#!/usr/bin/env python
"""Scaling sweep: the 10 scaling_subset questions x arms (indexed, agentic)
x subsets (S, M). L rows are reused from the main run -- not re-run.

Outputs: results/ws6a/scaling.csv

The S and M trees are the SAME persisted directories scripts/code-retrieval/index_ws6a.py
indexed (data/ws6a_subset_s, data/ws6a_subset_m), not fresh temporary ones:
claude-context keys an index by absolute codebase path, so indexing one path
and searching another would silently return nothing and the sweep would
measure an empty index rather than a small one. The agentic arm greps the same
tree, so both arms see exactly the same corpus at each size point.

Run:  python scripts/code-retrieval/scale_ws6a.py
"""

import asyncio
import functools
import sys
from pathlib import Path

import anyio.to_thread
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchlib import manifest, agent_harness
from benchlib.code_retrieval import bench as ws6a
from benchlib.config import DATA_DIR, WS6A_BUDGET_USD, ws6a_checkout, ws6a_dir
from index_ws6a import collection_prefix, materialise_subset

SWEEP_ARMS = ("indexed", "agentic")


async def sweep_one(client, subset, questions, prices, out):
    tree = DATA_DIR / f"ws6a_subset_{subset.lower()}"
    if not tree.exists():
        print(f"materialising {subset} subset at {tree}")
        materialise_subset(ws6a_checkout(), subset, tree)

    # Pre-flight per size point: the agentic arm must be runnable against THIS
    # tree before anything is billed against it.
    agentic = agent_harness.make_agentic_tools(tree)
    print(f"[{subset}] agentic toolchain OK: {[t.name for t in agentic]}")

    ckpt = DATA_DIR / "ws6a_cache" / f"ws6a_scaling_{subset.lower()}.jsonl"
    governor = ws6a.CostGovernor(
        limit=WS6A_BUDGET_USD,
        spent=agent_harness.total_spent_across_checkpoints(DATA_DIR),
    )
    print(f"[{subset}] governor: ${governor.spent:.2f} spent, "
          f"${governor.remaining:.2f} left")

    async with agent_harness.claude_context_session(
            tree, collection_prefix(subset),
            stderr_log=DATA_DIR / "ws6a_cache"
            / f"mcp_scale_{subset.lower()}.stderr.log") as (
            session, mcp_tools, raw):
        print(f"[{subset}] claude-context tools: {[t.name for t in raw]}")
        df = await anyio.to_thread.run_sync(functools.partial(
            agent_harness.execute_run,
            client, questions, SWEEP_ARMS,
            root=tree, prefix_text="", mcp_tools=mcp_tools,
            prices=prices, checkpoint_path=ckpt,
            judge_prompt=agent_harness.load_judge_prompt(out / "judge_prompt.txt"),
            governor=governor,
        ))

    ws6a.assert_no_errors(df)
    ws6a.assert_run_complete(df, list(questions["qid"]), SWEEP_ARMS)
    df["size_point"] = subset
    return df


async def main() -> None:
    out = ws6a_dir()
    client = agent_harness.make_client()

    questions = pd.read_csv(out / "questions.csv", dtype=str,
                            keep_default_na=False)
    scaling_q = questions[questions["scaling_subset"].str.lower() == "true"]
    print(f"{len(scaling_q)} scaling questions")

    prices = ws6a.load_pricing(out / "pricing.csv", regime="standard")
    frames = []
    for subset in ("S", "M"):
        frames.append(await sweep_one(client, subset, scaling_q, prices, out))

    # L comes free from the main run -- 60 new runs, not 90.
    runs = pd.read_csv(out / "runs.csv")
    l_rows = runs[runs["arm"].isin(SWEEP_ARMS)
                  & runs["qid"].isin(scaling_q["qid"])].copy()
    l_rows["size_point"] = "L"
    if len(l_rows) != len(scaling_q) * len(SWEEP_ARMS):
        raise SystemExit(
            f"expected {len(scaling_q) * len(SWEEP_ARMS)} reusable L rows from "
            f"runs.csv, found {len(l_rows)} -- the main run is incomplete")
    frames.append(l_rows)

    stats = pd.read_csv(out / "repo_stats.csv").set_index("scope")
    combined = pd.concat(frames, ignore_index=True)
    combined["corpus_tokens"] = combined["size_point"].map(stats["count_tokens"])
    combined["corpus_files"] = combined["size_point"].map(stats["files"])

    # The x-axis must be monotonic in TOKENS. File counts step 20x then 2.6x --
    # badly conditioned spacing that would make the curve's shape an artifact
    # of how the subsets happen to divide.
    axis = combined.groupby("size_point")["corpus_tokens"].first()
    if not (axis["S"] < axis["M"] < axis["L"]):
        raise SystemExit(f"corpus_tokens must increase S<M<L, got {axis.to_dict()}")

    path = out / "scaling.csv"
    combined.to_csv(path, index=False)
    manifest.record(path)
    print(combined.groupby(["size_point", "arm"]).agg(
        n=("qid", "count"),
        median_input=("input_tokens", "median"),
        judge_acc=("judge_correct", "mean")).to_string())
    print(f"\nwrote {len(combined)} rows to {path}")


if __name__ == "__main__":
    asyncio.run(main())
