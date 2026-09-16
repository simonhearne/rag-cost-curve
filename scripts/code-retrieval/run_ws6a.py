#!/usr/bin/env python
"""Full WS6a run: 40 questions x 4 arms, interleaved, checkpointed.

Resumable -- rerun after an abort and it continues from the checkpoint.

Run:  python scripts/code-retrieval/run_ws6a.py
"""

import asyncio
import functools
import sys
from pathlib import Path

import anyio.to_thread
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest, agent_harness
from benchlib.code_retrieval import bench as ws6a
from benchlib.config import (
    DATA_DIR, WS6A_ARMS, WS6A_BUDGET_USD, ws6a_checkout, ws6a_dir,
)

CHECKPOINT = DATA_DIR / "ws6a_cache" / "ws6a_checkpoint.jsonl"


async def main() -> None:
    root = ws6a_checkout()
    out = ws6a_dir()
    client = agent_harness.make_client()

    # PRE-FLIGHT before any spend: a missing ripgrep must fail here, not 120
    # pairs in with the agentic arm silently absent from the results.
    agentic = agent_harness.make_agentic_tools(root)
    print(f"agentic toolchain OK: {[t.name for t in agentic]}")

    questions = pd.read_csv(out / "questions.csv", dtype=str,
                            keep_default_na=False)
    prefix_text = agent_harness.load_prefix_text(root)
    prices = ws6a.load_pricing(out / "pricing.csv", regime="standard")
    # I-2: cumulative across every phase's checkpoint, never a single file.
    governor = ws6a.CostGovernor(
        limit=WS6A_BUDGET_USD,
        spent=agent_harness.total_spent_across_checkpoints(DATA_DIR),
    )
    print(f"governor: ${governor.spent:.2f} spent across all phases, "
          f"${governor.remaining:.2f} of ${WS6A_BUDGET_USD:.2f} left")

    async with agent_harness.claude_context_session(
            root, "ws6a_l",
            stderr_log=DATA_DIR / "ws6a_cache" / "mcp_run.stderr.log") as (
            session, mcp_tools, raw):
        print(f"claude-context tools: {[t.name for t in raw]}")
        if not mcp_tools:
            raise SystemExit("claude-context exposed no tools; indexed arm "
                             "cannot run")
        # execute_run is synchronous by design (the arms must differ only in
        # the tools list). The MCP tools are sync wrappers that bridge back to
        # THIS event loop via anyio.from_thread.run, which requires the sync
        # code to run inside anyio.to_thread.run_sync. Calling execute_run
        # directly here would deadlock the first time the indexed arm calls a
        # tool.
        try:
            df = await anyio.to_thread.run_sync(functools.partial(
                agent_harness.execute_run,
                client, questions, WS6A_ARMS,
                root=root, prefix_text=prefix_text, mcp_tools=mcp_tools,
                prices=prices, checkpoint_path=CHECKPOINT,
                judge_prompt=agent_harness.load_judge_prompt(
                    out / "judge_prompt.txt"),
                governor=governor,
                transcript_dir=out / "transcripts",
            ))
        except ws6a.BudgetExceeded as exc:
            print(f"\nGOVERNOR TRIPPED: {exc}")
            df = pd.DataFrame(
                list(agent_harness.load_checkpoint(CHECKPOINT).values()))

    # Both gates, in this order. assert_no_errors catches an errored row that
    # somehow reached the checkpoint; assert_run_complete catches the failure
    # that assert_no_errors structurally cannot -- since C-2, a broken arm
    # leaves MISSING rows, not errored ones, so a 120-row runs.csv would
    # otherwise be written as a finished 160-pair experiment.
    ws6a.assert_no_errors(df)
    ws6a.assert_run_complete(df, list(questions["qid"]), WS6A_ARMS)
    print(f"gates passed: {len(df)} rows, "
          f"{df.groupby('arm')['qid'].count().to_dict()}")

    runs_path = out / "runs.csv"
    df.to_csv(runs_path, index=False)
    manifest.record(runs_path)
    print(f"\nwrote {len(df)} rows to {runs_path}")
    print(df.groupby("arm")["cost_total_billed"].sum().round(2).to_string())
    print(f"total billed: ${governor.spent:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
