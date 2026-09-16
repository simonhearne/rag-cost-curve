#!/usr/bin/env python
"""Phase 3: the private-repo proxy run. 40 questions x 4 arms on the corpus
the Phase 2 gate selected, interleaved, checkpointed.

Arms: indexed (server default top-k), indexed_topk<K>, agentic, parametric.
Arm (c) does not exist here -- all three candidate corpora measured 4.4x-15.6x
over the 960k stuffed-prefix budget, so `prefix_text` is None and
`in_stuffed_prefix` stays blank on every row.

TWO MCP SESSIONS, ONE CHECKPOINT. claude-context's tool list is built at
session entry, so binding search_code's `limit` needs its own session; the two
indexed arms therefore cannot be interleaved WITH EACH OTHER. They are each
interleaved against the same question order, under the same seed, against the
same index. Recorded in results/ws6c/run_provenance.txt rather than left for a
reader to infer.

Resumable -- rerun after an abort and it continues from the checkpoint.

Run:  python scripts/code-retrieval/run_ws6c.py --smoke 2      # pre-flight, 8 pairs
      python scripts/code-retrieval/run_ws6c.py                # the full 160
"""

import argparse
import asyncio
import functools
import sys
from pathlib import Path

import anyio.to_thread
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest, agent_harness
from benchlib.code_retrieval import bench as ws6a
from benchlib.code_retrieval import topk as ws6c
from benchlib.config import (
    DATA_DIR, WS6C_BUDGET_USD, WS6C_CANDIDATES, WS6C_TOPK, ws6a_dir,
    ws6c_checkout, ws6c_dir,
)
# Shared with scripts/code-retrieval/gate_ws6c.py -- ONE implementation of the pin guard.
from benchlib.code_retrieval.topk import assert_pinned_checkouts

TOPK_ARM = f"indexed_topk{WS6C_TOPK}"
ARMS = ("indexed", TOPK_ARM, "agentic", "parametric")
# indexed_topk<K> runs the `indexed` code path under its own published label,
# so the two never collide in the shared checkpoint.
ARM_KINDS = {TOPK_ARM: "indexed"}

# Pre-registered in the spec, section 3 Phase 2. Read, never decided here.
GATE_PREFERENCE = ("S1", "S2")


def selected_slug() -> str:
    """The corpus the Phase 2 gate selected -- looked up, never chosen here.

    Takes the FIRST S1 candidate in WS6C_CANDIDATES order; if no candidate is
    S1, the first S2; all-S3 stops WS6c at Phase 1.
    """
    gate = pd.read_csv(ws6c_dir() / "nonmemorization_gate.csv").set_index("slug")
    order = [c["slug"] for c in WS6C_CANDIDATES]
    for want in GATE_PREFERENCE:
        for slug in order:
            if gate.loc[slug, "branch"] == want:
                print(f"gate selected {slug} ({want}, parametric judge acc "
                      f"{gate.loc[slug, 'judge_accuracy']:.3f})")
                return slug
    raise SystemExit("every candidate scored S3: WS6c stops at Phase 1, and "
                     "that outcome is itself reportable (see the spec)")


def new_governor() -> ws6a.CostGovernor:
    """I-2: cumulative across every phase's checkpoint, never a single file.

    Re-seeded before each execute_run call so the second MCP session starts
    from the spend the first one just recorded.
    """
    gov = ws6a.CostGovernor(
        limit=WS6C_BUDGET_USD,
        spent=agent_harness.total_spent_across_checkpoints(DATA_DIR))
    print(f"governor: ${gov.spent:.2f} spent across all phases, "
          f"${gov.remaining:.2f} of ${WS6C_BUDGET_USD:.2f} left")
    return gov


def write_provenance(out: Path, slug: str, n_questions: int,
                     derived: dict) -> None:
    stats = pd.read_csv(ws6c_dir() / "corpus_stats.csv").set_index("slug")
    fastapi = pd.read_csv(ws6a_dir() / "repo_stats.csv").set_index("scope")
    row, ref = stats.loc[slug], fastapi.loc["L"]
    path = out / "run_provenance.txt"
    path.write_text(
        "# results/ws6c/runs.csv -- how it was produced\n"
        f"corpus: {row['repo']} @ {row['commit']} ({row['license']})\n"
        f"arms: {', '.join(ARMS)}   questions: {n_questions}   "
        f"pairs: {n_questions * len(ARMS)}\n"
        f"topk arm binds search_code limit={WS6C_TOPK}; the `indexed` arm "
        "leaves the server default in place\n"
        "\n"
        "THE TREE. Both tool-using arms run against data/ws6c_tree_<slug>, a\n"
        "sha256-verified copy of the pinned checkout with its root-level\n"
        ".dockerignore dropped. The pinned package merges every root `.*ignore`\n"
        "file into its own ignore set, and this repo's `**`-style .dockerignore\n"
        "makes it index ZERO files while reporting success. The dropped file is\n"
        "not in WS6A_TEXT_EXTENSIONS, so it is none of the 342 files or\n"
        "4,307,363 count_tokens corpus_stats.csv pins; no question references\n"
        "it; and it would never be indexed anyway (dot-prefixed). See\n"
        "benchlib/code_retrieval/topk.py for the reproduction.\n"
        f"  tree: {derived['files_copied']} files, dropped "
        f"{derived['dropped_ignore_files']}\n"
        f"  tree_sha256: {derived['tree_sha256']}\n"
        "  (sha256 over '<path> <file sha256>' lines, sorted by path)\n"
        "\n"
        "SCOPE MATCHING (WS6a section 7's rule: by measured tokens, never by\n"
        "file count). This corpus is the whole repository -- no subset was\n"
        "cut, because it lands BETWEEN fastapi's M and L cells, nearest L,\n"
        "and a corpus cannot be cut UP to a larger scope.\n"
        f"  count_tokens: {int(row['count_tokens']):,} vs fastapi L "
        f"{int(ref['count_tokens']):,}  ratio {row['scope_ratio']}\n"
        f"  files:        {int(row['files']):,} vs fastapi L "
        f"{int(ref['files']):,}  ratio "
        f"{int(row['files']) / int(ref['files']):.3f}\n"
        "  The two corpora are matched on tokens and MISMATCHED on file\n"
        "  count. Grep's cost scales with how much tree it must search, so a\n"
        "  small-file-count haystack favours the AGENTIC arm -- this cuts\n"
        "  against the hypothesis the phase is testing, not for it.\n"
        "\n"
        "INTERLEAVING. claude-context builds its tool list at session entry,\n"
        "so binding search_code's limit needs a second MCP session. The run\n"
        "is therefore two sequential execute_run calls against ONE\n"
        f"checkpoint: ('indexed', 'agentic', 'parametric') first, then\n"
        f"('{TOPK_ARM}',). The two indexed arms are NOT interleaved with each\n"
        "other; each is interleaved against the same question order under the\n"
        "same seed, against the same index.\n"
        "\n"
        "ARM (c) IS ABSENT. All three candidates measured 4.4x-15.6x over the\n"
        "960k stuffed-prefix budget, so prefix_text is None and\n"
        "in_stuffed_prefix is blank on every row.\n")
    manifest.record(path)
    print(f"wrote {path}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0,
                    help="run only the first N questions across all four "
                         "arms and write no CSV")
    args = ap.parse_args()

    slug = selected_slug()
    checkout = ws6c_checkout(slug)
    out = ws6c_dir()

    # BEFORE any spend: the corpus indexed in Phase 3a must still be the
    # corpus being measured now.
    print("checking the pin before any spend")
    assert_pinned_checkouts([slug])

    # BOTH tool-using arms see the same derived tree Phase 3a indexed --
    # re-verified here byte by byte against the pinned checkout, so the arms
    # still differ only in retrieval strategy. See benchlib.code_retrieval.topk for why the
    # checkout itself cannot be indexed.
    derived = ws6c.materialise_index_tree(
        checkout, ws6a.git_ls_files(checkout), ws6c.index_tree(slug))
    root = Path(derived["tree"])
    print(f"derived tree verified: {derived['files_copied']} files, dropped "
          f"{derived['dropped_ignore_files'] or '(nothing)'}")

    client = agent_harness.make_client()

    # PRE-FLIGHT before any spend: a missing ripgrep must fail here, not 120
    # pairs in with the agentic arm silently absent from the results.
    agentic = agent_harness.make_agentic_tools(root)
    print(f"agentic toolchain OK: {[t.name for t in agentic]}")

    questions = pd.read_csv(out / "questions.csv", dtype=str,
                            keep_default_na=False)
    if args.smoke:
        questions = questions.head(args.smoke)
        print(f"SMOKE: {len(questions)} question(s) x {len(ARMS)} arms")

    prices = ws6a.load_pricing(ws6a_dir() / "pricing.csv", regime="standard")
    judge_prompt = agent_harness.load_judge_prompt(ws6a_dir() / "judge_prompt.txt")
    checkpoint = DATA_DIR / "ws6c_cache" / f"run_{slug}.jsonl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    # Two sessions, one checkpoint. topk=None first so the default-top-k arm
    # is measured against exactly the frozen WS6a tool binding.
    sessions = (
        (None, ("indexed", "agentic", "parametric"), "default"),
        (WS6C_TOPK, (TOPK_ARM,), f"topk{WS6C_TOPK}"),
    )

    df = None
    try:
        for topk, arms, tag in sessions:
            print(f"\n=== MCP session: topk={topk} arms={arms} ===")
            async with agent_harness.claude_context_session(
                    root, f"ws6c_{slug}", topk=topk,
                    stderr_log=DATA_DIR / "ws6c_cache"
                    / f"mcp_run_{slug}_{tag}.stderr.log") as (
                    session, mcp_tools, raw):
                print(f"claude-context tools: {[t.name for t in raw]}")
                if not mcp_tools:
                    raise SystemExit("claude-context exposed no tools; the "
                                     "indexed arms cannot run")
                # execute_run is synchronous by design (the arms must differ
                # only in the tools list). The MCP tools are sync wrappers
                # bridging back to THIS event loop via anyio.from_thread.run,
                # which requires the sync code to run inside
                # anyio.to_thread.run_sync -- calling execute_run directly
                # here would deadlock on the first tool call.
                df = await anyio.to_thread.run_sync(functools.partial(
                    agent_harness.execute_run,
                    client, questions, arms,
                    root=root, prefix_text=None, mcp_tools=mcp_tools,
                    prices=prices, checkpoint_path=checkpoint,
                    judge_prompt=judge_prompt, governor=new_governor(),
                    transcript_dir=out / f"transcripts_{slug}",
                    arm_kinds=ARM_KINDS,
                ))
    except ws6a.BudgetExceeded as exc:
        print(f"\nGOVERNOR TRIPPED: {exc}")
        df = pd.DataFrame(
            list(agent_harness.load_checkpoint(checkpoint).values()))

    qids = list(questions["qid"])
    df = df[df["qid"].isin(qids)].copy()

    # Both gates, in this order. assert_no_errors catches an errored row that
    # somehow reached the checkpoint; assert_run_complete catches the failure
    # it structurally cannot -- since C-2 a broken arm leaves MISSING rows,
    # not errored ones, so a 120-row runs.csv would otherwise be written as a
    # finished 160-pair experiment.
    ws6a.assert_no_errors(df)
    ws6a.assert_run_complete(df, qids, ARMS)
    print(f"gates passed: {len(df)} rows, "
          f"{df.groupby('arm')['qid'].count().to_dict()}")
    print(df.groupby("arm")["turns"].describe()[["min", "50%", "max"]]
          .to_string())

    if args.smoke:
        print("\nSMOKE ONLY -- no CSV written. Both indexed arms must show "
              "turns > 1: a 1-turn indexed row answered from parametric "
              "memory is the exact WS6a failure this check exists to catch.")
        return

    runs_path = out / "runs.csv"
    df.to_csv(runs_path, index=False)
    manifest.record(runs_path)
    write_provenance(out, slug, len(questions), derived)
    print(f"\nwrote {len(df)} rows to {runs_path}")
    print(df.groupby("arm")["cost_total_billed"].sum().round(2).to_string())
    print(f"cumulative billed across all phases: "
          f"${agent_harness.total_spent_across_checkpoints(DATA_DIR):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
