#!/usr/bin/env python
"""Phase 1: re-run arm (a) on fastapi with search_code's limit bound to
WS6C_TOPK. One variable changes. No re-index, no embedding spend.

Resumable -- rerun after an abort and it continues from the checkpoint.

Run:  python scripts/code-retrieval/run_ws6c_topk.py
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
    DATA_DIR, WS6C_BUDGET_USD, WS6C_TOPK, ws6a_checkout, ws6a_dir, ws6c_dir,
)

CHECKPOINT = DATA_DIR / "ws6c_cache" / "topk_checkpoint.jsonl"
ARM_LABEL = f"indexed_topk{WS6C_TOPK}"


async def main() -> None:
    root = ws6a_checkout()
    out6a, out6c = ws6a_dir(), ws6c_dir()
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    client = agent_harness.make_client()

    questions = pd.read_csv(out6a / "questions.csv", dtype=str,
                            keep_default_na=False)
    prices = ws6a.load_pricing(out6a / "pricing.csv", regime="standard")
    governor = ws6a.CostGovernor(
        limit=WS6C_BUDGET_USD,
        spent=agent_harness.total_spent_across_checkpoints(DATA_DIR),
    )
    print(f"governor: ${governor.spent:.2f} spent across all phases, "
          f"${governor.remaining:.2f} of ${WS6C_BUDGET_USD:.2f} left")

    async with agent_harness.claude_context_session(
            root, "ws6a_l", topk=WS6C_TOPK,
            stderr_log=DATA_DIR / "ws6c_cache" / "mcp_topk.stderr.log") as (
            session, mcp_tools, raw):
        if not mcp_tools:
            raise SystemExit("claude-context exposed no tools")
        df = await anyio.to_thread.run_sync(functools.partial(
            agent_harness.execute_run,
            client, questions, ("indexed",),
            root=root, prefix_text=None, mcp_tools=mcp_tools,
            prices=prices, checkpoint_path=CHECKPOINT,
            judge_prompt=agent_harness.load_judge_prompt(
                out6a / "judge_prompt.txt"),
            governor=governor,
            transcript_dir=out6c / "transcripts_topk",
        ))

    ws6a.assert_no_errors(df)
    ws6a.assert_run_complete(df, list(questions["qid"]), ("indexed",))
    df["arm"] = ARM_LABEL
    path = out6c / "topk_runs.csv"
    df.to_csv(path, index=False)
    manifest.record(path)
    print(f"wrote {path}: {len(df)} rows")


if __name__ == "__main__":
    asyncio.run(main())
