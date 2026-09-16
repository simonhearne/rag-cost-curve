#!/usr/bin/env python
"""Phase 2b: non-memorization gate. Arm (d) only, on all three candidates,
BEFORE any retrieval spend.

Stars and creation date are priors; this is the measurement. All three scores
are published whether or not the repo is used.

Run:  python scripts/code-retrieval/gate_ws6c.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest, agent_harness
from benchlib.code_retrieval import bench as ws6a
from benchlib.config import (
    DATA_DIR, WS6C_BUDGET_USD, WS6C_CANDIDATES, ws6a_dir, ws6c_checkout,
    ws6c_dir,
)
# Shared with scripts/code-retrieval/index_ws6c.py -- ONE implementation of the pin guard.
from benchlib.code_retrieval.topk import assert_pinned_checkouts

# Pre-registered in the spec, section 3 Phase 2. Read, never decided here.
S1_MAX, S2_MAX = 0.15, 0.40


def branch(acc: float) -> str:
    return "S1" if acc <= S1_MAX else ("S2" if acc <= S2_MAX else "S3")


def main() -> None:
    out = ws6c_dir()
    print("checking candidate pins before any spend")
    assert_pinned_checkouts()

    client = agent_harness.make_client()
    prices = ws6a.load_pricing(ws6a_dir() / "pricing.csv", regime="standard")
    judge_prompt = agent_harness.load_judge_prompt(ws6a_dir() / "judge_prompt.txt")
    allq = pd.read_csv(out / "gate_questions.csv", dtype=str, keep_default_na=False)

    rows = []
    for cand in WS6C_CANDIDATES:
        slug = cand["slug"]
        q = allq[allq["qid"].str.startswith(slug)].copy()
        ckpt = DATA_DIR / "ws6c_cache" / f"gate_{slug}.jsonl"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        governor = ws6a.CostGovernor(
            limit=WS6C_BUDGET_USD,
            spent=agent_harness.total_spent_across_checkpoints(DATA_DIR),
        )
        df = agent_harness.execute_run(
            client, q, ("parametric",),
            root=ws6c_checkout(slug), prefix_text=None, mcp_tools=None,
            prices=prices, checkpoint_path=ckpt, judge_prompt=judge_prompt,
            governor=governor, transcript_dir=out / f"transcripts_gate_{slug}",
        )
        ws6a.assert_no_errors(df)
        # Published columns are judge_accuracy and any_file_hit_rate only.
        # `symbol_hit` in data/ws6c_cache/gate_*.jsonl is NOT a signal here: an
        # arm that declines ("I don't have access to the codebase...") echoes
        # any symbol named in the question back inside its refusal, which the
        # substring scorer counts as a hit. Deliberately not reported.
        acc = float(df["judge_correct"].astype(bool).mean())
        hit = float(df["any_file_hit"].astype(bool).mean())
        rows.append({"slug": slug, "repo": cand["repo"], "n": len(df),
                     "judge_accuracy": acc, "any_file_hit_rate": hit,
                     "branch": branch(acc)})
        print(f"{slug:16s} n={len(df)}  parametric judge acc={acc:.3f}  "
              f"any_file_hit={hit:.3f}  -> {branch(acc)}")

    path = out / "nonmemorization_gate.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    manifest.record(path)
    print(f"\nwrote {path}")
    print("Take the FIRST S1 candidate in WS6C_CANDIDATES order. If none is "
          "S1, see the spec: S2 only if no S1 exists; all-S3 stops WS6c at "
          "Phase 1 and that outcome is itself reportable.")


if __name__ == "__main__":
    main()
