#!/usr/bin/env python
"""WS4 judge reliability (spec 5.4): re-judge a seeded sample of completed
rows a second time. Checkpointed; charged to the same governor.

    python scripts/recall-quality/run_ws4_rejudge.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib.code_retrieval import bench as ws6a  # noqa: E402
from benchlib.recall_quality import runner as ws4_runner  # noqa: E402
from benchlib.config import SEED, WS4_JUDGE_RELIABILITY_N  # noqa: E402
from benchlib.agent_harness import load_checkpoint  # noqa: E402


def main():
    done = load_checkpoint(ws4_runner.CHECKPOINT)
    rows = [r for r in done.values() if r["subset"] == "main"]
    if len(rows) < WS4_JUDGE_RELIABILITY_N:
        sys.exit(f"only {len(rows)} main rows checkpointed")
    rows.sort(key=lambda r: (r["arm"], r["qid"]))
    sample = random.Random(f"{SEED}:rejudge").sample(rows, WS4_JUDGE_RELIABILITY_N)
    _, judge_p = ws4_runner.load_prompts()
    prices = ws6a.load_pricing(Path("results/ws6a/pricing.csv"))
    gov = ws4_runner.governor_from_checkpoints()
    out = ws4_runner.rejudge_sample(ws4_runner.make_client(), sample, judge_p,
                                    prices, gov)
    agree = sum(r["first_verdict"] == r["second_verdict"] for r in out.values())
    print(f"agreement {agree}/{len(out)} = {agree / len(out):.3f}; governor "
          f"${gov.spent:.4f}")


if __name__ == "__main__":
    main()
