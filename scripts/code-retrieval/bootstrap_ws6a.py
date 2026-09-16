#!/usr/bin/env python
"""Phase 0: paired bootstrap CIs over the FROZEN results/ws6a/runs.csv.

Zero API spend. Reads WS6a, writes results/ws6c/paired_ci.csv. Never writes
into results/ws6a/.

Run:  python scripts/code-retrieval/bootstrap_ws6a.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.code_retrieval import topk as ws6c
from benchlib.config import (
    WS6C_ALPHA, WS6C_BOOT_N, WS6C_BOOT_SEED, ws6a_dir, ws6c_dir,
)

COMPARISONS = (
    ("indexed", "agentic"),
    ("indexed", "parametric"),
    ("agentic", "parametric"),
    ("stuffed", "agentic"),
)
METRICS = ("prompt_tokens", "judge_correct", "turns", "wall_clock_s")

# ------------------------------------------------------------ branch labels
#
# The `branch` column is a LOOKUP against spec section 3 Phase 0, so a cell may
# only carry an A-label where the spec actually pre-registers one. The spec
# defines the A-series for exactly one contrast -- its "Primary statistic" is
# d_q = prompt_tokens(indexed, q) - prompt_tokens(agentic, q) -- and for that
# contrast's two metrics:
#
#   A1  token-delta CI EXCLUDES 0   -> the inversion stands with its CI
#   A2  token-delta CI INCLUDES 0   -> C11 downgraded to "no detectable difference"
#   A3  accuracy-delta CI INCLUDES 0 -> accuracy "not separable at n=40"
#
# Everything else in this sweep is a secondary diagnostic. Labelling those
# cells A1/A2 would assert a pre-registration that does not exist, which is
# precisely the thing pre-registration is supposed to make impossible to fake.
#
# NOTE THE ASYMMETRY on judge_correct: the spec names A3 only for the
# NO-separation case, because it never expected accuracy to separate at n=40.
# It therefore names NO label for accuracy separating. Rather than silently
# reuse an A-label the spec did not define, that case emits "A3-REFUTED" --
# unmistakably not a pre-registered branch, and loud if it ever fires.
NOT_PREREG = ("n/a (not pre-registered)",) * 3
# (positive-separation, negative-separation, no-separation)
BRANCH_LABELS = {
    ("indexed", "agentic", "prompt_tokens"): ("A1", "A1", "A2"),
    ("indexed", "agentic", "judge_correct"): ("A3-REFUTED", "A3-REFUTED", "A3"),
}


def main() -> None:
    runs = pd.read_csv(ws6a_dir() / "runs.csv")
    rows = []
    for arm_a, arm_b in COMPARISONS:
        for metric in METRICS:
            qids, d = ws6c.paired_deltas(runs, metric, arm_a, arm_b)
            s = ws6c.paired_summary(d, n_boot=WS6C_BOOT_N,
                                    seed=WS6C_BOOT_SEED, alpha=WS6C_ALPHA)
            labels = BRANCH_LABELS.get((arm_a, arm_b, metric), NOT_PREREG)
            branch = ws6c.resolve_branch(
                s["excludes_zero"], s["median"], labels)
            rows.append({"arm_a": arm_a, "arm_b": arm_b, "metric": metric,
                         "branch": branch, **s})
            print(f"{arm_a:10s} - {arm_b:10s}  {metric:14s}  "
                  f"median {s['median']:+12.1f}  "
                  f"95% CI [{s['ci_low']:+.1f}, {s['ci_high']:+.1f}]  "
                  f"{s['n_positive']}/{s['n']} positive  "
                  f"sign p={s['sign_p']:.4g}  -> {branch}")

    out = ws6c_dir() / "paired_ci.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    manifest.record(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
