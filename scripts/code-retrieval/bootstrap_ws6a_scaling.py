#!/usr/bin/env python
"""Phase 0, second half: paired bootstrap CIs over the FROZEN WS6a section 7 rows.

`scripts/code-retrieval/bootstrap_ws6a.py` puts the four main-run arm comparisons under a
paired interval. This does the same for results/code-retrieval.md's dilution
curve, which has the SAME defect as the headline: "agentic grows 35%
(6,040 -> 8,151)" is a DIFFERENCE OF MEDIANS across size points, not a paired
statistic, even though the same 10 `scaling_subset` questions run at every
size point. Spec branch A3 pre-registers the restatement.

The pairing here is over SIZE POINTS within one arm (L - S, and L - M), not
over arms: `size_point` is copied into the `arm` column so `paired_deltas`
can join on qid the same way it does everywhere else.

Zero API spend. Reads results/ws6a/scaling.csv, writes
results/ws6c/paired_ci_scaling.csv. Never writes into results/ws6a/.

Run:  python scripts/code-retrieval/bootstrap_ws6a_scaling.py
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

ARMS = ("agentic", "indexed")
# (larger, smaller) size points; the delta is larger - smaller, so a positive
# median means the arm got MORE expensive as the corpus grew.
SPANS = (("L", "S"), ("L", "M"))
METRICS = ("prompt_tokens", "judge_correct", "turns")

# ------------------------------------------------------------ branch labels
#
# Same rule as scripts/code-retrieval/bootstrap_ws6a.py: a cell may only carry an A-label
# where the spec pre-registers one. A3 is the branch that covers this file --
# it restates section 7's "neither arm scales meaningfully" as "no growth
# detectable at n=10 per cell", which is a statement about the S/M/L curve's
# cells, so both spans of that one curve fall under it for both arms.
#
# `turns` is a secondary diagnostic the A-series never mentions, so it gets no
# A-label. And as in the sibling script, the spec names A3 only for the
# NO-separation case -- it never expected growth to separate at n=10 -- so a
# separating cell emits the deliberately non-pre-registered "A3-REFUTED"
# rather than borrowing a letter the spec did not define.
NOT_PREREG = ("n/a (not pre-registered)",) * 3
# (positive-separation, negative-separation, no-separation)
A3_SERIES = ("A3-REFUTED", "A3-REFUTED", "A3")
BRANCH_LABELS = {"prompt_tokens": A3_SERIES, "judge_correct": A3_SERIES}


def main() -> None:
    scaling = pd.read_csv(ws6a_dir() / "scaling.csv")
    rows = []
    for arm in ARMS:
        sub = scaling[scaling["arm"] == arm].copy()
        # paired_deltas joins on the `arm` column; here the two "arms" being
        # compared are two size points of the SAME arm.
        sub["arm"] = sub["size_point"]
        for big, small in SPANS:
            for metric in METRICS:
                _, d = ws6c.paired_deltas(sub, metric, big, small)
                s = ws6c.paired_summary(d, n_boot=WS6C_BOOT_N,
                                        seed=WS6C_BOOT_SEED, alpha=WS6C_ALPHA)
                branch = ws6c.resolve_branch(
                    s["excludes_zero"], s["median"],
                    BRANCH_LABELS.get(metric, NOT_PREREG))
                rows.append({"arm": arm, "span": f"{big}-{small}",
                             "metric": metric, "branch": branch, **s})
                print(f"{arm:8s} {big}-{small}  {metric:14s}  "
                      f"median {s['median']:+10.1f}  "
                      f"95% CI [{s['ci_low']:+.1f}, {s['ci_high']:+.1f}]  "
                      f"{s['n_positive']}/{s['n']} grew  "
                      f"sign p={s['sign_p']:.4g}  -> {branch}")

    out = ws6c_dir() / "paired_ci_scaling.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    manifest.record(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
