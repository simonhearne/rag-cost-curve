#!/usr/bin/env python
"""WS9: paired CIs over results/ws6c/scaling.csv. Zero API spend.

Outputs: results/ws6c/paired_ci_unseen_scaling.csv

TWO FAMILIES, one file, distinguished by `contrast_kind`:

  arm_at_size  -- the pre-registered U-series. At each size, the three arm
                  contrasts. Pairing is over QUESTIONS within one size point.
  size_span    -- the dilution curve. Within each arm, L-S and L-M. Pairing is
                  over QUESTIONS across two size points, which works because
                  the SAME 37 questions run at every size.

A NEW FILE, NOT AN APPEND. results/ws6c/paired_ci_scaling.csv holds WS6a's
frozen FASTAPI rows and has no corpus column; appending agentic_hil rows to it
is how a cross-corpus interval gets computed by accident. WS6c is explicit that
no interval may be paired across the two corpora.

Run:  .venv/bin/python scripts/code-retrieval/bootstrap_ws9.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.code_retrieval import topk as ws6c
from benchlib.config import (
    WS6C_ALPHA, WS6C_BOOT_N, WS6C_BOOT_SEED, WS6C_TOPK, WS9_SCOPES, WS9_SLUG,
    ws6c_dir,
)

TOPK_ARM = f"indexed_topk{WS6C_TOPK}"
CONTRASTS = (("indexed", "agentic"), (TOPK_ARM, "agentic"), (TOPK_ARM, "indexed"))
SPANS = (("L", "S"), ("L", "M"))
METRICS = ("prompt_tokens", "judge_correct", "turns")

# ------------------------------------------------------------ branch labels
#
# Spec section 4, U-series, over prompt_tokens ONLY:
#   U1  CI excludes zero and is NEGATIVE (the index arm costs fewer tokens)
#   U2  CI excludes zero and is POSITIVE
#   U3  CI includes zero -- report the point estimate, claim NEITHER direction
#
# resolve_branch takes (positive-separation, negative-separation, none).
U_SERIES = ("U2", "U1", "U3")

# judge_correct and turns are supporting mechanism the U-series never names, so
# they get no U-letter -- the same rule bootstrap_ws6a.py applies to the
# A-series. A cell that borrowed a letter would be claiming a pre-registration
# that does not cover it.
NOT_PREREG = ("n/a (not pre-registered)",) * 3

# The size spans are exploratory in full: WS6a's equivalent found no detectable
# growth over a 24x span at n=10, and WS9 has 3.74x, so nothing is expected and
# nothing is claimed. No metric of this family carries a U-letter.
BRANCH_LABELS = {"prompt_tokens": U_SERIES}


def main() -> None:
    out = ws6c_dir()
    scaling = pd.read_csv(out / "scaling.csv")
    rows = []

    # ---- family 1: the U-series, arm contrasts within each size ----
    for size in WS9_SCOPES:
        sub = scaling[scaling["size_point"] == size]
        for arm_a, arm_b in CONTRASTS:
            for metric in METRICS:
                _, d = ws6c.paired_deltas(sub, metric, arm_a, arm_b)
                s = ws6c.paired_summary(d, n_boot=WS6C_BOOT_N,
                                        seed=WS6C_BOOT_SEED, alpha=WS6C_ALPHA)
                branch = ws6c.resolve_branch(
                    s["excludes_zero"], s["median"],
                    BRANCH_LABELS.get(metric, NOT_PREREG))
                rows.append({"corpus": WS9_SLUG, "contrast_kind": "arm_at_size",
                             "arm_a": arm_a, "arm_b": arm_b, "size": size,
                             "span": "", "metric": metric, "branch": branch, **s})
                print(f"[{size}] {arm_a:14s} - {arm_b:9s} {metric:14s} "
                      f"median {s['median']:+10.1f}  "
                      f"95% CI [{s['ci_low']:+.1f}, {s['ci_high']:+.1f}]  "
                      f"{s['n_positive']}/{s['n']} positive  "
                      f"sign p={s['sign_p']:.4g}  -> {branch}")

    # ---- family 2: the dilution curve, size spans within each arm ----
    for arm in ("agentic", "indexed", TOPK_ARM):
        sub = scaling[scaling["arm"] == arm].copy()
        # paired_deltas joins on the `arm` column; here the two "arms" being
        # compared are two size points of the SAME arm.
        sub["arm"] = sub["size_point"]
        for big, small in SPANS:
            for metric in METRICS:
                _, d = ws6c.paired_deltas(sub, metric, big, small)
                s = ws6c.paired_summary(d, n_boot=WS6C_BOOT_N,
                                        seed=WS6C_BOOT_SEED, alpha=WS6C_ALPHA)
                rows.append({"corpus": WS9_SLUG, "contrast_kind": "size_span",
                             "arm_a": arm, "arm_b": arm, "size": "",
                             "span": f"{big}-{small}", "metric": metric,
                             "branch": NOT_PREREG[0], **s})
                print(f"[{arm:14s}] {big}-{small} {metric:14s} "
                      f"median {s['median']:+10.1f}  "
                      f"95% CI [{s['ci_low']:+.1f}, {s['ci_high']:+.1f}]  "
                      f"-> exploratory")

    df = pd.DataFrame(rows)
    path = out / "paired_ci_unseen_scaling.csv"
    df.to_csv(path, index=False)
    manifest.record(path)

    # The multiplicity disclosure the spec requires, printed where it cannot
    # be missed: nine unadjusted cells at alpha=0.05.
    u = df[(df.contrast_kind == "arm_at_size") & (df.metric == "prompt_tokens")]
    resolved = u[u.branch != "U3"]
    print(f"\nU-SERIES: {len(u)} unadjusted cells at alpha={WS6C_ALPHA} "
          f"(3 contrasts x 3 sizes), no multiplicity correction. Under a "
          f"global null ~{len(u) * WS6C_ALPHA:.2f} cells resolve by chance.")
    print(f"resolved away from U3: {len(resolved)} "
          f"{resolved[['size', 'arm_a', 'arm_b', 'branch']].to_dict('records')}")
    if 0 < len(resolved) <= 1:
        print("A SINGLE isolated resolution among nine cells is NOT evidence "
              "of anything and may not be reported as a finding (spec 4).")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
