#!/usr/bin/env python
"""Resolve the pre-registered T-branch for Phase 1.

The spec defines T1/T2/T3 on TOKENS only ("Constrained indexed still costs more
tokens than agentic, CI excludes 0"), so only a prompt_tokens row may carry one;
judge_correct and turns emit NOT_PREREG rather than borrowing a letter the spec
does not define for them. K_MORE/K_LESS/K_FLAT are NOT in the spec at all -- they
come from this plan's Phase 1 summariser, committed at 5323b38 before any data
existed -- so they are prefixed `plan:` to keep that visible in the CSV. See
scripts/code-retrieval/summarise_ws6c.py for the full statement of the rule; the two scripts
must agree.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.code_retrieval import topk as ws6c
from benchlib.config import (
    WS6C_ALPHA, WS6C_BOOT_N, WS6C_BOOT_SEED, WS6C_TOPK, ws6a_dir, ws6c_dir,
)


NOT_PREREG = ("n/a (not pre-registered)",) * 3
T_LABELS = ("T1", "T2", "T3")                             # spec, tokens only
K_LABELS = ("plan:K_MORE", "plan:K_LESS", "plan:K_FLAT")  # plan 5323b38, not spec


def main() -> None:
    runs6a = pd.read_csv(ws6a_dir() / "runs.csv")
    topk = pd.read_csv(ws6c_dir() / "topk_runs.csv")
    both = pd.concat([runs6a[runs6a["arm"].isin(["agentic", "indexed"])], topk])

    rows = []
    for a, b, labels in (
        (f"indexed_topk{WS6C_TOPK}", "agentic", T_LABELS),
        (f"indexed_topk{WS6C_TOPK}", "indexed", K_LABELS),
    ):
        for metric in ("prompt_tokens", "judge_correct", "turns"):
            _, d = ws6c.paired_deltas(both, metric, a, b)
            s = ws6c.paired_summary(d, WS6C_BOOT_N, WS6C_BOOT_SEED, WS6C_ALPHA)
            branch = ws6c.resolve_branch(
                s["excludes_zero"], s["median"],
                labels if metric == "prompt_tokens" else NOT_PREREG)
            rows.append({"arm_a": a, "arm_b": b, "metric": metric,
                         "branch": branch, **s})
            print(f"{a} - {b}  {metric:14s}  median {s['median']:+11.1f}  "
                  f"CI [{s['ci_low']:+.1f}, {s['ci_high']:+.1f}]  -> {branch}")

    out = ws6c_dir() / "topk_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    manifest.record(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
