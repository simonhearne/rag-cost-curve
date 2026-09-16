#!/usr/bin/env python
"""Resolve every pre-registered WS6c branch, across BOTH repositories.

Outputs:
  results/ws6c/summary.csv   -- per-arm aggregates (ws6a.summarise_runs), one
                                block per corpus, tagged with `corpus`
  results/ws6c/branches.csv  -- the paired branch resolutions

Extends scripts/code-retrieval/summarise_ws6c_topk.py, which is deliberately left in place
rather than renamed: it is the committed producer of topk_summary.csv, and
renaming it would leave the Phase 1 commit naming a script that no longer
exists, breaking the chain from a published CSV back to the code that made it.
This script recomputes Phase 1's T-branches from the same inputs, so the two
must agree -- if they ever disagree, something moved that should not have.

Every branch is a LOOKUP against the pre-registered spec via
ws6c.resolve_branch, never a judgement made after seeing the data. The label
triples below are read from the spec; nothing here decides them.

Run:  python scripts/code-retrieval/summarise_ws6c.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.code_retrieval import bench as ws6a
from benchlib.code_retrieval import topk as ws6c
from benchlib.config import (
    WS6C_ALPHA, WS6C_BOOT_N, WS6C_BOOT_SEED, WS6C_TOPK, ws6a_dir, ws6c_dir,
)

TOPK_ARM = f"indexed_topk{WS6C_TOPK}"

METRICS = ("prompt_tokens", "judge_correct", "turns")

# ------------------------------------------------------------ branch labels
#
# Label triples are (positive separation, negative, none). "positive" means
# arm_a - arm_b > 0 with a CI excluding zero. On prompt_tokens a POSITIVE delta
# means arm_a spent MORE, so for indexed - agentic the "indexed wins" branch is
# the NEGATIVE one.
#
# THE SPEC'S T- AND P-SERIES ARE DEFINED ON TOKENS ONLY:
#   T1/T2/T3  "Constrained indexed still costs more TOKENS than agentic, ..."
#   P1/P2/P3  "Indexed beats agentic ON TOKENS, CI excludes 0"
# so they may only be attached to a prompt_tokens row. Applying one triple
# across every metric -- which this script originally did -- put P1, the branch
# we are explicitly NOT claiming, into the `turns` row of the very file the talk
# cites for the P-verdict. Labels are therefore keyed on (arm_a, arm_b, metric)
# and everything the spec does not define emits NOT_PREREG.
#
# THE K-SERIES IS NOT PART OF THE SPEC. K_MORE/K_LESS/K_FLAT were introduced by
# the implementation plan's Phase 1 summariser, committed at 5323b38 BEFORE any
# data existed -- so they are fixed-in-advance in the weaker sense, but they are
# not part of the spec's binding A/T/S/P scheme. They are kept, because the
# comparison is real and correctly computed, and prefixed `plan:` so a reader of
# the CSV can see the distinction without opening the plan.
NOT_PREREG = ("n/a (not pre-registered)",) * 3
T_LABELS = ("T1", "T2", "T3")                  # spec section 3, Phase 1
P_LABELS = ("P2", "P1", "P3")                  # spec section 3, Phase 3
K_LABELS = ("plan:K_MORE", "plan:K_LESS", "plan:K_FLAT")  # plan 5323b38, not spec


def labels_for(arm_a: str, arm_b: str, metric: str, series):
    """`series` only applies to the token delta it was written about."""
    return series if metric == "prompt_tokens" else NOT_PREREG


# WS6a's `stuffed` arm has no counterpart on the uncontaminated corpus (that
# corpus is 4.5x over the prefix budget), so it is left out of this summary
# and stays in results/ws6a/summary.csv. `parametric` IS carried over, because
# summarise_runs derives judge_lift_over_parametric from it -- without it the
# fastapi block would report every lift against a floor of 0.0.
FASTAPI_ARMS = ("agentic", "indexed", "parametric")


def fastapi_frame() -> pd.DataFrame:
    """WS6a's frozen rows plus Phase 1's topk rows."""
    runs = pd.read_csv(ws6a_dir() / "runs.csv")
    topk = pd.read_csv(ws6c_dir() / "topk_runs.csv")
    return pd.concat([runs[runs["arm"].isin(FASTAPI_ARMS)], topk],
                     ignore_index=True)


def compare(df, corpus: str, pairs) -> list[dict]:
    rows = []
    for arm_a, arm_b, labels in pairs:
        for metric in METRICS:
            _, deltas = ws6c.paired_deltas(df, metric, arm_a, arm_b)
            s = ws6c.paired_summary(deltas, WS6C_BOOT_N, WS6C_BOOT_SEED,
                                    WS6C_ALPHA)
            branch = ws6c.resolve_branch(
                s["excludes_zero"], s["median"],
                labels_for(arm_a, arm_b, metric, labels))
            rows.append({"corpus": corpus, "arm_a": arm_a, "arm_b": arm_b,
                         "metric": metric, "branch": branch, **s})
            print(f"  {arm_a:14s} - {arm_b:10s} {metric:14s} "
                  f"median {s['median']:+11.1f}  "
                  f"CI [{s['ci_low']:+.1f}, {s['ci_high']:+.1f}]  "
                  f"sign p={s['sign_p']:.4f}  -> {branch}")
    return rows


def main() -> None:
    out = ws6c_dir()
    fastapi = fastapi_frame()
    uncontam = pd.read_csv(out / "runs.csv")

    branches = []

    print("\n=== fastapi (contaminated; WS6a frozen rows + Phase 1) ===")
    branches += compare(fastapi, "fastapi", (
        (TOPK_ARM, "agentic", T_LABELS),
        (TOPK_ARM, "indexed", K_LABELS),
    ))

    print("\n=== agentic_hil (uncontaminated; Phase 3) ===")
    print("PRE-REGISTERED: P1 = indexed beats agentic on prompt_tokens with "
          "the CI excluding 0;\n                P2 = agentic still beats "
          "indexed, CI excluding 0;\n                P3 = CI includes 0 -- "
          "underpowered, NEITHER may be claimed.")
    branches += compare(uncontam, "agentic_hil", (
        ("indexed", "agentic", P_LABELS),
        (TOPK_ARM, "agentic", P_LABELS),
        (TOPK_ARM, "indexed", K_LABELS),
    ))

    bpath = out / "branches.csv"
    pd.DataFrame(branches).to_csv(bpath, index=False)
    manifest.record(bpath)

    blocks = []
    for corpus, df in (("fastapi", fastapi), ("agentic_hil", uncontam)):
        block = ws6a.summarise_runs(df)
        block.insert(0, "corpus", corpus)
        blocks.append(block)
    summary = pd.concat(blocks, ignore_index=True)
    spath = out / "summary.csv"
    summary.to_csv(spath, index=False)
    manifest.record(spath)

    print("\n=== per-arm aggregates ===")
    print(summary[["corpus", "arm", "n", "median_prompt_tokens",
                   "median_turns", "judge_accuracy", "total_cost_billed"]]
          .to_string(index=False))

    headline = [r for r in branches
                if r["corpus"] == "agentic_hil" and r["arm_a"] == "indexed"
                and r["metric"] == "prompt_tokens"][0]
    print(f"\nHEADLINE: indexed - agentic on prompt_tokens, uncontaminated "
          f"corpus -> {headline['branch']}")
    print(f"wrote {bpath}\nwrote {spath}")


if __name__ == "__main__":
    main()
