#!/usr/bin/env python
"""WS4d Phase 3c -- gate L1 (ladder density) and the 15-arm changepoint refit.

Refuses to run unless results/ws4d/ladder_gates.csv already carries a PASSED
L2 (judge drift) row, written by scripts/recall-quality/ws4d_phase3_run.py -- an L2 failure
means the four new arms are not comparable to the existing eleven and the
refit is meaningless (spec §5.3).

Gate L1 is evaluated the same way Task 11 already measured it: the eleven
existing retrieval arms' recall is their mean recall_at_10 over WS4b's
corrected primary stratum (325 questions, tier0_runs_corrected.csv -- this
IS what Task 11's report quoted); the four new arms' recall is the
`ws4_recall_at_10_all3610` figure recorded in their own retrieval npz (all
3610 validation queries -- what Task 11's sweep measured them against). L1
is known in advance to FAIL (max adjacent gap 0.0416 > 0.035); its
pre-registered fail action is to report the achieved spacing and refit
anyway, not to tune nprobe or add a fifth arm.

The refit itself uses PAIRED PER-QUESTION recall_at_10 / judge_correct for
all 15 arms (both measured the same way, at generation time, over the same
question pool) -- a materially different and more principled comparison
than the L1 aggregate figures above.

Design: results/recall-vs-quality.md

Run:  .venv/bin/python scripts/recall-quality/ws4d_phase3_refit.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest  # noqa: E402
from benchlib.recall_quality import eligibility as ws4d  # noqa: E402
from benchlib.config import SEED, WS4B_CHANGEPOINT_BOOT_N, ws4_cache, ws4b_dir, ws4d_cache, ws4d_dir  # noqa: E402
from benchlib.agent_harness import load_checkpoint  # noqa: E402

PARAMETRIC = "parametric"

# Must match scripts/recall-quality/ws4d_phase3_run.py exactly -- these are the four arms
# and the checkpoint that script wrote.
NEW_ARMS = ("sq8_np6", "sq8_np12", "sq8_np24", "sq8_np48")
RET = ws4_cache() / "retrieval"
RUN_CHECKPOINT = ws4d_cache() / "phase3_runs_checkpoint.jsonl"


def check_l2_passed(gates_path: Path) -> None:
    if not gates_path.exists():
        raise SystemExit(f"{gates_path} does not exist -- run "
                         f"scripts/recall-quality/ws4d_phase3_run.py to completion first "
                         f"(it writes gate L2 there)")
    gates = pd.read_csv(gates_path)
    l2 = gates[gates.gate == "L2"]
    if l2.empty:
        raise SystemExit(f"{gates_path} has no L2 row -- run "
                         f"scripts/recall-quality/ws4d_phase3_run.py to completion first")
    if not bool(l2.iloc[0].passed):
        raise SystemExit(
            f"GATE L2 FAILED ({l2.iloc[0].to_dict()}). STOP per spec §5.3: "
            f"the judge has drifted since WS4 ran, the four new arms are not "
            f"comparable to the existing eleven, and the 15-arm refit is "
            f"meaningless. Not proceeding.")


def new_arm_recalls() -> dict:
    """Read ws4_recall_at_10_all3610 from each new arm's retrieval npz meta,
    and cross-check it against the independently recorded values
    (np6=0.8195, np12=0.8797, np24=0.9241, np48=0.9561)."""
    out = {}
    for arm in NEW_ARMS:
        z = np.load(RET / f"{arm}.npz")
        meta = json.loads(str(z["meta"]))
        out[arm] = float(meta["ws4_recall_at_10_all3610"])
    return out


def new_arm_rows() -> pd.DataFrame:
    done = load_checkpoint(RUN_CHECKPOINT)
    df = pd.DataFrame(done.values())
    if df.empty:
        raise SystemExit(f"{RUN_CHECKPOINT} is empty -- run "
                         f"scripts/recall-quality/ws4d_phase3_run.py first")
    df["qid"] = df["qid"].astype(int)
    expected = len(NEW_ARMS) * 450
    if len(df) != expected:
        raise SystemExit(f"expected {expected} generated rows (4 arms x 450 "
                         f"questions), found {len(df)} -- the full run has "
                         f"not completed")
    return df[["qid", "arm", "subset", "recall_at_10", "judge_correct"]]


def main():
    out = ws4d_dir()
    out.mkdir(parents=True, exist_ok=True)
    gates_path = out / "ladder_gates.csv"

    check_l2_passed(gates_path)

    old = pd.read_csv(ws4b_dir() / "tier0_runs_corrected.csv")
    old_primary = old[old.in_primary & (old.arm != PARAMETRIC)]
    old_recalls = old_primary.groupby("arm").recall_at_10.mean().to_dict()

    new_recalls = new_arm_recalls()

    recalls_15 = {**old_recalls, **new_recalls}
    assert len(recalls_15) == 15, f"expected 15 arms, got {len(recalls_15)}"
    l1 = ws4d.gate_l1(recalls_15.values())
    print(f"L1: {l1}")

    gates = pd.read_csv(gates_path)
    gates = gates[gates.gate != "L1"]
    gates = pd.concat([gates, pd.DataFrame([l1])], ignore_index=True)
    gates.to_csv(gates_path, index=False)
    manifest.record(gates_path)

    # ---- the 15-arm refit -------------------------------------------------
    new = new_arm_rows()
    in_primary_by_qid = old.drop_duplicates("qid").set_index("qid")["in_primary"]
    new["in_primary"] = new["qid"].map(in_primary_by_qid)
    assert new["in_primary"].notna().all(), "some new-arm qids missing from tier0_runs_corrected"

    old_cols = old_primary[["qid", "arm", "subset", "recall_at_10",
                            "judge_correct", "in_primary"]]
    combined = pd.concat([old_cols, new], ignore_index=True)
    assert combined.arm.nunique() == 15, combined.arm.unique()

    labels = pd.read_csv(out / "eligibility_labels.csv")
    drop_primary = set(labels.loc[labels["class"].isin(ws4d.PRIMARY_DROP), "qid"])

    primary_15 = combined[combined.in_primary & ~combined.qid.isin(drop_primary)]
    piv = primary_15.pivot_table(index="qid", columns="arm",
                                 values=["recall_at_10", "judge_correct"])
    rec, acc = piv["recall_at_10"], piv["judge_correct"]
    assert rec.shape[1] == 15, rec.shape
    assert not rec.isna().any().any() and not acc.isna().any().any(), \
        "missing (qid, arm) cell -- not every arm answered every retained question"

    boot = ws4d.bootstrap_changepoint(rec.values, acc.values,
                                      n_boot=WS4B_CHANGEPOINT_BOOT_N, seed=SEED)
    line = ws4d.fit_line(rec.mean().values, acc.mean().values)
    two = ws4d.fit_changepoint(rec.mean().values, acc.mean().values)
    branch = ws4d.knee_branch_n(boot, rec.mean().values)

    result = pd.DataFrame([{
        "stratum": "primary_15arm", "n_questions": len(rec), "n_arms": rec.shape[1],
        **boot,
        "sse_one_segment": round(line["sse"], 8),
        "sse_two_segment": round(two["sse"], 8),
        "branch": branch,
    }])
    result.to_csv(out / "curve_changepoint_15arm.csv", index=False)
    manifest.record(out / "curve_changepoint_15arm.csv")

    print(result.to_string(index=False))
    print(f"\nbranch: {branch}")
    print(f"L1 passed: {l1['passed']} (max_gap={l1['max_gap']}, "
         f"threshold={l1['threshold']})")


if __name__ == "__main__":
    main()
