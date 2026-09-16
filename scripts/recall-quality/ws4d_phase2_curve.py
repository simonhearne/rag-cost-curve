#!/usr/bin/env python
"""WS4d Phase 2 -- the cleaned recall-vs-quality curve. $0, no API calls.

Design: results/recall-vs-quality.md

Gated on Phase 1: refuses to run unless eligibility_gates.csv shows E1-E4 all
passed, because a curve computed on labels that failed their gates is exactly
the number the spec's sequencing exists to prevent.

Run:  .venv/bin/python scripts/recall-quality/ws4d_phase2_curve.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.recall_quality import eligibility as ws4d
from benchlib.config import SEED, WS4B_CHANGEPOINT_BOOT_N, ws4b_dir, ws4d_dir


def main():
    out = ws4d_dir()
    out.mkdir(parents=True, exist_ok=True)

    gates = pd.read_csv(ws4d_dir() / "eligibility_gates.csv")
    if not gates.passed.all():
        raise SystemExit(f"Phase 1 gates not all passed:\n{gates.to_string()}\n"
                         f"Phase 2 does not run. See spec §9 branches W-C / W-D.")

    labels = pd.read_csv(ws4d_dir() / "eligibility_labels.csv")
    runs = pd.read_csv(ws4b_dir() / "tier0_runs_corrected.csv")
    prim = runs[runs.in_primary & (runs.arm != "parametric")]

    inel = labels.set_index("qid")["class"].to_dict()
    drop_primary = {q for q, c in inel.items() if c in ws4d.PRIMARY_DROP}
    drop_plus = drop_primary | {q for q, c in inel.items() if c == "unanswerable"}
    drop_disputed = drop_primary | {q for q, c in inel.items() if c == ws4d.DISPUTED}

    # spec §4.2 Secondary A: eligible questions on which EVERY arm answered.
    # Arm-independent by construction -- the same question set for all arms --
    # so no collider is introduced. NOT accuracy-given-that-arm-answered,
    # which conditions per-arm on the generator's own decision (§4.2, "not
    # used").
    answered_all = {q for q, g in prim.groupby("qid") if not g.idk_corrected.any()}

    STRATA = {
        "primary":        prim[~prim.qid.isin(drop_primary)],
        "plus_unanswerable": prim[~prim.qid.isin(drop_plus)],
        "disputed_included": prim[~prim.qid.isin(drop_disputed)],
        "secondary_a":    prim[~prim.qid.isin(drop_primary) & prim.qid.isin(answered_all)],
        "unfiltered_325": prim,
    }

    rows = []
    for name, g in STRATA.items():
        piv = g.pivot_table(index="qid", columns="arm",
                            values=["recall_at_10", "judge_correct"])
        rec, acc = piv["recall_at_10"], piv["judge_correct"]
        boot = ws4d.bootstrap_changepoint(rec.values, acc.values,
                                          n_boot=WS4B_CHANGEPOINT_BOOT_N, seed=SEED)
        line = ws4d.fit_line(rec.mean().values, acc.mean().values)
        two = ws4d.fit_changepoint(rec.mean().values, acc.mean().values)
        rows.append({"stratum": name, "n_questions": len(rec),
                     "n_arms": rec.shape[1], **boot,
                     "sse_one_segment": round(line["sse"], 8),
                     "sse_two_segment": round(two["sse"], 8),
                     "branch": ws4d.knee_branch_n(boot, rec.mean().values)})

    sens = pd.DataFrame(rows)
    sens.to_csv(out / "curve_sensitivities.csv", index=False)
    sens[sens.stratum == "primary"].to_csv(out / "curve_changepoint.csv", index=False)

    # Gate F1 counts, on the primary filtered stratum.
    kept = STRATA["primary"]
    wrong = kept[~kept.judge_correct]
    f1 = ws4d.gate_f1(n_declines=int(wrong.idk_corrected.sum()),
                      n_errors=int((~wrong.idk_corrected).sum()))
    pd.DataFrame([f1]).to_csv(out / "curve_gates.csv", index=False)

    # pre-flight ruling C3: spec §4.5 names five CSVs; write all five before
    # recording any. curve_points.csv is the chart's source (one row per
    # arm); curve_slope.csv is the single-slope fit the N-ladder reads.
    piv = STRATA["primary"].pivot_table(index="qid", columns="arm",
                                        values=["recall_at_10", "judge_correct"])
    pd.DataFrame({"arm": piv["recall_at_10"].columns,
                  "recall_at_10": piv["recall_at_10"].mean().values,
                  "judge_correct": piv["judge_correct"].mean().values}
                 ).sort_values("recall_at_10", ascending=False).to_csv(
                     out / "curve_points.csv", index=False)
    primary = sens[sens.stratum == "primary"].iloc[0]
    pd.DataFrame([{"stratum": "primary", "n_questions": int(primary.n_questions),
                   "n_arms": int(primary.n_arms), "slope": primary.slope,
                   "ci_lo": primary.slope_ci_lo, "ci_hi": primary.slope_ci_hi,
                   "excludes_zero": bool(primary.slope_ci_lo > 0),
                   "n_boot": int(primary.n_boot), "seed": int(primary.seed)}
                  ]).to_csv(out / "curve_slope.csv", index=False)

    for name in ("curve_sensitivities.csv", "curve_changepoint.csv",
                 "curve_gates.csv", "curve_points.csv", "curve_slope.csv"):
        manifest.record(out / name)

    print(sens.to_string())
    print(f"\nF1: {f1}")


if __name__ == "__main__":
    main()
