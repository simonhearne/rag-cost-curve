#!/usr/bin/env python
"""results/ws4e/rstar_final.csv -- task 10(3): the consolidated final
record of every r*/tau estimate this project produced and its status, so a
reader never has to reconstruct it from four RESULTS files. $0 -- reads
only already-committed CSVs and asserts against the numbers this task was
briefed with rather than retyping them blind.

Sources (all committed, all frozen -- this script only reads them):
    results/ws4/rstar_sensitivity.csv    (WS4's own plateau r*, branch K1)
    results/ws4d/curve_changepoint.csv   (WS4d's 11-arm changepoint tau)
    results/ws4d/solvable_changepoint.csv (WS4d's 15-arm ladder, all_eligible row --
                                           the densest ladder the project built)
    results/ws4e/slope_by_k.csv          (documents changepoint_fitted=False on
                                           4 arms -- WS4e's own NOT_FITTED reason)

Design: results/recall-vs-quality.md
Run:
    PYTHONPATH=. .venv/bin/python scripts/recall-quality/ws4e_rstar_final.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest  # noqa: E402
from benchlib.config import ws4_dir, ws4d_dir, ws4e_dir  # noqa: E402


def approx(a, b, tol=1e-4):
    return abs(a - b) < tol


def main():
    ws4_sens = pd.read_csv(ws4_dir() / "rstar_sensitivity.csv")
    curve = pd.read_csv(ws4d_dir() / "curve_changepoint.csv")
    solvable = pd.read_csv(ws4d_dir() / "solvable_changepoint.csv")
    slope_by_k = pd.read_csv(ws4e_dir() / "slope_by_k.csv")

    sens = ws4_sens.set_index(["diagnostic", "statistic"])["value"]
    sens_ci_lo = ws4_sens.set_index(["diagnostic", "statistic"])["ci_lo"]
    sens_ci_hi = ws4_sens.set_index(["diagnostic", "statistic"])["ci_hi"]

    ws4_point = sens[("rstar", "point_estimate")]
    ws4_k1_share = sens[("rstar", "branch_K1_share_fixed_width")]
    ws4_undefined_share = sens[("rstar", "undefined_share_fixed_width")]
    ws4_median = sens[("rstar", "median_fixed_width")]
    ws4_median_lo = sens_ci_lo[("rstar", "median_fixed_width")]
    ws4_median_hi = sens_ci_hi[("rstar", "median_fixed_width")]
    ws4_p_within = sens[("rstar", "p_within_0.01_of_point_fixed_width")]

    # Verify against the values this task was briefed with, rather than
    # retype them blind (task instructions: "verify each against its
    # source -- do not retype from here without checking").
    assert approx(ws4_point, 0.9231111111111113)
    assert approx(ws4_k1_share, 0.6749)
    assert approx(ws4_undefined_share, 0.2787)
    assert approx(ws4_median, 0.9217777777777755)
    assert approx(ws4_median_lo, 0.8095555555555547)
    assert approx(ws4_median_hi, 0.9724444444444433)
    assert approx(ws4_p_within, 0.1904)

    c = curve[curve.stratum == "primary"].iloc[0]
    assert c.n_arms == 11 and c.n_questions == 264
    assert approx(c.tau, 0.789429) and approx(c.ci_lo, 0.756942)
    assert approx(c.ci_hi, 0.969948) and approx(c.width, 0.213005)
    assert c.branch == "N2"

    s = solvable[solvable.stratum == "all_eligible"].iloc[0]
    assert s.n_arms == 15 and s.n_questions == 264
    assert approx(s.tau, 0.789429)
    assert approx(round(s.width, 4), 0.2126)
    assert approx(round(s.slope, 3), 0.235) and approx(round(s.slope_ci_lo, 3), 0.122) \
        and approx(round(s.slope_ci_hi, 3), 0.353)
    assert s.branch == "N2"

    # WS4e: confirm the zero-residual-df fact this row's NOT_FITTED status
    # rests on is really what slope_by_k.csv records, rather than asserting
    # it from prose alone.
    assert (slope_by_k.n_arms == 4).all()
    assert (slope_by_k.residual_df == 2).all()
    assert (slope_by_k.changepoint_fitted == False).all()  # noqa: E712

    verdict_width = round(float(s.width), 4)
    verdict_ratio = round(verdict_width / 0.05, 2)
    assert approx(verdict_ratio, 4.25, tol=0.01)

    rows = [
        {
            "stage": "WS4",
            "n_arms": 11, "n_questions": 450,
            "rstar_or_tau": ws4_point,
            "ci_lo": ws4_median_lo, "ci_hi": ws4_median_hi,
            "ci_width": round(ws4_median_hi - ws4_median_lo, 10),
            "branch": "K1",
            "status": "WITHDRAWN",
            "source_csv": "results/ws4/rstar_sensitivity.csv",
            "note": (f"Spec 7.1 plateau rule on the frozen 11-arm/450-question "
                     f"ladder; branch K1 reproduces in only "
                     f"{ws4_k1_share:.4f} of 10,000 paired resamples "
                     f"(fixed_width CI), {ws4_undefined_share:.4f} of "
                     f"resamples find no plateau at all (branch K3, no r* "
                     f"exists), and only {ws4_p_within:.4f} of ALL "
                     f"resamples land within +/-0.01 of the published "
                     f"point estimate. ci_lo/ci_hi above are the "
                     f"median_fixed_width row's 95% resampling interval, "
                     f"not a CI on the point estimate itself -- no such CI "
                     f"exists because the estimator is a plateau rule, not "
                     f"a fit."),
        },
        {
            "stage": "WS4d 11-arm",
            "n_arms": int(c.n_arms), "n_questions": int(c.n_questions),
            "rstar_or_tau": c.tau,
            "ci_lo": c.ci_lo, "ci_hi": c.ci_hi, "ci_width": c.width,
            "branch": c.branch,
            "status": "SUPERSEDED",
            "source_csv": "results/ws4d/curve_changepoint.csv",
            "note": ("Two-segment continuous piecewise-linear changepoint "
                     "fit, primary 264-question stratum, 11 arms. N2: the "
                     "CI (width 0.213) is far above the 0.05 quotable bar "
                     "and/or touches the recall range's bounds -- no "
                     "knee is locatable. Superseded by the 15-arm ladder "
                     "below, built specifically to narrow this CI further."),
        },
        {
            "stage": "WS4d 15-arm",
            "n_arms": int(s.n_arms), "n_questions": int(s.n_questions),
            "rstar_or_tau": s.tau,
            "ci_lo": s.ci_lo, "ci_hi": s.ci_hi, "ci_width": s.width,
            "branch": s.branch,
            "status": "UNRESOLVED",
            "source_csv": "results/ws4d/solvable_changepoint.csv",
            "note": (f"The densest ladder the project built (all_eligible "
                     f"row, 15 arms, all 264 eligible questions). Still "
                     f"N2: width {s.width:.6f}, {verdict_ratio:.2f}x the "
                     f"0.05 quotable bar. Adding four more arms over the "
                     f"11-arm fit above moved the CI width by <0.001 -- "
                     f"tau is arm-limited (results/recall-vs-quality.md), not resolvable "
                     f"by adding more arms to this ladder. The slope "
                     f"replacement this stratum supports is quotable: "
                     f"{s.slope:.3f} [{s.slope_ci_lo:.3f}, "
                     f"{s.slope_ci_hi:.3f}] judge-accuracy points per unit "
                     f"recall."),
        },
        {
            "stage": "WS4e",
            "n_arms": 4, "n_questions": 264,
            "rstar_or_tau": "",
            "ci_lo": "", "ci_hi": "", "ci_width": "",
            "branch": "",
            "status": "NOT_FITTED",
            "source_csv": "results/ws4e/slope_by_k.csv",
            "note": ("4 arms against WS4d's 4-parameter two-segment "
                     "changepoint model is zero residual degrees of "
                     "freedom (n_arms=4, residual_df would be 0 for a "
                     "changepoint vs. 2 for the 2-parameter slope actually "
                     "fitted, per slope_by_k.csv's own "
                     "changepoint_fitted=False column) -- the fit would be "
                     "saturated, its CI undefined, and the number "
                     "numerology. No changepoint was fitted at any of the "
                     "four swept depths; a single slope of judge accuracy "
                     "on recall@10 was fitted instead (spec 6.4), and it "
                     "is not a tau/r* estimate."),
        },
        {
            "stage": "verdict",
            "n_arms": "", "n_questions": "",
            "rstar_or_tau": "", "ci_lo": "", "ci_hi": "",
            "ci_width": verdict_width,
            "branch": "",
            "status": "RETIRED",
            "source_csv": "results/ws4d/solvable_changepoint.csv",
            "note": (f"The quotable bar (WS4b spec 7.3, N1) was a CI width "
                     f"of 0.05. The narrowest width this project ever "
                     f"achieved is {verdict_width} (WS4d's 15-arm "
                     f"all_eligible ladder, the densest it could build) -- "
                     f"{verdict_ratio:.2f}x too wide. r* / tau is retired "
                     f"as a locatable quantity on this task; the quotable "
                     f"replacement is the slope, 0.235 [0.122, 0.353] "
                     f"judge-accuracy points per unit recall (same "
                     f"15-arm row)."),
        },
    ]
    out = pd.DataFrame(rows, columns=["stage", "n_arms", "n_questions",
                                      "rstar_or_tau", "ci_lo", "ci_hi",
                                      "ci_width", "branch", "status",
                                      "source_csv", "note"])
    out_dir = ws4e_dir()
    out.to_csv(out_dir / "rstar_final.csv", index=False)
    manifest.record(out_dir / "rstar_final.csv")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
