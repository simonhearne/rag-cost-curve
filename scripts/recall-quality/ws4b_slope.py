#!/usr/bin/env python
"""WS4's single-hop recall->quality slope on the CORRECTED primary stratum. $0.

WS4c spec §6.1 defines branches B1/B2 by comparing WS4c's multi-hop slope
against WS4's single-hop slope. That number was a placeholder ("+0.221 [to be
re-derived]") in the pre-registration, and a branch defined against a
placeholder is not pre-registered. This script derives it, once, from the
frozen Tier 0 output so the comparison has a committed CSV behind it.

Stratum: WS4b Tier 0's corrected primary stratum (325 questions, 11 retrieval
arms). Protocol: WS4's own -- paired bootstrap over QUESTIONS, seed 42,
WS4_N_BOOTSTRAP replicates, arms refit inside every replicate.

Run:  python scripts/recall-quality/ws4b_slope.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib.recall_quality import core
from benchlib.config import SEED, WS4_N_BOOTSTRAP, ws4b_dir

PARAMETRIC = "parametric"


def slope(acc: dict, rec: dict, arms: list) -> float:
    """OLS slope of accuracy on recall across arms. Each arm is one point;
    the bootstrap resamples the QUESTIONS underlying every point together, so
    the arms stay paired."""
    return float(np.polyfit(np.array([rec[a] for a in arms]),
                            np.array([acc[a] for a in arms]), 1)[0])


def main() -> None:
    out = ws4b_dir()
    r = pd.read_csv(out / "tier0_runs_corrected.csv")
    d = r[r.in_primary & (r.arm != PARAMETRIC)]
    acc = d.pivot(index="qid", columns="arm", values="judge_correct").astype(float)
    rec = d.pivot(index="qid", columns="arm", values="recall_at_10").astype(float)
    arms = list(acc.columns)

    point = slope({a: acc[a].mean() for a in arms},
                  {a: rec[a].mean() for a in arms}, arms)

    idx = core.bootstrap_indices(len(acc), WS4_N_BOOTSTRAP, SEED)
    av = {a: acc[a].to_numpy() for a in arms}
    rv = {a: rec[a].to_numpy() for a in arms}
    boot = np.array([slope({a: av[a][i].mean() for a in arms},
                           {a: rv[a][i].mean() for a in arms}, arms) for i in idx])
    lo, hi = core.percentile_ci(boot, 0.05)

    pd.DataFrame([{
        "stratum": "ws4b_tier0_corrected_primary",
        "n_questions": len(acc), "n_arms": len(arms),
        "slope": round(point, 6), "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
        "excludes_zero": bool(lo > 0), "n_boot": WS4_N_BOOTSTRAP, "seed": SEED,
    }]).to_csv(out / "tier0_slope.csv", index=False)

    per = pd.DataFrame([{"arm": a, "recall_at_10": round(float(rec[a].mean()), 6),
                         "judge_correct": round(float(acc[a].mean()), 6)}
                        for a in arms]).sort_values("recall_at_10", ascending=False)
    per.to_csv(out / "tier0_slope_points.csv", index=False)

    print(f"n={len(acc)} questions x {len(arms)} arms")
    print(f"slope b = {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
          f"(n_boot={WS4_N_BOOTSTRAP}, seed={SEED})")
    print(per.to_string(index=False))
    print(f"\nwrote {out}/tier0_slope.csv and tier0_slope_points.csv")


if __name__ == "__main__":
    main()
