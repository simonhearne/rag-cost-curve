#!/usr/bin/env python
"""WS4d Phase 0 -- the scoping evidence in the spec's section 2. $0, no API calls.

Design: results/recall-vs-quality.md

Everything here is computed from already-committed data and answers one
question asked BEFORE any of WS4d is run: would a changepoint fit on a
cleaned question set produce an r* worth quoting?

It deliberately writes NOTHING into results/. These are not results -- they
are the evidence for a pre-registration, and section 2 of the spec says so.
Run it to check the spec's tables; do not cite it as a finding.

Run:  python scripts/recall-quality/ws4d_scoping.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib.config import SEED, ws4b_dir

TAU_GRID = 200          # scoping resolution; the real run uses spec 6.1's joint fit
BOOT_CURVE = 2000       # scoping only -- WS4B_CHANGEPOINT_BOOT_N is 10_000
BOOT_POWER = 1200
SUBSAMPLE_SIZES = (80, 160, 240, 325)
ARTIFACT = ("ambiguous_question", "stale_gold", "unresolved")


def fit_tau(x, y, grid=TAU_GRID):
    """spec 6.1: q(r) = a + b1*min(r,tau) + b2*max(r-tau,0), tau free.
    Grid over tau with least squares at each -- exact for fixed tau."""
    best = None
    for tau in np.linspace(x.min() + 1e-6, x.max() - 1e-6, grid):
        X = np.column_stack([np.ones_like(x), np.minimum(x, tau),
                             np.maximum(x - tau, 0)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(((y - X @ beta) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, tau, beta)
    return best[1], best[2], best[0]


def fit_line(x, y):
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(((y - X @ beta) ** 2).sum()), beta[1]


def load():
    out = ws4b_dir()
    runs = pd.read_csv(out / "tier0_runs_corrected.csv")
    lab = pd.read_csv(out / "tier0_staleness_labelled.csv")
    prim = runs[runs.in_primary & (runs.arm != "parametric")]
    artifact = set(lab[lab.label.isin(ARTIFACT)].qid)
    return prim, artifact


def matrices(prim, drop):
    g = prim[~prim.qid.isin(drop)]
    piv = g.pivot_table(index="qid", columns="arm",
                        values=["recall_at_10", "judge_correct"])
    return piv["recall_at_10"], piv["judge_correct"]


def section_2_1_and_2_2(prim, artifact):
    print("\n2.1 / 2.2  where the existing data lands, and the fitted shape")
    print(f'{"stratum":34} {"n":>4} {"tau":>7} {"95% CI":>18} {"width":>7} '
          f'{"end?":>5} {"b1":>7} {"b2":>7} {"F":>5} {"branch":>7}')
    for name, drop in [("corrected primary", set()),
                       ("minus 54 in-stratum artifacts", artifact)]:
        rec, acc = matrices(prim, drop)
        x, y = rec.mean().values, acc.mean().values
        tau, beta, sse2 = fit_tau(x, y)
        sse1, slope1 = fit_line(x, y)
        n_arms = len(x)
        F = ((sse1 - sse2) / 2) / (sse2 / (n_arms - 4))
        rng = np.random.default_rng(SEED)
        taus, slopes = [], []
        for _ in range(BOOT_CURVE):
            idx = rng.choice(len(rec), len(rec), replace=True)   # ONE shared draw
            taus.append(fit_tau(rec.values[idx].mean(0), acc.values[idx].mean(0))[0])
            slopes.append(fit_line(rec.values[idx].mean(0), acc.values[idx].mean(0))[1])
        lo, hi = np.percentile(taus, [2.5, 97.5])
        slo, _ = np.percentile(slopes, [2.5, 97.5])
        ends = lo <= x.min() + 0.005 or hi >= x.max() - 0.005
        n1 = (hi - lo) <= 0.05 and not ends
        branch = "N1" if n1 else ("N3" if slo <= 0 else "N2")
        print(f"{name:34} {len(rec):>4} {tau:>7.3f} [{lo:.3f}, {hi:.3f}]{'':>4} "
              f"{hi-lo:>7.4f} {str(ends):>5} {beta[1]:>+7.3f} {beta[2]:>+7.3f} "
              f"{F:>5.2f} {branch:>7}")
    print("   a knee in C6's direction needs b1 STEEP and b2 FLAT.")


def section_2_3(prim):
    print("\n2.3  can more questions fix it?  tau CI width vs question count")
    rec, acc = matrices(prim, set())
    R, A = rec.values, acc.values
    rng = np.random.default_rng(SEED)
    widths = []
    for n in SUBSAMPLE_SIZES:
        reps = [fit_tau(R[i].mean(0), A[i].mean(0), grid=120)[0]
                for i in (rng.choice(len(R), n, replace=True)
                          for _ in range(BOOT_POWER))]
        lo, hi = np.percentile(reps, [2.5, 97.5])
        widths.append(hi - lo)
        print(f"   n = {n:>4}   width = {hi-lo:.4f}")
    p = np.polyfit(np.log(SUBSAMPLE_SIZES), np.log(widths), 1)
    print(f"   width ~ n^{p[0]:+.3f}   (pure sampling noise would be n^-0.500)")
    print("   => more questions cannot reach N1's 0.05. The exponent is the "
          "finding; the extrapolation is degenerate and is not quoted.")


def section_2_4(prim):
    print("\n2.4  what the interval IS limited by: leave one arm out")
    rec, acc = matrices(prim, set())
    arms = list(rec.columns)
    x, y = rec.mean().values, acc.mean().values
    t0, _, _ = fit_tau(x, y, grid=300)
    print(f"   tau on all {len(arms)} arms = {t0:.4f}")
    shifts = []
    for i, a in enumerate(arms):
        m = np.ones(len(arms), bool)
        m[i] = False
        t, _, _ = fit_tau(x[m], y[m], grid=300)
        shifts.append(t)
        if abs(t - t0) > 1e-9:
            print(f"     drop {a:24} tau = {t:.4f}  shift {t-t0:+.4f}")
    print(f"   leave-one-arm-out spread = {max(shifts)-min(shifts):.4f}")
    print("\n   adjacent-arm recall gaps:")
    o = np.argsort(x)
    for j in range(len(o) - 1):
        gap = x[o[j+1]] - x[o[j]]
        flag = "  <- hole" if gap > 0.035 else ("  <- near-duplicate" if gap < 0.005 else "")
        print(f"     {x[o[j]]:.4f} -> {x[o[j+1]]:.4f}   {gap:.4f}{flag}")


def section_4_4(prim, artifact):
    print("\n4.4  gate F1: after filtering, what is the residual wrongness made of?")
    kept = prim[~prim.qid.isin(artifact)]
    wrong = kept[~kept.judge_correct]
    dec = int(wrong.idk_corrected.sum())
    print(f"   across {kept.arm.nunique()} arms: wrong = {len(wrong)}, "
          f"declines = {dec} ({100*dec/len(wrong):.1f}%), errors = {len(wrong)-dec}")
    errs = prim[(~prim.judge_correct) & (~prim.idk_corrected)]
    share = errs.groupby("arm").apply(lambda g: g.qid.isin(artifact).mean(),
                                      include_groups=False)
    print(f"   share of each arm's ERROR rows removed by the filter: "
          f"{100*share.min():.1f}% - {100*share.max():.1f}%")
    print(f"   => F1's threshold is {0.80:.2f}; this analogue sits at "
          f"{dec/len(wrong):.3f} and F1 is EXPECTED TO FIRE.")


def main():
    prim, artifact = load()
    print("Scoping pass for the recall-vs-quality knee -- the methodology "
          "and every verdict are in results/recall-vs-quality.md")
    print("$0, no API calls, nothing written to results/. NOT a result.")
    section_2_1_and_2_2(prim, artifact)
    section_2_3(prim)
    section_2_4(prim)
    section_4_4(prim, artifact)


if __name__ == "__main__":
    main()
