#!/usr/bin/env python
"""WS8 review round: is the `hop1` separation an outlier artifact, and is the
test that reports it actually calibrated?

Output (sha256'd into data/MANIFEST.json):
  results/ws8/robustness.csv

Reads results/ws8/runs.csv and NOTHING else. No network, no model calls, no
spend: every row is deterministic local computation under `config.SEED`.
results/non-code-retrieval.md quotes these three figures, and this repo's rule is
that no number reaches a slide without a committed CSV behind it -- this file
is that CSV.

Three checks, one per `check` value in the CSV:

  leave_k_out -- the hostile question is "isn't the separation just the two or
      three biggest deltas?". For k = 1, 2, 3 this enumerates EVERY subset of
      size n-k (30, 435 and 4,060 of them) and reports the WORST lower bound
      any of them produces. k=0 is emitted too, as the anchor: one subset, the
      full sample, whose ci_low is the number paired_ci.csv publishes.

  paired_fpr -- the separation is only evidence if the test that found it fires
      at its nominal rate on data shaped like THIS data. `hop1`'s deltas are
      heavy-tailed (MAD ~100k, range to +1.35M), and a percentile bootstrap on
      a median is not guaranteed to be calibrated there. Measured, not assumed.

  contrast_fpr -- the same question for the PRIMARY test (the hop3plus-hop1
      contrast), under a true null built from the observed dispersion.

The null-pool choice for `contrast_fpr` is a real methodological fork, so all
three defensible pools are computed and the `null_pool` column names which is
which. `pooled_hop1_hop3plus` is the headline: under the null the two strata
ARE one distribution, so the pooled centred deltas are the natural estimate of
it, and it does not privilege hop1's heavier tail. The `per_stratum` row is the
construction already committed in mde_retrospective.csv's shift=0 row and is
reproduced here at this file's own parameters so the two can be compared.

A false-positive rate measured by simulation is itself an estimate: `mc_se` is
its binomial Monte Carlo standard error, so a reader can see that the third
decimal place is noise.

Run:  python scripts/non-code-retrieval/robustness_ws8.py     (deterministic, ~30s, $0)
"""

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.non_code_retrieval import stats as ws8_stats
from benchlib.code_retrieval import topk as ws6c
from benchlib.config import SEED, WS6C_ALPHA, WS8_STRATA, ws8_dir

# The stratum whose separation section 6.2 defends. hop3plus separates too, but
# H4 -- "the hop1 cell separating means the contrast cannot be read as a hop
# effect at all" -- is the branch that turns on hop1 alone, so hop1 is the cell
# a hostile question will aim at.
STRATUM = "hop1"
METRIC = "prompt_tokens"

# Leave-k-out enumerates C(30, k) subsets and runs a full bootstrap on each:
# 4,525 bootstraps in total at k <= 3. 2,000 resamples rather than
# config.WS6C_BOOT_N's 10,000 is a deliberate trade of bootstrap noise for
# EXHAUSTIVE enumeration -- a floor over all 4,060 subsets is a stronger
# statement than a floor over a sample of them, and the k=0 anchor below shows
# the two n_boot values agree on this data anyway (both give ci_low 17,402.0,
# the figure paired_ci.csv publishes at 10,000).
LEAVE_K_BOOT = 2_000
LEAVE_K_VALUES = (0, 1, 2, 3)

# False-positive rates. 5,000 simulated studies x 2,000 inner resamples puts the
# Monte Carlo standard error near 0.003 -- small enough to distinguish "about
# nominal" from "badly miscalibrated", which is all these rows are asked to do.
FPR_REPS = 5_000
FPR_BOOT = 2_000
# n=30 per powered cell is the pre-registered design (spec section 3), read from
# the pin rather than written out, so a change to the design cannot leave this
# simulation quietly measuring a different n from the one that was run.
FPR_N = WS8_STRATA[STRATUM]

# One column order for both checks, fixed here rather than left to dict
# insertion order, so the committed file's schema is a decision and not an
# artifact of which row type happened to be built first.
COLUMNS = (
    "check", "stratum", "metric", "role", "null_pool",
    "k", "n_retained", "n_subsets", "n_subsets_excluding_zero",
    "min_ci_low", "max_ci_low", "fpr", "mc_se",
    "n_per_cell", "n_reps", "n_boot", "seed", "alpha", "note",
)
INT_COLUMNS = ("k", "n_retained", "n_subsets", "n_subsets_excluding_zero",
               "n_per_cell", "n_reps", "n_boot", "seed")


def centred(deltas: np.ndarray) -> np.ndarray:
    """Deltas shifted to a median of exactly zero -- a true null with this
    sample's own shape and dispersion, which is the whole point: a null built
    from a normal or a t would measure the calibration of a test on data we did
    not collect."""
    d = np.asarray(deltas, dtype=float)
    return d - np.median(d)


def leave_k_out_rows(deltas, k_values=LEAVE_K_VALUES, *, n_boot=LEAVE_K_BOOT,
                     seed=SEED, alpha=WS6C_ALPHA) -> list[dict]:
    """Exhaustive leave-k-out floors on the paired bootstrap CI.

    EVERY subset, not a sample of them: the claim being defended is "no k
    observations can be removed to un-separate this cell", and that is a
    statement about the worst case, which only enumeration can establish.
    """
    d = np.asarray(deltas, dtype=float)
    n = int(d.size)
    rows = []
    for k in k_values:
        lows, n_excl = [], 0
        for drop in itertools.combinations(range(n), k):
            s = ws6c.paired_summary(np.delete(d, drop), n_boot=n_boot,
                                    seed=seed, alpha=alpha)
            lows.append(s["ci_low"])
            n_excl += bool(s["excludes_zero"])
        rows.append({
            "check": "leave_k_out",
            "stratum": STRATUM,
            "metric": METRIC,
            "role": "anchor" if k == 0 else "headline",
            "k": k,
            "n_retained": n - k,
            "n_subsets": len(lows),
            "n_subsets_excluding_zero": n_excl,
            "min_ci_low": float(min(lows)),
            "max_ci_low": float(max(lows)),
            "n_boot": n_boot,
            "seed": seed,
            "alpha": alpha,
            "note": ("ANCHOR: k=0 is the full sample; this ci_low is the "
                     "figure results/ws8/paired_ci.csv publishes"
                     if k == 0 else
                     f"every one of the {len(lows)} subsets of size {n - k}"),
        })
    return rows


def paired_fpr(pool, *, n=FPR_N, n_reps=FPR_REPS, n_boot=FPR_BOOT, seed=SEED,
               alpha=WS6C_ALPHA) -> float:
    """Fraction of null studies whose paired CI wrongly excludes zero.

    `pool` is already centred, so the truth is median = 0 and EVERY rejection
    is a false positive. Each study resamples n observations with replacement
    from the pool, then runs the same ws6c.paired_summary the real analysis
    ran.

    Each study gets a FRESH inner seed drawn from the outer generator. Passing
    the constant SEED instead would make every study reuse ONE bootstrap index
    matrix -- ws4.bootstrap_indices re-seeds from its seed argument on every
    call -- so the reported rate would be conditional on a single resampling
    draw rather than averaged over draws. The whole result stays reproducible
    because every inner seed descends from SEED.
    """
    p = np.asarray(pool, dtype=float)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_reps):
        sample = rng.choice(p, n)
        hits += ws6c.paired_summary(
            sample, n_boot=n_boot, seed=int(rng.integers(0, 2 ** 31 - 1)),
            alpha=alpha)["excludes_zero"]
    return hits / n_reps


def contrast_fpr(pool_a, pool_b, *, n_a=FPR_N, n_b=FPR_N, n_reps=FPR_REPS,
                 n_boot=FPR_BOOT, seed=SEED, alpha=WS6C_ALPHA) -> float:
    """Fraction of null studies whose CONTRAST CI wrongly excludes zero.

    Both pools are centred on their own medians, so median(a) - median(b) = 0
    in truth and every rejection is a false positive. Passing the SAME array as
    both pools is the pooled null; passing two different arrays keeps each
    cell's own dispersion. Fresh inner seeds, for the reason paired_fpr gives.
    """
    a = np.asarray(pool_a, dtype=float)
    b = np.asarray(pool_b, dtype=float)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_reps):
        sa, sb = rng.choice(a, n_a), rng.choice(b, n_b)
        hits += ws8_stats.contrast_summary(
            sa, sb, n_boot=n_boot, seed=int(rng.integers(0, 2 ** 31 - 1)),
            alpha=alpha)["excludes_zero"]
    return hits / n_reps


def mc_se(rate: float, n_reps: int) -> float:
    """Binomial Monte Carlo standard error of a simulated rate. Reported so the
    third decimal is visibly noise rather than read as precision."""
    return math.sqrt(rate * (1 - rate) / n_reps)


def fpr_rows(hop1: np.ndarray, hop3plus: np.ndarray) -> list[dict]:
    c1, c3 = centred(hop1), centred(hop3plus)
    pooled = np.concatenate([c1, c3])

    def row(check, null_pool, role, rate, note):
        return {
            "check": check,
            "stratum": STRATUM if check == "paired_fpr" else "hop3plus_vs_hop1",
            "metric": METRIC,
            "null_pool": null_pool,
            "role": role,
            "fpr": rate,
            "mc_se": round(mc_se(rate, FPR_REPS), 4),
            "n_per_cell": FPR_N,
            "n_reps": FPR_REPS,
            "n_boot": FPR_BOOT,
            "seed": SEED,
            # The SAME `alpha` column the leave_k_out rows carry: there it is
            # the CI's alpha, here it is the nominal rate `fpr` must be read
            # against. One column, because they are one number -- a separate
            # `nominal_alpha` invites a reader to think the test was run at a
            # different alpha from the one being compared to.
            "alpha": WS6C_ALPHA,
            "note": note,
        }

    rows = [row(
        "paired_fpr", "hop1_centred", "headline",
        paired_fpr(c1),
        "n=30 drawn with replacement from hop1's OWN deltas centred on their "
        "median; the test is ws6c.paired_summary, i.e. the one that fires H4",
    )]

    for null_pool, pa, pb, role, note in (
        ("pooled_hop1_hop3plus", pooled, pooled, "headline",
         "HEADLINE: under the null the two strata are ONE distribution, so "
         "both cells are drawn from the 60 pooled centred deltas"),
        ("hop1_centred", c1, c1, "sensitivity",
         "SENSITIVITY: both cells drawn from hop1's centred deltas alone -- "
         "the heavier-tailed of the two pools"),
        ("per_stratum_centred", c3, c1, "sensitivity",
         "SENSITIVITY: each cell keeps its own dispersion (hop3plus MAD is "
         "~0.4x hop1's); same construction as mde_retrospective.csv's "
         "injected_gradient_tokens=0 row, re-run at this file's parameters"),
    ):
        rows.append(row("contrast_fpr", null_pool, role,
                        contrast_fpr(pa, pb), note))
    return rows


def main() -> None:
    out = ws8_dir()
    runs = pd.read_csv(out / "runs.csv")
    hop1 = ws8_stats.stratum_deltas(runs, STRATUM, metric=METRIC)
    hop3plus = ws8_stats.stratum_deltas(runs, "hop3plus", metric=METRIC)
    assert hop1.size == WS8_STRATA[STRATUM], (
        f"hop1 has {hop1.size} paired deltas, spec section 3 registers "
        f"{WS8_STRATA[STRATUM]}; runs.csv is not the full run")

    rows = leave_k_out_rows(hop1) + fpr_rows(hop1, hop3plus)
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    # Nullable Int64, not float: the two checks populate different columns, and
    # NaN would otherwise render a count of 4,060 subsets as "4060.0".
    df = df.astype({c: "Int64" for c in INT_COLUMNS})
    path = out / "robustness.csv"
    df.to_csv(path, index=False)
    manifest.record(path)
    print(f"wrote {path}")

    lk = df[df["check"] == "leave_k_out"]
    print("\n=== leave-k-out floors on the hop1 paired CI "
          f"(n_boot={LEAVE_K_BOOT}, seed={SEED}, alpha={WS6C_ALPHA}) ===")
    print(lk[["k", "n_retained", "n_subsets", "n_subsets_excluding_zero",
              "min_ci_low", "max_ci_low"]].to_string(index=False))
    worst = lk[lk["k"] == max(LEAVE_K_VALUES)].iloc[0]
    if worst["n_subsets_excluding_zero"] == worst["n_subsets"]:
        print(f"ALL {int(worst['n_subsets'])} leave-{int(worst['k'])}-out "
              "subsets still exclude zero.")
    else:
        print(f"WARNING: {int(worst['n_subsets'] - worst['n_subsets_excluding_zero'])} "
              f"leave-{int(worst['k'])}-out subset(s) DO NOT exclude zero -- "
              "section 6.2's 'no k observations can be removed' wording is "
              "WRONG at this k and must be corrected.")

    fp = df[df["check"] != "leave_k_out"]
    print(f"\n=== false-positive rates (n_reps={FPR_REPS}, n_boot={FPR_BOOT}, "
          f"seed={SEED}, nominal alpha={WS6C_ALPHA}) ===")
    print(fp[["check", "null_pool", "role", "fpr", "mc_se"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
