#!/usr/bin/env python
"""WS8 Phase 4: the hop contrast, the H1-H4 branch lookup, and the secondaries.

Outputs (all sha256'd into data/MANIFEST.json):
  results/ws8/paired_ci.csv  -- per (variant, stratum, metric) paired summary
  results/ws8/branches.csv   -- the PRIMARY contrast and its branch, in both
                                pre-registered sensitivity variants
  results/ws8/summary.csv    -- per (stratum, arm) descriptives: the mechanism
                                (turns), dollars, accuracy, grep_handle
  results/ws8/mde_retrospective.csv -- POST-HOC: what gradient this design
                                could actually have detected, given the
                                dispersion it turned out to face

The PRIMARY test is the CONTRAST between hop3plus and hop1 on prompt tokens
(spec section 7, Phase 3) -- not either cell alone. Everything else in these
files is a declared secondary and cannot carry a claim by itself.

Two pre-registered handling rules are implemented here and nowhere else:

  R22 (spec section 7 amendment, 2026-09-11) -- ws8_q067 indexed hit the shared
      15-turn cap with an EMPTY answer. Its COST is kept and counts (dropping
      the index's worst case would bias the cost comparison toward the index);
      its ACCURACY is CENSORED, because the judge short-circuits an empty
      answer to False without evaluating content, so False there asserts a
      measurement never taken. The primary contrast is published BOTH with and
      without the row, and if the two disagree neither is preferred.

  hop2 is NOT powered -- n=15 by design (spec section 3). The `powered` column
      says so in the CSV itself, not only in the prose.

Run:  python scripts/non-code-retrieval/summarise_ws8.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.non_code_retrieval import stats as ws8_stats
from benchlib.code_retrieval import topk as ws6c
from benchlib.config import (
    SEED, WS6C_ALPHA, WS6C_BOOT_N, WS8_STRATA, WS8_TURN_CAP, ws8_dir,
)

# The censored row, named in the spec BEFORE the analysis ran. It is a
# constant, not a filter: nothing here may drop a row the spec did not name.
CENSORED_QID = "ws8_q067"
CENSORED_ARM = "indexed"

VARIANTS = {
    "all": (),                          # every pair, as run
    "excl_ws8_q067": (CENSORED_QID,),   # the declared R22 sensitivity
}

# prompt_tokens is the pre-registered outcome. turns and cost are SECONDARIES:
# declared secondary in advance and not claimable alone (spec section 7).
PRIMARY_METRIC = "prompt_tokens"
SECONDARY_METRICS = ("turns", "cost_total_billed")
METRICS = (PRIMARY_METRIC, *SECONDARY_METRICS)

NOT_PREREG = "n/a (secondary, no pre-registered branch)"

# spec section 3 DECLARED, in advance, that n=30 per cell powers a ~25,000-token
# effect at 80%. It is recorded in branches.csv so a null cannot be read as "no
# effect" rather than "none this design could see" -- but the declared figure is
# not the achieved one: mde_retrospective.csv re-runs section 3's own simulation
# at the dispersion actually observed and puts the power at this floor near
# alpha. branches.csv therefore carries the declared floor AND its measured
# power side by side, so the CSV refutes the optimistic reading by itself
# rather than needing a second file to do it.
DETECTABLE_FLOOR_TOKENS = 25_000

# The power a design is normally sized for, and the target `gradient_for_80pc_
# power` reports the smallest grid point reaching.
TARGET_POWER = 0.80

HOP2_NOTE = ("NOT POWERED: n=15 by design (spec section 3); reported as-is, "
             "cannot carry a claim")


def censored_mask(df: pd.DataFrame) -> pd.Series:
    """Rows whose judge_correct is CENSORED, not measured (R22)."""
    return (df["qid"].astype(str) == CENSORED_QID) & (df["arm"] == CENSORED_ARM)


def truthy(series: pd.Series) -> pd.Series:
    """Explicit boolean parse -- the SAME rule ws6c.metric_series applies to
    judge_correct, and the two must not diverge (a test pins them equal).

    `.astype(bool)` is correct here only by accident: pandas happens to read
    runs.csv's judge_correct and hit_turn_cap columns as bool dtype. ONE blank
    cell makes the column object/str, and astype(bool) then scores every
    non-empty string -- including "False" -- as True, i.e. 100% accuracy. That
    failure is silent and flatters us, which is the worst combination.
    """
    return series.astype(str).str.lower().isin(("true", "1"))


def per_stratum_rows(runs: pd.DataFrame) -> list[dict]:
    """Paired medians and CIs per (variant, stratum, metric)."""
    rows = []
    for variant, excluded in VARIANTS.items():
        for stratum in WS8_STRATA:
            for metric in METRICS:
                d = ws8_stats.stratum_deltas(runs, stratum, metric=metric,
                                             exclude_qids=excluded)
                s = ws6c.paired_summary(d, n_boot=WS6C_BOOT_N, seed=SEED,
                                        alpha=WS6C_ALPHA)
                # WS8_STRATA is the pre-registered n per cell, not just an
                # iteration order. ws6c.paired_deltas guards UNPAIRED qids; it
                # cannot see a MISSING pair, so a truncated or partially
                # resumed runs.csv would be summarised at a smaller n without
                # complaint. Asserted on the `all` variant only -- the
                # excl_ws8_q067 variant is legitimately one pair short.
                if variant == "all":
                    assert s["n"] == WS8_STRATA[stratum], (
                        f"{stratum}/{metric}: n={s['n']} pairs, spec section 3 "
                        f"registers {WS8_STRATA[stratum]}; runs.csv is "
                        f"incomplete -- do NOT publish this run")
                # Filtered by `excluded` too: otherwise the excl_ws8_q067 rows
                # report the all-pairs covariate. Cosmetic with one excluded
                # row; a trap the moment exclude_qids grows.
                sub = runs[(runs["stratum"] == stratum)
                           & ~runs["qid"].astype(str).isin(set(excluded))]
                rows.append({
                    "variant": variant,
                    "stratum": stratum,
                    "metric": metric,
                    "role": "primary-input" if metric == PRIMARY_METRIC
                            else "secondary",
                    "n": s["n"],
                    "powered": stratum != "hop2",
                    "median_delta": s["median"],
                    "mean_delta": s["mean"],
                    "ci_lo": s["ci_low"],
                    "ci_hi": s["ci_high"],
                    "excludes_zero": s["excludes_zero"],
                    "sign_p": s["sign_p"],
                    "n_indexed_dearer": s["n_positive"],
                    "n_indexed_cheaper": int((d < 0).sum()),
                    "mean_grep_handle": round(float(sub["grep_handle"].mean()), 4),
                    "median_grep_handle": round(float(sub["grep_handle"].median()), 4),
                    "note": HOP2_NOTE if stratum == "hop2" else "",
                })
    return rows


def power_at_gradient(mde: list[dict], gradient: int) -> float:
    """The measured power at one injected gradient, from THIS run's simulation.

    Looked up rather than hardcoded: a power figure copied into a second place
    is a power figure that will eventually disagree with the file it came from.
    """
    for row in mde:
        if row["injected_gradient_tokens"] == gradient:
            return row["power"]
    raise KeyError(f"no simulated row at a {gradient:,}-token gradient; "
                   f"MDE_SHIFTS has {[r['injected_gradient_tokens'] for r in mde]}")


def gradient_for_power(mde: list[dict], target: float) -> str:
    """Smallest simulated gradient reaching `target` power, or ">max" if none.

    Deliberately a grid lookup, not an interpolation: the grid points are the
    only gradients actually simulated, and a fitted curve would put a number on
    a slide that no simulation produced.
    """
    reaching = [r["injected_gradient_tokens"] for r in mde
                if r["power"] >= target]
    if reaching:
        return str(int(min(reaching)))
    return f">{int(max(r['injected_gradient_tokens'] for r in mde))}"


def contrast_rows(runs: pd.DataFrame, mde: list[dict]) -> list[dict]:
    """The primary contrast (and secondary contrasts), both variants.

    `mde` is this run's own retrospective power simulation (`mde_rows`), passed
    in rather than recomputed so the power figures printed beside the declared
    floor are the same numbers mde_retrospective.csv ships.
    """
    floor_power = power_at_gradient(mde, DETECTABLE_FLOOR_TOKENS)
    gradient_80 = gradient_for_power(mde, TARGET_POWER)
    rows = []
    for variant, excluded in VARIANTS.items():
        for metric in METRICS:
            a = ws8_stats.stratum_deltas(runs, "hop3plus", metric=metric,
                                         exclude_qids=excluded)
            b = ws8_stats.stratum_deltas(runs, "hop1", metric=metric,
                                         exclude_qids=excluded)
            c = ws8_stats.contrast_summary(a, b, n_boot=WS6C_BOOT_N, seed=SEED,
                                           alpha=WS6C_ALPHA)
            hop1 = ws6c.paired_summary(b, n_boot=WS6C_BOOT_N, seed=SEED,
                                       alpha=WS6C_ALPHA)
            hop3 = ws6c.paired_summary(a, n_boot=WS6C_BOOT_N, seed=SEED,
                                       alpha=WS6C_ALPHA)
            primary = metric == PRIMARY_METRIC
            rows.append({
                "variant": variant,
                "comparison": "hop3plus_minus_hop1",
                "metric": metric,
                "role": "PRIMARY" if primary else "secondary",
                "median": c["median"],
                "ci_lo": c["lo"],
                "ci_hi": c["hi"],
                "excludes_zero": c["excludes_zero"],
                "n_a_hop3plus": c["n_a"],
                "n_b_hop1": c["n_b"],
                "hop3plus_median": hop3["median"],
                "hop3plus_ci_lo": hop3["ci_low"],
                "hop3plus_ci_hi": hop3["ci_high"],
                "hop1_median": hop1["median"],
                "hop1_ci_lo": hop1["ci_low"],
                "hop1_ci_hi": hop1["ci_high"],
                "hop1_separates": hop1["excludes_zero"],
                "branch": (ws8_stats.resolve_hop_branch(c, hop1) if primary
                           else NOT_PREREG),
                # `branch = H4-confounded` ALONE reads as "there was a finding
                # and a confound spoils it". There was not: the contrast spans
                # zero, so with hop1 forced quiet the SAME pre-registered
                # lookup returns H2 -- a null. This column is that lookup, not
                # a post-hoc judgement: the only thing changed is the flag the
                # branch table already keys on.
                "branch_if_hop1_were_null": (
                    ws8_stats.resolve_hop_branch(c, {"excludes_zero": False})
                    if primary else NOT_PREREG),
                "declared_floor_tokens": (DETECTABLE_FLOOR_TOKENS if primary
                                          else ""),
                # ... and, in the same row, what that declared floor was
                # actually worth at the observed dispersion. Both computed by
                # this run, not copied from the prose.
                "observed_power_at_declared_floor": (floor_power if primary
                                                     else ""),
                "gradient_for_80pc_power": (gradient_80 if primary else ""),
            })
    return rows


def summary_rows(runs: pd.DataFrame, parametric: pd.DataFrame) -> list[dict]:
    """Per (stratum, arm) descriptives -- the mechanism and the dollars."""
    censored = censored_mask(runs)

    rows = []
    frames = [("indexed", runs), ("agentic", runs), ("parametric", parametric)]
    for arm, frame in frames:
        for stratum in WS8_STRATA:
            g = frame[(frame["arm"] == arm) & (frame["stratum"] == stratum)]
            if g.empty:
                continue
            # R22: accuracy is scored over MEASURED rows only. The censored row
            # stays in every cost and turn statistic below -- only the judged
            # denominator loses it.
            cens = int(censored.reindex(g.index, fill_value=False).sum())
            judged = g.loc[~censored.reindex(g.index, fill_value=False)]
            # metric_series returns qid -> value, so it is only correct on a
            # SINGLE-ARM frame: on the paired frame the two arms of one qid
            # collapse onto whichever row pandas saw last, and every arm then
            # reports its partner's tokens. `g` is one (arm, stratum) cell.
            tok = (np.fromiter(ws6c.metric_series(g, PRIMARY_METRIC).values(),
                               dtype=float)
                   if "input_tokens" in g else np.array([]))
            rows.append({
                "stratum": stratum,
                "arm": arm,
                "n": len(g),
                "powered": stratum != "hop2",
                "median_prompt_tokens": (float(np.median(tok)) if tok.size
                                         else ""),
                "mean_prompt_tokens": (round(float(tok.mean()), 1) if tok.size
                                       else ""),
                "median_turns": float(g["turns"].median()),
                "mean_turns": round(float(g["turns"].mean()), 3),
                "max_turns": int(g["turns"].max()),
                # Two columns, because they disagree: ws8_q035 reached 15
                # turns and still terminated at end_turn, so it is AT the cap
                # without having been cut off by it. The recorded flag is the
                # measurement; the count is arithmetic on turns.
                "n_hit_turn_cap": (int(truthy(g["hit_turn_cap"]).sum())
                                   if "hit_turn_cap" in g else 0),
                "n_turns_at_cap": int((g["turns"] >= WS8_TURN_CAP).sum()),
                "total_cost_usd": round(float(g["cost_total_billed"].sum()), 4),
                "mean_cost_usd": round(float(g["cost_total_billed"].mean()), 4),
                "max_cost_usd": round(float(g["cost_total_billed"].max()), 4),
                "n_judged": len(judged),
                "n_censored": cens,
                "accuracy_measured": (round(float(truthy(judged["judge_correct"])
                                                  .mean()), 4)
                                      if len(judged) else ""),
                "mean_grep_handle": (round(float(g["grep_handle"].mean()), 4)
                                     if "grep_handle" in g else ""),
                "note": HOP2_NOTE if stratum == "hop2" else "",
            })
    return rows


# Injected-shift grid for the retrospective MDE (spec section 3's own method,
# re-run on the dispersion actually observed). POST-HOC and labelled as such:
# it cannot and does not change the branch verdict, which is fixed by the
# contrast CI alone. It exists because section 7 declares a ~25,000-token
# detectable floor in advance, and a null is only honestly reportable if that
# declared floor is checked against the data rather than assumed.
# The 0 row is the file's OWN CALIBRATION, not padding: under a true null the
# reported "power" is the test's false-positive rate and must land near
# alpha=0.05. Without it every other row is an unchecked assertion, and
# this repo's rule is that no number reaches a slide without a CSV behind it --
# including the number that says the other numbers can be trusted.
# 125k and 150k exist to stop `gradient_for_power` reporting a grid artifact.
# Power is ~0.73 at 100,000 and ~0.98 at 200,000, so on the coarse grid the
# smallest point reaching 0.80 was 200,000 -- close to 2x the real crossing,
# and it would have been read on a slide as the achieved MDE. Added because the
# column is a DIAGNOSTIC of this design's resolution, so the grid's own
# resolution is part of the measurement; no verdict depends on it.
MDE_SHIFTS = (0, 25_000, 50_000, 100_000, 125_000, 150_000, 200_000, 400_000)
# Spec section 3's own simulation size, verbatim: S=400 studies x B=1,000
# resamples. The retrospective is only "the design's method re-run at the
# dispersion actually observed" if it uses the design's own S and B; a cheaper
# approximation of it would be a different claim.
MDE_STUDIES = 400
MDE_BOOT = 1000


def mde_rows(runs: pd.DataFrame) -> list[dict]:
    """Power of the primary contrast against an injected gradient, at the
    OBSERVED dispersion. Strata are centred on their own medians, so the only
    effect present is the one injected."""
    a = ws8_stats.stratum_deltas(runs, "hop3plus")
    b = ws8_stats.stratum_deltas(runs, "hop1")
    ac, bc = a - np.median(a), b - np.median(b)
    # MAD of the centred deltas: the dispersion this simulation is
    # parameterised BY. It belongs in the file, because the file's whole point
    # is that the observed dispersion is not what the design assumed, and that
    # should not need a second CSV to see.
    mad_a, mad_b = float(np.median(np.abs(ac))), float(np.median(np.abs(bc)))
    rng = np.random.default_rng(SEED)
    rows = []
    for shift in MDE_SHIFTS:
        hits = 0
        for _ in range(MDE_STUDIES):
            sa = rng.choice(ac, ac.size) - shift
            sb = rng.choice(bc, bc.size)
            # A FRESH inner seed per study, drawn from the outer rng. Passing
            # the constant SEED here makes every simulated study reuse ONE
            # bootstrap index matrix, so the reported power is conditional on a
            # single resampling draw rather than averaged over draws. The whole
            # grid stays reproducible: every inner seed descends from SEED.
            hits += ws8_stats.contrast_summary(
                sa, sb, n_boot=MDE_BOOT,
                seed=int(rng.integers(0, 2 ** 31 - 1)), alpha=WS6C_ALPHA
            )["excludes_zero"]
        rows.append({
            "role": "post-hoc diagnostic (does NOT affect the branch verdict)",
            "injected_gradient_tokens": shift,
            "n_a_hop3plus": int(ac.size),
            "n_b_hop1": int(bc.size),
            "power": round(hits / MDE_STUDIES, 3),
            "n_studies": MDE_STUDIES,
            "n_boot": MDE_BOOT,
            "declared_floor_tokens": DETECTABLE_FLOOR_TOKENS,
            "mad_hop3plus": round(mad_a, 1),
            "mad_hop1": round(mad_b, 1),
            "note": ("NULL CALIBRATION: no gradient injected, so this power IS "
                     "the false-positive rate; it must land near alpha=0.05"
                     if shift == 0 else ""),
        })
    return rows


def main() -> None:
    out = ws8_dir()
    runs = pd.read_csv(out / "runs.csv")
    parametric = pd.read_csv(out / "runs_parametric.csv")

    paired = pd.DataFrame(per_stratum_rows(runs))
    paired.to_csv(out / "paired_ci.csv", index=False)

    # The retrospective runs FIRST: branches.csv quotes its power at the
    # declared floor, and the two files must not be able to disagree.
    mde_data = mde_rows(runs)
    mde = pd.DataFrame(mde_data)
    mde.to_csv(out / "mde_retrospective.csv", index=False)

    branches = pd.DataFrame(contrast_rows(runs, mde_data))
    branches.to_csv(out / "branches.csv", index=False)

    summary = pd.DataFrame(summary_rows(runs, parametric))
    summary.to_csv(out / "summary.csv", index=False)

    for name in ("paired_ci.csv", "branches.csv", "summary.csv",
                 "mde_retrospective.csv"):
        manifest.record(out / name)

    # ------------------------------------------------------------- report
    print("\n=== per-stratum paired deltas (indexed - agentic), all pairs ===")
    show = paired[(paired["variant"] == "all")
                  & (paired["metric"] == PRIMARY_METRIC)]
    print(show[["stratum", "n", "powered", "median_delta", "ci_lo", "ci_hi",
                "excludes_zero", "sign_p", "n_indexed_cheaper",
                "mean_grep_handle"]].to_string(index=False))

    print("\n=== PRIMARY: hop3plus - hop1 on prompt tokens ===")
    prim = branches[branches["role"] == "PRIMARY"]
    for _, r in prim.iterrows():
        print(f"  [{r['variant']:>14s}] n={r['n_a_hop3plus']}/{r['n_b_hop1']}  "
              f"median {r['median']:+,.0f}  "
              f"CI [{r['ci_lo']:+,.0f}, {r['ci_hi']:+,.0f}]  "
              f"hop1_separates={bool(r['hop1_separates'])}  -> {r['branch']}")
    verdicts = sorted(set(prim["branch"]))
    if len(verdicts) > 1:
        print(f"  !! THE SENSITIVITY VARIANTS DISAGREE {verdicts}: spec section 7 "
              "says publish both and prefer neither.")
    else:
        print(f"  both sensitivity variants agree: {verdicts[0]}")
    would_be = sorted(set(prim["branch_if_hop1_were_null"]))
    if verdicts == ["H4-confounded"]:
        print("  hop1 separated; the contrast cannot be read as a hop effect.")
        # H4 alone reads as "a finding, spoiled by a confound". Say what the
        # same pre-registered lookup returns with hop1 forced quiet, so the
        # confound cannot be mistaken for the reason there is no claim.
        print(f"  absent H4 the SAME lookup returns {would_be}"
              + (" -- the contrast spans zero, so there was no finding for a "
                 "confound to spoil." if would_be == ["H2"] else "."))
    floor_power = float(prim["observed_power_at_declared_floor"].iloc[0])
    print(f"  spec section 3 DECLARED a ~{DETECTABLE_FLOOR_TOKENS:,}-token floor; "
          f"at the observed dispersion its measured power is {floor_power:.3f}. "
          f"80% needs a {prim['gradient_for_80pc_power'].iloc[0]}-token gradient.")
    if "H2" in verdicts or "H2" in would_be:
        print("  an H2 is a NULL, not an absence of effect -- and at this power "
              "it is barely even a null.")

    print("\n=== secondary: turns per arm per stratum (the mechanism) ===")
    piv = summary.pivot_table(index="stratum", columns="arm",
                              values="median_turns")
    print(piv.to_string())

    print("\n=== post-hoc: power of the primary contrast at OBSERVED dispersion ===")
    print(mde[["injected_gradient_tokens", "power", "mad_hop3plus",
               "mad_hop1"]].to_string(index=False))
    print("  the 0 row is the null calibration: it IS the false-positive rate.")

    print("\nwrote paired_ci.csv, branches.csv, summary.csv, "
          "mde_retrospective.csv -> {}".format(out))


if __name__ == "__main__":
    main()
