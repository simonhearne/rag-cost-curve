"""WS4e pure logic: does the discriminating share depend on retrieval depth?

Design: results/recall-vs-quality.md

Nothing here makes a network call or reads a file it was not handed. The
live half lives in depth_runner.py.
"""

from __future__ import annotations

import numpy as np

from . import eligibility as ws4d

STRATA = ("primary", "plus_unanswerable", "disputed_included")


def build_strata(labels_df, tier0_df) -> dict[str, list[int]]:
    """The three question sets of spec 3.1, all subsets of `primary`.

    `primary` reproduces results/recall-vs-quality.md's 264: the in_primary stratum
    (the gold is in the exact GT top-10) minus the questions WS4d Phase 1's
    blind labellers dropped as ambiguous / wrong_gold / stale_gold.

    The eligibility labels are reused UNCHANGED and are not a function of k:
    they were assigned against the arm-independent GT top-10, before this
    experiment existed.
    """
    in_primary = set(
        int(q) for q in
        tier0_df[tier0_df.in_primary & (tier0_df.arm != "parametric")].qid.unique())
    cls = dict(zip(labels_df.qid.astype(int), labels_df["class"]))
    drop = {q for q, c in cls.items() if c in ws4d.PRIMARY_DROP}
    primary = sorted(in_primary - drop)
    return {
        "primary": primary,
        "plus_unanswerable": sorted(
            set(primary) - {q for q, c in cls.items() if c == "unanswerable"}),
        "disputed_included": sorted(
            set(primary) - {q for q, c in cls.items() if c == ws4d.DISPUTED}),
    }


def bucket_of(correct_by_arm: dict, expected_arms=None) -> str:
    """always / never / discriminating, exactly as results/recall-vs-quality.md
    defines them, over whatever arm set is handed in.

    An incomplete row is an error, never a bucket: a missing arm would turn
    'always' into 'discriminating' silently. `expected_arms` makes that
    check explicit -- pass an int (the expected count) or a set/iterable of
    arm names to reject a row that does not match; None (the default)
    preserves the old behaviour for callers that only want the empty-row
    guard. `decompose` is the caller that should actually pass this: it
    knows the real arm set from the other rows.
    """
    vals = list(correct_by_arm.values())
    if not vals:
        raise ValueError("no verdicts: cannot assign a bucket")
    if expected_arms is not None:
        if isinstance(expected_arms, int):
            if len(correct_by_arm) != expected_arms:
                raise ValueError(
                    f"expected {expected_arms} arms, got {len(correct_by_arm)}: "
                    f"{sorted(correct_by_arm)}")
        else:
            want = set(expected_arms)
            got = set(correct_by_arm)
            if got != want:
                raise ValueError(
                    f"expected arms {sorted(want)}, got {sorted(got)}")
    n_true = sum(1 for v in vals if v)
    if n_true == len(vals):
        return "always"
    if n_true == 0:
        return "never"
    return "discriminating"


def separable_at(ids_by_arm: dict, qids, k: int) -> dict[int, bool]:
    """qid -> do the arms' top-k id lists DIFFER?

    Spec 2.2: two arms handing the generator the same passage list are not
    being compared, so a question below this line cannot discriminate for any
    retrieval reason. Spec 2.3 is why it is a floor rather than a ceiling --
    the generator samples, so identical prompts can still disagree.
    """
    out = {}
    for q in qids:
        lists = {tuple(int(x) for x in ids_by_arm[a][int(q)][:k] if int(x) >= 0)
                 for a in ids_by_arm}
        out[int(q)] = len(lists) > 1
    return out


def decompose(verdicts: dict, separable: dict, qids) -> dict:
    """Spec 6.1's four statistics for one (arm set, depth) cell.

    `null` and `signal` are None when no question has identical passages:
    reporting a noise floor of 0.0 measured on zero questions would be a lie.

    This is the function that actually knows the cell's arm set, so it is
    the real guard against a ragged verdicts dict: the arm set is derived
    from the first question and every other question must match it exactly,
    or a partial row (e.g. 3 of 4 arms) would silently mislabel a bucket
    instead of raising.
    """
    qids = [int(q) for q in qids]
    arm_set = frozenset(verdicts[qids[0]]) if qids else frozenset()
    for q in qids:
        if frozenset(verdicts[q]) != arm_set:
            raise ValueError(
                f"ragged verdicts: qid {q} has arms {sorted(verdicts[q])}, "
                f"expected {sorted(arm_set)}")
    buckets = {q: bucket_of(verdicts[q], expected_arms=arm_set) for q in qids}
    counts = {b: sum(1 for q in qids if buckets[q] == b)
              for b in ("always", "never", "discriminating")}
    sep = [q for q in qids if separable[q]]
    ident = [q for q in qids if not separable[q]]
    sep_rate = (sum(1 for q in sep if buckets[q] == "discriminating") / len(sep)
                if sep else None)
    null = (sum(1 for q in ident if buckets[q] == "discriminating") / len(ident)
            if ident else None)
    return {
        "n": len(qids),
        "always": counts["always"],
        "never": counts["never"],
        "discriminating": counts["discriminating"],
        "share": counts["discriminating"] / len(qids),
        "n_separable": len(sep),
        "n_identical": len(ident),
        "separable_rate": sep_rate,
        "null": null,
        "signal": (None if (sep_rate is None or null is None)
                   else sep_rate - null),
        "buckets": buckets,
    }


def _signal_from(bucket_map, sep_map, draw) -> float | None:
    """signal on one resampled question index. `draw` may repeat questions."""
    ds = ns = di = ni = 0
    for q in draw:
        disc = bucket_map[q] == "discriminating"
        if sep_map[q]:
            ns += 1
            ds += disc
        else:
            ni += 1
            di += disc
    if ns == 0 or ni == 0:
        return None
    return ds / ns - di / ni


def bootstrap_signal(bucket_by_k, separable_by_k, qids, n_boot: int,
                     seed: int) -> dict:
    """Paired bootstrap over QUESTIONS -- one shared index draw per replicate,
    so every depth is resampled on the same questions (WS4d section 4.3's
    machinery, unchanged).

    ⚠️ Docstring corrected in fix round 1 (S7): the PAIR (k_lo, k_hi) is
    chosen ONCE, from the point estimates, before the bootstrap loop --
    NOT re-selected as an argmax inside each replicate (that would make
    |max diff| positive by construction and its CI could never exclude
    zero, which is the degenerate reading of spec 4.1 the code deliberately
    avoids). What IS recomputed inside every replicate is that FIXED pair's
    difference, on the replicate's own shared draw, which is what makes
    spec 4.1's CI a real (non-post-hoc) interval around one pre-committed
    comparison rather than two independently-built per-depth intervals
    subtracted after the fact.
    """
    qids = [int(q) for q in qids]
    ks = sorted(bucket_by_k)
    point = {k: _signal_from(bucket_by_k[k], separable_by_k[k], qids)
             for k in ks}
    pairs = [(a, b) for i, a in enumerate(ks) for b in ks[i + 1:]]
    k_lo, k_hi = max(pairs, key=lambda p: abs(point[p[1]] - point[p[0]]))
    rng = np.random.default_rng(seed)
    n_q = len(qids)
    per_k = {k: np.empty(n_boot) for k in ks}
    maxdiff = np.empty(n_boot)
    for i in range(n_boot):
        draw = [qids[j] for j in rng.choice(n_q, n_q, replace=True)]
        vals = {k: _signal_from(bucket_by_k[k], separable_by_k[k], draw)
                for k in ks}
        for k in ks:
            per_k[k][i] = np.nan if vals[k] is None else vals[k]
        maxdiff[i] = (np.nan if None in (vals[k_lo], vals[k_hi])
                      else vals[k_hi] - vals[k_lo])
    lo, hi = np.nanpercentile(maxdiff, [2.5, 97.5])
    return {
        "signal_by_k": {k: point[k] for k in ks},
        "ci_by_k": {k: tuple(round(float(v), 6) for v in
                             np.nanpercentile(per_k[k], [2.5, 97.5]))
                    for k in ks},
        "max_pair": round(abs(point[k_hi] - point[k_lo]), 6),
        "max_pair_ci_lo": round(float(min(lo, hi)), 6),
        "max_pair_ci_hi": round(float(max(lo, hi)), 6),
        "k_lo": k_lo, "k_hi": k_hi,
        "n_boot": n_boot, "seed": seed,
    }


def bootstrap_all_pairs_signal(bucket_by_k, separable_by_k, qids, n_boot: int,
                               seed: int) -> list[dict]:
    """Fix round 1, S7: a Bonferroni-corrected CI for EVERY pairwise
    signal(k) difference, not just the single pre-registered comparison
    `bootstrap_signal` reports.

    A SEPARATE bootstrap run from `bootstrap_signal` (same shared
    per-replicate question draw pattern, same seed -- `bootstrap_signal`
    does not expose its per-replicate per-k values, so this recomputes
    them rather than reusing an internal we chose not to return). With m
    pairs, the two-sided per-pair CI is widened to level 1 - 0.05/m so
    that calling more than one of the six comparisons "significant" does
    not inflate the family-wise false-positive rate.
    """
    qids = [int(q) for q in qids]
    ks = sorted(bucket_by_k)
    point = {k: _signal_from(bucket_by_k[k], separable_by_k[k], qids)
             for k in ks}
    pairs = [(a, b) for i, a in enumerate(ks) for b in ks[i + 1:]]
    m = len(pairs)
    alpha_bonf = 0.05 / m
    lo_pct, hi_pct = 100 * alpha_bonf / 2, 100 * (1 - alpha_bonf / 2)
    rng = np.random.default_rng(seed)
    n_q = len(qids)
    per_k = {k: np.empty(n_boot) for k in ks}
    for i in range(n_boot):
        draw = [qids[j] for j in rng.choice(n_q, n_q, replace=True)]
        for k in ks:
            v = _signal_from(bucket_by_k[k], separable_by_k[k], draw)
            per_k[k][i] = np.nan if v is None else v
    rows = []
    for a, b in pairs:
        diff = per_k[b] - per_k[a]
        lo, hi = np.nanpercentile(diff, [lo_pct, hi_pct])
        lo, hi = float(min(lo, hi)), float(max(lo, hi))
        rows.append({
            "k_lo": a, "k_hi": b,
            "diff": round(point[b] - point[a], 6),
            "n_comparisons": m,
            "ci_level_pct": round(100 - 2 * lo_pct, 4),
            "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
            "significant": bool(lo > 0 or hi < 0),
        })
    return rows


def fixed_composition_signal(bucket_by_k, fixed_separable_qids,
                             fixed_identical_qids) -> dict:
    """Fix round 1, CRITICAL 1: per-k discrimination rates on a population
    held FIXED across k, rather than on separable(k)/identical(k)'s own
    k-dependent sets.

    separable(k) is nested and monotone in k (98 subset 172 subset 206
    subset 241), so signal(k)'s denominator fills up with different
    questions as k grows -- a question can enter `separable` at some k
    without ever leaving. `signal(k)` therefore mixes a fixed population's
    trend with a composition-driven fall as it is diluted by newcomers.
    This is a DIAGNOSTIC only: it does not replace, gate, or change
    signal(k), the branch rule, or any committed number (same posture as
    results/recall-vs-quality.md's exploratory cut).
    """
    fixed_sep = sorted(int(q) for q in fixed_separable_qids)
    fixed_ident = sorted(int(q) for q in fixed_identical_qids)
    out = {}
    for k, buckets in bucket_by_k.items():
        sep_rate = float(np.mean([buckets[q] == "discriminating"
                                  for q in fixed_sep])) if fixed_sep else None
        ident_rate = float(np.mean([buckets[q] == "discriminating"
                                    for q in fixed_ident])) if fixed_ident else None
        out[k] = {
            "fixed_separable_rate": sep_rate,
            "fixed_identical_rate": ident_rate,
            "fixed_signal": (None if sep_rate is None or ident_rate is None
                             else sep_rate - ident_rate),
        }
    return out


def bootstrap_fixed_composition_signal(bucket_by_k, fixed_separable_qids,
                                       fixed_identical_qids, n_boot: int,
                                       seed: int) -> dict:
    """CI for the fixed-composition signal's largest pairwise difference --
    same paired-bootstrap posture as `bootstrap_signal` (one shared draw of
    the FIXED populations per replicate, so every k in that replicate is
    evaluated on the same resampled questions), applied to the diagnostic
    of `fixed_composition_signal` instead of the committed statistic.

    ⚠️ Task 10 SHOULD FIX 6 -- pair-selection parity with `bootstrap_signal`,
    documented here because it was previously documented only on that
    function: the compared pair (k_lo, k_hi) is chosen ONCE below, from the
    POINT estimates, before the bootstrap loop -- exactly like
    `bootstrap_signal`, and NOT re-selected as an argmax inside each
    replicate. On this data the pair is (1, 3), not the committed
    statistic's (1, 5) -- the two diagnostics need not agree on which pair
    is largest, and don't. The resulting CI, `[-0.2223, 0.1815]`, is driven
    mostly by the 23-question fixed-identical term: resampling ONLY the
    98-question fixed-separable set (holding the 23 fixed) yields a CI
    about half as wide as resampling both, while resampling ONLY the
    23-question set (holding the 98 fixed) recovers most of the full width
    -- the smaller population, not the larger one, sets the CI's precision.
    """
    fixed_sep = [int(q) for q in fixed_separable_qids]
    fixed_ident = [int(q) for q in fixed_identical_qids]
    ks = sorted(bucket_by_k)
    point = fixed_composition_signal(bucket_by_k, fixed_sep, fixed_ident)
    pairs = [(a, b) for i, a in enumerate(ks) for b in ks[i + 1:]]
    k_lo, k_hi = max(
        pairs, key=lambda p: abs(point[p[1]]["fixed_signal"]
                                  - point[p[0]]["fixed_signal"]))
    rng = np.random.default_rng(seed)
    n_sep, n_ident = len(fixed_sep), len(fixed_ident)
    maxdiff = np.empty(n_boot)
    for i in range(n_boot):
        draw_sep = [fixed_sep[j] for j in rng.integers(0, n_sep, n_sep)]
        draw_ident = [fixed_ident[j] for j in rng.integers(0, n_ident, n_ident)]
        vals = {}
        for k in ks:
            b = bucket_by_k[k]
            sr = np.mean([b[q] == "discriminating" for q in draw_sep])
            ir = np.mean([b[q] == "discriminating" for q in draw_ident])
            vals[k] = sr - ir
        maxdiff[i] = vals[k_hi] - vals[k_lo]
    lo, hi = np.percentile(maxdiff, [2.5, 97.5])
    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    return {"point": point, "k_lo": k_lo, "k_hi": k_hi,
            "max_pair": round(abs(point[k_hi]["fixed_signal"]
                                  - point[k_lo]["fixed_signal"]), 6),
            "max_pair_ci_lo": round(lo, 6), "max_pair_ci_hi": round(hi, 6),
            "n_boot": n_boot, "seed": seed}


def fixed_population_membership_check(bucket_by_k, qids) -> dict:
    """For ONE fixed set of question ids, the
    discriminating COUNT and its overlap against the k=1 (lowest-k)
    discriminating set, at every depth.

    A flat RATE on a fixed population (`fixed_composition_signal`) does not
    mean the same questions discriminate at every depth -- the rate can be
    identical while the underlying set churns. This answers the membership
    question the rate cannot: on the committed data, the fixed-98
    always-separable set discriminates on 47 questions at k=1 and 47 at
    k=10, but only 32 of those are the SAME questions (15 drop out, 15
    enter); the fixed-23 always-identical set discriminates on 2 at k=1 and
    2 at k=10, with ZERO overlap.
    """
    qids = [int(q) for q in qids]
    ks = sorted(bucket_by_k)
    k_ref = ks[0]
    disc = {k: {q for q in qids if bucket_by_k[k][q] == "discriminating"}
            for k in ks}
    ref = disc[k_ref]
    return {k: {"n_disc": len(disc[k]),
                "n_disc_overlap_vs_k1": len(disc[k] & ref)}
            for k in ks}


def fixed_population_rate(bucket_by_k, qids) -> dict:
    """Task 10 SHOULD FIX 7: the per-k discriminating RATE on one fixed set
    of question ids -- no null subtraction, unlike `fixed_composition_signal`
    which pairs a separable set against an identical one. Lets a population
    defined at ONE k (e.g. the 241 questions separable at k=10, or the 143
    of those that are newcomers relative to the fixed-98) be read back at
    every OTHER k too, rather than only at the k where it was defined.
    """
    qids = [int(q) for q in qids]
    return {k: float(np.mean([b[q] == "discriminating" for q in qids]))
            for k, b in bucket_by_k.items()}


def retrieval_attributable_excess(signal_by_k: dict, n_separable_by_k: dict) -> dict:
    """Fix round 1, CRITICAL 1 point 2: signal(k) x n_separable(k) -- the
    ABSOLUTE count of questions the corrected rate implies carry retrieval
    signal, as opposed to signal(k)'s RATE. Reported alongside because the
    two can (and here do) point in opposite directions as k varies: the
    rate falls while the count that rate is applied to grows."""
    return {k: (None if signal_by_k[k] is None
               else round(signal_by_k[k] * n_separable_by_k[k], 4))
            for k in signal_by_k}


def evaluate_p1(cells_by_k: dict) -> dict:
    """Spec 4, prediction P1: the always-correct bucket shrinks as k falls,
    i.e. is non-decreasing as k rises: always(1) <= always(3) <= always(5)
    <= always(10). Never evaluated in code before fix round 1 (S4)."""
    ks = sorted(cells_by_k)
    always = {k: cells_by_k[k]["always"] for k in ks}
    holds = all(always[ks[i]] <= always[ks[i + 1]] for i in range(len(ks) - 1))
    return {"prediction": "P1", "holds": bool(holds), "always_by_k": always}


def evaluate_p2(cells_by_k: dict) -> dict:
    """Spec 4, prediction P2: the never-correct bucket grows as k falls,
    i.e. is non-increasing as k rises: never(1) >= never(3) >= never(5) >=
    never(10). Never evaluated in code before fix round 1 (S4)."""
    ks = sorted(cells_by_k)
    never = {k: cells_by_k[k]["never"] for k in ks}
    holds = all(never[ks[i]] >= never[ks[i + 1]] for i in range(len(ks) - 1))
    return {"prediction": "P2", "holds": bool(holds), "never_by_k": never}


def gate_k5(*, spent: float, limit: float) -> dict:
    """K5: the governor's own budget line (spec 7). Recorded even when it
    never fired -- omitting a gate because it did not trip is the same
    under-reporting Phase 0's by-construction K1 committed."""
    return {"gate": "K5", "spent": round(spent, 6), "limit": limit,
            "passed": bool(spent <= limit)}


def inverted_fraction(verdicts: dict, separable: dict, worst_arm: str,
                      best_arm: str, qids) -> float | None:
    """Fix round 1, IMPORTANT 3: among DISCRIMINATING AND SEPARABLE
    questions (an inversion requires the passages to actually differ), the
    share where the WORST arm is right and the BEST arm is wrong --
    inverted relative to what recall@10 would predict. `null(k)` cannot
    subtract this: it only removes generator variance measured on
    IDENTICAL-passage questions, which by construction cannot exhibit a
    retrieval-driven inversion.
    """
    qids = [int(q) for q in qids]
    disc_sep = [q for q in qids
               if separable[q] and len({v for v in verdicts[q].values()}) > 1]
    if not disc_sep:
        return None
    inverted = sum(1 for q in disc_sep
                   if verdicts[q][worst_arm] and not verdicts[q][best_arm])
    return round(inverted / len(disc_sep), 6)


def bootstrap_discrimination_stats(bucket_by_k, separable_by_k, qids,
                                   n_boot: int, seed: int) -> dict:
    """Fix round 1, S2: 95% bootstrap CIs for share(k), null(k) and
    separable_rate(k) -- discrimination_by_k.csv previously carried a CI
    for signal(k) only. Same paired-bootstrap posture as `bootstrap_signal`
    (one shared per-replicate question draw), but its OWN RNG stream: kept
    separate rather than threaded through bootstrap_signal so that function
    is untouched (fix round 1 explicitly forbids changing signal(k) or the
    branch rule).

    null(10) is measured on only 23 questions and its CI is expected to be
    the widest of the four depths for exactly that reason (spec 12).
    """
    qids = [int(q) for q in qids]
    ks = sorted(bucket_by_k)
    rng = np.random.default_rng(seed)
    n_q = len(qids)
    share_b = {k: np.empty(n_boot) for k in ks}
    sep_b = {k: np.empty(n_boot) for k in ks}
    null_b = {k: np.empty(n_boot) for k in ks}
    for i in range(n_boot):
        draw = [qids[j] for j in rng.choice(n_q, n_q, replace=True)]
        for k in ks:
            b, s = bucket_by_k[k], separable_by_k[k]
            share_b[k][i] = np.mean([b[q] == "discriminating" for q in draw])
            sep_qs = [q for q in draw if s[q]]
            ident_qs = [q for q in draw if not s[q]]
            sep_b[k][i] = (np.mean([b[q] == "discriminating" for q in sep_qs])
                           if sep_qs else np.nan)
            null_b[k][i] = (np.mean([b[q] == "discriminating" for q in ident_qs])
                           if ident_qs else np.nan)
    out = {}
    for k in ks:
        out[k] = {
            "share_ci": tuple(round(float(v), 6) for v in
                              np.nanpercentile(share_b[k], [2.5, 97.5])),
            "separable_rate_ci": tuple(round(float(v), 6) for v in
                                       np.nanpercentile(sep_b[k], [2.5, 97.5])),
            "null_ci": tuple(round(float(v), 6) for v in
                             np.nanpercentile(null_b[k], [2.5, 97.5])),
        }
    return out


def flatness_branch(boot: dict, min_diff: float) -> str:
    """Spec 4.1. NON-flat requires BOTH a CI excluding zero AND a difference
    of at least min_diff. Requiring both biases the test against this
    project's own hypothesis, by design.
    """
    excludes_zero = (boot["max_pair_ci_lo"] > 0) or (boot["max_pair_ci_hi"] < 0)
    return ("non_flat"
            if excludes_zero and abs(boot["max_pair"]) >= min_diff
            else "flat")


def peak_branch(signal_by_k: dict, k_values) -> str:
    """Spec 9, given a non-flat result: where does signal(k) peak?"""
    ks = sorted(k_values)
    peak = max(ks, key=lambda k: signal_by_k[k])
    if peak == ks[0]:
        return "X-B"
    if peak == ks[-1]:
        return "X-C"
    return "X-A"


def gate_k1(qids_by_cell: dict, *, qids, arms, arms_by_cell: dict,
            x_axis: str) -> dict:
    """K1: every (arm, k) cell measured on the SAME questions and the SAME
    arms, with one x-axis. WS4d's interim ladder report failed exactly this
    by comparing recall over 3,610 sweep queries against a 325-question
    stratum, so the basis is recorded in the CSV rather than assumed.
    """
    want_q, want_a = sorted(int(q) for q in qids), tuple(arms)
    bad = [c for c, qs in qids_by_cell.items()
           if sorted(int(q) for q in qs) != want_q]
    bad += [c for c, a in arms_by_cell.items() if tuple(a) != want_a]
    return {"gate": "K1", "passed": not bad,
            "n_cells": len(qids_by_cell),
            "basis": f"{len(want_q)} qids x {len(want_a)} arms; x-axis {x_axis}",
            "offending_cells": sorted(str(c) for c in set(bad))}


def gate_k2(ids_by_arm: dict, scores_by_arm: dict, *, qids, committed: dict,
            top_k: int) -> dict:
    """K2: is a k-prefix of the cache really what a limit=k search returns?

    ⚠️ Asserting that `ids[:k]` is a prefix of `ids` would be tautological.
    The claim's real content is two things the cache CAN be wrong about:

    1. **Score ordering.** A prefix equals a limit=k result only because the
       cached ids descend by score. An arm whose scores ascend anywhere
       breaks the whole design.
    2. **Provenance.** The cache must be what the committed rows were
       generated from, or every reused k=10 verdict is attached to the wrong
       passages.

    `committed`: (qid, arm) -> retrieved_ids as recorded on a committed row.
    """
    ordered, bad_order = 0, []
    for a, sc in scores_by_arm.items():
        rows = np.asarray(sc)[[int(q) for q in qids]]
        if np.all(np.diff(rows, axis=1) <= 1e-6):
            ordered += 1
        else:
            bad_order.append(a)
    bad_shape = [a for a, ids in ids_by_arm.items()
                 if np.asarray(ids).shape[1] != top_k]
    mismatched = [key for key, got in committed.items()
                  if [int(x) for x in got]
                  != [int(x) for x in ids_by_arm[key[1]][int(key[0])]
                      if int(x) >= 0]]
    return {"gate": "K2",
            "passed": not bad_order and not bad_shape and not mismatched,
            "n_arms_score_ordered": ordered,
            "arms_not_score_ordered": sorted(bad_order),
            "arms_wrong_width": sorted(bad_shape),
            "n_committed_checked": len(committed),
            "n_committed_mismatched": len(mismatched)}


def gate_k3(*, n: int, n_reproduced: int, threshold: int) -> dict:
    """K3: judge drift. Deliberately looser than WS4d's L2 (60/60) because
    the judge is sampled too -- requiring exact reproduction twice running is
    requiring luck rather than measuring drift (spec 7).
    """
    return {"gate": "K3", "n": n, "n_reproduced": n_reproduced,
            "threshold": threshold, "passed": bool(n_reproduced >= threshold)}


def gate_k4(cells_by_k: dict) -> dict:
    """K4: at which depths is discrimination indistinguishable from sampling
    noise? Reported per depth; those depths go to branch X-E and are NOT
    interpreted.
    """
    at_noise = sorted(
        k for k, c in cells_by_k.items()
        if c["null"] is not None and c["separable_rate"] is not None
        and c["null"] >= c["separable_rate"])
    return {"gate": "K4", "passed": not at_noise,
            "depths_at_noise": at_noise,
            "n_depths": len(cells_by_k)}


def bootstrap_rate(flags, *, n_boot: int, seed: int) -> dict:
    """Bootstrap CI for a boolean rate, resampling WITH replacement over the
    given units (spec 5.3: a disagreement rate over questions, or -- for the
    supplementary accidental-replicate set -- over duplicated
    (qid, arm, k) keys). Percentile interval, matching bootstrap_signal.

    Returns R = None (and a degenerate [None, None] interval) when `flags`
    is empty: an untested rate must not be reported as 0.0.
    """
    flags = list(flags)
    n = len(flags)
    if n == 0:
        return {"R": None, "n": 0, "n_disagree": 0,
                "ci_lo": None, "ci_hi": None, "n_boot": n_boot, "seed": seed}
    arr = np.asarray(flags, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    boot = arr[draws].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"R": float(arr.mean()), "n": n, "n_disagree": int(arr.sum()),
            "ci_lo": round(float(lo), 6), "ci_hi": round(float(hi), 6),
            "n_boot": n_boot, "seed": seed}


def four_sample_consistency_check(R) -> dict | None:
    """Spec 5.3's descriptive cross-check, published as an order-of-magnitude
    consistency check and NEVER as a test (per-question p is plainly not
    homogeneous).

    Under a homogeneous-p model, a two-sample disagreement rate
    R = 2p(1-p) implies a four-sample "not all identical" rate of
    1 - p**4 - (1-p)**4. Solves for the root p <= 0.5.
    """
    if R is None or not (0 <= R <= 0.5):
        return None
    p = (1 - (1 - 2 * R) ** 0.5) / 2
    predicted = 1 - p ** 4 - (1 - p) ** 4
    return {"p": round(p, 6), "predicted_four_sample_rate": round(predicted, 6)}


def gate_k6(*, R, ci_lo, ci_hi, n_disagree: int, min_n: int) -> dict:
    """K6 (REDEFINED, Amendment 1) / spec 5.3 V1: does replication show real
    within-prompt generator variance?

    Passes iff the replicate disagreement rate R's 95% bootstrap CI EXCLUDES
    zero AND at least `min_n` questions disagree -- a tight CI on fewer than
    that is one flaky row, not a measurement (spec 10, WS4E_REPLICATE_MIN_N).

    A two-sided gate (spec 7): passing licenses subtracting `null` in the
    estimator; failing is the MORE consequential result -- byte-identical
    prompts agreeing with themselves would mean the noise this project has
    been subtracting has some other, unidentified cause -- and routes to
    branch X-G.
    """
    excludes_zero = (ci_lo is not None and ci_hi is not None
                      and (ci_lo > 0 or ci_hi < 0))
    passed = bool(excludes_zero and n_disagree >= min_n)
    return {"gate": "K6", "R": R, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "n_disagree": n_disagree, "min_n": min_n, "passed": passed}


def mediator_point_estimates(cell_data: dict, qids) -> dict:
    """Task 10: the mediation chain's four point estimates per (arm, k)
    cell -- presence@k, exact match, judge accuracy, and conversion =
    P(judge correct | gold present at k). Mirrors what
    `results/ws4/chart_mediator.png` shows at k=10 (recall -> presence ->
    EM -> judge), but computed per cell so it can be read at every depth.

    `cell_data`: {(arm, k): {qid: (presence: bool, em: bool, judge: bool)}}.
    `conversion` is None for a cell where no question in `qids` has
    `presence` True -- reporting 0.0 on an empty denominator would be a lie,
    same posture as `decompose`'s `null`/`signal`.
    """
    qids = [int(q) for q in qids]
    out = {}
    for cell, d in cell_data.items():
        pres = [bool(d[q][0]) for q in qids]
        em = [bool(d[q][1]) for q in qids]
        judge = [bool(d[q][2]) for q in qids]
        present_judge = [j for p, j in zip(pres, judge) if p]
        out[cell] = {
            "n": len(qids),
            "presence_at_k": float(np.mean(pres)),
            "em": float(np.mean(em)),
            "judge_accuracy": float(np.mean(judge)),
            "conversion": (float(np.mean(present_judge))
                          if present_judge else None),
        }
    return out


def bootstrap_mediator_stats(cell_data: dict, qids, n_boot: int,
                             seed: int) -> dict:
    """Paired bootstrap over QUESTIONS for the mediation chain -- ONE shared
    question-index draw per replicate, reused across every (arm, k) cell and
    every one of the four outcome measures, same posture as
    `bootstrap_discrimination_stats` (WS4d section 4.3's machinery,
    unchanged).

    `cell_data`: {(arm, k): {qid: (presence: bool, em: bool, judge: bool)}}.
    `conversion`'s bootstrap distribution can have a replicate where the
    resampled draw contains no `presence`-True question (possible even
    though the point estimate's denominator is nonzero, because resampling
    can drop every present question by chance) -- that replicate contributes
    NaN and is excluded by `nanpercentile`, never treated as a conversion of
    0.0.
    """
    qids = [int(q) for q in qids]
    cells = sorted(cell_data, key=lambda c: (c[0], c[1]))
    rng = np.random.default_rng(seed)
    n_q = len(qids)
    pres_b = {c: np.empty(n_boot) for c in cells}
    em_b = {c: np.empty(n_boot) for c in cells}
    judge_b = {c: np.empty(n_boot) for c in cells}
    conv_b = {c: np.empty(n_boot) for c in cells}
    for i in range(n_boot):
        draw = [qids[j] for j in rng.choice(n_q, n_q, replace=True)]
        for c in cells:
            d = cell_data[c]
            pres = np.array([d[q][0] for q in draw], dtype=bool)
            em = np.array([d[q][1] for q in draw], dtype=float)
            judge = np.array([d[q][2] for q in draw], dtype=float)
            pres_b[c][i] = pres.mean()
            em_b[c][i] = em.mean()
            judge_b[c][i] = judge.mean()
            conv_b[c][i] = judge[pres].mean() if pres.any() else np.nan
    out = {}
    for c in cells:
        out[c] = {
            "presence_ci": tuple(round(float(v), 6) for v in
                                 np.percentile(pres_b[c], [2.5, 97.5])),
            "em_ci": tuple(round(float(v), 6) for v in
                          np.percentile(em_b[c], [2.5, 97.5])),
            "judge_accuracy_ci": tuple(round(float(v), 6) for v in
                                      np.percentile(judge_b[c], [2.5, 97.5])),
            "conversion_ci": tuple(round(float(v), 6) for v in
                                   np.nanpercentile(conv_b[c], [2.5, 97.5])),
        }
    return out


def conversion_composition_check(cell_data: dict, qids, n_boot: int,
                                 seed: int) -> dict:
    """Task 10 correction: `conversion(k)` (`mediator_by_k.csv`) is P(judge
    correct | present at k), and presence(k) is NESTED in k -- the exact
    same Simpson's-paradox shape section 4 already documents for
    `separable(k)`. Holding the presence set FIXED to the questions already
    present at the lowest swept k (`WS4E_K_VALUES[0]`, per-arm -- presence
    is arm-specific) removes the apparent conversion fall almost entirely.

    `cell_data`: {(arm, k): {qid: (presence, em, judge)}}, the same shape
    `mediator_point_estimates` / `bootstrap_mediator_stats` consume.

    Per (arm, k), reports: the moving conversion (mediator_by_k.csv's own
    column, recomputed here from the same cell_data rather than re-read, so
    this file stands alone), the FIXED-set conversion (fixed = present at
    the lowest k, held identical at every k since presence only grows -- see
    `n_fixed`), and a decomposition of the moving set into the newly-present
    questions (present now, absent at the previous swept k) versus the
    already-present ones (present at both) -- mirrors
    `fixed_population_membership_check`'s "hold a population fixed, read it
    back at every depth" posture but for conversion instead of the
    discriminating bucket. At the first swept k there is no previous depth:
    every present question counts as newly-present and n_already_present is
    0 (fixed_conversion == moving_conversion there by construction, since
    the fixed set IS the k=1 present set).

    The paired k_hi - k_lo difference on the FIXED set gets a bootstrap CI:
    same posture as `bootstrap_fixed_composition_signal` -- the pair
    (k_lo, k_hi) is fixed to the lowest and highest swept k ONCE, before the
    loop, and each replicate draws ONE shared resample of the arm's fixed
    set, reused for both k's judge outcomes (paired, not independent, so the
    CI is about the within-question change, not two separately-noisy rates).
    A fresh `np.random.default_rng(seed)` is used per arm (each arm's fixed
    set is a different population, so there is nothing to share across
    arms -- unlike `bootstrap_mediator_stats`'s single question-index draw
    shared across every (arm, k) cell).
    """
    qids = [int(q) for q in qids]
    arms = sorted({a for a, _ in cell_data})
    ks = sorted({k for _, k in cell_data})
    k_lo, k_hi = ks[0], ks[-1]
    out = {}
    for arm in arms:
        fixed_set = [q for q in qids if cell_data[(arm, k_lo)][q][0]]
        n_fixed = len(fixed_set)
        prev_present = None
        per_k = {}
        for k in ks:
            d = cell_data[(arm, k)]
            present_now = {q for q in qids if d[q][0]}
            moving_judge = [d[q][2] for q in present_now]
            fixed_judge = [d[q][2] for q in fixed_set]
            if prev_present is None:
                newly, already = present_now, set()
            else:
                newly = present_now - prev_present
                already = present_now & prev_present
            newly_judge = [d[q][2] for q in newly]
            already_judge = [d[q][2] for q in already]
            per_k[k] = {
                "n_present": len(present_now),
                "moving_conversion": (float(np.mean(moving_judge))
                                      if moving_judge else None),
                "n_fixed": n_fixed,
                "fixed_conversion": (float(np.mean(fixed_judge))
                                     if fixed_judge else None),
                "n_newly_present": len(newly),
                "newly_present_conversion": (float(np.mean(newly_judge))
                                             if newly_judge else None),
                "n_already_present": len(already),
                "already_present_conversion": (float(np.mean(already_judge))
                                               if already_judge else None),
            }
            prev_present = present_now
        rng = np.random.default_rng(seed)
        diffs = np.empty(n_boot)
        d_lo, d_hi = cell_data[(arm, k_lo)], cell_data[(arm, k_hi)]
        for i in range(n_boot):
            if n_fixed == 0:
                diffs[i] = np.nan
                continue
            draw = [fixed_set[j] for j in rng.integers(0, n_fixed, n_fixed)]
            diffs[i] = (np.mean([d_hi[q][2] for q in draw])
                       - np.mean([d_lo[q][2] for q in draw]))
        lo, hi = (float(v) for v in np.nanpercentile(diffs, [2.5, 97.5]))
        lo, hi = min(lo, hi), max(lo, hi)
        fc_lo, fc_hi = per_k[k_lo]["fixed_conversion"], per_k[k_hi]["fixed_conversion"]
        out[arm] = {
            "k_lo": k_lo, "k_hi": k_hi, "per_k": per_k,
            "fixed_diff": (None if fc_lo is None or fc_hi is None
                          else round(fc_hi - fc_lo, 6)),
            "fixed_diff_ci_lo": round(lo, 6), "fixed_diff_ci_hi": round(hi, 6),
            "n_boot": n_boot, "seed": seed,
        }
    return out
