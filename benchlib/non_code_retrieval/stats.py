"""WS8 analysis: per-stratum paired summaries and the hop contrast.

The PRIMARY test is the CONTRAST between the `hop3plus` and `hop1` strata, not
either cell alone (spec section 7, Phase 3). Reporting one cell would reproduce
the difference-of-medians error WS6a made and WS6c had to correct.

Nothing here decides anything. `resolve_hop_branch` is a LOOKUP against four
branches (H1/H2/H3/H4) that were fixed in writing before any WS8 data existed;
it is deliberately incapable of expressing a judgement made after the fact.

Pure: no network, no I/O, no config reads. The seed, n_boot and alpha are
passed in by the caller from `benchlib.config`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..code_retrieval.topk import paired_deltas

# indexed - agentic. NEGATIVE means the index spent FEWER prompt tokens, i.e.
# index-favourable. Every direction test in this module depends on that, so it
# is named once here rather than re-derived at each comparison.
ARM_A = "indexed"
ARM_B = "agentic"


def stratum_deltas(df, stratum: str, metric: str = "prompt_tokens",
                   exclude_qids: Iterable[str] = ()) -> np.ndarray:
    """Paired (indexed - agentic) deltas within one stratum.

    `exclude_qids` drops the named questions from BOTH arms before pairing --
    dropping one arm's row alone would merely unpair it. It exists for the
    pre-registered `ws8_q067` sensitivity (spec section 7, amendment 2026-09-11),
    not as a general outlier filter: no row may be removed here that was not
    named in the spec before the analysis ran.
    """
    sub = df[df["stratum"] == stratum]
    drop = {str(q) for q in exclude_qids}
    if drop:
        sub = sub[~sub["qid"].astype(str).isin(drop)]
    # paired_deltas returns (qids, array). The qids are not needed here, but
    # the call is kept for its unpaired-qid check -- a silent inner join would
    # quietly redefine n, and n is the whole argument in an underpowered design.
    _qids, deltas = paired_deltas(sub, metric, ARM_A, ARM_B)
    return np.asarray(deltas, dtype=float)


def contrast_summary(deltas_a, deltas_b, n_boot: int, seed: int,
                     alpha: float) -> dict:
    """Bootstrap CI on median(a) - median(b). `a` is hop3plus, `b` is hop1.

    The two strata are DIFFERENT questions, so this is an UNPAIRED contrast of
    two PAIRED statistics: each stratum is resampled independently. Sharing one
    index draw across the cells -- which is exactly what makes ws6c's
    within-stratum bootstrap paired -- would manufacture a dependence between
    two independent samples and, on equal-sized cells, shrink the interval
    toward a point.

    `excludes_zero` is the separation flag the branch lookup reads. The point
    estimate is the difference of the OBSERVED medians, not the bootstrap
    mean, so the published number is the one in the data.
    """
    a = np.asarray(deltas_a, dtype=float)
    b = np.asarray(deltas_b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("both strata need at least one paired observation")
    rng = np.random.default_rng(seed)
    # Two draws from one generator: independent by construction, and still
    # fully determined by `seed`.
    boot_a = np.median(a[rng.integers(0, a.size, size=(n_boot, a.size))], axis=1)
    boot_b = np.median(b[rng.integers(0, b.size, size=(n_boot, b.size))], axis=1)
    lo, hi = np.percentile(boot_a - boot_b,
                           [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "median": float(np.median(a) - np.median(b)),
        "lo": float(lo),
        "hi": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_a": int(a.size),
        "n_b": int(b.size),
    }


def resolve_hop_branch(contrast: dict, hop1: dict) -> str:
    """Spec section 7, Phase 3. A lookup, never a judgement made after the fact.

    H4 is checked FIRST and unconditionally: if the hop1 cell separates, the
    contrast cannot be read as a hop effect "regardless of what hop>=3 does",
    so no contrast result can overturn it.

    `hop1` may be a `ws6c.paired_summary` dict verbatim; only its
    `excludes_zero` flag is read (its interval keys are ci_low/ci_high, NOT
    the lo/hi this module's own contrast dict uses).

    BOTH dicts are indexed with `[]`, never `.get`. A `.get` here fails OPEN:
    a mis-keyed `hop1` would read as "did not separate", H4 would silently
    never fire and the lookup would return H1/H2/H3 instead -- a silent flip
    of the one guard the whole workstream rests on. A KeyError is the only
    safe response to a dict of the wrong shape.
    """
    if hop1["excludes_zero"]:
        return "H4-confounded"
    if not contrast["excludes_zero"]:
        return "H2"
    # Spec section 7 words H1/H3 as the DIRECTION THE CI EXCLUDES ZERO IN, not
    # the sign of the point estimate, and at n=30 the median is discrete enough
    # that a percentile CI could in principle sit on the opposite side of zero
    # from the observed median. Negative is index-favourable, so a CI entirely
    # below zero (hi < 0) is H1; we only get here when the CI separates, so the
    # remaining case is lo > 0, i.e. H3.
    return "H1" if contrast["hi"] < 0 else "H3"
