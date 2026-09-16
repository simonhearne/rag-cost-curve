"""WS4d pure logic: eligibility labels, adjudication, agreement, the
changepoint estimator, and gates E1-E4 / F1 / L1.

Design: results/recall-vs-quality.md

No I/O and no API calls live here -- the live half is
benchlib/recall_quality/eligibility_runner.py.
"""

import numpy as np

LABELS = ("eligible", "ambiguous", "wrong_gold", "stale_gold", "unanswerable")
INELIGIBLE = frozenset(LABELS[1:])

# spec §4.1. `unanswerable` is deliberately NOT here: it is the class closest
# to "retrieval did not work", and although it is judged against the
# arm-independent ground-truth top-10 it is the one an audience can most
# easily recast as circular. It is reported as a declared sensitivity.
PRIMARY_DROP = frozenset({"ambiguous", "wrong_gold", "stale_gold"})

DISPUTED = "ineligible_class_disputed"


def validate_label(label: str) -> str | None:
    """None if `label` is one of the spec §3.2 five, else an error string."""
    if label not in LABELS:
        return f"unknown label {label!r}; expected one of {LABELS}"
    return None


def adjudicate(label_a: str, label_b: str) -> tuple[str, str]:
    """Spec §3.3, fixed before any label exists.

    Returns (binary, klass). A disagreement on the BINARY eligible/ineligible
    call resolves to eligible -- the conservative direction, because dropping
    is what inflates accuracy.
    """
    for lab in (label_a, label_b):
        err = validate_label(lab)
        if err:
            raise ValueError(err)
    a_inel, b_inel = label_a in INELIGIBLE, label_b in INELIGIBLE
    if not a_inel and not b_inel:
        return ("eligible", "eligible")
    if a_inel != b_inel:
        return ("eligible", "eligible")
    if label_a == label_b:
        return ("ineligible", label_a)
    return ("ineligible", DISPUTED)


def cohens_kappa(a, b) -> float:
    """Cohen's kappa for two raters over the same items.

    Degenerate case: when both raters use exactly one category, p_e == 1 and
    kappa is 0/0. Return 1.0 -- they agreed on everything, and returning nan
    would make gate E3's comparison silently False.
    """
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError(f"rater lengths differ: {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        raise ValueError("no items to score")
    cats = sorted(set(a) | set(b))
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    p_e = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def _binary(labels):
    return ["ineligible" if x in INELIGIBLE else "eligible" for x in labels]


def gate_e3(labels_a, labels_b) -> dict:
    """Spec §3.4 E3. The floor applies to the BINARY decision, because that is
    the decision that filters the curve. The 5-class kappa is reported, not
    gated."""
    from ..config import WS4D_KAPPA_MIN
    kb = cohens_kappa(_binary(labels_a), _binary(labels_b))
    k5 = cohens_kappa(labels_a, labels_b)
    return {"gate": "E3", "kappa_binary": round(kb, 6),
            "kappa_5class": round(k5, 6), "floor": WS4D_KAPPA_MIN,
            "n": len(list(labels_a)), "passed": bool(kb >= WS4D_KAPPA_MIN)}


N1_MAX_CI_WIDTH = 0.05          # WS4b spec §7.3, inherited verbatim
_ENDPOINT_TOL = 0.005


def fit_changepoint(x, y, grid: int = 200) -> dict:
    """WS4b spec §6.1: q(r) = a + b1*min(r,tau) + b2*max(r-tau,0), tau free.

    tau enters non-linearly, so it is profiled over a grid while a/b1/b2 are
    solved exactly by least squares at each candidate. The grid spans the
    OPEN interval: at tau == x.min() or x.max() one segment has no points and
    the design matrix is rank-deficient.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    best = None
    for tau in np.linspace(x.min() + 1e-6, x.max() - 1e-6, grid):
        X = np.column_stack([np.ones_like(x), np.minimum(x, tau),
                             np.maximum(x - tau, 0.0)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(((y - X @ beta) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, tau, beta)
    sse, tau, beta = best
    return {"tau": float(tau), "a": float(beta[0]), "b1": float(beta[1]),
            "b2": float(beta[2]), "sse": sse}


def fit_line(x, y) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"intercept": float(beta[0]), "slope": float(beta[1]),
            "sse": float(((y - X @ beta) ** 2).sum())}


def bootstrap_changepoint(rec, acc, n_boot: int, seed: int) -> dict:
    """Paired bootstrap over QUESTIONS -- one shared index draw per replicate,
    refitting inside every replicate (WS4b §6.1).

    rec, acc: (n_questions, n_arms). Arms are columns so a single row draw
    resamples the same questions for every arm, preserving the pairing that
    makes arm-to-arm differences comparable.
    """
    rec, acc = np.asarray(rec, float), np.asarray(acc, float)
    if rec.shape != acc.shape:
        raise ValueError(f"shape mismatch: {rec.shape} vs {acc.shape}")
    point = fit_changepoint(rec.mean(0), acc.mean(0))
    rng = np.random.default_rng(seed)
    n_q = rec.shape[0]
    taus, slopes = np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n_q, n_q, replace=True)      # ONE shared draw
        xb, yb = rec[idx].mean(0), acc[idx].mean(0)
        taus[i] = fit_changepoint(xb, yb)["tau"]
        slopes[i] = fit_line(xb, yb)["slope"]
    lo, hi = np.percentile(taus, [2.5, 97.5])
    slo, shi = np.percentile(slopes, [2.5, 97.5])
    return {"tau": round(point["tau"], 6), "b1": round(point["b1"], 6),
            "b2": round(point["b2"], 6),
            "ci_lo": round(float(lo), 6), "ci_hi": round(float(hi), 6),
            "width": round(float(hi - lo), 6),
            "slope": round(fit_line(rec.mean(0), acc.mean(0))["slope"], 6),
            "slope_ci_lo": round(float(slo), 6),
            "slope_ci_hi": round(float(shi), 6),
            "n_boot": n_boot, "seed": seed}


def bootstrap_slope(rec, acc, n_boot: int, seed: int) -> dict:
    """Paired bootstrap over QUESTIONS for a plain LINE fit -- no
    changepoint. Split out from `bootstrap_changepoint` (which also fits
    `fit_line` internally, but ALWAYS fits `fit_changepoint` too) so a
    caller for whom fitting a changepoint would be wrong -- WS4e spec 6.4:
    four arms against a four-parameter two-segment model is zero residual
    df -- never calls `fit_changepoint` at all, even transitively.

    rec, acc: (n_questions, n_arms). Arms are columns so a single row draw
    resamples the same questions for every arm, preserving the pairing
    that makes arm-to-arm differences comparable (WS4d section 4.3's
    machinery, unchanged).
    """
    rec, acc = np.asarray(rec, float), np.asarray(acc, float)
    if rec.shape != acc.shape:
        raise ValueError(f"shape mismatch: {rec.shape} vs {acc.shape}")
    point = fit_line(rec.mean(0), acc.mean(0))
    rng = np.random.default_rng(seed)
    n_q = rec.shape[0]
    slopes = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n_q, n_q, replace=True)      # ONE shared draw
        xb, yb = rec[idx].mean(0), acc[idx].mean(0)
        slopes[i] = fit_line(xb, yb)["slope"]
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return {"slope": round(point["slope"], 6),
            "intercept": round(point["intercept"], 6),
            "sse": round(point["sse"], 8),
            "slope_ci_lo": round(float(min(lo, hi)), 6),
            "slope_ci_hi": round(float(max(lo, hi)), 6),
            "n_boot": n_boot, "seed": seed}


def knee_branch_n(boot: dict, x) -> str:
    """WS4b spec §7.3, inherited verbatim. N3 is checked FIRST: 'recall does
    not measurably predict quality anywhere' is a stronger claim than 'no
    knee is locatable', so it must not be masked by a wide tau CI."""
    x = np.asarray(x, float)
    if boot["slope_ci_lo"] <= 0.0 <= boot["slope_ci_hi"]:
        return "N3"
    touches = (boot["ci_lo"] <= x.min() + _ENDPOINT_TOL
               or boot["ci_hi"] >= x.max() - _ENDPOINT_TOL)
    if boot["width"] <= N1_MAX_CI_WIDTH and not touches:
        return "N1"
    return "N2"


L1_BAND = (0.80, 0.97)


def gate_f1(n_declines: int, n_errors: int) -> dict:
    """Spec §4.4. After filtering, is what remains still an ANSWER-QUALITY
    curve, or has it become a curve about the generator declining?

    WS4b §6 established that more than half of all declines are CORRECT
    behaviour, so a residual that is almost entirely declines is not
    measuring answer quality and must not be labelled as though it were.
    """
    from ..config import WS4D_DECLINE_SHARE_MAX
    total = n_declines + n_errors
    share = (n_declines / total) if total else 0.0
    fired = bool(share >= WS4D_DECLINE_SHARE_MAX)
    return {"gate": "F1", "n_declines": n_declines, "n_errors": n_errors,
            "decline_share": round(share, 6),
            "threshold": WS4D_DECLINE_SHARE_MAX, "fired": fired,
            "curve_is": "decline-behaviour" if fired else "answer-quality"}


def gate_l1(recalls) -> dict:
    """Spec §5.3. After the four new arms land, is the candidate band densely
    enough covered to claim it? Gaps are measured only INSIDE [0.80, 0.97] --
    outside it there is no knee to resolve and an isolated low arm is not a
    ladder defect."""
    from ..config import WS4D_LADDER_MAX_GAP
    lo, hi = L1_BAND
    band = sorted(r for r in recalls if lo <= r <= hi)
    gaps = [b - a for a, b in zip(band, band[1:])]
    max_gap = max(gaps) if gaps else 0.0
    return {"gate": "L1", "n_arms": len(list(recalls)), "in_band": len(band),
            "max_gap": round(max_gap, 6), "threshold": WS4D_LADDER_MAX_GAP,
            "passed": bool(max_gap <= WS4D_LADDER_MAX_GAP)}


def gate_e1(rendered) -> dict:
    """Spec §3.4 E1. Every prompt actually sent is compared byte-for-byte
    against an independent reconstruction built from ONLY the §3.1 inputs.

    Rows with `actual` None are checkpoint-recovered: no call was made this
    invocation, so there is nothing to compare. They are counted separately
    rather than passed silently, so a fully-resumed run cannot report a
    vacuous blindness pass.
    """
    rows = list(rendered)
    compared = [r for r in rows if r.get("actual") is not None]
    bad = [(r["qid"], r["model"]) for r in compared
           if r["expected"] != r["actual"]]
    return {"gate": "E1", "n_rows": len(rows), "n_checked": len(compared),
            "n_uncompared": len(rows) - len(compared),
            "n_mismatched": len(bad), "examples": bad[:3],
            "passed": not bad}


E4_VERDICTS = {"agree": True, "disagree": False}


def gate_e4(verdicts) -> dict:
    """Spec §3.4 E4. BLIND: the human assigned a label without seeing the
    models'; `verdicts` records whether each matched after the fact.

    results/recall-vs-quality.md is why this is blind: its spot-check showed
    the reviewer the label and asked 'is this defensible?', which is
    confirmatory review and systematically more agreeable than re-derivation.
    """
    from ..config import WS4D_SPOTCHECK_MAX_DISAGREE, WS4D_SPOTCHECK_N
    verdicts = list(verdicts)
    for v in verdicts:
        if v not in E4_VERDICTS:
            raise ValueError(f"unrecognised verdict {v!r}; "
                             f"expected one of {sorted(E4_VERDICTS)}")
    n_dis = sum(1 for v in verdicts if not E4_VERDICTS[v])
    err = "" if len(verdicts) == WS4D_SPOTCHECK_N else \
        f"expected {WS4D_SPOTCHECK_N} rows, got {len(verdicts)}"
    return {"gate": "E4", "n": len(verdicts), "n_disagree": n_dis,
            "threshold": WS4D_SPOTCHECK_MAX_DISAGREE, "blind": True,
            "error": err,
            "passed": bool(not err and n_dis <= WS4D_SPOTCHECK_MAX_DISAGREE)}
