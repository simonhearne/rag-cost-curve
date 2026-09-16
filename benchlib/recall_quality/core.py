"""WS4 pure half: answer scoring, prompt rendering, the operating-point rule,
the question draw, the paired bootstrap, and the pre-registered
classification rules of spec section 7. No I/O, no API calls -- everything
here is unit-tested in tests/recall_quality/test_core.py. The live half is
benchlib/recall_quality/runner.py.

Design: results/recall-vs-quality.md
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

import numpy as np

# ------------------------------------------------------------ scoring ----


def norm_text(s: str) -> str:
    """WS1's answer-presence normalisation, byte-for-byte (notebook 01 5d):
    NFKD, lowercase, every non-alphanumeric becomes a space."""
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"[^a-z0-9 ]", " ", s)


def answer_present(golds, texts) -> bool:
    """Any normalised gold alias is a substring of the concatenated
    normalised texts. Texts are joined with a space so a match cannot span
    two passages by accident of adjacency."""
    if not texts:
        return False
    blob = " ".join(norm_text(t) for t in texts)
    return any(norm_text(g) in blob for g in golds)


_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """SQuAD-style: lowercase, strip punctuation, drop articles, collapse
    whitespace. Standard for NQ-Open EM/F1."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds) -> int:
    p = normalize_answer(pred)
    return int(any(p == normalize_answer(g) for g in golds))


def _f1_one(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def f1(pred: str, golds) -> float:
    """Max token-F1 over the gold aliases."""
    if not golds:
        return 0.0
    return max(_f1_one(pred, g) for g in golds)


IDK = "I don't know"


def is_idk(pred: str) -> bool:
    """The committed decline phrase, tolerant of trailing punctuation and
    apostrophe variants, and nothing else."""
    p = pred.strip().replace("’", "'").rstrip(".!").strip().lower()
    return p == IDK.lower()


# ------------------------------------------------------ prompt render ----

NO_PASSAGES = "(no passages were retrieved)"


def render_context(passages) -> str:
    """Rank-ordered '[k] title\\ntext' blocks, blank-line separated. The same
    renderer serves every arm; the parametric arm gets NO_PASSAGES."""
    if not passages:
        return NO_PASSAGES
    return "\n\n".join(f"[{k}] {p['title']}\n{p['text']}"
                       for k, p in enumerate(passages, start=1))


def user_message(context: str, question: str) -> str:
    return f"Passages:\n\n{context}\n\nQuestion: {question}"


# ---------------------------------------------------- operating points ----


def pick_matching_nprobe(target_recall: float, sweep: dict) -> int:
    """Spec 4.3: the nprobe whose GT-sweep recall@10 is nearest the target;
    ties go to the higher nprobe."""
    best = None
    for nprobe, r in sweep.items():
        d = round(abs(r - target_recall), 9)   # fp noise must not break a tie
        if best is None or d < best[0] or (d == best[0] and nprobe > best[1]):
            best = (d, nprobe)
    return int(best[1])


# ------------------------------------------------------ question draw ----


def select_questions(pass_flags, n_main: int, n_sanity: int, seed: int):
    """Spec 3.2/3.3. Main: n_main rows from the pass set. Sanity: n_sanity
    rows from ALL rows (no filter), disjoint from main. One RNG, two draws,
    so the sanity draw is reproducible only together with the main draw."""
    pass_flags = np.asarray(pass_flags, dtype=bool)
    pass_idx = np.flatnonzero(pass_flags)
    if len(pass_idx) < n_main:
        raise ValueError(f"only {len(pass_idx)} questions pass the filter; "
                         f"cannot draw {n_main}")
    rng = np.random.default_rng(seed)
    main = np.sort(rng.choice(pass_idx, n_main, replace=False))
    rest = np.setdiff1d(np.arange(len(pass_flags)), main)
    if len(rest) < n_sanity:
        raise ValueError("not enough rows left for the sanity sample")
    sanity = np.sort(rng.choice(rest, n_sanity, replace=False))
    return main, sanity


# ----------------------------------------------------------- bootstrap ----


def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) resampled row indices. ONE draw shared by every arm and
    every metric, which is what makes the differences paired.

    WARNING -- this rebuilds default_rng(seed) on EVERY call, so two calls
    with the same arguments return the SAME matrix. That is the point for a
    paired within-stratum bootstrap (ws6c.paired_summary) and it is wrong for
    any comparison of two INDEPENDENT samples: reusing one index draw across
    both manufactures a dependence between them and, on equal-sized cells,
    shrinks the interval toward a point. `ws8_stats.contrast_summary` is
    exactly that case -- two strata, different questions -- and deliberately
    draws from one generator itself instead of calling this. Do not "simplify"
    it to use this helper. Behaviour here is frozen: WS4's published numbers
    (results/ws4/) depend on this exact stream.
    """
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def boot_means(values, idx: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return v[idx].mean(axis=1)


def percentile_ci(samples, alpha: float):
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bonferroni_alpha(alpha: float, n_tests: int) -> float:
    return alpha / n_tests


# ------------------------------------------- spec 7.1 classification ----


def classify_delta(ci_lo: float, ci_hi: float, margin: float) -> str:
    """Delta = Q(ref) - Q(arm). DROP if the CI excludes zero from above;
    PLATEAU if a loss of `margin` or more is ruled out; else INCONCLUSIVE."""
    if ci_lo > 0:
        return "DROP"
    if ci_hi < margin:
        return "PLATEAU"
    return "INCONCLUSIVE"


def knee(arms) -> float | None:
    """arms: iterable of (name, measured_recall, class), reference excluded.
    r* = the lowest recall such that every arm at or above it is PLATEAU.
    None if the top arm is not PLATEAU."""
    ordered = sorted(arms, key=lambda a: -a[1])
    r_star = None
    for _, r, cls in ordered:
        if cls != "PLATEAU":
            break
        r_star = r
    return r_star


def knee_branch(arms, top: float = 0.97) -> str:
    """K1: a PLATEAU arm at recall <= `top` inside the plateau region.
    K2: every arm is DROP. K3: otherwise."""
    classes = [a[2] for a in arms]
    if classes and all(c == "DROP" for c in classes):
        return "K2"
    r_star = knee(arms)
    if r_star is not None and r_star <= top:
        return "K1"
    return "K3"


# ------------------------------------------- spec 7.2 mechanism test ----


def is_matched(r_a: float, r_b: float, tol: float) -> bool:
    return abs(r_a - r_b) <= tol + 1e-12


def _excludes_zero(ci) -> bool:
    lo, hi = ci
    return lo > 0 or hi < 0


def mechanism_branch(pairs, margin: float = 0.05) -> str:
    """pairs: dicts with pair ('A'..'F'), matched (bool), face (lo, hi) at
    alpha, adj (lo, hi) at alpha/5. Pair A is primary at face value; every
    other pair counts toward M1 only at the adjusted width. M2/M3 use the
    face-value CIs of every matched pair."""
    matched = [p for p in pairs if p["matched"]]
    if not matched:
        return "M3"
    for p in matched:
        ci = p["face"] if p["pair"] == "A" else p["adj"]
        if _excludes_zero(ci):
            return "M1"
    if all(-margin <= p["face"][0] and p["face"][1] <= margin for p in matched):
        return "M2"
    return "M3"


def parametric_reading(acc: float) -> str:
    """Spec 7.5 tiers for the leak floor under the grounded prompt."""
    if acc <= 0.10:
        return "holds"
    if acc <= 0.30:
        return "partial_leak"
    return "does_not_ground"


# ---------------------------------------------------- analysis helpers ----


def _ci_pair(samples, alpha):
    lo, hi = percentile_ci(samples, alpha)
    return round(lo, 6), round(hi, 6)


def summarise_arms(recall: dict, quality: dict, *, ref: str, n_boot: int,
                   seed: int, margin: float, alpha: float) -> list[dict]:
    """Per-arm means with paired-bootstrap CIs and the spec 7.1 class.

    recall / quality: arm -> per-question array over the SAME questions in
    the SAME order. One index draw serves every arm, so every difference is
    paired. Quality is any 0/1 (or [0,1]) per-question metric."""
    arms = list(quality)
    n = len(quality[ref])
    for a in arms:
        assert len(quality[a]) == n and len(recall[a]) == n, a
    idx = bootstrap_indices(n, n_boot, seed)
    q_boot = {a: boot_means(quality[a], idx) for a in arms}
    r_boot = {a: boot_means(recall[a], idx) for a in arms}
    out = []
    for a in arms:
        q_mean = float(np.mean(quality[a]))
        r_mean = float(np.mean(recall[a]))
        d = q_boot[ref] - q_boot[a]
        d_lo, d_hi = _ci_pair(d, alpha)
        q_lo, q_hi = _ci_pair(q_boot[a], alpha)
        r_lo, r_hi = _ci_pair(r_boot[a], alpha)
        delta = float(np.mean(quality[ref]) - q_mean)
        cls = "reference" if a == ref else classify_delta(d_lo, d_hi, margin)
        out.append(dict(arm=a, n=n, recall=r_mean, recall_ci_lo=r_lo,
                        recall_ci_hi=r_hi, quality=q_mean, quality_ci_lo=q_lo,
                        quality_ci_hi=q_hi, delta_vs_ref=round(delta, 6),
                        delta_ci_lo=d_lo, delta_ci_hi=d_hi, **{"class": cls}))
    return out


def adjacent_slopes(recall: dict, quality: dict, *, n_boot: int, seed: int,
                    alpha: float) -> list[dict]:
    """dQ/dr between arms adjacent in measured recall, with a joint
    bootstrap CI (recall and quality resampled together)."""
    arms = sorted(quality, key=lambda a: -float(np.mean(recall[a])))
    n = len(quality[arms[0]])
    idx = bootstrap_indices(n, n_boot, seed)
    out = []
    for hi_arm, lo_arm in zip(arms, arms[1:]):
        dq = float(np.mean(quality[hi_arm]) - np.mean(quality[lo_arm]))
        dr = float(np.mean(recall[hi_arm]) - np.mean(recall[lo_arm]))
        bq = boot_means(quality[hi_arm], idx) - boot_means(quality[lo_arm], idx)
        br = boot_means(recall[hi_arm], idx) - boot_means(recall[lo_arm], idx)
        with np.errstate(divide="ignore", invalid="ignore"):
            slopes = np.where(br != 0, bq / br, np.nan)
        s_lo, s_hi = _ci_pair(slopes[np.isfinite(slopes)], alpha) if np.isfinite(
            slopes).any() else (float("nan"), float("nan"))
        out.append(dict(upper=hi_arm, lower=lo_arm, d_recall=round(dr, 6),
                        d_quality=round(dq, 6),
                        slope=round(dq / dr, 6) if dr else float("nan"),
                        slope_ci_lo=s_lo, slope_ci_hi=s_hi))
    return out


def evaluate_pairs(plan, recall: dict, quality: dict, *, n_boot: int,
                   seed: int, tol: float, alpha: float, n_adjust: int) -> list[dict]:
    """Spec 7.2. plan: [(pair_label, arm_first, arm_second)]; diff =
    Q(first) - Q(second). Face-value CI at alpha and the Bonferroni CI at
    alpha / n_adjust are both reported; `matched` uses measured recall."""
    n = len(next(iter(quality.values())))
    idx = bootstrap_indices(n, n_boot, seed)
    out = []
    for label, a, b in plan:
        ra, rb = float(np.mean(recall[a])), float(np.mean(recall[b]))
        d = boot_means(quality[a], idx) - boot_means(quality[b], idx)
        f_lo, f_hi = _ci_pair(d, alpha)
        j_lo, j_hi = _ci_pair(d, bonferroni_alpha(alpha, n_adjust))
        out.append(dict(pair=label, first=a, second=b, recall_first=ra,
                        recall_second=rb, recall_gap=round(abs(ra - rb), 6),
                        matched=is_matched(ra, rb, tol),
                        diff=round(float(np.mean(quality[a]) - np.mean(quality[b])), 6),
                        face_lo=f_lo, face_hi=f_hi, adj_lo=j_lo, adj_hi=j_hi,
                        primary=(label == "A")))
    return out
