#!/usr/bin/env python
"""Post-hoc robustness of WS4's K1 knee (r*) and M2 mechanism verdict.

Zero API spend, no Milvus. Reads the FROZEN results/ws4/runs.csv and writes
three CSVs into results/ws4/. Never rewrites the pre-registered outputs
(summary.csv, pairs.csv, ...) -- those stand exactly as the notebook left them.

What this answers, and why it is a separate script rather than a notebook cell:
the notebook implements spec 7.1/7.2 as pre-registered. Nothing here changes
that. These are the sensitivity checks a hostile audience question forces, run
AFTER the branch was resolved, and they are labelled post-hoc everywhere they
appear.

  results/ws4/rstar_sensitivity.csv    scalar diagnostics (tidy)
  results/ws4/arm_tests_corrected.csv  per-arm: exact test, Holm, flip distance
  results/ws4/mechanism_pairs_all.csv  EVERY cross-family matched pair

Run:  python scripts/recall-quality/ws4_rstar_sensitivity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib.recall_quality import core
from benchlib.config import (
    SEED,
    WS4_EQUIVALENCE_MARGIN,
    WS4_MATCH_TOLERANCE,
    WS4_N_BOOTSTRAP,
    ws4_dir,
)

REF = "sq8_np512"
PRIMARY = "judge"  # spec 7.1: the judge is primary

# Mechanism family per arm, as labelled in spec 4.2 / results/recall-vs-quality.md.
FAMILY = {
    "sq8_np512": "reference",
    "sq8_np64": "pruning",
    "refine_np64": "pruning",
    "sq8_np16": "pruning",
    "sq8_np8": "pruning",
    "sq8_np4": "pruning",
    "pca_uc_512_sq8_np512": "truncation",
    "pca_uc_384_sq8_np512": "truncation",
    "mrl_512_sq8_np512": "truncation",
    "pq_np256": "quantization",
    "rabitq_np256": "quantization",
}

# The six pairs spec 7.2 pre-registered, so the exhaustive sweep can say which
# matched pairs the design never looked at.
PREREG_PAIRS = {
    frozenset(("pca_uc_512_sq8_np512", "refine_np64")): "A",
    frozenset(("pca_uc_512_sq8_np512", "sq8_np64")): "B",
    frozenset(("pca_uc_384_sq8_np512", "sq8_np16")): "C",
    frozenset(("mrl_512_sq8_np512", "sq8_np4")): "D",
    frozenset(("pq_np256", "sq8_np8")): "E",
    frozenset(("mrl_512_sq8_np512", "rabitq_np256")): "F",
}


# --------------------------------------------------------------- loading ----


def load_matrices():
    """(quality, presence, recall) as (n_questions x n_arms) frames over the
    450 MAIN questions, arms in descending measured recall. `parametric` is a
    floor, not a retrieval arm, and is excluded."""
    runs = pd.read_csv(ws4_dir() / "runs.csv")
    runs = runs[(runs.subset == "main") & (runs.arm != "parametric")]

    def piv(col):
        return runs.pivot(index="qid", columns="arm", values=col).astype(float)

    q, p, r = piv("judge_correct"), piv("answer_presence_at_10"), piv("recall_at_10")
    order = r.mean().sort_values(ascending=False).index
    return q[order], p[order], r[order]


def spec_classes(q, r, idx):
    """Reproduce the spec 7.1 per-arm class from the frozen runs, so every
    sensitivity number below is anchored to a classification we can show is
    identical to the published summary.csv."""
    ref_boot = core.boot_means(q[REF], idx)
    rows = []
    for arm in q.columns:
        if arm == REF:
            rows.append({"arm": arm, "recall": r[arm].mean(),
                         "delta": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                         "spec_class": "reference"})
            continue
        delta = core.boot_means(q[arm], idx)
        lo, hi = core.percentile_ci(ref_boot - delta, 0.05)
        rows.append({
            "arm": arm,
            "recall": float(r[arm].mean()),
            "delta": float(q[REF].mean() - q[arm].mean()),
            "ci_lo": lo, "ci_hi": hi,
            "spec_class": core.classify_delta(lo, hi, WS4_EQUIVALENCE_MARGIN),
        })
    return pd.DataFrame(rows)


def rstar_of(recalls, classes):
    """spec 7.1's r*, over (recall, class) with the reference included.

    Returns None when the highest-recall DEGRADED arm is not itself PLATEAU:
    there is then no plateau to have a lower edge, and spec 7.1's branch rule
    reads that as K3, not K1. Crediting the reference arm's own recall here
    would manufacture a knee out of the very case that has none."""
    arms = [(None, rc, cl) for rc, cl in zip(recalls, classes) if cl != "reference"]
    return core.knee(arms)


def branch_of(recalls, classes):
    """spec 7.1's K-branch for the same (recall, class) vector."""
    arms = [(None, rc, cl) for rc, cl in zip(recalls, classes) if cl != "reference"]
    return core.knee_branch(arms)


# ------------------------------------------------------ r* under resampling ----


def rstar_bootstrap(q, r, idx, base):
    """Re-apply the whole spec 7.1 rule inside the paired bootstrap.

    The published CIs are percentile CIs on each arm's delta. Re-deriving those
    by an inner bootstrap for every outer replicate is a double bootstrap and is
    not affordable here, so r* is recomputed two ways and both are reported:

      fixed_width  each replicate reuses the ORIGINAL CI half-widths, so only
                   the point deltas move. Understates the spread.
      normal       each replicate recomputes its own width as +/-1.96*sd/sqrt(n)
                   on that replicate's paired differences, so the width moves
                   too. Requires the normal approximation, which `agreement`
                   below checks against the published percentile CIs.

    If both give the same story, the story does not depend on the shortcut.
    """
    n = q.shape[0]
    arms = list(q.columns)
    ref_i = arms.index(REF)
    Q = q.values
    R = r.values

    half_lo = (base.delta - base.ci_lo).values
    half_hi = (base.ci_hi - base.delta).values

    out = {"fixed_width": [], "normal": []}
    branches = {"fixed_width": [], "normal": []}
    z = 1.959963984540054
    for ii in idx:
        qb, rb = Q[ii], R[ii]
        mean_b = qb.mean(0)
        delta_b = mean_b[ref_i] - mean_b
        rec_b = rb.mean(0)
        # per-replicate SE of each paired difference
        se_b = (qb[:, [ref_i]] - qb).std(0, ddof=1) / np.sqrt(n)

        for key, lo, hi in (
            ("fixed_width", delta_b - half_lo, delta_b + half_hi),
            ("normal", delta_b - z * se_b, delta_b + z * se_b),
        ):
            classes = [
                "reference" if a == REF
                else core.classify_delta(lo[i], hi[i], WS4_EQUIVALENCE_MARGIN)
                for i, a in enumerate(arms)
            ]
            out[key].append(rstar_of(rec_b, classes))
            branches[key].append(branch_of(rec_b, classes))
    return (
        {k: pd.Series(v, dtype=float) for k, v in out.items()},
        {k: pd.Series(v, dtype=str) for k, v in branches.items()},
    )


def normal_approx_agreement(q, base):
    """Does the normal approximation reproduce the published classification?
    If it does not, the `normal` column of the r* bootstrap is not usable."""
    n = q.shape[0]
    z = 1.959963984540054
    same = 0
    total = 0
    for _, row in base.iterrows():
        if row.spec_class == "reference":
            continue
        d = (q[REF] - q[row.arm]).values
        se = d.std(ddof=1) / np.sqrt(n)
        cls = core.classify_delta(d.mean() - z * se, d.mean() + z * se,
                                 WS4_EQUIVALENCE_MARGIN)
        same += int(cls == row.spec_class)
        total += 1
    return same, total


# ------------------------------------------------ exact tests + flip distance ----


def arm_tests(q, r, base, idx):
    """Per arm: McNemar exact (the standard paired test for these binary
    outcomes), Holm across the ten degraded arms, and the flip distance -- how
    many single-question judge verdicts must change before the arm leaves DROP,
    and where r* lands if it does."""
    rows = []
    for _, row in base.iterrows():
        arm = row.arm
        if arm == REF:
            continue
        ref_only = int(((q[REF] == 1) & (q[arm] == 0)).sum())
        arm_only = int(((q[REF] == 0) & (q[arm] == 1)).sum())
        p = binomtest(ref_only, ref_only + arm_only, 0.5).pvalue

        flips, rstar_if_flipped = None, None
        if row.spec_class == "DROP":
            flips, rstar_if_flipped = flip_distance(q, r, base, arm, idx)

        rows.append({
            "arm": arm,
            "family": FAMILY[arm],
            "recall": round(row.recall, 6),
            "judge_accuracy": round(float(q[arm].mean()), 6),
            "delta_vs_ref": round(row.delta, 6),
            "delta_ci_lo": round(row.ci_lo, 6),
            "delta_ci_hi": round(row.ci_hi, 6),
            "spec_class": row.spec_class,
            "discordant_ref_only": ref_only,
            "discordant_arm_only": arm_only,
            "mcnemar_exact_p": round(p, 6),
            "flips_to_leave_drop": flips,
            "rstar_if_this_arm_plateau": rstar_if_flipped,
        })
    df = pd.DataFrame(rows)

    # Holm-Bonferroni across the ten degraded arms.
    o = df.sort_values("mcnemar_exact_p").copy()
    m = len(o)
    adj = np.minimum.accumulate(
        (o.mcnemar_exact_p.values * (m - np.arange(m)))[::-1]
    )[::-1]
    o["holm_p"] = np.clip(adj, 0, 1).round(6)
    return df.merge(o[["arm", "holm_p"]], on="arm")


def flip_distance(q, r, base, arm, idx, max_flips=80):
    """Smallest number of questions that must flip from 'reference right, arm
    wrong' to a tie before `arm` stops being DROP, and the r* that results.

    A flip to a tie is the mildest possible perturbation: it does not invent a
    win for the arm, it only removes one of the reference's wins."""
    ref_boot = core.boot_means(q[REF], idx)
    candidates = np.where((q[REF].values == 1) & (q[arm].values == 0))[0]
    for k in range(1, min(max_flips, len(candidates)) + 1):
        qa = q[arm].values.copy()
        qa[candidates[:k]] = 1.0  # tie: the arm now also gets it right
        lo, hi = core.percentile_ci(ref_boot - core.boot_means(qa, idx), 0.05)
        cls = core.classify_delta(lo, hi, WS4_EQUIVALENCE_MARGIN)
        if cls != "DROP":
            classes = [
                cls if a == arm else c
                for a, c in zip(base.arm, base.spec_class)
            ]
            return k, rstar_of(base.recall.values, classes)
    return None, None


# ------------------------------------------------ exhaustive mechanism pairs ----


def all_matched_pairs(q, r, idx):
    """Every cross-family pair inside the spec's own matching tolerance, not
    just the six the spec happened to name. spec 7.2 defines `matched` as
    |r(a) - r(b)| <= WS4_MATCH_TOLERANCE on the MEASURED axis; applying that
    definition exhaustively is the check on whether M2's support is general or
    an artefact of which pairs were listed."""
    arms = [a for a in q.columns if a != REF]
    rows = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            if FAMILY[a] == FAMILY[b]:
                continue
            gap = abs(r[a].mean() - r[b].mean())
            if not core.is_matched(r[a].mean(), r[b].mean(), WS4_MATCH_TOLERANCE):
                continue
            d = (q[a] - q[b]).values
            lo, hi = core.percentile_ci(core.boot_means(q[a], idx)
                                       - core.boot_means(q[b], idx), 0.05)
            x = int(((q[a] == 1) & (q[b] == 0)).sum())
            y = int(((q[a] == 0) & (q[b] == 1)).sum())
            rows.append({
                "arm_a": a, "arm_b": b,
                "family_a": FAMILY[a], "family_b": FAMILY[b],
                "recall_a": round(float(r[a].mean()), 6),
                "recall_b": round(float(r[b].mean()), 6),
                "recall_gap": round(float(gap), 6),
                "judge_diff": round(float(d.mean()), 6),
                "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                "mcnemar_exact_p": round(binomtest(x, x + y, 0.5).pvalue, 6),
                "prereg_pair": PREREG_PAIRS.get(frozenset((a, b)), ""),
                "tested_by_spec": frozenset((a, b)) in PREREG_PAIRS,
            })
    df = pd.DataFrame(rows).sort_values("mcnemar_exact_p").reset_index(drop=True)
    df["bonferroni_p"] = (df.mcnemar_exact_p * len(df)).clip(upper=1).round(6)
    return df


# ------------------------------------------------------- shape of the curve ----


def curve_shape(q, r, p, idx):
    """Is there a knee, or a shallow line? Slope and Spearman of answer quality
    on measured recall across arms, plus the mediation through presence@10."""
    rec, qual, pres = r.mean().values, q.mean().values, p.mean().values
    Q, R, P = q.values, r.values, p.values

    slopes, rhos = [], []
    for ii in idx:
        rb, qb = R[ii].mean(0), Q[ii].mean(0)
        slopes.append(np.polyfit(rb, qb, 1)[0])
        rhos.append(spearmanr(rb, qb).statistic)
    slopes, rhos = np.array(slopes), np.array(rhos)

    ref_present = P[:, list(q.columns).index(REF)] == 1
    ref_right = Q[:, list(q.columns).index(REF)] == 1
    conversion = float((ref_present & ref_right).sum() / ref_present.sum())

    d_pres = pres[0] - pres
    d_qual = qual[0] - qual
    mediation_slope = float(np.polyfit(d_pres, d_qual, 1)[0])

    return {
        "slope": (float(np.polyfit(rec, qual, 1)[0]),
                  *core.percentile_ci(slopes, 0.05)),
        "spearman": (float(spearmanr(rec, qual).statistic),
                     *core.percentile_ci(rhos, 0.05)),
        "spearman_presence": float(spearmanr(pres, qual).statistic),
        "conversion": conversion,
        "mediation_slope": mediation_slope,
        "presence_span": float(pres.max() - pres.min()),
        "quality_span": float(qual.max() - qual.min()),
        "quality_span_ex_worst": float(np.sort(qual)[1:].max() - np.sort(qual)[1:].min()),
        "predicted_quality_span": float((pres.max() - pres.min()) * conversion),
    }


# ------------------------------------------------------------------- main ----


def main() -> None:
    q, p, r = load_matrices()
    n = q.shape[0]
    idx = core.bootstrap_indices(n, WS4_N_BOOTSTRAP, SEED)

    base = spec_classes(q, r, idx)
    published = pd.read_csv(ws4_dir() / "summary.csv")
    published = published[published.metric == PRIMARY].set_index("arm")
    mismatches = [
        a for a in base.arm
        if base.set_index("arm").loc[a, "spec_class"] != published.loc[a, "class"]
    ]
    assert not mismatches, f"reproduction differs from summary.csv: {mismatches}"
    print(f"reproduced summary.csv classification for all {len(base)} arms")

    point_rstar = rstar_of(base.recall.values, base.spec_class.values)
    point_branch = branch_of(base.recall.values, base.spec_class.values)
    boots, branches = rstar_bootstrap(q, r, idx, base)
    agree_same, agree_total = normal_approx_agreement(q, base)
    shape = curve_shape(q, r, p, idx)

    arms_df = arm_tests(q, r, base, idx)
    arms_df.to_csv(ws4_dir() / "arm_tests_corrected.csv", index=False)

    pairs_df = all_matched_pairs(q, r, idx)
    pairs_df.to_csv(ws4_dir() / "mechanism_pairs_all.csv", index=False)

    rows = [
        ("rstar", "point_estimate", point_rstar, "", "",
         f"spec 7.1 rule on the frozen runs; branch {point_branch}; "
         "matches results/recall-vs-quality.md"),
    ]
    for key, s in boots.items():
        d = s.dropna()
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows += [
            ("rstar", f"undefined_share_{key}", float(s.isna().mean()), "", "",
             "resamples with NO plateau at all: the top degraded arm is not "
             "PLATEAU, so spec 7.1 yields no r* and the branch is K3, not K1"),
            ("rstar", f"branch_K1_share_{key}",
             float((branches[key] == "K1").mean()), "", "",
             "share of resamples reproducing the PUBLISHED branch K1"),
            ("rstar", f"median_{key}", float(d.median()), float(lo), float(hi),
             f"{WS4_N_BOOTSTRAP} paired resamples, seed {SEED}, {key} CI width; "
             "median and 95% interval over the resamples where r* is defined"),
            ("rstar", f"p_within_0.01_of_point_{key}",
             float((s - point_rstar).abs().lt(0.01).mean()), "", "",
             "share of ALL resamples putting r* within +/-0.01 of the published "
             "knee (undefined counts against)"),
        ]
    rows += [
        ("rstar", "normal_approx_agreement", agree_same / agree_total, "", "",
         f"{agree_same}/{agree_total} arms classified identically by the normal "
         "approximation and the published percentile CI"),
        ("curve", "slope_quality_per_unit_recall", *shape["slope"],
         "OLS across all 11 arms; ~1 accuracy point per 10 recall points"),
        ("curve", "spearman_recall_quality", *shape["spearman"],
         "monotone across the whole range: there is a line, not a knee"),
        ("curve", "spearman_presence_quality", shape["spearman_presence"], "", "",
         "presence@10 predicts quality better than recall@10 does"),
        ("mediation", "conversion_p_correct_given_present", shape["conversion"],
         "", "", "reference arm: P(judge correct | gold in the top-10)"),
        ("mediation", "slope_quality_delta_on_presence_delta",
         shape["mediation_slope"], "", "",
         "close to the conversion rate: no saturation, quality tracks presence"),
        ("mediation", "presence_span_across_ladder", shape["presence_span"],
         "", "", "answer-presence@10, best arm minus worst"),
        ("mediation", "quality_span_across_ladder", shape["quality_span"],
         "", "", "judge accuracy, best arm minus worst"),
        ("mediation", "quality_span_excluding_worst_arm",
         shape["quality_span_ex_worst"], "", "",
         "judge accuracy span with sq8_np4 removed"),
        ("mediation", "predicted_quality_span", shape["predicted_quality_span"],
         "", "", "presence span x conversion: what pure mediation predicts"),
        ("power", "equivalence_margin", WS4_EQUIVALENCE_MARGIN, "", "",
         "spec 7.1 PLATEAU margin -- compare to quality_span_across_ladder"),
        ("power", "median_ci_half_width",
         float(((base.ci_hi - base.ci_lo) / 2)[base.spec_class != "reference"].median()),
         "", "", "smallest drop the design can resolve"),
        ("mechanism", "matched_cross_family_pairs_total", len(pairs_df), "", "",
         "spec 7.2 tolerance applied exhaustively"),
        ("mechanism", "matched_pairs_never_tested",
         int((~pairs_df.tested_by_spec).sum()), "", "",
         "matched pairs the spec's A-F list does not contain"),
        ("mechanism", "untested_pairs_significant_bonferroni",
         int(((~pairs_df.tested_by_spec) & (pairs_df.bonferroni_p < 0.05)).sum()),
         "", "", "M2 is not general: see mechanism_pairs_all.csv"),
    ]

    out = pd.DataFrame(rows, columns=["diagnostic", "statistic", "value",
                                      "ci_lo", "ci_hi", "note"])
    out.to_csv(ws4_dir() / "rstar_sensitivity.csv", index=False)

    print(f"\npoint r* = {point_rstar:.4f}  (branch {point_branch})")
    for key, s in boots.items():
        d = s.dropna()
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  {key:12s} undefined {s.isna().mean():.3f}  K1 "
              f"{(branches[key] == 'K1').mean():.3f}  median {d.median():.3f}  "
              f"95% [{lo:.3f}, {hi:.3f}]  "
              f"P(within 0.01) = {(s - point_rstar).abs().lt(0.01).mean():.3f}")
    print(f"\nnormal-approx agreement with published classes: "
          f"{agree_same}/{agree_total}")
    print("\nwrote rstar_sensitivity.csv, arm_tests_corrected.csv, "
          "mechanism_pairs_all.csv")


if __name__ == "__main__":
    main()
