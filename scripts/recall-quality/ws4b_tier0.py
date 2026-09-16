#!/usr/bin/env python
"""WS4b Tier 0: re-score the FROZEN WS4 runs.csv with the spec 3 corrections.

Zero API spend, no Milvus, no re-retrieval. Reads results/ws4/runs.csv and the
passage cache WS4 already built (data/ws4_cache/passages.parquet), writes into
results/ws4b/. results/ws4/ is NEVER written to.

Design: results/recall-vs-quality.md word-boundary answer presence      3.4 primary stratum
  3.2 decline detection                  3.5 gates G0 / G0b / G0c
  3.3 staleness audit candidates         7.1 branches T0-A..D

Run:  python scripts/recall-quality/ws4b_tier0.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib.recall_quality import gold as ws4b
from benchlib.recall_quality import core
from benchlib.config import (
    SEED,
    WS4_EQUIVALENCE_MARGIN,
    WS4_N_BOOTSTRAP,
    WS4_TOP_K,
    WS4B_T0_CONVERSION_DELTA,
    WS4B_WS4_CONVERSION,
    gt_path,
    ws4_cache,
    ws4_dir,
    ws4b_dir,
)

REF = "sq8_np512"
PARAMETRIC = "parametric"


# ------------------------------------------------------------- loading ----


def load_runs() -> pd.DataFrame:
    r = pd.read_csv(ws4_dir() / "runs.csv")
    r["ids"] = r.retrieved_ids.map(
        lambda s: [int(i) for i in ws4b.parse_golds(s)] if str(s).strip() != "[]" else []
    )
    return r


def load_passage_table() -> pd.DataFrame:
    p = ws4_cache() / "passages.parquet"
    if not p.exists():
        sys.exit(f"missing passage cache {p} -- WS4's cache is required; this "
                 "script does not re-read corpus shards")
    return pd.read_parquet(p).set_index("row")


def texts_for(ids, table) -> list[str]:
    """WS4 handed the generator title+text blocks but scored presence on the
    TEXT only (ws4_runner.passages_for -> render_context -> answer_present on
    p['text']). G0 is what proves this reading is right."""
    return [table.at[i, "text"] for i in ids[:WS4_TOP_K] if i >= 0 and i in table.index]


# --------------------------------------------------------------- gates ----


def gate_g0(runs, table) -> pd.DataFrame:
    """G0: re-derived texts must reproduce the PUBLISHED answer_presence_at_10
    under the OLD rule, on every row that had passages. G0b: the corrected
    rule may only ever remove matches. G0c: row counts unchanged."""
    rows = []
    for _, x in runs.iterrows():
        if x.arm == PARAMETRIC:
            rows.append((None, None))
            continue
        t = texts_for(x.ids, table)
        rows.append((ws4b.answer_present_old(x.golds, t),
                     ws4b.answer_present_wb(x.golds, t)))
    out = pd.DataFrame(rows, columns=["presence_old_rederived", "presence_wb"],
                       index=runs.index)
    return out


def check_gates(runs, derived) -> dict:
    ok = runs.arm != PARAMETRIC
    pub = runs.loc[ok, "answer_presence_at_10"].astype(bool)
    red = derived.loc[ok, "presence_old_rederived"].astype(bool)
    wb = derived.loc[ok, "presence_wb"].astype(bool)

    g0_bad = int((pub != red).sum())
    g0b_bad = int((wb & ~red).sum())
    g0c_ok = len(runs) == 6000

    print(f"G0  re-derived == published (old rule): {len(pub) - g0_bad}/{len(pub)}"
          f"   mismatches={g0_bad}")
    print(f"G0b corrected is a subset of old:        gains={g0b_bad}")
    print(f"G0c row count 6000:                      {g0c_ok} ({len(runs)})")
    if g0_bad or g0b_bad or not g0c_ok:
        sys.exit("GATE FAILURE -- nothing downstream is trustworthy")
    return {"g0_mismatches": g0_bad, "g0b_gains": g0b_bad, "g0c_rows": len(runs)}


# -------------------------------------------------- corrected GT stratum ----


def corrected_gt_stratum(runs, table) -> pd.DataFrame:
    """spec 3.4: the primary stratum is 'the EXACT top-10 carries the gold'.
    Recomputed under the word-boundary rule from the WS1 exact GT, not from any
    arm's retrieval. qid is the row index into ws4_indices."""
    gt = np.load(gt_path("10m"))["ws4_indices"]
    q = runs.drop_duplicates("qid")[["qid", "golds"]]
    rows = []
    for _, x in q.iterrows():
        ids = [int(i) for i in gt[int(x.qid), :WS4_TOP_K]]
        t = texts_for(ids, table)
        rows.append({
            "qid": int(x.qid),
            "gt_presence_old": ws4b.answer_present_old(x.golds, t),
            "gt_presence_wb": ws4b.answer_present_wb(x.golds, t),
            "degenerate_golds": ws4b.degenerate_golds(x.golds),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------- re-scoring ----


def rescore(runs, derived, strat) -> pd.DataFrame:
    r = runs.join(derived)
    r["idk_corrected"] = r.answer.map(ws4b.is_idk_prefix)
    r = r.merge(strat, on="qid", how="left")
    # spec 3.4: primary stratum, with unscoreable questions removed (3.1)
    r["in_primary"] = r.gt_presence_wb & ~r.degenerate_golds & (r.subset == "main")
    return r


def conversion_table(r) -> pd.DataFrame:
    """P(judge correct | gold present) and the decline/error split, old rule
    vs corrected, per arm. This is the denominator the whole compression
    argument is stated in."""
    rows = []
    for arm, g in r[r.arm != PARAMETRIC].groupby("arm"):
        for label, mask in (
            ("ws4_published", g.answer_presence_at_10.astype(bool) & (g.subset == "main")),
            ("corrected", g.presence_wb.astype(bool) & g.in_primary),
        ):
            s = g[mask]
            if not len(s):
                continue
            idk = s.idk if label == "ws4_published" else s.idk_corrected
            rows.append({
                "arm": arm,
                "rule": label,
                "n_present": len(s),
                "conversion": round(float(s.judge_correct.mean()), 6),
                "decline_rate": round(float(idk.mean()), 6),
                "error_rate": round(float(((~idk.astype(bool))
                                           & (~s.judge_correct.astype(bool))).mean()), 6),
            })
    return pd.DataFrame(rows).sort_values(["rule", "arm"]).reset_index(drop=True)


def classification(r, metric="judge_correct") -> pd.DataFrame:
    """spec 7.1 DROP/PLATEAU on the CORRECTED primary stratum, so branch T0-D
    can be decided. Same bootstrap protocol and seed as WS4."""
    d = r[r.in_primary & (r.arm != PARAMETRIC)]
    q = d.pivot(index="qid", columns="arm", values=metric).astype(float)
    rec = d.pivot(index="qid", columns="arm", values="recall_at_10").astype(float)
    idx = core.bootstrap_indices(len(q), WS4_N_BOOTSTRAP, SEED)
    ref_boot = core.boot_means(q[REF], idx)
    rows = []
    for arm in q.columns:
        if arm == REF:
            rows.append({"arm": arm, "recall": float(rec[arm].mean()),
                         "quality": float(q[arm].mean()), "delta": 0.0,
                         "ci_lo": 0.0, "ci_hi": 0.0, "class": "reference"})
            continue
        lo, hi = core.percentile_ci(ref_boot - core.boot_means(q[arm], idx), 0.05)
        rows.append({
            "arm": arm, "recall": float(rec[arm].mean()),
            "quality": float(q[arm].mean()),
            "delta": float(q[REF].mean() - q[arm].mean()),
            "ci_lo": lo, "ci_hi": hi,
            "class": core.classify_delta(lo, hi, WS4_EQUIVALENCE_MARGIN),
        })
    out = pd.DataFrame(rows).sort_values("recall", ascending=False)
    return out.round(6).reset_index(drop=True)


def knee_and_branch(cls) -> tuple:
    arms = [(a, rc, c) for a, rc, c in zip(cls.arm, cls.recall, cls["class"])
            if c != "reference"]
    r_star = core.knee(arms)
    arm = next((a for a, rc, _ in arms if r_star is not None and rc == r_star), None)
    return r_star, core.knee_branch(arms), arm


# ------------------------------------------------------------ branches ----


WS4_RSTAR = 0.9231111111111113
WS4_RSTAR_ARM = "pca_uc_384_sq8_np512"


def resolve_branches(conv, r_star, branch, r_star_arm, n_stale_candidates) -> pd.DataFrame:
    """spec 7.1. T0-C needs the BLIND MANUAL PASS of 3.3 and is therefore
    reported as pending here -- the script emits its candidates and refuses to
    guess the labels.

    ⚠️ SPEC DEFECT, found by running it (amendment 2026-09-11 in the design
    doc). T0-D reads "branch K1 or r* changes". It does not say whether r*
    means the VALUE or the ARM, and on corrected data they disagree: the arm
    is identical, its measured recall moved 0.0015 only because recall is now
    averaged over a different question subset. Both readings are emitted and
    NEITHER is suppressed -- picking the flattering one after seeing the
    number is the exact failure pre-registration exists to prevent.
    """
    c_new = float(conv[(conv.arm == REF) & (conv.rule == "corrected")].conversion.iloc[0])
    delta = c_new - WS4B_WS4_CONVERSION
    same_branch = branch == "K1"
    same_value = r_star is not None and abs(r_star - WS4_RSTAR) < 1e-9
    same_arm = r_star_arm == WS4_RSTAR_ARM
    literal = same_branch and same_value        # T0-D as written
    substantive = same_branch and same_arm      # T0-D as intended

    rows = [
        {"branch": "T0-A", "reading": "literal",
         "fires": bool(abs(delta) < WS4B_T0_CONVERSION_DELTA and literal),
         "evidence": f"conversion delta {delta:+.4f}; branch {branch}; "
                     f"r* {r_star:.6f} vs {WS4_RSTAR:.6f}"},
        {"branch": "T0-A", "reading": "substantive",
         "fires": bool(abs(delta) < WS4B_T0_CONVERSION_DELTA and substantive),
         "evidence": f"conversion delta {delta:+.4f}; branch {branch}; "
                     f"r* arm {r_star_arm} vs {WS4_RSTAR_ARM}"},
        {"branch": "T0-B", "reading": "literal",
         "fires": bool(abs(delta) >= WS4B_T0_CONVERSION_DELTA and literal),
         "evidence": f"conversion delta {delta:+.4f} vs threshold "
                     f"{WS4B_T0_CONVERSION_DELTA}"},
        {"branch": "T0-B", "reading": "substantive",
         "fires": bool(abs(delta) >= WS4B_T0_CONVERSION_DELTA and substantive),
         "evidence": f"conversion delta {delta:+.4f} vs threshold "
                     f"{WS4B_T0_CONVERSION_DELTA}"},
        {"branch": "T0-C", "reading": "both", "fires": None,
         "evidence": f"PENDING blind audit of {n_stale_candidates} candidates "
                     "(spec 3.3); threshold WS4B_T0C_STALE_SHARE"},
        {"branch": "T0-D", "reading": "literal", "fires": bool(not literal),
         "evidence": f"r* value {r_star:.6f} != {WS4_RSTAR:.6f} "
                     f"(delta {r_star - WS4_RSTAR:+.6f}); branch {branch}"},
        {"branch": "T0-D", "reading": "substantive", "fires": bool(not substantive),
         "evidence": f"r* arm {r_star_arm}; branch {branch}"},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main ----


def main() -> None:
    out = ws4b_dir()
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs()
    table = load_passage_table()
    print(f"loaded {len(runs)} rows, {len(table)} cached passages\n")

    derived = gate_g0(runs, table)
    gates = check_gates(runs, derived)

    strat = corrected_gt_stratum(runs, table)
    r = rescore(runs, derived, strat)

    # ---- spec 3.1 / 3.2 exposure
    qs = r.drop_duplicates("qid")
    degen = qs[qs.degenerate_golds]
    degen_out = degen[["qid", "question", "golds", "subset",
                       "gt_presence_old", "gt_presence_wb"]]
    degen_out.to_csv(out / "tier0_degenerate_golds.csv", index=False)

    main_rows = r[(r.subset == "main") & (r.arm != PARAMETRIC)]
    lost = int((main_rows.answer_presence_at_10.astype(bool)
                & ~main_rows.presence_wb.astype(bool)).sum())
    idk_gained = int((r.idk_corrected & ~r.idk.astype(bool)).sum())

    print(f"\nspec 3.1  questions with NO usable gold alias: {len(degen)} of {qs.qid.nunique()}")
    print(f"          arm-rows losing presence under word boundaries: {lost} "
          f"of {len(main_rows)} ({lost / len(main_rows):.1%})")
    print(f"spec 3.2  declines recovered from the error bucket: {idk_gained}")

    # ---- spec 3.3 staleness audit candidates (blind: no arm, no recall)
    cand = r[(r.subset == "main") & (r.arm != PARAMETRIC)
             & (~r.idk_corrected) & (~r.judge_correct.astype(bool))]
    cand = (cand.drop_duplicates("qid")[["qid", "question", "golds", "answer",
                                         "judge_reason"]]
            .sort_values("qid").reset_index(drop=True))
    cand["label"] = ""   # stale_gold | ambiguous_question | model_error | judge_strict
    cand["auditor"] = ""
    cand.to_csv(out / "tier0_staleness_candidates.csv", index=False)
    print(f"spec 3.3  staleness audit candidates emitted: {len(cand)} (labels blank)")

    # ---- spec 3.4 + re-scoring
    conv = conversion_table(r)
    conv.to_csv(out / "tier0_conversion.csv", index=False)

    cls = classification(r)
    cls.to_csv(out / "tier0_classification.csv", index=False)
    r_star, branch, r_star_arm = knee_and_branch(cls)

    keep = ["qid", "arm", "subset", "recall_at_10", "answer_presence_at_10",
            "presence_wb", "idk", "idk_corrected", "judge_correct",
            "gt_presence_old", "gt_presence_wb", "degenerate_golds", "in_primary"]
    r[keep].to_csv(out / "tier0_runs_corrected.csv", index=False)

    branches = resolve_branches(conv, r_star, branch, r_star_arm, len(cand))
    branches.to_csv(out / "tier0_branches.csv", index=False)

    gate_df = pd.DataFrame([
        {"gate": k, "value": v, "passed": True} for k, v in gates.items()
    ])
    gate_df.to_csv(out / "tier0_gates.csv", index=False)

    # ---- report
    ref = conv[conv.arm == REF].set_index("rule")
    print("\n--- reference arm, spec 3.4 primary stratum ---")
    print(ref[["n_present", "conversion", "decline_rate", "error_rate"]].to_string())
    print(f"\nspec 7.1  r* = {r_star:.6f} on {r_star_arm}   branch = {branch}"
          f"   (WS4 published K1 / 0.923111 on {WS4_RSTAR_ARM})")
    print("\nbranches:")
    print(branches.to_string(index=False))
    print(f"\nwrote 6 CSVs into {out}/")


if __name__ == "__main__":
    main()
