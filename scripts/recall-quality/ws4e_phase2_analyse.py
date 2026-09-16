#!/usr/bin/env python
"""WS4e Phase 2 -- the decomposition, gates K1 (real data)/K4/K5/K6
(redefined, Amendment 1), predictions P1/P2, the fix-round-1 diagnostics,
and every remaining committed output. Costs $0.

Design: results/recall-vs-quality.md

⚠️ NO changepoint is fitted. Four arms against a four-parameter two-segment
model leaves zero residual degrees of freedom; the single slope is fitted
instead, on a recall@10 x-axis held fixed across every depth (spec 6.4).

⚠️ There is no temperature-0 cell. Amendment 1 (spec 3.5) withdrew it as
impossible on the pinned model and replaced it with a replicate cell:
WS4E_REPLICATE_ARM re-run at WS4E_REPLICATE_K, on its own checkpoint. Gate
K6 is redefined here to test that cell's disagreement rate R rather than a
temperature-0 null.

⚠️ Phase 0's gate K1 (results/ws4e/prefix_integrity.csv) handed EVERY cell
the same `qids` object -- it passed by construction and verified nothing.
This script reruns K1 against the REAL per-cell question and arm sets read
back from Phase 1's checkpoints, and appends that result to gates.csv
alongside Phase 0's.

⚠️ FIX ROUND 1 (adversarial review, CRITICAL 1): separable(k) is nested and
monotone in k (98 subset 172 subset 206 subset 241 on this data), so the
committed signal(k) statistic's own denominator changes composition as k
grows -- a Simpson's-paradox risk with the pre-registered estimator as the
collapsing variable. `composition_check.csv` reports a population held
FIXED across k as a diagnostic. It does NOT replace, gate, or change
signal(k), the branch rule, or any committed number.

⚠️ TASK 10 (final whole-branch review): composition_check.csv gained
membership-churn columns (a flat RATE on the fixed-98/fixed-23 is not the
same as a fixed discriminating SET -- see ws4e.fixed_population_membership_
check) and two cross-population rate columns (ws4e.fixed_population_rate).
dedup_sensitivity.csv gained the SAME fixed-composition diagnostic
recomputed under first_wins, because it is not robust to the dedup rule
either. noise_share_at_k10.csv is new: WS4e's own noise-share figure on
WS4E_ARMS (2/76), alongside spec 2.3's better-powered but DIFFERENT-arm-set
reference figure (12/57 on np4/np12/np48/np512), committed together so
neither is quoted from a non-CSV source again. None of this changes
signal(k), the branch rule, or any previously committed number -- verified
by re-running this script and diffing every existing column.

Run:
    PYTHONPATH=. .venv/bin/python scripts/recall-quality/ws4e_phase2_analyse.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest  # noqa: E402
from benchlib.recall_quality import depth_runner as ws4e_runner  # noqa: E402
from benchlib.recall_quality import depth as ws4e  # noqa: E402
from benchlib.recall_quality import eligibility as ws4d  # noqa: E402
from benchlib.config import (SEED, WS4_SCALE, WS4_TOP_K,  # noqa: E402
                             WS4B_CHANGEPOINT_BOOT_N, WS4E_ARMS,
                             WS4E_CAP_USD, WS4E_FLAT_MIN_DIFF,
                             WS4E_K_VALUES, WS4E_REPLICATE_ARM,
                             WS4E_REPLICATE_K, WS4E_REPLICATE_MIN_N,
                             WS4E_TEMPERATURE, gt_path, ws4_cache, ws4b_dir,
                             ws4d_cache, ws4d_dir, ws4e_dir)
from benchlib.quantization import recall_at_k  # noqa: E402
from benchlib.agent_harness import load_checkpoint  # noqa: E402

RET = ws4_cache() / "retrieval"
COMMITTED_K10_ARMS = ("sq8_np4", "sq8_np48", "sq8_np512")
WORST_ARM, BEST_ARM = WS4E_ARMS[0], WS4E_ARMS[-1]   # sq8_np1, sq8_np512

# Spec 2.3's "12 of 57" reference figure was measured on
# a DIFFERENT, better-powered four-arm set (np4/np12/np48/np512) than
# WS4E_ARMS (np1/np4/np48/np512) -- np1 was swapped in for np12 specifically
# to lift the k=1 separability ceiling (spec 2.2). Kept distinct from
# WS4E_ARMS/COMMITTED_K10_ARMS so the two are never silently conflated.
SPEC23_REF_ARMS = ("sq8_np4", "sq8_np12", "sq8_np48", "sq8_np512")

# Two concurrency incidents wrote duplicate raw lines for the same
# (qid, arm) key -- 549 in the k=1 checkpoint, 6 in k=5. Fix round 1,
# IMPORTANT 4 checks the dedup-rule sensitivity only at these depths;
# k=3 and k=10 have zero duplicate lines (verified in the round-1 review).
DEDUP_SENSITIVE_K = (1, 5)

# sq8_np48's committed k=10 rows live ONLY in WS4d's Phase 3 checkpoint, not
# in results/ws4/runs.csv -- see scripts/recall-quality/ws4e_phase1_run.py's own comment on
# this exact point. Duplicated here rather than imported: scripts/ is
# deliberately NOT a package, so importing
# it would require making it one, which would change how every other script
# resolves its imports.
PHASE3_CHECKPOINT = ws4d_cache() / "phase3_runs_checkpoint.jsonl"

# Fix round 1, IMPORTANT 5: k=10's four cells were NOT all generated at the
# same time. np4/np512 are WS4's own 2026-09-06/07 rows, np48 is WS4d Phase
# 3's 2026-09-12 rows, and np1 is WS4e's own 2026-09-13 row -- while every
# k in {1, 3, 5} is entirely WS4e-generated on 2026-09-13. Gate K3 re-judges
# committed WS4 rows to test JUDGE reproduction; it says nothing about
# GENERATOR drift across these vintages, and none of this project's gates
# test for that. Recorded, not mitigated.
GENERATION_VINTAGE_NOTE = (
    "k=10's four cells are NOT one generation vintage: sq8_np4/sq8_np512 "
    "are WS4's 2026-09-06/07 rows, sq8_np48 is WS4d Phase 3's 2026-09-12 "
    "rows, sq8_np1 is WS4e's own 2026-09-13 row. k in {1,3,5} are entirely "
    "2026-09-13 (WS4e). Gate K3 re-judges committed rows to test JUDGE "
    "reproduction only -- it does not test generator drift across these "
    "vintages, and no gate here does.")


def load_committed_k10(qids, arms=COMMITTED_K10_ARMS):
    """(qid, arm) judge_correct for `arms` at k=10, from both sources those
    arms actually live in. Generalised beyond
    COMMITTED_K10_ARMS so SPEC23_REF_ARMS can reuse it for sq8_np12 too --
    np12 lives only in PHASE3_CHECKPOINT, exactly like np48."""
    want = {int(q) for q in qids}
    frames = []
    runs = pd.read_csv("results/ws4/runs.csv")
    runs = runs[(runs.subset == "main") & runs.arm.isin(arms)]
    frames.append(runs[["qid", "arm", "judge_correct"]])
    if PHASE3_CHECKPOINT.exists():
        rows = [r for r in load_checkpoint(PHASE3_CHECKPOINT).values()
                if r.get("subset") == "main" and r["arm"] in arms]
        if rows:
            frames.append(pd.DataFrame(rows)[["qid", "arm", "judge_correct"]])
    committed = pd.concat(frames, ignore_index=True)
    committed["qid"] = committed["qid"].astype(int)
    committed = committed[committed.qid.isin(want)]
    dup = committed.duplicated(subset=["qid", "arm"]).sum()
    assert dup == 0, f"{dup} duplicate (qid, arm) rows in committed k=10 data"
    return committed


def verdicts_for(k, qids, *, first_wins=False):
    """qid -> {arm: judge_correct} for one PRIMARY cell.

    ⚠️ Reads ONLY the primary checkpoint (replicate=False, the default) --
    the Amendment 1 replicate cell lives on its own checkpoint file
    (ws4e_runner.cell_checkpoint(..., replicate=True)) precisely so it
    cannot contaminate this one. It is read separately by
    `replicate_verdicts` below and never merged in here.

    `first_wins`: fix round 1, IMPORTANT 4 -- a sensitivity check on the
    checkpoint dedup rule. Every COMMITTED number uses first_wins=False
    (ws4e_runner.cell_rows / load_checkpoint's last-wins) unchanged.
    """
    loader = (ws4e_runner.cell_rows_first_wins if first_wins
              else ws4e_runner.cell_rows)
    out = {int(q): {} for q in qids}
    for (qid, arm), row in loader(k, WS4E_TEMPERATURE).items():
        q = int(qid)
        if q in out:
            out[q][arm] = bool(row["judge_correct"])
    if k == WS4_TOP_K:
        for r in load_committed_k10(qids).itertuples():
            q = int(r.qid)
            if q in out:
                out[q][r.arm] = bool(r.judge_correct)
    return out


def replicate_verdicts(qids):
    """qid -> judge_correct for the Amendment 1 replicate cell alone."""
    want = {int(q) for q in qids}
    rows = ws4e_runner.cell_rows(WS4E_REPLICATE_K, WS4E_TEMPERATURE,
                                 replicate=True)
    out = {}
    for (qid, arm), row in rows.items():
        if arm == WS4E_REPLICATE_ARM and int(qid) in want:
            out[int(qid)] = bool(row["judge_correct"])
    return out


def real_cell_basis(k, qids):
    """Task 10(a): the qids and arms ACTUALLY recorded for this depth, read
    back from the checkpoints -- not assumed. A qid missing from even one
    arm drops out of the intersection (a short cell then fails K1 against
    the full 264); an arm that should not exist shows up in `arms_present`
    (an unexpected arm then fails K1's arms_by_cell check)."""
    by_arm = defaultdict(set)
    for (qid, arm), _ in ws4e_runner.cell_rows(k, WS4E_TEMPERATURE).items():
        by_arm[arm].add(int(qid))
    if k == WS4_TOP_K:
        for r in load_committed_k10(qids).itertuples():
            by_arm[r.arm].add(int(r.qid))
    arms_present = tuple(sorted(by_arm))
    common = set.intersection(*by_arm.values()) if by_arm else set()
    want = {int(q) for q in qids}
    qids_present = sorted(q for q in common if q in want)
    return qids_present, arms_present


def accidental_replicate_flags(qids):
    """Task 10(c): two concurrency incidents wrote two independent raw lines
    for the same (qid, arm, k) key -- 549 in the k=1 primary checkpoint, 6
    in k=5. ws4e_runner.cell_rows/load_checkpoint DEDUPS these away by
    (qid, arm), so this reads the RAW jsonl lines directly.

    Returns (disagree_flags, detail_rows). ⚠️ Supplementary corroboration
    only -- never promoted to primary; the pre-registered sq8_np512/k=1
    replicate cell remains what gate K6 reads.
    """
    want = {int(q) for q in qids}
    flags, detail = [], []
    for k in WS4E_K_VALUES:
        path = ws4e_runner.cell_checkpoint(k, WS4E_TEMPERATURE)
        if not path.exists():
            continue
        by_key = defaultdict(list)
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                qid = int(row["qid"])
                if qid not in want:
                    continue
                by_key[(qid, row["arm"])].append(bool(row["judge_correct"]))
        for (qid, arm), verdicts in by_key.items():
            if len(verdicts) == 1:
                continue
            if len(verdicts) != 2:
                sys.exit(f"unexpected {len(verdicts)} raw lines for "
                         f"(qid={qid}, arm={arm}, k={k}) -- the accidental "
                         "replicate analysis assumes exactly one duplicate")
            agree = verdicts[0] == verdicts[1]
            flags.append(agree)
            detail.append({"k": k, "qid": qid, "arm": arm, "agree": agree})
    return [not a for a in flags], detail   # disagreement flags


def recall_matrix(ids, gt, qids):
    """(n_questions, n_arms) recall@10 -- the arm's FIXED identity, spec 6.4.
    Rows follow `qids`' order; columns follow WS4E_ARMS' order."""
    return np.array([[recall_at_k(ids[a][q:q + 1], gt[q:q + 1], WS4_TOP_K)
                      for a in WS4E_ARMS] for q in qids])


def accuracy_matrix(verdicts, qids):
    """(n_questions, n_arms) judge_correct as 0/1, for one depth's verdicts."""
    return np.array([[float(verdicts[q][a]) for a in WS4E_ARMS] for q in qids])


def main():
    out = ws4e_dir()
    strata = ws4e.build_strata(
        pd.read_csv(ws4d_dir() / "eligibility_labels.csv"),
        pd.read_csv(ws4b_dir() / "tier0_runs_corrected.csv"))
    qids = strata["primary"]
    ids = {a: np.load(RET / f"{a}.npz")["ids"] for a in WS4E_ARMS}
    separable = {k: ws4e.separable_at(ids, qids, k) for k in WS4E_K_VALUES}

    # --- gate K1, rerun against REAL per-cell data (task 10a) -------------
    real_qids_by_cell, real_arms_by_cell = {}, {}
    for k in WS4E_K_VALUES:
        real_qids_by_cell[k], real_arms_by_cell[k] = real_cell_basis(k, qids)
    k1_real = ws4e.gate_k1(real_qids_by_cell, qids=qids,
                           arms=tuple(sorted(WS4E_ARMS)),
                           arms_by_cell=real_arms_by_cell,
                           x_axis="recall_at_10")
    k1_real["phase"] = "phase2_real_data"
    k1_real["kind"] = "gate"
    print("K1 (real per-cell data):", json.dumps(k1_real, indent=2))

    prior_gates = pd.read_csv(out / "prefix_integrity.csv").to_dict("records")
    for g in prior_gates:
        g["phase"] = "phase0_by_construction"
        g["kind"] = "gate"
    k3 = json.loads((out / "k3_rejudge.json").read_text())
    k3["phase"] = "phase1"
    k3["kind"] = "gate"
    manifest.record(out / "k3_rejudge.json")   # fix round 1, S6

    if not k1_real["passed"]:
        gates = prior_gates + [k3, k1_real]
        pd.DataFrame(gates).to_csv(out / "gates.csv", index=False)
        manifest.record(out / "gates.csv")
        sys.exit("GATE K1 FAILURE on real per-cell data -- nothing "
                 "downstream is interpretable; see gates.csv")

    # --- the headline decomposition ----------------------------------------
    cells, buckets, all_verdicts = {}, {}, {}
    for k in WS4E_K_VALUES:
        v = verdicts_for(k, qids)
        missing = [q for q in qids if len(v[q]) != len(WS4E_ARMS)]
        if missing:
            sys.exit(f"k={k}: {len(missing)} question(s) missing an arm "
                     f"(first: {missing[:3]}) -- Phase 1 is incomplete")
        all_verdicts[k] = v
        d = ws4e.decompose(v, separable[k], qids)
        buckets[k] = d.pop("buckets")
        cells[k] = d

    boot = ws4e.bootstrap_signal(buckets, separable, qids,
                                 n_boot=WS4B_CHANGEPOINT_BOOT_N, seed=SEED)
    flat = ws4e.flatness_branch(boot, WS4E_FLAT_MIN_DIFF)
    branch = "X-D" if flat == "flat" else ws4e.peak_branch(
        boot["signal_by_k"], WS4E_K_VALUES)

    # --- S4: P1/P2 evaluated explicitly, X-F checked ------------------------
    p1 = ws4e.evaluate_p1(cells)
    p2 = ws4e.evaluate_p2(cells)
    if not p1["holds"]:
        branch = "X-F"

    # --- Phase 1b (amended): the replicate validation cell -> gate K6 ------
    # Computed BEFORE discrimination_by_k.csv is written (fix round 1, S5):
    # an X-G outcome must withdraw `signal` from the headline file, not just
    # from the printed summary.
    v_k1 = all_verdicts[WS4E_REPLICATE_K]
    primary_arm = {q: v_k1[q][WS4E_REPLICATE_ARM] for q in qids}
    rep = replicate_verdicts(qids)
    missing_rep = [q for q in qids if q not in rep]
    if missing_rep:
        sys.exit(f"replicate cell missing {len(missing_rep)} question(s) "
                 f"(first: {missing_rep[:3]}) -- Phase 1b is incomplete")

    sep_map_k1 = separable[WS4E_REPLICATE_K]
    disagree_all = [primary_arm[q] != rep[q] for q in qids]
    disagree_ident = [primary_arm[q] != rep[q] for q in qids if not sep_map_k1[q]]
    disagree_sep = [primary_arm[q] != rep[q] for q in qids if sep_map_k1[q]]

    r_all = ws4e.bootstrap_rate(disagree_all, n_boot=WS4B_CHANGEPOINT_BOOT_N,
                                seed=SEED)
    r_ident = ws4e.bootstrap_rate(disagree_ident,
                                  n_boot=WS4B_CHANGEPOINT_BOOT_N, seed=SEED)
    r_sep = ws4e.bootstrap_rate(disagree_sep, n_boot=WS4B_CHANGEPOINT_BOOT_N,
                                seed=SEED)

    k6 = ws4e.gate_k6(R=r_all["R"], ci_lo=r_all["ci_lo"], ci_hi=r_all["ci_hi"],
                      n_disagree=r_all["n_disagree"], min_n=WS4E_REPLICATE_MIN_N)
    v1_passed = k6["passed"]
    v2_ci_overlap = None
    if r_ident["ci_lo"] is not None and r_sep["ci_lo"] is not None:
        v2_ci_overlap = not (r_ident["ci_hi"] < r_sep["ci_lo"]
                              or r_sep["ci_hi"] < r_ident["ci_lo"])
    check = ws4e.four_sample_consistency_check(r_all["R"])

    if not k6["passed"]:
        branch = "X-G"   # spec 7/9: the most consequential failure, outranks X-F too

    # --- S2: CIs for share/null/separable_rate, not just signal -------------
    disc_stats = ws4e.bootstrap_discrimination_stats(
        buckets, separable, qids, n_boot=WS4B_CHANGEPOINT_BOOT_N, seed=SEED)

    # --- IMPORTANT 3: inverted discrimination (worst-right, best-wrong) -----
    inverted = {k: ws4e.inverted_fraction(all_verdicts[k], separable[k],
                                          WORST_ARM, BEST_ARM, qids)
               for k in WS4E_K_VALUES}

    signal_valid = k6["passed"]   # S5: withdraw signal from the headline on X-G
    rows = []
    for k in WS4E_K_VALUES:
        c = cells[k]
        share_lo, share_hi = disc_stats[k]["share_ci"]
        sep_lo, sep_hi = disc_stats[k]["separable_rate_ci"]
        null_lo, null_hi = disc_stats[k]["null_ci"]
        sig_lo, sig_hi = boot["ci_by_k"][k]
        rows.append({
            "k": k, **c,
            "share_ci_lo": share_lo, "share_ci_hi": share_hi,
            "separable_rate_ci_lo": sep_lo, "separable_rate_ci_hi": sep_hi,
            "null_ci_lo": null_lo, "null_ci_hi": null_hi,
            "signal": c["signal"] if signal_valid else None,
            "signal_ci_lo": sig_lo if signal_valid else None,
            "signal_ci_hi": sig_hi if signal_valid else None,
            "signal_withdrawn_reason": (
                None if signal_valid else
                "gate K6 failed (branch X-G): replication did not show "
                "within-prompt variance, so subtracting null(k) is not "
                "justified -- see gates.csv and replicate_cell.csv"),
            "inverted_fraction": inverted[k],
            "stratum": "primary", "arms": "|".join(WS4E_ARMS),
            "note": ("Per-question discrimination among eligible questions "
                     "(bucket decomposition). Compare against "
                     "slope_by_k.csv, which answers a DIFFERENT question -- "
                     "how strongly recall@10 predicts judge accuracy -- and "
                     "moves in the OPPOSITE direction with k on this data. "
                     "See also "
                     "composition_check.csv for a fixed-population "
                     "diagnostic of this statistic."),
        })
    pd.DataFrame(rows).to_csv(out / "discrimination_by_k.csv", index=False)
    manifest.record(out / "discrimination_by_k.csv")

    # --- single slope per k, x-axis FIXED at recall@10, WITH a CI ----------
    gt = np.load(gt_path(WS4_SCALE))["ws4_indices"]
    rec = recall_matrix(ids, gt, qids)          # (264, 4), same at every k
    x = rec.mean(0).tolist()

    slope_rows = []
    for k in WS4E_K_VALUES:
        acc = accuracy_matrix(all_verdicts[k], qids)      # (264, 4)
        fit = ws4d.fit_line(x, acc.mean(0).tolist())
        sboot = ws4d.bootstrap_slope(rec, acc, n_boot=WS4B_CHANGEPOINT_BOOT_N,
                                     seed=SEED)
        slope_rows.append({"k": k, "slope": round(fit["slope"], 6),
                           "slope_ci_lo": sboot["slope_ci_lo"],
                           "slope_ci_hi": sboot["slope_ci_hi"],
                           "intercept": round(fit["intercept"], 6),
                           "sse": round(fit["sse"], 8),
                           "n_arms": len(WS4E_ARMS),
                           "residual_df": len(WS4E_ARMS) - 2,
                           "x_axis": "recall_at_10",
                           "changepoint_fitted": False,
                           "note": ("How strongly recall@10 predicts judge "
                                    "accuracy ACROSS ARMS (an arm-level "
                                    "regression, n_arms=4 points). This is a "
                                    "DIFFERENT question from "
                                    "discrimination_by_k.csv's signal(k) "
                                    "(per-question discrimination among "
                                    "eligible questions) -- the two move in "
                                    "OPPOSITE directions with k here; read "
                                    "both before quoting either.")})
    pd.DataFrame(slope_rows).to_csv(out / "slope_by_k.csv", index=False)
    manifest.record(out / "slope_by_k.csv")

    # --- sensitivities: both are free subsets of the primary ---------------
    sens = []
    for name in ("plus_unanswerable", "disputed_included"):
        sq = strata[name]
        for k in WS4E_K_VALUES:
            d = ws4e.decompose(verdicts_for(k, sq),
                               ws4e.separable_at(ids, sq, k), sq)
            d.pop("buckets")
            sens.append({"stratum": name, "k": k, **d})
    pd.DataFrame(sens).to_csv(out / "sensitivities.csv", index=False)
    manifest.record(out / "sensitivities.csv")

    # --- CRITICAL 1: fixed-composition diagnostic (does NOT change signal) -
    fixed_sep = sorted(q for q in qids if separable[WS4E_K_VALUES[0]][q])
    fixed_ident = sorted(q for q in qids if not separable[WS4E_K_VALUES[-1]][q])
    fc_point = ws4e.fixed_composition_signal(buckets, fixed_sep, fixed_ident)
    fc_boot = ws4e.bootstrap_fixed_composition_signal(
        buckets, fixed_sep, fixed_ident, n_boot=WS4B_CHANGEPOINT_BOOT_N,
        seed=SEED)
    excess = ws4e.retrieval_attributable_excess(
        boot["signal_by_k"], {k: cells[k]["n_separable"] for k in WS4E_K_VALUES})

    # The rate can be flat on a fixed population while
    # its discriminating MEMBERSHIP still churns -- report the churn, not
    # just the rate, for both fixed populations.
    membership_98 = ws4e.fixed_population_membership_check(buckets, fixed_sep)
    membership_23 = ws4e.fixed_population_membership_check(buckets, fixed_ident)

    # Task 10 SHOULD FIX 7: two populations defined at k=10 (separable-at-10,
    # and the newcomers within it relative to the fixed-98), read back at
    # every k -- not just the k where each was defined. Reproduces the
    # 0.2365 / 0.0699 pair at k=1 that results/recall-vs-quality.md quotes.
    sep10 = sorted(q for q in qids if separable[WS4E_K_VALUES[-1]][q])
    newcomers10 = sorted(set(sep10) - set(fixed_sep))
    sep10_rate_by_k = ws4e.fixed_population_rate(buckets, sep10)
    newcomers10_rate_by_k = ws4e.fixed_population_rate(buckets, newcomers10)

    comp_rows = []
    for k in WS4E_K_VALUES:
        sep_k = [q for q in qids if separable[k][q]]
        newcomers = sorted(set(sep_k) - set(fixed_sep))
        b = buckets[k]
        newcomer_rate = (float(np.mean([b[q] == "discriminating" for q in newcomers]))
                         if newcomers else None)
        comp_rows.append({
            "k": k,
            "n_fixed_separable": len(fixed_sep),
            "n_fixed_identical": len(fixed_ident),
            "fixed_separable_rate": fc_point[k]["fixed_separable_rate"],
            "fixed_identical_rate": fc_point[k]["fixed_identical_rate"],
            "fixed_signal": fc_point[k]["fixed_signal"],
            "n_newcomers_vs_k1": len(newcomers),
            "newcomer_discriminating_rate": newcomer_rate,
            "committed_signal": boot["signal_by_k"][k] if signal_valid else None,
            "retrieval_attributable_excess": excess[k] if signal_valid else None,
            "fixed_signal_k_lo": fc_boot["k_lo"], "fixed_signal_k_hi": fc_boot["k_hi"],
            "fixed_signal_max_pair": fc_boot["max_pair"],
            "fixed_signal_max_pair_ci_lo": fc_boot["max_pair_ci_lo"],
            "fixed_signal_max_pair_ci_hi": fc_boot["max_pair_ci_hi"],
            "fixed_signal_max_pair_below_flat_min_diff":
                fc_boot["max_pair"] < WS4E_FLAT_MIN_DIFF,
            "n_fixed_disc": membership_98[k]["n_disc"],
            "n_fixed_disc_overlap_vs_k1": membership_98[k]["n_disc_overlap_vs_k1"],
            "n_fixed_identical_disc": membership_23[k]["n_disc"],
            "n_fixed_identical_disc_overlap_vs_k1":
                membership_23[k]["n_disc_overlap_vs_k1"],
            "sep_at_k10_rate_at_this_k": sep10_rate_by_k[k],
            "newcomers_at_k10_rate_at_this_k": newcomers10_rate_by_k[k],
            "note": ("DIAGNOSTIC ONLY: "
                     "separable(k) is nested/monotone in k, so the "
                     "committed signal(k)'s denominator changes composition "
                     "as k grows. fixed_separable_rate/fixed_identical_rate "
                     "hold the SAME 98/23 questions fixed at every k. This "
                     "does NOT change signal(k), the branch rule, or any "
                     "committed number. n_fixed_disc*/n_fixed_identical_disc* "
                     "report the fixed-98/fixed-23 "
                     "discriminating SET's overlap against k=1, not just its "
                     "rate: the rate can be identical while the set churns. "
                     "sep_at_k10_rate_at_this_k / newcomers_at_k10_rate_at_this_k "
                     "are the 241-separable-at-k=10 and "
                     "143-newcomer populations, each FIXED at their k=10 "
                     "definition and read back at every other k."),
        })
    pd.DataFrame(comp_rows).to_csv(out / "composition_check.csv", index=False)
    manifest.record(out / "composition_check.csv")

    # --- S7: Bonferroni-corrected CI over all six pairwise comparisons -----
    pairs = ws4e.bootstrap_all_pairs_signal(
        buckets, separable, qids, n_boot=WS4B_CHANGEPOINT_BOOT_N, seed=SEED)
    pd.DataFrame(pairs).to_csv(out / "signal_pairwise_bonferroni.csv",
                               index=False)
    manifest.record(out / "signal_pairwise_bonferroni.csv")

    # --- IMPORTANT 4: dedup-rule sensitivity (first-wins vs last-wins) -----
    # The fixed-composition diagnostic (composition_
    # check.csv) is ALSO not robust to the dedup rule -- extend the
    # sensitivity check to cover it, not just the per-k decomposition.
    # k=3 and k=10 have zero duplicate raw lines (module docstring), so
    # their first-wins buckets are identical to last-wins; only k=1 and k=5
    # (DEDUP_SENSITIVE_K) actually differ.
    buckets_by_rule = {"last_wins": buckets}
    verdicts_first_wins_by_k = {
        k: (verdicts_for(k, qids, first_wins=True) if k in DEDUP_SENSITIVE_K
            else all_verdicts[k])
        for k in WS4E_K_VALUES}
    buckets_first_wins = {}
    for k in WS4E_K_VALUES:
        d = ws4e.decompose(verdicts_first_wins_by_k[k], separable[k], qids)
        buckets_first_wins[k] = d.pop("buckets")
    buckets_by_rule["first_wins"] = buckets_first_wins

    fc_by_rule = {}
    for rule, b_by_k in buckets_by_rule.items():
        fc_point_r = ws4e.fixed_composition_signal(b_by_k, fixed_sep, fixed_ident)
        fc_boot_r = ws4e.bootstrap_fixed_composition_signal(
            b_by_k, fixed_sep, fixed_ident, n_boot=WS4B_CHANGEPOINT_BOOT_N,
            seed=SEED)
        fc_by_rule[rule] = (fc_point_r, fc_boot_r)

    dedup_rows = []
    for k in DEDUP_SENSITIVE_K:
        for rule, v in (("last_wins", all_verdicts[k]),
                        ("first_wins", verdicts_for(k, qids, first_wins=True))):
            d = ws4e.decompose(v, separable[k], qids)
            d.pop("buckets")
            fc_point_r, fc_boot_r = fc_by_rule[rule]
            dedup_rows.append({
                "k": k, "dedup_rule": rule, **d,
                "fixed_separable_rate": fc_point_r[k]["fixed_separable_rate"],
                "fixed_identical_rate": fc_point_r[k]["fixed_identical_rate"],
                "fixed_signal": fc_point_r[k]["fixed_signal"],
                "fixed_signal_k_lo": fc_boot_r["k_lo"],
                "fixed_signal_k_hi": fc_boot_r["k_hi"],
                "fixed_signal_max_pair": fc_boot_r["max_pair"],
                "fixed_signal_max_pair_ci_lo": fc_boot_r["max_pair_ci_lo"],
                "fixed_signal_max_pair_ci_hi": fc_boot_r["max_pair_ci_hi"],
                "fixed_signal_margin_vs_flat_min_diff":
                    round(WS4E_FLAT_MIN_DIFF / fc_boot_r["max_pair"], 4),
                "note": ("fixed_signal_* columns are the "
                         "SAME fixed-98/fixed-23 diagnostic as "
                         "composition_check.csv, recomputed under this row's "
                         "dedup_rule across all four depths (k=3/10 unchanged "
                         "since they have zero duplicate raw lines). Under "
                         "first_wins the fixed-98 rate at k=1 moves "
                         "0.4796->0.5000, the max pairwise difference moves "
                         "0.0204->0.0843, and the margin against "
                         "WS4E_FLAT_MIN_DIFF shrinks 4.90x->1.19x. The "
                         "conclusion (non-flat CI still includes zero) is "
                         "unchanged; the margin is not."),
            })
    pd.DataFrame(dedup_rows).to_csv(out / "dedup_sensitivity.csv", index=False)
    manifest.record(out / "dedup_sensitivity.csv")

    # --- Phase 1b replicate_cell.csv ----------------------------------------
    # 555 duplicate rows from two concurrency incidents (549 at k=1, 6 at
    # k=5). Supplementary corroboration ONLY; never gates anything.
    acc_disagree, acc_detail = accidental_replicate_flags(qids)
    r_acc = ws4e.bootstrap_rate(acc_disagree, n_boot=WS4B_CHANGEPOINT_BOOT_N,
                                seed=SEED)
    check_acc = ws4e.four_sample_consistency_check(r_acc["R"])

    replicate_rows = [
        {"cell": "replicate_sq8_np512_k1", "subset": "all",
         "pre_registered": True, "n": r_all["n"],
         "n_disagree": r_all["n_disagree"], "R": r_all["R"],
         "ci_lo": r_all["ci_lo"], "ci_hi": r_all["ci_hi"],
         "v1_passed": v1_passed, "v2_ci_overlap": v2_ci_overlap,
         "consistency_p": None if check is None else check["p"],
         "consistency_predicted_rate":
             None if check is None else check["predicted_four_sample_rate"],
         "consistency_compared_null_at_k1": cells[WS4E_REPLICATE_K]["null"],
         "note": "V1 test; gate K6 reads this row"},
        {"cell": "replicate_sq8_np512_k1", "subset": "identical_passage",
         "pre_registered": True, "n": r_ident["n"],
         "n_disagree": r_ident["n_disagree"], "R": r_ident["R"],
         "ci_lo": r_ident["ci_lo"], "ci_hi": r_ident["ci_hi"],
         "v1_passed": None, "v2_ci_overlap": v2_ci_overlap,
         "consistency_p": None, "consistency_predicted_rate": None,
         "consistency_compared_null_at_k1": None,
         "note": "V2: compare against the separable subset below"},
        {"cell": "replicate_sq8_np512_k1", "subset": "separable",
         "pre_registered": True, "n": r_sep["n"],
         "n_disagree": r_sep["n_disagree"], "R": r_sep["R"],
         "ci_lo": r_sep["ci_lo"], "ci_hi": r_sep["ci_hi"],
         "v1_passed": None, "v2_ci_overlap": v2_ci_overlap,
         "consistency_p": None, "consistency_predicted_rate": None,
         "consistency_compared_null_at_k1": None,
         "note": "V2: compare against the identical_passage subset above"},
        {"cell": "accidental_duplicate_rows", "subset": "all",
         "pre_registered": False, "n": r_acc["n"],
         "n_disagree": r_acc["n_disagree"], "R": r_acc["R"],
         "ci_lo": r_acc["ci_lo"], "ci_hi": r_acc["ci_hi"],
         "v1_passed": None, "v2_ci_overlap": None,
         "consistency_p": None if check_acc is None else check_acc["p"],
         "consistency_predicted_rate":
             None if check_acc is None
             else check_acc["predicted_four_sample_rate"],
         "consistency_compared_null_at_k1": None,
         "note": ("ACCIDENTAL -- 555 duplicate (qid, arm, k) rows from two "
                  "concurrency incidents (549 at k=1, 6 at k=5), spanning "
                  "all four arms. Supplementary corroboration ONLY, NOT "
                  "pre-registered, and NEVER promoted to primary or used "
                  "by any gate.")},
    ]
    pd.DataFrame(replicate_rows).to_csv(out / "replicate_cell.csv", index=False)
    manifest.record(out / "replicate_cell.csv")

    # --- Noise share at k=10, WS4e's own arms vs -------------------------
    # spec 2.3's better-powered but DIFFERENT arm set. results/recall-vs-quality.md's 30.7%
    # (15 arms) has no internal null control; spec 2.3's 12/57 does, but on
    # np4/np12/np48/np512, not WS4E_ARMS (np1/np4/np48/np512) -- np1 was
    # swapped in specifically to raise the k=1 separability ceiling (spec
    # 2.2), so the two arm sets are not interchangeable. Both committed here;
    # WS4e's own figure (on its own arms) is primary.
    n_ident_10 = cells[WS4_TOP_K]["n_identical"]
    null_10 = cells[WS4_TOP_K]["null"]
    n_disc_10 = cells[WS4_TOP_K]["discriminating"]
    n_noise_10 = round(null_10 * n_ident_10)

    ref_qids = sorted(qids)
    ref_ids = {a: np.load(RET / f"{a}.npz")["ids"] for a in SPEC23_REF_ARMS}
    ref_sep = ws4e.separable_at(ref_ids, ref_qids, WS4_TOP_K)
    ref_committed = load_committed_k10(ref_qids, arms=SPEC23_REF_ARMS)
    ref_pivot = ref_committed.pivot(index="qid", columns="arm",
                                    values="judge_correct").dropna()
    assert set(ref_pivot.index) == set(ref_qids), (
        "SPEC23_REF_ARMS committed k=10 data does not cover the full "
        "264-question primary stratum")
    ref_verdicts = {q: {a: bool(ref_pivot.loc[q, a]) for a in SPEC23_REF_ARMS}
                    for q in ref_pivot.index}
    ref_d = ws4e.decompose(ref_verdicts, ref_sep, ref_qids)
    ref_d.pop("buckets")
    ref_n_noise = round(ref_d["null"] * ref_d["n_identical"])

    noise_share_rows = [
        {"source": "ws4e_own_arms", "k": WS4_TOP_K, "arms": "|".join(WS4E_ARMS),
         "n": cells[WS4_TOP_K]["n"], "n_identical": n_ident_10, "null": null_10,
         "n_discriminating": n_disc_10,
         "n_noise_attributable": n_noise_10,
         "noise_share": round(n_noise_10 / n_disc_10, 6),
         "primary": True,
         "note": ("WS4e's own committed arm set (WS4E_ARMS) at k=10: "
                  "null(10) x n_identical(10) questions are discriminating "
                  "for a reason null(k) can attribute to generator variance "
                  "rather than retrieval. This is the figure "
                  "results/recall-vs-quality.md reports as primary.")},
        {"source": "spec_2_3_reference_arm_set", "k": WS4_TOP_K,
         "arms": "|".join(SPEC23_REF_ARMS),
         "n": ref_d["n"], "n_identical": ref_d["n_identical"], "null": ref_d["null"],
         "n_discriminating": ref_d["discriminating"],
         "n_noise_attributable": ref_n_noise,
         "noise_share": round(ref_n_noise / ref_d["discriminating"], 6),
         "primary": False,
         "note": ("Spec 2.3's reference figure, reproduced here at $0 from "
                  "committed data on a DIFFERENT, better-powered four-arm "
                  "set (np4/np12/np48/np512 -- NOT WS4E_ARMS, which swaps "
                  "np12 for np1). Quote only with that qualifier; it is not "
                  "interchangeable with the ws4e_own_arms row above.")},
    ]
    pd.DataFrame(noise_share_rows).to_csv(out / "noise_share_at_k10.csv",
                                          index=False)
    manifest.record(out / "noise_share_at_k10.csv")

    # --- gates.csv -----------------------------------------------------------
    k4 = ws4e.gate_k4(cells)
    k4["kind"] = "gate"
    k5 = ws4e.gate_k5(spent=ws4e_runner.tier_spent(), limit=WS4E_CAP_USD)
    k5["kind"] = "gate"
    k6["kind"] = "gate"
    p1_row = {"gate": "P1", "kind": "prediction", "passed": p1["holds"],
             "always_by_k": json.dumps(p1["always_by_k"])}
    p2_row = {"gate": "P2", "kind": "prediction", "passed": p2["holds"],
             "never_by_k": json.dumps(p2["never_by_k"])}
    vintage_row = {"gate": "generation_vintage_note", "kind": "note",
                  "passed": None, "detail": GENERATION_VINTAGE_NOTE}

    gates = prior_gates + [k3, k1_real, k4, k5, k6, p1_row, p2_row, vintage_row, {
        "gate": "flatness", "kind": "diagnostic",
        "passed": flat == "non_flat",
        "max_pair": boot["max_pair"],
        "ci_lo": boot["max_pair_ci_lo"], "ci_hi": boot["max_pair_ci_hi"],
        "k_lo": boot["k_lo"], "k_hi": boot["k_hi"],
        "min_diff": WS4E_FLAT_MIN_DIFF,
        "basis": f"{len(qids)} qids x {len(WS4E_ARMS)} arms; "
                 f"x-axis recall_at_10",
        "branch": branch, "phase": "phase2"}]
    pd.DataFrame(gates).to_csv(out / "gates.csv", index=False)
    manifest.record(out / "gates.csv")

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nmax pair signal diff k={boot['k_lo']} vs k={boot['k_hi']}: "
          f"{boot['max_pair']:.4f} "
          f"[{boot['max_pair_ci_lo']:.4f}, {boot['max_pair_ci_hi']:.4f}]")
    print(f"flatness: {flat}   K4: {k4['passed']} {k4['depths_at_noise']}   "
          f"K5: {k5['passed']} (${k5['spent']:.4f} of ${k5['limit']:.2f})   "
          f"K6: {k6['passed']} (R={r_all['R']}, "
          f"n_disagree={r_all['n_disagree']}, "
          f"CI=[{r_all['ci_lo']}, {r_all['ci_hi']}])")
    print(f"P1: {p1['holds']}   P2: {p2['holds']}")
    print(f"accidental (supplementary, NOT pre-registered): "
          f"R={r_acc['R']} n={r_acc['n']} "
          f"CI=[{r_acc['ci_lo']}, {r_acc['ci_hi']}]")
    print("\ncomposition_check.csv fixed-98/fixed-23 signal "
          f"(max pair k={fc_boot['k_lo']} vs k={fc_boot['k_hi']}): "
          f"{fc_boot['max_pair']:.4f} "
          f"[{fc_boot['max_pair_ci_lo']:.4f}, {fc_boot['max_pair_ci_hi']:.4f}] "
          f"-- below WS4E_FLAT_MIN_DIFF: "
          f"{fc_boot['max_pair'] < WS4E_FLAT_MIN_DIFF}")
    print(f"BRANCH: {branch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
