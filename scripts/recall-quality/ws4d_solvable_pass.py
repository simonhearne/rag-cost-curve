#!/usr/bin/env python
"""WS4d supplementary pass -- exclude questions no arm ever answered correctly.

Requested by Simon 2026-09-13, after the 15-arm refit. Stored SEPARATELY from the
pre-registered Phase 2/3 outputs: every file this writes is prefixed `solvable_`
and nothing it produces overwrites a pre-registered artifact.

⚠️ THIS FILTER CONDITIONS ON THE OUTCOME, and that is a different kind of filter
from the one WS4d pre-registered. Phase 1's eligibility labels are computed from
(question, golds, ground-truth top-10) and never see any arm's answer -- which is
why they can filter a primary. "Never answered correctly by any arm" is defined BY
the arms' answers. It is reported here as an exploratory cut, never as a primary,
and results/recall-vs-quality.md must say so.

There is a second, more mechanical reason it cannot rescue the analysis. If every
arm is wrong on a question, dropping it removes ZERO correct answers and exactly
ONE wrong answer from every arm. So each arm's accuracy goes from correct/n to
correct/(n-k): a UNIFORM multiplicative rescale by n/(n-k), identical for all
arms. A positive rescale of y leaves argmin_tau SSE(tau) unchanged, so the fitted
changepoint cannot move. The script asserts this rather than asserting it in prose.

Run:  python scripts/recall-quality/ws4d_solvable_pass.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.recall_quality import eligibility as ws4d
from benchlib.config import (SEED, WS4B_CHANGEPOINT_BOOT_N, ws4b_dir,
                             ws4d_cache, ws4d_dir)

NEW_ARMS = {"sq8_np6", "sq8_np12", "sq8_np24", "sq8_np48"}


def load_merged():
    """The 15-arm ladder on Phase 1's eligibility-filtered primary stratum."""
    new = pd.DataFrame([json.loads(l) for l
                        in open(ws4d_cache() / "phase3_runs_checkpoint.jsonl")
                        if l.strip()])
    new["qid"] = new.qid.astype(int)
    new["idk_corrected"] = new["idk"]
    cols = ["qid", "arm", "recall_at_10", "judge_correct", "idk_corrected"]
    old = pd.read_csv(ws4b_dir() / "tier0_runs_corrected.csv")
    old = old[old.in_primary & (old.arm != "parametric")][cols]
    labels = pd.read_csv(ws4d_dir() / "eligibility_labels.csv")
    drop = set(labels[labels["class"].isin(ws4d.PRIMARY_DROP)].qid)
    m = pd.concat([old, new[cols]])
    return m[m.qid.isin(set(old.qid) - drop)]


def fit(m, name):
    piv = m.pivot_table(index="qid", columns="arm",
                        values=["recall_at_10", "judge_correct"])
    rec, acc = piv["recall_at_10"], piv["judge_correct"]
    boot = ws4d.bootstrap_changepoint(rec.values, acc.values,
                                      n_boot=WS4B_CHANGEPOINT_BOOT_N, seed=SEED)
    x = rec.mean().values
    one, two = ws4d.fit_line(x, acc.mean().values), ws4d.fit_changepoint(x, acc.mean().values)
    return {"stratum": name, "n_questions": len(rec), "n_arms": rec.shape[1],
            **boot, "sse_one_segment": round(one["sse"], 8),
            "sse_two_segment": round(two["sse"], 8),
            "branch": ws4d.knee_branch_n(boot, x)}, rec, acc


def main():
    out = ws4d_dir()
    m = load_merged()
    byq = m.groupby("qid").judge_correct.agg(["sum", "count"])
    n_arms = int(byq["count"].iloc[0])
    never = set(byq[byq["sum"] == 0].index)
    always = set(byq[byq["sum"] == n_arms].index)
    n_all = len(byq)

    # ---- how much of the stratum can discriminate between arms at all?
    disc = pd.DataFrame([{
        "n_questions": n_all, "n_arms": n_arms,
        "n_never_correct": len(never), "n_always_correct": len(always),
        "n_discriminating": n_all - len(never) - len(always),
        "share_discriminating": round((n_all - len(never) - len(always)) / n_all, 6),
        "predicted_rescale": round(n_all / (n_all - len(never)), 6),
    }])
    disc.to_csv(out / "solvable_discrimination.csv", index=False)

    full, _, _ = fit(m, "all_eligible")
    solv, rec, acc = fit(m[~m.qid.isin(never)], "solvable_only")
    # ⚠️ DIAGNOSTIC ONLY, and MORE outcome-conditioned than solvable_only: this
    # drops the always-correct questions too, so both tails are defined by the
    # arms' own answers. It exists to answer one mechanistic question -- is the
    # flat curve a dilution effect? -- and must never be quoted as a result.
    disc_fit, _, _ = fit(m[~m.qid.isin(never | always)], "discriminating_only")

    # ---- the rescale property, asserted not assumed
    piv_f = m.pivot_table(index="qid", columns="arm", values="judge_correct")
    piv_s = m[~m.qid.isin(never)].pivot_table(index="qid", columns="arm",
                                              values="judge_correct")
    ratio = (piv_s.mean() / piv_f.mean()).round(9)
    uniform = bool(ratio.nunique() == 1)
    print(f"uniform rescale across all {n_arms} arms: {uniform}  "
          f"(factor {ratio.iloc[0]:.6f}, predicted {n_all/(n_all-len(never)):.6f})")
    print(f"tau unchanged: {full['tau'] == solv['tau']}  "
          f"({full['tau']:.6f} -> {solv['tau']:.6f})")

    pd.DataFrame([full, solv, disc_fit]).to_csv(out / "solvable_changepoint.csv",
                                                index=False)

    rows = []
    for a, g in m[~m.qid.isin(never)].groupby("arm"):
        w = g[~g.judge_correct]
        rows.append({"arm": a, "is_new": a in NEW_ARMS, "n": len(g),
                     "recall_at_10": round(g.recall_at_10.mean(), 6),
                     "judge_correct": round(g.judge_correct.mean(), 6),
                     "declined": round(w.idk_corrected.sum() / len(g), 6),
                     "error": round((len(w) - w.idk_corrected.sum()) / len(g), 6),
                     "n_correct": int(g.judge_correct.sum()),
                     "n_declined": int(w.idk_corrected.sum()),
                     "n_error": int(len(w) - w.idk_corrected.sum())})
    pts = pd.DataFrame(rows).sort_values("recall_at_10", ascending=False)
    pts.to_csv(out / "solvable_points.csv", index=False)

    for f in ("solvable_discrimination.csv", "solvable_changepoint.csv",
              "solvable_points.csv"):
        manifest.record(out / f)

    print(f"\n{n_all} eligible -> {len(never)} never correct, {len(always)} always "
          f"correct, {n_all-len(never)-len(always)} discriminating "
          f"({100*(n_all-len(never)-len(always))/n_all:.1f}%)")
    print(f"all_eligible : n={full['n_questions']} tau={full['tau']:.4f} "
          f"width={full['width']:.4f} slope={full['slope']:.4f} {full['branch']}")
    print(f"solvable_only: n={solv['n_questions']} tau={solv['tau']:.4f} "
          f"width={solv['width']:.4f} slope={solv['slope']:.4f} {solv['branch']}")
    print(f"discriminating_only (DIAGNOSTIC): n={disc_fit['n_questions']} "
          f"tau={disc_fit['tau']:.4f} width={disc_fit['width']:.4f} "
          f"slope={disc_fit['slope']:.4f} b1={disc_fit['b1']:+.3f} "
          f"b2={disc_fit['b2']:+.3f} {disc_fit['branch']}")
    print(f"wrote 3 CSVs into {out}/ (all prefixed solvable_)")


if __name__ == "__main__":
    main()
