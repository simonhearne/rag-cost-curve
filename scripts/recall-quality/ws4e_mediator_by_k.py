#!/usr/bin/env python
"""WS4e task 10(1) -- the mediation chain per (arm, k) over the 264-question
primary stratum: recall@10 -> presence@k -> exact match -> judge accuracy,
plus conversion = P(judge correct | gold present at k). Mirrors what
`results/ws4/chart_mediator.png` shows at k=10, across all four swept
depths. $0 -- reads only committed/checkpointed rows already on disk.

Design: results/recall-vs-quality.md
(the mediation chain is spec 6.4's motivation; the estimator here is new
to task 10 and does not touch signal(k), the branch rule, or any
previously committed number).

⚠️ Reads the SAME sources as scripts/recall-quality/ws4e_phase1_run.py /
scripts/recall-quality/ws4e_phase2_analyse.py: WS4e's own per-cell checkpoints under
data/ws4e_cache/ (gitignored) and, for k=10's three committed arms,
results/ws4/runs.csv and data/ws4d_cache/phase3_runs_checkpoint.jsonl.
scripts/ is deliberately not a package (see ws4e_phase2_analyse.py's own
note on this point), so the loading logic is duplicated rather than
imported.

Run:
    PYTHONPATH=. .venv/bin/python scripts/recall-quality/ws4e_mediator_by_k.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest  # noqa: E402
from benchlib.recall_quality import depth_runner as ws4e_runner  # noqa: E402
from benchlib.recall_quality import depth as ws4e  # noqa: E402
from benchlib.config import (SEED, WS4_TOP_K, WS4B_CHANGEPOINT_BOOT_N,  # noqa: E402
                             WS4E_ARMS, WS4E_K_VALUES, WS4E_TEMPERATURE,
                             ws4b_dir, ws4d_cache, ws4d_dir, ws4e_dir)
from benchlib.agent_harness import load_checkpoint  # noqa: E402

COMMITTED_K10_ARMS = ("sq8_np4", "sq8_np48", "sq8_np512")
PHASE3_CHECKPOINT = ws4d_cache() / "phase3_runs_checkpoint.jsonl"


def load_committed_k10(qids) -> pd.DataFrame:
    """qid, arm, em, judge_correct, presence for COMMITTED_K10_ARMS at
    k=10, from both sources those arms actually live in -- sq8_np48 lives
    ONLY in WS4d Phase 3's checkpoint, never in results/ws4/runs.csv (see
    ws4e_phase1_run.py's own comment on exactly this point)."""
    want = {int(q) for q in qids}
    cols = ["qid", "arm", "em", "judge_correct", "answer_presence_at_10"]
    frames = []
    runs = pd.read_csv("results/ws4/runs.csv")
    runs = runs[(runs.subset == "main") & runs.arm.isin(COMMITTED_K10_ARMS)]
    frames.append(runs[cols])
    if PHASE3_CHECKPOINT.exists():
        rows = [r for r in load_checkpoint(PHASE3_CHECKPOINT).values()
                if r.get("subset") == "main" and r["arm"] in COMMITTED_K10_ARMS]
        if rows:
            frames.append(pd.DataFrame(rows)[cols])
    committed = pd.concat(frames, ignore_index=True)
    committed["qid"] = committed["qid"].astype(int)
    committed = committed[committed.qid.isin(want)]
    dup = committed.duplicated(subset=["qid", "arm"]).sum()
    assert dup == 0, f"{dup} duplicate (qid, arm) rows in committed k=10 data"
    return committed.rename(columns={"answer_presence_at_10": "presence"})


def rows_for_k(k: int, qids) -> pd.DataFrame:
    """qid, arm, em, judge_correct, presence, recall_at_10 for one depth,
    all four WS4E_ARMS -- WS4e's own checkpointed cells plus (at k=10) the
    three committed arms."""
    want = {int(q) for q in qids}
    own_arms = tuple(a for a in WS4E_ARMS
                     if not (k == WS4_TOP_K and a in COMMITTED_K10_ARMS))
    checked = ws4e_runner.cell_rows(k, WS4E_TEMPERATURE)
    own = [r for (qid, arm), r in checked.items()
          if arm in own_arms and int(qid) in want]
    frames = []
    if own:
        df = pd.DataFrame(own)
        df["qid"] = df["qid"].astype(int)
        df = df.rename(columns={"answer_presence_at_k": "presence"})
        frames.append(df[["qid", "arm", "em", "judge_correct", "presence",
                          "recall_at_10"]])
    if k == WS4_TOP_K:
        committed = load_committed_k10(qids)
        # recall@10 is per-question and is the arm's fixed identity (spec
        # 6.4) -- not carried on results/ws4/runs.csv under a name that
        # survives the concat cleanly, so read it straight from a k<10 cell
        # of the SAME arm, keyed by (qid, arm), rather than duplicate WS2's
        # recall computation here. Every WS4E_ARMS arm has at least one
        # non-committed cell (k in {1,3,5}) to read it from.
        r10_by_qid_arm = {
            (int(qid), a): checked_r["recall_at_10"]
            for (qid, a), checked_r in
            ws4e_runner.cell_rows(WS4E_K_VALUES[0], WS4E_TEMPERATURE).items()
            if a in COMMITTED_K10_ARMS}
        committed = committed.copy()
        committed["recall_at_10"] = [
            r10_by_qid_arm[(int(q), a)]
            for q, a in zip(committed.qid, committed.arm)]
        frames.append(committed[["qid", "arm", "em", "judge_correct",
                                 "presence", "recall_at_10"]])
    out = pd.concat(frames, ignore_index=True)
    dup = out.duplicated(subset=["qid", "arm"]).sum()
    assert dup == 0, f"k={k}: {dup} duplicate (qid, arm) rows"
    missing = want - set(out.qid)
    assert not missing, f"k={k}: missing {len(missing)} question(s), e.g. {sorted(missing)[:3]}"
    return out


def main():
    out_dir = ws4e_dir()
    strata = ws4e.build_strata(
        pd.read_csv(ws4d_dir() / "eligibility_labels.csv"),
        pd.read_csv(ws4b_dir() / "tier0_runs_corrected.csv"))
    qids = strata["primary"]

    frames = [rows_for_k(k, qids).assign(k=k) for k in WS4E_K_VALUES]
    all_rows = pd.concat(frames, ignore_index=True)

    # {(arm, k): {qid: (presence, em, judge)}} for the bootstrap and point
    # estimates -- ws4e.mediator_point_estimates / bootstrap_mediator_stats.
    cell_data = {}
    recall_by_arm_k = {}
    for (arm, k), g in all_rows.groupby(["arm", "k"]):
        g = g.set_index("qid")
        cell_data[(arm, k)] = {
            int(q): (bool(g.loc[q, "presence"]), bool(g.loc[q, "em"]),
                     bool(g.loc[q, "judge_correct"]))
            for q in qids}
        recall_by_arm_k[(arm, k)] = float(g["recall_at_10"].mean())

    # recall@10 is the arm's fixed identity (spec 6.4): assert it really
    # does not move with k before trusting the chart's x-axis (rounded to
    # 1e-9 -- float summation order across the four cells' row order can
    # differ in its last bit or two without the underlying value differing).
    for arm in WS4E_ARMS:
        vals = {round(recall_by_arm_k[(arm, k)], 9) for k in WS4E_K_VALUES}
        assert len(vals) == 1, f"{arm}: recall@10 varies with k: {vals}"

    points = ws4e.mediator_point_estimates(cell_data, qids)
    boot = ws4e.bootstrap_mediator_stats(
        cell_data, qids, n_boot=WS4B_CHANGEPOINT_BOOT_N, seed=SEED)

    rows = []
    for k in WS4E_K_VALUES:
        for arm in WS4E_ARMS:
            c = (arm, k)
            p = points[c]
            b = boot[c]
            pres_lo, pres_hi = b["presence_ci"]
            em_lo, em_hi = b["em_ci"]
            judge_lo, judge_hi = b["judge_accuracy_ci"]
            conv_lo, conv_hi = b["conversion_ci"]
            rows.append({
                "arm": arm, "k": k, "n": p["n"],
                "recall_at_10": round(recall_by_arm_k[c], 6),
                "presence_at_k": round(p["presence_at_k"], 6),
                "presence_at_k_ci_lo": pres_lo, "presence_at_k_ci_hi": pres_hi,
                "em": round(p["em"], 6),
                "em_ci_lo": em_lo, "em_ci_hi": em_hi,
                "judge_accuracy": round(p["judge_accuracy"], 6),
                "judge_accuracy_ci_lo": judge_lo,
                "judge_accuracy_ci_hi": judge_hi,
                "conversion": (None if p["conversion"] is None
                              else round(p["conversion"], 6)),
                "conversion_ci_lo": conv_lo, "conversion_ci_hi": conv_hi,
                "stratum": "primary", "arms": "|".join(WS4E_ARMS),
                "note": ("Mediation chain (recall@10 -> presence@k -> EM -> "
                         "judge accuracy) per (arm, k) over the 264-question "
                         "primary stratum, mirroring "
                         "results/ws4/chart_mediator.png's k=10 slice across "
                         "every swept depth. conversion = P(judge correct | "
                         "gold present at k). CIs are paired-bootstrap over "
                         "questions, one shared question-index draw per "
                         "replicate (WS4B_CHANGEPOINT_BOOT_N, seed=42), the "
                         "same posture as discrimination_by_k.csv."),
            })
    df = pd.DataFrame(rows).sort_values(["arm", "k"]).reset_index(drop=True)
    df.to_csv(out_dir / "mediator_by_k.csv", index=False)
    manifest.record(out_dir / "mediator_by_k.csv")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
