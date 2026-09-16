#!/usr/bin/env python
"""results/ws4e/conversion_composition_check.csv -- correcting a composition
artifact in `mediator_by_k.csv`'s conversion column and
`chart_mediator_by_depth.png`'s conversion panel.

`conversion(k) = P(judge correct | gold present at k)` conditions on the
presence set, and that set is NESTED in k (sq8_np512: 140 present at k=1,
subset of 213 at k=3, subset of 236 at k=5, subset of 264 at k=10) -- the
exact same Simpson's-paradox shape results/recall-vs-quality.md already
documents for `separable(k)`/`signal(k)`. Holding the presence set FIXED to
the questions already present at k=1 removes almost all of the apparent
conversion fall; what is real is that the newly-surfaced questions at each
depth convert far worse than the ones already present. $0 -- reads only
committed/checkpointed rows already on disk, same sources as
scripts/recall-quality/ws4e_mediator_by_k.py (loading logic duplicated rather than
imported, per that script's own note: scripts/ is deliberately not a
package).

Design: results/recall-vs-quality.md (the
mediation chain is spec 6.4's motivation; this diagnostic is new to the
composition-artifact fix and does not touch signal(k), the branch rule,
mediator_by_k.csv, or any other previously committed number).

Run:
    PYTHONPATH=. .venv/bin/python scripts/recall-quality/ws4e_conversion_composition_check.py
"""

import sys
from pathlib import Path

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
    k=10 -- identical to ws4e_mediator_by_k.py's own function (duplicated,
    not imported: scripts/ is deliberately not a package)."""
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
    """qid, arm, em, judge_correct, presence for one depth, all four
    WS4E_ARMS -- identical to ws4e_mediator_by_k.py's own function (no
    recall_at_10 needed here, unlike that script)."""
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
        frames.append(df[["qid", "arm", "em", "judge_correct", "presence"]])
    if k == WS4_TOP_K:
        frames.append(load_committed_k10(qids)[
            ["qid", "arm", "em", "judge_correct", "presence"]])
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

    cell_data = {}
    for (arm, k), g in all_rows.groupby(["arm", "k"]):
        g = g.set_index("qid")
        cell_data[(arm, k)] = {
            int(q): (bool(g.loc[q, "presence"]), bool(g.loc[q, "em"]),
                     bool(g.loc[q, "judge_correct"]))
            for q in qids}

    check = ws4e.conversion_composition_check(
        cell_data, qids, n_boot=WS4B_CHANGEPOINT_BOOT_N, seed=SEED)

    note = ("Second instance of the composition artifact "
            "results/recall-vs-quality.md documents for separable(k)/signal(k): conversion(k) "
            "= P(judge correct | present at k) conditions on presence(k), "
            "which is nested/monotone in k. fixed_conversion holds the "
            "presence set fixed at k=1 (per arm) and reads conversion back "
            "at every depth; n_newly_present/n_already_present split the "
            "MOVING present set at each k against the previous swept k. "
            "fixed_diff (k_hi - k_lo) and its CI are a paired bootstrap "
            "over the arm's fixed set only, one shared resample per "
            "replicate reused for both k's judge outcomes "
            "(WS4B_CHANGEPOINT_BOOT_N, seed=42), same posture as "
            "bootstrap_fixed_composition_signal in composition_check.csv. "
            "DIAGNOSTIC: does not change mediator_by_k.csv, signal(k), the "
            "branch rule, or any other previously committed number.")

    rows = []
    for arm in WS4E_ARMS:
        c = check[arm]
        for k in WS4E_K_VALUES:
            p = c["per_k"][k]
            rows.append({
                "arm": arm, "k": k, "n": len(qids),
                "n_present": p["n_present"],
                "moving_conversion": (None if p["moving_conversion"] is None
                                      else round(p["moving_conversion"], 6)),
                "n_fixed": p["n_fixed"],
                "fixed_conversion": (None if p["fixed_conversion"] is None
                                     else round(p["fixed_conversion"], 6)),
                "n_newly_present": p["n_newly_present"],
                "newly_present_conversion":
                    (None if p["newly_present_conversion"] is None
                     else round(p["newly_present_conversion"], 6)),
                "n_already_present": p["n_already_present"],
                "already_present_conversion":
                    (None if p["already_present_conversion"] is None
                     else round(p["already_present_conversion"], 6)),
                "k_lo": c["k_lo"], "k_hi": c["k_hi"],
                "fixed_diff_k_hi_minus_k_lo": c["fixed_diff"],
                "fixed_diff_ci_lo": c["fixed_diff_ci_lo"],
                "fixed_diff_ci_hi": c["fixed_diff_ci_hi"],
                "stratum": "primary", "arms": "|".join(WS4E_ARMS),
                "note": note,
            })
    df = pd.DataFrame(rows).sort_values(["arm", "k"]).reset_index(drop=True)
    df.to_csv(out_dir / "conversion_composition_check.csv", index=False)
    manifest.record(out_dir / "conversion_composition_check.csv")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
