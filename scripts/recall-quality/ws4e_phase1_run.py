#!/usr/bin/env python
"""WS4e Phase 1 + Phase 1b -- the depth sweep, the replicate validation
cell (Amendment 1), and gate K3.

Spends against WS4E_CAP_USD, a budget line SEPARATE from WS4's $75:
ws4_runner.governor_from_checkpoints() must NOT be used here.

Held fixed per spec 3.4: WS4_GEN_MODEL, WS4_JUDGE_MODEL, WS4_GEN_MAX_TOKENS,
WS4's own system/judge prompts, and temperature (WS4E_TEMPERATURE = None
throughout, including the replicate cell -- see spec 3.5 Amendment 1). Only
the retrieval DEPTH changes; the one Phase 1b cell instead re-runs the SAME
(arm, k, temperature) a second, independent time.

Design: results/recall-vs-quality.md

Run:
    PYTHONPATH=. .venv/bin/python scripts/recall-quality/ws4e_phase1_run.py --limit 8  # ~5c
    PYTHONPATH=. .venv/bin/python scripts/recall-quality/ws4e_phase1_run.py            # ~$23
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest  # noqa: E402
from benchlib.code_retrieval import bench as ws6a  # noqa: E402
from benchlib.recall_quality import depth_runner as ws4e_runner  # noqa: E402
from benchlib.recall_quality import depth as ws4e  # noqa: E402
from benchlib.recall_quality import runner as ws4_runner  # noqa: E402
from benchlib.config import (SEED, WS4_SCALE, WS4_TOP_K, WS4E_ARMS,  # noqa: E402
                             WS4E_K_VALUES, WS4E_REJUDGE_MIN, WS4E_REJUDGE_N,
                             WS4E_REPLICATE_ARM, WS4E_REPLICATE_K,
                             WS4E_TEMPERATURE, gt_path,
                             ws4_cache, ws4b_dir, ws4d_cache, ws4d_dir,
                             ws4e_dir)
from benchlib.quantization import recall_at_k  # noqa: E402
from benchlib.agent_harness import load_checkpoint  # noqa: E402

RET = ws4_cache() / "retrieval"
COMMITTED_K10_ARMS = ("sq8_np4", "sq8_np48", "sq8_np512")

# Where each COMMITTED_K10_ARMS arm's committed k=10 rows actually live.
# sq8_np4 and sq8_np512 are WS4's own generations, in results/ws4/runs.csv.
# sq8_np48 is NOT there -- it was generated in WS4d Phase 3 and lives only in
# that phase's checkpoint. ws4e_phase0_scope.py's own committed_ids() reads
# from these same two sources for exactly this reason; a version of this
# script that read results/ws4/runs.csv alone would silently score sq8_np48
# as having zero k=10 rows instead of raising -- caught by cross-checking
# against ws4e_phase0_scope.py before this script was run for real.
PHASE3_CHECKPOINT = ws4d_cache() / "phase3_runs_checkpoint.jsonl"


def cells():
    """(k, temperature, arms, replicate) for every cell this phase runs.

    The three arms with committed k=10 rows are deliberately NOT re-run:
    reusing them is what keeps WS4e's k=10 point comparable to
    results/recall-vs-quality.md (spec 3.5).

    The last cell is the Amendment 1 replicate (spec 5.3): WS4E_REPLICATE_ARM
    re-run at WS4E_REPLICATE_K, at the SAME WS4E_TEMPERATURE as every other
    cell -- it is `replicate=True` alone that keeps its checkpoint from
    colliding with the primary k=1 cell's (qid, arm) rows.
    """
    out = []
    for k in WS4E_K_VALUES:
        arms = (tuple(a for a in WS4E_ARMS if a not in COMMITTED_K10_ARMS)
                if k == WS4_TOP_K else WS4E_ARMS)
        if arms:
            out.append((k, WS4E_TEMPERATURE, arms, False))
    out.append((WS4E_REPLICATE_K, WS4E_TEMPERATURE, (WS4E_REPLICATE_ARM,), True))
    return out


def build_inputs(qids):
    qdf = pd.read_csv("results/ws4/questions.csv")
    qdf = qdf[qdf.subset == "main"]
    qdf = qdf[qdf.row.astype(int).isin(set(qids))].sort_values("row")
    assert len(qdf) == len(qids), f"{len(qdf)} questions for {len(qids)} qids"
    gt = np.load(gt_path(WS4_SCALE))["ws4_indices"]
    questions = {int(r.row): dict(question=r.question, golds=json.loads(r.golds),
                                  subset=r.subset, recall={})
                 for r in qdf.itertuples()}
    retrieval, needed = {}, set()
    for arm in WS4E_ARMS:
        ids = np.load(RET / f"{arm}.npz")["ids"]
        assert ids.shape[1] == WS4_TOP_K, ids.shape
        retrieval[arm] = ids
        for q in qids:
            # recall@10 is the arm's FIXED identity -- the x-axis must not
            # move with the treatment (spec 6.4).
            questions[q]["recall"][arm] = recall_at_k(
                ids[q:q + 1], gt[q:q + 1], WS4_TOP_K)
            needed.update(int(i) for i in ids[q] if i >= 0)
    passages = ws4_runner.load_passages(sorted(needed), WS4_SCALE)
    return questions, retrieval, passages


def gate_k3(client, judge_p, prices, gov, log):
    """K3: re-judge WS4E_REJUDGE_N committed WS4 rows under WS4e's OWN
    checkpoint, so the judge is actually called again now."""
    rows = [r for r in load_checkpoint(ws4_runner.CHECKPOINT).values()
            if r["subset"] == "main"]
    if len(rows) < WS4E_REJUDGE_N:
        sys.exit(f"only {len(rows)} WS4 main rows checkpointed; need "
                 f"{WS4E_REJUDGE_N} for gate K3")
    rows.sort(key=lambda r: (r["arm"], r["qid"]))
    sample = random.Random(f"{SEED}:ws4e_k3").sample(rows, WS4E_REJUDGE_N)
    out = ws4_runner.rejudge_sample(
        client, sample, judge_p, prices, gov,
        checkpoint=ws4e_runner.rejudge_checkpoint(), log=log)
    n_ok = sum(1 for r in out.values()
               if r["first_verdict"] == r["second_verdict"])
    return ws4e.gate_k3(n=len(out), n_reproduced=n_ok,
                        threshold=WS4E_REJUDGE_MIN)


def load_committed_k10(qids):
    """Committed k=10 rows for COMMITTED_K10_ARMS, from BOTH sources those
    arms actually live in (see PHASE3_CHECKPOINT comment above). Reading
    only results/ws4/runs.csv -- as an earlier draft of this script did --
    silently drops sq8_np48 entirely, because it was never a WS4 arm; it
    only exists in WS4d's Phase 3 checkpoint."""
    frames = []
    runs = pd.read_csv("results/ws4/runs.csv")
    runs = runs[(runs.subset == "main") & runs.arm.isin(COMMITTED_K10_ARMS)]
    frames.append(runs)
    if PHASE3_CHECKPOINT.exists():
        rows = [r for r in load_checkpoint(PHASE3_CHECKPOINT).values()
                if r.get("subset") == "main" and r["arm"] in COMMITTED_K10_ARMS]
        if rows:
            frames.append(pd.DataFrame(rows))
    committed = pd.concat(frames, ignore_index=True)
    committed["qid"] = committed["qid"].astype(int)
    committed = committed[committed.qid.isin(set(qids))]
    # A checkpoint can carry more than one line for the same (qid, arm) if a
    # run was ever resumed; load_checkpoint already dedupes that within the
    # phase3 file, but the concat with runs.csv cannot introduce duplicates
    # since the two sources cover disjoint arms -- still assert it.
    dup = committed.duplicated(subset=["qid", "arm"]).sum()
    assert dup == 0, f"{dup} duplicate (qid, arm) rows in committed k=10 data"
    missing = {a: len(qids) - (committed.arm == a).sum()
               for a in COMMITTED_K10_ARMS}
    assert all(v == 0 for v in missing.values()), (
        f"committed k=10 coverage incomplete: {missing}")
    committed = committed.copy()
    committed["k"] = WS4_TOP_K
    committed["temperature"] = None
    committed["answer_presence_at_k"] = committed["answer_presence_at_10"]
    return committed


def write_depth_points(out_dir, qids):
    """Per (arm, k, temperature) aggregate. No answers, no corpus text.

    ⚠️ The replicate cell (spec 5.3) is excluded here: it is never a source
    of any headline number and no point from it enters this table -- it
    validates the estimator in Phase 2, on its own checkpoint.
    """
    frames = []
    for k, temp, _, replicate in cells():
        if replicate:
            continue
        rows = list(ws4e_runner.cell_rows(k, temp).values())
        if rows:
            frames.append(pd.DataFrame(rows))
    frames.append(load_committed_k10(qids))
    df = pd.concat(frames, ignore_index=True)
    df["qid"] = df["qid"].astype(int)
    df = df[df.qid.isin(set(qids))]
    agg = (df.groupby(["arm", "k", df.temperature.isna()], dropna=False)
             .agg(n_questions=("qid", "nunique"),
                  recall_at_10=("recall_at_10", "mean"),
                  presence_at_k=("answer_presence_at_k", "mean"),
                  accuracy=("judge_correct", "mean"),
                  declined=("idk", "mean"),
                  cost_usd=("cost_total_billed", "sum"))
             .reset_index()
             .rename(columns={"temperature": "temperature_is_default"}))
    agg.to_csv(out_dir / "depth_points.csv", index=False)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="truncate EACH cell for a smoke run; skips gate K3")
    args = ap.parse_args()

    out = ws4e_dir()
    out.mkdir(parents=True, exist_ok=True)

    if not (out / "ceiling.csv").exists():
        sys.exit("run scripts/recall-quality/ws4e_phase0_scope.py first -- ceiling.csv must "
                 "be pre-registered before any row is generated")

    strata = ws4e.build_strata(
        pd.read_csv(ws4d_dir() / "eligibility_labels.csv"),
        pd.read_csv(ws4b_dir() / "tier0_runs_corrected.csv"))
    qids = strata["primary"]
    questions, retrieval, passages = build_inputs(qids)

    sys_p, judge_p = ws4_runner.load_prompts()
    prices = ws6a.load_pricing(Path("results/ws6a/pricing.csv"))
    gov = ws4e_runner.tier_governor()
    client = ws4_runner.make_client()
    print(f"governor ${gov.spent:.4f} of ${gov.limit:.2f}", flush=True)

    for k, temp, arms, replicate in cells():
        pairs = [(q, a) for a in arms for q in qids]
        # Amendment 1 (spec 3.5) added the trailing `:{replicate}` component
        # so the replicate cell's shuffle cannot collide with the primary
        # k=1 cell's -- but it also means every cell's shuffle order changed
        # from what it was before Amendment 1 (the seed string used to be
        # f"{SEED}:ws4e:{k}:{temp}" with no replicate component). A
        # `--limit` smoke run therefore no longer samples the same
        # question/arm prefix it would have pre-Amendment-1; this is fine
        # for reproducibility (the full run is unaffected and deterministic
        # under the new seed) but worth knowing before comparing a smoke's
        # sampled pairs against a pre-Amendment-1 run.
        random.Random(f"{SEED}:ws4e:{k}:{temp}:{replicate}").shuffle(pairs)
        if args.limit is not None:
            pairs = pairs[:args.limit]
        label = f"k={k} temp={'default' if temp is None else temp} arms={len(arms)}"
        if replicate:
            label += " [REPLICATE]"
        print(f"\n=== cell {label}: {len(pairs)} pair(s) ===", flush=True)
        try:
            ws4_runner.run_pairs(
                client, pairs, questions=questions, retrieval=retrieval,
                passages=passages, system_prompt=sys_p, judge_prompt=judge_p,
                prices=prices, governor=gov,
                checkpoint=ws4e_runner.cell_checkpoint(k, temp, replicate=replicate),
                log=lambda m: print(m, flush=True),
                top_k=k, temperature=temp)
        except ws6a.BudgetExceeded as exc:
            print(f"GOVERNOR STOP: {exc}", flush=True)
            return 2

    print(f"\ngeneration done; governor ${gov.spent:.4f} of ${gov.limit:.2f}",
          flush=True)

    agg = write_depth_points(out, qids)
    manifest.record(out / "depth_points.csv")
    print(agg.to_string(index=False))

    if args.limit is not None:
        print("smoke run (--limit): skipping gate K3")
        return 0

    k3 = gate_k3(client, judge_p, prices, gov, lambda m: print(m, flush=True))
    print(json.dumps(k3, indent=2))
    (out / "k3_rejudge.json").write_text(json.dumps(k3, indent=2))
    if not k3["passed"]:
        sys.exit("GATE K3 FAILURE -- the judge has drifted; the new rows are "
                 "not comparable to the committed ones")
    print(f"Phase 1 OK; total spend ${gov.spent:.4f} of ${gov.limit:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
