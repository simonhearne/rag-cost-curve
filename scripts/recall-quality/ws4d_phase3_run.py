#!/usr/bin/env python
"""WS4d Phase 3b/3c -- generate + judge the four new ladder arms, then gate L2
(has the judge drifted since WS4 ran?).

Spends against WS4D_PHASE3_CAP_USD, a budget line SEPARATE from WS4's own
$75 (benchlib.recall_quality.runner.governor_from_checkpoints() must NOT be used here:
it would read WS4's own checkpoints and conflate two budget lines). This
script builds its own CostGovernor, seeded from THIS task's own checkpoint,
recovered the way benchlib.recall_quality.eligibility_runner.tier_spent recovers Phase 1's spend --
summed over every raw checkpoint line, not deduped by (qid, arm).

Held fixed, per WS4b spec 4.4's "what does not change": WS4_GEN_MODEL,
WS4_JUDGE_MODEL, WS4_TOP_K, and WS4's own system/judge prompts via
ws4_runner.load_prompts(). Only the retrieval arm changes.

Design: results/recall-vs-quality.md

Run:
    .venv/bin/python scripts/recall-quality/ws4d_phase3_run.py --limit 8   # smoke, a few cents
    .venv/bin/python scripts/recall-quality/ws4d_phase3_run.py              # full, ~$17
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
from benchlib.recall_quality import runner as ws4_runner  # noqa: E402
from benchlib.recall_quality import eligibility_runner as ws4d_runner  # noqa: E402
from benchlib.config import (SEED, WS4_JUDGE_RELIABILITY_N, WS4_SCALE,  # noqa: E402
                             WS4_TOP_K, WS4D_PHASE3_CAP_USD, gt_path,
                             ws4_cache, ws4d_cache, ws4d_dir)
from benchlib.quantization import recall_at_k  # noqa: E402
from benchlib.agent_harness import load_checkpoint  # noqa: E402

NEW_ARMS = ("sq8_np6", "sq8_np12", "sq8_np24", "sq8_np48")
RET = ws4_cache() / "retrieval"
RUN_CHECKPOINT = ws4d_cache() / "phase3_runs_checkpoint.jsonl"
L2_CHECKPOINT = ws4d_cache() / "phase3_l2_rejudge_checkpoint.jsonl"


def load_all_450():
    """The WS4 `main` questions -- the same 450 the eleven existing arms and
    Phase 1's eligibility labels were drawn over. `row` in questions.csv is
    the same id space as `qid` everywhere downstream."""
    qdf = pd.read_csv(Path("results/ws4/questions.csv"))
    qdf = qdf[qdf.subset == "main"].sort_values("row").reset_index(drop=True)
    assert len(qdf) == 450, f"expected 450 main questions, got {len(qdf)}"
    return qdf


def build_inputs(qdf):
    ws4_gt = np.load(gt_path(WS4_SCALE))["ws4_indices"]
    questions = {int(r.row): dict(question=r.question, golds=json.loads(r.golds),
                                  subset=r.subset, recall={})
                 for r in qdf.itertuples()}
    retrieval, needed = {}, set()
    for arm in NEW_ARMS:
        f = RET / f"{arm}.npz"
        if not f.exists():
            sys.exit(f"missing retrieval cache for {arm}: {f} "
                     f"(Task 11's sweep has not produced it)")
        ids = np.load(f)["ids"]
        assert ids.shape == (ws4_gt.shape[0], WS4_TOP_K), ids.shape
        retrieval[arm] = ids
        for r in qdf["row"]:
            r = int(r)
            questions[r]["recall"][arm] = recall_at_k(
                ids[r:r + 1], ws4_gt[r:r + 1], WS4_TOP_K)
            needed.update(int(i) for i in ids[r] if i >= 0)
    passages = ws4_runner.load_passages(sorted(needed), WS4_SCALE)
    return questions, retrieval, passages


def gate_l2(client, judge_p, prices, gov, log):
    """Spec 5.3 L2: re-judge WS4_JUDGE_RELIABILITY_N (60) WS4 rows a SECOND
    time, under a Phase-3-specific checkpoint so the judge is actually
    called again now (not served from WS4's own already-cached reliability
    checkpoint, which would prove nothing about drift since WS4 ran)."""
    done = load_checkpoint(ws4_runner.CHECKPOINT)
    rows = [r for r in done.values() if r["subset"] == "main"]
    if len(rows) < WS4_JUDGE_RELIABILITY_N:
        sys.exit(f"only {len(rows)} WS4 main rows checkpointed; need "
                 f"{WS4_JUDGE_RELIABILITY_N} for gate L2")
    rows.sort(key=lambda r: (r["arm"], r["qid"]))
    sample = random.Random(f"{SEED}:ws4d_phase3_l2").sample(
        rows, WS4_JUDGE_RELIABILITY_N)
    out = ws4_runner.rejudge_sample(client, sample, judge_p, prices, gov,
                                    checkpoint=L2_CHECKPOINT, log=log)
    n_reproduced = sum(1 for r in out.values()
                       if r["first_verdict"] == r["second_verdict"])
    return {"gate": "L2", "n": len(out), "n_reproduced": n_reproduced,
            "threshold": WS4_JUDGE_RELIABILITY_N,
            "passed": bool(n_reproduced == WS4_JUDGE_RELIABILITY_N)}


def write_ladder_runs(out_dir):
    """Aggregate per-arm summary of what THIS phase generated -- no raw
    generations, no corpus text, no answer text. One row per new arm."""
    done = load_checkpoint(RUN_CHECKPOINT)
    df = pd.DataFrame(done.values())
    if df.empty:
        return df
    df["qid"] = df["qid"].astype(int)
    agg = (df.groupby("arm")
             .agg(n_questions=("qid", "nunique"),
                  n_rows=("qid", "size"),
                  mean_recall_at_10=("recall_at_10", "mean"),
                  mean_judge_correct=("judge_correct", "mean"),
                  mean_idk=("idk", "mean"),
                  mean_em=("em", "mean"),
                  mean_f1=("f1", "mean"),
                  cost_total_usd=("cost_total_billed", "sum"))
             .reset_index().sort_values("mean_recall_at_10"))
    agg.to_csv(out_dir / "ladder_runs.csv", index=False)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="truncate the pair list for a smoke run; skips gate L2")
    args = ap.parse_args()

    out = ws4d_dir()
    out.mkdir(parents=True, exist_ok=True)

    qdf = load_all_450()
    questions, retrieval, passages = build_inputs(qdf)

    pairs = [(int(q), arm) for arm in NEW_ARMS for q in qdf["row"]]
    if args.limit is not None:
        pairs = pairs[:args.limit]

    sys_p, judge_p = ws4_runner.load_prompts()
    prices = ws6a.load_pricing(Path("results/ws6a/pricing.csv"))

    spent = ws4d_runner.tier_spent(path=RUN_CHECKPOINT)
    gov = ws6a.CostGovernor(WS4D_PHASE3_CAP_USD, spent=spent)
    client = ws4_runner.make_client()
    print(f"{len(pairs)} pair(s) requested; governor ${gov.spent:.4f} of "
         f"${gov.limit:.2f}", flush=True)

    try:
        ws4_runner.run_pairs(client, pairs, questions=questions,
                             retrieval=retrieval, passages=passages,
                             system_prompt=sys_p, judge_prompt=judge_p,
                             prices=prices, governor=gov,
                             checkpoint=RUN_CHECKPOINT,
                             log=lambda m: print(m, flush=True))
    except ws6a.BudgetExceeded as exc:
        print(f"GOVERNOR STOP: {exc}", flush=True)
        return 2

    print(f"generation done; governor ${gov.spent:.4f} of ${gov.limit:.2f}",
         flush=True)

    agg = write_ladder_runs(out)
    if not agg.empty:
        manifest.record(out / "ladder_runs.csv")
        print(agg.to_string(index=False))

    if args.limit is not None:
        print("smoke run (--limit): skipping gate L2, run the full pass first")
        return 0

    n_done = sum(1 for _ in RUN_CHECKPOINT.open()) if RUN_CHECKPOINT.exists() else 0
    if n_done < len(NEW_ARMS) * 450:
        print(f"only {n_done} of {len(NEW_ARMS) * 450} rows checkpointed -- "
             f"not running gate L2 or writing ladder_gates.csv yet")
        return 0

    l2 = gate_l2(client, judge_p, prices, gov, log=lambda m: print(m, flush=True))
    pd.DataFrame([l2]).to_csv(out / "ladder_gates.csv", index=False)
    manifest.record(out / "ladder_gates.csv")
    print(f"\nL2: {l2}")
    print(f"final governor spend: ${gov.spent:.4f} of ${gov.limit:.2f}")
    if not l2["passed"]:
        print("\nGATE L2 FAILED: the judge has drifted since WS4 ran. STOP -- "
             "the four new arms are not comparable to the existing eleven. "
             "Do not run scripts/recall-quality/ws4d_phase3_refit.py.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
