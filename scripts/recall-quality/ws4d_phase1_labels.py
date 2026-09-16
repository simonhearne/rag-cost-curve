#!/usr/bin/env python
"""WS4d Phase 1 -- blind question eligibility over all 450 WS4 questions.

Design: results/recall-vs-quality.md

Two labellers, identical prompts, neither sees the other. Each sees only the
question, its golds, and the ARM-INDEPENDENT ground-truth top-10 passages.

Run:  python scripts/recall-quality/ws4d_phase1_labels.py [--limit N] [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.recall_quality import gold as ws4b_gold
from benchlib.recall_quality import eligibility as ws4d
from benchlib.recall_quality import eligibility_runner as p1
from benchlib.config import (RESULTS_DIR, SEED, WS4D_LABEL_MODEL_A,
                             WS4D_LABEL_MODEL_B, WS4D_N_QUESTIONS,
                             WS4D_PHASE1_CAP_USD, ws4d_dir)
from benchlib.recall_quality.eligibility_runner import gt_passages   # pre-flight ruling C1
# Cost/governor helpers are shared machinery, not code-retrieval-specific.
from benchlib.code_retrieval.bench import BudgetExceeded, load_pricing

MODELS = {"a": WS4D_LABEL_MODEL_A, "b": WS4D_LABEL_MODEL_B}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = ws4d_dir()
    out.mkdir(parents=True, exist_ok=True)
    templates = {k: (out / f"label_prompt_{k}.txt").read_text() for k in MODELS}

    # Every WS4 MAIN question (spec §3.1's 450). The 325-question primary
    # stratum is the curve's denominator in Phase 2 and is a strict subset of
    # these; the remaining 125 are labelled so a benchmark-wide ineligibility
    # rate can be reported.
    #
    # Source is results/ws4/runs.csv, NOT tier0_runs_corrected.csv
    # (pre-flight ruling P2, benchlib/config.py commit 385aaf1). Two reasons:
    #   1. tier0_runs_corrected.csv carries 500 distinct qids -- 450 `main`
    #      plus 50 `sanity` -- so an unfiltered read gives 500, not 450.
    #   2. it has no `question` or `golds` column at all, which the label
    #      prompt and ruling C2 both need.
    # results/ws4/ is READ here and never written; this repo's "results/ws4/
    # is unmodified" rule is respected.
    runs = pd.read_csv(RESULTS_DIR / "ws4" / "runs.csv")
    qs = (runs[runs.subset == "main"].drop_duplicates("qid")
              [["qid", "question", "golds"]]
              .sort_values("qid").reset_index(drop=True))
    assert len(qs) == WS4D_N_QUESTIONS, f"expected {WS4D_N_QUESTIONS}, got {len(qs)}"

    passages = gt_passages(qs.qid)
    if args.dry_run:
        print(f"would render {len(qs) * len(MODELS)} prompts; "
              f"mean passage chars {np.mean([len(v) for v in passages.values()]):.0f}")
        return

    client = p1.make_client()
    prices = load_pricing("results/ws6a/pricing.csv")
    gov = p1.tier_governor()
    done = {}
    if p1.CHECKPOINT.exists():
        with open(p1.CHECKPOINT) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    done[(int(r["qid"]), r["labeller"])] = r

    rendered, rows, n_new = [], [], 0
    for _, w in qs.iterrows():
        qid = int(w.qid)
        for key, model in MODELS.items():
            prior = done.get((qid, key))
            # A row checkpointed mid-abort is INCOMPLETE, not done: an empty
            # label means the charge landed but the result was never written.
            if prior and prior.get("label"):
                rows.append(prior)
                rendered.append({"qid": qid, "model": key,
                                 "expected": p1.fill(templates[key],
                                                     question=w.question,
                                                     golds=w.golds,
                                                     passages=passages[qid]),
                                 "actual": None})
                continue
            if args.limit is not None and n_new >= args.limit:
                break
            row = {"qid": qid, "labeller": key, "model": model,
                   "question": w.question, "golds": w.golds,
                   "label": "", "reason": "", "cost_total_billed": 0.0}
            try:
                r = p1.label(client, templates[key], question=w.question,
                             golds=w.golds, passages=passages[qid], model=model)
                # row.update() BEFORE charge(): charge() can raise
                # BudgetExceeded after the call is already billed, and the row
                # must carry a real label before that can happen or an aborted
                # row checkpoints blank and the resume check above re-runs it.
                row.update(label=r["label"], reason=r["reason"][:300])
                rendered.append({"qid": qid, "model": key,
                                 "expected": p1.fill(templates[key],
                                                     question=w.question,
                                                     golds=w.golds,
                                                     passages=passages[qid]),
                                 "actual": r["prompt"]})
                p1.charge(gov, r["usage"], prices, model, row)
            except BudgetExceeded as exc:
                print(f"phase1  STOP: {exc}")
                if row["cost_total_billed"] > 0:
                    p1.append_checkpoint(p1.CHECKPOINT, row)
                rows.append(row)
                _write(out, rows, rendered)
                return
            p1.append_checkpoint(p1.CHECKPOINT, row)
            rows.append(row)
            n_new += 1
    _write(out, rows, rendered)


def _write(out, rows, rendered):
    """Adjudicate, redact, gate, record. Redaction uses case_insensitive=True:
    the reasons are model-written but the guard is cheap and the spec's
    §3.5 requires it."""
    df = pd.DataFrame(rows)
    wide = df.pivot(index="qid", columns="labeller",
                    values=["label", "reason"]).reset_index()
    wide.columns = ["qid", "label_a", "label_b", "reason_a", "reason_b"]
    # pre-flight ruling C2: Task 9's blind sheet reads question and golds from
    # THIS file. Without them emit() would silently produce an unanswerable
    # sheet rather than failing loudly, destroying E4's validity.
    meta = df.drop_duplicates("qid").set_index("qid")[["question", "golds"]]
    wide = wide.join(meta, on="qid")
    adj = [ws4d.adjudicate(a, b) for a, b in zip(wide.label_a, wide.label_b)]
    wide["adjudicated"] = [x[0] for x in adj]
    wide["class"] = [x[1] for x in adj]

    passages = gt_passages(wide.qid)
    for col in ("reason_a", "reason_b"):
        wide[col] = [
            ws4b_gold.redact_verbatim_passage_text(str(t), passages[int(q)],
                                                   case_insensitive=True)
            for q, t in zip(wide.qid, wide[col])]
    wide.to_csv(out / "eligibility_labels.csv", index=False)

    e1 = ws4d.gate_e1(rendered)
    e2 = {"gate": "E2", "n": len(wide), "expected": WS4D_N_QUESTIONS,
          "passed": bool(len(wide) == WS4D_N_QUESTIONS
                         and wide.label_a.ne("").all()
                         and wide.label_b.ne("").all())}
    e3 = ws4d.gate_e3(wide.label_a.tolist(), wide.label_b.tolist())
    pd.DataFrame([e1, e2, e3]).to_csv(out / "eligibility_gates.csv", index=False)

    conf = pd.crosstab(wide.label_a, wide.label_b)
    conf.to_csv(out / "eligibility_agreement.csv")

    for name in ("eligibility_labels.csv", "eligibility_gates.csv",
                 "eligibility_agreement.csv"):
        manifest.record(out / name)
    print(f"E1 {e1['passed']}  E2 {e2['passed']}  "
          f"E3 kappa_binary={e3['kappa_binary']:.4f} {e3['passed']}")
    print(f"spend ${p1.tier_spent():.4f} of ${WS4D_PHASE1_CAP_USD:.2f}")


if __name__ == "__main__":
    main()
