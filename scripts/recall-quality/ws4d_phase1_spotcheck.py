#!/usr/bin/env python
"""WS4d Phase 1 -- gate E4, a BLIND human spot-check of 20 of the 450
model-assigned eligibility labels.

Design: results/recall-vs-quality.md

This is the deliberate correction of the spot-check recorded in
results/recall-vs-quality.md, which showed the reviewer the model's label
first and asked "is this defensible?",
which is confirmatory review, not independent re-derivation. Here the human
sees only the question, golds and the arm-independent ground-truth top-10
passages -- never the model labels -- and assigns their own label before
anything is compared.

Two modes:
  --emit   build data/ws4d_cache/e4_blind_sheet.csv (gitignored: it carries
           raw corpus passages) for a human to fill in `your_label` for all
           20 rows, WITHOUT reading eligibility_labels.csv.
  --score  re-derive the seeded sample, compare the human's BINARY call
           against the adjudicated class, run gate E4, and write the
           REDACTED results/ws4d/eligibility_spotcheck_20.csv.

Run:  python scripts/recall-quality/ws4d_phase1_spotcheck.py --emit
      # ... a human fills in `your_label` for all 20 rows, blind ...
      python scripts/recall-quality/ws4d_phase1_spotcheck.py --score
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd

from benchlib import manifest
from benchlib.recall_quality import gold as ws4b_gold
from benchlib.recall_quality import eligibility as ws4d
from benchlib.config import SEED, WS4D_SPOTCHECK_N, ws4d_cache, ws4d_dir
from benchlib.recall_quality.eligibility_runner import gt_passages

REVIEWER = "Simon Hearne (human), 2026-09-12; BLIND -- did not see either model's label"
BLIND_COLUMNS = ["qid", "question", "golds", "passages", "your_label", "your_note"]
LEAKS = ("label_a", "label_b", "adjudicated", "class", "reason_a", "reason_b")
SHEET = ws4d_cache() / "e4_blind_sheet.csv"


def emit():
    labels = pd.read_csv(ws4d_dir() / "eligibility_labels.csv")
    sample = labels.sample(n=WS4D_SPOTCHECK_N, random_state=SEED)
    passages = gt_passages(sample.qid)
    sheet = pd.DataFrame({
        "qid": sample.qid.values,
        "question": sample.question.values,     # present per ruling C2
        "golds": sample.golds.values,
        "passages": [passages[int(q)] for q in sample.qid],
        "your_label": "", "your_note": "",
    })[BLIND_COLUMNS]
    leaked = [c for c in LEAKS if c in sheet.columns]
    assert not leaked, f"sheet would leak the model labels: {leaked}"
    ws4d_cache().mkdir(parents=True, exist_ok=True)
    sheet.to_csv(SHEET, index=False)
    print(f"wrote {SHEET} -- fill in `your_label` for all "
          f"{WS4D_SPOTCHECK_N} rows WITHOUT reading eligibility_labels.csv.")
    print(f"valid labels: {ws4d.LABELS}")


def score():
    labels = pd.read_csv(ws4d_dir() / "eligibility_labels.csv")
    sheet = pd.read_csv(SHEET)
    expect = sorted(labels.sample(n=WS4D_SPOTCHECK_N,
                                  random_state=SEED).qid.tolist())
    if sorted(int(q) for q in sheet.qid) != expect:
        raise SystemExit("sheet is not the seeded sample; provenance checked, "
                         "not asserted")
    adjudicated = labels.set_index("qid")["class"].to_dict()
    verdicts, rows = [], []
    for _, r in sheet.iterrows():
        err = ws4d.validate_label(str(r.your_label))
        if err:
            raise SystemExit(f"qid {r.qid}: {err}")
        # The human labelled a CLASS; E4 scores the BINARY decision, which is
        # what filters the curve -- same reason E3 gates the binary kappa.
        model_binary = ("eligible" if adjudicated[int(r.qid)] == "eligible"
                        else "ineligible")
        human_binary = ("eligible" if str(r.your_label) == "eligible"
                        else "ineligible")
        v = "agree" if model_binary == human_binary else "disagree"
        verdicts.append(v)
        rows.append({"qid": int(r.qid), "human_label": r.your_label,
                     "model_class": adjudicated[int(r.qid)],
                     "verdict": v, "your_note": r.your_note})
    e4 = ws4d.gate_e4(verdicts)
    passages = gt_passages(sheet.qid)
    out = pd.DataFrame(rows)
    out["your_note"] = [
        ws4b_gold.redact_verbatim_passage_text(str(t), passages[int(q)],
                                               case_insensitive=True)
        for q, t in zip(out.qid, out.your_note)]
    out["reviewer"] = REVIEWER
    out.to_csv(ws4d_dir() / "eligibility_spotcheck_20.csv", index=False)
    manifest.record(ws4d_dir() / "eligibility_spotcheck_20.csv")
    # E4 joins the gate file Phase 2 reads, or Phase 2's all-passed check
    # would never see it.
    gates = pd.read_csv(ws4d_dir() / "eligibility_gates.csv")
    gates = pd.concat([gates[gates.gate != "E4"], pd.DataFrame([e4])])
    gates.to_csv(ws4d_dir() / "eligibility_gates.csv", index=False)
    manifest.record(ws4d_dir() / "eligibility_gates.csv")
    print(f"E4 {e4['n_disagree']}/{e4['n']} disagree "
          f"(threshold >{e4['threshold']}): passed={e4['passed']}")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.emit:
        emit()
    else:
        score()


if __name__ == "__main__":
    main()
