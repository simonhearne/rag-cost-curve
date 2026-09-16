#!/usr/bin/env python
"""WS4b decline audit -- apply the completed HUMAN spot-check and commit it.

Context. results/recall-vs-quality.md made every label from the in-session
auditor (Claude Opus 5, the same system that proposed WS4b's hypothesis)
provisional "pending a human spot-check of a random 20 rows; if that
disagrees on more than 3 of 20, the audit should be redone by a human". The
decline audit inherits that threshold by reference -- it never states its
own. Tier 0b 9.4 then moved the check from recommended to REQUIRED. This
script is what lifts it for the decline audit.

The sheet. Simon Hearne completed `data/ws4b_cache/decline_spotcheck_20.csv`
(gitignored; corpus-derived notes) with a per-row verdict and note on all 20
rows. The 20 rows are reproduced here from the committed labels with the
project seed, and the run ABORTS if they do not match -- the sample's
provenance is checked, not asserted.

  NOT BLIND. The sheet carries the auditor's `label` and `justification`
  columns, and `your_verdict` is a judgement ON that label. This is
  confirmatory review, a weaker instrument than Tier 0d's blind
  re-derivation, and 6.4 says so in as many words. It clears the forced-redo
  trigger; it does not turn 6 into a headline.

Redaction. Several notes quote the retrieved passage windows verbatim
(qid 2144 is the clearest: ~100 characters). This repo's hard rule is that
corpus text never gets committed -- the embeddings dataset has no license
tag. Every free-text field is therefore run through
`benchlib.recall_quality.gold.redact_verbatim_passage_text` against THAT ROW'S OWN ten
retrieved passages, exactly as Tier 0d did for `tier0d_gold_diff.csv`, with
one difference: `case_insensitive=True`, because a human re-types a quote in
lower case and the case-sensitive default left most of qid 2144 standing.
`tests/recall_quality/test_gold.py` re-runs this check against whatever is committed.

Passage source. The auditor read the ten passages the REFERENCE ARM
(`sq8_np512`) actually retrieved -- `retrieved_ids` in
`data/ws4_cache/runs_checkpoint.jsonl` -- not the 10M ground-truth top-10
that Tier 0d used. Redacting against the wrong row's, or the wrong arm's,
passages would neither remove a real leak nor detect one.

Run:  python scripts/recall-quality/ws4b_decline_spotcheck.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.recall_quality import gold as ws4b_gold
from benchlib.config import DATA_DIR, SEED, WS4_TOP_K, ws4b_dir

SHEET = DATA_DIR / "ws4b_cache" / "decline_spotcheck_20.csv"
RUNS = DATA_DIR / "ws4_cache" / "runs_checkpoint.jsonl"
PASSAGES = DATA_DIR / "ws4_cache" / "passages.parquet"

REF_ARM = "sq8_np512"          # the 93 declines in 6 are this arm's
SPOTCHECK_N = 20               # 5's "a random 20 rows"
REDO_THRESHOLD = 3             # 5: "disagrees on more than 3 of 20" -> redo
FREE_TEXT = ("justification", "your_note")

REVIEWER = "Simon Hearne (human), 2026-09-12; NOT blind -- saw label + justification"

# `your_verdict` -> does the reviewer agree with the auditor's LABEL?
# The threshold 5 sets counts label disagreements, nothing else.
AGREES = {
    "agree": True,
    "agree_borderline": True,
    # 513: the label is right and the auditor's stated reason for it is wrong.
    # A distinct defect from a wrong label, and NOT what the threshold
    # measures -- counted as agreement here and reported separately in 6.4.
    "agree_label_wrong_justification": True,
    "disagree_precedence": False,
    "DISAGREE": False,
}


def reference_arm_passages(qids) -> dict[int, str]:
    """qid -> the concatenated text of the ten passages REF_ARM retrieved."""
    want = set(int(q) for q in qids)
    ids: dict[int, list[int]] = {}
    with open(RUNS) as f:
        for line in f:
            r = json.loads(line)
            if r["arm"] == REF_ARM and int(r["qid"]) in want:
                ids[int(r["qid"])] = [int(x) for x in json.loads(r["retrieved_ids"])]
    missing = want - set(ids)
    if missing:
        raise SystemExit(f"no {REF_ARM} retrieval for qids {sorted(missing)}")
    tbl = pd.read_parquet(PASSAGES).set_index("row")
    return {
        q: " ".join(str(tbl.at[x, "text"]) for x in rows[:WS4_TOP_K] if x in tbl.index)
        for q, rows in ids.items()
    }


def main() -> None:
    out = ws4b_dir()
    labelled = pd.read_csv(out / "decline_labelled.csv")
    sheet = pd.read_csv(SHEET)

    # ---- provenance: the 20 rows must BE the seeded sample, not merely claim to be
    expect = sorted(labelled.sample(n=SPOTCHECK_N, random_state=SEED).qid.tolist())
    got = sorted(int(q) for q in sheet.qid)
    if expect != got:
        raise SystemExit(
            f"sheet is not decline_labelled.csv.sample(n={SPOTCHECK_N}, "
            f"random_state={SEED}); expected {expect}, got {got}")
    bad = set(sheet.your_verdict) - set(AGREES)
    if bad:
        raise SystemExit(f"unrecognised your_verdict value(s): {sorted(bad)}")

    # ---- redact every free-text field against that row's own passages
    blobs = reference_arm_passages(sheet.qid)
    red = sheet.copy()
    n_redacted = 0
    for i, r in red.iterrows():
        for col in FREE_TEXT:
            text = r[col]
            if pd.isna(text) or not str(text):
                continue
            clean = ws4b_gold.redact_verbatim_passage_text(
                str(text), blobs[int(r.qid)], case_insensitive=True)
            if clean != str(text):
                n_redacted += 1
            red.at[i, col] = clean
    red["reviewer"] = REVIEWER
    red.to_csv(out / "decline_spotcheck_20.csv", index=False)

    # ---- the tally the threshold is actually about
    agree = sheet.your_verdict.map(AGREES)
    n_disagree = int((~agree).sum())
    summary = pd.DataFrame([{
        "n": len(sheet),
        "n_agree_label": int(agree.sum()),
        "n_disagree_label": n_disagree,
        "redo_threshold": REDO_THRESHOLD,
        "redo_forced": bool(n_disagree > REDO_THRESHOLD),
        "n_label_right_justification_wrong": int(
            (sheet.your_verdict == "agree_label_wrong_justification").sum()),
        "n_free_text_fields_redacted": n_redacted,
        "blind": False,
        "reviewer": REVIEWER,
        "seed": SEED,
    }])
    summary.to_csv(out / "decline_spotcheck_summary.csv", index=False)

    for p in (out / "decline_spotcheck_20.csv", out / "decline_spotcheck_summary.csv"):
        manifest.record(p)

    print(f"{int(agree.sum())}/{len(sheet)} agree with the auditor's label, "
          f"{n_disagree} disagree (threshold: redo if > {REDO_THRESHOLD})")
    print(f"redo forced: {n_disagree > REDO_THRESHOLD}")
    print(f"redacted {n_redacted} free-text field(s) against the row's own "
          f"{REF_ARM} passages")
    print(f"wrote 2 CSVs into {out}/ and recorded both in data/MANIFEST.json")


if __name__ == "__main__":
    main()
