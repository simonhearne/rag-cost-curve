#!/usr/bin/env python
"""WS4b Tier 0d -- G4 diagnosis, committed evidence for results/recall-vs-quality.md.

Design: results/recall-vs-quality.md
Gate:   benchlib.recall_quality.gold.gate_g4's docstring now cites this script's own
        committed collision measurement -- 51 of 105 (48.6%) four-plus-
        character audited answers occur verbatim in their own gold passages
        (results/ws4b/tier0d_g4_diagnosis.csv, kind=collision_measurement) --
        which superseded an earlier, uncommitted 19/38 estimate that could
        not be reproduced from committed inputs (see the collision_measurement
        row's own `note` field for that history). A coordinator log
        separately asserted the five committed term-scan hits were "four
        golds-collide-with-a-different-row's-answer, plus one question
        containing that row's own answer ('Herne')". Both were traceable
        only to a gitignored log, not to a committed file -- this script
        fixes that by RECOMPUTING both from scratch against read-only,
        already-committed/frozen inputs, at $0, using gate_g4's OWN matching
        rule (case-sensitive, unnormalised, word-boundary regex --
        benchlib/recall_quality/gold.py's gate_g4) rather than a different (normalised)
        one, so "would this collide under G4" is answered faithfully rather
        than approximately.

Writes results/ws4b/tier0d_g4_diagnosis.csv with two kinds of row:

  kind=collision_measurement, one row: of the 107 audited rows' own
  (pooled) wrong answers, how many are >= 4 raw characters ("testable"),
  and of those, how many appear verbatim (gate_g4's own case-sensitive,
  word-boundary rule) in that row's own top-10 GT passage TEXT (texts_for's
  convention: TEXT only, not title, matching scripts/recall-quality/ws4b_tier0.py's
  G0-proven reading of WS4's own presence scoring).

  kind=term_violation, one row per G4 SECONDARY-check hit, reconstructed
  from results/ws4b/tier0d_gold_diff.csv exactly as
  scripts/recall-quality/ws4b_tier0d_refresh.py builds `g4_rows` (one entry for the
  propose prompt, plus a SECOND entry -- scanning the same question/golds
  again -- for rows that also had a verify call), then run through the
  real `benchlib.recall_quality.gold.gate_g4`. Verified below to reproduce
  tier0d_gates.csv's G4 row exactly (n_rows=152, n_term_violations=5,
  examples='126,6,Kansas,Stephen Curry') before being written out with,
  per hit: the flagged term, which field it matched (question or golds),
  whether the term is that row's own audited answer or a different row's,
  and the row's own recorded golds.

Run: .venv/bin/python scripts/recall-quality/ws4b_tier0d_g4_diagnosis.py
Cost: $0 -- no API calls. Reads only already-frozen/committed inputs.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.recall_quality import gold as ws4b_gold
from benchlib.recall_quality import gold_runner as t0d
from benchlib.config import DATA_DIR, WS4_TOP_K, ws4b_dir

LABELLED = "results/ws4b/tier0_staleness_labelled.csv"
GOLD_DIFF = "results/ws4b/tier0d_gold_diff.csv"
PASSAGES = DATA_DIR / "ws4_cache" / "passages.parquet"
GT = DATA_DIR / "gt_10m_top100.npz"
MIN_CHARS = 4   # gate_g4's own docstring threshold for this measurement --
                # distinct from WS4B_MIN_GOLD_CHARS (3), which governs
                # alias_usable/degenerate_golds elsewhere.

# gate_g4's OWN matching rule (benchlib/recall_quality/gold.py), copied verbatim so the
# collision measurement tests exactly what the gate tests: case-sensitive,
# no normalisation, word-boundary on both ends.
_TERM_RE = "(?<![A-Za-z0-9]){}(?![A-Za-z0-9])"


def gate_g4_match(term: str, text: str) -> bool:
    return bool(re.search(_TERM_RE.format(re.escape(term)), text))


def texts_for(qid: int, idx: np.ndarray, table: pd.DataFrame) -> list[str]:
    """Top-WS4_TOP_K GT passage TEXTS for `qid`, read-only against the frozen
    passages.parquet -- same convention as scripts/recall-quality/ws4b_tier0.py's texts_for
    (TEXT only, not title; G0 there proves that reading matches WS4's own
    presence scoring)."""
    ids = [int(i) for i in idx[qid, :WS4_TOP_K]]
    return [table.at[i, "text"] for i in ids if i >= 0 and i in table.index]


def main():
    lab = pd.read_csv(LABELLED)
    diff = pd.read_csv(GOLD_DIFF)
    idx = np.load(GT)["ws4_indices"]
    table = pd.read_parquet(PASSAGES).set_index("row")
    answer_by_qid = dict(zip(lab.qid, lab.answer.astype(str)))

    # ---- (a) collision measurement, gate_g4's own matching rule ----------
    n_tested = n_verbatim = 0
    for _, r in lab.iterrows():
        answer = str(r.answer).strip()
        if len(answer) < MIN_CHARS:
            continue
        n_tested += 1
        blob = "\n\n".join(texts_for(int(r.qid), idx, table))
        if gate_g4_match(answer, blob):
            n_verbatim += 1

    measured_share = n_verbatim / n_tested if n_tested else 0.0
    print(f"collision measurement: {n_verbatim}/{n_tested} "
          f"(prior, uncommitted estimate was 19/38 -- see note)")

    rows = [{
        "kind": "collision_measurement", "qid": "", "term": "", "region": "",
        "is_own_answer": "", "row_golds": "",
        "n_tested": n_tested, "n_verbatim": n_verbatim,
        "note": (f"of the {len(lab)} audited rows' own pooled wrong answers, "
                 f"{n_tested} are >= {MIN_CHARS} raw characters (testable); "
                 f"{n_verbatim} of those appear verbatim, by gate_g4's own "
                 f"case-sensitive word-boundary rule, in that row's own "
                 f"top-{WS4_TOP_K} GT passage TEXT. share={measured_share:.4f}. "
                 f"A prior, uncommitted coordinator-log estimate of 19/38 "
                 f"could not be reproduced from committed inputs and is "
                 f"superseded by this measurement -- see results/recall-vs-quality.md.")
    }]

    # ---- (b) the five term-scan violations, reconstructed exactly --------
    # Mirrors scripts/recall-quality/ws4b_tier0d_refresh.py's g4_rows construction: one
    # entry per row for the propose prompt (question, old_golds), plus a
    # SECOND entry -- scanning the identical question/old_golds again --
    # whenever a verify call also happened (verdict augment/replace, or a
    # VerifyError that still billed a verify call). Both entries always
    # carry the SAME question/golds text, so a hit on such a row is counted
    # (and reported) twice, which is why n_term_violations can exceed the
    # number of distinct flagged terms.
    forbidden = sorted({str(a) for a in lab.answer.dropna()}
                       | set(lab.label.dropna().unique()))
    g4_rows = []
    for _, r in diff.iterrows():
        qid = int(r.qid)
        entry = {"qid": qid, "question": r.question, "golds": r.old_golds,
                 "expected": "", "actual": None}
        g4_rows.append(("propose", entry))
        had_verify = (r.verdict in ("augment", "replace")
                      or r.get("verify_reason") == t0d.VERIFIER_EXHAUSTED_REASON)
        if had_verify:
            g4_rows.append(("verify", dict(entry)))

    # Verify this reconstruction reproduces the committed gate result
    # exactly before trusting it for the per-hit breakdown below.
    check = ws4b_gold.gate_g4([e for _, e in g4_rows], forbidden)
    committed = pd.read_csv("results/ws4b/tier0d_gates.csv")
    committed_g4 = committed[committed.gate == "G4"].iloc[0]
    assert check["n_rows"] == int(committed_g4.n_rows), check
    assert check["n_term_violations"] == int(committed_g4.n_term_violations), check
    assert check["examples"] == committed_g4.examples, check
    print(f"reconstruction verified against committed tier0d_gates.csv: "
          f"n_rows={check['n_rows']}, n_term_violations={check['n_term_violations']}")

    for instance, entry in g4_rows:
        qid = entry["qid"]
        for field in ("question", "golds"):
            text = str(entry[field])
            for term in forbidden:
                t = str(term).strip()
                if not t:
                    continue
                if gate_g4_match(t, text):
                    is_own = (t == answer_by_qid.get(qid))
                    rows.append({
                        "kind": "term_violation", "qid": qid, "term": t,
                        "region": field, "is_own_answer": bool(is_own),
                        "row_golds": entry["golds"], "n_tested": "",
                        "n_verbatim": "",
                        "note": (f"{instance} prompt; row's own audited "
                                 f"answer: {answer_by_qid.get(qid)!r}"),
                    })

    df = pd.DataFrame(rows)
    n_violations = int((df.kind == "term_violation").sum())
    print(f"term-scan violations written: {n_violations}")

    # The line above reproduces tier0d_gates.csv's TOTAL (n_rows,
    # n_term_violations, examples) via `check`, but nothing yet asserts the
    # per-hit BREAKDOWN below sums to the same committed n_term_violations --
    # assert that too before trusting this file as a faithful expansion.
    assert n_violations == int(committed_g4.n_term_violations), (
        f"breakdown violation count {n_violations} != committed "
        f"n_term_violations {committed_g4.n_term_violations}")

    out = ws4b_dir() / "tier0d_g4_diagnosis.csv"
    df.to_csv(out, index=False)
    manifest.record(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
