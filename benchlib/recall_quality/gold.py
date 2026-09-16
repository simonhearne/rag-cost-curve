"""WS4b scoring corrections, and WS4b Tier 0d pure logic: verdict validation,
gold diffing, stratum recomputation, gates and branches. No API, no network,
no filesystem for either half -- the Tier 0d live half is gold_runner.py,
mirroring core.py/runner.py.

Design: results/recall-vs-quality.md
  (both halves: the scoring corrections, and the Tier 0d re-judge R1-R3)

SCORING CORRECTIONS. These are deliberately written as NEW functions rather
than edits to `benchlib.recall_quality.core`. WS4's published numbers were
produced by those functions and must stay reproducible from them; a silent
fix would make `results/ws4/` irreproducible from the code that claims to
produce it.

The two corrections (spec 3.1, 3.2):

  answer_present_wb   word-boundary alias matching, dropping aliases too short
                      to be matched safely. `core.answer_present` is a raw
                      substring test, so gold "12" matches inside "2012" and
                      "II" inside "hawaii".
  is_idk_prefix       a decline is also a decline when the model adds an
                      explanation after it. `core.is_idk` requires exact
                      equality, so three reference-arm declines were scored
                      as errors.

TIER 0D. The pass re-derives gold answers for the 107 rows where some arm
erred. That selection is ONE-SIDED by construction (spec 2): a row where
every arm was correct against a bad gold is never examined. Everything here
therefore supports an UPPER BOUND on corrected accuracy and never a
corrected accuracy.

`in_primary`, `stratum_membership`, `branch_r12` and `branch_r3` belong to
the re-judge/rescore stage (spec 3.4-3.5, R1-R3) and have NO production
caller: gate G1 stopped the pass (0.4348 < 0.6, see results/recall-vs-quality.md)
before the rescore that would have called them ever ran. Their presence
here is the correct, pre-registered implementation waiting on a rescore
that never happened -- not evidence the rescore happened.
"""

from __future__ import annotations

import difflib
import json
import re

from .core import IDK, norm_text
from ..config import (
    WS4B_MIN_GOLD_CHARS,
    WS4B_T0D_CONCORDANCE_MIN, WS4B_T0D_FALSE_CORRECT_MIN,
    WS4B_T0D_MIN_CHANGES, WS4B_T0D_REDACT_MIN_CHARS, WS4B_T0D_UPHOLD_CEILING
)

__all__ = [
    "parse_golds", "norm_alias", "alias_usable", "degenerate_golds",
    "answer_present_old", "answer_present_wb", "is_idk_prefix",
    "VERDICTS", "validate_verdict", "apply_verdict", "operation_of",
    "in_primary", "stratum_membership", "gate_g1", "gate_g2", "gate_g4",
    "branch_r12", "branch_r3", "redact_verbatim_passage_text",
    "REDACTED_PLACEHOLDER",
]


def parse_golds(golds) -> list[str]:
    """runs.csv stores golds as a JSON array string; questions.csv the same.
    Tolerates an already-parsed list and a bare string."""
    if isinstance(golds, (list, tuple)):
        return [str(g) for g in golds]
    s = str(golds)
    if s.startswith("["):
        return [str(g) for g in json.loads(s)]
    return [s]


def norm_alias(gold: str) -> str:
    """WS4's normalisation, then collapse the runs of spaces that
    non-alphanumerics leave behind, so alias length means what it looks like.

    `core.norm_text` maps every non-alphanumeric to a space, so "---" becomes
    "   " -- which strips to the EMPTY STRING and is a substring of every
    passage. That single gold guarantees a presence hit under the old rule.
    """
    return " ".join(norm_text(gold).split())


def alias_usable(gold: str) -> bool:
    """An alias is usable if its normalised form is at least
    WS4B_MIN_GOLD_CHARS characters. Shorter aliases ("12", "II", "", "*")
    cannot be located by string matching without false positives."""
    return len(norm_alias(gold)) >= WS4B_MIN_GOLD_CHARS


def degenerate_golds(golds) -> bool:
    """True when NO alias is usable -- the question cannot be scored by string
    presence at all and is dropped from the corrected primary stratum."""
    return not any(alias_usable(g) for g in parse_golds(golds))


def _blob(texts) -> str:
    """The same normalised, space-joined blob WS4 built, so the old and new
    rules are compared on identical input."""
    return " ".join(norm_text(t) for t in texts)


def answer_present_old(golds, texts) -> bool:
    """Byte-for-byte `core.answer_present`, re-implemented here so gate G0 can
    reproduce the published column from re-derived passage text without
    importing WS4's parsing. Any divergence from `core.answer_present` is a
    bug in this file, and G0 is what catches it."""
    if not texts:
        return False
    blob = _blob(texts)
    return any(norm_text(g) in blob for g in parse_golds(golds))


def answer_present_wb(golds, texts) -> bool:
    """Word-boundary alias matching over the same blob, ignoring aliases too
    short to be usable. Strictly stronger than `answer_present_old`: every
    match here is also a match there, which gate G0b asserts."""
    if not texts:
        return False
    blob = _blob(texts)
    for g in parse_golds(golds):
        if not alias_usable(g):
            continue
        a = norm_alias(g)
        if re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", blob):
            return True
    return False


def is_idk_prefix(pred: str) -> bool:
    """A decline, including one the model then explains.

    Matches `core.is_idk` exactly, plus the case where the decline phrase is
    the whole of the FIRST line. It deliberately does not match the phrase
    appearing later in an answer: the judge already marks "names a gold answer
    only inside a statement denying that it can answer" incorrect, and that
    verdict stands.
    """
    if pred is None:
        return False
    first = str(pred).strip().replace("’", "'").split("\n", 1)[0]
    return first.rstrip(".!").strip().lower() == IDK.lower()


# This repo's hard rule: never commit corpus data. propose_reason/verify_reason
# are free-text model justifications that naturally quote the passage they
# reason about, and are written straight into a COMMITTED CSV.
# WS4B_T0D_REDACT_MIN_CHARS (benchlib/config.py) is the pinned threshold.
REDACTED_PLACEHOLDER = "[passage text redacted]"

VERDICTS = ("keep", "augment", "replace", "unanswerable_from_corpus")


def validate_verdict(verdict: str, old_golds, new_golds) -> str | None:
    """None when the verdict is internally consistent; otherwise why it is not.

    Each verdict carries an obligation, and a stage that emits a verdict whose
    payload contradicts it has not made the judgement it claims to have made.
    """
    if verdict not in VERDICTS:
        return f"unknown verdict {verdict!r}"
    old = set(parse_golds(old_golds))
    new = set(parse_golds(new_golds)) if new_golds else set()

    if verdict == "keep":
        return None if not new or new == old else "keep must not change the golds"
    if verdict == "unanswerable_from_corpus":
        return None if not new else "unanswerable_from_corpus must supply no golds"
    if verdict == "augment":
        if not new > old:
            return "augment must add at least one alias and drop none"
    if verdict == "replace":
        if not new:
            return "replace must supply a non-empty gold list"
        if new == old:
            return "replace must differ from the existing golds"
    if not any(alias_usable(g) for g in new):
        # A correction nobody can match is not a correction. WS4 lost 14
        # questions to unmatchable golds; Tier 0d must not mint more.
        return "no proposed alias is long enough to be matchable"
    return None


def apply_verdict(verdict: str, old_golds, new_golds) -> list[str]:
    """The gold list after the verdict.

    `keep` and `unanswerable_from_corpus` both leave the golds untouched: the
    second is a statement about CORPUS COVERAGE, not about the gold's quality.

    Expect the payload to have already passed `validate_verdict`: the guards
    below fail loudly rather than silently corrupting a gold list.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")

    old = parse_golds(old_golds)
    if verdict in ("keep", "unanswerable_from_corpus"):
        return old

    # For augment and replace, guard against falsy new_golds becoming "None"
    new = parse_golds(new_golds) if new_golds else []

    if verdict == "augment":
        # Union, preserving the original order first, so WS4's published
        # golds remain a prefix and its scoring stays derivable.
        return old + [g for g in new if g not in old]

    # verdict == "replace"
    # Empty gold list makes every answer wrong; reject this.
    if not new:
        raise ValueError("replace verdict with empty gold list")
    return new


def operation_of(verdict: str, old_golds, new_golds) -> str:
    """'changed' when the gold list actually moves, else 'unchanged'. The
    re-judge in spec 3.4 is driven by this, not by the verdict name.

    Expect the payload to have already passed `validate_verdict`: the guards
    in `apply_verdict` fail loudly rather than silently corrupting a gold list.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")

    return ("changed"
            if apply_verdict(verdict, old_golds, new_golds) != parse_golds(old_golds)
            else "unchanged")


# ------------------------------------------------------------- stratum ----
# Spec 3.5 as amended. Membership is a FUNCTION OF THE GOLDS, so a refresh
# moves the population as well as the scores. Tier 0d reports the frozen
# stratum as primary and the recomputed one as a labelled sensitivity.


def in_primary(golds, gt_texts, subset: str) -> bool:
    """WS4b Tier 0's corrected primary stratum, byte-for-byte with
    scripts/recall-quality/ws4b_tier0.py:137:

        gt_presence_wb & ~degenerate_golds & (subset == "main")

    Re-implemented rather than imported because that line lives inside a
    script's main flow; any divergence from it is a bug in this function.
    """
    return bool(subset == "main"
                and not degenerate_golds(golds)
                and answer_present_wb(golds, gt_texts))


def stratum_membership(rows, golds_by_qid: dict, gt_texts_by_qid: dict) -> dict:
    """qid -> in_primary, for one gold assignment.

    Call twice -- once with WS4's golds, once with Tier 0d's -- to see which
    questions the refresh moves into or out of the stratum.
    """
    out = {}
    for r in rows:
        qid = r["qid"]
        out[qid] = in_primary(golds_by_qid[qid], gt_texts_by_qid.get(qid, []),
                              r["subset"])
    return out


# --------------------------------------------------------------- gates ----


def gate_g1(diff_rows, labels_by_qid: dict) -> dict:
    """G1: of the rows WS4b labelled `stale_gold`, what share did Tier 0d
    independently decide to change, and did the verifier uphold it?

    This is the gate Tier 0d is worth running for even if nothing else fires.
    WS4b's labels are provisional pending a human spot-check that has not
    happened, and were applied by the same model that proposed the hypothesis
    they test. Tier 0d re-reads the same rows blind to those labels, so
    disagreement here impeaches one of the two passes -- and
    results/recall-vs-quality.md's numbers are built on those labels.
    """
    stale = [r for r in diff_rows if labels_by_qid.get(r["qid"]) == "stale_gold"]
    upheld = [r for r in stale if r["operation"] == "changed" and r["uphold"]]
    n = len(stale)
    share = (len(upheld) / n) if n else 0.0
    evaluable = n > 0
    return {"gate": "G1", "n_stale_gold": n, "n_upheld_change": len(upheld),
            "concordance": share, "min_concordance": WS4B_T0D_CONCORDANCE_MIN,
            "evaluable": bool(evaluable),
            "pass": bool(n == 0 or share >= WS4B_T0D_CONCORDANCE_MIN)}


def gate_g2(diff_rows) -> dict:
    """G2: is the verifier actually biting?

    The denominator is PROPOSED CHANGES ONLY (spec 5 G2) -- a `keep` is not
    something the verifier upholds, and counting keeps would drown the signal.
    Below WS4B_T0D_MIN_CHANGES the rate is noise, so the gate records
    `not evaluable` rather than passing or failing on three data points.
    """
    proposed = [r for r in diff_rows if r["verdict"] in ("augment", "replace")]
    n = len(proposed)
    rate = (sum(1 for r in proposed if r["uphold"]) / n) if n else 0.0
    evaluable = n >= WS4B_T0D_MIN_CHANGES
    return {"gate": "G2", "n_proposed_changes": n, "uphold_rate": rate,
            "ceiling": WS4B_T0D_UPHOLD_CEILING, "evaluable": bool(evaluable),
            "pass": bool(not evaluable or rate <= WS4B_T0D_UPHOLD_CEILING)}


def gate_g4(rows, forbidden) -> dict:
    """G4: is the pass mechanically blind?

    G4 originally scanned every rendered prompt in full for forbidden terms.
    That is unsound: the audited arms' wrong answers were themselves derived
    FROM the retrieved passages, so a forbidden term routinely reappears in
    the corpus text on its own merits (measured: 51 of 105 (48.6%)
    four-plus-character audited answers occur verbatim in their own gold
    passages -- results/ws4b/tier0d_g4_diagnosis.csv,
    kind=collision_measurement), and
    `core.render_context`'s own `[k]` index markers collide with single-digit
    forbidden terms on almost every prompt. A full-prompt scan can therefore
    only ever fail, on rows where nothing leaked at all.

    G4 now checks two different things:

    PRIMARY -- structural reconstruction. Each entry in `rows` carries
    `expected` (the prompt independently rebuilt via `fill(template,
    question=..., golds=..., passages=...)`, using ONLY the columns Tier 0d
    is allowed to see) and `actual` (the prompt really sent this invocation,
    or `None` for a row recovered from checkpoint without a fresh call, since
    `fill` is deterministic and there is nothing to compare a resumed row
    against). Wherever `actual` is present it must be byte-identical to
    `expected`: any divergence means something outside the allow-list reached
    the model, whatever it is. This is airtight and immune to coincidence,
    unlike a term scan -- and unlike the term scan it also validates a row's
    passages, since `expected` is rebuilt from the same frozen
    `passages.parquet` lookup used for the live call.

    SECONDARY -- a forbidden-term scan, word-boundary matched (never
    substring -- '126' must not fire on '1260'; WS4's presence metric died of
    exactly that, spec 4), restricted to each row's `question` and `golds`
    fields ONLY, never the passages. Those two fields are short and
    controlled, and are the only place a withheld column (an arm's answer, an
    arm name, a prior label) could plausibly be threaded into a prompt by a
    bug without also producing a structural mismatch above.

    Fails if either check fires on any row. A failure here is not a quality
    problem: it means every downstream number is contaminated and the run
    stops.
    """
    rows = list(rows)
    mismatched, hits = [], []
    for r in rows:
        if r.get("actual") is not None and r["actual"] != r["expected"]:
            mismatched.append(r["qid"])
        for field in ("question", "golds"):
            text = str(r.get(field, ""))
            for term in forbidden:
                t = str(term).strip()
                if not t:
                    continue
                if re.search(rf"(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])", text):
                    hits.append(t)
    return {"gate": "G4", "n_rows": len(rows),
            "n_structural_mismatches": len(mismatched),
            "mismatched_qids": ",".join(str(q) for q in mismatched[:5]),
            "n_term_violations": len(hits),
            "examples": ",".join(sorted(set(hits))[:5]),
            "pass": not mismatched and not hits}


# ------------------------------------------------------------ branches ----


def branch_r12(cls_before: str, cls_after: str) -> dict:
    """R1 if WS4 section 7.1's classification moves, R2 if it does not.
    Mutually exclusive and jointly exhaustive (spec 5)."""
    changed = cls_before != cls_after
    return {"branch": "R1" if changed else "R2",
            "classification_before": cls_before, "classification_after": cls_after,
            "changed": bool(changed)}


def branch_r3(flips) -> dict:
    """R3: did re-judging surface false-corrects -- arms scored right against a
    gold that was wrong? INDEPENDENT of R1/R2; it may fire alongside either.

    Spec 2: the one-sided row selection means this rate is the pass's own
    measurement of what that bias costs.
    """
    flips = list(flips)
    n = len(flips)
    bad = sum(1 for f in flips if f == "correct_to_error")
    rate = (bad / n) if n else 0.0
    return {"branch": "R3", "n_rejudged": n, "n_correct_to_error": bad,
            "false_correct_rate": rate, "threshold": WS4B_T0D_FALSE_CORRECT_MIN,
            "fires": bool(rate >= WS4B_T0D_FALSE_CORRECT_MIN)}


# --------------------------------------------------------------- redaction ----


def _fold(s: str) -> str:
    """Lowercase `s` WITHOUT changing its length, so an index into the folded
    string is also an index into the original. `str.lower()` is not
    length-preserving for every codepoint (U+0130 lowercases to two chars),
    so any character whose lowercase is not a single character is left alone
    -- it simply will not case-fold-match, which is the safe direction.
    """
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in s)


def redact_verbatim_passage_text(text: str, passages_blob: str,
                                  min_len: int = WS4B_T0D_REDACT_MIN_CHARS,
                                  case_insensitive: bool = False) -> str:
    """Replace any run of `text` that also appears verbatim in
    `passages_blob`, `min_len` characters or longer, with
    `REDACTED_PLACEHOLDER`. The rest of `text` is returned unchanged.

    A model reasoning about a passage naturally quotes it, and the free-text
    `propose_reason`/`verify_reason` columns are written to a CSV that gets
    committed to a public repo -- corpus text (the embeddings dataset has no
    license tag) must not ride along inside them. This is a structural fix,
    not a one-off scrub: call it wherever a reason is about to be written to
    a committed artifact, on the passages belonging to THAT SAME row (never
    another row's -- that would neither redact real leaks nor catch them).

    Repeatedly finds THE single longest common substring (`find_longest_match`
    over the full, current text) and redacts it, until nothing left is at
    least `min_len` long. A one-shot call to `get_matching_blocks` was tried
    first and rejected: its recursive find-longest-then-split-and-recurse
    decomposition is not guaranteed to surface every maximal common
    substring once there are multiple competing candidates elsewhere in a
    multi-thousand-character passage blob, and it missed a real quote in
    practice (round 2 review fix). Recomputing the true longest match against
    the ORIGINAL blob on every pass, against the text as redacted so far, is
    slower but cannot miss one: every pass either finds and removes a real
    match or proves none remain.

    `case_insensitive` matches on a length-preserving lowercase fold of both
    sides while still redacting the span out of the ORIGINAL `text`, so a
    quote that differs from the corpus only in capitalisation is still caught.
    It defaults to False, which is Tier 0d's exact behaviour: Tier 0d only
    ever fed this function MODEL-written reasons, which quote a passage with
    its original casing, and `tier0d_gold_diff.csv` is merged and pinned in
    `data/MANIFEST.json`. Pass True for HUMAN-written free text, which
    routinely re-types a quote in lower case -- see
    `scripts/recall-quality/ws4b_decline_spotcheck.py`, where the case-sensitive default
    left ~100 characters of a retrieved passage standing in one note
    (qid 2144) because the note lowercased it. results/recall-vs-quality.md
    records the four short spans that flipping this default would newly flag in the
    already-committed Tier 0d file.
    """
    if not text or not passages_blob:
        return text
    hay = _fold(passages_blob) if case_insensitive else passages_blob
    while True:
        needle = _fold(text) if case_insensitive else text
        sm = difflib.SequenceMatcher(None, needle, hay, autojunk=False)
        m = sm.find_longest_match(0, len(needle), 0, len(hay))
        if m.size < min_len:
            return text
        text = text[:m.a] + REDACTED_PLACEHOLDER + text[m.a + m.size:]
