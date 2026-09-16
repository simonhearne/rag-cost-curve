"""WS4c pure logic: entity-bridge chain construction, the free filters, and
the gates. Nothing here touches an API, the network, or Milvus -- the live
half is benchlib/recall_quality/provenance_runner.py, exactly as core.py/
runner.py are split.

Design: results/recall-vs-quality.md
sections 2A and 3A (amendment 2026-09-11).

WARNING: sections 2 and 3 of that document are SUPERSEDED. C0/C1/C2 ran and
C1 failed at 0.074 (see results/recall-vs-quality.md). Nothing here implements them.

The decisive property of the amendment: the two gold chunk ROWS are known
before the question exists, so a gold chunk never has to be LOCATED. That is
what C1 spent 3,421 questions failing to do.
"""

from __future__ import annotations

import random
import re
import unicodedata
from collections import Counter

from ..config import (
    SEED,
    WS4C_MAX_PAGE_SHARE,
    WS4C_MIN_BRIDGE_CHARS,
    WS4C_N_QUESTIONS,
    WS4_TOP_K,
    WS4C_BUILD_BUDGET_USD,
    WS4C_FLOOR_MIN_SHARE,
)
from .gold import answer_present_wb

__all__ = [
    "norm_title", "title_eligible", "mentions", "nested", "bridges",
    "page_cap", "select_pairs",
    "f1_gold_in_b", "f2_bridge_absent", "hops_retrieved",
    "gate_a", "gate_b", "gate_c", "gate_d",
]

_WORD = re.compile(r"[a-z0-9]+")
MAX_BRIDGE_WORDS = 10   # n-gram proposal ceiling; longer titles are not used


def norm_title(title: str) -> str:
    """Lowercase alphanumeric words joined by single spaces, with NFKD decomposition.

    A MATCHING normalisation only -- the committed artifact always keeps the
    corpus spelling. Exactly shaped like gold.norm_alias (via core.norm_text):
    NFKD first to preserve accented letters as base + combining mark, then
    extract [a-z0-9]+ tokens. WS4b's lesson is that a normaliser able to emit
    the empty string matches everything, so callers must check for it
    (title_eligible does).
    """
    s = unicodedata.normalize("NFKD", str(title).lower())
    return " ".join(_WORD.findall(s))


def title_eligible(title: str) -> bool:
    """Spec 2A.1: a bridge title must be multi-word and at least
    WS4C_MIN_BRIDGE_CHARS characters. Word count is measured on the RAW title
    (whitespace-split parts that contain at least one alphanumeric), to reject
    accented single-word titles like 'Müller' that normalise to multi-token
    forms. Length is measured on the NORMALISED form, so '---' can never qualify."""
    # Count raw words: whitespace-split parts with at least one alphanumeric
    raw_words = [w for w in str(title).split() if re.search(r"[a-z0-9]", w, re.IGNORECASE)]
    n = norm_title(title)
    return len(raw_words) >= 2 and len(n) >= WS4C_MIN_BRIDGE_CHARS


def _wb_search(needle_norm: str, haystack_norm: str) -> bool:
    if not needle_norm:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(needle_norm)}(?![a-z0-9])",
                     haystack_norm) is not None


def mentions(title: str, text: str) -> bool:
    """Word-boundary containment of `title` in `text`, case-insensitive.

    Not a raw substring test: spec 4 records that WS4's presence metric was a
    substring test and that Tier 0 found it over-counting 9.3% of arm-rows.
    """
    return _wb_search(norm_title(title), norm_title(text))


def nested(a: str, b: str) -> bool:
    """True when one normalised title contains the other as a word run.

    NOT in spec 2A.1 -- an addition, recorded in results/recall-vs-quality.md.
    'Daniel Boone' bridged off the page 'Daniel Boone National Forest' is not
    a second hop, it is the same entity named twice. Empty normalisations
    return True so they are always rejected.
    """
    na, nb = norm_title(a), norm_title(b)
    if not na or not nb:
        return True
    return _wb_search(na, nb) or _wb_search(nb, na)


def bridges(text: str, page_title: str, lookup: dict,
            max_words: int = MAX_BRIDGE_WORDS) -> list[str]:
    """Eligible bridge titles mentioned in `text`, in first-appearance order.

    `lookup` maps norm_title(t) -> t for every ELIGIBLE slice title. Word
    n-grams from the text PROPOSE candidates in O(len(text)) rather than
    scanning 1,525,747 titles per chunk; because the n-gram key and the
    lookup key use the same normalisation, a hit IS a word-boundary match and
    needs no second confirmation.
    """
    words = _WORD.findall(norm_title(text))
    seen: set[str] = set()
    out: list[str] = []
    for i in range(len(words)):
        for n in range(2, max_words + 1):
            if i + n > len(words):
                break
            key = " ".join(words[i:i + n])
            if key in seen:
                continue
            title = lookup.get(key)
            if title is None:
                continue
            seen.add(key)
            if nested(title, page_title):
                continue
            out.append(title)
    return out


def page_cap(n_questions: int = WS4C_N_QUESTIONS,
             share: float = WS4C_MAX_PAGE_SHARE) -> int:
    """Spec 2A.1's concentration cap as a COUNT. floor(), so 2% of 600 is 12
    and the 13th question from one page is refused."""
    return max(1, int(n_questions * share))


def select_pairs(candidates, *, chunks_of: dict, lookup: dict,
                 n_pairs: int, seed: int = SEED, cap: int | None = None):
    """Deterministic (chunk A, bridge, chunk B) triples. $0 -- no API, no
    Milvus. Spec 2A.1.

    candidates  ordered [(row_a, page_title, text_a)]; order is the caller's
                and is itself seeded, so this function adds no hidden order
    chunks_of   bridge title -> ordered corpus row positions of its chunks
    lookup      norm_title -> title, ELIGIBLE titles only
    cap         max pairs one title may supply IN EITHER ROLE. Enforced on
                the POOL, not on the accepted set: filters can only remove
                questions, so a pool cap of `cap` bounds the final set too and
                G-C (spec 3A) cannot fail after the money is spent.

    Only the FIRST usable bridge in a chunk is taken, so one chunk A never
    floods the pool with near-duplicate questions.
    """
    cap = page_cap() if cap is None else cap
    rng = random.Random(f"{seed}:ws4c:pairs")
    used: dict[str, int] = {}
    out: list[dict] = []
    for row_a, page_title, text_a in candidates:
        if len(out) >= n_pairs:
            break
        if used.get(page_title, 0) >= cap:
            continue
        for bridge in bridges(text_a, page_title, lookup):
            if used.get(bridge, 0) >= cap:
                continue
            b_rows = [int(r) for r in chunks_of.get(bridge, ()) if int(r) != int(row_a)]
            if not b_rows:
                continue
            out.append({"row_a": int(row_a), "row_b": int(rng.choice(b_rows)),
                        "page_title": page_title, "bridge_title": bridge})
            used[page_title] = used.get(page_title, 0) + 1
            used[bridge] = used.get(bridge, 0) + 1
            break
    return out


# ------------------------------------------------------- free filters ----
# Spec 2A.2 runs the filters cheapest-first, so nothing pays for a question a
# free check can kill. F1 and F2 are the free ones. F3/F4/F5 cost money and
# live in provenance_runner.py.


def f1_gold_in_b(gold: str, text_b: str) -> bool:
    """F1: the gold answer is present in chunk B, word-boundary.

    Deliberately gold.answer_present_wb and not a new matcher (spec 2A.2).
    It also drops aliases shorter than WS4B_MIN_GOLD_CHARS, so a gold of '12'
    -- unmatchable without false positives -- fails F1 rather than entering
    the set as an unscoreable question, which is what cost WS4 14 of them.
    """
    return answer_present_wb([gold], [text_b])


def f2_bridge_absent(bridge: str, question: str) -> bool:
    """F2: the bridge title does not appear in the question (spec 2A.2). A
    question that names the bridge has leaked the hop and is single-hop."""
    return not mentions(bridge, question)


# ------------------------------------------------------- provenance ------


def hops_retrieved(gold_rows, retrieved_ids, k: int = WS4_TOP_K) -> int:
    """Spec 4: how many DISTINCT gold chunk rows are in the top-k. The
    mediator WS4 wanted, measured without touching the answer string.
    Padding ids (-1) are skipped, as ws4_runner.passages_for does."""
    top = {int(i) for i in list(retrieved_ids)[:k] if int(i) >= 0}
    return sum(1 for r in {int(g) for g in gold_rows} if r in top)


# ------------------------------------------------------------- gates -----


def gate_a(n_accepted: int, spent: float, *,
           target: int = WS4C_N_QUESTIONS,
           budget: float = WS4C_BUILD_BUDGET_USD) -> dict:
    """G-A: >= target questions survive F1-F5 inside the build governor.

    On failure the action is to report the per-filter yield and STOP. The
    governor is NOT raised to reach the number (spec 3A, spec 7): a build
    that cannot produce `target` for `budget` is reporting something.
    """
    return {"gate": "G-A", "n_accepted": int(n_accepted), "target": int(target),
            "spent_usd": round(float(spent), 6), "budget_usd": float(budget),
            "pass": bool(n_accepted >= target)}


def gate_b(questions, text_of: dict) -> dict:
    """G-B: for 100% of questions both gold rows are recorded AND the gold
    answer is present in chunk B. A failure means the build is broken."""
    bad = []
    for q in questions:
        if q.get("row_a") is None or q.get("row_b") is None:
            bad.append(q["qid"])
            continue
        if not f1_gold_in_b(q["gold_answer"], text_of[int(q["row_b"])]):
            bad.append(q["qid"])
    return {"gate": "G-B", "n": len(list(questions)), "n_bad": len(bad),
            "bad_qids": ",".join(map(str, bad[:20])), "pass": not bad}


def gate_c(questions, max_share: float = 0.05) -> dict:
    """G-C (the old C2): no single page supplies more than `max_share` of the
    questions. select_pairs caps the POOL at 2% in either role, so a failure
    here means the cap was not applied."""
    questions = list(questions)
    counts = Counter()
    for q in questions:
        counts[q["page_title"]] += 1
        counts[q["bridge_title"]] += 1
    n = len(questions) or 1
    title, top = counts.most_common(1)[0] if counts else ("", 0)
    return {"gate": "G-C", "top_title": title, "top_count": int(top),
            "top_share": top / n, "max_share": float(max_share),
            "pass": bool(top / n <= max_share)}


def gate_d(hops, min_share: float = WS4C_FLOOR_MIN_SHARE) -> dict:
    """G-D, THE FLOOR GATE (spec 3A). After retrieval and BEFORE any eval
    spend: the reference arm must reach hops_retrieved@10 == 2 on at least
    `min_share` of questions.

    A bridge question hides the second entity and single-shot dense retrieval
    is known to be poor at second hops. If the mediator pins at 1 the curve
    is flat at a FLOOR and WS4c would answer 'retrieval does not bind' from
    the measurement rather than from the world. Failing costs ~$13, not ~$73.
    On failure: report the floor and STOP. Do not run Phase 3 to see what
    happens.
    """
    hops = [int(h) for h in hops]
    n = len(hops)
    two = sum(1 for h in hops if h >= 2)
    share = (two / n) if n else 0.0
    return {"gate": "G-D", "n": n, "n_hops2": two, "share_hops2": share,
            "min_share": float(min_share), "pass": bool(share >= min_share)}
