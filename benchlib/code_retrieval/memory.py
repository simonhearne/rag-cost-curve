"""WS6b pure half: lexicon, corpus model, depth arithmetic, probe derivation,
structural validators, aggregation. No I/O, no API calls -- everything here is
unit-tested. The live half lives in ws6b_runner.py.

Design: results/code-retrieval.md
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Spec 3.6. Values are drawn from a space of ~7,900, so a blind guess is
# right about once in 7,900 -- which is what makes the parametric gate (5.5)
# passable by construction rather than by luck. Round numbers and powers of
# two are excluded because those are precisely what a model guesses when it
# is guessing.
VALUE_POOL: tuple[int, ...] = tuple(
    v for v in range(17, 9974) if v % 5 != 0 and (v & (v - 1)) != 0
)

# Coined syllables. Deliberately not morphemes of real product, library or
# person names: a model cannot have read about a system that does not exist.
_COMP_A = ("quill", "tang", "pel", "hob", "wras", "brim", "fen", "gald",
           "murk", "thax", "vell", "sprig", "corb", "dray", "flin", "gorse")
_COMP_B = ("rack", "wood", "met", "nail", "sey", "ley", "dle", "ric",
           "ton", "vex", "mure", "gate", "cote", "well", "fold", "ridge")
_PARAM_A = ("sluice", "gantry", "kerf", "swale", "plinth", "camber",
            "girth", "reave", "tine", "warp")
_PARAM_B = ("depth", "width", "budget", "ladder", "quota", "stride",
            "margin", "ceiling")
_PERSON_A = ("Marn", "Vess", "Dorl", "Kip", "Sela", "Tor", "Ilse", "Bran",
             "Rue", "Quen")
_PERSON_B = ("adou", "iker", "olm", "esna", "urrow", "ipley")
_MECH_A = ("shunt", "ladder", "relay", "sieve", "spindle", "bailer")
_MECH_B = ("queue", "buffer", "gateway", "arbiter", "ledger", "harness")


@dataclass(frozen=True)
class Lexicon:
    """Coined names for one generated world. Deterministic in the RNG."""
    components: tuple[str, ...]
    params: tuple[str, ...]
    persons: tuple[str, ...]
    mechanisms: tuple[str, ...]


def _cross(rng: random.Random, a: tuple[str, ...], b: tuple[str, ...],
           sep: str = "") -> tuple[str, ...]:
    """Every a x b combination, shuffled deterministically.

    A list comprehension over two ordered tuples, never a set: set iteration
    order is not stable across processes and would break byte-exact
    regeneration (spec 3.2).
    """
    pairs = [x + sep + y for x in a for y in b]
    rng.shuffle(pairs)
    return tuple(pairs)


def make_lexicon(rng: random.Random) -> Lexicon:
    """Build one world's vocabulary. The four name spaces are disjoint.

    Disjointness matters: if a component and a parameter could share a name,
    a probe asking about `quillrack` could be answered from a bullet about a
    different `quillrack`, and the probe would be measuring nothing.
    """
    components = _cross(rng, _COMP_A, _COMP_B)
    params = _cross(rng, _PARAM_A, _PARAM_B)
    persons = _cross(rng, _PERSON_A, _PERSON_B)
    mechanisms = _cross(rng, _MECH_A, _MECH_B, sep="-")

    taken: set[str] = set()
    out = []
    for space in (components, params, persons, mechanisms):
        kept = tuple(n for n in space if n not in taken)
        taken.update(kept)
        out.append(kept)
    return Lexicon(*out)


import uuid


def seeded_uuid(rng: random.Random) -> str:
    """A UUID drawn from the seeded RNG.

    NEVER uuid.uuid4() -- that would make the corpus different on every
    generation and break the byte-exact regeneration guarantee (spec 3.2).
    """
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


@dataclass(frozen=True)
class Turn:
    time_hhmm: str
    session_id: str
    turn_id: str
    bullets: tuple[str, ...]


@dataclass(frozen=True)
class Session:
    time_hhmm: str
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class Day:
    date_iso: str
    sessions: tuple[Session, ...]


# The anchor comment is what the real format carries and is arm (a)'s L3
# entry point. memsearch's clean_content_for_embedding() strips HTML comments
# before embedding, so it dilutes no vector.
_ANCHOR = ("<!-- session:{session_id} turn:{turn_id} "
           "transcript:/synthetic/ws6b/{session_id}.jsonl -->")


def render_turn(turn: Turn) -> str:
    lines = [f"### {turn.time_hhmm}",
             _ANCHOR.format(session_id=turn.session_id, turn_id=turn.turn_id)]
    lines += [f"- {b}" for b in turn.bullets]
    return "\n".join(lines)


def render_day(day: Day) -> str:
    """Render one day file. Layout matches .memsearch/memory/YYYY-MM-DD.md."""
    parts = [""]
    for session in day.sessions:
        parts.append(f"## Session {session.time_hhmm}\n")
        for turn in session.turns:
            parts.append(render_turn(turn) + "\n")
    return "\n".join(parts).rstrip() + "\n"


# ----------------------------------------------- depth, sizes, truncation


def cumulative_before(unit_tokens: list[int]) -> list[int]:
    """Exclusive prefix sums: tokens appearing BEFORE each unit.

    Exclusive, not inclusive: a fact's depth is how much history precedes it,
    so the first unit sits at depth 0.
    """
    out, running = [], 0
    for n in unit_tokens:
        out.append(running)
        running += n
    return out


def depth_of(cum_before: int, total: int) -> float:
    """Relative position of a fact in a history, in MEASURED tokens.

    Spec 3.7: never a file, session or turn index. File counts made WS6a's
    curve shape an artifact of how the subsets happened to divide.
    """
    if total <= 0:
        raise ValueError("total tokens must be positive")
    return cum_before / total


def size_prefix(day_tokens: list[int], budget: int) -> int:
    """How many leading whole day files fit in `budget` tokens.

    Greedy and never overshooting, so a size's measured total is always <=
    its target. The published x-axis is the measured total, not the budget.
    """
    total, n = 0, 0
    for t in day_tokens:
        if total + t > budget:
            break
        total += t
        n += 1
    return n


def truncate_recent(day_tokens: list[int], budget: int) -> tuple[int, float]:
    """Index of the first retained day file, and the retained fraction.

    Replay-truncated keeps the MOST RECENT whole day files -- what a real
    system does when history outgrows the window. Whole files only (spec
    5.3): splitting a file mid-way would put half a session in context and
    make `fact_in_window` ambiguous for the facts in it.
    """
    total = sum(day_tokens)
    if total <= budget:
        return 0, 1.0
    kept, first = 0, len(day_tokens)
    for i in range(len(day_tokens) - 1, -1, -1):
        if kept + day_tokens[i] > budget:
            break
        kept += day_tokens[i]
        first = i
    return first, (kept / total if total else 1.0)


# ------------------------------------------- planted facts and probe text

FACT_TYPES = ("decision", "value", "owner", "superseded")


@dataclass(frozen=True)
class PlantedFact:
    """One (subject, attribute, value) triple planted in the history.

    `alternative` is the rejected option for `decision` facts; `stale_value`
    is the pre-revision value for `superseded` facts. Both are rendered into
    the corpus and NEVER into the derived question.
    """
    fact_id: str
    fact_type: str
    subject: str
    attribute: str
    value: str
    alternative: str = ""
    stale_value: str = ""


def render_fact_bullet(fact: PlantedFact) -> str:
    """The bullet planted at the fact's location.

    For `superseded` this is the ORIGINAL statement; the revision is a
    separate bullet (render_revision_bullet) planted later in the history.
    """
    if fact.fact_type == "decision":
        return (f"We decided to route `{fact.subject}` through "
                f"`{fact.value}` rather than `{fact.alternative}`.")
    if fact.fact_type == "value":
        return (f"We set `{fact.attribute}` on `{fact.subject}` "
                f"to {fact.value}.")
    if fact.fact_type == "owner":
        return f"{fact.value} owns `{fact.subject}`."
    if fact.fact_type == "superseded":
        return (f"We decided `{fact.attribute}` on `{fact.subject}` "
                f"would be {fact.stale_value}.")
    raise ValueError(f"unknown fact type {fact.fact_type!r}")


def render_revision_bullet(fact: PlantedFact) -> str:
    """The later statement that supersedes the original."""
    if fact.fact_type != "superseded":
        raise ValueError("only superseded facts have a revision")
    return (f"We revisited `{fact.attribute}` on `{fact.subject}` and "
            f"changed it from {fact.stale_value} to {fact.value}.")


def derive_question(fact: PlantedFact) -> str:
    """The probe question, derived mechanically from the triple.

    Spec 3.4. Questions name only subject and attribute -- never the answer,
    never a synonym of it, and for `decision` never the rejected alternative,
    which would reduce recall to a coin flip.
    """
    if fact.fact_type == "decision":
        return f"What did we decide to route `{fact.subject}` through?"
    if fact.fact_type == "value":
        return (f"What value did we set `{fact.attribute}` on "
                f"`{fact.subject}` to?")
    if fact.fact_type == "owner":
        return f"Who owns `{fact.subject}`?"
    if fact.fact_type == "superseded":
        return (f"What is `{fact.attribute}` on `{fact.subject}` "
                f"currently set to?")
    raise ValueError(f"unknown fact type {fact.fact_type!r}")


# --------------------------------------------------- structural validators
# Spec 3.5, 3.6 and 4.3. These are the guarantees that should make the
# parametric gate (5.5) pass. They run against the GENERATED corpus, so a
# generator change that quietly breaks one is caught before any spend.
# Each returns a list of violations; empty means clean.


def check_answer_not_in_question(facts) -> list[str]:
    """No probe may contain its own answer, or its superseded value."""
    bad = []
    for f in facts:
        q = derive_question(f).lower()
        if f.value.lower() in q:
            bad.append(f"{f.fact_id}: answer {f.value!r} appears in its question")
        if f.stale_value and f.stale_value.lower() in q:
            bad.append(f"{f.fact_id}: stale value {f.stale_value!r} in question")
        if f.alternative and f.alternative.lower() in q:
            bad.append(f"{f.fact_id}: alternative {f.alternative!r} in question")
    return bad


def check_distractor_density(corpus_text: str, facts, min_components: int,
                             min_mentions: int) -> list[str]:
    """Every probe parameter must be shared, and every subject must be noisy.

    Without this the benchmark is degenerate: BM25 on the parameter name
    alone returns the one chunk that contains it, and nothing is measured.
    """
    import re
    bad = []
    for f in facts:
        if f.attribute:
            pattern = re.compile(
                r"`" + re.escape(f.attribute) + r"` on `([^`]+)`")
            others = {m for m in pattern.findall(corpus_text) if m != f.subject}
            if len(others) < min_components:
                bad.append(
                    f"{f.fact_id}: parameter {f.attribute!r} used by only "
                    f"{len(others)} other components (need {min_components})")
        mentions = corpus_text.count(f"`{f.subject}`")
        if mentions < min_mentions:
            bad.append(f"{f.fact_id}: subject {f.subject!r} mentioned "
                       f"{mentions} times (need {min_mentions})")
    return bad


def bullet_text_only(corpus_text: str) -> str:
    """The corpus with its HTML anchor comments removed.

    The anchors carry two UUIDs per turn -- roughly 30,000 random hex
    characters per hundred turns. A four-digit decimal answer occurs inside
    that hex by chance about once per 10,000 characters, so a uniqueness
    check run over the raw text reports collisions that no arm could ever
    exploit: nothing can answer a question from a substring of a UUID, and
    memsearch's clean_content_for_embedding() strips these comments before
    embedding anyway. The guarantee is therefore about bullet text.
    """
    import re
    return re.sub(r"<!--.*?-->", "", corpus_text, flags=re.S)


def check_answer_unique_in_corpus(corpus_text: str, facts) -> list[str]:
    """An answer may appear only in its own planted bullet.

    A second occurrence means an arm can retrieve the wrong chunk and still
    score, which would credit retrieval that did not happen.

    Matched on word boundaries over bullet text only (see bullet_text_only),
    so `271` inside `4271` is not a collision and neither is a digit run
    inside a UUID.
    """
    import re
    text = bullet_text_only(corpus_text)
    bad = []
    for f in facts:
        allowed = 2 if f.fact_type == "superseded" else 1
        n = len(re.findall(r"\b" + re.escape(str(f.value)) + r"\b", text))
        if n > allowed:
            bad.append(f"{f.fact_id}: answer {f.value!r} appears {n} times "
                       f"in bullet text (allowed {allowed})")
    return bad


def check_depth_bands(probe_depths: dict, bands, tolerance: float) -> list[str]:
    """Every probe must sit within `tolerance` of one of the target bands."""
    bad = []
    for pid, d in sorted(probe_depths.items()):
        if not any(abs(d - b) <= tolerance for b in bands):
            bad.append(f"{pid}: measured depth {d:.4f} is not within "
                       f"{tolerance} of any band {bands}")
    return bad


def check_superseded_originals_in_window(original_depths: dict,
                                         fit_fraction: float,
                                         margin: float) -> list[str]:
    """Both statements of every superseded pair must be visible to BOTH arms.

    Spec 4.3. replay_trunc retains the most recent `fit_fraction` of the
    history, so anything below depth (1 - fit_fraction) is invisible to it.
    An original planted there would let truncated replay return the current
    value without ever seeing the superseded one -- a zero stale_rate it did
    not earn, scored as recency reasoning when it is really just
    invisibility. The subset would then flatter truncated replay on the exact
    failure mode it exists to detect.

    The threshold is computed from the MEASURED fit_fraction rather than a
    fixed band, so the guarantee survives L overshooting its ~1.2M target.
    """
    floor = (1.0 - fit_fraction) + margin
    return [
        f"{pid}: superseded original at depth {d:.4f} is below the "
        f"visibility floor {floor:.4f} (fit_fraction={fit_fraction:.4f}, "
        f"margin={margin}) -- truncated replay would never see it"
        for pid, d in sorted(original_depths.items()) if d <= floor
    ]


# ------------------------------------------------- scoring and aggregation

import statistics


def prompt_tokens(usage) -> int:
    """Total tokens the request actually carried.

    Spec 7.1. `input_tokens` alone is the UNCACHED REMAINDER; on the replay
    arm it is a two-digit number next to a 960,000-token prompt. Charting it
    would rank whole-history replay as the cheapest arm.
    """
    return (usage.input_tokens
            + usage.cache_read_input_tokens
            + usage.cache_creation_input_tokens)


def score_mechanical(answer: str, expected: str) -> bool:
    """Substring match. Published alongside the judge, NEVER primary.

    WS6a section 6 measured why: a model that names the expected value inside
    an explicit refusal ("there is no 4271 in the history") is credited. The
    coined, unique values here should behave far better than WS6a's file
    paths did, but the failure mode is identical in shape.
    """
    return bool(expected) and str(expected).lower() in (answer or "").lower()


def score_stale_mentioned(answer: str, stale: str) -> bool:
    """True when the superseded value appears ANYWHERE in the answer.

    This is NOT the stale rate. It fires on the correct answer "we changed
    kerfquota from 3319 to 8123", which mentions the superseded value while
    committing to the current one. Published for auditability; never the
    headline.
    """
    return bool(stale) and str(stale).lower() in (answer or "").lower()


# Back-compat alias: the orchestrator writes the `stale` column with this.
score_stale = score_stale_mentioned


def score_stale_as_answer(answer: str, stale: str, judge_correct: bool) -> bool:
    """True when the answer COMMITS to the superseded value.

    This is the stale rate, and it needs the judge. A substring match cannot
    tell "it is 3319" from "it was 3319, now 8123" -- the first is the
    recency failure the superseded subset exists to detect, the second is a
    correct answer that happens to narrate the change.

    Measured, not theorised: on the first superseded probes to complete, all
    three had the superseded value in the answer AND were judged correct,
    because the model explained the change. Reporting the substring rate as
    the stale rate would have published a 100% recency-failure rate for an
    arm that got every one of them right.

    This is WS6a section 6's lesson in a new place: there, `any_file_hit`
    credited answers that named the expected file inside an explicit refusal.
    Same shape -- a string can appear in a semantic role the matcher cannot
    see. The judge is primary; the mechanical column is published beside it.
    """
    return bool(stale) and (not judge_correct) and \
        str(stale).lower() in (answer or "").lower()


def summarise(rows: list[dict]) -> dict:
    """Per-arm aggregates. One dict per arm, keyed by arm name."""
    out: dict[str, dict] = {}
    arms = []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        correct = sum(1 for r in sub if r["judge_correct"])
        billed = round(sum(r["cost_billed_usd"] for r in sub), 6)
        sup = [r for r in sub if r["probe_set"] == "superseded"]
        out[arm] = {
            "arm": arm,
            "n": len(sub),
            "judge_accuracy": correct / len(sub),
            "median_prompt_tokens": statistics.median(
                r["prompt_tokens"] for r in sub),
            "total_billed_usd": billed,
            "total_list_usd": round(sum(r["cost_list_usd"] for r in sub), 6),
            "median_billed_usd": statistics.median(
                r["cost_billed_usd"] for r in sub),
            # None, never 0.0: a zero here reads as "free" on a chart when it
            # actually means "spent money and got nothing right".
            "cost_per_correct_usd": round(billed / correct, 6) if correct else None,
            # The judge-gated definition: committed to the superseded value,
            # not merely mentioned it. See score_stale_as_answer.
            "stale_rate": (
                sum(1 for r in sup
                    if r.get("stale") and not r["judge_correct"]) / len(sup)
                if sup else None),
            "stale_mentioned_rate": (sum(1 for r in sup if r.get("stale"))
                                     / len(sup) if sup else None),
            "turn_cap_rate": sum(1 for r in sub if r["hit_turn_cap"]) / len(sub),
        }
    return out
