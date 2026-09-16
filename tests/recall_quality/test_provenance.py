"""WS4c: entity-bridge question construction, its filters, and its gates.

Design: results/recall-vs-quality.mdA, §3A (amendment 2026-09-11). §2/§3 are SUPERSEDED.
"""

import json

import pytest

from benchlib import config
from benchlib.recall_quality import provenance_runner as ws4c_runner
from benchlib.recall_quality import provenance as ws4c
# Cost/governor helpers are shared machinery, not code-retrieval-specific --
# WS4c lives here in recall_quality, not in code_retrieval.
from benchlib.code_retrieval.bench import BudgetExceeded


def test_ws4c_pins_match_spec_2a4_exactly():
    """Spec 2A.4 is the single source of truth for all nine. A drifted pin
    makes the committed question set unattributable to a design."""
    assert config.WS4C_BUDGET_USD == 75.00
    assert config.WS4C_BUILD_BUDGET_USD == 14.00
    assert config.WS4C_N_QUESTIONS == 600
    assert config.WS4C_QGEN_MODEL == "claude-sonnet-5"
    assert config.WS4C_QJUDGE_MODEL == "claude-opus-5"
    assert config.WS4C_MIN_BRIDGE_CHARS == 10
    assert config.WS4C_MAX_PAGE_SHARE == 0.02
    assert config.WS4C_FLOOR_MIN_SHARE == 0.30
    assert config.WS4C_CANDIDATE_TARGET == 3_000


def test_ws4c_build_budget_is_separate_from_and_below_the_total():
    """Spec 7: a low filter yield must halt the BUILD, not quietly eat the
    eval's money."""
    assert config.WS4C_BUILD_BUDGET_USD < config.WS4C_BUDGET_USD


def test_ws4c_adds_no_pin_and_changes_none():
    """This repo's hard rule. WS4c may only ADD to config.py (spec 2A.4)."""
    assert config.SEED == 42
    assert config.WS4_GEN_MODEL == "claude-sonnet-5"
    assert config.WS4_JUDGE_MODEL == "claude-opus-5"
    assert config.WS4_TOP_K == 10
    assert config.WS4_BUDGET_USD == 75.00
    assert config.WS4B_MIN_GOLD_CHARS == 3


def test_ws4c_dirs_are_distinct_from_ws4s_frozen_cache():
    """data/ws4_cache/passages.parquet is FROZEN input to WS4b Tier 0/0b."""
    assert config.ws4c_dir() == config.RESULTS_DIR / "ws4c"
    assert config.ws4c_cache() == config.DATA_DIR / "ws4c_cache"
    assert config.ws4c_cache() != config.ws4_cache()


def test_norm_title_collapses_to_alphanumeric_words():
    assert ws4c.norm_title("Daniel Boone National Forest") == "daniel boone national forest"
    assert ws4c.norm_title("Zilpo  Road!") == "zilpo road"
    assert ws4c.norm_title("---") == ""


@pytest.mark.parametrize("title,ok", [
    ("Daniel Boone National Forest", True),
    ("Zilpo Road", True),            # 10 normalised chars exactly
    ("Ohio", False),                 # one word
    ("Elm St", False),               # 6 chars, under WS4C_MIN_BRIDGE_CHARS
    ("---", False),                  # normalises to the empty string
])
def test_title_eligible_requires_multi_word_and_min_chars(title, ok):
    """Spec 2A.1, carrying forward WS4b's lesson: a short token match is a
    false positive waiting to happen (benchlib/recall_quality/gold.py)."""
    assert ws4c.title_eligible(title) is ok


def test_mentions_is_word_boundary_not_substring():
    """The WS4 metric died of raw substring matching (spec 4)."""
    assert ws4c.mentions("Daniel Boone National Forest",
                         "it runs through the Daniel Boone National Forest.")
    assert not ws4c.mentions("Boone National", "Danielboone Nationalpark")
    assert not ws4c.mentions("Zilpo Road", "the Zilpo Roadway is closed")


def test_mentions_is_case_insensitive_and_punctuation_tolerant():
    assert ws4c.mentions("Zilpo Road", "The ZILPO ROAD, a byway, is scenic.")


def test_nested_rejects_a_title_contained_in_the_other():
    """NOT in spec 2A.1 -- an addition. 'Daniel Boone' as a bridge off the
    page 'Daniel Boone National Forest' is not a hop."""
    assert ws4c.nested("Daniel Boone", "Daniel Boone National Forest")
    assert ws4c.nested("Daniel Boone National Forest", "Daniel Boone")
    assert not ws4c.nested("Zilpo Road", "Daniel Boone National Forest")


def test_bridges_proposes_by_ngram_and_returns_first_appearance_order():
    lookup = {ws4c.norm_title(t): t for t in
              ["Daniel Boone National Forest", "Cave Run Lake", "Zilpo Road"]}
    text = ("Zilpo Road is a byway near Cave Run Lake that travels through "
            "the Daniel Boone National Forest.")
    got = ws4c.bridges(text, page_title="Zilpo Road", lookup=lookup)
    assert got == ["Cave Run Lake", "Daniel Boone National Forest"]


def test_bridges_excludes_the_page_itself_and_nested_titles():
    lookup = {ws4c.norm_title(t): t for t in
              ["Daniel Boone", "Daniel Boone National Forest"]}
    text = "Daniel Boone gave his name to the Daniel Boone National Forest."
    assert ws4c.bridges(text, page_title="Daniel Boone National Forest",
                        lookup=lookup) == []


def test_bridges_is_deterministic():
    lookup = {ws4c.norm_title(t): t for t in ["Cave Run Lake", "Zilpo Road"]}
    text = "Cave Run Lake sits beside Zilpo Road."
    runs = [ws4c.bridges(text, "Morehead", lookup) for _ in range(5)]
    assert all(r == runs[0] for r in runs)


def test_norm_title_preserves_accented_base_letters_via_nfkd():
    """Finding 1: norm_title must match gold.norm_alias via NFKD decomposition.
    Accented letters normalise to base + combining mark, and [a-z0-9]+ extracts
    the base letter, enabling 'Beyoncé' and 'Beyonce' to match."""
    assert ws4c.norm_title("Beyoncé Knowles") == "beyonce knowles"
    assert ws4c.norm_title("café") == "cafe"
    assert ws4c.norm_title("São Paulo") == "sa o paulo"


def test_mentions_finds_accented_title_in_unaccented_text_and_vice_versa():
    """Finding 1: mentions() should find 'Beyoncé' in text written as 'Beyonce'
    because both normalise identically via NFKD."""
    assert ws4c.mentions("Beyoncé Knowles", "Beyonce Knowles performed.")
    assert ws4c.mentions("Beyonce Knowles", "Beyoncé Knowles performed.")


def test_title_eligible_rejects_single_accented_word():
    """Finding 2: a single accented word like 'Müller' (raw: 1 word) must fail
    title_eligible, even though its normalised form 'mu ller' has 2 tokens.
    Word count is checked on RAW title (whitespace-split), not normalised tokens."""
    assert not ws4c.title_eligible("Müller")
    assert not ws4c.title_eligible("café")
    # But two-word accented titles pass if min_chars OK
    assert ws4c.title_eligible("São Paulo")  # 2 raw words, 10 norm chars


def test_bridges_respects_max_words_boundary():
    """The max_words parameter bounds n-gram length. An 11-word title is never
    proposed (default max_words=10); a 10-word title is."""
    ten_words = " ".join(["word"] * 10)
    eleven_words = " ".join(["word"] * 11)
    lookup_10 = {ws4c.norm_title(ten_words): ten_words}
    lookup_11 = {ws4c.norm_title(eleven_words): eleven_words}
    text = ten_words + " " + eleven_words
    # With default max_words=10, only the 10-word title is proposed
    assert ws4c.bridges(text, "Morehead", lookup_10) == [ten_words]
    # With max_words=11, the 11-word title is also proposed
    assert ws4c.bridges(text, "Morehead", lookup_11, max_words=11) == [eleven_words]


def test_bridges_on_empty_text_returns_empty_list():
    """No n-grams can be extracted from empty text."""
    lookup = {ws4c.norm_title("Cave Run Lake"): "Cave Run Lake"}
    assert ws4c.bridges("", "Morehead", lookup) == []


def test_bridges_on_one_word_text_returns_empty_list():
    """N-grams require at least 2 words (the inner loop starts at n=2)."""
    lookup = {ws4c.norm_title("Cave Run Lake"): "Cave Run Lake"}
    assert ws4c.bridges("Morehead", "Morehead", lookup) == []


def test_page_cap_floors_so_the_boundary_question_is_refused():
    """2% of 600 is 12; the 13th question from one page is refused."""
    assert ws4c.page_cap(600, 0.02) == 12
    assert ws4c.page_cap(10, 0.02) == 1     # never zero, or nothing selects


def _fixture_pool():
    lookup = {ws4c.norm_title(t): t for t in
              ["Daniel Boone National Forest", "Cave Run Lake"]}
    chunks_of = {"Daniel Boone National Forest": [500, 501, 502],
                 "Cave Run Lake": [900, 901]}
    candidates = [
        (10, "Zilpo Road", "Zilpo Road runs through the Daniel Boone National Forest."),
        (11, "Zilpo Road", "Zilpo Road passes Cave Run Lake on its way north."),
        (12, "Morehead",   "Morehead lies near Cave Run Lake."),
        (13, "Nowhere",    "Nothing here names any other page."),
    ]
    return candidates, chunks_of, lookup


def test_select_pairs_records_both_gold_rows_and_is_deterministic():
    """The whole point of spec 2A: the gold chunk rows exist BEFORE the
    question does. C1 failed trying to recover them afterwards."""
    candidates, chunks_of, lookup = _fixture_pool()
    runs = [ws4c.select_pairs(candidates, chunks_of=chunks_of, lookup=lookup,
                              n_pairs=10, seed=42, cap=10) for _ in range(5)]
    assert all(r == runs[0] for r in runs)
    first = runs[0][0]
    assert first["row_a"] == 10
    assert first["bridge_title"] == "Daniel Boone National Forest"
    assert first["row_b"] in (500, 501, 502)
    assert first["page_title"] == "Zilpo Road"


def test_select_pairs_skips_a_candidate_with_no_eligible_bridge():
    candidates, chunks_of, lookup = _fixture_pool()
    got = ws4c.select_pairs(candidates, chunks_of=chunks_of, lookup=lookup,
                            n_pairs=10, seed=42, cap=10)
    assert 13 not in [p["row_a"] for p in got]


def test_select_pairs_enforces_the_cap_in_either_role():
    """Spec 2A.1's cap is applied to the POOL, counting a title as page OR
    bridge -- stronger than the spec, so G-C cannot fail late."""
    candidates, chunks_of, lookup = _fixture_pool()
    got = ws4c.select_pairs(candidates, chunks_of=chunks_of, lookup=lookup,
                            n_pairs=10, seed=42, cap=1)
    pages = [p["page_title"] for p in got]
    bridges_ = [p["bridge_title"] for p in got]
    assert pages.count("Zilpo Road") == 1
    for t in set(pages + bridges_):
        assert pages.count(t) + bridges_.count(t) <= 1


def test_select_pairs_stops_at_n_pairs():
    candidates, chunks_of, lookup = _fixture_pool()
    got = ws4c.select_pairs(candidates, chunks_of=chunks_of, lookup=lookup,
                            n_pairs=1, seed=42, cap=10)
    assert len(got) == 1


def test_select_pairs_never_pairs_a_chunk_with_itself():
    candidates = [(500, "Daniel Boone National Forest",
                   "It contains Cave Run Lake.")]
    chunks_of = {"Cave Run Lake": [500]}
    lookup = {ws4c.norm_title("Cave Run Lake"): "Cave Run Lake"}
    assert ws4c.select_pairs(candidates, chunks_of=chunks_of, lookup=lookup,
                             n_pairs=5, seed=42, cap=10) == []


def test_f1_reuses_ws4b_word_boundary_matching():
    """Spec 2A.2: F1 is benchlib.recall_quality.gold.answer_present_wb, not a new matcher.
    A gold of '12' is unmatchable and must NOT pass."""
    assert ws4c.f1_gold_in_b("Cave Run Lake", "It borders Cave Run Lake today.")
    assert not ws4c.f1_gold_in_b("Cave Run Lake", "It borders a lake today.")
    assert not ws4c.f1_gold_in_b("12", "Built in 2012 near the ridge.")


def test_f2_rejects_a_question_that_names_the_bridge():
    assert not ws4c.f2_bridge_absent(
        "Cave Run Lake", "What is the surface area of Cave Run Lake?")
    assert ws4c.f2_bridge_absent(
        "Cave Run Lake", "What is the surface area of the lake the byway passes?")


def test_hops_retrieved_counts_distinct_gold_rows_in_the_top_k():
    assert ws4c.hops_retrieved([500, 900], [900, 7, 500, 8], k=10) == 2
    assert ws4c.hops_retrieved([500, 900], [900, 7, 8, 9], k=10) == 1
    assert ws4c.hops_retrieved([500, 900], [1, 2, 3], k=10) == 0


def test_hops_retrieved_respects_k_and_skips_padding():
    assert ws4c.hops_retrieved([500, 900], [900] + [0] * 9 + [500], k=10) == 1
    assert ws4c.hops_retrieved([500, 900], [-1, -1, 500], k=10) == 1


def test_gate_a_fails_below_target_and_never_suggests_raising_the_governor():
    fail = ws4c.gate_a(412, 13.98, target=600, budget=14.00)
    assert fail["pass"] is False and fail["n_accepted"] == 412
    assert ws4c.gate_a(600, 12.70, target=600, budget=14.00)["pass"] is True


def test_gate_b_fails_when_a_gold_row_is_missing_or_the_answer_is_not_in_b():
    text_of = {900: "It borders Cave Run Lake today."}
    good = [{"qid": "q1", "row_a": 10, "row_b": 900, "gold_answer": "Cave Run Lake"}]
    assert ws4c.gate_b(good, text_of)["pass"] is True
    assert ws4c.gate_b(
        [{"qid": "q2", "row_a": 10, "row_b": None, "gold_answer": "Cave Run Lake"}],
        text_of)["pass"] is False
    assert ws4c.gate_b(
        [{"qid": "q3", "row_a": 10, "row_b": 900, "gold_answer": "Nolin Lake"}],
        text_of)["pass"] is False


def test_gate_c_counts_a_title_in_either_role():
    """G-C's denominator is the QUESTION count (spec 3A). nested() forbids
    page == bridge within one question, so a title contributes at most 1 per
    question and the Counter value is 'questions this title appears in'."""
    qs = [{"page_title": "A", "bridge_title": "B"},
          {"page_title": "C", "bridge_title": "B"},
          {"page_title": "D", "bridge_title": "E"},
          {"page_title": "F", "bridge_title": "G"}]
    assert ws4c.gate_c(qs, max_share=0.5)["top_title"] == "B"
    assert ws4c.gate_c(qs, max_share=0.5)["top_share"] == 0.5
    assert ws4c.gate_c(qs, max_share=0.5)["pass"] is True
    assert ws4c.gate_c(qs, max_share=0.4)["pass"] is False


def test_gate_c_denominator_is_questions_not_title_slots():
    """A title in EVERY question is a 1.0 share, not 0.5. The *2 denominator
    would halve it and make G-C twice as permissive as spec 3A."""
    qs = [{"page_title": "A", "bridge_title": "B"},
          {"page_title": "C", "bridge_title": "B"}]
    got = ws4c.gate_c(qs, max_share=0.9)
    assert got["top_count"] == 2
    assert got["top_share"] == 1.0
    assert got["pass"] is False


def test_gate_d_is_the_floor_gate_at_0_30():
    """Spec 3A G-D. Below the floor the mediator is pinned and the curve is
    flat for a measurement reason, not a world reason."""
    assert ws4c.gate_d([2] * 30 + [1] * 70, min_share=0.30)["pass"] is True
    assert ws4c.gate_d([2] * 29 + [1] * 71, min_share=0.30)["pass"] is False
    assert ws4c.gate_d([1] * 100, min_share=0.30)["share_hops2"] == 0.0


def test_gate_d_defaults_to_the_pinned_floor():
    assert ws4c.gate_d([2] * 50 + [1] * 50)["min_share"] == config.WS4C_FLOOR_MIN_SHARE


def test_every_gate_row_is_csv_shaped():
    rows = [ws4c.gate_a(600, 12.7), ws4c.gate_c([{"page_title": "A", "bridge_title": "B"}]),
            ws4c.gate_d([2, 1])]
    for r in rows:
        assert isinstance(r["gate"], str) and isinstance(r["pass"], bool)
        json.dumps(r)   # every value must survive a CSV/JSON round trip


def test_build_spent_counts_duplicate_lines_unlike_checkpoint_spent(tmp_path):
    """WS6c went ~$0.3 over a raised cap because the
    governor could not see spend on deduplicated checkpoint lines (see
    README.md). A retried
    candidate is billed twice and the governor must see both charges."""
    from benchlib.agent_harness import checkpoint_spent
    p = tmp_path / "build_checkpoint.jsonl"
    p.write_text(
        json.dumps({"qid": "c1", "arm": "build", "cost_total_billed": 0.10}) + "\n"
        + json.dumps({"qid": "c1", "arm": "build", "cost_total_billed": 0.10}) + "\n"
        + json.dumps({"qid": "c2", "arm": "build", "cost_total_billed": 0.05}) + "\n")
    assert ws4c_runner.build_spent(p) == 0.25
    assert checkpoint_spent(p) == 0.15          # the bug this guards against


def test_build_spent_tolerates_a_truncated_final_line(tmp_path):
    p = tmp_path / "build_checkpoint.jsonl"
    p.write_text(json.dumps({"qid": "c1", "cost_total_billed": 0.10}) + "\n"
                 + '{"qid": "c2", "cost_tot')
    assert ws4c_runner.build_spent(p) == 0.10


def test_build_spent_is_zero_when_no_checkpoint_exists(tmp_path):
    assert ws4c_runner.build_spent(tmp_path / "nope.jsonl") == 0.0


def test_build_governor_is_seeded_from_the_checkpoint_and_is_separate(tmp_path):
    p = tmp_path / "build_checkpoint.jsonl"
    p.write_text(json.dumps({"qid": "c1", "cost_total_billed": 13.99}) + "\n")
    gov = ws4c_runner.build_governor(p)
    assert gov.limit == config.WS4C_BUILD_BUDGET_USD == 14.00
    assert gov.spent == 13.99
    with pytest.raises(BudgetExceeded):
        gov.charge(0.02)


class _FakeResp:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.stop_reason = stop_reason
        self.usage = type("U", (), {"input_tokens": 100, "output_tokens": 20,
                                    "cache_creation_input_tokens": 0,
                                    "cache_read_input_tokens": 0})()
        self._request_id = "req_test"


class _FakeMessages:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResp(self.texts.pop(0))


class _FakeClient:
    def __init__(self, texts):
        self.messages = _FakeMessages(texts)


def test_fill_survives_braces_in_chunk_text():
    """Wikipedia chunks contain braces; str.format() would raise on them and
    kill a paid run mid-build."""
    out = ws4c_runner.fill("A: {chunk_a}|B: {chunk_b}",
                           chunk_a="see {{cite}} and {x}", chunk_b="plain")
    assert out == "A: see {{cite}} and {x}|B: plain"


def test_generate_question_parses_a_decline_without_inventing_one():
    """Spec 2A.2: declines are COUNTED, not retried into existence."""
    client = _FakeClient([json.dumps(
        {"decline": True, "question": "", "gold_answer": "", "reason": "no fact"})])
    got = ws4c_runner.generate_question(
        client, "A {chunk_a} B {chunk_b} {page_title} {bridge_title}",
        text_a="a", text_b="b", page_title="P", bridge_title="Q")
    assert got["decline"] is True and got["question"] == ""
    assert got["usage"].input_tokens == 100


def test_generate_question_disables_thinking_and_uses_the_pinned_model():
    client = _FakeClient([json.dumps(
        {"decline": False, "question": "q?", "gold_answer": "g", "reason": "ok"})])
    ws4c_runner.generate_question(client, "{chunk_a}{chunk_b}{page_title}{bridge_title}",
                                  text_a="a", text_b="b", page_title="P",
                                  bridge_title="Q")
    kw = client.messages.calls[0]
    assert kw["model"] == config.WS4C_QGEN_MODEL
    assert kw["thinking"] == {"type": "disabled"}


def test_answer_with_uses_ws4s_conditions_so_build_and_eval_mean_the_same():
    """Spec 2A.2: the answering model in F3/F4 is WS4C_QGEN_MODEL with WS4's
    answer prompt, thinking off -- the same conditions the eval uses."""
    client = _FakeClient(["Cave Run Lake"])
    got = ws4c_runner.answer_with(client, "sys", [], "what lake?")
    kw = client.messages.calls[0]
    assert kw["model"] == config.WS4C_QGEN_MODEL
    assert kw["thinking"] == {"type": "disabled"}
    assert kw["max_tokens"] == config.WS4_GEN_MAX_TOKENS
    assert got["answer"] == "Cave Run Lake"


def test_f3_and_f4_are_scored_by_exact_match_not_by_a_judge():
    """Spec 2A.2 is explicit: exact_match, NOT an LLM judge. A judge here
    would cost more and entangle question-building with the eval's scoring."""
    from benchlib.recall_quality.core import exact_match
    assert exact_match("Cave Run Lake", ["Cave Run Lake"]) == 1
    assert exact_match("the Cave Run Lake", ["Cave Run Lake"]) == 1  # articles
    assert exact_match("Nolin Lake", ["Cave Run Lake"]) == 0


def test_judge_validity_returns_the_verdict_and_accumulates_usage():
    client = _FakeClient([json.dumps({"valid": False, "reason": "ambiguous"})])
    got = ws4c_runner.judge_validity(
        client, "{question}{gold_answer}{chunk_a}{chunk_b}",
        question="q?", gold_answer="g", text_a="a", text_b="b")
    assert got["valid"] is False and got["reason"] == "ambiguous"
    assert client.messages.calls[0]["model"] == config.WS4C_QJUDGE_MODEL


def test_judge_validity_escalates_max_tokens_when_opus_thinks_the_budget_away():
    """Opus 5 runs adaptive thinking by default and max_tokens caps thinking
    plus output together -- agent_harness's JUDGE_MAX_TOKENS_SCHEDULE comment
    records this killing a multi-hour run."""
    client = _FakeClient(["", json.dumps({"valid": True, "reason": "ok"})])
    got = ws4c_runner.judge_validity(
        client, "{question}{gold_answer}{chunk_a}{chunk_b}",
        question="q?", gold_answer="g", text_a="a", text_b="b")
    assert got["valid"] is True
    budgets = [c["max_tokens"] for c in client.messages.calls]
    assert budgets == sorted(budgets) and len(budgets) == 2


def test_committed_prompts_exist_and_carry_no_corpus_text():
    """Licensing: prompts are templates. Chunk text is never committed."""
    for name in ("qgen_prompt.txt", "qvalidity_prompt.txt"):
        p = config.ws4c_dir() / name
        assert p.exists(), f"{name} must be committed"
        body = p.read_text()
        assert "{chunk_a}" in body and "{chunk_b}" in body


def test_judge_validity_raises_qvalidity_error_with_summed_usage_when_schedule_exhausts():
    """Regression guard for the fix-round-1 finding: two exhausted Opus
    attempts must not make their billed usage vanish -- the governor needs to
    see the full spend, not one attempt's worth, and not zero."""
    client = _FakeClient(["", "still not json"])
    with pytest.raises(ws4c_runner.QValidityError) as exc_info:
        ws4c_runner.judge_validity(
            client, "{question}{gold_answer}{chunk_a}{chunk_b}",
            question="q?", gold_answer="g", text_a="a", text_b="b")
    usage = exc_info.value.usage
    assert usage.input_tokens == 200    # 100 + 100, summed across both calls
    assert usage.output_tokens == 40    # 20 + 20
    assert len(client.messages.calls) == 2


def test_judge_validity_recovers_from_unparseable_text_on_the_first_attempt():
    """Only the empty-then-success escalation path was previously exercised;
    this covers non-empty-but-unparseable text on the first attempt too."""
    client = _FakeClient(["not json at all",
                          json.dumps({"valid": True, "reason": "ok"})])
    got = ws4c_runner.judge_validity(
        client, "{question}{gold_answer}{chunk_a}{chunk_b}",
        question="q?", gold_answer="g", text_a="a", text_b="b")
    assert got["valid"] is True
    assert len(client.messages.calls) == 2


def test_generate_question_returns_a_decline_shaped_dict_on_unparseable_json():
    """The model can return non-JSON text; this must not raise mid-build."""
    client = _FakeClient(["not json"])
    got = ws4c_runner.generate_question(
        client, "{chunk_a}{chunk_b}{page_title}{bridge_title}",
        text_a="a", text_b="b", page_title="P", bridge_title="Q")
    assert got["decline"] is True
    assert got["question"] == "" and got["gold_answer"] == ""
    assert got["usage"].input_tokens == 100


def test_generate_question_passes_through_empty_fields_on_a_non_decline_unmodified():
    """Task 8 relies on seeing an empty question/gold_answer as-is rather
    than having generate_question coerce it into something else."""
    client = _FakeClient([json.dumps(
        {"decline": False, "question": "", "gold_answer": "", "reason": "oops"})])
    got = ws4c_runner.generate_question(
        client, "{chunk_a}{chunk_b}{page_title}{bridge_title}",
        text_a="a", text_b="b", page_title="P", bridge_title="Q")
    assert got["decline"] is False
    assert got["question"] == "" and got["gold_answer"] == ""


def test_pool_size_is_a_multiple_of_the_candidate_target_not_a_gate():
    """WS4C_CANDIDATE_TARGET is 'an estimate, not a gate' (spec 2A.4). The
    pool is deliberately larger so the $14 governor, not the pool, is what
    stops the build."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_ws4c_questions", "scripts/recall-quality/build_ws4c_questions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.N_POOL >= 3 * config.WS4C_CANDIDATE_TARGET
    assert mod.N_SAMPLE_A >= 4 * mod.N_POOL


def test_f4_verdict_requires_both_single_chunk_answers_to_fail():
    """Spec 2A.2: answering from A alone AND from B alone must BOTH fail."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_ws4c_questions", "scripts/recall-quality/build_ws4c_questions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.f4_verdict("wrong", "wrong", ["Cave Run Lake"]) is True
    assert mod.f4_verdict("Cave Run Lake", "wrong", ["Cave Run Lake"]) is False
    assert mod.f4_verdict("wrong", "Cave Run Lake", ["Cave Run Lake"]) is False


def test_canary_reading_flags_a_filter_that_drops_nothing():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "build_ws4c_questions", "scripts/recall-quality/build_ws4c_questions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.canary_reading(n=15, n_dropped=15)["pass"] is True
    assert mod.canary_reading(n=15, n_dropped=12)["pass"] is True
    assert mod.canary_reading(n=15, n_dropped=6)["pass"] is False
    assert mod.canary_reading(n=15, n_dropped=0)["pass"] is False


def test_budget_exceeded_on_the_first_call_still_checkpoints_its_real_cost(
        tmp_path, monkeypatch):
    """Fix round 1, Finding 1: CostGovernor.charge() raises BEFORE recording
    (correct for ws6a, which charges before the call happens). The build
    loop inverts that -- the call is already made and billed by the time
    charge() runs -- so a BudgetExceeded, even on a candidate's very FIRST
    call, must not erase the row's cost_total_billed. If it does, the
    candidate is checkpointed with cost 0.0 (or not checkpointed at all,
    since the `if row["cost_total_billed"] > 0` guard would then skip it),
    and a resumed run starts from an under-counted governor.

    This drives run_build() against a synthetic one-candidate pool and a
    governor set up to reject the very first charge, then verifies the
    checkpoint -- not just an in-memory value -- carries the real cost, and
    that ws4c_runner.build_spent() (what a resumed run actually reads) sees
    it.
    """
    import importlib.util

    import pandas as pd

    # Shared cost machinery (not code-retrieval-specific).
    from benchlib.code_retrieval.bench import CostGovernor

    spec = importlib.util.spec_from_file_location(
        "build_ws4c_questions", "scripts/recall-quality/build_ws4c_questions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pairs_path = tmp_path / "pairs.parquet"
    pd.DataFrame([{"row_a": 1, "row_b": 2, "page_title": "P",
                   "bridge_title": "Q"}]).to_parquet(pairs_path, index=False)
    chunks_path = tmp_path / "pair_chunks.parquet"
    pd.DataFrame([{"row": 1, "title": "P", "text": "chunk a text"},
                  {"row": 2, "title": "Q", "text": "chunk b text"}]
                 ).to_parquet(chunks_path, index=False)
    monkeypatch.setattr(mod, "PAIRS", pairs_path)
    monkeypatch.setattr(mod, "PAIR_CHUNKS", chunks_path)

    checkpoint_path = tmp_path / "build_checkpoint.jsonl"
    monkeypatch.setattr(ws4c_runner, "BUILD_CHECKPOINT", checkpoint_path)

    # Limit set below the cost of a single call (100 in / 20 out tokens at
    # claude-sonnet-5 pricing = $0.0004), so generate_question's OWN charge
    # -- the first call a candidate makes -- is the one that trips it.
    monkeypatch.setattr(ws4c_runner, "build_governor",
                        lambda: CostGovernor(0.0001, spent=0.0))
    fake_client = _FakeClient([json.dumps(
        {"decline": False, "question": "q?", "gold_answer": "a",
         "reason": ""})])
    monkeypatch.setattr(ws4c_runner, "make_client", lambda: fake_client)

    mod.run_build(limit_new=1)

    assert checkpoint_path.exists()
    lines = [json.loads(l) for l in checkpoint_path.read_text().splitlines() if l]
    assert len(lines) == 1
    row = lines[0]
    assert row["outcome"] == "aborted"
    assert row["cost_total_billed"] == 0.0004     # NOT 0.0 -- the fix
    assert ws4c_runner.build_spent(checkpoint_path) == 0.0004
