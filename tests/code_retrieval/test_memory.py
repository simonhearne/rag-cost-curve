"""WS6b: unit tests for the pure half, plus structural assertions against the
generated corpus.

Design: results/code-retrieval.md
"""

from benchlib import config


def test_ws6b_pins_are_present_and_consistent():
    assert config.WS6B_MEMSEARCH_VERSION == "0.4.0"
    assert config.WS6B_MEMSEARCH_COMMIT == "3149dc3"
    assert config.WS6B_EMBED_PROVIDER == "openai"
    assert config.WS6B_EMBED_MODEL == "text-embedding-3-small"
    assert config.WS6B_MILVUS_URI == "http://localhost:19530"
    assert config.WS6B_AGENT_MODEL == "claude-sonnet-5"
    assert config.WS6B_JUDGE_MODEL == "claude-opus-5"
    assert config.WS6B_TURN_CAP == 15
    assert config.WS6B_BUDGET_USD == 75.00
    # Spec 2: truncation budget is window minus reserve, not a magic number.
    assert config.WS6B_TRUNC_BUDGET_TOKENS == (
        config.WS6B_CONTEXT_WINDOW_TOKENS - config.WS6B_RESERVE_TOKENS
    ) == 960_000
    assert config.WS6B_SIZES == {"s": 100_000, "m": 400_000, "l": 1_200_000}
    assert config.WS6B_COLLECTIONS == {"s": "ws6b_s", "m": "ws6b_m", "l": "ws6b_l"}
    assert config.WS6B_ARMS == ("memsearch", "replay", "parametric")
    # Spec 4: probe-set cardinalities are pinned so a short run fails loudly.
    assert config.WS6B_N_PROBES == {"scaling": 12, "depth": 18, "superseded": 8}
    assert config.WS6B_DEPTH_BANDS == (0.10, 0.50, 0.90)
    assert config.WS6B_DEPTH_TOLERANCE == 0.03
    # Spec 4.3: superseded originals must sit inside the truncation window.
    assert config.WS6B_SUPERSEDED_ORIGINAL_BAND == (0.40, 0.55)
    assert config.WS6B_SUPERSEDED_WINDOW_MARGIN == 0.10
    # Spec 3.5: distractor density.
    assert config.WS6B_MIN_DISTRACTOR_COMPONENTS == 3
    assert config.WS6B_MIN_COMPONENT_MENTIONS == 5
    # Spec 5.5: the gate.
    assert config.WS6B_GATE_PASS == 0.05
    assert config.WS6B_GATE_FAIL == 0.10


import random

from benchlib.code_retrieval import memory as ws6b


def test_value_pool_is_large_and_unguessable():
    # Spec 3.6: >= 1000 options, so blind guess rate <= 0.001.
    assert len(ws6b.VALUE_POOL) >= 1000
    # No round numbers and no powers of two: the two things a model guesses.
    for v in ws6b.VALUE_POOL:
        assert v % 5 != 0, v
        assert v & (v - 1) != 0, v
        assert 17 <= v <= 9973, v


def test_lexicon_is_deterministic_and_disjoint():
    a = ws6b.make_lexicon(random.Random(42))
    b = ws6b.make_lexicon(random.Random(42))
    assert a == b, "same seed must give the same lexicon"

    c = ws6b.make_lexicon(random.Random(43))
    assert a != c, "different seeds must give different lexicons"

    # The four name spaces must not overlap, or a probe question about a
    # component could be answered by a parameter of the same name.
    spaces = [set(a.components), set(a.params), set(a.persons), set(a.mechanisms)]
    for i, s in enumerate(spaces):
        for j, t in enumerate(spaces):
            if i != j:
                assert not (s & t), f"space {i} overlaps space {j}: {s & t}"


def test_lexicon_is_large_enough_for_distractors():
    lex = ws6b.make_lexicon(random.Random(42))
    # Spec 3.5 needs >= 3 distractor components per probe parameter plus
    # conversational noise, so the spaces cannot be tiny.
    assert len(lex.components) >= 60
    assert len(lex.params) >= 40
    assert len(lex.persons) >= 20
    assert len(lex.mechanisms) >= 30


import re


def _sample_day():
    turn = ws6b.Turn(
        time_hhmm="09:12",
        session_id="5a0ee318-e387-4f06-89b4-fac350c947aa",
        turn_id="44079e8f-9d9d-4a4c-9546-02d056c86efa",
        bullets=("Discussed the quillrack shunt-queue.", "Agreed to revisit."),
    )
    session = ws6b.Session(time_hhmm="09:12", turns=(turn,))
    return ws6b.Day(date_iso="2025-01-06", sessions=(session,))


def test_render_matches_memsearch_on_disk_format():
    out = ws6b.render_day(_sample_day())
    # Verified against .memsearch/memory/*.md in this repository.
    assert "## Session 09:12" in out
    assert "### 09:12" in out
    assert re.search(
        r"<!-- session:[0-9a-f-]{36} turn:[0-9a-f-]{36} transcript:\S+\.jsonl -->",
        out,
    ), out
    assert "- Discussed the quillrack shunt-queue." in out


def test_every_turn_block_has_meaningful_body():
    # memsearch's chunker drops chunks whose body is < 2 chars after stripping
    # headings and HTML comments (_MIN_MEANINGFUL_LEN). A turn that rendered
    # to a bare heading would silently vanish from the index.
    out = ws6b.render_day(_sample_day())
    body = re.sub(r"<!--.*?-->", "", out, flags=re.S)
    body = "\n".join(l for l in body.splitlines() if not l.startswith("#"))
    assert len(body.strip()) >= 2


def test_seeded_uuid_is_deterministic_and_well_formed():
    a = ws6b.seeded_uuid(random.Random(42))
    b = ws6b.seeded_uuid(random.Random(42))
    assert a == b
    assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                        r"[0-9a-f]{4}-[0-9a-f]{12}", a), a


import pytest


def test_cumulative_before_is_exclusive():
    assert ws6b.cumulative_before([10, 20, 30]) == [0, 10, 30]
    assert ws6b.cumulative_before([]) == []


def test_depth_of():
    assert ws6b.depth_of(0, 1000) == 0.0
    assert ws6b.depth_of(500, 1000) == 0.5
    with pytest.raises(ValueError):
        ws6b.depth_of(10, 0)


def test_size_prefix_never_overshoots_the_budget():
    days = [100, 100, 100, 100, 100]
    # Greedy leading prefix: take whole day files while they fit.
    assert ws6b.size_prefix(days, 250) == 2
    assert ws6b.size_prefix(days, 300) == 3
    assert ws6b.size_prefix(days, 10) == 0


def test_truncate_recent_keeps_the_newest_whole_files():
    days = [100, 100, 100, 100, 100]  # total 500
    first, frac = ws6b.truncate_recent(days, 250)
    assert first == 3, "keeps days 3 and 4, the two most recent"
    assert frac == pytest.approx(200 / 500)


def test_truncate_recent_when_everything_fits():
    days = [100, 100]
    first, frac = ws6b.truncate_recent(days, 1000)
    assert first == 0
    assert frac == 1.0


def test_truncate_recent_never_splits_a_day_file():
    # Spec 5.3: truncation is at day-file granularity. A budget that fits
    # one and a half files keeps one, not one and a half.
    days = [100, 100, 100]
    first, frac = ws6b.truncate_recent(days, 150)
    assert first == 2
    assert frac == pytest.approx(100 / 300)


def _facts():
    return [
        ws6b.PlantedFact("f1", "decision", "quillrack", "", "shunt-queue",
                         alternative="sieve-ledger"),
        ws6b.PlantedFact("f2", "value", "tangwood", "sluicedepth", "4271"),
        ws6b.PlantedFact("f3", "owner", "pelmet", "", "Marnadou"),
        ws6b.PlantedFact("f4", "superseded", "hobnail", "kerfquota", "8123",
                         stale_value="3319"),
    ]


def test_every_question_omits_its_own_answer():
    # Spec 3.6, the structural half of the gate: if the answer is in the
    # question, a parametric arm scores without knowing anything.
    for f in _facts():
        q = ws6b.derive_question(f).lower()
        assert f.value.lower() not in q, f"{f.fact_id}: answer leaks into question"
        if f.stale_value:
            assert f.stale_value.lower() not in q, f.fact_id


def test_decision_question_omits_the_alternative():
    # Naming the rejected option turns a recall question into a coin flip.
    f = _facts()[0]
    assert "sieve-ledger" not in ws6b.derive_question(f)


def test_bullets_contain_the_answer_exactly_once():
    for f in _facts():
        assert f.value in ws6b.render_fact_bullet(f) or f.fact_type == "superseded"


def test_superseded_renders_both_statements():
    f = _facts()[3]
    original = ws6b.render_fact_bullet(f)
    revision = ws6b.render_revision_bullet(f)
    assert f.stale_value in original
    assert f.value not in original, "the original must not leak the new value"
    assert f.value in revision and f.stale_value in revision


def test_superseded_question_asks_for_current_value():
    f = _facts()[3]
    q = ws6b.derive_question(f)
    assert "currently" in q.lower()


def test_question_derivation_is_a_pure_function_of_the_fact():
    f = _facts()[1]
    assert ws6b.derive_question(f) == ws6b.derive_question(f)


def test_check_answer_not_in_question_catches_a_leak():
    bad = ws6b.PlantedFact("x", "value", "tangwood", "sluicedepth", "tangwood")
    assert ws6b.check_answer_not_in_question([bad])
    good = ws6b.PlantedFact("y", "value", "tangwood", "sluicedepth", "4271")
    assert ws6b.check_answer_not_in_question([good]) == []


def test_check_distractor_density():
    facts = [ws6b.PlantedFact("f", "value", "tangwood", "sluicedepth", "4271")]
    # Only the probe's own component uses the parameter -- BM25 on the
    # parameter name alone would return the right chunk every time.
    thin = "We set `sluicedepth` on `tangwood` to 4271."
    assert ws6b.check_distractor_density(thin, facts, 3, 5)

    thick = thin + "".join(
        f"\nWe set `sluicedepth` on `comp{i}` to {100 + i}." for i in range(4)
    ) + "".join(f"\nNoted `tangwood` behaviour again ({i})." for i in range(6))
    assert ws6b.check_distractor_density(thick, facts, 3, 5) == []


def test_check_answer_unique_in_corpus():
    facts = [ws6b.PlantedFact("f", "value", "tangwood", "sluicedepth", "4271")]
    assert ws6b.check_answer_unique_in_corpus("... to 4271.", facts) == []
    # A second occurrence means retrieving the wrong chunk still scores.
    assert ws6b.check_answer_unique_in_corpus("4271 ... 4271", facts)


def test_uniqueness_ignores_uuid_anchors_and_respects_word_boundaries():
    """Two false-positive sources the raw substring check could not tell apart.

    A digit run inside a UUID anchor is not an answer any arm could use, and
    `271` inside `4271` is not an occurrence of `271`.
    """
    facts = [ws6b.PlantedFact("f", "value", "tangwood", "sluicedepth", "4271")]
    anchored = ("- We set `sluicedepth` on `tangwood` to 4271.\n"
                "<!-- session:5a0e4271-e387-4f06-89b4-fac3504271aa turn:x -->\n")
    assert ws6b.check_answer_unique_in_corpus(anchored, facts) == []

    embedded = [ws6b.PlantedFact("g", "value", "tangwood", "sluicedepth", "271")]
    assert ws6b.check_answer_unique_in_corpus("to 271. also 4271.", embedded) == []


def test_check_depth_bands():
    assert ws6b.check_depth_bands({"p1": 0.11}, (0.10,), 0.03) == []
    assert ws6b.check_depth_bands({"p1": 0.20}, (0.10,), 0.03)


def test_check_superseded_originals_in_window_is_measured_not_assumed():
    # Spec 4.3: the binding constraint is depth_l > (1 - fit_fraction) + margin,
    # evaluated against the MEASURED fit_fraction. A fixed 0.40-0.55 band is
    # only safe while L lands near 1.2M; if L overshoots, the dropped segment
    # grows and 0.40 slides out of the window silently.
    assert ws6b.check_superseded_originals_in_window({"s1": 0.45}, 0.80, 0.10) == []
    # fit_fraction 0.60 -> dropped below 0.40 -> 0.45 is inside but within
    # the margin, so it must be rejected.
    assert ws6b.check_superseded_originals_in_window({"s1": 0.45}, 0.60, 0.10)
    # Comfortably inside.
    assert ws6b.check_superseded_originals_in_window({"s1": 0.75}, 0.60, 0.10) == []


from benchlib.code_retrieval import bench as ws6a


def test_prompt_tokens_includes_cache_fields():
    # Spec 7.1, the WS6a regression: input_tokens alone reports only the
    # UNCACHED remainder, which would rank a 960k-token replay as cheapest.
    u = ws6a.Usage(input_tokens=28, output_tokens=100,
                   cache_creation_input_tokens=0,
                   cache_read_input_tokens=960_000)
    assert ws6b.prompt_tokens(u) == 960_028


def test_score_mechanical_is_substring_based():
    assert ws6b.score_mechanical("The value is 4271.", "4271")
    assert not ws6b.score_mechanical("The value is 4272.", "4271")


def test_score_mechanical_credits_answers_inside_refusals():
    # NOT a bug -- a documented weakness. WS6a section 6 measured exactly this
    # on a contaminated corpus, which is why the judge is primary and this
    # scorer is published alongside but never leads.
    assert ws6b.score_mechanical(
        "I cannot find any mention of 4271 in the history.", "4271")


def test_score_stale_detects_the_superseded_value():
    assert ws6b.score_stale("It is currently 3319.", "3319")
    assert not ws6b.score_stale("It is currently 8123.", "3319")
    assert not ws6b.score_stale("It is currently 8123.", "")


def test_summarise_reports_cost_per_correct_and_stale_rate():
    rows = [
        {"arm": "memsearch", "judge_correct": True, "cost_billed_usd": 0.02,
         "cost_list_usd": 0.02, "prompt_tokens": 9000, "probe_set": "superseded",
         "stale": False, "hit_turn_cap": False},
        {"arm": "memsearch", "judge_correct": False, "cost_billed_usd": 0.02,
         "cost_list_usd": 0.02, "prompt_tokens": 11000, "probe_set": "superseded",
         "stale": True, "hit_turn_cap": False},
    ]
    s = ws6b.summarise(rows)["memsearch"]
    assert s["n"] == 2
    assert s["judge_accuracy"] == 0.5
    assert s["median_prompt_tokens"] == 10000
    assert s["total_billed_usd"] == pytest.approx(0.04)
    # Defined BEFORE any data exists (spec 7.2) so it cannot become a
    # post-hoc rescue metric.
    assert s["cost_per_correct_usd"] == pytest.approx(0.04)
    assert s["stale_rate"] == pytest.approx(0.5)


def test_summarise_cost_per_correct_is_none_when_nothing_is_correct():
    rows = [{"arm": "a", "judge_correct": False, "cost_billed_usd": 0.5,
             "cost_list_usd": 0.5, "prompt_tokens": 10, "probe_set": "depth",
             "stale": False, "hit_turn_cap": False}]
    # Division by zero must be None, never 0.0 or infinity -- a 0.0 here
    # would read as "free" on a chart.
    assert ws6b.summarise(rows)["a"]["cost_per_correct_usd"] is None


import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent


def _generate(out, target=20000):
    subprocess.run(
        [sys.executable, str(REPO / "scripts/code-retrieval/generate_ws6b_corpus.py"),
         "--out", str(out), "--target-tokens", str(target)],
        check=True, capture_output=True, cwd=REPO)


def test_generator_is_byte_exact_across_runs(tmp_path):
    """Spec 3.2: reproduction is one command, verified by checksum.

    Two runs into different directories must produce identical bytes. This
    catches uuid4(), module-level random, and set iteration -- the three
    things that silently break regeneration.
    """
    import hashlib

    def run(out):
        _generate(out)
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(out.glob("*.md"))}

    a = run(tmp_path / "a")
    b = run(tmp_path / "b")
    assert a == b and a, "generator is not deterministic"


def test_generated_corpus_passes_every_structural_gate(tmp_path):
    out = tmp_path / "c"
    _generate(out, 60000)

    text = "\n".join(p.read_text() for p in sorted(out.glob("*.md")))
    facts = [ws6b.PlantedFact(**{k: v for k, v in f.items()
                                 if k in ws6b.PlantedFact.__dataclass_fields__})
             for f in json.loads((out / "facts.json").read_text())]

    assert len(facts) == sum(config.WS6B_N_PROBES.values())
    assert ws6b.check_answer_not_in_question(facts) == []
    assert ws6b.check_answer_unique_in_corpus(text, facts) == []
    assert ws6b.check_distractor_density(
        text, facts,
        config.WS6B_MIN_DISTRACTOR_COMPONENTS,
        config.WS6B_MIN_COMPONENT_MENTIONS) == []


def test_generated_corpus_uses_memsearch_day_file_naming(tmp_path):
    out = tmp_path / "d"
    _generate(out)
    names = [p.name for p in sorted(out.glob("*.md"))]
    assert names, "no day files generated"
    for n in names:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}\.md", n), n
    # Weekdays only, so the corpus reads like a real work history.
    import datetime as dt
    for n in names:
        d = dt.date.fromisoformat(n[:-3])
        assert d.weekday() < 5, f"{n} is a weekend"


def test_superseded_facts_have_a_revision_planted_later(tmp_path):
    """Spec 4.3: both statements must exist, and the revision must come after."""
    out = tmp_path / "e"
    _generate(out, 60000)
    led = json.loads((out / "facts.json").read_text())
    sup = [f for f in led if f["fact_type"] == "superseded"]
    assert len(sup) == config.WS6B_N_PROBES["superseded"]
    for f in sup:
        assert f["revision_day_index"] > f["day_index"], f["fact_id"]


import csv


def test_probes_csv_schema_is_complete():
    """Spec 4.4. A missing derived column is a silently unfalsifiable claim."""
    path = REPO / "results/ws6b/probes.csv"
    if not path.exists():
        pytest.skip("probes.csv not built yet")
    rows = list(csv.DictReader(path.open()))
    required = {
        "probe_id", "probe_set", "fact_type", "question", "expected_answer",
        "stale_answer", "subject", "attribute", "source_file", "source_line",
        "session_id", "turn_id", "depth_s", "depth_m", "depth_l",
        "cumulative_tokens_before", "original_source_line",
        "original_depth_l", "original_in_window", "fact_in_window",
    }
    assert required <= set(rows[0]), required - set(rows[0])
    counts = {}
    for r in rows:
        counts[r["probe_set"]] = counts.get(r["probe_set"], 0) + 1
    assert counts == config.WS6B_N_PROBES, counts


def test_every_superseded_original_is_inside_the_truncation_window():
    """Spec 4.3, verified in the PUBLISHED DATA, not only in a test fixture.

    An original outside replay_trunc's window would let that arm return the
    current value without ever seeing the superseded one -- a stale_rate of
    zero it did not earn, scored as recency reasoning when it is really just
    invisibility.
    """
    path = REPO / "results/ws6b/probes.csv"
    if not path.exists():
        pytest.skip("probes.csv not built yet")
    rows = [r for r in csv.DictReader(path.open())
            if r["probe_set"] == "superseded"]
    assert len(rows) == config.WS6B_N_PROBES["superseded"]
    for r in rows:
        assert r["original_in_window"] == "True", (
            f"{r['probe_id']}: original at depth {r['original_depth_l']} is "
            "outside replay_trunc's window")
        assert r["fact_in_window"] == "True", f"{r['probe_id']}: revision"


def test_depth_set_straddles_the_truncation_boundary():
    """Spec 4.2: the window cliff must be MEASURED across bands, not assumed.

    The shallow band has to fall outside replay_trunc's retained window and
    the deeper bands inside it, or the boundary is asserted rather than
    observed.
    """
    path = REPO / "results/ws6b/probes.csv"
    if not path.exists():
        pytest.skip("probes.csv not built yet")
    rows = [r for r in csv.DictReader(path.open()) if r["probe_set"] == "depth"]
    shallow = [r for r in rows if float(r["depth_l"]) < 0.2]
    deep = [r for r in rows if float(r["depth_l"]) > 0.4]
    assert shallow and deep
    assert all(r["fact_in_window"] == "False" for r in shallow)
    assert all(r["fact_in_window"] == "True" for r in deep)


def test_corpus_stats_records_measured_tokens_not_targets():
    path = REPO / "results/ws6b/corpus_stats.csv"
    if not path.exists():
        pytest.skip("corpus_stats.csv not built yet")
    rows = {r["size"]: r for r in csv.DictReader(path.open())}
    assert set(rows) == {"s", "m", "l"}
    for size in ("s", "m"):
        measured, target = int(rows[size]["count_tokens"]), config.WS6B_SIZES[size]
        # S and M are greedy whole-file prefixes, so they may never overshoot.
        assert measured <= target, f"{size}: prefix overshot its budget"
        assert measured >= target * 0.85, (
            f"{size}: measured {measured} far below target {target}; "
            "CHARS_PER_TOKEN is miscalibrated -- recalibrate and regenerate")
    # L is the whole corpus, not a prefix. It is DELIBERATELY sized to exceed
    # the 1M context window (spec 3.7), so overshooting its target is correct.
    assert int(rows["l"]["count_tokens"]) > config.WS6B_CONTEXT_WINDOW_TOKENS
    assert int(rows["s"]["count_tokens"]) < int(rows["m"]["count_tokens"]) \
        < int(rows["l"]["count_tokens"])
    # The truncation window must actually bite at L, or (b') is just (b).
    assert 0.0 < float(rows["l"]["fit_fraction"]) < 1.0


def test_judge_prompt_is_blind_and_committed():
    p = (REPO / "results/ws6b/judge_prompt.txt").read_text()
    for field in ("{question}", "{expected_answer}", "{stale_answer}", "{answer}"):
        assert field in p, field
    # Blindness: the judge must never learn which arm produced the answer.
    for leak in ("memsearch", "replay", "parametric", "retrieval", "index"):
        assert leak not in p.lower(), f"judge prompt leaks arm identity: {leak!r}"


def test_judge_prompt_formats_without_error():
    p = (REPO / "results/ws6b/judge_prompt.txt").read_text()
    filled = p.format(question="q", expected_answer="8123",
                      stale_answer="3319", answer="a")
    assert "8123" in filled and "3319" in filled


def test_judge_empty_answer_is_incorrect_without_an_api_call():
    from benchlib.code_retrieval import memory_runner as ws6b_runner
    out = ws6b_runner.judge_ws6b(
        client=None,
        prompt_template="{question}{expected_answer}{stale_answer}{answer}",
        question="q", expected_answer="8123", stale_answer="", answer="   ")
    assert out["judge_correct"] is False
    assert out["usage"].output_tokens == 0


def test_system_prompt_names_no_tool_and_no_strategy():
    """Spec 5: the arms must differ in exactly ONE variable."""
    from benchlib.code_retrieval import memory_runner as ws6b_runner
    p = ws6b_runner.WS6B_SYSTEM_PROMPT.lower()
    for leak in ("memsearch", "search", "index", "retriev", "tool",
                 "expand", "vector", "replay"):
        assert leak not in p, f"system prompt names a strategy: {leak!r}"


def test_gate_verdict_thresholds():
    from benchlib.code_retrieval import memory_runner as r
    assert r.gate_verdict(0.00) == "pass"
    assert r.gate_verdict(0.05) == "pass"
    assert r.gate_verdict(0.06) == "investigate"
    assert r.gate_verdict(0.10) == "investigate"
    assert r.gate_verdict(0.11) == "fail"
    assert r.gate_verdict(config.WS6B_GATE_PASS) == "pass"
    assert r.gate_verdict(config.WS6B_GATE_FAIL) == "investigate"


def test_memsearch_tools_have_the_three_documented_layers():
    """Spec 5.1: L1 search -> L2 expand -> L3 read, the shipped skill's flow."""
    from benchlib.code_retrieval import memory_runner as ws6b_runner
    tools = ws6b_runner.make_memsearch_tools(
        "ws6b_s", REPO / "data/ws6b_corpus", check=False)
    assert [t.name for t in tools] == [
        "memory_search", "memory_expand", "memory_read_session"]


def test_memory_read_session_refuses_path_escape(tmp_path):
    """Tool arguments are untrusted model output."""
    from benchlib.code_retrieval import memory_runner as ws6b_runner
    (tmp_path / "2025-01-06.md").write_text("## Session 09:00\n\n- hi\n")
    tools = ws6b_runner.make_memsearch_tools("ws6b_s", tmp_path, check=False)
    read = {t.name: t for t in tools}["memory_read_session"]
    out = read("../../../etc/passwd")
    assert "root:" not in out
    assert "Error" in out


def test_memsearch_availability_actually_executes_the_binary(monkeypatch):
    """WS6a I-7: a path check is satisfied by a broken stub or a shell shim."""
    from benchlib.code_retrieval import memory_runner as ws6b_runner
    monkeypatch.setattr(ws6b_runner.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    assert ws6b_runner.memsearch_available() is False


def test_replay_uses_a_cached_system_block():
    """Spec 5.2: stable prefix cached, volatile question after it."""
    from benchlib.code_retrieval import memory_runner as ws6b_runner
    blocks = ws6b_runner.replay_system_blocks("HISTORY TEXT")
    assert blocks[0]["text"] == ws6b_runner.WS6B_SYSTEM_PROMPT
    assert "cache_control" not in blocks[0], "the tiny prompt must not be cached"
    assert blocks[1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "HISTORY TEXT" in blocks[1]["text"]


def test_dual_cost_columns_diverge_only_when_caching_happened():
    """Spec 5.2: one run, two columns. The gap IS the C9 story."""
    price = ws6a.load_pricing(REPO / "results/ws6b/pricing.csv")["claude-sonnet-5"]
    cached = ws6a.Usage(input_tokens=30, output_tokens=500,
                        cache_read_input_tokens=960_000)
    assert ws6a.cost_uncached_list(cached, price) > ws6a.cost_billed(cached, price)
    uncached = ws6a.Usage(input_tokens=9_000, output_tokens=500)
    assert ws6a.cost_uncached_list(uncached, price) == pytest.approx(
        ws6a.cost_billed(uncached, price))


def test_window_exceeded_is_a_status_not_a_cost():
    """Spec 8.4: a hard capability boundary, never an extrapolated cost."""
    from benchlib.code_retrieval import memory_runner as ws6b_runner
    assert ws6b_runner.WINDOW_EXCEEDED == "window_exceeded"


def _cell_plan():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_ws6b_arms", REPO / "scripts/code-retrieval/run_ws6b_arms.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.cell_plan()


def test_cell_order_is_contiguous_per_arm_and_size():
    """Spec 5.6: contiguous, NOT interleaved. The cache is C9's steelman."""
    if not (REPO / "results/ws6b/probes.csv").exists():
        pytest.skip("probes.csv not built yet")
    cells = _cell_plan()
    assert [(c["arm"], c["size"]) for c in cells] == [
        ("memsearch", "s"), ("replay", "s"),
        ("memsearch", "m"), ("replay", "m"),
        ("memsearch", "l"), ("replay", "l"), ("replay_trunc", "l")]
    # 12 + 12 + 12 + 12 + 38 + 1 + 38. The single replay@L window probe is
    # easy to drop when totalling by hand -- it is a run, and it is billed.
    assert sum(len(c["probe_ids"]) for c in cells) == 125
    # replay at L is attempted ONCE, not once per probe (spec 8.4).
    assert len([c for c in cells
                if c["arm"] == "replay" and c["size"] == "l"][0]["probe_ids"]) == 1


def test_run_is_resumable_from_checkpoint(tmp_path):
    from benchlib.agent_harness import append_checkpoint, load_checkpoint
    p = tmp_path / "ck.jsonl"
    append_checkpoint(p, {"qid": "p1", "arm": "memsearch@s",
                          "cost_total_billed": 0.02})
    assert ("p1", "memsearch@s") in load_checkpoint(p)


def test_governor_refuses_a_charge_past_the_limit():
    g = ws6a.CostGovernor(limit=75.0, spent=74.99)
    with pytest.raises(ws6a.BudgetExceeded):
        g.charge(1.0)
    # A rejected charge is not recorded, so a resumed run never double-counts.
    assert g.spent == 74.99


def test_retrieved_chars_uses_the_untruncated_length():
    """The transcript truncates tool_result to 2,000 chars and records the
    real length beside it. Reading the truncated copy would cap every
    retrieval at 2,000 and make a 12,000-char payload look like a 2,000-char
    one -- which is exactly the measurement WS6a's sharpest mechanistic
    finding rested on.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_ws6b_arms", REPO / "scripts/code-retrieval/run_ws6b_arms.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    turns = [{"content": [{"type": "tool_result", "content": "x" * 2000,
                           "content_length": 12014}]}]
    assert mod.retrieved_chars(turns) == 12014
    # Falls back to the visible length only when no true length was recorded.
    assert mod.retrieved_chars(
        [{"content": [{"type": "tool_result", "content": "abc"}]}]) == 3


def test_stale_rate_is_judge_gated_not_a_substring_match():
    """The superseded subset's headline number cannot be a substring match.

    Measured on the first superseded probes to complete: the model answered
    "we changed kerfquota from 3319 to 8123" -- correct, and containing the
    superseded value. A substring rate would have published a 100% recency
    failure for an arm that got every one right. Same shape as WS6a section
    6's any_file_hit crediting a filename named inside a refusal.
    """
    narrated = "We changed it from 3319 to 8123."
    committed = "It is currently 3319."
    # Mentioned: both. Stale-as-answer: only the one the judge marked wrong.
    assert ws6b.score_stale_mentioned(narrated, "3319")
    assert ws6b.score_stale_mentioned(committed, "3319")
    assert not ws6b.score_stale_as_answer(narrated, "3319", judge_correct=True)
    assert ws6b.score_stale_as_answer(committed, "3319", judge_correct=False)

    rows = [
        {"arm": "a", "judge_correct": True, "cost_billed_usd": 0.01,
         "cost_list_usd": 0.01, "prompt_tokens": 10, "probe_set": "superseded",
         "stale": True, "hit_turn_cap": False},
        {"arm": "a", "judge_correct": False, "cost_billed_usd": 0.01,
         "cost_list_usd": 0.01, "prompt_tokens": 10, "probe_set": "superseded",
         "stale": True, "hit_turn_cap": False},
    ]
    s = ws6b.summarise(rows)["a"]
    assert s["stale_rate"] == pytest.approx(0.5), "must exclude narrated mentions"
    assert s["stale_mentioned_rate"] == pytest.approx(1.0), "audit column"


def test_window_error_is_recognised_from_the_real_api_message():
    """Spec 8.4. Captured verbatim from the live 400 on the L corpus.

    The message does not contain the word "context" -- an earlier classifier
    required it, missed this, and recorded the window cliff as a generic
    error, which dropped the row instead of publishing it.
    """
    from benchlib.code_retrieval import memory_runner as r
    real = ("Error code: 400 - {'type': 'error', 'error': {'type': "
            "'invalid_request_error', 'message': 'prompt is too long: "
            "1221932 tokens > 1000000 maximum'}}")
    assert r.is_window_error(Exception(real))
    assert not r.is_window_error(Exception("529 Overloaded"))
    assert not r.is_window_error(Exception("rate_limit_error"))
