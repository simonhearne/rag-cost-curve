"""WS4b Tier 0d: gold refresh on the audited error bucket.

Design: results/recall-vs-quality.md
"""

import json

import pytest

from benchlib import config
from benchlib.recall_quality import gold
from benchlib.recall_quality import gold_runner as t0d
# Cost/governor helpers are shared machinery, not code-retrieval-specific --
# WS4b lives here in recall_quality, not in code_retrieval.
from benchlib.code_retrieval.bench import BudgetExceeded, CostGovernor, Usage


def test_ws4b_t0d_pins_match_the_spec_table():
    """Spec §4 is the single source of truth for all nine -- the ninth
    (added in review) is a licensing pin, not a tuning knob: the threshold
    above which a verbatim span shared with a row's own passages must be
    redacted before propose_reason/verify_reason are committed."""
    assert config.WS4B_T0D_N_ROWS == 107
    assert config.WS4B_T0D_PROPOSE_MODEL == "claude-sonnet-5"
    assert config.WS4B_T0D_VERIFY_MODEL == "claude-opus-5"
    assert config.WS4B_T0D_TIER_CAP_USD == 12.00
    assert config.WS4B_T0D_UPHOLD_CEILING == 0.90
    assert config.WS4B_T0D_CONCORDANCE_MIN == 0.60
    assert config.WS4B_T0D_MIN_CHANGES == 10
    assert config.WS4B_T0D_FALSE_CORRECT_MIN == 0.05
    assert config.WS4B_T0D_REDACT_MIN_CHARS == 20


def test_tier_cap_sits_inside_the_ws4b_governor():
    """Spec §6: Tier 0d spends inside WS4b's budget, never beside it."""
    assert config.WS4B_T0D_TIER_CAP_USD < config.WS4B_BUDGET_USD


def test_ws4b_t0d_changes_no_existing_pin():
    assert config.WS4B_BUDGET_USD == 90.00
    assert config.WS4B_MIN_GOLD_CHARS == 3
    assert config.WS4B_PRIMARY_STRATUM == "gold_in_exact_top10"
    assert config.WS4B_CHANGEPOINT_BOOT_N == 10_000
    assert config.WS4_TOP_K == 10
    assert config.SEED == 42


def test_ws4b_cache_is_distinct_from_the_frozen_ws4_cache():
    """data/ws4_cache/passages.parquet is frozen input; Tier 0d gets its own."""
    assert config.ws4b_cache() == config.DATA_DIR / "ws4b_cache"
    assert config.ws4b_cache() != config.ws4_cache()
    assert config.ws4b_dir() == config.RESULTS_DIR / "ws4b"


def test_verdicts_are_exactly_the_four_the_spec_names():
    assert gold.VERDICTS == (
        "keep", "augment", "replace", "unanswerable_from_corpus")


@pytest.mark.parametrize("verdict,old,new,ok", [
    ("keep", '["71%"]', '[]', True),
    ("keep", '["71%"]', '["78%"]', False),          # keep may not change golds
    ("augment", '["Austria-Hungary"]', '["Austria-Hungary", "Germany"]', True),
    ("augment", '["Austria-Hungary"]', '["Germany"]', False),   # must not drop
    ("augment", '["Austria-Hungary"]', '["Austria-Hungary"]', False),  # must add
    ("replace", '["Ronald Reagan"]', '["Joe Biden"]', True),
    ("replace", '["78%"]', '[]', False),            # must supply a list
    ("replace", '["78%"]', '["78%"]', False),       # must actually differ
    ("unanswerable_from_corpus", '["78%"]', '[]', True),
    ("unanswerable_from_corpus", '["78%"]', '["71%"]', False),
    ("nonsense", '["78%"]', '[]', False),
])
def test_validate_verdict_enforces_each_operation(verdict, old, new, ok):
    assert (gold.validate_verdict(verdict, old, new) is None) is ok


def test_apply_verdict_keep_and_unanswerable_leave_golds_untouched():
    """unanswerable_from_corpus is a COVERAGE statement, not a gold change."""
    assert gold.apply_verdict("keep", '["71%"]', '[]') == ["71%"]
    assert gold.apply_verdict(
        "unanswerable_from_corpus", '["71%"]', '[]') == ["71%"]


def test_apply_verdict_augment_unions_and_replace_overwrites():
    got = gold.apply_verdict(
        "augment", '["Austria-Hungary"]', '["Austria-Hungary", "Germany"]')
    assert sorted(got) == ["Austria-Hungary", "Germany"]
    assert gold.apply_verdict("replace", '["78%"]', '["71%"]') == ["71%"]


def test_augment_never_drops_an_existing_alias():
    """WS4's published scoring must stay derivable from an augmented row."""
    got = gold.apply_verdict("augment", '["a", "b"]', '["b", "c"]')
    assert set(got) >= {"a", "b"}


def test_validate_rejects_a_correction_nobody_could_match():
    """gold.alias_usable drops aliases under WS4B_MIN_GOLD_CHARS (3 normalised
    chars). MEASURED: norm_alias("71%") == "71", two chars, NOT usable -- so
    the audit's own water-percentage row cannot be rescued by correcting 78% to
    71%, because neither is matchable. Pinning that here because it is a real
    limit on what Tier 0d can reach, not a quirk of one fixture."""
    assert gold.validate_verdict(
        "replace", '["78%"]', '["71%"]') is not None
    assert gold.validate_verdict(
        "replace", '["78%"]', '["71 percent"]') is None
    assert gold.validate_verdict(
        "replace", '["Ronald Reagan"]', '["Joe Biden"]') is None


def test_operation_of_reports_whether_the_gold_actually_moved():
    assert gold.operation_of("keep", '["a"]', '[]') == "unchanged"
    assert gold.operation_of(
        "unanswerable_from_corpus", '["a"]', '[]') == "unchanged"
    assert gold.operation_of("augment", '["a"]', '["a", "b"]') == "changed"
    assert gold.operation_of("replace", '["a"]', '["b"]') == "changed"


def test_apply_verdict_raises_on_unknown_verdict():
    """Finding 1: apply_verdict must not silently default to destructive behavior."""
    with pytest.raises(ValueError, match="unknown verdict"):
        gold.apply_verdict("nonsense", '["a"]', '["b"]')


def test_operation_of_raises_on_unknown_verdict():
    """Finding 1: operation_of must not silently default to any behavior."""
    with pytest.raises(ValueError, match="unknown verdict"):
        gold.operation_of("nonsense", '["a"]', '["b"]')


def test_apply_verdict_replace_raises_on_falsy_new_golds():
    """Finding 2: empty gold list makes every answer wrong; reject loudly."""
    with pytest.raises(ValueError):
        gold.apply_verdict("replace", '["a"]', None)
    with pytest.raises(ValueError):
        gold.apply_verdict("replace", '["a"]', "")


def test_apply_verdict_augment_treats_falsy_new_golds_as_empty():
    """Finding 2: augmenting with nothing is a no-op (no additions made)."""
    result = gold.apply_verdict("augment", '["a"]', None)
    assert result == ["a"]
    result = gold.apply_verdict("augment", '["a"]', "")
    assert result == ["a"]


def test_apply_verdict_keep_and_unanswerable_with_falsy_new_golds():
    """Finding 2: these verdicts legitimately receive no payload."""
    assert gold.apply_verdict("keep", '["a"]', None) == ["a"]
    assert gold.apply_verdict("keep", '["a"]', "") == ["a"]
    assert gold.apply_verdict(
        "unanswerable_from_corpus", '["a"]', None) == ["a"]
    assert gold.apply_verdict(
        "unanswerable_from_corpus", '["a"]', "") == ["a"]


def test_in_primary_mirrors_ws4b_tier0s_definition():
    """scripts/recall-quality/ws4b_tier0.py:137 -- gt_presence_wb & ~degenerate & main."""
    texts = ["Joe Biden was sworn in as president in January 2021."]
    assert gold.in_primary('["Joe Biden"]', texts, "main") is True
    assert gold.in_primary('["Ronald Reagan"]', texts, "main") is False
    assert gold.in_primary('["Joe Biden"]', texts, "sanity") is False


def test_in_primary_excludes_degenerate_golds():
    """A gold with no matchable alias cannot be scored by presence at all, so
    it is out of the stratum however well the passages answer the question.
    MEASURED: '71%' normalises to '71' (two chars) and is degenerate, which is
    why a percentage answer can never enter the primary stratum."""
    assert gold.in_primary('["--"]', ["anything at all"], "main") is False
    assert gold.in_primary(
        '["71%"]', ["about 71% water"], "main") is False


def test_stratum_membership_can_move_when_a_gold_is_corrected():
    """THE reason spec 3.5 was amended: membership is a function of the golds,
    so a refresh moves the population as well as the scores."""
    gt = {"q1": ["Joe Biden was sworn in as president in January 2021."]}
    before = gold.stratum_membership(
        [{"qid": "q1", "subset": "main"}], {"q1": '["Ronald Reagan"]'}, gt)
    after = gold.stratum_membership(
        [{"qid": "q1", "subset": "main"}], {"q1": '["Joe Biden"]'}, gt)
    assert before == {"q1": False}
    assert after == {"q1": True}


def test_stratum_membership_is_keyed_by_qid_and_covers_every_row():
    gt = {"q1": ["Joe Biden was sworn in"], "q2": ["nothing relevant"]}
    got = gold.stratum_membership(
        [{"qid": "q1", "subset": "main"}, {"qid": "q2", "subset": "main"}],
        {"q1": '["Joe Biden"]', "q2": '["absent name here"]'}, gt)
    assert set(got) == {"q1", "q2"}


def _diff(qid, verdict, uphold, op="changed"):
    return {"qid": qid, "verdict": verdict, "uphold": uphold, "operation": op}


def test_gate_g1_measures_concordance_against_the_provisional_labels():
    """G1 is the independent read on WS4b's audit -- its labels are still
    pending a human spot-check and were applied by the model that proposed
    the hypothesis they test."""
    labels = {f"q{i}": "stale_gold" for i in range(10)}
    rows = [_diff(f"q{i}", "replace", i < 7) for i in range(10)]
    got = gold.gate_g1(rows, labels)
    assert got["n_stale_gold"] == 10
    assert got["n_upheld_change"] == 7
    assert got["concordance"] == 0.7
    assert got["pass"] is True
    assert gold.gate_g1([_diff(f"q{i}", "replace", i < 5)
                              for i in range(10)], labels)["pass"] is False


def test_gate_g1_ignores_rows_not_labelled_stale_gold():
    labels = {"q0": "stale_gold", "q1": "model_error"}
    got = gold.gate_g1(
        [_diff("q0", "replace", True), _diff("q1", "replace", True)], labels)
    assert got["n_stale_gold"] == 1


def test_gate_g2_uphold_rate_is_over_proposed_changes_only():
    """A `keep` is not something the verifier upholds, so it is not in the
    denominator (spec 5 G2)."""
    rows = ([_diff(f"c{i}", "replace", True) for i in range(9)]
            + [_diff("c9", "replace", False)]
            + [_diff(f"k{i}", "keep", True, op="unchanged") for i in range(50)])
    got = gold.gate_g2(rows)
    assert got["n_proposed_changes"] == 10
    assert got["uphold_rate"] == 0.9
    assert got["pass"] is True


def test_gate_g2_fails_when_the_verifier_upholds_everything():
    rows = [_diff(f"c{i}", "replace", True) for i in range(12)]
    got = gold.gate_g2(rows)
    assert got["uphold_rate"] == 1.0
    assert got["pass"] is False


def test_gate_g2_is_not_evaluable_below_the_floor():
    """Below WS4B_T0D_MIN_CHANGES the rate is noise, not evidence."""
    got = gold.gate_g2([_diff(f"c{i}", "replace", True) for i in range(3)])
    assert got["evaluable"] is False
    assert got["pass"] is True          # not evaluable never fails the run


def test_gate_g4_catches_a_leaked_answer_arm_or_label():
    """The blindness claim has to be mechanical, not asserted in prose. A
    forbidden term inside the QUESTION or GOLDS field is a real leak -- those
    are exactly the two columns a withheld arm answer or label could be
    threaded into by a bug (the passages are scanned by the structural check
    instead; see test_gate_g4_catches_a_structural_mismatch below)."""
    ok = gold.gate_g4(
        [{"qid": 1, "question": "who won?", "golds": '["71%"]',
          "expected": "p", "actual": "p"}],
        forbidden=["sq8_np512", "126", "stale_gold"])
    assert ok["pass"] is True and ok["n_term_violations"] == 0

    bad = gold.gate_g4(
        [{"qid": 1, "question": "who won?", "golds": '["126"]',
          "expected": "p", "actual": "p"}],
        forbidden=["sq8_np512", "126", "stale_gold"])
    assert bad["pass"] is False and bad["n_term_violations"] == 1
    assert "126" in bad["examples"]


def test_gate_g4_matches_on_word_boundaries_not_substrings():
    """'126' must not fire on '1260'; the repo has been burned by substring
    matching before (WS4's presence metric, spec 4)."""
    row = {"qid": 1, "question": "a total of 1260 seats", "golds": "[]",
           "expected": "p", "actual": "p"}
    assert gold.gate_g4([row], forbidden=["126"])["pass"] is True


def test_gate_g4_catches_a_structural_mismatch():
    """A withheld column could reach the prompt without ever touching the
    question or golds fields the term scan can see -- e.g. threaded into the
    passages, or via a bug in how the prompt gets assembled. The structural
    reconstruction check is what actually catches that: `actual` must be
    byte-identical to `expected`, whatever the divergence is."""
    row = {"qid": 1, "question": "who won?", "golds": '["71%"]',
           "expected": 'Q:who won? G:["71%"] P:clean passage',
           "actual": 'Q:who won? G:["71%"] P:clean passage, model said 126'}
    got = gold.gate_g4([row], forbidden=["stale_gold"])
    assert got["n_structural_mismatches"] == 1
    assert got["pass"] is False


def test_gate_g4_skips_structural_check_without_a_fresh_call():
    """A row recovered from checkpoint has no `actual` prompt this
    invocation -- `fill` is deterministic and free, so there is nothing to
    compare a resumed row against, and it is checked on the term scan alone."""
    row = {"qid": 1, "question": "who won?", "golds": '["71%"]',
           "expected": "Q:who won? P:clean passage", "actual": None}
    got = gold.gate_g4([row], forbidden=["stale_gold"])
    assert got["n_structural_mismatches"] == 0
    assert got["pass"] is True


def test_branch_r12_is_exclusive_and_exhaustive():
    assert gold.branch_r12("K1", "K3")["branch"] == "R1"
    assert gold.branch_r12("K1", "K1")["branch"] == "R2"


def test_branch_r3_fires_on_the_false_correct_rate():
    """R3 is INDEPENDENT of R1/R2 and may fire alongside either."""
    flips = ["correct_to_error"] * 5 + ["error_to_correct"] * 95
    got = gold.branch_r3(flips)
    assert got["n_correct_to_error"] == 5
    assert got["false_correct_rate"] == 0.05
    assert got["fires"] is True
    assert gold.branch_r3(["error_to_correct"] * 100)["fires"] is False


def test_redact_verbatim_passage_text_replaces_a_long_verbatim_quote():
    """This repo's hard rule: never commit corpus data. A model's free-text
    reasoning naturally quotes the passage it read; that quote must not
    survive into a committed CSV."""
    blob = "The treaty was signed in the spring of 1919 after months of talks."
    reason = ('The passage says "the treaty was signed in the spring of '
              '1919 after months of talks" so the recorded answer is wrong.')
    got = gold.redact_verbatim_passage_text(reason, blob)
    assert gold.REDACTED_PLACEHOLDER in got
    assert "spring of 1919" not in got
    assert "so the recorded answer is wrong." in got     # surrounding text survives


def test_redact_verbatim_passage_text_leaves_short_coincidences_alone():
    """A shared word or number is not a leak; only a real quote is. Below
    the threshold nothing is touched -- this is what makes G4's own
    global-forbidden-term false positives (fix round 1) a different failure
    mode from this one, not evidence this redaction is too aggressive."""
    blob = "the population reached 6 million by 2010"
    reason = "the recorded answer of 6 is plausible for this era"
    assert gold.redact_verbatim_passage_text(reason, blob) == reason


def test_redact_verbatim_passage_text_handles_empty_and_no_match():
    assert gold.redact_verbatim_passage_text("", "some passage") == ""
    assert gold.redact_verbatim_passage_text("a reason", "") == "a reason"
    clean = "nothing here matches the corpus at all"
    assert gold.redact_verbatim_passage_text(clean, "totally unrelated text"
                                                  ) == clean


def test_redact_verbatim_passage_text_redacts_multiple_disjoint_spans():
    blob = ("the population of the region reached six million residents "
            "and the capital was relocated to the northern province in 1958")
    reason = ("Note: the population of the region reached six million "
              "residents ### completely unrelated interstitial reasoning ### "
              "the capital was relocated to the northern province according "
              "to the passage.")
    got = gold.redact_verbatim_passage_text(reason, blob, min_len=15)
    assert got.count(gold.REDACTED_PLACEHOLDER) >= 2
    assert "completely unrelated interstitial reasoning" in got
    assert "according to the passage." in got


def test_redact_verbatim_passage_text_is_case_sensitive_by_default():
    """Tier 0d's exact behaviour, kept as the default because
    tier0d_gold_diff.csv is merged and pinned in data/MANIFEST.json. A model
    quotes a passage with its original casing, so this was sufficient there.
    """
    blob = "Elena Begins Transition In The Departed, Which Sets Up Season Four."
    reason = 'the note says "elena begins transition in the departed" so it is reachable'
    assert gold.redact_verbatim_passage_text(reason, blob) == reason


def test_redact_verbatim_passage_text_case_insensitive_catches_a_retyped_quote():
    """A HUMAN re-types a quote in lower case, and the case-sensitive default
    then leaves the corpus span standing (this is the real qid 2144 shape from
    the decline spot-check). The match is found on a folded copy; the span is
    still cut out of the ORIGINAL text."""
    blob = "Elena Begins Transition In The Departed, Which Sets Up Season Four."
    reason = 'the note says "elena begins transition in the departed" so it is reachable'
    got = gold.redact_verbatim_passage_text(reason, blob,
                                                 case_insensitive=True)
    assert gold.REDACTED_PLACEHOLDER in got
    assert "elena begins transition" not in got
    assert "so it is reachable" in got


def test_fold_preserves_length_so_indices_stay_aligned():
    """The folded copy indexes the original, so the fold MUST NOT change
    length. U+0130 lowercases to two characters and is therefore left alone
    rather than silently shifting every later index."""
    for s in ("\u0130stanbul", "\u00c4\u00d6\u00dc \u1e9e", "plain ASCII", ""):
        assert len(gold._fold(s)) == len(s)
    assert gold._fold("\u0130stanbul") == "\u0130stanbul"   # untouched, not lowered
    assert gold._fold("\u00c4\u00d6\u00dc") == "\u00e4\u00f6\u00fc"


def test_every_gate_and_branch_row_is_csv_shaped():
    labels = {"q0": "stale_gold"}
    rows = [gold.gate_g1([_diff("q0", "replace", True)], labels),
            gold.gate_g2([_diff("c", "replace", True)]),
            gold.gate_g4(
                [{"qid": "q0", "question": "what color?", "golds": "[]",
                  "expected": "x", "actual": "x"}], forbidden=["test"]),
            gold.branch_r12("K1", "K1"),
            gold.branch_r3(["error_to_correct"])]
    for r in rows:
        json.dumps(r)


def test_gate_g4_consumes_rows_once_not_twice():
    """gate_g4 must convert rows to a list before iterating. For a generator
    (one-shot iterable), a second iteration would find it exhausted and
    report n_rows=0 -- silently wrong, no exception."""
    rows = (r for r in [
        {"qid": 1, "question": "clean", "golds": "[]",
         "expected": "p", "actual": "p"},
        {"qid": 2, "question": "the model said 126", "golds": "[]",
         "expected": "p", "actual": "p"}])
    got = gold.gate_g4(rows, forbidden=["126"])
    assert got["n_rows"] == 2
    assert got["n_term_violations"] == 1
    assert got["pass"] is False


def test_tier_spent_counts_duplicate_lines(tmp_path):
    """A governor that cannot see a retried row's second
    charge is how WS6c went over a raised cap (see README.md)."""
    from benchlib.agent_harness import checkpoint_spent
    p = tmp_path / "t0d_checkpoint.jsonl"
    p.write_text(
        json.dumps({"qid": "q1", "arm": "propose", "cost_total_billed": 0.10}) + "\n"
        + json.dumps({"qid": "q1", "arm": "propose", "cost_total_billed": 0.10}) + "\n")
    assert t0d.tier_spent(p) == 0.20
    assert checkpoint_spent(p) == 0.10          # the bug this guards against


def test_tier_governor_is_capped_by_the_tier_not_the_workstream(tmp_path):
    gov = t0d.tier_governor(tmp_path / "none.jsonl")
    assert gov.limit == config.WS4B_T0D_TIER_CAP_USD == 12.00
    assert gov.limit < config.WS4B_BUDGET_USD


def test_charge_records_the_cost_before_the_governor_can_reject_it():
    """The call is already billed by the time charge() runs, so a rejection
    must not erase it from the row or from gov.spent."""
    gov = CostGovernor(0.0001)
    prices = {"m": {"input_usd_per_mtok": 2.0, "output_usd_per_mtok": 10.0,
                    "cache_write_1h_multiplier": 2.0, "cache_read_multiplier": 0.1}}
    row = {"cost_total_billed": 0.0}
    usage = Usage(input_tokens=100, output_tokens=20)
    with pytest.raises(BudgetExceeded):
        t0d.charge(gov, usage, prices, "m", row)
    assert row["cost_total_billed"] > 0.0       # survives the rejection
    assert gov.spent > 0.0                      # and the governor sees it


def test_charge_adds_each_cost_exactly_once():
    gov = CostGovernor(10.0)
    prices = {"m": {"input_usd_per_mtok": 2.0, "output_usd_per_mtok": 10.0,
                    "cache_write_1h_multiplier": 2.0, "cache_read_multiplier": 0.1}}
    row = {"cost_total_billed": 0.0}
    u = Usage(input_tokens=1_000_000, output_tokens=0)
    t0d.charge(gov, u, prices, "m", row)
    t0d.charge(gov, u, prices, "m", row)
    assert row["cost_total_billed"] == pytest.approx(4.0)
    assert gov.spent == pytest.approx(4.0)


def test_budget_exceeded_inside_the_verifyerror_handler_still_checkpoints(tmp_path):
    """spec §8.1. charge() inside `except VerifyError` can raise
    BudgetExceeded; the sibling `except BudgetExceeded` on the same try does
    NOT catch it, so the row was never checkpointed and tier_spent()
    under-counted money already billed."""
    ckpt = tmp_path / "ck.jsonl"
    gov = CostGovernor(0.01, spent=0.0)
    row = {"qid": 1, "cost_total_billed": 0.0, "verdict": "replace"}
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    prices = {"m": {"input_usd_per_mtok": 5.0, "output_usd_per_mtok": 25.0,
                    "cache_write_1h_multiplier": 1.0, "cache_read_multiplier": 1.0}}

    with pytest.raises(BudgetExceeded):
        try:
            raise t0d.VerifyError("exhausted", usage, "p")
        except t0d.VerifyError as exc:
            t0d.charge(gov, exc.usage, prices, "m", row)
            row.update(verdict="keep", verify_reason="verifier exhausted")
        except BudgetExceeded:
            pytest.fail("sibling handler cannot catch this -- that is the bug")

    # the money was really billed, so it must be recoverable from the row
    assert row["cost_total_billed"] > 0
    t0d.append_checkpoint(ckpt, row)
    assert t0d.tier_spent(ckpt) == row["cost_total_billed"]


def test_fill_survives_braces_in_passage_text():
    """Corpus passages carry braces; str.format would raise mid-run."""
    assert t0d.fill("P: {passages}", passages="see {{cite}} and {x}") == \
        "P: see {{cite}} and {x}"


class _Resp:
    def __init__(self, text):
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.stop_reason = "end_turn"
        self.usage = type("U", (), {"input_tokens": 100, "output_tokens": 20,
                                    "cache_creation_input_tokens": 0,
                                    "cache_read_input_tokens": 0})()


class _Msgs:
    def __init__(self, texts):
        self.texts, self.calls = list(texts), []

    def create(self, **kw):
        self.calls.append(kw)
        return _Resp(self.texts.pop(0))


class _Client:
    def __init__(self, texts):
        self.messages = _Msgs(texts)


def test_propose_returns_the_verdict_and_the_rendered_prompt():
    c = _Client([json.dumps({"verdict": "replace", "new_golds": ["71%"],
                             "reason": "corpus says 71%"})])
    got = t0d.propose(c, "Q:{question} G:{golds} P:{passages}",
                      question="how much water", golds='["78%"]',
                      passages="about 71% water")
    assert got["verdict"] == "replace" and got["new_golds"] == ["71%"]
    assert "how much water" in got["prompt"]
    assert c.messages.calls[0]["model"] == config.WS4B_T0D_PROPOSE_MODEL
    assert c.messages.calls[0]["thinking"] == {"type": "disabled"}


def test_propose_never_renders_an_answer_or_an_arm_into_the_prompt():
    """Blindness is spec 3.1's load-bearing property: the arms' answers would
    anchor the re-derivation toward whatever the models said, which given the
    one-sided row selection is exactly the failure mode."""
    c = _Client([json.dumps({"verdict": "keep", "new_golds": [],
                             "reason": "correct"})])
    got = t0d.propose(c, "Q:{question} G:{golds} P:{passages}",
                      question="q", golds='["a"]', passages="p")
    row = {"qid": 1, "question": "q", "golds": '["a"]',
           "expected": got["prompt"], "actual": got["prompt"]}
    assert gold.gate_g4([row],
                             forbidden=["sq8_np512", "126", "stale_gold"])["pass"]


def test_verify_sees_the_proposal_but_not_the_proposers_reasoning():
    c = _Client([json.dumps({"uphold": False, "reason": "corpus does not say"})])
    got = t0d.verify(c, "Q:{question} G:{golds} P:{passages} N:{proposed}",
                     question="q", golds='["78%"]', passages="p",
                     proposed=["71%"])
    assert got["uphold"] is False
    assert "corpus says 71%" not in got["prompt"]   # stage 1's reason withheld
    assert c.messages.calls[0]["model"] == config.WS4B_T0D_VERIFY_MODEL


def test_verify_escalates_max_tokens_when_opus_returns_no_text():
    """Opus runs adaptive thinking by default and max_tokens caps thinking plus
    output together; agent_harness records this killing a multi-hour run."""
    c = _Client(["", json.dumps({"uphold": True, "reason": "ok"})])
    got = t0d.verify(c, "{question}{golds}{passages}{proposed}",
                     question="q", golds='["a"]', passages="p", proposed=["b"])
    assert got["uphold"] is True
    budgets = [k["max_tokens"] for k in c.messages.calls]
    assert budgets == sorted(budgets) and len(budgets) == 2


class _VaryingResp:
    """Like _Resp but with per-call token counts, so a test can pin an
    accumulated total to a concrete sum rather than a fixed constant that
    would pass even if accumulation were broken."""

    def __init__(self, text, input_tokens, output_tokens):
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.stop_reason = "end_turn"
        self.usage = type("U", (), {"input_tokens": input_tokens,
                                    "output_tokens": output_tokens,
                                    "cache_creation_input_tokens": 0,
                                    "cache_read_input_tokens": 0})()


class _VaryingMsgs:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def create(self, **kw):
        self.calls.append(kw)
        return self.responses.pop(0)


class _VaryingClient:
    def __init__(self, responses):
        self.messages = _VaryingMsgs(responses)


def test_verify_raises_verifyerror_and_sums_usage_when_every_attempt_is_unusable():
    """The reviewer flagged this as the spot most likely to silently regress:
    total.add() must fire on EVERY attempt, including the ones that never
    produce a parseable payload, so a rejected run's real spend is still
    chargeable to the governor."""
    c = _VaryingClient([
        _VaryingResp("", 100, 20),           # empty text block (thinking ate it)
        _VaryingResp("not json", 300, 40),   # text present but unparseable
    ])
    with pytest.raises(t0d.VerifyError) as exc_info:
        t0d.verify(c, "{question}{golds}{passages}{proposed}",
                  question="q", golds='["a"]', passages="p", proposed=["b"])
    assert len(c.messages.calls) == 2          # both budgets were attempted
    usage = exc_info.value.usage
    assert usage.input_tokens == 400           # 100 + 300, not one attempt's
    assert usage.output_tokens == 60           # 20 + 40
    # The prompt of a billed-but-unparseable call must still be recoverable,
    # so a caller can hand it to gate G4 -- a call that was made and paid for
    # is a call whose blindness still needs asserting.
    assert exc_info.value.prompt == 'q["a"]p["b"]'


def test_both_prompts_are_committed_and_carry_no_corpus_text():
    for name in ("tier0d_propose_prompt.txt", "tier0d_verify_prompt.txt"):
        p = config.ws4b_dir() / name
        assert p.exists(), f"{name} must be committed"
        assert "{passages}" in p.read_text()


_PASSAGES_PARQUET = config.DATA_DIR / "ws4_cache" / "passages.parquet"
_GT_10M = config.DATA_DIR / "gt_10m_top100.npz"
_RUNS_CHECKPOINT = config.DATA_DIR / "ws4_cache" / "runs_checkpoint.jsonl"
_GOLD_DIFF_CSV = config.ws4b_dir() / "tier0d_gold_diff.csv"
_SPOTCHECK_CSV = config.ws4b_dir() / "decline_spotcheck_20.csv"

# The committed artifacts that carry free text written while reading corpus
# passages, and -- per file -- where THAT file's rows got their passages from.
# Redacting against the wrong row's or the wrong arm's passages would neither
# remove a real leak nor detect one, so the source is part of the fixture.
#
#   gt10m  Tier 0d read the 10M ground-truth top-10.
#   ref    the decline audit (and its spot-check) read the ten passages the
#          REFERENCE ARM sq8_np512 actually retrieved.
#
# `case_insensitive` is False for Tier 0d -- its reasons are model-written and
# quote with the passage's own casing, and that file is merged and pinned --
# and True for the spot-check, whose notes are human-written and re-type
# quotes in lower case. See results/recall-vs-quality.md.
_CORPUS_TEXT_GUARD_CASES = [
    pytest.param(_GOLD_DIFF_CSV, ("propose_reason", "verify_reason"),
                 "gt10m", False, id="tier0d_gold_diff"),
    pytest.param(_SPOTCHECK_CSV, ("justification", "your_note"),
                 "ref", True, id="decline_spotcheck_20"),
    pytest.param(config.ws4d_dir() / "eligibility_labels.csv",
                 ("reason_a", "reason_b"), "gt10m", True,
                 id="ws4d_eligibility_labels"),
    pytest.param(config.ws4d_dir() / "eligibility_spotcheck_20.csv",
                 ("your_note",), "gt10m", True,
                 id="ws4d_eligibility_spotcheck_20"),
]


def _blobs_gt10m(qids):
    """qid -> the 10M ground-truth top-10 passages, as Tier 0d saw them."""
    import numpy as np
    import pandas as pd

    from benchlib.config import WS4_TOP_K

    tbl = pd.read_parquet(_PASSAGES_PARQUET).set_index("row")
    idx = np.load(_GT_10M)["ws4_indices"]
    out = {}
    for q in qids:
        rows = [int(x) for x in idx[int(q)][:WS4_TOP_K] if int(x) >= 0]
        out[int(q)] = " ".join(str(tbl.at[x, "text"]) for x in rows
                               if x in tbl.index)
    return out


def _blobs_reference_arm(qids):
    """qid -> the ten passages sq8_np512 actually retrieved, as the decline
    auditor and the human spot-checker saw them."""
    import pandas as pd

    from benchlib.config import WS4_TOP_K

    want = {int(q) for q in qids}
    ids = {}
    with open(_RUNS_CHECKPOINT) as f:
        for line in f:
            r = json.loads(line)
            if r["arm"] == "sq8_np512" and int(r["qid"]) in want:
                ids[int(r["qid"])] = [int(x) for x in json.loads(r["retrieved_ids"])]
    assert not (want - set(ids)), f"no sq8_np512 retrieval for {want - set(ids)}"
    tbl = pd.read_parquet(_PASSAGES_PARQUET).set_index("row")
    return {q: " ".join(str(tbl.at[x, "text"]) for x in rows[:WS4_TOP_K]
                        if x in tbl.index)
            for q, rows in ids.items()}


_BLOB_SOURCES = {"gt10m": _blobs_gt10m, "ref": _blobs_reference_arm}


@pytest.mark.parametrize("csv_path,fields,blob_source,case_insensitive",
                         _CORPUS_TEXT_GUARD_CASES)
def test_committed_free_text_has_no_verbatim_passage_text(
        csv_path, fields, blob_source, case_insensitive):
    """This repo's hard rule: never commit corpus data. This is the actual fix
    for the corpus-text leak found in review (fix round 2), extended to cover
    every committed artifact carrying free text written while reading
    passages. It runs against whatever is COMMITTED, so a future run that
    reintroduces an unredacted quote fails here rather than quietly landing
    in the repo again.

    WARNING: this needs the gitignored passages cache, so in a public
    checkout and in CI it SKIPS rather than fails. The hard rule is enforced
    by this test only on a machine that holds the corpus.
    """
    import pandas as pd

    needed = [_PASSAGES_PARQUET, csv_path,
              _GT_10M if blob_source == "gt10m" else _RUNS_CHECKPOINT]
    missing = [p for p in needed if not p.exists()]
    if missing:
        pytest.skip(f"needs local corpus cache / completed run: "
                    f"{[p.name for p in missing]}")

    df = pd.read_csv(csv_path)
    blobs = _BLOB_SOURCES[blob_source](df.qid)

    violations = []
    for _, r in df.iterrows():
        blob = blobs[int(r.qid)]
        for field in fields:
            text = r[field]
            if pd.isna(text) or not text:
                continue
            redacted = gold.redact_verbatim_passage_text(
                str(text), blob, case_insensitive=case_insensitive)
            if redacted != str(text):
                violations.append((int(r.qid), field, str(text)))
    assert not violations, (
        f"{len(violations)} committed field(s) in {csv_path.name} still carry "
        f"verbatim passage text: {violations[:3]}")


def test_the_spot_check_sample_is_the_seeded_sample():
    """results/recall-vs-quality.md's 18/20 is only meaningful if the 20 rows are the
    pre-agreed random sample and not a convenience selection. The committed
    spot-check must reproduce from the committed labels under the project
    seed -- provenance checked, not asserted."""
    import pandas as pd

    labelled = config.ws4b_dir() / "decline_labelled.csv"
    if not (labelled.exists() and _SPOTCHECK_CSV.exists()):
        pytest.skip("needs decline_labelled.csv and the committed spot-check")
    lab = pd.read_csv(labelled)
    got = sorted(int(q) for q in pd.read_csv(_SPOTCHECK_CSV).qid)
    expect = sorted(lab.sample(n=20, random_state=config.SEED).qid.tolist())
    assert got == expect
