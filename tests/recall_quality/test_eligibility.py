"""WS4d -- a legitimate r*, or a pre-registered negative.

Design: results/recall-vs-quality.md
"""

import numpy as np
import pytest

from benchlib import config
from benchlib.recall_quality import eligibility


def test_ws4d_pins_match_the_spec_table():
    """Spec §7 is the single source of truth for all ten."""
    assert config.WS4D_LABEL_MODEL_A == "claude-sonnet-5"
    assert config.WS4D_LABEL_MODEL_B == "claude-opus-5"
    assert config.WS4D_N_QUESTIONS == 450
    assert config.WS4D_KAPPA_MIN == 0.60
    assert config.WS4D_SPOTCHECK_N == 20
    assert config.WS4D_SPOTCHECK_MAX_DISAGREE == 3
    assert config.WS4D_DECLINE_SHARE_MAX == 0.80
    assert config.WS4D_LADDER_MAX_GAP == 0.035
    assert config.WS4D_PHASE1_CAP_USD == 25.00
    assert config.WS4D_PHASE3_CAP_USD == 25.00


def test_both_tier_caps_sit_inside_the_ws4b_governor():
    """A runaway phase must not consume the workstream's budget."""
    assert (config.WS4D_PHASE1_CAP_USD + config.WS4D_PHASE3_CAP_USD
            <= config.WS4B_BUDGET_USD)


def test_ws4d_dirs_are_distinct_and_never_write_into_ws4_or_ws4b():
    assert config.ws4d_dir().name == "ws4d"
    assert config.ws4d_dir() != config.ws4b_dir()
    assert "ws4d_cache" in str(config.ws4d_cache())


def test_label_vocabulary_is_exactly_the_spec_five():
    assert eligibility.LABELS == ("eligible", "ambiguous", "wrong_gold",
                           "stale_gold", "unanswerable")
    assert eligibility.INELIGIBLE == frozenset(eligibility.LABELS[1:])
    # spec §4.1: `unanswerable` is a SENSITIVITY, not part of the primary drop
    assert eligibility.PRIMARY_DROP == frozenset({"ambiguous", "wrong_gold", "stale_gold"})
    assert "unanswerable" not in eligibility.PRIMARY_DROP


def test_validate_label_rejects_anything_off_the_list():
    assert eligibility.validate_label("eligible") is None
    assert eligibility.validate_label("ambiguous") is None
    assert "unknown label" in eligibility.validate_label("ambiguous_question")
    assert "unknown label" in eligibility.validate_label("")


def test_adjudicate_both_eligible():
    assert eligibility.adjudicate("eligible", "eligible") == ("eligible", "eligible")


def test_adjudicate_agreed_ineligible_keeps_the_class():
    assert eligibility.adjudicate("stale_gold", "stale_gold") == ("ineligible", "stale_gold")


def test_adjudicate_ineligible_but_different_classes_is_disputed():
    got = eligibility.adjudicate("ambiguous", "wrong_gold")
    assert got == ("ineligible", "ineligible_class_disputed")


def test_adjudicate_split_on_the_binary_call_resolves_to_ELIGIBLE():
    """Spec §3.3: the conservative direction. It minimises the drop, and
    dropping is the move that inflates accuracy -- so the rule biases
    AGAINST this document's own hypothesis. Both orders, to prove the rule
    is symmetric and not an artefact of which labeller is 'A'."""
    assert eligibility.adjudicate("eligible", "stale_gold") == ("eligible", "eligible")
    assert eligibility.adjudicate("stale_gold", "eligible") == ("eligible", "eligible")


def test_adjudicate_rejects_an_invalid_label_rather_than_guessing():
    with pytest.raises(ValueError):
        eligibility.adjudicate("eligible", "not_a_label")


def test_cohens_kappa_is_1_for_perfect_agreement():
    a = ["eligible", "ambiguous", "stale_gold", "eligible"]
    assert eligibility.cohens_kappa(a, list(a)) == pytest.approx(1.0)


def test_cohens_kappa_is_0_for_chance_agreement():
    """Two raters who agree exactly as often as chance predicts score 0."""
    a = ["eligible"] * 50 + ["ambiguous"] * 50
    b = ["eligible"] * 25 + ["ambiguous"] * 25 + ["eligible"] * 25 + ["ambiguous"] * 25
    assert eligibility.cohens_kappa(a, b) == pytest.approx(0.0, abs=1e-9)


def test_cohens_kappa_is_negative_when_raters_systematically_disagree():
    a = ["eligible", "ambiguous"] * 10
    b = ["ambiguous", "eligible"] * 10
    assert eligibility.cohens_kappa(a, b) < 0


def test_cohens_kappa_handles_a_degenerate_single_category():
    """Both raters say 'eligible' for everything: observed agreement is 1 and
    so is expected, which is 0/0. Return 1.0 rather than nan -- perfect
    agreement is perfect agreement, and a nan would silently fail the gate."""
    a = ["eligible"] * 20
    assert eligibility.cohens_kappa(a, list(a)) == pytest.approx(1.0)


def test_gate_e3_gates_the_BINARY_decision_not_the_five_classes():
    """Spec §3.4: gate what you use. Two raters who agree on every
    eligible/ineligible call but differ on WHICH ineligible class must PASS."""
    a = ["ambiguous"] * 10 + ["eligible"] * 10
    b = ["stale_gold"] * 10 + ["eligible"] * 10
    got = eligibility.gate_e3(a, b)
    assert got["kappa_binary"] == pytest.approx(1.0)
    assert got["kappa_5class"] < 1.0
    assert got["passed"] is True
    assert got["floor"] == config.WS4D_KAPPA_MIN


def test_gate_e3_fails_below_the_floor():
    a = ["eligible", "ambiguous"] * 10
    b = ["ambiguous", "eligible"] * 10
    got = eligibility.gate_e3(a, b)
    assert got["passed"] is False
    assert got["n"] == 20


def _two_segment(x, tau, a, b1, b2):
    return a + b1 * np.minimum(x, tau) + b2 * np.maximum(x - tau, 0.0)


def test_fit_changepoint_recovers_a_clean_synthetic_knee():
    x = np.linspace(0.70, 1.00, 25)
    y = _two_segment(x, 0.85, 0.1, 0.8, 0.0)
    got = eligibility.fit_changepoint(x, y, grid=400)
    assert got["tau"] == pytest.approx(0.85, abs=0.01)
    assert got["b1"] == pytest.approx(0.8, abs=0.05)
    assert got["b2"] == pytest.approx(0.0, abs=0.05)


def test_fit_line_recovers_a_straight_line():
    x = np.linspace(0.7, 1.0, 20)
    got = eligibility.fit_line(x, 0.25 * x + 0.3)
    assert got["slope"] == pytest.approx(0.25, abs=1e-9)
    assert got["sse"] == pytest.approx(0.0, abs=1e-18)


def test_two_segments_never_fit_worse_than_one():
    """A straight line is a two-segment fit with b1 == b2, so SSE cannot rise.
    Guards against a grid that excludes the optimum."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.7, 1.0, 11)
    y = 0.2 * x + rng.normal(0, 0.01, 11)
    assert eligibility.fit_changepoint(x, y)["sse"] <= eligibility.fit_line(x, y)["sse"] + 1e-12


def test_bootstrap_uses_ONE_shared_question_draw_per_replicate():
    """WS4b §6.1: arms are paired on questions, so each replicate draws ONE
    question index and applies it to every arm. Drawing per-arm would break
    the pairing and understate the CI.

    The fixture makes this decisive. Recall is a fixed constant per arm, so x
    never moves; accuracy is the SAME per-question column for every arm. Under
    a shared draw all three arm means are equal in every replicate, so the
    fitted slope is exactly 0 every time and its CI collapses to [0, 0]. Under
    per-arm draws the means would differ and the CI would open up.
    """
    rng = np.random.default_rng(1)
    col = rng.random(200)
    acc = np.column_stack([col, col, col])
    rec = np.column_stack([np.full(200, 0.70), np.full(200, 0.80),
                           np.full(200, 0.90)])
    got = eligibility.bootstrap_changepoint(rec, acc, n_boot=50, seed=42)
    assert got["slope_ci_lo"] == pytest.approx(0.0, abs=1e-9)
    assert got["slope_ci_hi"] == pytest.approx(0.0, abs=1e-9)


def test_bootstrap_is_deterministic_under_the_pinned_seed():
    rng = np.random.default_rng(2)
    rec, acc = rng.random((60, 5)), rng.random((60, 5))
    a = eligibility.bootstrap_changepoint(rec, acc, n_boot=40, seed=42)
    b = eligibility.bootstrap_changepoint(rec, acc, n_boot=40, seed=42)
    assert a == b


def test_bootstrap_slope_matches_bootstrap_changepoints_own_slope():
    """bootstrap_slope is split out of bootstrap_changepoint precisely so a
    caller can get the same line-fit CI WITHOUT fitting a changepoint (WS4e
    fix round 1, S1/S7). Its numbers must agree with the slope
    bootstrap_changepoint already computes internally."""
    rng = np.random.default_rng(3)
    rec, acc = rng.random((80, 4)), rng.random((80, 4))
    cp = eligibility.bootstrap_changepoint(rec, acc, n_boot=50, seed=42)
    sl = eligibility.bootstrap_slope(rec, acc, n_boot=50, seed=42)
    assert sl["slope"] == cp["slope"]
    assert sl["slope_ci_lo"] == cp["slope_ci_lo"]
    assert sl["slope_ci_hi"] == cp["slope_ci_hi"]


def test_bootstrap_slope_uses_one_shared_question_draw_per_replicate():
    """Same fixture and reasoning as the changepoint version above."""
    rng = np.random.default_rng(1)
    col = rng.random(200)
    acc = np.column_stack([col, col, col])
    rec = np.column_stack([np.full(200, 0.70), np.full(200, 0.80),
                           np.full(200, 0.90)])
    got = eligibility.bootstrap_slope(rec, acc, n_boot=50, seed=42)
    assert got["slope_ci_lo"] == pytest.approx(0.0, abs=1e-9)
    assert got["slope_ci_hi"] == pytest.approx(0.0, abs=1e-9)


def test_bootstrap_slope_is_deterministic_under_the_pinned_seed():
    rng = np.random.default_rng(2)
    rec, acc = rng.random((60, 5)), rng.random((60, 5))
    a = eligibility.bootstrap_slope(rec, acc, n_boot=40, seed=42)
    b = eligibility.bootstrap_slope(rec, acc, n_boot=40, seed=42)
    assert a == b


def test_branch_n1_needs_a_narrow_ci_that_clears_both_endpoints():
    x = np.array([0.70, 0.80, 0.90, 1.00])
    boot = {"ci_lo": 0.84, "ci_hi": 0.87, "width": 0.03,
            "slope_ci_lo": 0.1, "slope_ci_hi": 0.3}
    assert eligibility.knee_branch_n(boot, x) == "N1"


def test_branch_n2_when_the_ci_touches_an_endpoint_even_if_narrow():
    x = np.array([0.70, 0.80, 0.90, 1.00])
    boot = {"ci_lo": 0.70, "ci_hi": 0.73, "width": 0.03,
            "slope_ci_lo": 0.1, "slope_ci_hi": 0.3}
    assert eligibility.knee_branch_n(boot, x) == "N2"


def test_branch_n3_when_the_single_slope_ci_includes_zero():
    """N3 outranks N2: 'recall does not predict quality anywhere' is a
    stronger statement than 'we could not locate the knee'."""
    x = np.array([0.70, 0.80, 0.90, 1.00])
    boot = {"ci_lo": 0.72, "ci_hi": 0.98, "width": 0.26,
            "slope_ci_lo": -0.05, "slope_ci_hi": 0.30}
    assert eligibility.knee_branch_n(boot, x) == "N3"


def test_gate_f1_fires_on_the_real_ws4b_analogue():
    """Spec §4.4: after dropping the artifact questions, 982 of 1038 wrong
    rows across 11 arms are declines. F1 is EXPECTED to fire."""
    got = eligibility.gate_f1(n_declines=982, n_errors=56)
    assert got["decline_share"] == pytest.approx(0.9461, abs=1e-4)
    assert got["fired"] is True
    assert got["curve_is"] == "decline-behaviour"


def test_gate_f1_quiet_when_errors_still_dominate():
    got = eligibility.gate_f1(n_declines=100, n_errors=400)
    assert got["fired"] is False
    assert got["curve_is"] == "answer-quality"


def test_gate_f1_on_a_perfect_stratum_does_not_divide_by_zero():
    got = eligibility.gate_f1(n_declines=0, n_errors=0)
    assert got["decline_share"] == 0.0
    assert got["fired"] is False


def test_gate_l1_passes_on_an_evenly_spaced_ladder():
    got = eligibility.gate_l1([0.80, 0.83, 0.86, 0.89, 0.92, 0.95, 0.97])
    assert got["max_gap"] == pytest.approx(0.03, abs=1e-9)
    assert got["passed"] is True


def test_gate_l1_fails_on_the_current_11_arm_ladder():
    """Spec §2.4: three gaps exceed 0.035 today, the widest 0.0514."""
    current = [0.7625, 0.7837, 0.7975, 0.8209, 0.8625, 0.9138,
               0.9246, 0.9634, 0.9674, 0.9677, 0.9905]
    got = eligibility.gate_l1(current)
    assert got["passed"] is False
    # 0.0513 from these 4-dp recalls; 0.0514 at full precision (spec §2.4).
    assert got["max_gap"] == pytest.approx(0.0513, abs=5e-5)
    # the three arms below 0.80 are outside the band and are not ladder defects
    assert got["in_band"] == 7


def test_gate_l1_only_measures_gaps_inside_the_0_80_to_0_97_band():
    """An arm at 0.55 is outside the band; the gap up to 0.80 must not be
    counted as a ladder failure."""
    got = eligibility.gate_l1([0.55, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90,
                        0.92, 0.94, 0.96])
    assert got["passed"] is True
    assert got["in_band"] == 9


class _FakeResp:
    def __init__(self, text, usage):
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.usage = usage
        self.stop_reason = "end_turn"


class _FakeMessages:
    def __init__(self, text):
        self._text, self.calls = text, []

    def create(self, **kw):
        self.calls.append(kw)
        usage = type("U", (), {"input_tokens": 100, "output_tokens": 10,
                               "cache_creation_input_tokens": 0,
                               "cache_read_input_tokens": 0})()
        return _FakeResp(self._text, usage)


class _FakeClient:
    def __init__(self, text):
        self.messages = _FakeMessages(text)


def test_label_returns_the_parsed_label_and_the_exact_prompt_sent():
    from benchlib.recall_quality import eligibility_runner as ws4d_runner
    client = _FakeClient('{"label": "stale_gold", "reason": "corpus says 2010"}')
    got = ws4d_runner.label(client, "Q={question} G={golds} P={passages}",
                            question="when did it snow", golds='["2009"]',
                            passages="the corpus text", model="claude-sonnet-5")
    assert got["label"] == "stale_gold"
    assert got["reason"] == "corpus says 2010"
    assert got["prompt"] == "Q=when did it snow G=[\"2009\"] P=the corpus text"
    assert client.messages.calls[0]["model"] == "claude-sonnet-5"


def test_label_downgrades_an_unparseable_response_to_eligible():
    """Never fabricate an ineligible label. An unreadable response must not
    silently drop a question from the benchmark -- `eligible` is the
    conservative fallback, matching §3.3's adjudication direction, and the
    reason records what happened."""
    from benchlib.recall_quality import eligibility_runner as ws4d_runner
    client = _FakeClient("not json at all")
    got = ws4d_runner.label(client, "{question}{golds}{passages}",
                            question="q", golds="[]", passages="p",
                            model="claude-sonnet-5")
    assert got["label"] == "eligible"
    assert "unparseable" in got["reason"]


def test_label_downgrades_an_off_vocabulary_label_to_eligible():
    from benchlib.recall_quality import eligibility_runner as ws4d_runner
    client = _FakeClient('{"label": "ambiguous_question", "reason": "x"}')
    got = ws4d_runner.label(client, "{question}{golds}{passages}",
                            question="q", golds="[]", passages="p",
                            model="claude-sonnet-5")
    assert got["label"] == "eligible"
    assert "unknown label" in got["reason"]


def test_tier_spent_sums_every_raw_line_including_retries(tmp_path):
    """Same defect ws4b_gold_runner.tier_spent guards: a row retried after a
    crash writes a second line with the same qid, and a dict-keyed sum would
    make that real charge invisible."""
    from benchlib.recall_quality import eligibility_runner as ws4d_runner
    p = tmp_path / "ck.jsonl"
    p.write_text('{"qid": 1, "cost_total_billed": 0.5}\n'
                 '{"qid": 1, "cost_total_billed": 0.25}\n')
    assert ws4d_runner.tier_spent(p) == 0.75


def test_gate_e1_passes_when_every_prompt_matches_its_reconstruction():
    rows = [{"qid": 1, "model": "a", "expected": "P1", "actual": "P1"},
            {"qid": 2, "model": "b", "expected": "P2", "actual": "P2"}]
    got = eligibility.gate_e1(rows)
    assert got["n_checked"] == 2 and got["n_mismatched"] == 0
    assert got["passed"] is True


def test_gate_e1_fails_on_a_single_byte_of_drift():
    rows = [{"qid": 1, "model": "a", "expected": "P1", "actual": "P1 "}]
    got = eligibility.gate_e1(rows)
    assert got["n_mismatched"] == 1 and got["passed"] is False


def test_gate_e1_counts_but_does_not_fail_on_checkpoint_recovered_rows():
    """A row recovered from checkpoint made no fresh call this invocation, so
    there is no actual prompt to compare. That is not a blindness failure --
    but it must be COUNTED and reported, or a fully-resumed run would show a
    vacuous pass."""
    rows = [{"qid": 1, "model": "a", "expected": "P1", "actual": None},
            {"qid": 2, "model": "b", "expected": "P2", "actual": "P2"}]
    got = eligibility.gate_e1(rows)
    assert got["n_uncompared"] == 1 and got["n_checked"] == 1
    assert got["passed"] is True


def test_both_label_prompts_are_committed_and_carry_no_corpus_text():
    for name in ("label_prompt_a.txt", "label_prompt_b.txt"):
        p = config.ws4d_dir() / name
        assert p.exists(), f"{name} must be committed"
        body = p.read_text()
        assert "{passages}" in body and "{question}" in body and "{golds}" in body
        for forbidden in ("recall", "arm", "judge", "declin", "presence"):
            assert forbidden not in body.lower(), (
                f"{name} mentions {forbidden!r} -- the labeller must not learn "
                f"that arms, recall or judging exist")


def test_gate_e4_passes_at_the_threshold_and_fails_above_it():
    """Spec §3.4: redo if MORE THAN 3 of 20 disagree. Exactly 3 passes."""
    assert eligibility.gate_e4(["agree"] * 17 + ["disagree"] * 3)["passed"] is True
    assert eligibility.gate_e4(["agree"] * 16 + ["disagree"] * 4)["passed"] is False


def test_gate_e4_requires_the_pinned_sample_size():
    got = eligibility.gate_e4(["agree"] * 19)
    assert got["passed"] is False
    assert "expected 20" in got["error"]


def test_gate_e4_rejects_an_unrecognised_verdict_rather_than_counting_it_as_agreement():
    with pytest.raises(ValueError):
        eligibility.gate_e4(["agree"] * 19 + ["probably?"])
