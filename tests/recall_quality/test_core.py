"""WS4: unit tests for the pure half (scoring, selection rules, bootstrap,
pre-registered classification).

Design: results/recall-vs-quality.md
"""

import numpy as np
import pytest

from benchlib import config
from benchlib.recall_quality import core


def test_ws4_pins_are_present_and_consistent():
    assert config.WS4_SCALE == "10m"
    assert config.WS4_TOP_K == 10
    assert config.WS4_GEN_MODEL == "claude-sonnet-5"
    assert config.WS4_JUDGE_MODEL == "claude-opus-5"
    assert config.WS4_GEN_MODEL != config.WS4_JUDGE_MODEL
    assert config.WS4_N_MAIN == 450 and config.WS4_N_SANITY == 50
    assert config.WS4_N_BOOTSTRAP == 10_000
    assert config.WS4_EQUIVALENCE_MARGIN == 0.05
    assert config.WS4_MATCH_TOLERANCE == 0.03
    assert config.WS4_PAIR_BONFERRONI_N == 5
    assert config.WS4_BUDGET_USD == 75.00
    assert config.WS4_SHARED_COLLECTION == "ws2_10m_shared"
    assert set(config.WS4_TRUNC_ARMS) == {
        "pca_uc_512_sq8", "pca_uc_384_sq8", "mrl_512_sq8"}
    assert set(config.WS4_CEILING_1M) == set(config.WS4_TRUNC_ARMS)
    assert config.WS4_TRUNC_NPROBE == 512 == max(config.WS4_SQ8_NPROBES)


# ------------------------------------------------------------ scoring ----


def test_norm_text_matches_ws1_definition():
    # NFKD splits the accent into a combining mark, which the WS1 regex turns
    # into a space -- reproduced byte-for-byte, quirk included
    assert core.norm_text("Élan, 1972!") == "e lan  1972 "
    assert core.norm_text("ABC") == "abc"


def test_answer_present_substring_over_concatenated_texts():
    golds = ["December 1972", "14 December 1972 UTC"]
    texts = ["Apollo 17 left the Moon in", "december 1972 after three days."]
    assert core.answer_present(golds, texts) is True
    assert core.answer_present(golds, ["nothing here"]) is False
    assert core.answer_present(golds, []) is False


def test_normalize_answer_is_squad_style():
    assert core.normalize_answer("The  Beatles.") == "beatles"
    assert core.normalize_answer("An answer, a  test") == "answer test"
    assert core.normalize_answer("  1972 ") == "1972"


def test_exact_match_and_f1():
    assert core.exact_match("the beatles", ["Beatles", "Rolling Stones"]) == 1
    assert core.exact_match("beatle", ["Beatles"]) == 0
    assert core.f1("Bob Russell and Bobby Scott", ["Bobby Scott"]) == pytest.approx(
        2 * (2 / 5) * 1.0 / ((2 / 5) + 1.0))
    assert core.f1("", ["x"]) == 0.0
    assert core.f1("x", ["x"]) == 1.0
    # max over aliases, not the first
    assert core.f1("Bob Russell", ["Bobby Scott", "Bob Russell"]) == 1.0


def test_is_idk_matches_the_exact_committed_phrase_only():
    assert core.is_idk("I don't know") is True
    assert core.is_idk("  i don't know.\n") is True
    assert core.is_idk("I don’t know") is True     # curly apostrophe
    assert core.is_idk("I don't know, but maybe 1972") is False
    assert core.is_idk("1972") is False


# ------------------------------------------------------- prompt render ----


def test_render_context_is_rank_ordered_and_deterministic():
    ps = [{"title": "A", "text": "first"}, {"title": "B", "text": "second"}]
    out = core.render_context(ps)
    assert out == "[1] A\nfirst\n\n[2] B\nsecond"
    assert core.render_context([]) == core.NO_PASSAGES


def test_user_message_puts_question_after_passages():
    m = core.user_message("CTX", "who?")
    assert m.index("CTX") < m.index("who?")
    assert m.rstrip().endswith("who?")


# ---------------------------------------------------- operating points ----


def test_pick_matching_nprobe_nearest_ties_to_higher():
    sweep = {1: 0.55, 2: 0.69, 4: 0.81, 8: 0.85, 16: 0.90, 64: 0.965}
    assert core.pick_matching_nprobe(0.70, sweep) == 2
    assert core.pick_matching_nprobe(0.91, sweep) == 16
    assert core.pick_matching_nprobe(0.83, sweep) == 8      # tie -> higher
    assert core.pick_matching_nprobe(0.99, sweep) == 64


# ------------------------------------------------------ question draw ----


def test_select_questions_main_from_pass_set_sanity_unfiltered_disjoint():
    rng_pass = np.zeros(1000, dtype=bool)
    rng_pass[::3] = True                     # 334 pass
    main, sanity = core.select_questions(rng_pass, 100, 20, seed=42)
    assert len(main) == 100 and len(sanity) == 20
    assert rng_pass[main].all()
    assert not set(main) & set(sanity)
    assert len(set(main)) == 100 and len(set(sanity)) == 20
    # deterministic
    m2, s2 = core.select_questions(rng_pass, 100, 20, seed=42)
    assert np.array_equal(main, m2) and np.array_equal(sanity, s2)
    # sanity is drawn WITHOUT the filter: some non-passing rows expected
    assert (~rng_pass[sanity]).any()


def test_select_questions_refuses_when_pass_set_too_small():
    with pytest.raises(ValueError):
        core.select_questions(np.zeros(10, dtype=bool), 5, 2, seed=42)


# ----------------------------------------------------------- bootstrap ----


def test_bootstrap_indices_shape_and_determinism():
    a = core.bootstrap_indices(7, 50, seed=1)
    b = core.bootstrap_indices(7, 50, seed=1)
    assert a.shape == (50, 7) and a.max() < 7 and a.min() >= 0
    assert np.array_equal(a, b)
    assert not np.array_equal(a, core.bootstrap_indices(7, 50, seed=2))


def test_boot_means_and_percentile_ci():
    idx = core.bootstrap_indices(1000, 2000, seed=0)
    x = np.zeros(1000); x[:500] = 1.0
    means = core.boot_means(x, idx)
    lo, hi = core.percentile_ci(means, 0.05)
    assert 0.45 < lo < 0.5 < hi < 0.55
    lo99, hi99 = core.percentile_ci(means, 0.01)
    assert lo99 < lo and hi99 > hi


def test_paired_difference_uses_the_same_resamples():
    idx = core.bootstrap_indices(400, 3000, seed=3)
    a = np.random.default_rng(0).random(400) < 0.8
    b = a.copy()                              # identical -> delta exactly 0
    d = core.boot_means(a.astype(float), idx) - core.boot_means(b.astype(float), idx)
    assert np.all(d == 0)


# ------------------------------------------- spec 7.1 classification ----


@pytest.mark.parametrize("lo,hi,expected", [
    (0.01, 0.06, "DROP"),            # excludes zero from above
    (-0.02, 0.03, "PLATEAU"),        # upper < margin
    (-0.01, 0.07, "INCONCLUSIVE"),   # includes zero, upper >= margin
    (0.0, 0.04, "PLATEAU"),          # lower == 0 is not strictly > 0
    (0.06, 0.09, "DROP"),            # both: DROP wins over anything
])
def test_classify_delta(lo, hi, expected):
    assert core.classify_delta(lo, hi, margin=0.05) == expected


def test_knee_is_lowest_recall_with_all_above_plateau():
    arms = [("a", 0.96, "PLATEAU"), ("b", 0.90, "PLATEAU"),
            ("c", 0.82, "DROP"), ("d", 0.70, "DROP")]
    assert core.knee(arms) == 0.90
    arms2 = [("a", 0.96, "PLATEAU"), ("b", 0.90, "DROP"),
             ("c", 0.82, "PLATEAU")]
    assert core.knee(arms2) == 0.96            # the interleaved c does not count
    assert core.knee([("a", 0.96, "DROP")]) is None
    assert core.knee([("a", 0.96, "INCONCLUSIVE"), ("b", 0.9, "PLATEAU")]) is None


def test_knee_branch():
    assert core.knee_branch([("a", 0.96, "PLATEAU"), ("b", 0.8, "DROP")]) == "K1"
    assert core.knee_branch([("a", 0.98, "PLATEAU"), ("b", 0.8, "DROP")]) == "K3"
    assert core.knee_branch([("a", 0.96, "DROP"), ("b", 0.8, "DROP")]) == "K2"
    assert core.knee_branch([("a", 0.96, "INCONCLUSIVE"),
                            ("b", 0.8, "INCONCLUSIVE")]) == "K3"
    assert core.knee_branch([("a", 0.96, "PLATEAU"), ("b", 0.9, "DROP"),
                            ("c", 0.8, "PLATEAU")]) == "K1"


# ------------------------------------------- spec 7.2 mechanism test ----


def test_is_matched_uses_the_tolerance_inclusively():
    assert core.is_matched(0.960, 0.965, tol=0.03)
    assert core.is_matched(0.70, 0.73, tol=0.03)
    assert not core.is_matched(0.78, 0.70, tol=0.03)


def test_mechanism_branch_primary_face_value_others_adjusted():
    # pair A face CI excludes zero -> M1 regardless of the rest
    pairs = [dict(pair="A", matched=True, face=(0.01, 0.08), adj=(-0.01, 0.10)),
             dict(pair="B", matched=True, face=(-0.02, 0.02), adj=(-0.03, 0.03))]
    assert core.mechanism_branch(pairs) == "M1"
    # pair B excludes zero at face value but NOT at adjusted width -> not M1
    pairs = [dict(pair="A", matched=True, face=(-0.02, 0.02), adj=(-0.03, 0.03)),
             dict(pair="B", matched=True, face=(0.005, 0.045), adj=(-0.005, 0.055))]
    assert core.mechanism_branch(pairs) == "M2"
    # pair B excludes zero at adjusted width -> M1
    pairs[1]["adj"] = (0.002, 0.06)
    assert core.mechanism_branch(pairs) == "M1"
    # all include zero but one face CI wider than +-0.05 -> M3
    pairs = [dict(pair="A", matched=True, face=(-0.06, 0.02), adj=(-0.08, 0.04)),
             dict(pair="B", matched=True, face=(-0.02, 0.02), adj=(-0.03, 0.03))]
    assert core.mechanism_branch(pairs) == "M3"
    # unmatched pairs are ignored
    pairs = [dict(pair="A", matched=True, face=(-0.02, 0.02), adj=(-0.03, 0.03)),
             dict(pair="F", matched=False, face=(0.1, 0.2), adj=(0.1, 0.2))]
    assert core.mechanism_branch(pairs) == "M2"


def test_mechanism_branch_with_no_matched_pairs_is_m3():
    assert core.mechanism_branch([dict(pair="A", matched=False,
                                      face=(0, 0), adj=(0, 0))]) == "M3"


def test_parametric_reading_tiers():
    assert core.parametric_reading(0.05) == "holds"
    assert core.parametric_reading(0.10) == "holds"
    assert core.parametric_reading(0.2) == "partial_leak"
    assert core.parametric_reading(0.31) == "does_not_ground"


def test_bonferroni_alpha():
    assert core.bonferroni_alpha(0.05, 5) == pytest.approx(0.01)


# ------------------------------------------------ analysis helpers ----


def _fake_runs(n=300, seed=0):
    """Per-question judge outcomes for three arms with known structure:
    ref and 'same' identical; 'worse' loses 20 points."""
    rng = np.random.default_rng(seed)
    ref = rng.random(n) < 0.8
    worse = ref & (rng.random(n) < 0.75)
    recall = {"ref": np.full(n, 0.99), "same": np.full(n, 0.96),
              "worse": np.full(n, 0.80)}
    quality = {"ref": ref, "same": ref.copy(), "worse": worse}
    return recall, quality


def test_summarise_arms_paired_and_classified():
    recall, quality = _fake_runs()
    out = core.summarise_arms(recall, quality, ref="ref", n_boot=2000, seed=1,
                             margin=0.05, alpha=0.05)
    by = {r["arm"]: r for r in out}
    assert by["ref"]["delta_vs_ref"] == 0.0 and by["ref"]["class"] == "reference"
    assert by["same"]["delta_vs_ref"] == 0.0
    assert by["same"]["class"] == "PLATEAU"       # identical -> CI is [0, 0]
    assert by["worse"]["class"] == "DROP"
    assert by["worse"]["delta_ci_lo"] > 0
    assert 0.95 < by["same"]["recall_ci_lo"] <= by["same"]["recall"] <= by["same"]["recall_ci_hi"]
    assert set(out[0]) >= {"arm", "n", "recall", "recall_ci_lo", "recall_ci_hi",
                           "quality", "quality_ci_lo", "quality_ci_hi",
                           "delta_vs_ref", "delta_ci_lo", "delta_ci_hi",
                           "class"}


def test_adjacent_slopes_reports_joint_ci():
    recall, quality = _fake_runs()
    out = core.adjacent_slopes(recall, quality, n_boot=500, seed=2, alpha=0.05)
    assert [(r["upper"], r["lower"]) for r in out] == [("ref", "same"), ("same", "worse")]
    assert out[0]["slope"] == 0.0                      # identical quality
    assert out[1]["slope"] > 0 and out[1]["slope_ci_lo"] > 0


def test_evaluate_pairs_face_and_adjusted():
    recall, quality = _fake_runs()
    plan = [("A", "same", "ref"), ("B", "worse", "same"), ("F", "worse", "ref")]
    out = core.evaluate_pairs(plan, recall, quality, n_boot=2000, seed=3,
                             tol=0.03, alpha=0.05, n_adjust=5)
    by = {r["pair"]: r for r in out}
    assert by["A"]["matched"] is True and by["A"]["diff"] == 0.0
    assert by["B"]["matched"] is False                  # 0.80 vs 0.96
    assert by["F"]["matched"] is False
    # adjusted interval is wider than the face-value one
    assert by["B"]["adj_lo"] <= by["B"]["face_lo"] and by["B"]["adj_hi"] >= by["B"]["face_hi"]
    assert core.mechanism_branch([dict(pair=r["pair"], matched=r["matched"],
                                      face=(r["face_lo"], r["face_hi"]),
                                      adj=(r["adj_lo"], r["adj_hi"]))
                                 for r in out]) == "M2"
