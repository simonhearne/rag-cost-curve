"""Unit tests for the pure half of WS9. No network, no API keys, no Milvus."""

import pandas as pd
import pytest

from benchlib import config
from benchlib.code_retrieval import scaling as ws9


def test_ws9_pins_are_present_and_frozen():
    assert config.WS9_SLUG == "agentic_hil"
    assert config.WS9_SCOPES == ("S", "M", "L")
    assert config.WS9_SWEEP_SIZES == ("S", "M")
    assert config.WS9_ARMS == ("agentic", "indexed", "indexed_topk3")
    assert config.WS9_ELIGIBILITY_SCOPE == "S"
    assert config.WS9_PANEL_N == 37
    assert config.WS9_BUDGET_USD == 55.00
    assert config.WS9_MATCH_BAND == 2.0
    assert config.WS9_SMOKE_QUESTIONS == 2
    # The tiers, exactly as pre-registered in spec section 2.2.
    assert config.WS9_SUBSETS["S"] == {"include": ("src/",), "exclude": ()}
    assert config.WS9_SUBSETS["M"] == {"include": (), "exclude": ("tests/",)}
    assert config.WS9_SUBSETS["L"] == {"include": (), "exclude": ()}


def test_ws9_adds_nothing_to_the_frozen_pins():
    # WS9 only ADDS. If any of these moved, the workstream broke the house rule.
    assert config.WS6A_COMMIT == "a1fa70d4237d50aae6586a0d9b229df583463d21"
    assert config.WS6A_TURN_CAP == 15
    assert config.WS6C_TOPK == 3
    assert config.WS6C_BOOT_N == 10_000
    assert config.WS6C_BOOT_SEED == 42
    assert config.WS6C_ALPHA == 0.05
    assert config.WS6C_BUDGET_USD == 45.00
    assert config.WS6C_COMMITS["agentic_hil"] == (
        "837f39813e1421c7b8efc6e621c1368507f2dbb5")


PATHS = [
    "src/agentic_hil/tools.py",
    "src/agentic_hil/backends/stlink.py",
    "tests/test_bench_mutex.py",
    "tools/bench_battery.py",
    "docs/index.md",
    "README.md",
]


def test_s_is_the_src_tier_only():
    assert ws9.ws9_subset_files(PATHS, "S") == [
        "src/agentic_hil/tools.py", "src/agentic_hil/backends/stlink.py"]


def test_m_is_everything_except_tests_including_root_files():
    # The whole point of the exclude tier: root-level files have no directory
    # prefix, so an include-only prefix list cannot express "all but tests/".
    assert ws9.ws9_subset_files(PATHS, "M") == [
        "src/agentic_hil/tools.py", "src/agentic_hil/backends/stlink.py",
        "tools/bench_battery.py", "docs/index.md", "README.md"]


def test_l_is_everything_and_preserves_input_order():
    assert ws9.ws9_subset_files(PATHS, "L") == PATHS


def test_tests_prefix_does_not_swallow_a_similarly_named_directory():
    # "tests/" must exclude tests/, not testsuite/ -- str.startswith on a bare
    # "tests" would take both.
    paths = ["tests/a.py", "testsuite/b.py", "src/c.py"]
    assert ws9.ws9_subset_files(paths, "M") == ["testsuite/b.py", "src/c.py"]


def test_scopes_are_nested():
    assert ws9.ws9_scopes_are_nested(PATHS)


def test_nesting_guard_rejects_paths_the_include_prefixes_never_match():
    # Absolute paths: "src/" matches nothing, so S is empty while M and L are
    # full. A plain subset test would pass this -- and the caller would index
    # an EMPTY S tier and be told nothing was wrong.
    absolute = ["/repo/" + p for p in PATHS]
    assert ws9.ws9_subset_files(absolute, "S") == []
    assert not ws9.ws9_scopes_are_nested(absolute)


def test_nesting_guard_rejects_tiers_that_do_not_actually_grow():
    # No tests/ anywhere, so M == L: three size points, two distinct corpora.
    assert not ws9.ws9_scopes_are_nested(["src/a.py", "docs/b.md"])
    # And S == M when everything lives under src/.
    assert not ws9.ws9_scopes_are_nested(["src/a.py", "src/b.py"])


def test_unknown_scope_raises():
    with pytest.raises(ValueError, match="unknown WS9 scope"):
        ws9.ws9_subset_files(PATHS, "XL")


def _questions(**rows):
    return pd.DataFrame([{"qid": q, "expected_files": f}
                         for q, f in rows.items()])


def test_eligibility_requires_every_gold_file_inside_the_scope():
    q = _questions(
        q1="src/a.py",                       # in
        q2="src/a.py;src/backends/b.py",     # in
        q3="src/a.py;tools/c.py",            # OUT: one gold file outside S
        q4="tests/d.py",                     # OUT
    )
    assert ws9.ws9_eligible_qids(q, "S") == ["q1", "q2"]


def test_eligibility_rejects_a_question_with_no_gold_files():
    q = _questions(q1="", q2="src/a.py")
    assert ws9.ws9_eligible_qids(q, "S") == ["q2"]


def test_the_real_panel_is_thirty_seven_of_forty():
    """Against the committed WS6c questions -- the number the spec pins."""
    q = pd.read_csv(config.ws6c_dir() / "questions.csv")
    panel = ws9.ws9_eligible_qids(q, config.WS9_ELIGIBILITY_SCOPE)
    assert len(panel) == config.WS9_PANEL_N == 37
    # The three the spec names, and no others.
    assert set(q["qid"]) - set(panel) == {
        "agentic_hil_q09", "agentic_hil_q30", "agentic_hil_q31"}


def test_the_panel_is_a_superset_of_the_pre_registered_scaling_ten():
    q = pd.read_csv(config.ws6c_dir() / "questions.csv")
    panel = set(ws9.ws9_eligible_qids(q, config.WS9_ELIGIBILITY_SCOPE))
    ten = set(q[q["scaling_subset"].astype(str).str.lower() == "true"]["qid"])
    assert len(ten) == 10
    assert ten <= panel


def test_collection_prefix_is_scope_keyed_and_lowercase():
    assert ws9.ws9_collection_prefix("S") == "ws9_s"
    assert ws9.ws9_collection_prefix("M") == "ws9_m"


import json

from benchlib import agent_harness


def _checkpoint(path, billed):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"qid": "q1", "arm": "agentic", "cost_total_billed": billed}) + "\n")


def test_ws9_governor_sees_only_ws9_spend(tmp_path):
    _checkpoint(tmp_path / "ws6a_cache" / "ws6a_checkpoint.jsonl", 25.14)
    _checkpoint(tmp_path / "ws6c_cache" / "run_agentic_hil.jsonl", 44.00)
    _checkpoint(tmp_path / "ws9_cache" / "scale_s.jsonl", 3.50)

    ws9 = agent_harness.total_spent_across_checkpoints(
        tmp_path, globs=agent_harness.WS9_SPEND_CHECKPOINT_GLOBS)
    assert ws9 == 3.50, (
        "the WS9 governor must not inherit WS6a's $75 and WS6c's $45 caps' "
        "spend -- it would trip on the first call against a $55 cap")


def test_default_globs_still_fail_closed_and_now_include_ws9(tmp_path):
    _checkpoint(tmp_path / "ws6a_cache" / "ws6a_checkpoint.jsonl", 25.14)
    _checkpoint(tmp_path / "ws6c_cache" / "run_agentic_hil.jsonl", 44.00)
    _checkpoint(tmp_path / "ws9_cache" / "scale_s.jsonl", 3.50)

    # A WS6a or WS6c script re-run later sees WS9's spend too. Over-counting a
    # shared cap can only make a governor trip EARLIER, which is the safe
    # direction and is the existing docstring's own rule.
    assert agent_harness.total_spent_across_checkpoints(tmp_path) == 72.64


def test_ws9_globs_cover_every_ws9_checkpoint_shape(tmp_path):
    for name in ("scale_s.jsonl", "scale_m.jsonl", "smoke_s.jsonl"):
        _checkpoint(tmp_path / "ws9_cache" / name, 1.00)
    assert agent_harness.total_spent_across_checkpoints(
        tmp_path, globs=agent_harness.WS9_SPEND_CHECKPOINT_GLOBS) == 3.00
