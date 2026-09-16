"""Unit tests for the pure half of WS6a. No network, no API keys."""

import os
from benchlib import config
from benchlib.code_retrieval import bench as ws6a


def test_ws6a_pins_are_present_and_consistent():
    assert config.WS6A_COMMIT == "a1fa70d4237d50aae6586a0d9b229df583463d21"
    assert len(config.WS6A_COMMIT) == 40
    assert config.WS6A_AGENT_MODEL != config.WS6A_JUDGE_MODEL, (
        "judge model must differ from the agent model"
    )
    assert config.WS6A_PREFIX_BUDGET_TOKENS == 960_000
    assert config.WS6A_TURN_CAP == 15
    assert set(config.WS6A_SUBSETS) == {"S", "M", "L"}
    assert config.WS6A_SUBSETS["L"] == ()
    # S must be a subset of M, or the nested-subset design is broken.
    assert set(config.WS6A_SUBSETS["S"]) <= set(config.WS6A_SUBSETS["M"])


def test_is_text_file_matches_allowlist():
    assert ws6a.is_text_file("fastapi/routing.py")
    assert ws6a.is_text_file("docs/en/index.md")
    assert not ws6a.is_text_file("docs/img/logo.png")
    assert not ws6a.is_text_file("Makefile")  # no extension -> excluded


def test_subset_files_is_nested():
    paths = [
        "fastapi/routing.py",
        "docs_src/app_testing/tutorial001.py",
        "tests/test_router.py",
        "docs/en/index.md",
        "pyproject.toml",
    ]
    s = ws6a.subset_files(paths, "S")
    m = ws6a.subset_files(paths, "M")
    lg = ws6a.subset_files(paths, "L")
    assert s == ["fastapi/routing.py"]
    assert set(s) < set(m) < set(lg)
    assert lg == paths  # L is the whole repo, unfiltered


def test_subset_files_rejects_unknown_subset():
    import pytest
    with pytest.raises(ValueError, match="unknown subset"):
        ws6a.subset_files(["fastapi/a.py"], "XL")


def test_tiered_order_puts_library_before_docs():
    paths = [
        "docs/de/index.md",
        "tests/test_a.py",
        "fastapi/routing.py",
        "docs_src/tutorial001.py",
        "docs/en/index.md",
        "pyproject.toml",
    ]
    ordered = ws6a.tiered_order(paths)
    # Library first, translations last -- the whole point of the tiering.
    assert ordered[0] == "fastapi/routing.py"
    assert ordered.index("docs_src/tutorial001.py") < ordered.index("tests/test_a.py")
    assert ordered.index("tests/test_a.py") < ordered.index("docs/en/index.md")
    assert ordered.index("docs/en/index.md") < ordered.index("docs/de/index.md")


def test_tiered_order_is_lexicographic_within_a_tier():
    paths = ["fastapi/z.py", "fastapi/a.py", "fastapi/m.py"]
    assert ws6a.tiered_order(paths) == ["fastapi/a.py", "fastapi/m.py", "fastapi/z.py"]


def test_build_prefix_stops_at_budget():
    paths = ["fastapi/a.py", "fastapi/b.py", "fastapi/c.py"]
    counts = {"fastapi/a.py": 400, "fastapi/b.py": 400, "fastapi/c.py": 400}
    res = ws6a.build_prefix(paths, counts, budget=900)
    assert res.included == ["fastapi/a.py", "fastapi/b.py"]
    assert res.tokens == 800
    assert res.total_tokens == 1200
    assert res.fit_fraction == 800 / 1200


def test_build_prefix_skips_an_oversized_file_but_keeps_going():
    # A single file larger than the whole budget must not abort the build --
    # it is skipped and smaller later files still get their chance.
    paths = ["fastapi/huge.py", "fastapi/small.py"]
    counts = {"fastapi/huge.py": 5000, "fastapi/small.py": 100}
    res = ws6a.build_prefix(paths, counts, budget=900)
    assert res.included == ["fastapi/small.py"]
    assert res.tokens == 100


def test_classify_prefix_membership_three_states():
    included = ["fastapi/a.py", "fastapi/b.py"]
    assert ws6a.classify_prefix_membership(["fastapi/a.py"], included) == "all"
    assert ws6a.classify_prefix_membership(
        ["fastapi/a.py", "fastapi/b.py"], included) == "all"
    # Straddling question: MUST be `partial`, not `all`. A boolean column would
    # bucket this as in-prefix and reintroduce the confound the column exists
    # to remove.
    assert ws6a.classify_prefix_membership(
        ["fastapi/a.py", "tests/z.py"], included) == "partial"
    assert ws6a.classify_prefix_membership(["tests/z.py"], included) == "none"


def test_classify_prefix_membership_rejects_empty_expected_files():
    import pytest
    with pytest.raises(ValueError, match="expected_files"):
        ws6a.classify_prefix_membership([], ["fastapi/a.py"])


def test_usage_add_accumulates_every_field():
    u = ws6a.Usage()
    u.add(ws6a.Usage(input_tokens=10, output_tokens=2,
                     cache_creation_input_tokens=100,
                     cache_read_input_tokens=5))
    u.add(ws6a.Usage(input_tokens=1, output_tokens=3,
                     cache_creation_input_tokens=0,
                     cache_read_input_tokens=50))
    assert u.input_tokens == 11
    assert u.output_tokens == 5
    assert u.cache_creation_input_tokens == 100
    assert u.cache_read_input_tokens == 55


def test_cost_billed_applies_cache_multipliers():
    price = {"input_usd_per_mtok": 3.0, "output_usd_per_mtok": 15.0,
             "cache_write_1h_multiplier": 2.0, "cache_read_multiplier": 0.1}
    u = ws6a.Usage(input_tokens=1_000_000, output_tokens=1_000_000,
                   cache_creation_input_tokens=1_000_000,
                   cache_read_input_tokens=1_000_000)
    # 3.00 uncached + 15.00 output + 6.00 write(2x) + 0.30 read(0.1x)
    assert ws6a.cost_billed(u, price) == 24.30


def test_cost_uncached_list_prices_all_input_at_full_rate():
    price = {"input_usd_per_mtok": 3.0, "output_usd_per_mtok": 15.0,
             "cache_write_1h_multiplier": 2.0, "cache_read_multiplier": 0.1}
    u = ws6a.Usage(input_tokens=1_000_000, output_tokens=0,
                   cache_creation_input_tokens=1_000_000,
                   cache_read_input_tokens=1_000_000)
    # All 3M input tokens at the full rate -- what a naive model predicts.
    assert ws6a.cost_uncached_list(u, price) == 9.00


def test_governor_trips_at_the_limit():
    import pytest
    g = ws6a.CostGovernor(limit=1.00)
    g.charge(0.60)
    assert g.spent == 0.60
    assert round(g.remaining, 2) == 0.40
    with pytest.raises(ws6a.BudgetExceeded, match="0.60"):
        g.charge(0.50)
    # The rejected charge must NOT be recorded -- otherwise a resumed run
    # double-counts spend that never happened.
    assert g.spent == 0.60


def test_governor_allows_a_charge_landing_exactly_on_the_limit():
    g = ws6a.CostGovernor(limit=1.00)
    g.charge(1.00)
    assert g.spent == 1.00


def test_load_pricing_raises_on_empty_regime():
    import pytest
    import tempfile
    import os
    # Create a CSV with no rows matching the requested regime
    csv_content = """model,input_usd_per_mtok,output_usd_per_mtok,cache_write_1h_multiplier,cache_read_multiplier,regime,source_url,fetched_date
claude-sonnet-5,2.00,10.00,2.0,0.1,standard,https://example.com,2026-08-17"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        temp_path = f.name
    try:
        # Try to load a regime that doesn't exist in the CSV
        with pytest.raises(ValueError, match="no pricing rows for regime"):
            ws6a.load_pricing(temp_path, regime="nonexistent_regime")
    finally:
        os.unlink(temp_path)


def test_usage_from_api_coerces_none_to_zero():
    # Create a mock usage object with explicit None values
    class MockUsage:
        input_tokens = None
        output_tokens = None
        cache_creation_input_tokens = None
        cache_read_input_tokens = None

    u = ws6a.Usage.from_api(MockUsage())
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.cache_creation_input_tokens == 0
    assert u.cache_read_input_tokens == 0


def test_usage_from_api_handles_missing_fields():
    # Create a mock usage object with no attributes
    class MockUsage:
        pass

    u = ws6a.Usage.from_api(MockUsage())
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.cache_creation_input_tokens == 0
    assert u.cache_read_input_tokens == 0


def test_interleaved_order_covers_every_pair_exactly_once():
    qids = [f"q{i:02d}" for i in range(5)]
    arms = ("indexed", "agentic", "stuffed", "parametric")
    order = ws6a.interleaved_order(qids, arms, seed=42)
    assert len(order) == 20
    assert len(set(order)) == 20
    assert {a for _, a in order} == set(arms)


def test_interleaved_order_round_robins_arms():
    qids = [f"q{i:02d}" for i in range(5)]
    arms = ("indexed", "agentic", "stuffed", "parametric")
    order = ws6a.interleaved_order(qids, arms, seed=42)
    # Every consecutive window of len(arms) hits each arm exactly once, so no
    # arm is bunched into one stretch of wall-clock time.
    for i in range(0, len(order), len(arms)):
        window = [a for _, a in order[i:i + len(arms)]]
        assert sorted(window) == sorted(arms)


def test_interleaved_order_gives_each_arm_a_different_question_order():
    qids = [f"q{i:02d}" for i in range(20)]
    arms = ("indexed", "agentic")
    order = ws6a.interleaved_order(qids, arms, seed=42)
    per_arm = {a: [q for q, arm in order if arm == a] for a in arms}
    assert per_arm["indexed"] != per_arm["agentic"], (
        "arms must not share a question order"
    )
    # ...but each arm still sees the full question set.
    for a in arms:
        assert sorted(per_arm[a]) == sorted(qids)


def test_interleaved_order_is_deterministic_for_a_seed():
    qids = [f"q{i:02d}" for i in range(8)]
    arms = ("indexed", "agentic")
    assert (ws6a.interleaved_order(qids, arms, seed=42)
            == ws6a.interleaved_order(qids, arms, seed=42))
    assert (ws6a.interleaved_order(qids, arms, seed=42)
            != ws6a.interleaved_order(qids, arms, seed=43))


def test_score_programmatic_detects_a_plain_path_mention():
    r = ws6a.score_programmatic(
        "The dependency resolution happens in fastapi/dependencies/utils.py.",
        ["fastapi/dependencies/utils.py"], ["solve_dependencies"])
    assert r["any_file_hit"] is True
    assert r["all_files_hit"] is True
    assert r["symbol_hit"] is False


def test_score_programmatic_tolerates_leading_slash_and_backticks():
    r = ws6a.score_programmatic(
        "See `/fastapi/routing.py` for the APIRoute class.",
        ["fastapi/routing.py"], ["APIRoute"])
    assert r["any_file_hit"] is True
    assert r["symbol_hit"] is True


def test_score_programmatic_all_files_requires_every_file():
    r = ws6a.score_programmatic(
        "Only fastapi/routing.py matters here.",
        ["fastapi/routing.py", "fastapi/applications.py"], [])
    assert r["any_file_hit"] is True
    assert r["all_files_hit"] is False
    assert r["matched_files"] == ["fastapi/routing.py"]


def test_score_programmatic_does_not_match_a_different_file_with_same_basename():
    # `utils.py` alone must not count as `fastapi/dependencies/utils.py` --
    # basename matching would massively over-credit every arm.
    r = ws6a.score_programmatic(
        "It is handled in utils.py somewhere.",
        ["fastapi/dependencies/utils.py"], [])
    assert r["any_file_hit"] is False


def test_score_programmatic_symbol_match_is_word_bounded():
    r = ws6a.score_programmatic("We call solve_dependencies_v2 here.",
                                ["fastapi/a.py"], ["solve_dependencies"])
    assert r["symbol_hit"] is False


def test_normalise_path_preserves_dot_prefix():
    # .github, .pre-commit-config are valid dotfiles; must not strip the dot.
    assert ws6a.normalise_path(".github/workflows/ci.yml") == ".github/workflows/ci.yml"
    assert ws6a.normalise_path(".pre-commit-config.yaml") == ".pre-commit-config.yaml"
    # But ./ prefix is removed
    assert ws6a.normalise_path("./fastapi/a.py") == "fastapi/a.py"
    # And leading / is removed
    assert ws6a.normalise_path("/fastapi/a.py") == "fastapi/a.py"


def test_score_programmatic_rejects_vendor_prefix():
    # vendor/fastapi/routing.py in prose does NOT match fastapi/routing.py
    r = ws6a.score_programmatic(
        "See vendor/fastapi/routing.py for details.",
        ["fastapi/routing.py"], [])
    assert r["any_file_hit"] is False


def test_score_programmatic_rejects_merged_prefix():
    # notfastapi/routing.py in prose does NOT match fastapi/routing.py
    r = ws6a.score_programmatic(
        "Look at notfastapi/routing.py instead.",
        ["fastapi/routing.py"], [])
    assert r["any_file_hit"] is False


def test_score_programmatic_matches_sentence_final_period():
    # Sentence-final period is a common case; should match.
    r = ws6a.score_programmatic(
        "The handler lives in fastapi/routing.py.",
        ["fastapi/routing.py"], [])
    assert r["any_file_hit"] is True


def test_score_programmatic_all_files_hit_false_when_expected_files_empty():
    # Empty expected_files should give all_files_hit=False, not True.
    # A question with no expected files should not score as "fully correct".
    r = ws6a.score_programmatic(
        "Some answer text.", [], [])
    assert r["any_file_hit"] is False
    assert r["all_files_hit"] is False
    assert r["symbol_hit"] is False
    assert r["matched_files"] == []


# ----------------------------------------------- question validation tests


def _q(**kw):
    base = {"qid": "q01", "question": "where?", "expected_files": "fastapi/a.py",
            "expected_symbols": "f", "difficulty": "locate", "category": "routing",
            "notes": "", "scaling_subset": "false", "excluded": "false",
            "exclusion_reason": ""}
    base.update(kw)
    return base


def test_validate_questions_accepts_a_good_row():
    # I-8 added cardinality checks (exact question count, scaling count,
    # difficulty mix) that a single-row fixture can never satisfy; these
    # per-row tests opt out via check_cardinality=False.
    assert ws6a.validate_questions([_q()], {"fastapi/a.py"},
                                   check_cardinality=False) == []


def test_validate_questions_rejects_expected_file_not_in_repo():
    problems = ws6a.validate_questions([_q(expected_files="fastapi/nope.py")],
                                       {"fastapi/a.py"}, check_cardinality=False)
    assert any("not in the repo" in p for p in problems)


def test_validate_questions_rejects_scaling_question_outside_fastapi():
    # The S subset is fastapi/ only, so a scaling question answered in tests/
    # would be unanswerable at S -- confounding corpus size with answer
    # availability and invalidating the whole curve.
    problems = ws6a.validate_questions(
        [_q(expected_files="tests/test_a.py", scaling_subset="true")],
        {"tests/test_a.py"}, check_cardinality=False)
    assert any("scaling_subset" in p and "fastapi/" in p for p in problems)


def test_validate_questions_rejects_bad_difficulty():
    problems = ws6a.validate_questions([_q(difficulty="hard")], {"fastapi/a.py"},
                                       check_cardinality=False)
    assert any("difficulty" in p for p in problems)


def test_validate_questions_rejects_duplicate_qids():
    problems = ws6a.validate_questions([_q(qid="q01"), _q(qid="q01")],
                                       {"fastapi/a.py"}, check_cardinality=False)
    assert any("duplicate" in p for p in problems)


def test_validate_questions_rejects_empty_expected_files():
    problems = ws6a.validate_questions([_q(expected_files="")], {"fastapi/a.py"},
                                       check_cardinality=False)
    assert any("expected_files" in p for p in problems)


def test_validate_questions_reports_missing_qid():
    # Two rows completely valid except for missing qid must each report the
    # missing identifier by row index, not as duplicates.
    problems = ws6a.validate_questions(
        [_q(qid=None),
         _q(qid=None)],
        {"fastapi/a.py"}, check_cardinality=False)
    # Must report exactly two problems, one per missing qid
    assert len(problems) == 2
    # Each row should be identified with its row index
    assert any("row 0" in p for p in problems)
    assert any("row 1" in p for p in problems)
    # Should not be claimed as duplicates (each has a unique placeholder)
    assert not any("duplicate" in p for p in problems)


def test_validate_questions_rejects_empty_qid():
    # An empty string qid must be reported as missing, not treated as valid.
    problems = ws6a.validate_questions(
        [_q(qid="")],
        {"fastapi/a.py"}, check_cardinality=False)
    assert any("qid" in p and "missing" in p for p in problems)


def test_resolve_in_root_allows_a_path_inside(tmp_path):
    from benchlib import agent_harness
    (tmp_path / "a.py").write_text("x")
    assert agent_harness.resolve_in_root(tmp_path, "a.py") == (tmp_path / "a.py")


def test_resolve_in_root_blocks_traversal(tmp_path):
    import pytest
    from benchlib import agent_harness
    for bad in ["../etc/passwd", "a/../../etc/passwd", "/etc/passwd"]:
        with pytest.raises(ValueError, match="outside the repository"):
            agent_harness.resolve_in_root(tmp_path, bad)


def test_resolve_in_root_blocks_a_symlink_escape(tmp_path):
    import pytest
    from benchlib import agent_harness
    outside = tmp_path.parent / "outside_target"
    outside.mkdir(exist_ok=True)
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the repository"):
        agent_harness.resolve_in_root(tmp_path, "link/secret.txt")


# --------------------------------------------------- agentic tool tests


def test_glob_rejects_traversal_in_error_string(tmp_path):
    """glob('../outside/secret.txt') must not escape the root."""
    import shutil
    import pytest
    from benchlib import agent_harness
    if not shutil.which("rg"):
        pytest.skip("ripgrep not found on PATH")
    grep_tool, glob_tool, read_tool = agent_harness.make_agentic_tools(tmp_path)
    result = glob_tool("../outside/secret.txt")
    # Must return an error string, not raise or leak the path
    assert isinstance(result, str)
    assert "Error" in result or "outside" in result.lower()
    assert "outside" not in result or "../" not in result  # no leakage


def test_glob_rejects_absolute_pattern_in_error_string(tmp_path):
    """glob('/etc/*') must not raise NotImplementedError, must return error."""
    import shutil
    import pytest
    from benchlib import agent_harness
    if not shutil.which("rg"):
        pytest.skip("ripgrep not found on PATH")
    grep_tool, glob_tool, read_tool = agent_harness.make_agentic_tools(tmp_path)
    result = glob_tool("/etc/*")
    assert isinstance(result, str)
    assert "Error" in result or "absolute" in result.lower()


def test_grep_emits_truncation_marker_on_per_file_overflow(tmp_path):
    """grep must emit truncation marker even when matches come from one file."""
    import shutil
    import pytest
    from benchlib import agent_harness

    # Skip if ripgrep is not available
    if not shutil.which("rg"):
        pytest.skip("ripgrep not found on PATH")

    # Create a file with 150 lines, each matching a simple pattern
    test_file = tmp_path / "many_matches.txt"
    test_file.write_text("\n".join(f"match_{i}" for i in range(150)))

    grep_tool, glob_tool, read_tool = agent_harness.make_agentic_tools(tmp_path)
    result = grep_tool("match_", "many_matches.txt")

    # Should contain truncation marker because > 100 matches
    assert "truncated" in result.lower()


def test_grep_returns_error_string_when_ripgrep_missing(tmp_path):
    """grep must return error string if ripgrep is unavailable, not raise."""
    import shutil
    import pytest
    import subprocess as subprocess_module
    from benchlib import agent_harness

    # Only run if ripgrep IS available (we're testing the error handling)
    if not shutil.which("rg"):
        pytest.skip("ripgrep must be on PATH to test error handling")

    # Create a simple test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    grep_tool, glob_tool, read_tool = agent_harness.make_agentic_tools(tmp_path)

    # Temporarily mock subprocess to raise FileNotFoundError
    original_run = subprocess_module.run

    def mock_run_raises(*args, **kwargs):
        raise FileNotFoundError("rg not found")

    subprocess_module.run = mock_run_raises
    try:
        result = grep_tool("hello", "test.txt")
        # Must return error string, not raise
        assert isinstance(result, str)
        assert "Error" in result
    finally:
        subprocess_module.run = original_run


def test_read_first_line_is_numbered_1(tmp_path):
    """read() with offset=1 must label the first line as 1, not 0."""
    import shutil
    import pytest
    from benchlib import agent_harness
    if not shutil.which("rg"):
        pytest.skip("ripgrep not found on PATH")
    test_file = tmp_path / "lines.txt"
    test_file.write_text("first\nsecond\nthird")

    grep_tool, glob_tool, read_tool = agent_harness.make_agentic_tools(tmp_path)
    result = read_tool("lines.txt", offset=1, limit=1)

    # First line must be labeled "1", not "0"
    assert "     1\t" in result
    assert "first" in result


def test_read_clamps_limit_to_max_and_shows_marker(tmp_path):
    """read() must clamp limit > READ_MAX_LINES and show 'more lines' marker."""
    import shutil
    import pytest
    from benchlib import agent_harness
    if not shutil.which("rg"):
        pytest.skip("ripgrep not found on PATH")
    test_file = tmp_path / "many_lines.txt"
    # Create a file with more than READ_MAX_LINES (400) lines
    test_file.write_text("\n".join(f"line_{i}" for i in range(500)))

    grep_tool, glob_tool, read_tool = agent_harness.make_agentic_tools(tmp_path)

    # Request 600 lines (more than 400 limit)
    result = read_tool("many_lines.txt", offset=1, limit=600)

    # Must clamp to 400 lines and show the marker
    assert "more lines" in result
    # Count numbered lines: format is "     1\ttext" with right-justified 6-wide field,
    # so check the stripped prefix for leading digits
    lines_returned = len([l for l in result.split("\n") if l and l.strip()[0].isdigit()])
    assert lines_returned == 400


# ------------------------------------------------ stub ripgrep tests (no real ripgrep needed)


def test_grep_truncation_marker_at_101_matches(tmp_path, stub_ripgrep):
    """grep must emit truncation marker when matches exceed cap (101+ → marker at 100)."""
    from benchlib import agent_harness
    test_file = tmp_path / "many_matches.txt"
    test_file.write_text("\n".join(f"match_{i}" for i in range(150)))

    grep_tool, glob_tool, read_tool = agent_harness.make_agentic_tools(tmp_path)
    result = grep_tool("match_", "many_matches.txt")

    # With 150 matches, should truncate at 100 and show marker
    assert "truncated" in result.lower()
    lines = [l for l in result.split("\n") if l and not l.startswith("...")]
    assert len(lines) == 100


def test_grep_no_truncation_marker_at_exactly_100_matches(tmp_path, stub_ripgrep):
    """grep must NOT emit truncation marker when exactly 100 matches (boundary case)."""
    from benchlib import agent_harness
    test_file = tmp_path / "single_file.txt"
    test_file.write_text("\n".join(f"pattern match {i}" for i in range(100)))

    grep_tool, glob_tool, read_tool = agent_harness.make_agentic_tools(tmp_path)
    result = grep_tool("pattern", "single_file.txt")

    # With exactly 100 matches, no truncation marker should appear
    assert "truncated" not in result.lower()
    lines = [l for l in result.split("\n") if l and not l.startswith("No matches")]
    assert len(lines) == 100


def test_grep_rewrites_absolute_paths_to_relative(tmp_path, stub_ripgrep):
    """grep output must strip root prefix, showing repo-relative paths only."""
    from benchlib import agent_harness
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    grep_tool, glob_tool, read_tool = agent_harness.make_agentic_tools(tmp_path)
    result = grep_tool("hello", "test.txt")

    # Output must show repo-relative path, not absolute
    assert "test.txt" in result
    assert str(tmp_path) not in result  # no absolute path leakage


def test_grep_empty_output_with_exit_1_is_no_matches(tmp_path, stub_ripgrep):
    """grep exit code 1 with empty stdout is 'No matches.', not an error."""
    from benchlib import agent_harness
    # Request a pattern that won't match (stub returns exit 1)
    grep_tool, glob_tool, read_tool = agent_harness.make_agentic_tools(tmp_path)
    result = grep_tool("nomatch_pattern", "somefile.txt")

    # Exit 1 with no output should be reported as "No matches."
    assert "No matches." in result
    assert "Error" not in result


def test_grep_nonzero_non1_exit_is_error_string(tmp_path, stub_ripgrep):
    """grep with non-zero, non-1 exit code is reported as error string."""
    from benchlib import agent_harness
    # Request a pattern that triggers exit 2 (stub will handle "error_pattern")
    grep_tool, glob_tool, read_tool = agent_harness.make_agentic_tools(tmp_path)
    result = grep_tool("error_pattern", "somefile.txt")

    # Exit 2 should be reported as an error string, not raise or silence the error
    assert "Error" in result
    assert isinstance(result, str)


def test_system_prompt_is_arm_agnostic():
    from benchlib import agent_harness
    p = agent_harness.SYSTEM_PROMPT.lower()
    # The prompt is byte-identical across arms, so it must not name any tool
    # or retrieval strategy -- otherwise arms differ in more than tools.
    for banned in ("grep", "glob", "search_code", "index", "semantic",
                   "claude-context", "mcp"):
        assert banned not in p, f"system prompt must not mention {banned!r}"
    assert "file path" in p


def test_run_arm_system_prompt_default_is_unchanged():
    # run_arm grew an optional system_prompt parameter (WS8 needs a
    # non-Python-codebase framing) that MUST default to the original
    # module-level SYSTEM_PROMPT, byte-identical, so every frozen
    # WS6a/WS6b/WS6c call site that does not pass it is unaffected. A future
    # edit that changes the default silently would invalidate every prior
    # run -- this test exists so that edit fails loudly instead.
    import inspect

    from benchlib import agent_harness
    default = inspect.signature(agent_harness.run_arm).parameters[
        "system_prompt"].default
    assert default is None, (
        "run_arm must default system_prompt to None and resolve it to the "
        "module-level SYSTEM_PROMPT internally, not bake a copy into the "
        "signature (which could drift from the real constant)")


def test_run_arm_max_tokens_default_is_unchanged():
    # Same pattern as system_prompt (R19): run_arm grew an optional
    # max_tokens parameter (WS8 needs to re-run a truncated question at a
    # larger budget without changing every other row's comparability) that
    # MUST default to None and resolve to the module-level MAX_TOKENS
    # internally, so every frozen WS6a/WS6b/WS6c call site that does not
    # pass it stays byte-identical.
    import inspect

    from benchlib import agent_harness
    default = inspect.signature(agent_harness.run_arm).parameters[
        "max_tokens"].default
    assert default is None, (
        "run_arm must default max_tokens to None and resolve it to the "
        "module-level MAX_TOKENS internally, not bake a copy into the "
        "signature")


def test_run_result_defaults_are_safe():
    from benchlib import agent_harness
    r = agent_harness.RunResult(qid="q01", arm="agentic")
    assert r.turns == 0
    assert r.hit_turn_cap is False
    assert r.answer == ""
    assert r.usage.input_tokens == 0
    assert r.turns_detail == []


def test_judge_prompt_has_all_placeholders_and_no_arm_leak():
    from pathlib import Path
    text = Path("results/ws6a/judge_prompt.txt").read_text()
    for ph in ("{question}", "{expected_files}", "{expected_symbols}", "{answer}"):
        assert ph in text, f"judge prompt missing {ph}"
    lowered = text.lower()
    # Judging must be blind: naming the arms would let the judge infer which
    # system produced the answer and grade it differently.
    for banned in ("grep", "claude-context", "indexed arm", "parametric",
                   "stuffed"):
        assert banned not in lowered, f"judge prompt leaks arm identity: {banned!r}"


def test_load_judge_prompt_formats_cleanly():
    from benchlib import agent_harness
    p = agent_harness.load_judge_prompt("results/ws6a/judge_prompt.txt")
    filled = p.format(question="q", expected_files="f", expected_symbols="s",
                      answer="a")
    assert "{question}" not in filled


# ------------------------------------------------ checkpoint tests


def test_checkpoint_round_trips_and_dedupes(tmp_path):
    from benchlib import agent_harness
    p = tmp_path / "ck.jsonl"
    agent_harness.append_checkpoint(p, {"qid": "q01", "arm": "agentic", "cost_billed": 0.5})
    agent_harness.append_checkpoint(p, {"qid": "q02", "arm": "agentic", "cost_billed": 0.25})
    # Test deduping: append a duplicate key with changed value
    agent_harness.append_checkpoint(p, {"qid": "q01", "arm": "agentic", "cost_billed": 0.75})
    done = agent_harness.load_checkpoint(p)
    assert set(done) == {("q01", "agentic"), ("q02", "agentic")}
    # Last write wins -- deduping preserves the later value
    assert done[("q01", "agentic")]["cost_billed"] == 0.75
    assert done[("q02", "agentic")]["cost_billed"] == 0.25


def test_load_checkpoint_on_missing_file_is_empty(tmp_path):
    from benchlib import agent_harness
    assert agent_harness.load_checkpoint(tmp_path / "nope.jsonl") == {}


def test_checkpoint_spent_sums_prior_cost(tmp_path):
    # C-1 renamed the governor's charged quantity from "cost_billed" to
    # "cost_total_billed" (agent + judge, matching what governor.charge()
    # actually charges); checkpoint_spent's resume point must track it.
    from benchlib import agent_harness
    p = tmp_path / "ck.jsonl"
    agent_harness.append_checkpoint(p, {"qid": "q01", "arm": "a", "cost_total_billed": 1.5})
    agent_harness.append_checkpoint(p, {"qid": "q02", "arm": "a", "cost_total_billed": 2.0})
    # A resumed run must start its governor from spend already incurred,
    # otherwise resuming silently doubles the effective budget.
    assert agent_harness.checkpoint_spent(p) == 3.5


def test_load_checkpoint_tolerates_truncated_final_line(tmp_path, capsys):
    """load_checkpoint on truncated final line returns earlier rows."""
    from benchlib import agent_harness
    p = tmp_path / "ck.jsonl"
    # Write two complete rows
    agent_harness.append_checkpoint(p, {"qid": "q01", "arm": "a", "cost_billed": 1.0})
    agent_harness.append_checkpoint(p, {"qid": "q02", "arm": "a", "cost_billed": 2.0})
    # Truncate the file by appending a partial line (no newline, no closing brace)
    with open(p, "a") as fh:
        fh.write('{"qid": "q03", "arm": "a"')
        fh.flush()
        os.fsync(fh.fileno())
    # load_checkpoint must skip the truncated line and return the two good rows
    done = agent_harness.load_checkpoint(p)
    assert len(done) == 2
    assert ("q01", "a") in done
    assert ("q02", "a") in done
    # Should have printed a warning about the truncated line
    captured = capsys.readouterr()
    assert "truncated" in captured.err.lower() or "truncated" in captured.out.lower()


def test_execute_run_checkpoint_survives_transcript_write_failure(tmp_path, monkeypatch, capsys):
    """execute_run writes checkpoint before transcript, so a write failure doesn't lose the row.

    This test exercises the ordering invariant inside execute_run: if the transcript write
    fails, the checkpoint must already be durable. If someone moves append_checkpoint after
    the transcript write, this test will fail and catch the regression.
    """
    import pandas as pd
    from benchlib import agent_harness
    from benchlib.code_retrieval import bench as ws6a

    # Create a transcript dir path whose parent is a file (so mkdir will fail)
    bad_parent = tmp_path / "is_a_file.txt"
    bad_parent.write_text("this is a file, not a directory")
    bad_transcript_dir = bad_parent / "subdir"

    checkpoint_path = tmp_path / "ck.jsonl"

    # Minimal questions DataFrame with required columns
    questions_df = pd.DataFrame([{
        "qid": "q01",
        "question": "Where is the handler?",
        "expected_files": "fastapi/routing.py",
        "expected_symbols": "APIRoute",
        "excluded": "false",
        "in_stuffed_prefix": "",
        "difficulty": "locate",
        "scaling_subset": "false",
    }])

    # Monkeypatch run_arm to return a canned result
    def fake_run_arm(client, arm, qid, question, **kwargs):
        return agent_harness.RunResult(
            qid=qid,
            arm=arm,
            answer="The handler is in fastapi/routing.py with APIRoute class.",
            turns=1,
            hit_turn_cap=False,
            stop_reason="end_turn",
            wall_clock_s=0.1,
            started_at="2026-08-17T12:00:00Z",
            error="",
            usage=ws6a.Usage(input_tokens=100, output_tokens=50),
        )

    # Monkeypatch judge_answer to return a canned verdict
    def fake_judge_answer(client, prompt_template, question, expected_files, expected_symbols, answer, **kwargs):
        return {
            "judge_correct": True,
            "judge_reason": "Answer mentions the correct file and class.",
            "usage": ws6a.Usage(input_tokens=200, output_tokens=50),
        }

    monkeypatch.setattr(agent_harness, "run_arm", fake_run_arm)
    monkeypatch.setattr(agent_harness, "judge_answer", fake_judge_answer)

    # Create a minimal cost governor
    governor = ws6a.CostGovernor(limit=100.0)

    # Create minimal prices dict (matching config model names with cache multipliers)
    prices = {
        "claude-sonnet-5": {
            "input_usd_per_mtok": 3.0,
            "output_usd_per_mtok": 15.0,
            "cache_write_1h_multiplier": 2.0,
            "cache_read_multiplier": 0.1,
        },
        "claude-opus-5": {
            "input_usd_per_mtok": 15.0,
            "output_usd_per_mtok": 60.0,
            "cache_write_1h_multiplier": 2.0,
            "cache_read_multiplier": 0.1,
        },
    }

    # Call execute_run with transcript_dir pointing to impossible location
    result_df = agent_harness.execute_run(
        client=None,  # Not used (run_arm is monkeypatched)
        questions_df=questions_df,
        arms=["agentic"],
        root=tmp_path,
        prefix_text="",
        mcp_tools=None,
        prices=prices,
        checkpoint_path=checkpoint_path,
        judge_prompt="Question: {question}\nAnswer: {answer}",
        governor=governor,
        transcript_dir=bad_transcript_dir,  # Will fail to write
        seed=42,
    )

    # Despite transcript write failure, checkpoint must contain the row
    done = agent_harness.load_checkpoint(checkpoint_path)
    assert ("q01", "agentic") in done, "Checkpoint must contain row even if transcript write fails"
    assert done[("q01", "agentic")]["answer"] == "The handler is in fastapi/routing.py with APIRoute class."
    assert len(result_df) == 1

    # Verify a warning was printed about the transcript failure
    captured = capsys.readouterr()
    assert "transcript write failed" in captured.err.lower() or "transcript write failed" in captured.out.lower()


# ============================================================================
# Final whole-branch review fix wave (2026-08-17)
# ============================================================================


def _fix_wave_prices():
    return {
        "claude-sonnet-5": {
            "input_usd_per_mtok": 3.0, "output_usd_per_mtok": 15.0,
            "cache_write_1h_multiplier": 2.0, "cache_read_multiplier": 0.1,
        },
        "claude-opus-5": {
            "input_usd_per_mtok": 15.0, "output_usd_per_mtok": 60.0,
            "cache_write_1h_multiplier": 2.0, "cache_read_multiplier": 0.1,
        },
    }


def _fix_wave_questions_df():
    import pandas as pd
    return pd.DataFrame([{
        "qid": "q01", "question": "Where is the handler?",
        "expected_files": "fastapi/routing.py", "expected_symbols": "APIRoute",
        "excluded": "false", "in_stuffed_prefix": "", "difficulty": "locate",
        "scaling_subset": "false",
    }])


# ---------------------------------------------------------------- C-1 ----


def test_execute_run_cost_columns_are_agent_only_matched_pair_and_total_sums_judge(
        tmp_path, monkeypatch):
    """cost_agent_billed and cost_uncached_list must price the SAME (agent-only)
    token set; cost_total_billed (governor's charged quantity) must be their
    sum with cost_judge_billed. Before this fix, the summed field was named
    cost_billed and compared against cost_uncached_list -- an agent-only
    figure -- which for the no-cache parametric arm reported caching as
    making the run 160% MORE expensive.
    """
    from benchlib import agent_harness
    from benchlib.code_retrieval import bench as ws6a

    checkpoint_path = tmp_path / "ck.jsonl"

    def fake_run_arm(client, arm, qid, question, **kwargs):
        return agent_harness.RunResult(
            qid=qid, arm=arm, answer="fastapi/routing.py has APIRoute.",
            turns=1, hit_turn_cap=False, stop_reason="end_turn",
            wall_clock_s=0.1, started_at="2026-08-17T12:00:00Z", error="",
            usage=ws6a.Usage(input_tokens=1000, output_tokens=200),
        )

    def fake_judge_answer(client, prompt_template, question, expected_files,
                          expected_symbols, answer, **kwargs):
        return {"judge_correct": True, "judge_reason": "ok",
                "usage": ws6a.Usage(input_tokens=300, output_tokens=50)}

    monkeypatch.setattr(agent_harness, "run_arm", fake_run_arm)
    monkeypatch.setattr(agent_harness, "judge_answer", fake_judge_answer)

    prices = _fix_wave_prices()
    governor = ws6a.CostGovernor(limit=100.0)
    df = agent_harness.execute_run(
        client=None, questions_df=_fix_wave_questions_df(), arms=["agentic"],
        root=tmp_path, prefix_text="", mcp_tools=None, prices=prices,
        checkpoint_path=checkpoint_path,
        judge_prompt="Question: {question}\nAnswer: {answer}",
        governor=governor, seed=42,
    )
    row = df.iloc[0]

    agent_usage = ws6a.Usage(input_tokens=1000, output_tokens=200)
    expected_agent_billed = ws6a.cost_billed(agent_usage, prices["claude-sonnet-5"])
    expected_uncached_list = ws6a.cost_uncached_list(agent_usage, prices["claude-sonnet-5"])
    judge_usage = ws6a.Usage(input_tokens=300, output_tokens=50)
    expected_judge_billed = ws6a.cost_billed(judge_usage, prices["claude-opus-5"])

    assert row["cost_agent_billed"] == expected_agent_billed
    assert row["cost_uncached_list"] == expected_uncached_list
    assert row["cost_judge_billed"] == expected_judge_billed
    assert row["cost_total_billed"] == round(
        row["cost_agent_billed"] + row["cost_judge_billed"], 6)
    # The governor must be charged the total (agent + judge), not the
    # agent-only figure -- that is the entire point of the rename.
    assert governor.spent == row["cost_total_billed"]


# ---------------------------------------------------------------- C-2 ----


def test_execute_run_does_not_checkpoint_an_errored_pair_and_retries_it(
        tmp_path, capsys, monkeypatch):
    """A pair whose run_arm raises (e.g. ripgrep missing) must not be
    checkpointed as a confidently-wrong, near-zero-cost answer. It must be
    skipped with a warning and retried on the next execute_run call.
    """
    from benchlib import agent_harness
    from benchlib.code_retrieval import bench as ws6a

    checkpoint_path = tmp_path / "ck.jsonl"
    calls = {"n": 0}

    def flaky_run_arm(client, arm, qid, question, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return agent_harness.RunResult(
                qid=qid, arm=arm, answer="", turns=0, hit_turn_cap=False,
                stop_reason="", wall_clock_s=0.01,
                started_at="2026-08-17T12:00:00Z",
                error="RuntimeError: ripgrep not found on PATH",
                usage=ws6a.Usage(),
            )
        return agent_harness.RunResult(
            qid=qid, arm=arm, answer="fastapi/routing.py has APIRoute.",
            turns=1, hit_turn_cap=False, stop_reason="end_turn",
            wall_clock_s=0.1, started_at="2026-08-17T12:00:01Z", error="",
            usage=ws6a.Usage(input_tokens=100, output_tokens=50),
        )

    def fake_judge_answer(client, prompt_template, question, expected_files,
                          expected_symbols, answer, **kwargs):
        return {"judge_correct": True, "judge_reason": "ok",
                "usage": ws6a.Usage(input_tokens=10, output_tokens=5)}

    monkeypatch.setattr(agent_harness, "run_arm", flaky_run_arm)
    monkeypatch.setattr(agent_harness, "judge_answer", fake_judge_answer)

    prices = _fix_wave_prices()
    kwargs = dict(
        client=None, questions_df=_fix_wave_questions_df(), arms=["agentic"],
        root=tmp_path, prefix_text="", mcp_tools=None, prices=prices,
        checkpoint_path=checkpoint_path,
        judge_prompt="Question: {question}\nAnswer: {answer}", seed=42,
    )

    # First call: the only pair errors. Must not be checkpointed.
    df1 = agent_harness.execute_run(governor=ws6a.CostGovernor(limit=100.0), **kwargs)
    assert len(df1) == 0
    assert agent_harness.load_checkpoint(checkpoint_path) == {}
    captured = capsys.readouterr()
    assert "q01" in captured.err or "q01" in captured.out
    assert "agentic" in captured.err or "agentic" in captured.out
    assert "ripgrep" in captured.err.lower() or "ripgrep" in captured.out.lower()

    # Second call: resume must retry the pair (not treat it as already done).
    df2 = agent_harness.execute_run(governor=ws6a.CostGovernor(limit=100.0), **kwargs)
    done = agent_harness.load_checkpoint(checkpoint_path)
    assert ("q01", "agentic") in done
    assert done[("q01", "agentic")]["error"] == ""
    assert len(df2) == 1
    assert calls["n"] == 2


def test_execute_run_prints_prominent_error_summary(tmp_path, capsys, monkeypatch):
    """At the end of a run with any errored pair, a prominent summary listing
    every errored pair must be printed -- an operator resuming a run must be
    able to see at a glance that something needs fixing."""
    from benchlib import agent_harness
    from benchlib.code_retrieval import bench as ws6a

    checkpoint_path = tmp_path / "ck.jsonl"

    def always_fails(client, arm, qid, question, **kwargs):
        return agent_harness.RunResult(
            qid=qid, arm=arm, answer="", turns=0, hit_turn_cap=False,
            stop_reason="", wall_clock_s=0.01, started_at="2026-08-17T12:00:00Z",
            error="ValueError: boom", usage=ws6a.Usage(),
        )

    monkeypatch.setattr(agent_harness, "run_arm", always_fails)

    agent_harness.execute_run(
        client=None, questions_df=_fix_wave_questions_df(), arms=["agentic"],
        root=tmp_path, prefix_text="", mcp_tools=None, prices=_fix_wave_prices(),
        checkpoint_path=checkpoint_path,
        judge_prompt="Question: {question}\nAnswer: {answer}",
        governor=ws6a.CostGovernor(limit=100.0), seed=42,
    )
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "1" in out and "error" in out.lower()
    assert "q01" in out and "agentic" in out


def test_assert_no_errors_raises_on_any_errored_row():
    import pandas as pd
    import pytest
    from benchlib.code_retrieval import bench as ws6a

    df = pd.DataFrame([
        {"qid": "q01", "arm": "agentic", "error": ""},
        {"qid": "q02", "arm": "agentic", "error": "RuntimeError: boom"},
    ])
    with pytest.raises(RuntimeError, match="q02/agentic"):
        ws6a.assert_no_errors(df)


def test_assert_no_errors_passes_when_all_clean():
    import pandas as pd
    from benchlib.code_retrieval import bench as ws6a

    df = pd.DataFrame([{"qid": "q01", "arm": "agentic", "error": ""}])
    ws6a.assert_no_errors(df)  # must not raise


# ---------------------------------------------------------------- I-1 ----


def test_manifest_record_keys_relative_to_data_dir_when_under_it(tmp_path, monkeypatch):
    from benchlib import manifest

    monkeypatch.setattr(manifest, "DATA_DIR", tmp_path)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", tmp_path / "MANIFEST.json")
    f = tmp_path / "sub" / "file.bin"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"hello")
    digest = manifest.record(f)
    m = manifest.load_manifest()
    assert "sub/file.bin" in m["files"]
    assert m["files"]["sub/file.bin"]["sha256"] == digest


def test_manifest_record_keys_relative_to_repo_root_when_outside_data_dir(
        tmp_path, monkeypatch):
    """I-1: relative_to(DATA_DIR) used to raise ValueError for any results/
    path -- reproduced live by build_ws6a_corpus.py AFTER a ~2,900-call paid
    run had already completed. Must not raise; must key relative to REPO_ROOT.
    """
    from benchlib import manifest

    repo_root = tmp_path
    data_dir = repo_root / "data"
    data_dir.mkdir()
    monkeypatch.setattr(manifest, "DATA_DIR", data_dir)
    monkeypatch.setattr(manifest, "REPO_ROOT", repo_root)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", data_dir / "MANIFEST.json")

    results_dir = repo_root / "results" / "ws6a"
    results_dir.mkdir(parents=True)
    f = results_dir / "repo_stats.csv"
    f.write_text("a,b\n1,2\n")

    digest = manifest.record(f)  # must not raise
    m = manifest.load_manifest()
    assert "results/ws6a/repo_stats.csv" in m["files"]
    assert m["files"]["results/ws6a/repo_stats.csv"]["sha256"] == digest


def test_manifest_verify_finds_files_recorded_outside_data_dir(tmp_path, monkeypatch):
    from benchlib import manifest

    repo_root = tmp_path
    data_dir = repo_root / "data"
    data_dir.mkdir()
    monkeypatch.setattr(manifest, "DATA_DIR", data_dir)
    monkeypatch.setattr(manifest, "REPO_ROOT", repo_root)
    monkeypatch.setattr(manifest, "MANIFEST_PATH", data_dir / "MANIFEST.json")

    results_dir = repo_root / "results" / "ws6a"
    results_dir.mkdir(parents=True)
    f = results_dir / "repo_stats.csv"
    f.write_text("a,b\n1,2\n")
    manifest.record(f)

    assert manifest.verify() == []

    # A hash mismatch outside DATA_DIR must still be caught.
    f.write_text("a,b\n1,3\n")
    bad = manifest.verify()
    assert any("HASH MISMATCH" in b and "repo_stats.csv" in b for b in bad)


# ---------------------------------------------------------------- I-4 ----


def test_compact_blocks_formats_text_tool_use_and_truncates_tool_result():
    from benchlib import agent_harness

    class FakeText:
        type = "text"
        text = "The answer is fastapi/routing.py."

    class FakeToolUse:
        type = "tool_use"
        name = "grep"
        input = {"pattern": "APIRoute"}

    long_result = "x" * 3000
    tool_result_block = {"type": "tool_result", "tool_use_id": "toolu_1",
                         "content": long_result, "is_error": False}

    out = agent_harness._compact_blocks([FakeText(), FakeToolUse(), tool_result_block])

    assert out[0] == {"type": "text", "text": "The answer is fastapi/routing.py."}
    assert out[1] == {"type": "tool_use", "name": "grep", "input": {"pattern": "APIRoute"}}
    assert out[2]["type"] == "tool_result"
    assert len(out[2]["content"]) == 2000
    assert out[2]["content_length"] == 3000
    assert out[2]["is_error"] is False


def test_execute_run_writes_one_transcript_line_per_turn_not_the_aggregate_row(
        tmp_path, monkeypatch):
    """I-4: the transcript file must be the per-turn audit trail (role,
    stop_reason, per-turn usage, compact tool_use/tool_result content), not a
    byte-duplicate of the checkpoint row.
    """
    import json
    from benchlib import agent_harness
    from benchlib.code_retrieval import bench as ws6a

    checkpoint_path = tmp_path / "ck.jsonl"
    transcript_dir = tmp_path / "transcripts"

    def fake_run_arm(client, arm, qid, question, **kwargs):
        r = agent_harness.RunResult(
            qid=qid, arm=arm, answer="It's in fastapi/routing.py.",
            turns=2, hit_turn_cap=False, stop_reason="end_turn",
            wall_clock_s=0.2, started_at="2026-08-17T12:00:00Z", error="",
            usage=ws6a.Usage(input_tokens=500, output_tokens=80),
        )
        r.turns_detail = [
            {"role": "assistant", "stop_reason": "tool_use",
             "usage": {"input_tokens": 300, "output_tokens": 40,
                       "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
             "content": [{"type": "tool_use", "name": "grep",
                         "input": {"pattern": "APIRoute"}}]},
            {"role": "user",
             "usage": None,
             "content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                          "content": "fastapi/routing.py:10: class APIRoute",
                          "content_length": 39, "is_error": False}]},
            {"role": "assistant", "stop_reason": "end_turn",
             "usage": {"input_tokens": 200, "output_tokens": 40,
                       "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
             "content": [{"type": "text", "text": "It's in fastapi/routing.py."}]},
        ]
        return r

    def fake_judge_answer(client, prompt_template, question, expected_files,
                          expected_symbols, answer, **kwargs):
        return {"judge_correct": True, "judge_reason": "ok",
                "usage": ws6a.Usage(input_tokens=10, output_tokens=5)}

    monkeypatch.setattr(agent_harness, "run_arm", fake_run_arm)
    monkeypatch.setattr(agent_harness, "judge_answer", fake_judge_answer)

    agent_harness.execute_run(
        client=None, questions_df=_fix_wave_questions_df(), arms=["agentic"],
        root=tmp_path, prefix_text="", mcp_tools=None, prices=_fix_wave_prices(),
        checkpoint_path=checkpoint_path,
        judge_prompt="Question: {question}\nAnswer: {answer}",
        governor=ws6a.CostGovernor(limit=100.0),
        transcript_dir=transcript_dir, seed=42,
    )

    transcript_file = transcript_dir / "agentic" / "q01.jsonl"
    lines = transcript_file.read_text().strip().splitlines()
    assert len(lines) == 3  # one per turns_detail entry, not one aggregate row
    parsed = [json.loads(l) for l in lines]
    assert parsed[0]["role"] == "assistant"
    assert parsed[0]["usage"]["input_tokens"] == 300
    assert "cost_billed" not in parsed[0]
    assert "cost_total_billed" not in parsed[0]
    assert parsed[1]["content"][0]["type"] == "tool_result"
    assert parsed[2]["content"][0]["type"] == "text"


# ---------------------------------------------------------------- I-7 ----


def test_make_agentic_tools_raises_on_rg_present_but_nonfunctional(tmp_path, monkeypatch):
    """A file named `rg` that exists on PATH but isn't functional ripgrep (a
    broken stub, a name collision) must not satisfy the guard -- shutil.which
    alone would let it through and produce uniformly-wrong grep output for
    every agentic-arm pair, the most likely live trigger for C-2's fabricated
    0%-accuracy row.
    """
    import os
    import pytest
    from benchlib import agent_harness

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "rg"
    stub.write_text("#!/bin/sh\nexit 1\n")  # exists, executable, --version fails
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")

    with pytest.raises(RuntimeError, match="ripgrep"):
        agent_harness.make_agentic_tools(tmp_path)


def test_make_agentic_tools_succeeds_with_functional_stub_rg(tmp_path, stub_ripgrep):
    from benchlib import agent_harness
    tools = agent_harness.make_agentic_tools(tmp_path)
    assert len(tools) == 3


# ---------------------------------------------------------------- I-2 ----


def test_total_spent_across_checkpoints_sums_every_phase(tmp_path):
    """The governor must be seeded from spend across every phase's checkpoint
    -- otherwise the smoke run, the full run, and each scaling sweep each get
    their own fresh $75, ~4x the spec's cap.
    """
    from benchlib import agent_harness

    (tmp_path / "ws6a_cache").mkdir()
    agent_harness.append_checkpoint(
        tmp_path / "ws6a_checkpoint.jsonl",
        {"qid": "q01", "arm": "agentic", "cost_total_billed": 1.0})
    agent_harness.append_checkpoint(
        tmp_path / "ws6a_smoke_checkpoint.jsonl",
        {"qid": "q02", "arm": "agentic", "cost_total_billed": 0.5})
    agent_harness.append_checkpoint(
        tmp_path / "ws6a_scaling_S.jsonl",
        {"qid": "q03", "arm": "indexed", "cost_total_billed": 2.25})
    agent_harness.append_checkpoint(
        tmp_path / "ws6a_cache" / "ws6a_l.jsonl",
        {"qid": "q04", "arm": "indexed", "cost_total_billed": 0.1})
    # A file that isn't a WS6a checkpoint must not be picked up.
    agent_harness.append_checkpoint(
        tmp_path / "unrelated.jsonl",
        {"qid": "qX", "arm": "y", "cost_total_billed": 100.0})

    assert agent_harness.total_spent_across_checkpoints(tmp_path) == 3.85


# ---------------------------------------------------------------- I-3 ----


class _FakeUsage:
    def __init__(self, input_tokens=10, output_tokens=5):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0


class _FakeTextBlock:
    type = "text"

    def __init__(self, text="The answer."):
        self.text = text


class _FakeToolUseBlock:
    type = "tool_use"

    def __init__(self, name="grep", input=None, id="toolu_1"):
        self.name = name
        self.input = input or {"pattern": "x"}
        self.id = id


class _FakeMessage:
    def __init__(self, role, stop_reason, content):
        self.role = role
        self.stop_reason = stop_reason
        self.usage = _FakeUsage()
        self.content = content


class _FakeToolRunner:
    """Mimics the shape of client.beta.messages.tool_runner() enough for
    run_arm: iterable of messages, plus generate_tool_call_response() for the
    message most recently yielded."""

    def __init__(self, messages):
        self._messages = messages
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self._messages):
            raise StopIteration
        m = self._messages[self._i]
        self._i += 1
        return m

    def generate_tool_call_response(self):
        m = self._messages[self._i - 1]
        if any(getattr(b, "type", None) == "tool_use" for b in m.content):
            return {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": "ok", "is_error": False}]}
        return None


class _FakeClient:
    def __init__(self, messages):
        self._messages = messages

        class _Beta:
            class messages:
                @staticmethod
                def tool_runner(**kwargs):
                    return _FakeToolRunner(list(self._messages))
        self.beta = _Beta()


def test_hit_turn_cap_is_false_when_the_capped_turn_finished_naturally(
        tmp_path, stub_ripgrep, monkeypatch):
    """I-3: reaching WS6A_TURN_CAP on a message that was itself the final
    answer (stop_reason != 'tool_use') must NOT be flagged as capped -- the
    model was not cut off, it happened to finish on the last allowed turn.
    """
    from benchlib import agent_harness
    monkeypatch.setattr("benchlib.config.WS6A_TURN_CAP", 3)

    messages = (
        [_FakeMessage("assistant", "tool_use", [_FakeToolUseBlock()])] * 2
        + [_FakeMessage("assistant", "end_turn", [_FakeTextBlock("Done.")])]
    )
    client = _FakeClient(messages)
    result = agent_harness.run_arm(client, "agentic", "q01", "where?", root=tmp_path)

    assert result.error == ""
    assert result.turns == 3
    assert result.hit_turn_cap is False


def test_hit_turn_cap_is_true_when_the_capped_turn_still_wanted_to_continue(
        tmp_path, stub_ripgrep, monkeypatch):
    """I-3: if the message at the cap still has stop_reason == 'tool_use',
    the model was genuinely cut off -- that IS a capped run."""
    from benchlib import agent_harness
    monkeypatch.setattr("benchlib.config.WS6A_TURN_CAP", 3)

    messages = [_FakeMessage("assistant", "tool_use", [_FakeToolUseBlock()])] * 3
    client = _FakeClient(messages)
    result = agent_harness.run_arm(client, "agentic", "q01", "where?", root=tmp_path)

    assert result.error == ""
    assert result.turns == 3
    assert result.hit_turn_cap is True


def test_execute_run_forces_capped_runs_incorrect_but_keeps_answer_and_verdict(
        tmp_path, monkeypatch):
    """I-3 second half: a capped run must be scored incorrect (all three
    programmatic-hit flags forced False) but the answer text and judge
    verdict must still be recorded, so the divergence between 'judge liked
    it' and 'forced incorrect because capped' stays visible.
    """
    from benchlib import agent_harness
    from benchlib.code_retrieval import bench as ws6a

    checkpoint_path = tmp_path / "ck.jsonl"

    def fake_run_arm(client, arm, qid, question, **kwargs):
        # Answer genuinely contains the expected file -- would score correct
        # if not for the cap.
        return agent_harness.RunResult(
            qid=qid, arm=arm, answer="It's in fastapi/routing.py (APIRoute).",
            turns=15, hit_turn_cap=True, stop_reason="tool_use",
            wall_clock_s=1.0, started_at="2026-08-17T12:00:00Z", error="",
            usage=ws6a.Usage(input_tokens=500, output_tokens=80),
        )

    def fake_judge_answer(client, prompt_template, question, expected_files,
                          expected_symbols, answer, **kwargs):
        return {"judge_correct": True, "judge_reason": "mentions the file",
                "usage": ws6a.Usage(input_tokens=10, output_tokens=5)}

    monkeypatch.setattr(agent_harness, "run_arm", fake_run_arm)
    monkeypatch.setattr(agent_harness, "judge_answer", fake_judge_answer)

    df = agent_harness.execute_run(
        client=None, questions_df=_fix_wave_questions_df(), arms=["agentic"],
        root=tmp_path, prefix_text="", mcp_tools=None, prices=_fix_wave_prices(),
        checkpoint_path=checkpoint_path,
        judge_prompt="Question: {question}\nAnswer: {answer}",
        governor=ws6a.CostGovernor(limit=100.0), seed=42,
    )
    row = df.iloc[0]
    assert row["hit_turn_cap"] == True
    assert row["any_file_hit"] == False
    assert row["all_files_hit"] == False
    assert row["symbol_hit"] == False
    # The answer and judge verdict must still be visible.
    assert row["answer"] == "It's in fastapi/routing.py (APIRoute)."
    assert row["judge_correct"] == True


# ---------------------------------------------------------------- I-8 ----


def _mix_rows(n_locate=15, n_trace=15, n_multi_hop=10, n_scaling=10):
    rows = []
    i = 0
    for diff, n in (("locate", n_locate), ("trace", n_trace),
                    ("multi_hop", n_multi_hop)):
        for _ in range(n):
            rows.append(_q(qid=f"q{i:02d}", difficulty=diff,
                           scaling_subset="true" if i < n_scaling else "false"))
            i += 1
    return rows


def test_validate_questions_accepts_the_full_pinned_mix():
    rows = _mix_rows()
    assert ws6a.validate_questions(rows, {"fastapi/a.py"}) == []


def test_validate_questions_rejects_wrong_total_count():
    rows = _mix_rows(n_locate=14)  # 39 total, not WS6A_N_QUESTIONS (40)
    problems = ws6a.validate_questions(rows, {"fastapi/a.py"})
    assert any("40" in p and ("question" in p.lower()) for p in problems)


def test_validate_questions_rejects_wrong_scaling_count():
    rows = _mix_rows(n_scaling=9)
    problems = ws6a.validate_questions(rows, {"fastapi/a.py"})
    assert any("scaling" in p.lower() and "10" in p for p in problems)


def test_validate_questions_rejects_wrong_difficulty_mix():
    rows = _mix_rows(n_locate=16, n_trace=14)  # still 40 total, wrong mix
    problems = ws6a.validate_questions(rows, {"fastapi/a.py"})
    assert any("locate" in p and "15" in p for p in problems)


def test_validate_questions_rejects_empty_expected_symbols():
    rows = _mix_rows()
    rows[0] = _q(qid=rows[0]["qid"], difficulty=rows[0]["difficulty"],
                scaling_subset=rows[0]["scaling_subset"], expected_symbols="")
    problems = ws6a.validate_questions(rows, {"fastapi/a.py"})
    assert any("expected_symbols" in p for p in problems)


def test_validate_questions_check_cardinality_false_skips_the_new_checks():
    # A lone row would fail every cardinality check; opting out must suppress
    # all of them, leaving only per-row problems (there are none here).
    assert ws6a.validate_questions([_q()], {"fastapi/a.py"},
                                   check_cardinality=False) == []


# ---------------------------------------------------------- cheap minors ----


def test_git_ls_files_handles_paths_with_spaces(tmp_path):
    import subprocess
    from benchlib.code_retrieval import bench as ws6a

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "a file.py").write_text("x")
    (tmp_path / "b.py").write_text("y")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    files = ws6a.git_ls_files(tmp_path)
    assert "a file.py" in files
    assert "b.py" in files
    assert len(files) == 2


def test_load_prefix_text_default_manifest_path_is_under_data_dir():
    """The default manifest_path must not be CWD-relative -- running from
    notebooks/ (or anywhere but the repo root) must still read the right
    file."""
    import inspect
    from benchlib import agent_harness
    from benchlib.config import DATA_DIR

    default = inspect.signature(agent_harness.load_prefix_text).parameters[
        "manifest_path"].default
    assert str(default) == str(DATA_DIR / "ws6a_prefix_manifest.json")


def test_gitignore_allows_ws6a_prefix_manifest():
    """arm (c)'s exact file composition must ship in the public repo."""
    import subprocess
    from benchlib.config import REPO_ROOT

    result = subprocess.run(
        ["git", "check-ignore", "-q", "data/ws6a_prefix_manifest.json"],
        cwd=REPO_ROOT)
    # git check-ignore exits 1 when the path is NOT ignored.
    assert result.returncode == 1


# ------------------------------------------------------ deferred-minor #7 ----


def test_load_pricing_covers_both_pinned_models():
    from benchlib.code_retrieval import bench as ws6a
    from benchlib.config import WS6A_AGENT_MODEL, WS6A_JUDGE_MODEL

    prices = ws6a.load_pricing("results/ws6a/pricing.csv", regime="standard")
    assert WS6A_AGENT_MODEL in prices
    assert WS6A_JUDGE_MODEL in prices


# --------------------------------------------- token accounting: I-5 and I-6


class _FakeCountClient:
    """Counts 1 token per character plus a fixed per-request framing cost.

    Stands in for messages.count_tokens so the framing correction can be
    tested without a key. The framing constant is the whole point of the
    fixture: a real request carries one, and the code under test must remove
    exactly one per batch -- no more, no fewer.
    """

    def __init__(self, framing: int = 11):
        self.framing = framing
        self.calls = []

        class _Messages:
            def __init__(self, outer):
                self.outer = outer

            def count_tokens(self, *, model, messages):
                text = messages[0]["content"]
                self.outer.calls.append(text)

                class _R:
                    input_tokens = self.outer.framing + len(text)

                return _R()

        self.messages = _Messages(self)


def test_measure_request_overhead_recovers_the_framing_constant():
    from benchlib import agent_harness

    client = _FakeCountClient(framing=11)
    assert agent_harness.measure_request_overhead(client, "m") == 11
    assert len(client.calls) == 3, "three probes: c(A), c(B), c(A+B)"


def test_count_tokens_batched_subtracts_overhead_once_per_batch():
    """One batch -> one framing constant removed; two batches -> two.

    Without this, a corpus counted file-by-file carries the constant once per
    file (~2,900 times for fastapi) while the same corpus counted in one call
    carries it once -- the I-6 disagreement between two 'repo token totals'.
    """
    from benchlib import agent_harness

    client = _FakeCountClient(framing=11)
    # Single batch: 3 + 4 = 7 chars of content.
    assert agent_harness.count_tokens_batched(
        client, "m", ["abc", "defg"], max_chars=100, overhead=11) == 7
    # Uncorrected, the same call still reports the raw API figure.
    assert agent_harness.count_tokens_batched(
        client, "m", ["abc", "defg"], max_chars=100) == 18

    # Forced into two batches: still exactly 7 tokens of content.
    assert agent_harness.count_tokens_batched(
        client, "m", ["abc", "defg"], max_chars=4, overhead=11) == 7


def test_prefix_unit_text_reproduces_load_prefix_text(tmp_path):
    """The measured unit and the sent string must be the same string.

    I-5: selection and reporting used bare file bodies while the context
    window received "===== path =====" headers and blank-line separators.
    Building load_prefix_text out of prefix_unit_text is what stops those two
    definitions drifting apart again.
    """
    import json
    from benchlib import agent_harness

    (tmp_path / "fastapi").mkdir()
    (tmp_path / "fastapi" / "a.py").write_text("import os\n")
    (tmp_path / "fastapi" / "b.py").write_text("x = 1\n")
    files = ["fastapi/a.py", "fastapi/b.py"]

    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps({"files": files}))

    units = [agent_harness.prefix_unit_text(tmp_path, p) for p in files]
    text = agent_harness.load_prefix_text(tmp_path, manifest_path)

    assert "".join(units)[:-2] == text
    # Byte-identical to the original "\n\n".join(header + body) formulation --
    # arm (c)'s prompt cache depends on this string never moving.
    legacy = "\n\n".join(
        f"===== {p} =====\n" + (tmp_path / p).read_text() for p in files)
    assert text == legacy
    assert text.startswith("===== fastapi/a.py =====\nimport os\n")


def test_load_prefix_text_handles_an_empty_manifest(tmp_path):
    """Slicing off the trailing separator must not eat the last character of
    an empty prefix (or raise)."""
    import json
    from benchlib import agent_harness

    manifest_path = tmp_path / "m.json"
    manifest_path.write_text(json.dumps({"files": []}))
    assert agent_harness.load_prefix_text(tmp_path, manifest_path) == ""


def _run_row(qid, arm, error=""):
    return {"qid": qid, "arm": arm, "error": error}


def test_assert_run_complete_passes_on_a_full_grid():
    import pandas as pd
    df = pd.DataFrame([_run_row(q, a) for q in ("q01", "q02")
                       for a in ("indexed", "agentic")])
    ws6a.assert_run_complete(df, ["q01", "q02"], ("indexed", "agentic"))


def test_assert_run_complete_catches_a_wholly_missing_arm():
    """The C-2 failure mode: a broken arm leaves NO rows, so assert_no_errors
    sees a clean frame and a 120-row runs.csv looks like a finished 160-pair
    experiment."""
    import pandas as pd
    import pytest

    df = pd.DataFrame([_run_row(q, "indexed") for q in ("q01", "q02")])
    ws6a.assert_no_errors(df)  # passes -- this is exactly the problem
    with pytest.raises(RuntimeError, match="agentic"):
        ws6a.assert_run_complete(df, ["q01", "q02"], ("indexed", "agentic"))


def test_assert_run_complete_catches_a_duplicated_pair():
    import pandas as pd
    import pytest

    df = pd.DataFrame([_run_row("q01", "indexed"), _run_row("q01", "indexed")])
    with pytest.raises(RuntimeError, match="duplicated"):
        ws6a.assert_run_complete(df, ["q01"], ("indexed",))


def _summary_row(arm, qid, judge, cost_agent, cost_uncached, cost_judge):
    return {
        "arm": arm, "qid": qid, "input_tokens": 500, "output_tokens": 20,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
        "turns": 1, "hit_turn_cap": False, "wall_clock_s": 1.0,
        "judge_correct": judge, "any_file_hit": judge, "all_files_hit": judge,
        "cost_agent_billed": cost_agent, "cost_judge_billed": cost_judge,
        "cost_total_billed": round(cost_agent + cost_judge, 6),
        "cost_uncached_list": cost_uncached,
    }


def test_summarise_runs_reports_lift_over_parametric():
    import pandas as pd
    df = pd.DataFrame([
        {**_summary_row("parametric", "q1", True, 0.01, 0.01, 0.004),
         "input_tokens": 100},
        {**_summary_row("parametric", "q2", False, 0.01, 0.01, 0.004),
         "input_tokens": 100},
        _summary_row("indexed", "q1", True, 0.05, 0.05, 0.004),
        _summary_row("indexed", "q2", True, 0.05, 0.05, 0.004),
    ])
    s = ws6a.summarise_runs(df).set_index("arm")
    assert s.loc["parametric", "judge_accuracy"] == 0.5
    assert s.loc["indexed", "judge_accuracy"] == 1.0
    # Lift is measured against the parametric floor, not against zero.
    assert s.loc["indexed", "judge_lift_over_parametric"] == 0.5
    assert s.loc["parametric", "judge_lift_over_parametric"] == 0.0
    assert s.loc["indexed", "total_input_tokens"] == 1000


def test_cache_savings_ratio_is_agent_only_on_both_sides():
    """C-1, locked down. The parametric arm uses no cache at all, so its
    caching saving must be exactly zero. Dividing cost_total_billed (agent +
    Opus judge) by cost_uncached_list (agent only) instead reports it as made
    ~160% MORE expensive BY caching -- and distorts the stuffed arm by a few
    points, which is worse for being invisible."""
    import pandas as pd
    df = pd.DataFrame([
        _summary_row("parametric", "q1", True, 0.010, 0.010, 0.016),
        # A cached arm: billed 0.30 against a 1.00 uncached list price.
        _summary_row("stuffed", "q1", True, 0.300, 1.000, 0.016),
    ])
    s = ws6a.summarise_runs(df).set_index("arm")
    assert s.loc["parametric", "cache_savings_ratio"] == 0.0
    assert s.loc["stuffed", "cache_savings_ratio"] == 0.7


def test_mcp_env_uses_the_variable_claude_context_actually_reads(monkeypatch):
    """@zilliz/claude-context-mcp@0.1.15 recognises exactly one collection-naming
    variable, CODE_CHUNKS_COLLECTION_NAME_OVERRIDE (verified by reading its
    dist/config.js). The plan specified COLLECTION_NAME_PREFIX, which the
    package ignores -- under that name every scope would land in an auto-named
    hybrid_code_chunks_<pathHash> collection and the S/M/L sweep could not tell
    its own indexes apart."""
    from benchlib import agent_harness

    for var in ("MILVUS_ADDRESS", "MILVUS_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.setenv(var, "x")
    env = agent_harness.mcp_env("/tmp/repo", "ws6a_s")

    assert env["CODE_CHUNKS_COLLECTION_NAME_OVERRIDE"] == "ws6a_s"
    assert "COLLECTION_NAME_PREFIX" not in env
    # Background sync and the trigger watcher default to ON and would re-embed
    # the corpus mid-benchmark -- spending outside the governor and moving the
    # index under a run that is measuring a fixed one.
    assert env["CLAUDE_CONTEXT_BACKGROUND_SYNC"] == "false"
    assert env["CLAUDE_CONTEXT_TRIGGER_WATCHER"] == "false"
    assert env["EMBEDDING_MODEL"] == config.WS6A_EMBED_MODEL
    assert env["EMBEDDING_PROVIDER"] == config.WS6A_EMBED_PROVIDER


def test_mcp_env_raises_rather_than_indexing_into_the_wrong_place(monkeypatch):
    import pytest
    from benchlib import agent_harness

    monkeypatch.delenv("MILVUS_ADDRESS", raising=False)
    monkeypatch.setenv("MILVUS_TOKEN", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    with pytest.raises(RuntimeError, match="MILVUS_ADDRESS"):
        agent_harness.mcp_env("/tmp/repo", "ws6a_l")


# ------------------------------------------- judge robustness (thinking budget)


class _FakeJudgeClient:
    """Judge client that returns thinking-only responses for the first N calls.

    Reproduces the live failure: claude-opus-5 runs adaptive thinking by
    default and max_tokens caps thinking + output together, so a tight budget
    can yield a response whose only block is `thinking` -- no text at all.
    """

    def __init__(self, empty_for: int, verdict='{"correct": true, "reason": "ok"}'):
        self.empty_for = empty_for
        self.verdict = verdict
        self.budgets = []

        class _Blk:
            def __init__(self, type_, text=""):
                self.type = type_
                self.text = text

        class _Usage:
            input_tokens = 100
            output_tokens = 50
            cache_creation_input_tokens = 0
            cache_read_input_tokens = 0

        class _Messages:
            def __init__(self, outer):
                self.outer = outer

            def create(self, *, model, max_tokens, messages, output_config):
                o = self.outer
                o.budgets.append(max_tokens)
                n = len(o.budgets)

                class _R:
                    stop_reason = "max_tokens" if n <= o.empty_for else "end_turn"
                    usage = _Usage()
                    content = ([_Blk("thinking")] if n <= o.empty_for
                               else [_Blk("thinking"), _Blk("text", o.verdict)])

                return _R()

        self.messages = _Messages(self)


def test_judge_retries_with_a_bigger_budget_when_thinking_ate_it_all():
    from benchlib import agent_harness

    client = _FakeJudgeClient(empty_for=1)
    out = agent_harness.judge_answer(client, "{question}{expected_files}"
                                   "{expected_symbols}{answer}",
                                   "q?", ["a.py"], ["f"], "an answer")
    assert out["judge_correct"] is True
    assert client.budgets == list(agent_harness.JUDGE_MAX_TOKENS_SCHEDULE[:2])
    # Both attempts were billed, so both must be counted.
    assert out["usage"].output_tokens == 100


def test_judge_raises_rather_than_fabricating_a_wrong_answer():
    """A judge failure must never be recorded as judge_correct=False -- that
    fabricates a result. It raises, and execute_run skips the checkpoint so a
    resumed run retries the pair (the C-2 rule)."""
    import pytest
    from benchlib import agent_harness

    client = _FakeJudgeClient(empty_for=99)
    with pytest.raises(agent_harness.JudgeError):
        agent_harness.judge_answer(client, "{question}{expected_files}"
                                 "{expected_symbols}{answer}",
                                 "q?", ["a.py"], ["f"], "an answer")


def test_judge_empty_answer_short_circuits_without_calling_the_api():
    from benchlib import agent_harness

    client = _FakeJudgeClient(empty_for=99)
    out = agent_harness.judge_answer(client, "{question}{expected_files}"
                                   "{expected_symbols}{answer}",
                                   "q?", ["a.py"], ["f"], "   ")
    assert out["judge_correct"] is False
    assert client.budgets == [], "no API call for an empty answer"


def test_summarise_runs_prompt_tokens_include_cached_content():
    """input_tokens is the UNCACHED REMAINDER, not the prompt size.

    Arm (c) sends ~960k tokens per question and reports a median input_tokens
    of 28 because the rest is served from cache. Ranking arms on input_tokens
    would put the stuffed arm first as the cheapest -- the exact inversion this
    column exists to prevent."""
    import pandas as pd
    df = pd.DataFrame([
        {**_summary_row("stuffed", "q1", True, 0.2, 1.0, 0.01),
         "input_tokens": 28, "cache_read_input_tokens": 960_000},
        {**_summary_row("agentic", "q1", True, 0.02, 0.02, 0.01),
         "input_tokens": 9_485, "cache_read_input_tokens": 0},
    ])
    s = ws6a.summarise_runs(df).set_index("arm")
    assert s.loc["stuffed", "median_prompt_tokens"] == 960_028
    assert s.loc["agentic", "median_prompt_tokens"] == 9_485
    # The naive column still ranks stuffed as smallest -- that is why it must
    # never be the basis of a token comparison.
    assert s.loc["stuffed", "median_input_tokens"] < s.loc["agentic", "median_input_tokens"]
    assert s.loc["stuffed", "median_prompt_tokens"] > s.loc["agentic", "median_prompt_tokens"]
