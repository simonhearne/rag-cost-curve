"""Unit tests for the pure half of WS6c. No network, no API keys."""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchlib import config
from benchlib.code_retrieval import topk as ws6c


def _runs(**kw):
    """Two arms x 4 questions, prompt_tokens supplied per arm."""
    rows = []
    for arm, vals in kw.items():
        for i, v in enumerate(vals, start=1):
            rows.append({"qid": f"q{i:02d}", "arm": arm,
                         "input_tokens": v, "cache_read_input_tokens": 0,
                         "cache_creation_input_tokens": 0, "judge_correct": True})
    return pd.DataFrame(rows)


def test_ws6c_pins_are_present_and_frozen():
    assert config.WS6C_TOPK == 3
    assert config.WS6C_BOOT_N == 10_000
    assert config.WS6C_BOOT_SEED == 42
    assert config.WS6C_ALPHA == 0.05
    # Raised from 40.00 on explicit user authorisation mid-Phase-3; see
    # the comment at the constant. No other WS6c pin moved.
    assert config.WS6C_BUDGET_USD == 45.00
    # WS6a pins must be untouched by this workstream.
    assert config.WS6A_COMMIT == "a1fa70d4237d50aae6586a0d9b229df583463d21"
    assert config.WS6A_TURN_CAP == 15


def test_paired_deltas_are_ordered_by_qid_and_aligned():
    df = _runs(indexed=[100, 200, 300, 400], agentic=[10, 20, 30, 40])
    qids, d = ws6c.paired_deltas(df, "prompt_tokens", "indexed", "agentic")
    assert qids == ["q01", "q02", "q03", "q04"]
    np.testing.assert_array_equal(d, np.array([90.0, 180.0, 270.0, 360.0]))


def test_paired_deltas_uses_prompt_tokens_not_input_tokens():
    """prompt_tokens = input + cache_read + cache_creation. A cached arm whose
    input_tokens is near zero must not read as the cheapest."""
    df = pd.DataFrame([
        {"qid": "q01", "arm": "stuffed", "input_tokens": 28,
         "cache_read_input_tokens": 900_000, "cache_creation_input_tokens": 60_000,
         "judge_correct": True},
        {"qid": "q01", "arm": "agentic", "input_tokens": 9_000,
         "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
         "judge_correct": True},
    ])
    _, d = ws6c.paired_deltas(df, "prompt_tokens", "stuffed", "agentic")
    # 28 + 900_000 + 60_000 - 9_000 = 951_028 (the brief's own constant,
    # 951_000, is an arithmetic slip against its stated formula -- corrected
    # here).
    assert d[0] == pytest.approx(951_028.0)


def test_paired_deltas_rejects_unpaired_rows():
    df = _runs(indexed=[1, 2, 3], agentic=[1, 2])
    with pytest.raises(ValueError, match="unpaired"):
        ws6c.paired_deltas(df, "prompt_tokens", "indexed", "agentic")


def test_paired_summary_ci_excludes_zero_for_a_clear_shift():
    d = np.full(40, 7000.0) + np.arange(40)
    s = ws6c.paired_summary(d, n_boot=2000, seed=42, alpha=0.05)
    assert s["n"] == 40
    assert s["ci_low"] > 0
    assert s["excludes_zero"] is True
    assert s["n_positive"] == 40


def test_paired_summary_ci_includes_zero_for_symmetric_noise():
    d = np.array([100.0, -100.0] * 20)
    s = ws6c.paired_summary(d, n_boot=2000, seed=42, alpha=0.05)
    assert s["ci_low"] < 0 < s["ci_high"]
    assert s["excludes_zero"] is False


def test_paired_summary_is_deterministic_under_the_seed():
    d = np.random.default_rng(7).normal(500, 100, 40)
    a = ws6c.paired_summary(d, n_boot=2000, seed=42, alpha=0.05)
    b = ws6c.paired_summary(d, n_boot=2000, seed=42, alpha=0.05)
    assert a == b


def test_sign_test_p_is_exact_and_two_sided():
    assert ws6c.sign_test_p(20, 20) == pytest.approx(2 * 0.5 ** 20)
    assert ws6c.sign_test_p(10, 20) == pytest.approx(1.0)
    assert ws6c.sign_test_p(0, 0) == 1.0


def test_resolve_branch_picks_by_sign_then_by_separation():
    labels = ("T1", "T2", "T3")
    assert ws6c.resolve_branch(True, 5000.0, labels) == "T1"
    assert ws6c.resolve_branch(True, -5000.0, labels) == "T2"
    assert ws6c.resolve_branch(False, 5000.0, labels) == "T3"


class _FakeTool:
    def __init__(self):
        self.name = "search_code"
        self.description = "search"
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "query": {"type": "string"},
                "limit": {"type": "number"},
                "extensionFilter": {"type": "string"},
            },
            "required": ["path", "query"],
        }


def test_topk_none_leaves_limit_in_the_schema():
    from benchlib import agent_harness
    raw = [_FakeTool()]
    tools = agent_harness.claude_context_tools(None, raw, "/tmp/repo", topk=None)
    props = tools[0].input_schema["properties"]
    assert "limit" in props, "default behaviour must match the frozen WS6a run"
    assert "path" not in props, "path is bound and hidden, as before"


def test_topk_set_removes_limit_from_the_schema():
    from benchlib import agent_harness
    raw = [_FakeTool()]
    tools = agent_harness.claude_context_tools(None, raw, "/tmp/repo", topk=3)
    props = tools[0].input_schema["properties"]
    assert "limit" not in props, (
        "a bound limit must be invisible to the model -- the byte-identical "
        "system prompt cannot name a parameter, so leaving it visible would "
        "let the model override the one variable under test")
    assert "path" not in props


def test_topk_binding_does_not_mutate_the_servers_schema():
    from benchlib import agent_harness
    raw = [_FakeTool()]
    before = dict(raw[0].input_schema["properties"])
    agent_harness.claude_context_tools(None, raw, "/tmp/repo", topk=3)
    assert raw[0].input_schema["properties"] == before


def test_ws6c_commits_are_full_shas():
    for slug, sha in config.WS6C_COMMITS.items():
        assert len(sha) == 40, f"{slug}: not a full sha"
        assert set(sha) <= set("0123456789abcdef"), f"{slug}: not hex"


def test_ws6c_commits_match_the_measured_corpus_stats():
    """A well-formed but wrong-or-stale SHA would pass the shape test above
    silently and unpin the corpus -- the exact risk named in
    scripts/code-retrieval/select_ws6c_corpus.py's own docstring. dtype=str so a SHA that is
    all digits can't be coerced to a number."""
    df = pd.read_csv(config.ws6c_dir() / "corpus_stats.csv", dtype=str)
    measured = dict(zip(df["slug"], df["commit"]))
    for slug, sha in config.WS6C_COMMITS.items():
        assert measured[slug] == sha, (
            f"{slug}: config.WS6C_COMMITS pins {sha}, but corpus_stats.csv "
            f"measured {measured[slug]}")


def test_total_spent_across_checkpoints_includes_ws6c_cache(tmp_path):
    """The governor must be cumulative across WS6a AND WS6c phases -- seeding
    it from ws6a-only checkpoints would let each WS6c script authorise its
    own fresh budget on top of WS6a's spend, reproducing the exact I-2
    failure total_spent_across_checkpoints exists to prevent."""
    from benchlib import agent_harness

    (tmp_path / "ws6a_cache").mkdir()
    agent_harness.append_checkpoint(
        tmp_path / "ws6a_cache" / "ws6a_l.jsonl",
        {"qid": "q01", "arm": "indexed", "cost_total_billed": 1.0})
    (tmp_path / "ws6c_cache").mkdir()
    agent_harness.append_checkpoint(
        tmp_path / "ws6c_cache" / "topk_checkpoint.jsonl",
        {"qid": "q02", "arm": "indexed_topk3", "cost_total_billed": 2.5})

    assert agent_harness.total_spent_across_checkpoints(tmp_path) == 3.5


# ------------------------------------------------------- the Phase 2b guard

REPO = Path(__file__).resolve().parent.parent.parent


def _throwaway_checkout(tmp_path, slug):
    """A one-commit git repo standing in for a pinned candidate checkout."""
    root = tmp_path / f"ws6c_{slug}"
    (root / "src").mkdir(parents=True)
    (root / "src" / "mod.py").write_text("VALUE = 1\n")
    run = lambda *a: subprocess.run(  # noqa: E731
        ["git", "-C", str(root), *a], check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    run("config", "commit.gpgsign", "false")
    run("add", "-A")
    run("commit", "-qm", "initial")
    sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return root, sha


@pytest.fixture
def gate_on_temp_checkout(tmp_path, monkeypatch):
    """Point the guard at a throwaway repo instead of the real corpora.

    The guard now lives in benchlib.code_retrieval.topk and is imported by BOTH
    scripts/code-retrieval/gate_ws6c.py (Phase 2b) and scripts/code-retrieval/index_ws6c.py (Phase 3), so
    these tests exercise the one shared implementation.
    """
    root, sha = _throwaway_checkout(tmp_path, "probe")
    monkeypatch.setattr(ws6c, "WS6C_CANDIDATES", ({"slug": "probe"},))
    monkeypatch.setattr(ws6c, "WS6C_COMMITS", {"probe": sha})
    monkeypatch.setattr(ws6c, "ws6c_checkout", lambda slug: tmp_path / f"ws6c_{slug}")
    return ws6c, root, sha


def _script_module(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / "code-retrieval" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("script", ["gate_ws6c", "index_ws6c"])
def test_scripts_share_one_pin_guard_rather_than_a_copy(script):
    """R8: a second verbatim copy of the guard is a guard that can drift out
    of agreement with itself. Identity, not just equal behaviour."""
    mod = _script_module(script)
    assert mod.assert_pinned_checkouts is ws6c.assert_pinned_checkouts


def test_pin_guard_passes_on_a_pinned_clean_checkout(gate_on_temp_checkout):
    mod, _root, _sha = gate_on_temp_checkout
    assert mod.checkout_problems("probe") == []
    mod.assert_pinned_checkouts()  # must not raise


def test_pin_guard_rejects_a_drifted_head(gate_on_temp_checkout, monkeypatch):
    mod, _root, _sha = gate_on_temp_checkout
    monkeypatch.setattr(mod, "WS6C_COMMITS", {"probe": "0" * 40})
    problems = mod.checkout_problems("probe")
    assert len(problems) == 1
    assert "config.WS6C_COMMITS pins" in problems[0]
    with pytest.raises(SystemExit):
        mod.assert_pinned_checkouts()


def test_pin_guard_rejects_a_modified_tracked_file(gate_on_temp_checkout):
    """HEAD is still the pinned SHA, so only the cleanliness check can see
    this -- and Phase 3 would index the edited bytes."""
    mod, root, _sha = gate_on_temp_checkout
    (root / "src" / "mod.py").write_text("VALUE = 2\n")
    problems = mod.checkout_problems("probe")
    assert len(problems) == 1
    assert "working tree is not clean" in problems[0]
    assert "src/mod.py" in problems[0], "the offending path must be named"


def test_pin_guard_rejects_an_untracked_file(gate_on_temp_checkout):
    """--porcelain includes untracked files by default, and it must: a stray
    file in a source tree is indexed like any other."""
    mod, root, _sha = gate_on_temp_checkout
    (root / "src" / "stray.py").write_text("SECRET = 1\n")
    problems = mod.checkout_problems("probe")
    assert len(problems) == 1
    assert "src/stray.py" in problems[0]


def test_pin_guard_reports_head_and_dirt_together(gate_on_temp_checkout, monkeypatch):
    """All problems collected before raising, so one run reports everything."""
    mod, root, _sha = gate_on_temp_checkout
    monkeypatch.setattr(mod, "WS6C_COMMITS", {"probe": "0" * 40})
    (root / "src" / "mod.py").write_text("VALUE = 2\n")
    problems = mod.checkout_problems("probe")
    assert len(problems) == 2
    assert any("pins" in p for p in problems)
    assert any("not clean" in p for p in problems)


def test_pin_guard_rejects_a_missing_checkout(gate_on_temp_checkout, monkeypatch):
    mod, _root, _sha = gate_on_temp_checkout
    monkeypatch.setattr(mod, "ws6c_checkout", lambda slug: REPO / "no" / "such" / slug)
    assert mod.checkout_problems("probe") == [
        f"probe: no checkout at {REPO / 'no' / 'such' / 'probe'}"]


def test_pin_guard_truncates_a_very_dirty_tree(gate_on_temp_checkout):
    """A guard that dumps hundreds of paths is as unreadable as one that
    names none, so the list is capped and the remainder counted."""
    mod, root, _sha = gate_on_temp_checkout
    for i in range(mod.DIRTY_PATHS_SHOWN + 5):
        (root / "src" / f"stray{i:02d}.py").write_text("x = 1\n")
    problems = mod.checkout_problems("probe")
    assert len(problems) == 1
    assert f"({mod.DIRTY_PATHS_SHOWN + 5} entries)" in problems[0]
    assert "and 5 more" in problems[0]


# ------------------------------------- scaling_prefix on validate_questions
#
# WS6a hardcoded `fastapi/` as the directory a scaling_subset question had to
# be answerable inside. WS6c's corpus is a different repository, so the prefix
# is a parameter -- but the default must still be WS6a's, byte for byte, or
# every WS6a caller silently changes meaning.


def _prefix_q(**kw):
    """One row shaped like a questions.csv record. Mirrors tests/code_retrieval/test_bench.py's
    `_q`, deliberately duplicated rather than imported: test_ws6a.py is frozen
    alongside results/ws6a/ and must not grow a dependency from WS6c."""
    base = {"qid": "q01", "question": "where?", "expected_files": "fastapi/a.py",
            "expected_symbols": "f", "difficulty": "locate", "category": "routing",
            "notes": "", "scaling_subset": "false", "excluded": "false",
            "exclusion_reason": ""}
    base.update(kw)
    return base


def test_scaling_prefix_defaults_to_fastapi_and_still_accepts_ws6a_rows():
    from benchlib.code_retrieval import bench as ws6a

    row = _prefix_q(scaling_subset="true")
    assert ws6a.validate_questions([row], {"fastapi/a.py"},
                                   check_cardinality=False) == []


def test_scaling_prefix_default_message_is_unchanged_for_ws6a_callers():
    """The exact string WS6a's own test greps for, reproduced by the default."""
    from benchlib.code_retrieval import bench as ws6a

    problems = ws6a.validate_questions(
        [_prefix_q(expected_files="tests/test_a.py", scaling_subset="true")],
        {"tests/test_a.py"}, check_cardinality=False)
    assert problems == [
        "q01: scaling_subset question has expected_files outside fastapi/ "
        "(['tests/test_a.py']); the S subset is fastapi/ only, so it would be "
        "unanswerable at S"]


def test_scaling_prefix_accepts_a_custom_prefix():
    from benchlib.code_retrieval import bench as ws6a

    row = _prefix_q(expected_files="src/agentic_hil/elfsymbols.py",
                    scaling_subset="true")
    assert ws6a.validate_questions(
        [row], {"src/agentic_hil/elfsymbols.py"}, check_cardinality=False,
        scaling_prefix="src/agentic_hil/") == []


def test_scaling_prefix_rejects_a_file_outside_the_custom_prefix():
    from benchlib.code_retrieval import bench as ws6a

    problems = ws6a.validate_questions(
        [_prefix_q(expected_files="tools/bench_battery.py",
                   scaling_subset="true")],
        {"tools/bench_battery.py"}, check_cardinality=False,
        scaling_prefix="src/agentic_hil/")
    assert len(problems) == 1
    assert "src/agentic_hil/" in problems[0]
    assert "fastapi/" not in problems[0]


def test_scaling_prefix_leaves_non_scaling_rows_alone():
    """A row that is not scaling_subset is unaffected by either prefix."""
    from benchlib.code_retrieval import bench as ws6a

    row = _prefix_q(expected_files="tools/bench_battery.py")
    assert ws6a.validate_questions(
        [row], {"tools/bench_battery.py"}, check_cardinality=False,
        scaling_prefix="src/agentic_hil/") == []


# ------------------------------- what claude-context will actually index
#
# The indexed arm can only retrieve from files the pinned package indexes,
# and that set is NOT our WS6A_TEXT_EXTENSIONS allowlist. Phase 3 uses this
# as the third, independent source in its index-completeness cross-check, so
# the filters have to match the package's own dist/ exactly.


def test_indexable_keeps_supported_code_and_markdown():
    paths = ["src/a.py", "src/b.ts", "README.md", "docs/guide.markdown",
             "nb/demo.ipynb"]
    assert ws6c.claude_context_indexable(paths) == paths


def test_indexable_drops_extensions_the_package_commented_out():
    """.yaml/.json/.toml/.sh/.txt are in OUR text allowlist and NOT in the
    package's -- the agentic arm can grep files the indexed arm never saw."""
    paths = ["conf.yaml", "pkg.json", "pyproject.toml", "run.sh", "notes.txt",
             "page.html", "site.css"]
    assert ws6c.claude_context_indexable(paths) == []


def test_indexable_drops_any_dot_prefixed_segment():
    """hasHiddenSegment in dist/utils/ignore-matcher.js -- this is why
    fastapi's .github/**/*.md files are absent from its index."""
    assert ws6c.claude_context_indexable(
        [".github/CONTRIBUTING.md", "src/.hidden/a.py", ".eslintrc.js"]) == []


def test_indexable_drops_ignored_directories_and_globs():
    assert ws6c.claude_context_indexable(
        ["node_modules/x/index.js", "dist/bundle.js", "build/a.py",
         "src/vendor.min.js", "src/app.bundle.js", "trace.map"]) == []
    # ...but only when the ignored name is a DIRECTORY segment, never the file.
    assert ws6c.claude_context_indexable(["src/build.py", "src/dist.md"]) == [
        "src/build.py", "src/dist.md"]


def test_indexable_reproduces_the_frozen_fastapi_index_exactly():
    """CALIBRATION. The server's own completion log reported 2,825 files for
    the frozen WS6a L index; this predictor must land on the same number, or
    it is not usable as an independent completeness check.

    Skipped rather than failed when the gitignored checkout is absent -- the
    clone is not part of this repo's history.
    """
    root = config.ws6a_checkout()
    if not (root / ".git").exists():
        pytest.skip("data/ws6a_fastapi checkout not present")
    from benchlib.code_retrieval import bench as ws6a

    reported = int(pd.read_csv(config.ws6a_dir() / "index_cost.csv")
                   .set_index("scope").loc["L", "indexed_files_reported"])
    predicted = ws6c.claude_context_indexable(ws6a.git_ls_files(root))
    assert len(predicted) == reported


# --------------------------------------------- arm LABEL vs behaviour KIND
#
# Phase 3 runs the `indexed` code path twice under two labels against ONE
# checkpoint. Without a separable kind the second arm would either dispatch
# as an unknown arm or overwrite the first arm's rows.


def test_run_arm_dispatches_on_kind_not_label():
    from benchlib import agent_harness

    # No client is ever touched: both paths fail before the first API call.
    without = agent_harness.run_arm(None, "indexed_topk3", "q1", "?",
                                  mcp_tools=None)
    assert "unknown arm 'indexed_topk3'" in without.error

    withkind = agent_harness.run_arm(None, "indexed_topk3", "q1", "?",
                                   mcp_tools=None, kind="indexed")
    assert "non-empty tool list" in withkind.error, (
        "kind='indexed' must reach the indexed/agentic branch")
    assert withkind.arm == "indexed_topk3", "the LABEL is what gets published"


def test_run_arm_default_kind_is_the_label_itself():
    """Every WS6a call site passes no kind and must be unchanged."""
    from benchlib import agent_harness

    assert "unknown arm 'nonsense'" in agent_harness.run_arm(
        None, "nonsense", "q1", "?").error
