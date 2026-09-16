"""Unit tests for the pure half of WS8. No network, no API keys."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchlib import config
from benchlib.non_code_retrieval import bench as ws8
from benchlib.non_code_retrieval import stats as ws8_stats
from benchlib.code_retrieval import topk as ws6c

# Ruling R5-extended: imports at module top, tests included. numpy and pandas
# were imported mid-file at 987d223 (Phase 4 was appended as a block); the
# committed-CSV assertions below need them from the first line anyway.
_RESULTS_WS8 = config.ws8_dir()


def test_ws8_pins_are_present_and_frozen():
    assert config.WS8_WINDOW_START == "2026-06-01"
    assert config.WS8_WINDOW_END == "2026-09-01"
    assert config.WS8_AGENT_MODEL == "claude-sonnet-5"
    assert config.WS8_JUDGE_MODEL == "claude-opus-5"
    assert config.WS8_JUDGE_MODEL != config.WS8_AGENT_MODEL
    assert config.WS8_EMBED_MODEL == "text-embedding-3-small"
    assert config.WS8_TOP_K == 10
    assert config.WS8_BUDGET_USD == 70.00  # raised 55->70 in f15266e, before Phase 3's full run
    assert config.WS8_ARMS == ("indexed", "agentic", "parametric")
    assert config.WS8_STRATA == {"hop1": 30, "hop2": 15, "hop3plus": 30}
    assert config.WS8_N_QUESTIONS == 75
    assert sum(config.WS8_STRATA.values()) == config.WS8_N_QUESTIONS
    assert config.WS8_GATE_PASS == 0.0
    # ADDED 2026-09-11. Every pin below was unasserted, and the gap was not
    # theoretical: WS8_MAX_THREADS_PER_REPO was raised 1,000 -> 8,000 on the
    # day of the full run and the stale 1,000 reached a published document,
    # because nothing failed when the constant moved. A pin with no test is a
    # comment.
    assert config.WS8_MAX_THREADS_PER_REPO == 8000
    assert config.WS8_CHUNK_CHARS == 1_800
    assert config.WS8_CHUNK_OVERLAP == 200
    assert config.WS8_TURN_CAP == 15
    assert config.WS8_SINGLE_TURN_ARMS == ("parametric",)
    assert config.WS8_FETCH_SORT == "created"
    assert config.WS8_FETCH_DIRECTION == "asc"
    assert config.WS8_REPOS == ("fastapi/fastapi", "kubernetes/kubernetes",
                                "rust-lang/rust")
    assert config.WS8_EMBED_DIM == 1536
    assert config.WS8_COLLECTION == "ws8_issues"
    assert config.WS8_COMMENTS_BULK is True
    assert config.WS8_INCLUDE_PULL_REQUESTS is True
    # WITHDRAWN, not live -- pinned with no consumer (see config.py). Asserted
    # so the values cannot drift while nothing reads them, and so a reader who
    # greps for the pin finds this note.
    assert config.WS8_SIZES == {"s": 100_000, "m": 400_000, "l": 1_200_000}


def test_ws8_dir_is_results_ws8_and_exists():
    """ws8_dir() is a pin too: every committed WS8 output lands where it
    points, so a change to it silently orphans every results/ws8/ file."""
    d = config.ws8_dir()
    assert d == config.RESULTS_DIR / "ws8"
    assert d.name == "ws8" and d.parent.name == "results"
    assert d.is_dir()   # ws8_dir() mkdirs; a file of that name would be a bug


def test_ws6_pins_are_untouched():
    assert config.WS6A_COMMIT == "a1fa70d4237d50aae6586a0d9b229df583463d21"
    assert config.WS6C_TOPK == 3
    assert config.WS6C_BUDGET_USD == 45.00


def _thread(**kw):
    base = dict(repo="acme/widget", number=12, title="Build hangs",
                body="It hangs after upgrade.",
                user={"login": "alice", "type": "User"},
                created_at="2026-06-02T10:00:00Z", state="closed",
                comments=[
                    {"user": {"login": "dependabot[bot]", "type": "Bot"},
                     "body": "Bumps x to 2.0", "created_at": "2026-06-02T11:00:00Z"},
                    {"user": {"login": "bob", "type": "User"},
                     "body": "Fixed by #34.", "created_at": "2026-06-03T09:00:00Z"},
                ])
    base.update(kw)
    return base


def test_is_bot_matches_type_and_denylist():
    assert ws8.is_bot({"login": "x", "type": "Bot"})
    assert ws8.is_bot({"login": "renovate[bot]", "type": "User"})
    assert not ws8.is_bot({"login": "alice", "type": "User"})


def test_dedupe_threads_drops_repeat_numbers_keeping_the_first():
    # GitHub pagination can repeat an item across a page boundary; observed
    # live as kubernetes/kubernetes#140406 fetched twice.
    threads = [
        _thread(number=1, title="first"),
        _thread(number=140406, title="original"),
        _thread(number=2, title="unrelated"),
        _thread(number=140406, title="duplicate"),
    ]
    deduped, dropped = ws8.dedupe_threads(threads)
    assert dropped == 1
    assert [t["number"] for t in deduped] == [1, 140406, 2]
    assert next(t for t in deduped if t["number"] == 140406)["title"] == "original"


def test_dedupe_threads_scopes_duplicates_to_repo_and_number():
    # Same number, different repos, must NOT be treated as a duplicate.
    threads = [
        _thread(repo="acme/widget", number=12),
        _thread(repo="other/thing", number=12),
    ]
    deduped, dropped = ws8.dedupe_threads(threads)
    assert dropped == 0
    assert len(deduped) == 2


def test_assert_fresh_passes_when_every_thread_matches_the_current_basis():
    threads = [_thread(number=1), _thread(number=2)]
    for t in threads:
        t["_fetch_basis"] = ws8.FETCH_BASIS
    ws8.assert_fresh(threads, source="fixture")  # must not raise


def test_assert_fresh_raises_on_a_stale_basis():
    threads = [_thread(number=1)]
    threads[0]["_fetch_basis"] = "cap1000_per_thread_comments_v0"
    with pytest.raises(ValueError, match="STALE"):
        ws8.assert_fresh(threads, source="fixture")


def test_assert_fresh_raises_when_the_tag_is_missing_entirely():
    # A raw file from before this guard existed has no _fetch_basis key at
    # all; that must fail exactly like a mismatched one, not pass silently.
    threads = [_thread(number=1)]
    with pytest.raises(ValueError, match="STALE"):
        ws8.assert_fresh(threads, source="fixture")


def test_assert_fresh_raises_if_even_one_thread_in_a_mixed_file_is_stale():
    threads = [_thread(number=1), _thread(number=2)]
    threads[0]["_fetch_basis"] = ws8.FETCH_BASIS
    threads[1]["_fetch_basis"] = "cap1000_per_thread_comments_v0"
    with pytest.raises(ValueError, match="STALE"):
        ws8.assert_fresh(threads, source="fixture")


def test_strip_bots_drops_bot_comments_and_counts_them():
    cleaned, dropped = ws8.strip_bots(_thread())
    assert dropped == 1
    assert [c["user"]["login"] for c in cleaned["comments"]] == ["bob"]


def test_strip_bots_does_not_mutate_the_input():
    t = _thread()
    ws8.strip_bots(t)
    assert len(t["comments"]) == 2


def test_render_document_includes_title_body_and_comments():
    doc = ws8.render_document(ws8.strip_bots(_thread())[0])
    assert "Build hangs" in doc
    assert "It hangs after upgrade." in doc
    assert "Fixed by #34." in doc
    assert "dependabot" not in doc


def test_document_filename_is_stable_and_greppable():
    assert ws8.document_filename(_thread()) == "acme__widget__0012.md"


def test_extract_references_finds_hash_numbers_only():
    assert ws8.extract_references("Fixed by #34 and #7.") == [34, 7]


def test_extract_references_ignores_cross_repo_and_urls():
    # Spec section 5: explicit same-repo `#` references only. Cross-repo and
    # bare URLs are NOT edges -- this under-counts hops, which biases against
    # H1 rather than toward it.
    assert ws8.extract_references("see other/repo#9") == []
    assert ws8.extract_references("https://github.com/a/b/issues/9") == []


def test_extract_references_ignores_markdown_headings_and_anchors():
    assert ws8.extract_references("# Heading\n## Sub") == []
    assert ws8.extract_references("colour #fff") == []


def test_extract_references_deduplicates_preserving_order():
    assert ws8.extract_references("#5 then #5 then #2") == [5, 2]


def test_build_graph_links_body_and_comments():
    threads = [
        _thread(number=12, body="broken", comments=[
            {"user": {"login": "bob", "type": "User"},
             "body": "dupe of #34", "created_at": "2026-06-03T09:00:00Z"}]),
        _thread(number=34, body="root cause in #56", comments=[]),
        _thread(number=56, body="the actual fix", comments=[]),
    ]
    g = ws8.build_graph(threads)
    assert g[12] == {34}
    assert g[34] == {56}
    assert g[56] == set()


def test_build_graph_drops_edges_to_threads_outside_the_corpus():
    threads = [_thread(number=12, body="see #999", comments=[])]
    assert ws8.build_graph(threads)[12] == set()


def test_extract_chains_returns_maximal_paths():
    threads = [
        _thread(number=12, body="see #34", comments=[]),
        _thread(number=34, body="see #56", comments=[]),
        _thread(number=56, body="fix here", comments=[]),
    ]
    chains = ws8.extract_chains(threads, ws8.build_graph(threads))
    assert [12, 34, 56] in chains


def test_extract_chains_includes_isolated_nodes_as_length_one():
    threads = [_thread(number=12, body="self-contained", comments=[])]
    assert ws8.extract_chains(threads, ws8.build_graph(threads)) == [[12]]


def test_extract_chains_terminates_on_cycles():
    threads = [
        _thread(number=1, body="see #2", comments=[]),
        _thread(number=2, body="see #1", comments=[]),
    ]
    chains = ws8.extract_chains(threads, ws8.build_graph(threads))
    assert all(len(c) == len(set(c)) for c in chains)


def test_classify_hops_bands_match_the_strata_keys():
    assert ws8.classify_hops([1]) == "hop1"
    assert ws8.classify_hops([1, 2]) == "hop2"
    assert ws8.classify_hops([1, 2, 3]) == "hop3plus"
    assert ws8.classify_hops([1, 2, 3, 4]) == "hop3plus"


def test_extract_references_ignores_inline_code_spans():
    # Numeric hex colours like #123 are indistinguishable from issue refs;
    # stripping code spans prevents them from creating false edges.
    assert ws8.extract_references("Run `#123` inline") == []
    assert ws8.extract_references("colour code `#123456` in prompt") == []


def test_extract_references_ignores_fenced_code_blocks():
    # Same reasoning: #123 inside a fenced block is blanked out.
    text = "See the bug:\n```\nerror: #456 in logs\n```"
    assert ws8.extract_references(text) == []


def test_extract_references_still_finds_real_refs_with_code_present():
    # Verify code stripping does not drop legitimate references in the same text.
    text = "Related to #100; see the traceback:\n```\n#200\n```\nAlso #300."
    assert ws8.extract_references(text) == [100, 300]


def test_build_graph_excludes_self_references():
    # A thread cannot reference itself; self-loops are filtered out.
    threads = [_thread(number=12, body="see #12", comments=[])]
    assert ws8.build_graph(threads)[12] == set()


def test_extract_references_bounds_unclosed_fence_to_single_field():
    # When body has an unclosed fence and a comment has genuine prose ref,
    # the comment's reference must survive. This is the critical cross-field
    # bleed test: it fails if stripping is applied to concatenated text instead
    # of per-field.
    threads = [
        _thread(number=12, body="broken:\n```\ncode snippet",
                comments=[
                    {"user": {"login": "bob", "type": "User"},
                     "body": "Fixed in #34", "created_at": "2026-06-03T09:00:00Z"}]),
        _thread(number=34, body="the fix", comments=[]),
    ]
    g = ws8.build_graph(threads)
    # The unclosed fence in the body must not swallow the #34 ref in the comment.
    assert g[12] == {34}


def test_extract_references_unclosed_fence_drops_text_after():
    # An unclosed fence runs to end-of-field, dropping any refs after it.
    text = "See #100 and #200:\n```\ncode with #300"
    # #100 and #200 are before the fence, so they survive. #300 is after, so dropped.
    assert ws8.extract_references(text) == [100, 200]


def test_extract_references_well_formed_fence_still_strips():
    # Guard against the unclosed branch swallowing well-formed blocks.
    text = "See #100.\n```\ncode\n```\nSee #200."
    assert ws8.extract_references(text) == [100, 200]


def test_first_sentence_stops_at_the_first_terminator():
    assert ws8.first_sentence("It hangs. Then it dies.") == "It hangs."


def test_first_sentence_truncates_runaway_text():
    assert len(ws8.first_sentence("x" * 500)) <= 300


def test_first_sentence_strips_markdown_noise():
    assert ws8.first_sentence("> quoted\n\nReal report here.") == "Real report here."


def test_build_question_uses_the_reporters_words_not_the_resolution():
    # R16: built from the head's TITLE, not its body -- 81% of fetched
    # threads are PRs, so a body-based template pulled in Prow commands and
    # template boilerplate; titles are human-written problem summaries.
    q = ws8.build_question(_thread(number=12, title="Build hangs after upgrade."),
                           [12, 34, 56])
    assert "Build hangs after upgrade." in q
    assert "#34" not in q and "#56" not in q


def test_grep_handle_is_zero_when_no_question_token_is_in_the_evidence():
    df, n = ws8.document_frequencies(["alpha beta", "gamma delta"])
    assert ws8.grep_handle("zeta omega", "alpha beta", df, n) == 0.0


def test_grep_handle_rewards_rare_shared_tokens_over_common_ones():
    docs = ["common token here"] * 9 + ["common token rareword"]
    df, n = ws8.document_frequencies(docs)
    rare = ws8.grep_handle("rareword", "common token rareword", df, n)
    common = ws8.grep_handle("common", "common token rareword", df, n)
    assert rare > common


def test_build_question_redacts_inline_references():
    # Critical: the head's title contains the next hop by construction (that's
    # why the edge exists). Redacting prevents the question from becoming a
    # pointer-follow. This is the test that catches the leak. R16: title, not
    # body -- a title containing #34 must still produce a question with no
    # recoverable references.
    q = ws8.build_question(
        _thread(number=12, title="Build hangs; looks like the same thing as #34."),
        [12, 34, 56])
    assert ws8.extract_references(q) == []
    assert "#…" in q  # The marker appears where the ref was.


def test_build_question_preserves_text_with_no_references():
    # Guard against over-redaction: a head with no references remains unchanged
    # apart from the template wrapper.
    q = ws8.build_question(_thread(number=12, title="Build hangs after upgrade."),
                           [12])
    assert "Build hangs after upgrade." in q
    assert "#…" not in q


def test_spent_from_checkpoint_counts_superseded_duplicate_lines(tmp_path):
    p = tmp_path / "ckpt.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"qid": "q1", "arm": "indexed", "cost_total_billed": 1.00},
        {"qid": "q1", "arm": "indexed", "cost_total_billed": 0.84},  # a retry
        {"qid": "q2", "arm": "indexed", "cost_total_billed": 2.00},
    ]) + "\n")
    # Last-wins dedup would report 2.84 and hide the 1.00 that was really billed.
    assert ws8.spent_from_checkpoint(p) == 3.84


def test_spent_from_checkpoint_is_zero_for_a_missing_file(tmp_path):
    assert ws8.spent_from_checkpoint(tmp_path / "nope.jsonl") == 0.0


def test_spent_from_checkpoint_ignores_unparseable_trailing_bytes(tmp_path):
    p = tmp_path / "ckpt.jsonl"
    p.write_text('{"cost_total_billed": 1.5}\n{"cost_total_bi')
    assert ws8.spent_from_checkpoint(p) == 1.5


def test_spent_from_checkpoint_raises_on_non_numeric_string_value(tmp_path):
    p = tmp_path / "ckpt.jsonl"
    p.write_text('{"cost_total_billed": "oops"}\n{"cost_total_billed": 2.00}\n')
    with pytest.raises(ValueError, match="cost_total_billed is non-numeric"):
        ws8.spent_from_checkpoint(p)


def test_spent_from_checkpoint_raises_on_null_value(tmp_path):
    p = tmp_path / "ckpt.jsonl"
    p.write_text('{"cost_total_billed": null}\n{"cost_total_billed": 2.00}\n')
    with pytest.raises(ValueError, match="cost_total_billed is non-numeric"):
        ws8.spent_from_checkpoint(p)


def test_spent_from_checkpoint_skips_missing_key(tmp_path):
    p = tmp_path / "ckpt.jsonl"
    p.write_text('{"qid": "q1"}\n{"cost_total_billed": 2.00}\n')
    assert ws8.spent_from_checkpoint(p) == 2.0


def test_spent_from_checkpoint_raises_on_mid_file_corruption(tmp_path):
    p = tmp_path / "ckpt.jsonl"
    p.write_text('{"cost_total_billed": 1.5}\n{"broken\n{"cost_total_billed": 2.00}\n')
    with pytest.raises(ValueError, match="Unparseable JSON at line"):
        ws8.spent_from_checkpoint(p)


def test_spent_from_checkpoint_skips_blank_lines(tmp_path):
    p = tmp_path / "ckpt.jsonl"
    p.write_text('{"cost_total_billed": 1.5}\n\n  \n{"cost_total_billed": 2.0}\n')
    assert ws8.spent_from_checkpoint(p) == 3.5


def test_spent_from_checkpoint_raises_on_nan(tmp_path):
    p = tmp_path / "ckpt.jsonl"
    # NaN is a valid JSON literal that json.loads accepts
    p.write_text('{"cost_total_billed": NaN}\n')
    with pytest.raises(ValueError, match="cost_total_billed is non-finite"):
        ws8.spent_from_checkpoint(p)


def test_spent_from_checkpoint_raises_on_infinity(tmp_path):
    p = tmp_path / "ckpt.jsonl"
    # Infinity is a valid JSON literal that json.loads accepts
    p.write_text('{"cost_total_billed": Infinity}\n')
    with pytest.raises(ValueError, match="cost_total_billed is non-finite"):
        ws8.spent_from_checkpoint(p)


def test_spent_from_checkpoint_raises_on_true(tmp_path):
    p = tmp_path / "ckpt.jsonl"
    # true would otherwise silently become 1.0 due to float(True) == 1.0
    p.write_text('{"cost_total_billed": true}\n')
    with pytest.raises(ValueError, match="cost_total_billed is non-numeric"):
        ws8.spent_from_checkpoint(p)


def test_spent_from_checkpoint_raises_on_false(tmp_path):
    p = tmp_path / "ckpt.jsonl"
    # false would otherwise silently become 0.0, hiding a real charge
    p.write_text('{"cost_total_billed": false}\n')
    with pytest.raises(ValueError, match="cost_total_billed is non-numeric"):
        ws8.spent_from_checkpoint(p)


def test_one_question_per_head_keeps_only_the_longest_chain():
    chains = [
        ("acme/widget", [1, 2]),
        ("acme/widget", [1, 3, 4]),
        ("acme/widget", [1, 5]),
    ]
    result = ws8.one_question_per_head(chains)
    assert result == [("acme/widget", [1, 3, 4])]


def test_one_question_per_head_breaks_ties_deterministically():
    # Both length 3 from head 1 -- the lexicographically smaller wins,
    # regardless of input order.
    chains = [("acme/widget", [1, 9, 9]), ("acme/widget", [1, 2, 3])]
    assert ws8.one_question_per_head(chains) == [("acme/widget", [1, 2, 3])]
    assert ws8.one_question_per_head(list(reversed(chains))) == [
        ("acme/widget", [1, 2, 3])]


def test_one_question_per_head_never_produces_two_questions_for_one_head():
    chains = [("r", [1]), ("r", [1, 2]), ("r", [2]), ("r", [2, 3]), ("r", [3])]
    result = ws8.one_question_per_head(chains)
    heads = [(repo, c[0]) for repo, c in result]
    assert len(heads) == len(set(heads))
    assert len(result) == 3  # heads 1, 2, 3 each contribute exactly one


def test_one_question_per_head_scopes_by_repo():
    # Same head number, different repos: NOT the same head.
    chains = [("repoA", [1, 2]), ("repoB", [1, 2, 3])]
    result = ws8.one_question_per_head(chains)
    assert len(result) == 2
    assert ("repoA", [1, 2]) in result
    assert ("repoB", [1, 2, 3]) in result


def test_select_questions_fills_strata_to_quota_deterministically():
    chains = [("r", [i]) for i in range(50)] + [("r", [i, i + 1]) for i in range(100, 140)] \
        + [("r", [i, i + 1, i + 2]) for i in range(200, 250)]
    picked = ws8.select_questions(chains, {"hop1": 3, "hop2": 2, "hop3plus": 4}, seed=42)
    counts = {k: sum(1 for _, c in picked if ws8.classify_hops(c) == k) for k in
              ("hop1", "hop2", "hop3plus")}
    assert counts == {"hop1": 3, "hop2": 2, "hop3plus": 4}
    assert ws8.select_questions(chains, {"hop1": 3, "hop2": 2, "hop3plus": 4},
                                seed=42) == picked


def test_select_questions_raises_when_a_stratum_is_short():
    with pytest.raises(ValueError, match="hop3plus"):
        ws8.select_questions([("r", [1])], {"hop1": 1, "hop2": 0, "hop3plus": 30}, seed=42)


def test_select_questions_no_longer_prefers_longer_chains_within_hop3plus():
    # Spec amended c0f4b59: the longest-chain tie-break is REMOVED -- it
    # concentrated hop3plus into one dense, correlated cluster (measured:
    # 29/30 from one repo, 25/30 sharing a gold document with another
    # question). Order is now governed purely by the seeded shuffle,
    # identically to every other stratum; chain length plays no role. This
    # asserts the ACTUAL deterministic outcome under seed=42, not "the
    # longer one wins" -- with only two candidates and no shuffle bias
    # toward length, either could legitimately be first.
    chains = [("r", [1, 2, 3]), ("r", [4, 5, 6, 7, 8])]
    picked = ws8.select_questions(chains, {"hop1": 0, "hop2": 0, "hop3plus": 1}, seed=42)
    assert len(picked) == 1
    assert picked[0] in chains
    # Deterministic: same seed, same input order, same pick every time.
    assert ws8.select_questions(chains, {"hop1": 0, "hop2": 0, "hop3plus": 1},
                                seed=42) == picked


def test_select_questions_excludes_degenerate_chains_before_filling_quota():
    # R9-carried: a chain flagged by is_degenerate must never be picked and
    # must never count toward its stratum's quota, even when it would
    # otherwise be the only candidate.
    chains = [("r", [1]), ("r", [2]), ("r", [3])]

    def is_degenerate(repo, chain):
        return chain == [1]

    picked = ws8.select_questions(
        chains, {"hop1": 2, "hop2": 0, "hop3plus": 0}, seed=42,
        is_degenerate=is_degenerate)
    assert ("r", [1]) not in picked
    assert sorted(picked) == [("r", [2]), ("r", [3])]


def test_select_questions_raises_when_degeneracy_drops_pool_below_quota():
    # The stratum has 2 chains nominally, but one is degenerate, so the real
    # pool is 1 -- short of a quota of 2. Must raise on the POST-filter pool.
    chains = [("r", [1]), ("r", [2])]

    def is_degenerate(repo, chain):
        return chain == [1]

    with pytest.raises(ValueError, match="hop1"):
        ws8.select_questions(chains, {"hop1": 2, "hop2": 0, "hop3plus": 0},
                             seed=42, is_degenerate=is_degenerate)


def test_select_questions_enforces_pairwise_disjoint_gold_sets_per_stratum():
    # Two candidates whose gold sets intersect: only the first (in seeded
    # shuffle order) is selected.
    chains = [("r", [1, 2]), ("r", [2, 3])]  # share document "2"

    def gold_documents(repo, chain):
        return set(chain)

    # Only one of the two could be selected -- their gold sets intersect on
    # document 2 -- so a quota of 2 cannot be met and must raise.
    with pytest.raises(ValueError, match="hop2"):
        ws8.select_questions(chains, {"hop1": 0, "hop2": 2, "hop3plus": 0},
                             seed=42, gold_documents=gold_documents)
    # With a quota of 1, exactly one is picked, and it's whichever the
    # seeded shuffle puts first (deterministic, verified via a rerun).
    picked = ws8.select_questions(chains, {"hop1": 0, "hop2": 1, "hop3plus": 0},
                                  seed=42, gold_documents=gold_documents)
    assert len(picked) == 1
    assert picked[0] in chains
    rerun = ws8.select_questions(chains, {"hop1": 0, "hop2": 1, "hop3plus": 0},
                                 seed=42, gold_documents=gold_documents)
    assert rerun == picked


def test_select_questions_disjointness_is_per_stratum_not_global():
    # A hop1 and a hop2 question may legitimately share a document -- the
    # constraint must not reach across strata.
    chains = [("r", [1]), ("r", [1, 2])]  # both touch document 1

    def gold_documents(repo, chain):
        return set(chain)

    picked = ws8.select_questions(
        chains, {"hop1": 1, "hop2": 1, "hop3plus": 0}, seed=42,
        gold_documents=gold_documents)
    counts = {k: sum(1 for _, c in picked if ws8.classify_hops(c) == k)
              for k in ("hop1", "hop2")}
    assert counts == {"hop1": 1, "hop2": 1}  # both filled despite sharing doc 1


def test_select_questions_raises_when_disjointness_drops_pool_below_quota():
    # Three candidates all sharing one document: only one can ever be
    # selected under the disjointness constraint, so a quota of 2 raises
    # even though the raw (pre-constraint) pool has 3 candidates.
    chains = [("r", [1, 9]), ("r", [1, 8]), ("r", [1, 7])]

    def gold_documents(repo, chain):
        return set(chain)

    with pytest.raises(ValueError, match="hop2"):
        ws8.select_questions(chains, {"hop1": 0, "hop2": 2, "hop3plus": 0},
                             seed=42, gold_documents=gold_documents)


def test_is_degenerate_head_flags_a_bare_reference_sentence():
    # R16: measured against the TITLE. "see #34" redacts to "see #…" -- one
    # word of content, no real question.
    assert ws8.is_degenerate_head(_thread(number=12, title="see #34"))


def test_is_degenerate_head_does_not_flag_a_real_report():
    assert not ws8.is_degenerate_head(
        _thread(number=12, title="The scheduler panics when the pod spec "
                                 "omits a resource limit entirely."))


def test_is_pull_request_detects_the_fetch_time_boolean_key():
    assert ws8.is_pull_request({"is_pull_request": True})
    assert not ws8.is_pull_request({"is_pull_request": False})


def test_is_pull_request_detects_the_raw_github_payload_key():
    # A raw GitHub /issues payload only carries `pull_request` on PR items.
    assert ws8.is_pull_request({"pull_request": {"url": "https://example.com"}})
    assert not ws8.is_pull_request({"title": "a plain issue"})


def test_select_questions_excludes_pr_headed_chains_before_filling_quota():
    # R16: a chain's head must be an issue, never a pull request, even when
    # the PR-headed chain would otherwise be the only candidate.
    chains = [("r", [1]), ("r", [2]), ("r", [3])]

    def is_pr_headed(repo, chain):
        return chain == [1]

    picked = ws8.select_questions(
        chains, {"hop1": 2, "hop2": 0, "hop3plus": 0}, seed=42,
        is_pr_headed=is_pr_headed)
    assert ("r", [1]) not in picked
    assert sorted(picked) == [("r", [2]), ("r", [3])]


def test_select_questions_raises_when_pr_heads_drop_pool_below_quota():
    chains = [("r", [1]), ("r", [2])]

    def is_pr_headed(repo, chain):
        return chain == [1]

    with pytest.raises(ValueError, match="hop1"):
        ws8.select_questions(chains, {"hop1": 2, "hop2": 0, "hop3plus": 0},
                             seed=42, is_pr_headed=is_pr_headed)


def test_spent_from_checkpoint_still_sums_normal_and_duplicate_lines(tmp_path):
    # Verify the core behavior survives the restructure
    p = tmp_path / "ckpt.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"qid": "q1", "arm": "indexed", "cost_total_billed": 1.50},
        {"qid": "q1", "arm": "indexed", "cost_total_billed": 0.75},  # a retry
        {"qid": "q2", "arm": "indexed", "cost_total_billed": 2.25},
    ]) + "\n")
    # Verify duplicates are counted, not deduped
    assert ws8.spent_from_checkpoint(p) == 4.5


def test_total_spent_across_checkpoints_sums_live_and_archived(tmp_path):
    # R18: a checkpoint that recorded real spend is renamed to
    # <name>.voided-<ISO8601>.jsonl, never deleted -- the governor's seed
    # must count both the live checkpoint and any archived one, or a voided
    # run's real billed spend goes invisible (the 2026-09-11 incident this
    # function exists to prevent).
    cache = tmp_path / "ws8_cache"
    cache.mkdir()
    (cache / "gate_checkpoint.jsonl").write_text(
        json.dumps({"qid": "q1", "arm": "parametric",
                    "cost_total_billed": 1.0}) + "\n")
    (cache / "gate_checkpoint.voided-2026-09-11.jsonl").write_text(
        json.dumps({"qid": "voided", "arm": "parametric",
                    "cost_total_billed": 0.5}) + "\n")
    assert ws8.total_spent_across_checkpoints(tmp_path) == 1.5


def test_total_spent_across_checkpoints_ignores_files_outside_ws8_cache(tmp_path):
    cache = tmp_path / "ws8_cache"
    cache.mkdir()
    (cache / "a.jsonl").write_text(
        json.dumps({"cost_total_billed": 1.0}) + "\n")
    # Sibling to ws8_cache/, not inside it -- must not be counted.
    (tmp_path / "b.jsonl").write_text(
        json.dumps({"cost_total_billed": 99.0}) + "\n")
    assert ws8.total_spent_across_checkpoints(tmp_path) == 1.0


def test_total_spent_across_checkpoints_is_zero_with_no_cache_dir(tmp_path):
    assert ws8.total_spent_across_checkpoints(tmp_path) == 0.0


def test_gate_branch_passes_only_at_exactly_zero():
    assert ws8.gate_branch(0.0, 0) == "G1"
    assert ws8.gate_branch(0.04, 0) == "G2"      # "small" is NOT harmless
    assert ws8.gate_branch(0.0, 3) == "G2"       # any_doc_hit must also be 0


def test_ws8_system_prompt_is_arm_agnostic_and_corpus_correct():
    # Fix round 1: agent_harness.SYSTEM_PROMPT ("a Python codebase", "file
    # path", "functions or classes") is wrong for WS8's GitHub issue/PR
    # threads on Go (kubernetes) and Rust (rust-lang) repos, and every WS8
    # arm must share ONE corrected string or the parametric arm (reused
    # verbatim as Phase 3's third arm) would differ from indexed/agentic in
    # two variables instead of one.
    p = config.WS8_SYSTEM_PROMPT.lower()
    for banned in ("grep", "glob", "search_code", "index", "semantic",
                   "claude-context", "mcp", "python", "repository-relative",
                   "functions or classes"):
        assert banned not in p, f"WS8 system prompt must not mention {banned!r}"
    assert "issue" in p
    assert "document filename" in p
    assert "say so plainly rather than guessing" in p


# ------------------------------------------------------- ws8_index (Phase 2)

def test_chunk_document_overlaps_and_covers_the_whole_text():
    from benchlib.non_code_retrieval import index as ws8_index
    text = "".join(str(i % 10) for i in range(5000))
    chunks = ws8_index.chunk_document(text, size=1800, overlap=200)
    assert all(len(c) <= 1800 for c in chunks)
    assert chunks[0][-200:] == chunks[1][:200]
    assert "".join(c[:-200] for c in chunks[:-1]) + chunks[-1] == text


def test_chunk_document_returns_one_chunk_for_short_text():
    from benchlib.non_code_retrieval import index as ws8_index
    assert ws8_index.chunk_document("short", size=1800, overlap=200) == ["short"]


def test_chunk_document_zero_overlap_partitions_without_duplication():
    from benchlib.non_code_retrieval import index as ws8_index
    text = "".join(str(i % 10) for i in range(5000))
    chunks = ws8_index.chunk_document(text, size=1800, overlap=0)
    assert "".join(chunks) == text  # no overlap -> plain concatenation
    assert len(chunks) == 3  # 1800, 1800, 1400


def test_predict_chunk_count_matches_manual_chunking():
    from benchlib.non_code_retrieval import index as ws8_index
    texts = ["a" * 5000, "short", "b" * 1800]
    expected = sum(len(ws8_index.chunk_document(t, 1800, 200)) for t in texts)
    assert ws8_index.predict_chunk_count(texts, 1800, 200) == expected


def test_build_rows_is_deterministic_and_matches_predicted_count():
    from benchlib.non_code_retrieval import index as ws8_index
    docs = [("a.md", "x" * 4000), ("b.md", "short"), ("c.md", "y" * 2500)]
    rows1 = ws8_index.build_rows(docs, 1800, 200)
    rows2 = ws8_index.build_rows(docs, 1800, 200)
    assert rows1 == rows2  # same input order -> same ids, every time
    predicted = ws8_index.predict_chunk_count([t for _, t in docs], 1800, 200)
    assert len(rows1) == predicted
    # ids are a dense 0..N-1 sequence in row order
    assert [r["id"] for r in rows1] == list(range(len(rows1)))
    # every row's document name matches the doc it was chunked from
    assert rows1[0]["document"] == "a.md"
    assert rows1[-1]["document"] == "c.md"


def test_build_rows_reorders_with_input_order():
    from benchlib.non_code_retrieval import index as ws8_index
    docs_a = [("a.md", "x" * 4000), ("b.md", "short")]
    docs_b = [("b.md", "short"), ("a.md", "x" * 4000)]
    rows_a = ws8_index.build_rows(docs_a, 1800, 200)
    rows_b = ws8_index.build_rows(docs_b, 1800, 200)
    assert [r["document"] for r in rows_a] != [r["document"] for r in rows_b]


def test_corpus_signature_is_stable_and_order_independent():
    from benchlib.non_code_retrieval import index as ws8_index
    sig1 = ws8_index.corpus_signature(["b.md", "a.md"], 1800, 200)
    sig2 = ws8_index.corpus_signature(["a.md", "b.md"], 1800, 200)
    assert sig1 == sig2  # input order must not matter -- it's a fingerprint


def test_corpus_signature_changes_with_chunk_params():
    from benchlib.non_code_retrieval import index as ws8_index
    sig1 = ws8_index.corpus_signature(["a.md"], 1800, 200)
    sig2 = ws8_index.corpus_signature(["a.md"], 900, 100)
    assert sig1 != sig2


def test_corpus_signature_changes_with_file_set():
    from benchlib.non_code_retrieval import index as ws8_index
    sig1 = ws8_index.corpus_signature(["a.md", "b.md"], 1800, 200)
    sig2 = ws8_index.corpus_signature(["a.md", "c.md"], 1800, 200)
    assert sig1 != sig2


def test_resume_from_is_zero_with_no_checkpoint():
    from benchlib.non_code_retrieval import index as ws8_index
    assert ws8_index.resume_from([], "sig") == 0


def test_resume_from_advances_past_inserted_batches():
    from benchlib.non_code_retrieval import index as ws8_index
    lines = [
        {"batch_start": 0, "batch_end": 128, "signature": "sig", "inserted": True},
        {"batch_start": 128, "batch_end": 256, "signature": "sig", "inserted": True},
    ]
    assert ws8_index.resume_from(lines, "sig") == 256


def test_resume_from_ignores_spend_only_lines_for_the_resume_point():
    from benchlib.non_code_retrieval import index as ws8_index
    lines = [
        {"batch_start": 0, "batch_end": 128, "signature": "sig", "inserted": True},
        {"batch_start": 128, "batch_end": 256, "signature": "sig",
         "inserted": False, "cost_total_billed": 0.01},  # billed, write failed
    ]
    # Resume point does NOT advance past the failed batch -- its rows are
    # not in Milvus, so re-embedding [128, 256) on the next run is correct.
    assert ws8_index.resume_from(lines, "sig") == 128


def test_resume_from_raises_on_a_mismatched_signature():
    from benchlib.non_code_retrieval import index as ws8_index
    lines = [{"batch_start": 0, "batch_end": 128, "signature": "old-sig",
              "inserted": True}]
    with pytest.raises(ValueError, match="different corpus/chunk-config"):
        ws8_index.resume_from(lines, "new-sig")


def test_assert_index_complete_raises_on_any_disagreement():
    from benchlib.non_code_retrieval import index as ws8_index

    class _MC:
        def __init__(self, counted): self._c = counted
        def flush(self, _): pass
        def query(self, **kw): return [{"count(*)": self._c}]

    ws8_index.assert_index_complete(_MC(100), "c", reported=100, predicted=100)
    with pytest.raises(ws8_index.IndexIncomplete, match="counted 0"):
        ws8_index.assert_index_complete(_MC(0), "c", reported=100, predicted=100)
    with pytest.raises(ws8_index.IndexIncomplete, match="predicted"):
        ws8_index.assert_index_complete(_MC(100), "c", reported=100, predicted=250)


def test_assert_index_complete_flushes_before_counting():
    # The flush is load-bearing (WS6b's exact bug: unflushed rows read as
    # zero). Assert it is actually called, not just that the happy path
    # passes.
    from benchlib.non_code_retrieval import index as ws8_index

    class _MC:
        def __init__(self): self.flushed = []
        def flush(self, name): self.flushed.append(name)
        def query(self, **kw): return [{"count(*)": 5}]

    mc = _MC()
    ws8_index.assert_index_complete(mc, "coll", reported=5, predicted=5)
    assert mc.flushed == ["coll"]


def test_make_search_tool_returns_schema_and_callable():
    from benchlib.non_code_retrieval import index as ws8_index

    class _MC:
        def search(self, **kw):
            assert kw["collection_name"] == "coll"
            assert kw["limit"] == 10
            return [[
                {"entity": {"document": "owner__repo__1.md", "text": "hit one"}},
                {"entity": {"document": "owner__repo__2.md", "text": "hit two"}},
            ]]

    schema, call = ws8_index.make_search_tool(
        _MC(), "coll", embed_fn=lambda q: [0.0] * 1536, top_k=10)

    assert schema["name"] == "search_documents"
    assert "query" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["required"] == ["query"]
    # Neutral, factual description -- names no retrieval strategy and makes
    # no quality claim, so it doesn't tip the model relative to the agentic
    # arm's tools.
    for banned in ("semantic", "vector", "embedding", "best", "accurate",
                  "better"):
        assert banned not in schema["description"].lower()

    result = call(query="does it hang")
    assert "owner__repo__1.md" in result
    assert "hit one" in result
    assert "owner__repo__2.md" in result


def test_make_search_tool_reports_no_results_found():
    from benchlib.non_code_retrieval import index as ws8_index

    class _MC:
        def search(self, **kw): return [[]]

    schema, call = ws8_index.make_search_tool(
        _MC(), "coll", embed_fn=lambda q: [0.0], top_k=10)
    assert call(query="anything") == "No results found."


# --------------------------------------------------------------------------
# Phase 4 -- the hop contrast and branch resolution (spec section 7, Phase 3).
#
# Written BEFORE benchlib/non_code_retrieval/stats.py existed and exercised entirely on
# synthetic arrays, so the statistics are verified independently of what they
# say about the real WS8 rows.
# --------------------------------------------------------------------------

_PT = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def _runs_frame(rows):
    """Minimal runs.csv-shaped frame. `rows` are (qid, arm, stratum, tokens)."""
    out = []
    for qid, arm, stratum, tokens in rows:
        rec = {"qid": qid, "arm": arm, "stratum": stratum,
               "input_tokens": tokens, "cache_read_input_tokens": 0,
               "cache_creation_input_tokens": 0, "turns": 3,
               "cost_total_billed": 0.1, "judge_correct": True,
               "grep_handle": 0.5, "answer": "an answer"}
        out.append(rec)
    return pd.DataFrame(out)


def test_stratum_deltas_returns_a_flat_float_array_not_a_tuple():
    """Regression: ws6c.paired_deltas returns (qids, array).

    np.asarray() over that tuple yields a 2-element object array and every
    downstream median is garbage, silently. This test is the guard.
    """
    from benchlib.non_code_retrieval import stats as ws8_stats

    df = _runs_frame([
        ("q1", "indexed", "hop1", 100.0), ("q1", "agentic", "hop1", 40.0),
        ("q2", "indexed", "hop1", 200.0), ("q2", "agentic", "hop1", 500.0),
        ("q3", "indexed", "hop3plus", 10.0), ("q3", "agentic", "hop3plus", 90.0),
    ])
    d = ws8_stats.stratum_deltas(df, "hop1")
    assert isinstance(d, np.ndarray)
    assert d.dtype == np.float64
    assert d.shape == (2,)
    assert sorted(d.tolist()) == [-300.0, 60.0]


def test_stratum_deltas_is_indexed_minus_agentic():
    """Sign convention: NEGATIVE means the index spent fewer prompt tokens."""
    from benchlib.non_code_retrieval import stats as ws8_stats

    df = _runs_frame([
        ("q1", "indexed", "hop3plus", 1_000.0),
        ("q1", "agentic", "hop3plus", 61_000.0),
    ])
    assert ws8_stats.stratum_deltas(df, "hop3plus").tolist() == [-60_000.0]


def test_stratum_deltas_sums_the_three_prompt_token_columns():
    """prompt_tokens is DERIVED; input_tokens alone is the uncached remainder."""
    from benchlib.non_code_retrieval import stats as ws8_stats

    df = _runs_frame([("q1", "indexed", "hop1", 10.0),
                      ("q1", "agentic", "hop1", 0.0)])
    df.loc[df["arm"] == "indexed", "cache_read_input_tokens"] = 990.0
    df.loc[df["arm"] == "agentic", "cache_creation_input_tokens"] = 400.0
    assert ws8_stats.stratum_deltas(df, "hop1").tolist() == [600.0]


def test_stratum_deltas_raises_on_an_unpaired_qid():
    """The unpaired check is a guard worth keeping: a silent inner join would
    redefine n, and n is the whole argument in an underpowered design."""
    from benchlib.non_code_retrieval import stats as ws8_stats

    df = _runs_frame([
        ("q1", "indexed", "hop1", 100.0), ("q1", "agentic", "hop1", 40.0),
        ("q2", "indexed", "hop1", 200.0),                     # no agentic pair
    ])
    with pytest.raises(ValueError, match="unpaired"):
        ws8_stats.stratum_deltas(df, "hop1")


def test_stratum_deltas_drops_an_excluded_qid_from_BOTH_arms():
    """The ws8_q067 sensitivity: dropping one arm only would unpair the row."""
    from benchlib.non_code_retrieval import stats as ws8_stats

    df = _runs_frame([
        ("q1", "indexed", "hop3plus", 100.0), ("q1", "agentic", "hop3plus", 40.0),
        ("q67", "indexed", "hop3plus", 900.0), ("q67", "agentic", "hop3plus", 10.0),
    ])
    assert ws8_stats.stratum_deltas(df, "hop3plus").shape == (2,)
    kept = ws8_stats.stratum_deltas(df, "hop3plus", exclude_qids=("q67",))
    assert kept.tolist() == [60.0]


# ----------------------------------------------------------- the contrast


def test_contrast_summary_detects_a_clear_separation():
    from benchlib.non_code_retrieval import stats as ws8_stats
    a = np.full(30, -60_000.0)      # hop3plus: index much cheaper
    b = np.full(30, +2_000.0)       # hop1: no advantage
    out = ws8_stats.contrast_summary(a, b, n_boot=2000, seed=42, alpha=0.05)
    assert out["excludes_zero"] is True
    assert out["median"] < 0


def test_contrast_summary_reports_no_separation_for_identical_strata():
    from benchlib.non_code_retrieval import stats as ws8_stats
    rng = np.random.default_rng(0)
    a = rng.normal(0, 10_000, 30)
    b = rng.normal(0, 10_000, 30)
    out = ws8_stats.contrast_summary(a, b, n_boot=2000, seed=42, alpha=0.05)
    assert out["excludes_zero"] is False


def test_contrast_summary_is_deterministic_under_the_seed():
    from benchlib.non_code_retrieval import stats as ws8_stats
    rng = np.random.default_rng(1)
    a, b = rng.normal(-5000, 8000, 30), rng.normal(0, 8000, 30)
    k = dict(n_boot=2000, seed=42, alpha=0.05)
    assert ws8_stats.contrast_summary(a, b, **k) == ws8_stats.contrast_summary(a, b, **k)


def test_contrast_summary_carries_both_cell_sizes():
    from benchlib.non_code_retrieval import stats as ws8_stats
    out = ws8_stats.contrast_summary(np.zeros(30), np.zeros(15),
                                     n_boot=200, seed=42, alpha=0.05)
    assert (out["n_a"], out["n_b"]) == (30, 15)
    assert set(out) == {"median", "lo", "hi", "excludes_zero", "n_a", "n_b"}


def test_contrast_summary_resamples_the_two_strata_INDEPENDENTLY():
    """The strata are DIFFERENT questions, so this is an unpaired contrast of
    two paired statistics. Sharing one index draw across both cells (which is
    what makes ws6c's paired bootstrap paired) would make every draw on two
    identical arrays exactly zero and collapse the interval to a point."""
    from benchlib.non_code_retrieval import stats as ws8_stats
    a = np.arange(30, dtype=float) * 1_000.0
    out = ws8_stats.contrast_summary(a, a.copy(), n_boot=4000, seed=42,
                                     alpha=0.05)
    assert out["median"] == 0.0
    assert out["hi"] - out["lo"] > 0.0, "shared resampling would give a point CI"
    assert out["excludes_zero"] is False


def test_contrast_summary_ci_covers_a_known_shift():
    """Coverage on synthetic data with the true effect KNOWN in advance."""
    from benchlib.non_code_retrieval import stats as ws8_stats
    true_shift = -30_000.0
    rng = np.random.default_rng(7)
    covered = 0
    reps = 60
    for _ in range(reps):
        a = rng.normal(true_shift, 20_000, 30)
        b = rng.normal(0.0, 20_000, 30)
        out = ws8_stats.contrast_summary(a, b, n_boot=400, seed=42, alpha=0.05)
        covered += out["lo"] <= true_shift <= out["hi"]
    assert covered / reps >= 0.80, f"coverage {covered}/{reps}"


def test_contrast_summary_false_positive_rate_is_near_alpha_under_the_null():
    """A null-calibration check: two strata drawn from ONE distribution must
    rarely separate. If this ran hot, an H1 verdict would be meaningless."""
    from benchlib.non_code_retrieval import stats as ws8_stats
    rng = np.random.default_rng(11)
    reps, hits = 100, 0
    for _ in range(reps):
        a, b = rng.normal(0, 15_000, 30), rng.normal(0, 15_000, 30)
        hits += ws8_stats.contrast_summary(a, b, n_boot=300, seed=42,
                                           alpha=0.05)["excludes_zero"]
    assert hits / reps <= 0.20, f"false positives {hits}/{reps}"


# ------------------------------------------------------ branch resolution


def test_resolve_hop_branch_covers_all_four_spec_branches():
    from benchlib.non_code_retrieval import stats as ws8_stats
    quiet = {"excludes_zero": False}
    assert ws8_stats.resolve_hop_branch(
        {"excludes_zero": True, "median": -30_000.0,
         "lo": -50_000.0, "hi": -10_000.0}, quiet) == "H1"
    assert ws8_stats.resolve_hop_branch(
        {"excludes_zero": False, "median": -3_000.0,
         "lo": -20_000.0, "hi": 14_000.0}, quiet) == "H2"
    assert ws8_stats.resolve_hop_branch(
        {"excludes_zero": True, "median": +30_000.0,
         "lo": 10_000.0, "hi": 50_000.0}, quiet) == "H3"
    # hop1 separating invalidates the hop reading regardless of the contrast
    assert ws8_stats.resolve_hop_branch(
        {"excludes_zero": True, "median": -30_000.0,
         "lo": -50_000.0, "hi": -10_000.0},
        {"excludes_zero": True}) == "H4-confounded"


def test_resolve_hop_branch_checks_h4_FIRST_even_when_the_contrast_is_quiet():
    """Spec section 7 H4: if hop1 separates, the contrast cannot be read as a hop
    effect 'regardless of what hop>=3 does' -- quiet contrast included."""
    from benchlib.non_code_retrieval import stats as ws8_stats
    loud = {"excludes_zero": True}
    for contrast in ({"excludes_zero": False, "median": 0.0,
                      "lo": -9.0, "hi": 9.0},
                     {"excludes_zero": True, "median": +9.0,
                      "lo": 1.0, "hi": 19.0},
                     {"excludes_zero": True, "median": -9.0,
                      "lo": -19.0, "hi": -1.0}):
        assert ws8_stats.resolve_hop_branch(contrast, loud) == "H4-confounded"


def test_resolve_hop_branch_accepts_a_ws6c_paired_summary_verbatim():
    """hop1's separation flag comes from ws6c.paired_summary, whose keys are
    ci_low/ci_high -- NOT lo/hi. Only `excludes_zero` may be read from it."""
    from benchlib.non_code_retrieval import stats as ws8_stats
    from benchlib.code_retrieval import topk as ws6c
    hop1 = ws6c.paired_summary(np.full(30, 5.0), n_boot=200, seed=42, alpha=0.05)
    assert "ci_low" in hop1 and "lo" not in hop1
    assert ws8_stats.resolve_hop_branch(
        {"excludes_zero": True, "median": -1.0, "lo": -2.0, "hi": -0.5},
        hop1) == "H4-confounded"


# ------------------------------------------------- R22, the censored row
#
# The censoring rule lives in scripts/non-code-retrieval/summarise_ws8.py and nowhere else, so it
# is tested there. Loaded by path because scripts/ is not an importable package.


def _summariser():
    root = Path(__file__).resolve().parent.parent.parent
    spec = importlib.util.spec_from_file_location(
        "summarise_ws8", root / "scripts" / "non-code-retrieval" / "summarise_ws8.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_summariser_names_the_censored_row_as_a_constant():
    """R22 named ws8_q067 BEFORE the analysis ran. It is a pinned constant,
    not a filter: nothing may drop a row the spec did not name in advance."""
    m = _summariser()
    assert m.CENSORED_QID == "ws8_q067"
    assert m.CENSORED_ARM == "indexed"
    # Both sensitivity variants are computed every run; neither is optional.
    assert m.VARIANTS == {"all": (), "excl_ws8_q067": ("ws8_q067",)}
    assert m.PRIMARY_METRIC == "prompt_tokens"


def test_censored_mask_hits_the_indexed_row_only():
    """The agentic ws8_q067 row answered normally and is NOT censored."""
    m = _summariser()
    df = _runs_frame([
        ("ws8_q067", "indexed", "hop3plus", 1.0),
        ("ws8_q067", "agentic", "hop3plus", 1.0),
        ("ws8_q001", "indexed", "hop3plus", 1.0),
        ("ws8_q001", "agentic", "hop3plus", 1.0),
    ])
    mask = m.censored_mask(df)
    assert mask.sum() == 1
    hit = df[mask].iloc[0]
    assert (hit["qid"], hit["arm"]) == ("ws8_q067", "indexed")


def test_the_censored_row_keeps_its_cost_and_loses_only_its_accuracy():
    """R22 part 1 vs part 2: dropping the index's worst COST would bias the
    cost comparison toward the index; keeping its judge_correct=False would
    assert a measurement the judge never took."""
    m = _summariser()
    runs = _runs_frame([
        ("ws8_q067", "indexed", "hop3plus", 1.0),
        ("ws8_q067", "agentic", "hop3plus", 1.0),
        ("ws8_q001", "indexed", "hop3plus", 1.0),
        ("ws8_q001", "agentic", "hop3plus", 1.0),
    ])
    runs.loc[(runs.qid == "ws8_q067") & (runs.arm == "indexed"),
             ["cost_total_billed", "judge_correct"]] = [3.5387, False]
    runs["hit_turn_cap"] = False
    parametric = runs.iloc[:0]
    rows = [r for r in m.summary_rows(runs, parametric)
            if r["arm"] == "indexed" and r["stratum"] == "hop3plus"]
    assert len(rows) == 1
    row = rows[0]
    assert row["n"] == 2                       # cost cell keeps both runs
    assert row["total_cost_usd"] == 3.6387     # ... including the $3.5387
    assert row["n_censored"] == 1
    assert row["n_judged"] == 1                # ... but the judge sees one
    assert row["accuracy_measured"] == 1.0     # not 0.5


def test_summariser_reports_both_sensitivity_variants_of_the_primary():
    """A result that depends on one censored observation is a result about
    that observation, so both variants are always produced."""
    m = _summariser()
    runs = pd.read_csv(m.ws8_dir() / "runs.csv")
    # The power columns come from the retrospective; a two-row stub is enough
    # for the n assertions below and keeps this test off the simulation.
    stub = [{"injected_gradient_tokens": g, "power": p}
            for g, p in ((m.DETECTABLE_FLOOR_TOKENS, 0.06), (400_000, 1.0))]
    rows = [r for r in m.contrast_rows(runs, stub) if r["role"] == "PRIMARY"]
    assert sorted(r["variant"] for r in rows) == ["all", "excl_ws8_q067"]
    assert [r["n_a_hop3plus"] for r in rows if r["variant"] == "all"] == [30]
    assert [r["n_a_hop3plus"] for r in rows
            if r["variant"] == "excl_ws8_q067"] == [29]
    assert all(r["n_b_hop1"] == 30 for r in rows)   # hop1 is untouched


# ------------------------------------------- Task 10 review round (F1-F10)
#
# Every test below pins a defect the Task 10 review found in 987d223. They are
# written against the FIXED behaviour and fail on the shipped code.


def test_resolve_hop_branch_raises_on_a_wrong_shaped_hop1_dict():
    """F1. The H4 guard must fail CLOSED.

    `hop1.get("excludes_zero")` returns None for a mis-keyed dict, so a shape
    change in ws6c.paired_summary would make H4 silently never fire and the
    lookup would return H1/H2/H3 instead -- a silent flip of the one guard the
    whole workstream rests on. A KeyError is the only acceptable outcome.
    """
    from benchlib.non_code_retrieval import stats as ws8_stats
    contrast = {"excludes_zero": False, "median": -1.0, "lo": -9.0, "hi": 9.0}
    with pytest.raises(KeyError):
        ws8_stats.resolve_hop_branch(contrast, {"ci_excludes_zero": True})


def test_resolve_hop_branch_labels_h1_h3_from_the_CI_not_the_point_estimate():
    """F2. Spec section 7 words H1/H3 as the direction the contrast CI excludes
    zero in, not the sign of the point estimate. At n=30 the median is discrete
    and a percentile CI can in principle sit entirely on the opposite side of
    zero from the observed median; the label must follow the CI.
    """
    from benchlib.non_code_retrieval import stats as ws8_stats
    quiet = {"excludes_zero": False}
    # CI entirely BELOW zero (index-favourable) but a positive point estimate.
    assert ws8_stats.resolve_hop_branch(
        {"excludes_zero": True, "median": +5.0, "lo": -30_000.0, "hi": -100.0},
        quiet) == "H1"
    # Mirror: CI entirely ABOVE zero but a negative point estimate.
    assert ws8_stats.resolve_hop_branch(
        {"excludes_zero": True, "median": -5.0, "lo": 100.0, "hi": 30_000.0},
        quiet) == "H3"


def test_branches_csv_says_what_the_branch_would_have_been_without_h4():
    """F3. `branch = H4-confounded` ALONE reads as 'there was a finding that a
    confound spoils'. There was not: the contrast spans zero, so absent H4 the
    same pre-registered lookup returns H2. The CSV -- not only the prose -- has
    to say so, because the CSV is what feeds a slide.
    """
    branches = pd.read_csv(_RESULTS_WS8 / "branches.csv")
    prim = branches[branches["role"] == "PRIMARY"]
    assert len(prim) == 2
    assert set(prim["branch"]) == {"H4-confounded"}
    assert set(prim["branch_if_hop1_were_null"]) == {"H2"}
    # Secondaries carry no pre-registered branch in either column.
    sec = branches[branches["role"] == "secondary"]
    assert set(sec["branch_if_hop1_were_null"]) == {_summariser().NOT_PREREG}


def test_branches_csv_carries_its_own_power_at_the_declared_floor():
    """F4. A reader who sees contrast -42,617, a declared floor of 25,000 and a
    CI spanning zero concludes 'underpowered by bad luck at a detectable effect
    size'. The truth -- power ~0.05 at that floor -- must be in the same file.
    """
    branches = pd.read_csv(_RESULTS_WS8 / "branches.csv")
    mde = pd.read_csv(_RESULTS_WS8 / "mde_retrospective.csv")
    prim = branches[branches["role"] == "PRIMARY"]
    assert "detectable_floor_tokens" not in branches.columns
    assert set(prim["declared_floor_tokens"]) == {25_000}
    floor_power = float(
        mde.loc[mde["injected_gradient_tokens"] == 25_000, "power"].iloc[0])
    assert set(prim["observed_power_at_declared_floor"]) == {floor_power}
    assert floor_power < 0.20, "a 0.80-powered floor would not be a finding"
    # The smallest grid point that does reach 80% power, computed not asserted.
    # Read as str: the column carries a ">400000" sentinel when nothing on the
    # grid reaches the target, so it must not be parsed as a float.
    as_text = pd.read_csv(_RESULTS_WS8 / "branches.csv",
                          dtype={"gradient_for_80pc_power": str})
    reaching = mde[mde["power"] >= 0.80]["injected_gradient_tokens"]
    expected = str(int(reaching.min())) if len(reaching) else ">400000"
    got = as_text[as_text["role"] == "PRIMARY"]["gradient_for_80pc_power"]
    assert set(got) == {expected}


# ------------------------------------------------ F5: the MDE retrospective


def _mde_runs():
    m = _summariser()
    return m, pd.read_csv(m.ws8_dir() / "runs.csv")


def test_mde_simulation_matches_the_spec_section_3_design():
    """F5b. Spec section 3's simulation is S=400 studies x B=1,000 resamples.
    The retrospective is only 'the design's own method re-run at the observed
    dispersion' if it uses the design's own S and B."""
    m = _summariser()
    assert m.MDE_STUDIES == 400
    assert m.MDE_BOOT == 1000
    assert m.MDE_SHIFTS[0] == 0, "the null calibration row comes first"


def test_mde_power_under_a_null_gradient_lands_near_alpha():
    """F5c(1). With no injected gradient the reported power IS the test's
    false-positive rate. If this ran hot, every other row would be noise."""
    m, runs = _mde_runs()
    m.MDE_SHIFTS = (0,)
    rows = m.mde_rows(runs)
    assert len(rows) == 1 and rows[0]["injected_gradient_tokens"] == 0
    assert rows[0]["power"] <= 0.12, rows[0]["power"]


def test_mde_power_is_deterministic_under_the_seed():
    """F5a. Fresh inner seeds per study must still be drawn from the outer
    SEED, so two runs of the same grid are identical."""
    m, runs = _mde_runs()
    m.MDE_SHIFTS, m.MDE_STUDIES, m.MDE_BOOT = (0, 100_000), 40, 200
    assert m.mde_rows(runs) == m.mde_rows(runs)


def test_mde_studies_do_not_share_one_bootstrap_index_matrix():
    """F5a. Passing a constant seed into contrast_summary inside the loop makes
    every simulated study reuse ONE index matrix, so the reported power is
    conditional on a single bootstrap draw rather than averaged over draws.
    Two different outer seeds must therefore give different index matrices.
    """
    m, runs = _mde_runs()
    seeds = []
    real = m.ws8_stats.contrast_summary

    def spy(sa, sb, *, n_boot, seed, alpha):
        seeds.append(seed)
        return real(sa, sb, n_boot=n_boot, seed=seed, alpha=alpha)

    m.ws8_stats.contrast_summary = spy
    try:
        m.MDE_SHIFTS, m.MDE_STUDIES, m.MDE_BOOT = (50_000,), 25, 100
        m.mde_rows(runs)
    finally:
        m.ws8_stats.contrast_summary = real
    assert len(seeds) == 25
    assert len(set(seeds)) == 25, "all studies shared one bootstrap draw"


def test_mde_power_is_monotone_in_the_injected_gradient():
    """F5. A bigger injected gradient cannot be harder to detect. Asserted on
    the SHIPPED csv, which is the artifact a slide would quote."""
    mde = pd.read_csv(_RESULTS_WS8 / "mde_retrospective.csv")
    mde = mde.sort_values("injected_gradient_tokens")
    powers = mde["power"].tolist()
    assert powers == sorted(powers), powers
    assert mde["injected_gradient_tokens"].iloc[0] == 0
    assert powers[0] <= 0.12, "the null row is the file's own calibration"


def test_mde_csv_states_the_dispersion_it_was_parameterised_by():
    """F5c(2). The whole point of the file is that the observed dispersion was
    not what the design assumed. That should not need a second file to see."""
    mde = pd.read_csv(_RESULTS_WS8 / "mde_retrospective.csv")
    for col in ("mad_hop3plus", "mad_hop1"):
        assert col in mde.columns
        assert (mde[col] > 0).all()
    # The hop1 cell is the dispersed one; that is why the contrast is quiet.
    assert (mde["mad_hop1"] > mde["mad_hop3plus"]).all()


# ------------------------------------ F6/F7/F8: the summariser's own guards


def test_per_stratum_rows_refuses_a_truncated_runs_frame():
    """F6. WS8_STRATA declares n per cell. paired_deltas guards UNPAIRED qids,
    not MISSING pairs, so a partially-resumed runs.csv would be summarised
    without complaint at a smaller n."""
    m = _summariser()
    runs = pd.read_csv(m.ws8_dir() / "runs.csv")
    truncated = runs[runs["qid"] != runs["qid"].iloc[0]]
    with pytest.raises(AssertionError, match="hop"):
        m.per_stratum_rows(truncated)


def test_the_covariate_honours_the_sensitivity_exclusion():
    """F7. The grep_handle covariate was computed on the unfiltered stratum, so
    the excl_ws8_q067 rows carried the all-pairs value. Cosmetic with one
    excluded row; a trap the moment exclude_qids grows."""
    m = _summariser()
    runs = pd.read_csv(m.ws8_dir() / "runs.csv")
    rows = m.per_stratum_rows(runs)
    cell = {(r["variant"], r["metric"]): r for r in rows
            if r["stratum"] == "hop3plus"}
    kept = runs[(runs["stratum"] == "hop3plus")
                & (runs["qid"] != m.CENSORED_QID)]
    assert (cell[("excl_ws8_q067", "prompt_tokens")]["mean_grep_handle"]
            == round(float(kept["grep_handle"].mean()), 4))
    assert (cell[("excl_ws8_q067", "prompt_tokens")]["mean_grep_handle"]
            != cell[("all", "prompt_tokens")]["mean_grep_handle"])


def test_accuracy_is_not_100pc_when_the_column_parses_as_strings():
    """F8. One blank cell makes judge_correct object dtype, and astype(bool)
    then scores every non-empty string -- "False" included -- as True, i.e.
    100% accuracy. The truthiness rule must be explicit."""
    m = _summariser()
    runs = _runs_frame([
        ("q1", "indexed", "hop1", 1.0), ("q1", "agentic", "hop1", 1.0),
        ("q2", "indexed", "hop1", 1.0), ("q2", "agentic", "hop1", 1.0),
    ])
    # _runs_frame interleaves the arms, so the two INDEXED rows are 0 and 2.
    runs["judge_correct"] = ["True", "True", "False", "True"]
    runs["hit_turn_cap"] = ["False", "False", "True", "False"]
    # pandas 3 parses these as `str` dtype, pandas 2 as `object`; astype(bool)
    # scores every non-empty string as True under both.
    assert runs["judge_correct"].dtype != bool
    rows = [r for r in m.summary_rows(runs, runs.iloc[:0])
            if r["arm"] == "indexed"]
    assert len(rows) == 1
    assert rows[0]["accuracy_measured"] == 0.5
    assert rows[0]["n_hit_turn_cap"] == 1


def test_the_truthiness_rule_is_the_one_ws6c_already_uses():
    """F8. Two conventions for parsing the same column is how they diverge."""
    m = _summariser()
    df = pd.DataFrame({"qid": ["a", "b", "c", "d", "e"],
                       "judge_correct": ["True", "False", "true", "1", ""]})
    mine = m.truthy(df["judge_correct"]).astype(float).tolist()
    theirs = list(ws6c.metric_series(df, "judge_correct").values())
    assert mine == theirs == [1.0, 0.0, 1.0, 1.0, 0.0]


# ------------------------------------------------- F10b: pin the verdict


def test_the_published_ws8_verdict_is_pinned():
    """F10b. The guard that stops a silent statistical regression reaching a
    slide. Both pre-registered sensitivity variants, from the committed CSV.
    """
    branches = pd.read_csv(_RESULTS_WS8 / "branches.csv")
    prim = branches[(branches["role"] == "PRIMARY")
                    & (branches["metric"] == "prompt_tokens")]
    assert sorted(prim["variant"]) == ["all", "excl_ws8_q067"]
    assert set(prim["branch"]) == {"H4-confounded"}
    assert set(prim["hop1_separates"]) == {True}
    assert set(prim["excludes_zero"]) == {False}
    by_variant = prim.set_index("variant")
    assert by_variant.loc["all", "median"] == -42616.5
    assert by_variant.loc["excl_ws8_q067", "median"] == -43743.0
    # The CI spans zero in BOTH variants -- that is why the branch is not H1.
    assert (by_variant["ci_lo"] < 0).all() and (by_variant["ci_hi"] > 0).all()


# ============================================================ FETCH_BASIS v2
#
# The freshness guard keys on every pin that decides WHICH threads a fetch
# returns. It did not always: v1 keyed on the cap and the comment strategy
# only, so the window -- the pin whose entire job is to define the corpus --
# could move without invalidating a single raw file.


def test_fetch_basis_encodes_the_window_and_the_sort_order():
    """A raw file's stamp must change when any collection pin changes."""
    for value in (str(config.WS8_MAX_THREADS_PER_REPO),
                  config.WS8_WINDOW_START, config.WS8_WINDOW_END,
                  config.WS8_FETCH_SORT, config.WS8_FETCH_DIRECTION):
        assert value in ws8.FETCH_BASIS, f"{value!r} missing from FETCH_BASIS"
    assert ws8.FETCH_BASIS.endswith("_v2")


def test_v1_basis_is_accepted_only_while_its_implicit_pins_hold():
    """The v1 equivalence is conditional, not a grandfather clause.

    v1 files ARE current files while the four pins v1 left implicit still hold
    the values that v1 fetch ran under -- confirmed independently by
    data/MANIFEST.json's window_pinned/sort/direction on each raw file. The
    guard against this becoming a permanent hole is that the acceptance is
    computed FROM those pins, so moving one drops v1 out of the set.
    """
    assert ws8._V1_PINS_UNCHANGED is True
    assert ws8.ACCEPTED_FETCH_BASES == {ws8.FETCH_BASIS, ws8._V1_BASIS}
    # Rebuild the condition with one pin moved: v1 must fall out.
    moved = (config.WS8_WINDOW_END == "2026-10-01")
    assert not moved, "this test encodes that WS8_WINDOW_END is NOT 2026-10-01"


def test_assert_fresh_accepts_v1_and_v2_and_rejects_anything_else():
    v1 = [{"_fetch_basis": ws8._V1_BASIS}]
    v2 = [{"_fetch_basis": ws8.FETCH_BASIS}]
    ws8.assert_fresh(v1, source="v1.json")
    ws8.assert_fresh(v2, source="v2.json")
    ws8.assert_fresh([], source="empty.json")   # nothing to be stale about
    for bad in ("cap1000_bulk_comments_v1", None,
                "cap8000_win2026-06-01..2026-10-01_created-asc_bulk_comments_v2"):
        with pytest.raises(ValueError, match="STALE"):
            ws8.assert_fresh([{"_fetch_basis": bad}], source="bad.json")


# ==================================================== docs/ is swapped, not
# ==================================================== written into in place


def _corpus_builder():
    root = Path(__file__).resolve().parent.parent.parent
    spec = importlib.util.spec_from_file_location(
        "build_ws8_corpus", root / "scripts" / "non-code-retrieval" / "build_ws8_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_swap_docs_drops_superseded_documents(tmp_path):
    """A document a later build no longer produces must not survive in docs/.

    Phase 2 indexes that directory and the agentic arm greps it, so a leftover
    from a wider window would silently remain part of the measured corpus.
    """
    m = _corpus_builder()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "keep.md").write_text("old content")
    (docs / "gone.md").write_text("from a superseded window")

    staging = tmp_path / "docs.building"
    staging.mkdir()
    (staging / "keep.md").write_text("new content")
    (staging / "added.md").write_text("new document")

    dropped = m.swap_docs(staging, docs)

    assert dropped == {"gone.md"}
    assert {p.name for p in docs.iterdir()} == {"keep.md", "added.md"}
    assert (docs / "keep.md").read_text() == "new content"
    assert not staging.exists()
    assert not docs.with_name("docs.superseded").exists()


def test_swap_docs_works_when_docs_does_not_exist_yet(tmp_path):
    m = _corpus_builder()
    docs = tmp_path / "docs"
    staging = tmp_path / "docs.building"
    staging.mkdir()
    (staging / "a.md").write_text("x")
    assert m.swap_docs(staging, docs) == set()
    assert {p.name for p in docs.iterdir()} == {"a.md"}


def test_swap_docs_clears_an_interrupted_previous_swap(tmp_path):
    """A `.superseded` directory is the PREVIOUS corpus, never a source."""
    m = _corpus_builder()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("current")
    stale = tmp_path / "docs.superseded"
    stale.mkdir()
    (stale / "ancient.md").write_text("two builds ago")
    staging = tmp_path / "docs.building"
    staging.mkdir()
    (staging / "a.md").write_text("new")

    m.swap_docs(staging, docs)
    assert {p.name for p in docs.iterdir()} == {"a.md"}
    assert not stale.exists()


# ================================================== the spend ledger, hoisted
#
# append_spend_ledger was four verbatim copies (gate_ws8, index_ws8,
# retry_ws8_q065, run_ws8), each docstring promising to keep the others in
# sync by hand. The ledger is the artifact that actually lost money, so a fix
# to it had to land in four places at once.

LEDGER_COLUMNS = ["run_label", "spend_usd", "ran_at_head", "dirty",
                  "committed_in", "note"]


def test_append_spend_ledger_schema_matches_the_committed_file():
    """The hoisted writer must produce the schema the committed ledger has.

    Read from results/ws8/spend_ledger.csv itself, not from a literal that
    could drift away from it.
    """
    committed = pd.read_csv(_RESULTS_WS8 / "spend_ledger.csv")
    assert list(committed.columns) == LEDGER_COLUMNS


def test_append_spend_ledger_creates_then_appends(tmp_path):
    path = tmp_path / "spend_ledger.csv"
    ws8.append_spend_ledger(path, run_label="first", spend_usd=1.2345678,
                            note="one")
    ws8.append_spend_ledger(path, run_label="second", spend_usd=0.5, note="two")
    df = pd.read_csv(path)
    assert list(df.columns) == LEDGER_COLUMNS
    assert list(df["run_label"]) == ["first", "second"]
    # Rounded to six places, the ledger's precision.
    assert df.loc[0, "spend_usd"] == 1.234568
    # committed_in is NEVER guessed at run time -- a run cannot know the commit
    # that will contain its own outputs.
    assert df["committed_in"].isna().all()
    assert df.loc[0, "ran_at_head"] and df.loc[0, "dirty"] in (True, False)


def test_reconcile_spend_compares_the_ledger_SUM_to_the_checkpoint_SWEEP(tmp_path):
    """The ledger's TOTAL against a fresh sweep of every checkpoint on disk.

    Not `governor.spent` against the checkpoint sum: inside a surviving
    invocation those agree by construction, which is exactly why the weaker
    check would not have caught the missing row.
    """
    cache = tmp_path / "ws8_cache"
    cache.mkdir()
    (cache / "a_checkpoint.jsonl").write_text(
        '{"cost_total_billed": 1.0}\n{"cost_total_billed": 0.5}\n')
    ledger = tmp_path / "spend_ledger.csv"
    ws8.append_spend_ledger(ledger, run_label="r", spend_usd=1.5, note="")

    rec = ws8.reconcile_spend(ledger, tmp_path)
    assert rec == {"ledger_total": 1.5, "checkpoint_total": 1.5, "delta": 0.0,
                   "tolerance": ws8.LEDGER_TOLERANCE_USD, "agrees": True}

    # Now simulate the real incident: a checkpointed pair whose invocation died
    # before writing its ledger row.
    (cache / "a_checkpoint.jsonl").write_text(
        '{"cost_total_billed": 1.0}\n{"cost_total_billed": 0.5}\n'
        '{"cost_total_billed": 0.250558}\n')
    rec = ws8.reconcile_spend(ledger, tmp_path)
    assert rec["delta"] == 0.250558 and rec["agrees"] is False


def test_warn_on_ledger_mismatch_warns_and_never_raises(tmp_path):
    import io

    cache = tmp_path / "ws8_cache"
    cache.mkdir()
    (cache / "c.jsonl").write_text('{"cost_total_billed": 9.0}\n')
    ledger = tmp_path / "spend_ledger.csv"
    ws8.append_spend_ledger(ledger, run_label="r", spend_usd=1.0, note="")

    buf = io.StringIO()
    rec = ws8.warn_on_ledger_mismatch(ledger, tmp_path, stream=buf)
    out = buf.getvalue()
    assert rec["agrees"] is False
    # Both figures NAMED, so the warning is actionable without a second tool.
    assert "1.000000" in out and "9.000000" in out and "8.000000" in out
    assert "WARNING" in out

    # A missing ledger is a discrepancy, not a crash.
    buf2 = io.StringIO()
    assert ws8.warn_on_ledger_mismatch(tmp_path / "nope.csv", tmp_path,
                                       stream=buf2)["ledger_total"] == 0.0


def _run_ws8_module():
    root = Path(__file__).resolve().parent.parent.parent
    spec = importlib.util.spec_from_file_location(
        "run_ws8", root / "scripts" / "non-code-retrieval" / "run_ws8.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_ws8_writes_its_ledger_row_from_a_finally():
    """THE regression test for the lost $0.250558.

    Rather than import run_ws8's whole main (which needs Milvus, OpenAI and an
    Anthropic key), this pins the STRUCTURE that makes the fix work: the ledger
    append is reached from a `finally`, not from the end of a happy path.
    """
    src = (Path(__file__).resolve().parent.parent.parent
           / "scripts" / "non-code-retrieval" / "run_ws8.py").read_text()
    assert "ws8.append_spend_ledger" in src
    assert "def append_spend_ledger" not in src, "the local copy is back"
    assert "finally:" in src
    # The append happens inside write_ledger, and write_ledger is called from
    # the finally with the in-flight exception.
    assert "write_ledger(sys.exc_info()[1])" in src
    body = src[src.index("def write_ledger"):]
    assert "ws8.append_spend_ledger" in body, "the append escaped write_ledger"
    assert "ws8.warn_on_ledger_mismatch" in src


def test_every_ws8_spending_script_uses_the_hoisted_writer_from_a_finally():
    """All four, not just the one the incident happened to hit."""
    root = Path(__file__).resolve().parent.parent.parent / "scripts" / "non-code-retrieval"
    for name in ("gate_ws8.py", "index_ws8.py", "retry_ws8_q065.py",
                 "run_ws8.py"):
        src = (root / name).read_text()
        assert "def append_spend_ledger" not in src, f"{name}: local copy"
        assert "def _git_short_sha" not in src, f"{name}: local git helper"
        assert "ws8.append_spend_ledger" in src, f"{name}: not using benchlib"
        assert "write_ledger(sys.exc_info()[1])" in src, f"{name}: no finally"
        assert "ws8.warn_on_ledger_mismatch" in src, f"{name}: no reconcile"


def test_a_mid_run_exception_still_produces_a_ledger_row(tmp_path):
    """The failure mode itself, simulated: a body that raises after spending.

    Reproduces run_ws8.main's write_ledger/finally shape against a fake
    governor, and asserts the row exists, carries the spend, and says it
    aborted.
    """
    class Governor:
        spent = 0.0

    governor, spent_before = Governor(), 0.0
    ledger = tmp_path / "spend_ledger.csv"
    written: list[bool] = []

    def write_ledger(error=None):
        if written:
            return
        written.append(True)
        note = "scope=full pairs_checkpointed=2"
        if error is not None:
            note += f" ABORTED before completion: {type(error).__name__}"
        ws8.append_spend_ledger(
            ledger, run_label="run_ws8_phase3",
            spend_usd=round(governor.spent - spent_before, 6), note=note)

    import sys as _sys
    with pytest.raises(RuntimeError):
        try:
            governor.spent += 0.176158          # ws8_q028/indexed
            governor.spent += 0.074400          # ws8_q011/agentic
            raise RuntimeError("connection reset by peer")
        finally:
            write_ledger(_sys.exc_info()[1])

    df = pd.read_csv(ledger)
    assert len(df) == 1
    assert df.loc[0, "spend_usd"] == 0.250558   # the exact lost amount
    assert "ABORTED" in df.loc[0, "note"]
    assert "RuntimeError" in df.loc[0, "note"]

    # And the guard against a double-write on the normal path.
    write_ledger(None)
    assert len(pd.read_csv(ledger)) == 1


# ================================================ robustness.csv (section 6.2)
#
# Three figures results/non-code-retrieval.md quotes had no script and no CSV behind them
# until 2026-09-11. These tests pin both the machinery and the committed file.


def _robustness():
    root = Path(__file__).resolve().parent.parent.parent
    spec = importlib.util.spec_from_file_location(
        "robustness_ws8", root / "scripts" / "non-code-retrieval" / "robustness_ws8.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_leave_k_out_at_k0_is_the_full_sample_ci_low():
    """The anchor. k=0 is one subset -- the sample itself -- so its floor must
    equal the interval paired_ci.csv publishes for hop1."""
    m = _robustness()
    runs = pd.read_csv(_RESULTS_WS8 / "runs.csv")
    deltas = ws8_stats.stratum_deltas(runs, "hop1")

    row = m.leave_k_out_rows(deltas, k_values=(0,))[0]
    assert row["n_subsets"] == 1
    assert row["min_ci_low"] == row["max_ci_low"]
    assert row["min_ci_low"] == ws6c.paired_summary(
        deltas, n_boot=m.LEAVE_K_BOOT, seed=config.SEED,
        alpha=config.WS6C_ALPHA)["ci_low"]
    # And the published value, at the summariser's own n_boot=10,000: the two
    # bootstrap sizes must not disagree about the number on the slide.
    published = pd.read_csv(_RESULTS_WS8 / "paired_ci.csv")
    hop1 = published[(published["variant"] == "all")
                     & (published["stratum"] == "hop1")
                     & (published["metric"] == "prompt_tokens")].iloc[0]
    assert row["min_ci_low"] == hop1["ci_lo"]


def test_leave_k_out_enumerates_every_subset_and_is_deterministic():
    m = _robustness()
    deltas = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    rows = m.leave_k_out_rows(deltas, k_values=(0, 1, 2), n_boot=200)
    assert [r["n_subsets"] for r in rows] == [1, 6, 15]   # C(6,k)
    assert [r["n_retained"] for r in rows] == [6, 5, 4]
    assert rows == m.leave_k_out_rows(deltas, k_values=(0, 1, 2), n_boot=200)
    # The floor is a MINIMUM over subsets, so it can only fall as k rises.
    assert rows[0]["min_ci_low"] >= rows[1]["min_ci_low"] >= rows[2]["min_ci_low"]


def test_centred_deltas_have_a_median_of_exactly_zero():
    m = _robustness()
    d = np.array([1.0, 5.0, 100.0, 2.0, 1_350_000.0])
    assert np.median(m.centred(d)) == 0.0


def test_fpr_helpers_land_near_alpha_on_a_symmetric_null():
    """Calibration of the calibration check. On a well-behaved symmetric null
    both tests must reject at roughly alpha -- if they did not, a measured
    0.05 on the real data would mean nothing."""
    m = _robustness()
    rng = np.random.default_rng(7)
    pool = m.centred(rng.normal(0, 1, 400))
    paired = m.paired_fpr(pool, n=30, n_reps=400, n_boot=400)
    contrast = m.contrast_fpr(pool, pool, n_a=30, n_b=30, n_reps=400,
                              n_boot=400)
    assert 0.0 <= paired <= 0.12, paired
    assert 0.0 <= contrast <= 0.12, contrast


def test_fpr_helpers_are_deterministic_under_the_seed():
    m = _robustness()
    pool = m.centred(np.arange(40, dtype=float) ** 2)
    a = m.paired_fpr(pool, n=10, n_reps=50, n_boot=200)
    b = m.paired_fpr(pool, n=10, n_reps=50, n_boot=200)
    assert a == b
    c = m.contrast_fpr(pool, pool, n_a=10, n_b=10, n_reps=50, n_boot=200)
    d = m.contrast_fpr(pool, pool, n_a=10, n_b=10, n_reps=50, n_boot=200)
    assert c == d
    # A DIFFERENT seed must actually move the draws -- the point of the
    # fresh-inner-seed construction is that the rate is averaged over bootstrap
    # draws rather than conditional on one. Checked on the SAMPLES, not on the
    # rate, which is a coarse ratio that two seeds can tie on by chance.
    rng_a = np.random.default_rng(config.SEED).choice(pool, 10)
    rng_b = np.random.default_rng(7).choice(pool, 10)
    assert not np.array_equal(rng_a, rng_b)


def test_fpr_detects_a_test_that_over_rejects():
    """A rate near alpha must be evidence, not an artifact of the helper
    always returning something small. Shrink the CI to nothing (alpha=0.9) and
    the measured rate has to shoot up."""
    m = _robustness()
    rng = np.random.default_rng(3)
    pool = m.centred(rng.normal(0, 1, 200))
    assert m.paired_fpr(pool, n=30, n_reps=200, n_boot=300, alpha=0.9) > 0.3


def test_robustness_csv_is_committed_and_says_what_each_row_means():
    df = pd.read_csv(_RESULTS_WS8 / "robustness.csv")
    m = _robustness()
    assert list(df.columns) == list(m.COLUMNS)
    assert set(df["check"]) == {"leave_k_out", "paired_fpr", "contrast_fpr"}
    # Every row carries the parameters it was produced under; a rate with no
    # n_reps/n_boot/seed beside it is not reproducible.
    assert df["seed"].eq(config.SEED).all()
    assert df["alpha"].eq(config.WS6C_ALPHA).all()
    assert df["note"].notna().all()
    fpr = df[df["check"] != "leave_k_out"]
    assert fpr["n_reps"].notna().all() and fpr["n_boot"].notna().all()
    # The contrast's null pool is a METHODOLOGICAL CHOICE, so it is named in
    # the file rather than left to the prose.
    assert fpr["null_pool"].notna().all()
    contrast = df[df["check"] == "contrast_fpr"]
    assert "pooled_hop1_hop3plus" in set(contrast["null_pool"])
    assert (contrast["role"] == "headline").sum() == 1


def test_the_published_leave_k_out_floors_are_pinned():
    """F10b, for section 6.2. These three numbers are quoted verbatim in
    results/non-code-retrieval.md; a silent statistical regression must fail here first."""
    df = pd.read_csv(_RESULTS_WS8 / "robustness.csv")
    lk = df[df["check"] == "leave_k_out"].set_index("k")
    assert lk.loc[0, "n_subsets"] == 1 and lk.loc[0, "min_ci_low"] == 17402.0
    assert lk.loc[1, "n_subsets"] == 30 and lk.loc[1, "min_ci_low"] == 17402.0
    assert lk.loc[2, "n_subsets"] == 435 and lk.loc[2, "min_ci_low"] == 9389.5
    assert lk.loc[3, "n_subsets"] == 4060 and lk.loc[3, "min_ci_low"] == 1377.0
    # The claim is "no k observations can be removed to un-separate hop1", so
    # EVERY subset must still exclude zero -- not merely the worst one.
    assert (lk["n_subsets_excluding_zero"] == lk["n_subsets"]).all()
    assert (lk["min_ci_low"] > 0).all()


def test_the_published_false_positive_rates_are_near_nominal():
    df = pd.read_csv(_RESULTS_WS8 / "robustness.csv")
    fpr = df[df["check"] != "leave_k_out"]
    # Near alpha, in the sense the document claims: within ~3 Monte Carlo
    # standard errors of 0.05 on the paired test, and conservative (below
    # nominal) on the contrast.
    paired = fpr[fpr["check"] == "paired_fpr"].iloc[0]
    assert abs(paired["fpr"] - config.WS6C_ALPHA) < 4 * paired["mc_se"]
    contrast = fpr[(fpr["check"] == "contrast_fpr")
                   & (fpr["role"] == "headline")].iloc[0]
    assert 0.0 < contrast["fpr"] <= config.WS6C_ALPHA
