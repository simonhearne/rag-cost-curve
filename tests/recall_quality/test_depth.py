"""WS4e -- does the discriminating share depend on retrieval depth?

Design: results/recall-vs-quality.md
"""
from benchlib import config


def test_ws4e_pins_match_the_spec():
    assert config.WS4E_K_VALUES == (1, 3, 5, 10)
    assert config.WS4E_ARMS == ("sq8_np1", "sq8_np4", "sq8_np48", "sq8_np512")
    assert config.WS4E_STRATUM == "ws4d_primary_264"
    assert config.WS4E_FLAT_MIN_DIFF == 0.10
    assert config.WS4E_REJUDGE_N == 60
    assert config.WS4E_REJUDGE_MIN == 58
    assert config.WS4E_TEMPERATURE is None
    # RETIRED by Amendment 1 -- kept in place, never reused (spec 3.5, 10).
    assert config.WS4E_T0_CELL_K == 1
    assert config.WS4E_T0_NULL_MAX == 0.02
    # Amendment 1 replacement cell (spec 5.3, 10).
    assert config.WS4E_REPLICATE_ARM == "sq8_np512"
    assert config.WS4E_REPLICATE_K == 1
    assert config.WS4E_REPLICATE_MIN_N == 5
    assert config.WS4E_CAP_USD == 35.00


def test_ws4e_changes_no_existing_pin():
    """Spec global constraint: WS4e ADDS, it never edits."""
    assert config.WS4_TOP_K == 10
    assert config.SEED == 42
    assert config.WS4_GEN_MODEL == "claude-sonnet-5"
    assert config.WS4_JUDGE_MODEL == "claude-opus-5"
    assert config.WS4_GEN_MAX_TOKENS == 256
    assert config.WS4B_BUDGET_USD == 90.00
    assert config.WS4B_CHANGEPOINT_BOOT_N == 10_000


def test_t0_cell_depth_is_one_of_the_swept_depths():
    assert config.WS4E_T0_CELL_K in config.WS4E_K_VALUES


def test_cap_leaves_headroom_inside_the_ws4b_budget():
    """Spec 11: $27.31 already spent against WS4B_BUDGET_USD."""
    assert config.WS4E_CAP_USD <= config.WS4B_BUDGET_USD - 27.31


def test_ws4e_dirs_are_not_ws4_or_ws4d():
    assert config.ws4e_dir().name == "ws4e"
    assert config.ws4e_cache().name == "ws4e_cache"
    assert config.ws4e_dir() != config.ws4d_dir()
    assert config.ws4e_cache() != config.ws4d_cache()


import json
import types

import pandas as pd
import pytest

from benchlib.recall_quality import runner as ws4_runner
from benchlib.config import WS4_GEN_MODEL, WS4_TOP_K


# --- fresh-clone guard -------------------------------------------------
# data/ws4_cache/* and data/ws4d_cache/* are gitignored apart from the small
# committed *.json gate/build files, so a public checkout holds none of the
# retrieval npz, the passage cache or the run checkpoints. The replay tests
# below read those caches directly; without this guard a `git clone &&
# pytest` fails seven times with a bare FileNotFoundError.
#
# The skip is keyed STRICTLY on a file being absent -- never on an assertion
# outcome -- so on a machine that holds the cache every one of these tests
# still RUNS, and a real regression still fails.

_RETRIEVAL_DIR = config.DATA_DIR / "ws4_cache" / "retrieval"
_RETRIEVAL_REGEN = ("notebooks/04_recall_vs_quality.ipynb, whose Milvus "
                    "sweep runs scripts/recall-quality/run_ws4_milvus.py")


def _require_files(paths, regenerate):
    """Skip (never fail) when a gitignored local cache file is absent."""
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"gitignored local cache absent: {', '.join(missing)} "
                    f"-- regenerate with {regenerate}")


def _arm_ids(*arms):
    """{arm: retrieval id matrix}, or skip if an arm's npz is not present."""
    import numpy as np
    paths = {a: _RETRIEVAL_DIR / f"{a}.npz" for a in arms}
    _require_files(paths.values(), _RETRIEVAL_REGEN)
    return {a: np.load(p)["ids"] for a, p in paths.items()}


class _FakeUsage:
    input_tokens = 100
    output_tokens = 10
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _FakeMessages:
    """Records every create() kwarg so the test can assert on the request."""

    def __init__(self, text="an answer"):
        self.calls = []
        self._text = text

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=self._text)],
            stop_reason="end_turn", usage=_FakeUsage(), _request_id="req_1")


class _FakeClient:
    def __init__(self, text="an answer"):
        self.messages = _FakeMessages(text)


def test_generate_omits_temperature_by_default():
    """Spec 3.5: the main sweep INHERITS the API default. Passing
    temperature=1.0 explicitly would be a different request."""
    c = _FakeClient()
    ws4_runner.generate(c, "sys", "ctx", "q?")
    assert "temperature" not in c.messages.calls[0]
    assert c.messages.calls[0]["thinking"] == {"type": "disabled"}
    assert c.messages.calls[0]["model"] == WS4_GEN_MODEL


def test_generate_sends_temperature_when_asked():
    c = _FakeClient()
    ws4_runner.generate(c, "sys", "ctx", "q?", temperature=0)
    assert c.messages.calls[0]["temperature"] == 0


def _fixture_inputs():
    """Two questions, one arm, four cached passages -- enough to exercise the
    depth slice without touching the corpus."""
    questions = {
        7: {"question": "who?", "golds": ["ada"], "subset": "main",
            "recall": {"armA": 0.9}},
        8: {"question": "when?", "golds": ["1815"], "subset": "main",
            "recall": {"armA": 0.5}},
    }
    import numpy as np
    ids = np.full((9, WS4_TOP_K), -1, dtype=np.int64)
    ids[7, :4] = [100, 101, 102, 103]
    ids[8, :4] = [104, 105, 106, 107]
    passages = pd.DataFrame(
        {"title": [f"t{i}" for i in range(100, 108)],
         "text": ["ada lovelace", "filler one", "filler two", "filler three",
                  "born 1815", "filler four", "filler five", "filler six"]},
        index=pd.Index(range(100, 108), name="row"))
    prices = {WS4_GEN_MODEL: {"input_usd_per_mtok": 2.0, "output_usd_per_mtok": 10.0,
                              "cache_write_1h_multiplier": 2.0, "cache_read_multiplier": 0.1},
              "claude-opus-5": {"input_usd_per_mtok": 5.0, "output_usd_per_mtok": 25.0,
                               "cache_write_1h_multiplier": 2.0, "cache_read_multiplier": 0.1}}
    return questions, {"armA": ids}, passages, prices


class _NoGovernor:
    spent, limit = 0.0, 999.0

    def charge(self, amount):
        self.spent += amount


def _run(tmp_path, monkeypatch, *, top_k, temperature):
    questions, retrieval, passages, prices = _fixture_inputs()
    monkeypatch.setattr(ws4_runner, "judge", lambda *a, **k: {
        "judge_correct": True, "judge_reason": "ok", "usage": _FakeUsage()})
    cp = tmp_path / f"cp_{top_k}_{temperature}.jsonl"
    ws4_runner.run_pairs(
        _FakeClient(), [(7, "armA"), (8, "armA")], questions=questions,
        retrieval=retrieval, passages=passages, system_prompt="sys",
        judge_prompt="judge", prices=prices, governor=_NoGovernor(),
        checkpoint=cp, log=lambda m: None, top_k=top_k, temperature=temperature)
    return [json.loads(line) for line in cp.read_text().splitlines()]


def test_run_pairs_slices_passages_to_top_k(tmp_path, monkeypatch):
    rows = _run(tmp_path, monkeypatch, top_k=1, temperature=None)
    assert [json.loads(r["retrieved_ids"]) for r in rows] == [[100], [104]]
    assert all(r["k"] == 1 for r in rows)


def test_run_pairs_defaults_to_the_pinned_depth(tmp_path, monkeypatch):
    rows = _run(tmp_path, monkeypatch, top_k=WS4_TOP_K, temperature=None)
    assert [json.loads(r["retrieved_ids"]) for r in rows] == [
        [100, 101, 102, 103], [104, 105, 106, 107]]
    assert all(r["k"] == WS4_TOP_K for r in rows)


def test_presence_at_k_tracks_the_slice_not_the_pin(tmp_path, monkeypatch):
    """qid 7's gold is at rank 1, qid 8's at rank 1 of its own list."""
    at1 = _run(tmp_path, monkeypatch, top_k=1, temperature=None)
    assert [r["answer_presence_at_k"] for r in at1] == [True, True]
    # answer_presence_at_10 is only meaningful when the slice IS 10
    assert all(r["answer_presence_at_10"] is None for r in at1)
    at10 = _run(tmp_path, monkeypatch, top_k=WS4_TOP_K, temperature=None)
    assert all(r["answer_presence_at_10"] is not None for r in at10)
    assert [r["answer_presence_at_k"] for r in at10] == [
        r["answer_presence_at_10"] for r in at10]


def test_recall_at_10_is_the_arm_identity_at_every_depth(tmp_path, monkeypatch):
    """Spec 6.4: the x-axis is recall@10, held FIXED across depths."""
    for k in (1, 10):
        rows = _run(tmp_path, monkeypatch, top_k=k, temperature=None)
        assert [r["recall_at_10"] for r in rows] == [0.9, 0.5]


def test_temperature_is_recorded_on_every_row(tmp_path, monkeypatch):
    assert all(r["temperature"] is None
               for r in _run(tmp_path, monkeypatch, top_k=1, temperature=None))
    assert all(r["temperature"] == 0
               for r in _run(tmp_path, monkeypatch, top_k=1, temperature=0))


from benchlib.recall_quality import gold as ws4b
from benchlib.recall_quality import depth
from benchlib.recall_quality import eligibility as ws4d
from benchlib.config import ws4b_dir, ws4d_dir


@pytest.fixture(scope="module")
def strata():
    labels = pd.read_csv(ws4d_dir() / "eligibility_labels.csv")
    tier0 = pd.read_csv(ws4b_dir() / "tier0_runs_corrected.csv")
    return depth.build_strata(labels, tier0)


def test_primary_stratum_is_ws4ds_264(strata):
    """results/recall-vs-quality.md: 325 in_primary minus 61 in-stratum drops."""
    assert len(strata["primary"]) == 264


def test_sensitivity_strata_match_ws4d(strata):
    """results/recall-vs-quality.md's table, and both are SUBSETS of the primary
    so they cost nothing extra at any depth (spec 3.1)."""
    assert len(strata["plus_unanswerable"]) == 246
    assert len(strata["disputed_included"]) == 241
    assert set(strata["plus_unanswerable"]) <= set(strata["primary"])
    assert set(strata["disputed_included"]) <= set(strata["primary"])


def test_strata_are_sorted_unique_ints(strata):
    for name, qids in strata.items():
        assert qids == sorted(set(qids)), name
        assert all(isinstance(q, int) for q in qids), name


def test_every_primary_question_has_its_gold_in_the_gt_top_10(strata):
    """Spec 3.1: this is what makes capping k at 10 the right control -- every
    question is guaranteed rescuable at the sweep's maximum depth."""
    import numpy as np
    from benchlib.config import DATA_DIR, WS4_TOP_K
    _require_files(
        [DATA_DIR / "ws4_cache" / "passages.parquet",
         DATA_DIR / "gt_10m_top100.npz"],
        "notebooks/01_datasets.ipynb (the ground truth) then "
        "notebooks/04_recall_vs_quality.ipynb, whose filter cell runs "
        "scripts/recall-quality/build_ws4_filter.py (the passage cache)")
    tbl = pd.read_parquet(DATA_DIR / "ws4_cache" / "passages.parquet").set_index("row")
    gt = np.load(DATA_DIR / "gt_10m_top100.npz")["ws4_indices"]
    golds = (pd.read_csv("results/ws4/runs.csv")
             .drop_duplicates("qid").set_index("qid")["golds"])
    for q in strata["primary"][:40]:      # 40 is enough to catch an off-by-one
        texts = [tbl.at[int(i), "text"] for i in gt[q, :WS4_TOP_K]
                 if int(i) >= 0 and int(i) in tbl.index]
        assert ws4b.answer_present_wb(golds[q], texts), q


import numpy as np


def test_bucket_of_names_the_three_buckets():
    assert depth.bucket_of({"a": True, "b": True}) == "always"
    assert depth.bucket_of({"a": False, "b": False}) == "never"
    assert depth.bucket_of({"a": True, "b": False}) == "discriminating"


def test_bucket_of_rejects_an_incomplete_row():
    """A missing arm would silently turn 'always' into 'discriminating'."""
    with pytest.raises(ValueError):
        depth.bucket_of({})


def test_bucket_of_rejects_a_genuine_partial_arm_gap():
    """{} alone is too weak a regression test: a row with 3 of 4 intended
    arms present is exactly the silent 'always' -> 'discriminating' failure
    mode the docstring warns about, and it is non-empty so the {} guard
    above does not catch it. expected_arms makes the real arm set explicit."""
    row = {"a": True, "b": True, "c": True}   # a 4th arm, "d", is missing
    assert depth.bucket_of(row) == "always"    # unchanged behaviour if the
                                               # caller doesn't ask for a check
    with pytest.raises(ValueError):
        depth.bucket_of(row, expected_arms=4)
    with pytest.raises(ValueError):
        depth.bucket_of(row, expected_arms={"a", "b", "c", "d"})
    assert depth.bucket_of(row, expected_arms=3) == "always"
    assert depth.bucket_of(row, expected_arms={"a", "b", "c"}) == "always"


def test_separable_at_detects_identical_prefixes():
    ids = {
        "a": np.array([[1, 2, 3], [9, 8, 7]]),
        "b": np.array([[1, 5, 6], [9, 8, 7]]),
    }
    # qid 0: same at k=1, differs at k=3. qid 1: identical at every depth.
    assert depth.separable_at(ids, [0, 1], 1) == {0: False, 1: False}
    assert depth.separable_at(ids, [0, 1], 3) == {0: True, 1: False}


def test_decompose_splits_the_null_from_the_signal():
    """Spec 6.1: signal = separable_rate - null."""
    verdicts = {
        1: {"a": True, "b": False},    # separable, discriminating
        2: {"a": True, "b": True},     # separable, always
        3: {"a": True, "b": False},    # IDENTICAL, discriminating -> noise
        4: {"a": False, "b": False},   # identical, never
    }
    separable = {1: True, 2: True, 3: False, 4: False}
    d = depth.decompose(verdicts, separable, [1, 2, 3, 4])
    assert d["n"] == 4
    assert (d["always"], d["never"], d["discriminating"]) == (1, 1, 2)
    assert d["share"] == pytest.approx(0.5)
    assert d["n_separable"] == 2 and d["n_identical"] == 2
    assert d["separable_rate"] == pytest.approx(0.5)
    assert d["null"] == pytest.approx(0.5)
    assert d["signal"] == pytest.approx(0.0)


def test_decompose_reports_null_as_none_when_nothing_is_identical():
    """At high k the identical subset can empty out; 0.0 would be a lie."""
    verdicts = {1: {"a": True, "b": False}}
    d = depth.decompose(verdicts, {1: True}, [1])
    assert d["null"] is None
    assert d["signal"] is None


def test_decompose_rejects_a_ragged_verdicts_dict():
    """decompose is the function that knows the cell's real arm set, so it
    is the guard against a partial row -- qid 2 is missing arm 'b', which
    bucket_of alone (without expected_arms) would silently accept as
    'always' instead of an error."""
    verdicts = {
        1: {"a": True, "b": False},
        2: {"a": True},              # missing "b": a ragged row
    }
    with pytest.raises(ValueError):
        depth.decompose(verdicts, {1: True, 2: True}, [1, 2])


def test_decompose_reproduces_the_committed_k10_null_control():
    """Spec 2.3, measured on committed data: of the 90 questions whose four
    arms got byte-identical passages at k=10, 12 discriminate."""
    import json
    from benchlib.config import ws4b_dir, ws4d_cache, ws4d_dir
    arms = ["sq8_np4", "sq8_np12", "sq8_np48", "sq8_np512"]
    _require_files(
        [ws4d_cache() / "phase3_runs_checkpoint.jsonl"],
        "notebooks/04_recall_vs_quality.ipynb, whose WS4d phase 3 cell runs "
        "scripts/recall-quality/ws4d_phase3_run.py")
    labels = pd.read_csv(ws4d_dir() / "eligibility_labels.csv")
    tier0 = pd.read_csv(ws4b_dir() / "tier0_runs_corrected.csv")
    qids = depth.build_strata(labels, tier0)["primary"]
    old = tier0[tier0.arm.isin(arms)][["qid", "arm", "judge_correct"]]
    new = pd.DataFrame([json.loads(l) for l in
                        open(ws4d_cache() / "phase3_runs_checkpoint.jsonl")])
    new["qid"] = new.qid.astype(int)
    both = pd.concat([old, new[new.arm.isin(arms)][["qid", "arm", "judge_correct"]]])
    verdicts = {q: {} for q in qids}
    for r in both.itertuples():
        if int(r.qid) in verdicts:
            verdicts[int(r.qid)][r.arm] = bool(r.judge_correct)
    ids = _arm_ids(*arms)
    d = depth.decompose(verdicts, depth.separable_at(ids, qids, 10), qids)
    assert d["n_identical"] == 90
    assert round(d["null"] * 90) == 12
    assert d["discriminating"] == 57


def _synthetic_cells(shape):
    """shape: k -> (n_discriminating_separable, n_separable, n_disc_ident,
    n_identical). Builds bucket/separable dicts over a shared question index
    so the bootstrap's pairing is exercised."""
    qids, buckets, separable = [], {}, {}
    for k, (ds, ns, di, ni) in shape.items():
        buckets[k], separable[k] = {}, {}
    n = max(ns + ni for _, ns, _, ni in shape.values())
    qids = list(range(n))
    for k, (ds, ns, di, ni) in shape.items():
        for q in qids:
            if q < ns:
                separable[k][q] = True
                buckets[k][q] = "discriminating" if q < ds else "always"
            else:
                separable[k][q] = False
                buckets[k][q] = "discriminating" if q < ns + di else "never"
    return qids, buckets, separable


def test_bootstrap_signal_recovers_the_point_estimates():
    qids, buckets, separable = _synthetic_cells({
        1: (10, 50, 5, 50), 10: (30, 50, 5, 50)})
    b = depth.bootstrap_signal(buckets, separable, qids, n_boot=200, seed=42)
    assert b["signal_by_k"][1] == pytest.approx(10 / 50 - 5 / 50)
    assert b["signal_by_k"][10] == pytest.approx(30 / 50 - 5 / 50)
    assert b["max_pair"] == pytest.approx(0.4)
    assert (b["k_lo"], b["k_hi"]) == (1, 10)


def test_bootstrap_is_deterministic_for_a_seed():
    qids, buckets, separable = _synthetic_cells({
        1: (10, 50, 5, 50), 10: (30, 50, 5, 50)})
    a = depth.bootstrap_signal(buckets, separable, qids, n_boot=200, seed=42)
    b = depth.bootstrap_signal(buckets, separable, qids, n_boot=200, seed=42)
    assert a["max_pair_ci_lo"] == b["max_pair_ci_lo"]
    assert a["max_pair_ci_hi"] == b["max_pair_ci_hi"]


def test_bootstrap_max_pair_ci_is_degenerate_when_depths_are_identical():
    """Regression for two invariants a passing test suite must not let
    silently regress (spec 6.2): (1) the SAME question-index draw is used
    for every depth within one replicate (the 'paired' bootstrap), and
    (2) max_pair_ci is built from that shared draw's per-replicate max
    difference, never as a post-hoc difference of two independently-built
    per-depth CIs.

    Construction: k=1 and k=10 are given LITERALLY IDENTICAL bucket and
    separable maps. Under a correct paired draw, every replicate resamples
    the same questions for both depths, so vals[k_lo] == vals[k_hi] exactly
    in every replicate and the max-pair difference is exactly 0.0 always --
    the CI collapses to the single point [0.0, 0.0]. Either breakage widens
    it: an independent draw per depth resamples different questions per
    depth, and a naive ci_by_k[k_hi][0] - ci_by_k[k_lo][1] mixes the lower
    tail of one independent bootstrap with the upper tail of another.
    """
    qids, buckets, separable = _synthetic_cells({
        1: (10, 50, 5, 50), 10: (10, 50, 5, 50)})
    assert buckets[1] == buckets[10] and separable[1] == separable[10]
    b = depth.bootstrap_signal(buckets, separable, qids, n_boot=300, seed=7)
    assert b["k_lo"] == 1 and b["k_hi"] == 10
    assert b["max_pair_ci_lo"] == 0.0
    assert b["max_pair_ci_hi"] == 0.0


def test_flatness_needs_BOTH_an_excluding_ci_and_a_big_enough_difference():
    """Spec 4.1: requiring both biases the test AGAINST our own hypothesis."""
    big_and_sig = {"max_pair": 0.30, "max_pair_ci_lo": 0.10, "max_pair_ci_hi": 0.50}
    assert depth.flatness_branch(big_and_sig, 0.10) == "non_flat"
    # significant but too small to matter
    small = {"max_pair": 0.04, "max_pair_ci_lo": 0.01, "max_pair_ci_hi": 0.07}
    assert depth.flatness_branch(small, 0.10) == "flat"
    # big but the interval covers zero
    noisy = {"max_pair": 0.30, "max_pair_ci_lo": -0.05, "max_pair_ci_hi": 0.60}
    assert depth.flatness_branch(noisy, 0.10) == "flat"


def test_peak_branch_distinguishes_interior_from_the_ends():
    """Spec 9: X-A interior, X-B at k=1, X-C at k=10."""
    ks = (1, 3, 5, 10)
    assert depth.peak_branch({1: 0.1, 3: 0.4, 5: 0.2, 10: 0.1}, ks) == "X-A"
    assert depth.peak_branch({1: 0.5, 3: 0.4, 5: 0.2, 10: 0.1}, ks) == "X-B"
    assert depth.peak_branch({1: 0.1, 3: 0.2, 5: 0.4, 10: 0.5}, ks) == "X-C"


def test_gate_k1_requires_one_basis_across_every_cell():
    """Spec 7: the gate WS4d's interim ladder report failed."""
    cells = {(1, None): [1, 2, 3], (10, None): [1, 2, 3]}
    g = depth.gate_k1(cells, qids=[1, 2, 3], arms=("a", "b"),
                     arms_by_cell={(1, None): ("a", "b"), (10, None): ("a", "b")},
                     x_axis="recall_at_10")
    assert g["passed"] is True
    assert g["basis"] == "3 qids x 2 arms; x-axis recall_at_10"

    mismatched = {(1, None): [1, 2], (10, None): [1, 2, 3]}
    assert depth.gate_k1(mismatched, qids=[1, 2, 3], arms=("a", "b"),
                        arms_by_cell={(1, None): ("a", "b"),
                                      (10, None): ("a", "b")},
                        x_axis="recall_at_10")["passed"] is False


def test_gate_k2_checks_score_order_shape_and_the_committed_ids():
    """⚠️ The prefix claim's real content is SCORE ORDERING: a k-prefix equals
    what a limit=k search would return only because the cached scores
    descend. Asserting '[:k] is a prefix' would be tautological."""
    ids = {"a": np.array([[1, 2, 3, 4]]), "b": np.array([[5, 6, 7, 8]])}
    ok = {"a": np.array([[0.9, 0.8, 0.7, 0.6]]),
          "b": np.array([[0.9, 0.5, 0.4, 0.1]])}
    g = depth.gate_k2(ids, ok, qids=[0], committed={}, top_k=4)
    assert g["passed"] is True and g["n_arms_score_ordered"] == 2

    # an arm whose scores ASCEND breaks the prefix claim
    bad_order = {"a": np.array([[0.1, 0.8, 0.7, 0.6]]),
                 "b": np.array([[0.9, 0.5, 0.4, 0.1]])}
    assert depth.gate_k2(ids, bad_order, qids=[0], committed={},
                        top_k=4)["passed"] is False

    # the cache must be what the committed rows were generated from
    bad_ids = depth.gate_k2(ids, ok, qids=[0], top_k=4,
                           committed={(0, "a"): [9, 9, 9, 9]})
    assert bad_ids["passed"] is False and bad_ids["n_committed_mismatched"] == 1

    # a wrong cache width means the arm is not the one WS4 served
    assert depth.gate_k2(ids, ok, qids=[0], committed={},
                        top_k=10)["passed"] is False


def test_gate_k3_allows_two_sampling_flips():
    assert depth.gate_k3(n=60, n_reproduced=60, threshold=58)["passed"] is True
    assert depth.gate_k3(n=60, n_reproduced=58, threshold=58)["passed"] is True
    assert depth.gate_k3(n=60, n_reproduced=57, threshold=58)["passed"] is False


def test_gate_k4_flags_a_depth_whose_signal_drowns_in_noise():
    """Spec 7 K4 -> branch X-E, per depth."""
    g = depth.gate_k4({1: {"separable_rate": 0.30, "null": 0.10},
                      3: {"separable_rate": 0.10, "null": 0.15}})
    assert g["passed"] is False
    assert g["depths_at_noise"] == [3]


def test_gate_k6_passes_on_a_ci_excluding_zero_with_enough_disagreement():
    """Spec 5.3 V1 / 7 K6 (REDEFINED, Amendment 1)."""
    g = depth.gate_k6(R=0.08, ci_lo=0.04, ci_hi=0.12, n_disagree=21, min_n=5)
    assert g["passed"] is True


def test_gate_k6_fails_when_the_ci_includes_zero():
    """-> branch X-G: the more consequential failure (spec 7)."""
    g = depth.gate_k6(R=0.01, ci_lo=-0.01, ci_hi=0.03, n_disagree=3, min_n=5)
    assert g["passed"] is False


def test_gate_k6_fails_below_the_minimum_disagreeing_count_even_with_a_tight_ci():
    """WS4E_REPLICATE_MIN_N=5 (spec 10): a CI excluding zero on fewer than
    five disagreeing questions is one flaky row, not a measurement."""
    g = depth.gate_k6(R=0.02, ci_lo=0.005, ci_hi=0.04, n_disagree=4, min_n=5)
    assert g["passed"] is False


def test_gate_k6_ci_touching_zero_does_not_count_as_excluding_it():
    g = depth.gate_k6(R=0.0, ci_lo=0.0, ci_hi=0.05, n_disagree=10, min_n=5)
    assert g["passed"] is False


def test_bootstrap_rate_recovers_the_point_estimate_and_is_deterministic():
    flags = [True] * 20 + [False] * 80
    a = depth.bootstrap_rate(flags, n_boot=500, seed=42)
    b = depth.bootstrap_rate(flags, n_boot=500, seed=42)
    assert a["R"] == pytest.approx(0.20)
    assert a["n"] == 100 and a["n_disagree"] == 20
    assert a["ci_lo"] == b["ci_lo"] and a["ci_hi"] == b["ci_hi"]
    assert a["ci_lo"] < 0.20 < a["ci_hi"]


def test_bootstrap_rate_ci_excludes_zero_when_all_flags_agree():
    flags = [True] * 30
    r = depth.bootstrap_rate(flags, n_boot=500, seed=42)
    assert r["ci_lo"] == 1.0 and r["ci_hi"] == 1.0


def test_bootstrap_rate_reports_none_on_an_empty_set():
    """An untested rate must not be reported as 0.0 (mirrors decompose's
    null=None convention)."""
    r = depth.bootstrap_rate([], n_boot=500, seed=42)
    assert r["R"] is None and r["ci_lo"] is None and r["ci_hi"] is None
    assert r["n"] == 0


def test_four_sample_consistency_check_matches_a_hand_worked_example():
    """R = 2p(1-p): p = 0.1 -> R = 0.18 -> predicted = 1 - .1^4 - .9^4."""
    got = depth.four_sample_consistency_check(0.18)
    assert got["p"] == pytest.approx(0.1, abs=1e-4)
    assert got["predicted_four_sample_rate"] == pytest.approx(
        1 - 0.1 ** 4 - 0.9 ** 4, abs=1e-4)


def test_four_sample_consistency_check_is_none_outside_its_domain():
    assert depth.four_sample_consistency_check(None) is None
    assert depth.four_sample_consistency_check(0.6) is None


from benchlib.recall_quality import depth_runner as ws4e_runner


def test_each_cell_gets_its_own_checkpoint_file():
    """The (qid, arm) checkpoint key COLLIDES across depths -- one shared
    file would let a k=1 row satisfy the k=10 cell."""
    paths = {ws4e_runner.cell_checkpoint(k, None) for k in (1, 3, 5, 10)}
    assert len(paths) == 4
    assert ws4e_runner.cell_checkpoint(1, 0) != ws4e_runner.cell_checkpoint(1, None)


def test_checkpoint_names_are_readable_and_under_the_ws4e_cache():
    from benchlib.config import ws4e_cache
    p = ws4e_runner.cell_checkpoint(3, None)
    assert p.parent == ws4e_cache()
    assert p.name == "runs_k3_tdefault.jsonl"
    assert ws4e_runner.cell_checkpoint(1, 0).name == "runs_k1_t0.jsonl"


def test_replicate_checkpoint_does_not_collide_with_the_primary_k1_cell():
    """Amendment 1's replicate cell (spec 5.3) reuses the EXACT same (k=1,
    temperature=None) as the primary sweep cell. Without `replicate` as a
    third key, load_checkpoint's (qid, arm) dedup would treat every
    replicate row as already recorded by the primary cell, write zero new
    rows, and make R trivially 0.0."""
    primary = ws4e_runner.cell_checkpoint(1, None)
    replicate = ws4e_runner.cell_checkpoint(1, None, replicate=True)
    assert replicate != primary
    assert replicate.name == "runs_k1_tdefault_rep.jsonl"
    assert replicate.parent == primary.parent


def test_tier_spent_sums_every_cell_and_the_rejudge_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ws4e_runner, "_cache_dir", lambda: tmp_path)
    (tmp_path / "runs_k1_tdefault.jsonl").write_text(
        '{"cost_total_billed": 1.5}\n{"cost_total_billed": 0.5}\n')
    (tmp_path / "runs_k3_tdefault.jsonl").write_text('{"cost_total_billed": 2.0}\n')
    (tmp_path / "rejudge_checkpoint.jsonl").write_text('{"cost_total_billed": 0.25}\n')
    assert ws4e_runner.tier_spent() == pytest.approx(4.25)


def test_tier_spent_counts_every_raw_line_not_deduped_keys(tmp_path, monkeypatch):
    """Money already billed on a duplicate line is still spent -- this is the
    WS6c section 10.1 failure, where the governor could not see it."""
    monkeypatch.setattr(ws4e_runner, "_cache_dir", lambda: tmp_path)
    (tmp_path / "runs_k1_tdefault.jsonl").write_text(
        '{"qid": "7", "arm": "a", "cost_total_billed": 1.0}\n'
        '{"qid": "7", "arm": "a", "cost_total_billed": 1.0}\n')
    assert ws4e_runner.tier_spent() == pytest.approx(2.0)


def test_tier_governor_is_capped_at_the_ws4e_pin(tmp_path, monkeypatch):
    from benchlib.config import WS4E_CAP_USD
    monkeypatch.setattr(ws4e_runner, "_cache_dir", lambda: tmp_path)
    assert ws4e_runner.tier_governor().limit == WS4E_CAP_USD


def test_phase0_reproduces_the_specs_separability_table():
    """Spec 2.2, for WS4E_ARMS: 98/172/206/241 separable at k=1/3/5/10."""
    from benchlib.config import WS4E_ARMS, WS4E_K_VALUES, ws4b_dir, ws4d_dir
    labels = pd.read_csv(ws4d_dir() / "eligibility_labels.csv")
    tier0 = pd.read_csv(ws4b_dir() / "tier0_runs_corrected.csv")
    qids = depth.build_strata(labels, tier0)["primary"]
    ids = _arm_ids(*WS4E_ARMS)
    got = {k: sum(depth.separable_at(ids, qids, k).values())
           for k in WS4E_K_VALUES}
    assert got == {1: 98, 3: 172, 5: 206, 10: 241}


def test_the_validation_cell_depth_has_the_biggest_identical_subset():
    """Spec 5.3's reason for choosing k=1: 166 identical, vs 92/58/23.

    Retargeted at WS4E_REPLICATE_K in fix round 1 (Minor): WS4E_T0_CELL_K
    names the withdrawn temperature-0 design (Amendment 1) and, although its
    value still happens to equal WS4E_REPLICATE_K, asserting against it here
    would describe the wrong thing.
    """
    from benchlib.config import (WS4E_ARMS, WS4E_K_VALUES, WS4E_REPLICATE_K,
                                 ws4b_dir, ws4d_dir)
    labels = pd.read_csv(ws4d_dir() / "eligibility_labels.csv")
    tier0 = pd.read_csv(ws4b_dir() / "tier0_runs_corrected.csv")
    qids = depth.build_strata(labels, tier0)["primary"]
    ids = _arm_ids(*WS4E_ARMS)
    ident = {k: len(qids) - sum(depth.separable_at(ids, qids, k).values())
             for k in WS4E_K_VALUES}
    assert ident == {1: 166, 3: 92, 5: 58, 10: 23}
    assert WS4E_REPLICATE_K == max(ident, key=ident.get)


from pathlib import Path


def test_slope_uses_recall_at_10_as_a_fixed_x_axis():
    """Spec 6.4: four arms against a 4-parameter changepoint is zero residual
    df, so WS4e fits a LINE -- and the x-axis must not move with k."""
    from benchlib.recall_quality import eligibility as ws4d
    x = [0.5871, 0.7902, 0.9587, 0.9913]       # recall@10, identical at every k
    fit = ws4d.fit_line(x, [0.30, 0.45, 0.55, 0.60])
    assert fit["slope"] > 0
    assert len(x) - 2 == 2                      # two residual degrees of freedom


def test_no_changepoint_is_fitted_on_four_arms():
    """Guard against a future reader re-adding it: 4 points, 4 parameters.

    Read as text rather than imported -- `scripts/` is deliberately NOT a
    package in this repo, and making it one to satisfy a test would change
    how every other script resolves its imports.
    """
    src = Path("scripts/recall-quality/ws4e_phase2_analyse.py").read_text()
    assert "fit_changepoint" not in src
    assert "bootstrap_changepoint" not in src
    assert "fit_line" in src


# --- Fix round 1 -------------------------------------------------------
# CRITICAL 1: separable(k) is nested/monotone, so signal(k)'s own
# denominator changes composition with k. Diagnostic functions below do
# NOT change signal(k), the branch rule, or any committed number.


def test_separability_is_nested_across_depths_on_the_committed_data():
    """CRITICAL 1: separable(1) subset separable(3) subset separable(5)
    subset separable(10). A question that becomes separable at some depth
    never becomes identical again at a GREATER depth (separable_at compares
    exact id tuples, and extending an already-different prefix cannot make
    it equal again)."""
    from benchlib.config import WS4E_ARMS, WS4E_K_VALUES, ws4b_dir, ws4d_dir
    labels = pd.read_csv(ws4d_dir() / "eligibility_labels.csv")
    tier0 = pd.read_csv(ws4b_dir() / "tier0_runs_corrected.csv")
    qids = depth.build_strata(labels, tier0)["primary"]
    ids = _arm_ids(*WS4E_ARMS)
    sep = {k: {q for q, v in depth.separable_at(ids, qids, k).items() if v}
           for k in WS4E_K_VALUES}
    ks = sorted(WS4E_K_VALUES)
    for a, b in zip(ks, ks[1:]):
        assert sep[a] <= sep[b], (a, b)


def test_fixed_composition_signal_is_far_flatter_than_the_committed_signal():
    """CRITICAL 1: on the population held fixed across k (separable(1)'s 98
    questions vs identical(10)'s 23), the max pairwise fixed-composition
    signal difference is BELOW WS4E_FLAT_MIN_DIFF on the committed data --
    the opposite of the committed (non-flat, X-B) result, because the
    committed statistic's denominator is not the same population at every
    k."""
    import json as _json
    from benchlib.config import (WS4E_ARMS, WS4E_K_VALUES, WS4E_TEMPERATURE,
                                 WS4_TOP_K, WS4E_FLAT_MIN_DIFF,
                                 ws4b_dir, ws4d_cache, ws4d_dir)
    from benchlib.recall_quality import depth_runner as ws4e_runner
    labels = pd.read_csv(ws4d_dir() / "eligibility_labels.csv")
    tier0 = pd.read_csv(ws4b_dir() / "tier0_runs_corrected.csv")
    qids = depth.build_strata(labels, tier0)["primary"]
    ids = _arm_ids(*WS4E_ARMS)
    separable = {k: depth.separable_at(ids, qids, k) for k in WS4E_K_VALUES}
    fixed_sep = [q for q in qids if separable[1][q]]
    fixed_ident = [q for q in qids if not separable[10][q]]

    committed_k10_arms = ("sq8_np4", "sq8_np48", "sq8_np512")
    runs = pd.read_csv("results/ws4/runs.csv")
    runs = runs[(runs.subset == "main") & runs.arm.isin(committed_k10_arms)]
    phase3 = ws4d_cache() / "phase3_runs_checkpoint.jsonl"
    extra = []
    if phase3.exists():
        extra = [_json.loads(l) for l in open(phase3)]
        extra = [r for r in extra
                 if r.get("subset") == "main" and r["arm"] in committed_k10_arms]

    def verdicts_for(k):
        out = {int(q): {} for q in qids}
        for (qid, arm), row in ws4e_runner.cell_rows(k, WS4E_TEMPERATURE).items():
            q = int(qid)
            if q in out:
                out[q][arm] = bool(row["judge_correct"])
        if k == WS4_TOP_K:
            for r in runs.itertuples():
                q = int(r.qid)
                if q in out:
                    out[q][r.arm] = bool(r.judge_correct)
            for r in extra:
                q = int(r["qid"])
                if q in out:
                    out[q][r["arm"]] = bool(r["judge_correct"])
        return out

    buckets = {}
    for k in WS4E_K_VALUES:
        v = verdicts_for(k)
        d = depth.decompose(v, separable[k], qids)
        buckets[k] = d.pop("buckets")

    boot = depth.bootstrap_fixed_composition_signal(
        buckets, fixed_sep, fixed_ident, n_boot=1000, seed=42)
    assert boot["max_pair"] < WS4E_FLAT_MIN_DIFF


def test_fixed_composition_signal_shape():
    buckets = {
        1: {1: "discriminating", 2: "always", 3: "discriminating", 4: "never"},
        10: {1: "always", 2: "always", 3: "discriminating", 4: "never"},
    }
    got = depth.fixed_composition_signal(buckets, [1, 2], [3, 4])
    assert got[1] == {"fixed_separable_rate": pytest.approx(0.5),
                      "fixed_identical_rate": pytest.approx(0.5),
                      "fixed_signal": pytest.approx(0.0)}
    assert got[10]["fixed_separable_rate"] == pytest.approx(0.0)


def test_fixed_population_membership_check_reports_churn_not_just_rate():
    """Two questions discriminate at k=1 and two at
    k=10, same count -- but a different pair, so overlap is 0, not 2."""
    buckets = {
        1: {1: "discriminating", 2: "discriminating", 3: "never", 4: "never"},
        10: {1: "never", 2: "never", 3: "discriminating", 4: "discriminating"},
    }
    got = depth.fixed_population_membership_check(buckets, [1, 2, 3, 4])
    assert got[1] == {"n_disc": 2, "n_disc_overlap_vs_k1": 2}
    assert got[10] == {"n_disc": 2, "n_disc_overlap_vs_k1": 0}


def test_fixed_population_membership_check_full_overlap_when_set_is_stable():
    buckets = {
        1: {1: "discriminating", 2: "never"},
        10: {1: "discriminating", 2: "never"},
    }
    got = depth.fixed_population_membership_check(buckets, [1, 2])
    assert got[10] == {"n_disc": 1, "n_disc_overlap_vs_k1": 1}


def test_fixed_population_rate_reads_back_a_fixed_set_at_every_k():
    buckets = {
        1: {1: "discriminating", 2: "never", 3: "discriminating"},
        10: {1: "discriminating", 2: "discriminating", 3: "discriminating"},
    }
    got = depth.fixed_population_rate(buckets, [1, 2, 3])
    assert got == {1: pytest.approx(2 / 3), 10: pytest.approx(1.0)}


def test_retrieval_attributable_excess_multiplies_rate_by_count():
    got = depth.retrieval_attributable_excess(
        {1: 0.40, 10: 0.20}, {1: 100, 10: 250})
    assert got == {1: pytest.approx(40.0), 10: pytest.approx(50.0)}


def test_retrieval_attributable_excess_is_none_when_signal_is_none():
    got = depth.retrieval_attributable_excess({1: None}, {1: 100})
    assert got[1] is None


def test_bootstrap_all_pairs_signal_uses_a_bonferroni_widened_ci_level():
    """S7: with 6 pairwise comparisons the two-sided per-pair CI must be
    widened to 1 - 0.05/6 ~= 99.17%, not left at the uncorrected 95%."""
    qids, buckets, separable = _synthetic_cells({
        1: (10, 50, 5, 50), 3: (15, 50, 5, 50),
        5: (20, 50, 5, 50), 10: (30, 50, 5, 50)})
    pairs = depth.bootstrap_all_pairs_signal(buckets, separable, qids,
                                            n_boot=500, seed=42)
    assert len(pairs) == 6
    assert all(p["n_comparisons"] == 6 for p in pairs)
    assert all(p["ci_level_pct"] == pytest.approx(99.1667, abs=1e-3)
              for p in pairs)
    assert {(p["k_lo"], p["k_hi"]) for p in pairs} == {
        (1, 3), (1, 5), (1, 10), (3, 5), (3, 10), (5, 10)}


def test_evaluate_p1_holds_when_always_never_decreases_as_k_falls():
    cells = {1: {"always": 10}, 3: {"always": 12}, 5: {"always": 15},
            10: {"always": 20}}
    assert depth.evaluate_p1(cells)["holds"] is True
    bad = {1: {"always": 20}, 3: {"always": 12}, 5: {"always": 15},
          10: {"always": 20}}
    assert depth.evaluate_p1(bad)["holds"] is False


def test_evaluate_p2_holds_when_never_decreases_as_k_rises():
    cells = {1: {"never": 20}, 3: {"never": 15}, 5: {"never": 12},
            10: {"never": 10}}
    assert depth.evaluate_p2(cells)["holds"] is True
    bad = {1: {"never": 10}, 3: {"never": 15}, 5: {"never": 12},
          10: {"never": 10}}
    assert depth.evaluate_p2(bad)["holds"] is False


def test_p1_and_p2_hold_on_the_committed_data():
    """S4: never evaluated in code before fix round 1. Both hold: always
    104 -> 111 -> 116 -> 123 and never 101 -> 91 -> 83 -> 65."""
    df = pd.read_csv("results/ws4e/discrimination_by_k.csv").set_index("k")
    cells = {k: {"always": int(df.loc[k, "always"]),
                "never": int(df.loc[k, "never"])} for k in (1, 3, 5, 10)}
    assert depth.evaluate_p1(cells)["holds"] is True
    assert depth.evaluate_p2(cells)["holds"] is True


def test_gate_k5_reports_budget_even_when_it_never_fired():
    g = depth.gate_k5(spent=24.6845, limit=35.00)
    assert g["passed"] is True
    assert g["gate"] == "K5"
    assert depth.gate_k5(spent=40.0, limit=35.00)["passed"] is False


def test_inverted_fraction_among_discriminating_AND_separable_questions_only():
    """Fix round 1, IMPORTANT 3: the denominator is discriminating AND
    separable -- an inversion needs the passages to actually differ, so a
    discriminating-but-identical-passage question (generator noise, not
    retrieval) must not count."""
    verdicts = {
        1: {"worst": True, "mid": False, "best": False},   # inverted, separable
        2: {"worst": False, "mid": False, "best": True},   # not inverted, separable
        3: {"worst": True, "mid": True, "best": True},     # always: excluded
        4: {"worst": False, "mid": False, "best": False},  # never: excluded
        5: {"worst": True, "mid": False, "best": False},   # inverted but IDENTICAL: excluded
    }
    separable = {1: True, 2: True, 3: True, 4: True, 5: False}
    got = depth.inverted_fraction(verdicts, separable, "worst", "best",
                                 [1, 2, 3, 4, 5])
    assert got == pytest.approx(0.5)   # 1 of 2 discriminating+separable questions


def test_inverted_fraction_reproduces_the_committed_k1_and_k10_rates():
    """Spec-adjacent regression: the exact rates the round-1 review reported
    (0.2553 at k=1, 0.1081 at k=10) on the committed data.

    The test's own name promised both rates but only
    ever asserted k=1. k=10 needs its own verdicts helper because
    ws4e_runner.cell_rows(10, ...) holds ONLY sq8_np1's row -- sq8_np4,
    sq8_np48 and sq8_np512's committed k=10 verdicts live in
    results/ws4/runs.csv and data/ws4d_cache/phase3_runs_checkpoint.jsonl
    respectively (see scripts/recall-quality/ws4e_phase2_analyse.py's own
    load_committed_k10, duplicated here rather than imported since
    scripts/ is deliberately not a package)."""
    from benchlib.config import (WS4E_ARMS, WS4E_TEMPERATURE, WS4_TOP_K,
                                 ws4b_dir, ws4d_cache, ws4d_dir)
    from benchlib.recall_quality import depth_runner as ws4e_runner
    from benchlib.agent_harness import load_checkpoint
    labels = pd.read_csv(ws4d_dir() / "eligibility_labels.csv")
    tier0 = pd.read_csv(ws4b_dir() / "tier0_runs_corrected.csv")
    qids = depth.build_strata(labels, tier0)["primary"]
    ids = _arm_ids(*WS4E_ARMS)
    committed_k10_arms = ("sq8_np4", "sq8_np48", "sq8_np512")

    def verdicts_for(k):
        out = {int(q): {} for q in qids}
        for (qid, arm), row in ws4e_runner.cell_rows(k, WS4E_TEMPERATURE).items():
            q = int(qid)
            if q in out:
                out[q][arm] = bool(row["judge_correct"])
        if k == WS4_TOP_K:
            want = {int(q) for q in qids}
            frames = []
            runs = pd.read_csv("results/ws4/runs.csv")
            runs = runs[(runs.subset == "main")
                        & runs.arm.isin(committed_k10_arms)]
            frames.append(runs[["qid", "arm", "judge_correct"]])
            p3 = ws4d_cache() / "phase3_runs_checkpoint.jsonl"
            if p3.exists():
                rows = [r for r in load_checkpoint(p3).values()
                        if r.get("subset") == "main"
                        and r["arm"] in committed_k10_arms]
                if rows:
                    frames.append(
                        pd.DataFrame(rows)[["qid", "arm", "judge_correct"]])
            committed = pd.concat(frames, ignore_index=True)
            committed["qid"] = committed["qid"].astype(int)
            committed = committed[committed.qid.isin(want)]
            for r in committed.itertuples():
                q = int(r.qid)
                if q in out:
                    out[q][r.arm] = bool(r.judge_correct)
        return out

    sep1 = depth.separable_at(ids, qids, 1)
    got1 = depth.inverted_fraction(verdicts_for(1), sep1, "sq8_np1",
                                  "sq8_np512", qids)
    assert got1 == pytest.approx(0.2553, abs=1e-4)

    sep10 = depth.separable_at(ids, qids, WS4_TOP_K)
    got10 = depth.inverted_fraction(verdicts_for(WS4_TOP_K), sep10, "sq8_np1",
                                   "sq8_np512", qids)
    assert got10 == pytest.approx(0.1081, abs=1e-4)


def test_inverted_fraction_is_none_with_no_discriminating_separable_questions():
    verdicts = {1: {"worst": True, "best": True}}
    assert depth.inverted_fraction(verdicts, {1: True}, "worst", "best",
                                  [1]) is None


def test_bootstrap_discrimination_stats_returns_ci_for_all_three_statistics():
    qids, buckets, separable = _synthetic_cells({
        1: (10, 50, 5, 50), 10: (30, 50, 5, 50)})
    got = depth.bootstrap_discrimination_stats(buckets, separable, qids,
                                              n_boot=300, seed=42)
    for k in (1, 10):
        for stat in ("share_ci", "separable_rate_ci", "null_ci"):
            lo, hi = got[k][stat]
            assert lo <= hi


def test_bootstrap_discrimination_stats_is_deterministic_for_a_seed():
    qids, buckets, separable = _synthetic_cells({
        1: (10, 50, 5, 50), 10: (30, 50, 5, 50)})
    a = depth.bootstrap_discrimination_stats(buckets, separable, qids,
                                            n_boot=300, seed=42)
    b = depth.bootstrap_discrimination_stats(buckets, separable, qids,
                                            n_boot=300, seed=42)
    assert a == b


def test_cell_rows_first_wins_keeps_the_first_of_two_duplicate_lines(tmp_path, monkeypatch):
    """IMPORTANT 4: a sensitivity check on the dedup rule -- agent_harness's
    load_checkpoint (used by cell_rows) keeps the LAST line for a
    duplicated (qid, arm) key; this keeps the FIRST."""
    from benchlib.recall_quality import depth_runner as ws4e_runner
    monkeypatch.setattr(ws4e_runner, "_cache_dir", lambda: tmp_path)
    (tmp_path / "runs_k1_tdefault.jsonl").write_text(
        '{"qid": "7", "arm": "a", "judge_correct": true}\n'
        '{"qid": "7", "arm": "a", "judge_correct": false}\n'
        '{"qid": "8", "arm": "a", "judge_correct": false}\n')
    first = ws4e_runner.cell_rows_first_wins(1, None)
    last = ws4e_runner.cell_rows(1, None)
    assert first[("7", "a")]["judge_correct"] is True
    assert last[("7", "a")]["judge_correct"] is False
    assert first[("8", "a")]["judge_correct"] is False


# --- task 10: mediator_by_k.csv -- the mediation chain per (arm, k) --------

def test_mediator_point_estimates_computes_presence_em_judge_conversion():
    """Two questions, one arm, one k: presence@k, em, judge accuracy and
    conversion = P(judge correct | present) computed from first principles."""
    from benchlib.recall_quality import depth as _ws4e
    cell_data = {
        ("a", 1): {
            1: (True, True, True),     # present, EM, correct
            2: (True, False, False),   # present, no EM, wrong
            3: (False, False, False),  # absent, no EM, wrong
            4: (False, False, True),   # absent, no EM, correct (leak)
        }
    }
    pts = _ws4e.mediator_point_estimates(cell_data, [1, 2, 3, 4])
    p = pts[("a", 1)]
    assert p["n"] == 4
    assert p["presence_at_k"] == pytest.approx(0.5)
    assert p["em"] == pytest.approx(0.25)
    assert p["judge_accuracy"] == pytest.approx(0.5)
    assert p["conversion"] == pytest.approx(0.5)   # correct among {1, 2}


def test_mediator_point_estimates_conversion_is_none_with_no_presence():
    """No question present at this cell -> conversion must not silently
    read as 0.0 (same posture as decompose's null/signal)."""
    from benchlib.recall_quality import depth as _ws4e
    cell_data = {("a", 1): {1: (False, False, True), 2: (False, False, False)}}
    pts = _ws4e.mediator_point_estimates(cell_data, [1, 2])
    assert pts[("a", 1)]["conversion"] is None


def test_bootstrap_mediator_stats_is_deterministic_and_orders_ci():
    from benchlib.recall_quality import depth as _ws4e
    cell_data = {
        ("a", 1): {q: (q % 2 == 0, q % 3 == 0, q % 4 == 0) for q in range(40)},
        ("a", 10): {q: (q % 2 == 1, q % 3 == 0, q % 5 == 0) for q in range(40)},
    }
    qids = list(range(40))
    a = _ws4e.bootstrap_mediator_stats(cell_data, qids, n_boot=200, seed=42)
    b = _ws4e.bootstrap_mediator_stats(cell_data, qids, n_boot=200, seed=42)
    assert a == b
    for cell in cell_data:
        for stat in ("presence_ci", "em_ci", "judge_accuracy_ci", "conversion_ci"):
            lo, hi = a[cell][stat]
            assert lo <= hi


def test_mediator_by_k_csv_reproduces_verified_values():
    """Spot-check of values computed independently by hand, read back from
    the committed CSV rather than re-derived, so a regression in the
    generating script is caught even if nobody re-runs it. sq8_np512 at k=1
    and k=10 (see results/recall-vs-quality.md's mediator finding: presence
    rises, conversion falls) plus sq8_np1 at k=1 and k=10."""
    from benchlib.config import ws4e_dir
    path = ws4e_dir() / "mediator_by_k.csv"
    if not path.exists():
        pytest.skip("results/ws4e/mediator_by_k.csv not generated in this checkout")
    df = pd.read_csv(path).set_index(["arm", "k"])

    def check(arm, k, presence, em, judge, conversion):
        r = df.loc[(arm, k)]
        assert r.presence_at_k == pytest.approx(presence, abs=1e-4)
        assert r.em == pytest.approx(em, abs=1e-4)
        assert r.judge_accuracy == pytest.approx(judge, abs=1e-4)
        assert r.conversion == pytest.approx(conversion, abs=1e-4)

    check("sq8_np1", 1, 0.4470, 0.2955, 0.4735, 0.7458)
    check("sq8_np512", 1, 0.5303, 0.3106, 0.5379, 0.7714)
    check("sq8_np512", 3, 0.8068, 0.3144, 0.5871, 0.6761)
    check("sq8_np512", 5, 0.8939, 0.3258, 0.6174, 0.6695)
    check("sq8_np1", 10, 0.8182, 0.2689, 0.5303, 0.6343)
    check("sq8_np512", 10, 1.0000, 0.3561, 0.6742, 0.6742)

    # The mediator finding this CSV must make legible: conversion falls as
    # k rises on sq8_np512, even though presence rises. ⚠️ That fall is a
    # composition artifact (conversion_composition_check.csv, see
    # results/recall-vs-quality.md) -- this CSV is unmodified and this assertion documents
    # the MOVING statistic only, not a real depth effect.
    assert df.loc[("sq8_np512", 1), "conversion"] > df.loc[("sq8_np512", 10), "conversion"]
    assert df.loc[("sq8_np512", 1), "presence_at_k"] < df.loc[("sq8_np512", 10), "presence_at_k"]


# --- composition-artifact fix: conversion_composition_check.csv ------------

def test_conversion_composition_check_holds_presence_fixed_at_k1():
    """Two questions, two depths, one arm, from first principles: presence
    grows from k=1 to k=10 (question 3 newly appears), but the FIXED set
    (present at k=1) is held to {1, 2} at every depth, unlike the MOVING
    set which grows to {1, 2, 3}."""
    from benchlib.recall_quality import depth as _ws4e
    cell_data = {
        ("a", 1): {
            1: (True, False, True),     # present at k=1, correct
            2: (True, False, False),    # present at k=1, wrong
            3: (False, False, True),    # absent at k=1
        },
        ("a", 10): {
            1: (True, False, True),     # still present, still correct
            2: (True, False, True),     # still present, now correct
            3: (True, False, False),    # NEWLY present at k=10, wrong
        },
    }
    check = _ws4e.conversion_composition_check(cell_data, [1, 2, 3],
                                                n_boot=50, seed=42)
    c = check["a"]
    assert c["k_lo"] == 1 and c["k_hi"] == 10
    k1, k10 = c["per_k"][1], c["per_k"][10]
    assert k1["n_fixed"] == k10["n_fixed"] == 2
    assert k1["fixed_conversion"] == pytest.approx(0.5)     # {1, 2} at k=1
    assert k10["fixed_conversion"] == pytest.approx(1.0)    # {1, 2} at k=10
    # moving conversion at k=10 is diluted by question 3's wrong answer --
    # this is exactly the composition artifact the fix documents.
    assert k10["moving_conversion"] == pytest.approx(2 / 3)
    assert k10["n_newly_present"] == 1 and k10["n_already_present"] == 2
    assert k10["newly_present_conversion"] == pytest.approx(0.0)
    assert k10["already_present_conversion"] == pytest.approx(1.0)
    assert k1["n_newly_present"] == 2 and k1["n_already_present"] == 0
    assert k1["already_present_conversion"] is None
    assert c["fixed_diff"] == pytest.approx(0.5)   # fixed conversion RISES here


def test_conversion_composition_check_conversion_is_none_with_no_presence():
    from benchlib.recall_quality import depth as _ws4e
    cell_data = {("a", 1): {1: (False, False, True), 2: (False, False, False)},
                ("a", 10): {1: (False, False, True), 2: (False, False, False)}}
    check = _ws4e.conversion_composition_check(cell_data, [1, 2],
                                                n_boot=20, seed=42)
    c = check["a"]
    assert c["per_k"][1]["fixed_conversion"] is None
    assert c["per_k"][1]["n_fixed"] == 0
    assert c["fixed_diff"] is None


def test_conversion_composition_check_is_deterministic():
    from benchlib.recall_quality import depth as _ws4e
    cell_data = {
        ("a", 1): {q: (q % 2 == 0, q % 3 == 0, q % 4 == 0) for q in range(40)},
        ("a", 10): {q: (q % 2 == 0 or q % 5 == 0, q % 3 == 0, q % 5 == 0)
                   for q in range(40)},
    }
    qids = list(range(40))
    a = _ws4e.conversion_composition_check(cell_data, qids, n_boot=200, seed=42)
    b = _ws4e.conversion_composition_check(cell_data, qids, n_boot=200, seed=42)
    assert a == b
    assert a["a"]["fixed_diff_ci_lo"] <= a["a"]["fixed_diff_ci_hi"]


def test_conversion_composition_check_csv_reproduces_verified_values():
    """Spot-check mirroring
    test_mediator_by_k_csv_reproduces_verified_values's posture: values
    verified independently by hand, read back from the committed CSV so a
    regression in the generating script is caught even if nobody re-runs
    it."""
    from benchlib.config import ws4e_dir
    path = ws4e_dir() / "conversion_composition_check.csv"
    if not path.exists():
        pytest.skip("results/ws4e/conversion_composition_check.csv not "
                    "generated in this checkout")
    df = pd.read_csv(path).set_index(["arm", "k"])

    def check_fixed(arm, k, fixed_conversion, n_fixed):
        r = df.loc[(arm, k)]
        assert r.fixed_conversion == pytest.approx(fixed_conversion, abs=1e-3)
        assert r.n_fixed == n_fixed

    check_fixed("sq8_np512", 1, 0.7714, 140)
    check_fixed("sq8_np512", 10, 0.7571, 140)
    check_fixed("sq8_np48", 1, 0.7714, 140)
    check_fixed("sq8_np48", 10, 0.7714, 140)
    check_fixed("sq8_np4", 1, 0.7669, 133)
    check_fixed("sq8_np4", 10, 0.7444, 133)

    def check_newly(arm, k, newly_conversion, n_new, already_conversion):
        r = df.loc[(arm, k)]
        assert r.newly_present_conversion == pytest.approx(newly_conversion, abs=1e-3)
        assert r.n_newly_present == n_new
        assert r.already_present_conversion == pytest.approx(already_conversion, abs=1e-3)

    check_newly("sq8_np512", 3, 0.5342, 73, 0.7500)
    check_newly("sq8_np512", 5, 0.5217, 23, 0.6854)
    check_newly("sq8_np512", 10, 0.3571, 28, 0.7119)
    check_newly("sq8_np48", 10, 0.3704, 27, 0.7106)
    check_newly("sq8_np4", 10, 0.4211, 19, 0.6621)

    # The corrected finding this CSV must make legible: for EVERY arm, the
    # fixed-set conversion fall (k_lo -> k_hi) is smaller in magnitude than
    # the moving conversion fall -- the moving fall is inflated by
    # composition, not a real per-question effect.
    for arm in df.index.get_level_values("arm").unique():
        k_lo = int(df.loc[arm, "k_lo"].iloc[0])
        k_hi = int(df.loc[arm, "k_hi"].iloc[0])
        moving_fall = (df.loc[(arm, k_lo), "moving_conversion"]
                      - df.loc[(arm, k_hi), "moving_conversion"])
        fixed_fall = (df.loc[(arm, k_lo), "fixed_conversion"]
                     - df.loc[(arm, k_hi), "fixed_conversion"])
        assert abs(fixed_fall) < abs(moving_fall), (
            f"{arm}: fixed-set conversion fall {fixed_fall} is not smaller "
            f"than the moving conversion fall {moving_fall}")
        # And every arm's paired fixed-set CI must include zero -- depth
        # does not measurably change conversion once composition is held
        # fixed.
        ci_lo = df.loc[(arm, k_hi), "fixed_diff_ci_lo"]
        ci_hi = df.loc[(arm, k_hi), "fixed_diff_ci_hi"]
        assert ci_lo <= 0 <= ci_hi, (
            f"{arm}: fixed-set paired CI [{ci_lo}, {ci_hi}] excludes zero")
