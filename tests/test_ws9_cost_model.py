"""WS9's WS5 wiring: the new points are present and gated, and NOTHING that
was already there moved.

results/ws5/{break_even,churn,sensitivity}.csv are copied verbatim by the deck,
whose data/SOURCES.md documents their current values. Adding a workload must
append rows and (for churn/sensitivity) one column -- never change an existing
number.
"""

import pandas as pd
import pytest

from benchlib import config

WS5 = config.ws5_dir()
# Snapshot of results/ws5/ taken BEFORE the WS9 wiring landed, by Task 8 step 1.
# Gitignored: this is a one-shot migration guard, not a committed fixture. The
# two tests that use it skip when it is absent, so a clean clone still passes --
# at the cost of proving nothing, which is why the plan takes it up front.
BEFORE = config.DATA_DIR / "ws9_cache" / "ws5_before"
needs_snapshot = pytest.mark.skipif(
    not BEFORE.exists(),
    reason=f"no pre-WS9 snapshot at {BEFORE}; see the plan's Task 8 step 1")


@pytest.fixture(scope="module")
def be():
    return pd.read_csv(WS5 / "break_even.csv")


@pytest.fixture(scope="module")
def gate():
    return pd.read_csv(WS5 / "postdiction_gate.csv")


def test_code_unseen_points_exist_at_every_size_for_both_topk(be):
    u = be[be.workload == "code_unseen"]
    assert not u.empty, "no code_unseen rows in break_even.csv"
    assert set(u["size"]) == {"S", "M", "L"}
    assert set(u.index_arm) == {"indexed", "indexed_topk3"}
    assert set(u.live_arm) == {"agentic"}, (
        "agentic is the only live arm on this workload -- there is no stuffed "
        "arm at any size (the corpus is 4.4x over the prefix budget)")
    # A break_even_qpd per top-k at every size, under the headline settings.
    head = u[(u.hit_rate_basis == "measured") & (u.regime == "claude")
             & (u.footprint == "pca_uc_384_sq8") & (u.infra_mode == "dedicated")]
    assert len(head) == 6, head[["size", "index_arm"]].to_dict("records")


def test_every_code_unseen_point_is_gated_on_both_asserted_bases(gate):
    u = gate[gate.workload == "code_unseen"]
    assert not u.empty
    asserted = u[u.basis.isin(["mean_billed", "mean_uncached_list"])]
    # 4 arms at L (agentic, indexed, indexed_topk3, parametric) + 3 arms at
    # each of S and M = 10 points, x 2 asserted bases.
    assert len(asserted) == 20, len(asserted)
    assert asserted.within_10pct.all(), (
        "POSTDICTION GATE FAILED on code_unseen:\n"
        + asserted[~asserted.within_10pct][
            ["size", "arm", "basis", "rel_error"]].to_string())
    assert asserted.rel_error.abs().max() <= 0.10


def test_the_parametric_floor_is_at_l_only(gate):
    par = gate[(gate.workload == "code_unseen") & (gate.arm == "parametric")]
    assert set(par["size"]) == {"L"}, (
        "parametric has no corpus access, so it is size-independent by "
        "construction and must not be run or modelled at S or M")
    assert set(par.role) == {"floor"}


@needs_snapshot
def test_existing_workloads_are_byte_identical(be):
    """The deck copies this file. Adding a workload may only append rows."""
    before = pd.read_csv(BEFORE / "break_even.csv")
    after = be[be.workload != "code_unseen"].reset_index(drop=True)
    assert list(after.columns) == list(before.columns), "a column moved"
    pd.testing.assert_frame_equal(after, before, check_exact=True)


@needs_snapshot
@pytest.mark.parametrize("name", ["churn", "sensitivity"])
def test_churn_and_sensitivity_gain_a_workload_column_and_keep_memory_intact(name):
    after = pd.read_csv(WS5 / f"{name}.csv")
    before = pd.read_csv(BEFORE / f"{name}.csv")
    assert "workload" in after.columns, f"{name}.csv has no workload column"
    assert (after.workload == "code_unseen").any(), f"{name}.csv has no WS9 rows"
    mem = after[after.workload == "memory"]
    # churn.csv already carried `workload` before WS9 (be_row emits it, as
    # break_even.csv's does); sensitivity.csv did not. Drop it only where the
    # snapshot lacks it, so the column-for-column comparison is like for like.
    if "workload" not in before.columns:
        mem = mem.drop(columns=["workload"])
    mem = mem.reset_index(drop=True)
    assert list(mem.columns) == list(before.columns)
    pd.testing.assert_frame_equal(mem, before, check_exact=True)


def test_sensitivity_ranks_per_workload_not_across_them():
    """max_spread_ratio must be computed WITHIN a workload.

    Ranking across workloads would silently rewrite the memory rows' spreads --
    the numbers the deck's SOURCES.md documents knob by knob.

    What this proves: for every (workload, knob) the stored max_spread_ratio
    equals the spread recomputed from THAT workload's own rows, by the
    builder's own recipe (max over sizes of max/min of break_even_qpd across
    the knob's rows plus the base rows, feasible and finite only, rounded to
    2). A ranking computed across workloads and merged on knob alone would
    stamp one cross-workload value on both workloads' rows, which a mere
    "one value per (workload, knob)" check cannot see. This holds on a clean
    clone, where the snapshot regression skips.
    """
    import math

    import numpy as np

    s = pd.read_csv(WS5 / "sensitivity.csv")
    assert "workload" in s.columns
    checked = 0
    for workload, w in s.groupby("workload"):
        base = w[w.knob == "base"]
        for knob, k in w.groupby("knob"):
            if knob == "base":
                continue
            stored = k.max_spread_ratio.unique()
            assert len(stored) == 1, (workload, knob, stored)
            d = pd.concat([k, base])
            d = d[d.feasible.astype(bool) & np.isfinite(d.break_even_qpd.astype(float))]
            by_size = d.groupby("size").break_even_qpd
            want = round(float((by_size.max() / by_size.min()).max()), 2)
            assert math.isclose(float(stored[0]), want, abs_tol=1e-9), (
                f"{workload}/{knob}: stored max_spread_ratio {stored[0]} != {want} "
                "recomputed from this workload's own rows -- the ranking crossed "
                "workloads")
            checked += 1
    # Every distinct non-base knob was checked in BOTH workloads.
    knobs = set(s.knob) - {"base"}
    assert set(s.workload) == {"memory", "code_unseen"}, set(s.workload)
    assert len(knobs) >= 9 and checked == 2 * len(knobs), (checked, sorted(knobs))


def test_index_footprint_is_derived_from_the_measured_chunk_count(be):
    """The footprint is DERIVED, not estimated -- the same path WS6a's takes.

    This is the test that retires the deck's refusal of the marginal and
    serverless rows for this corpus ("agentic-hil's footprint was never
    measured"). It is measured chunk count x EMBED_DIMS x 4 bytes / the
    measured footprint multiplier, and nothing here is fitted or guessed.
    """
    from benchlib import cost_model

    EMBED_DIMS = 1536                      # text-embedding-3-small default
    # The multiplier the notebook prices with is the MEASURED value from
    # results/ws3/dim_verdict_10m.csv (8.6596..., quoted as 8.66x); the notebook
    # loads it from the same file and only asserts it within 0.01, so a literal
    # 8.66 would miss the derivation below at the 1e-6 this test holds it to.
    ws3 = pd.read_csv(config.RESULTS_DIR / "ws3" / "dim_verdict_10m.csv").set_index("arm")
    PCA_UC_384_SQ8 = float(ws3.loc["pca_uc_384_sq8", "footprint_compression_vs_flat_1024_1m"])
    assert abs(PCA_UC_384_SQ8 - 8.66) < 0.01
    idx = pd.read_csv(config.ws6c_dir() / "index_cost.csv").set_index("scope")
    u = be[(be.workload == "code_unseen") & (be.footprint == "pca_uc_384_sq8")
           & (be.infra_mode == "dedicated") & (be.index_arm == "indexed")]
    for size in ("S", "M", "L"):
        row = u[u["size"] == size].iloc[0]
        want = cost_model.index_footprint_bytes(
            int(idx.loc[size, "chunk_count"]), EMBED_DIMS, PCA_UC_384_SQ8) / 2 ** 30
        assert abs(row.footprint_gib - want) / want < 1e-6, (
            f"{size}: footprint_gib {row.footprint_gib} is not the derivation "
            f"over {int(idx.loc[size, 'chunk_count'])} measured chunks ({want})")


def test_the_dedicated_fixed_cost_is_flat_across_the_curve(be):
    """Spec section 3.1, declared in advance: a 16 GiB instance holds a 5 MB
    index as easily as a 1.5 MB one, so the curve's SHAPE is the per-query
    saving and nothing else. If this ever fails, the report's explanation of
    the curve is wrong and must change with it."""
    u = be[(be.workload == "code_unseen") & (be.regime == "claude")
           & (be.hit_rate_basis == "measured") & (be.footprint == "pca_uc_384_sq8")
           & (be.infra_mode == "dedicated") & (be.index_arm == "indexed")]
    fixed = u.set_index("size").fixed_daily
    assert fixed.max() - fixed.min() < 0.01, fixed.to_dict()
