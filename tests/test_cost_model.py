"""WS5: the cost model must be arithmetic anyone can check by hand, and it
must postdict the measured WS6 points it is built from."""

import math

import pandas as pd
import pytest

from benchlib import config, cost_model

CLAUDE = cost_model.Regime("claude", "claude-sonnet-5", 2.0, 10.0, 2.0, 0.1, 1_000_000)
OPENAI = cost_model.Regime("openai", "gpt-5.6-terra", 2.0, 12.0, 1.0, 0.1, 922_000,
                    long_context_threshold=272_000, long_input_usd_per_mtok=4.0,
                    long_cached_usd_per_mtok=0.4, long_output_usd_per_mtok=18.0)


def test_live_cost_by_hand():
    # 100k prefix, 20 dynamic, 30 out, 90% hits: 0.9*0.2 + 0.1*4.0 = 0.58 $/MTok on prefix
    c = cost_model.live_query_cost(CLAUDE, 100_000, 20, 30, 0.9)
    assert c == pytest.approx((100_000 * 0.58 + 20 * 2 + 30 * 10) / 1e6)
    # list price ignores the cache entirely
    assert cost_model.live_query_cost(CLAUDE, 100_000, 20, 30, 0.9, cached=False) == \
        pytest.approx((100_020 * 2 + 30 * 10) / 1e6)


def test_window_is_a_boundary_not_a_price():
    assert cost_model.live_query_cost(CLAUDE, 1_000_001, 0, 0, 1.0) is None
    assert cost_model.live_query_cost(OPENAI, 959_269, 18, 117, 0.97) is None
    assert cost_model.index_query_cost(OPENAI, 922_001, 1) is None


def test_openai_long_context_reprices_whole_request():
    inp, wr, rd, out = OPENAI.rates(272_001)
    assert (inp, wr, rd, out) == (4.0, 4.0, 0.4, 18.0)
    assert OPENAI.rates(272_000) == (2.0, 2.0, 0.2, 12.0)


def test_hit_rate_validation():
    with pytest.raises(ValueError):
        cost_model.live_query_cost(CLAUDE, 1, 1, 1, 1.5)


def test_footprint_uses_measured_ratio():
    raw = cost_model.raw_fp32_bytes(10_000_000, 1024)
    assert raw == 10_000_000 * 1024 * 4
    assert cost_model.index_footprint_bytes(10_000_000, 1024, 3.54) == pytest.approx(raw / 3.54)


def test_infra_modes():
    kw = dict(ram_headroom=1.0, instance_gib=16, instance_usd_per_hour=0.1323,
              usd_per_gib_month=0.1323 * cost_model.HOURS_PER_MONTH / 16)
    tiny = 2 * 2**20  # 2 MiB
    # dedicated: the floor is one whole instance
    assert cost_model.infra_monthly_usd(tiny, "dedicated", **kw) == pytest.approx(0.1323 * cost_model.HOURS_PER_MONTH)
    # marginal: 2 MiB of a $/GiB rate
    assert cost_model.infra_monthly_usd(tiny, "marginal", **kw) == pytest.approx(kw["usd_per_gib_month"] * 2 / 1024)
    # 17 GiB needs two instances
    assert cost_model.infra_monthly_usd(17 * 2**30, "dedicated", **kw) == pytest.approx(2 * 0.1323 * cost_model.HOURS_PER_MONTH)
    with pytest.raises(ValueError):
        cost_model.infra_monthly_usd(1, "serverless", **kw)


def test_churn_and_fixed_daily():
    # 1% of 1.2M tokens/day, mid-edit blast x5, $0.02/MTok
    assert cost_model.churn_daily_usd(1_200_000, 0.01, 0.02, 5.0) == pytest.approx(1_200_000 * 0.01 * 5 * 0.02 / 1e6)
    assert cost_model.fixed_daily_usd(30.4375, 365.25, 365.25, 0.5) == pytest.approx(1 + 1 + 0.5)


def test_break_even_branches():
    assert cost_model.break_even_qpd(0.02, 0.01, 3.0) == pytest.approx(300)
    assert cost_model.break_even_qpd(0.01, 0.02, 3.0) == math.inf     # live cheaper: never repays
    assert cost_model.break_even_qpd(None, 0.02, 3.0) is None          # infeasible
    assert cost_model.payback_days(10.0, 1.0, 200, 0.02, 0.01) == pytest.approx(10.0)
    assert cost_model.payback_days(10.0, 1.0, 50, 0.02, 0.01) == math.inf


def test_stretch_hit_rate_acts_on_misses():
    assert cost_model.stretch_hit_rate(0.9, 2.0) == pytest.approx(0.8)
    assert cost_model.stretch_hit_rate(0.9, 0.5) == pytest.approx(0.95)
    assert cost_model.stretch_hit_rate(0.0, 2.0) == 0.0


def test_gate_helpers():
    assert cost_model.postdiction_error(1.05, 1.0) == pytest.approx(0.05)
    assert cost_model.postdiction_error(None, 1.0) is None
    assert cost_model.gate_passes([0.05, -0.09]) and not cost_model.gate_passes([0.11]) \
        and not cost_model.gate_passes([None])


# ------------------------------------------------- postdiction on real data ----

@pytest.mark.skipif(not (config.RESULTS_DIR / "ws6b" / "runs.csv").exists(),
                    reason="committed WS6b runs not present")
def test_model_postdicts_ws6b_replay_at_every_size():
    """Corpus tokens + measured hit rate + measured output must reproduce the
    mean billed cost per query within 10% at S, M and L."""
    runs = pd.read_csv(config.RESULTS_DIR / "ws6b" / "runs.csv")
    stats = pd.read_csv(config.RESULTS_DIR / "ws6b" / "corpus_stats.csv").set_index("size")
    ok = runs[runs.status == "ok"]
    for size in ("s", "m", "l"):
        rep = ok[(ok["size"] == size) & (ok.arm_base.str.startswith("replay"))]
        n = len(rep)
        reads, writes = rep.cache_read_input_tokens.sum(), rep.cache_creation_input_tokens.sum()
        h = reads / (reads + writes)
        # the cacheable prefix is the measured corpus x the measured fit
        # fraction (1.0 at S/M; 0.785 at L where replay is truncated)
        prefix = stats.loc[size, "count_tokens"] * stats.loc[size, "fit_fraction"]
        modelled = cost_model.live_query_cost(CLAUDE, prefix, rep.input_tokens.mean(),
                                       rep.output_tokens.mean(), h)
        measured = rep.cost_billed_usd.mean()
        assert abs(cost_model.postdiction_error(modelled, measured)) <= 0.10, (size, modelled, measured)
