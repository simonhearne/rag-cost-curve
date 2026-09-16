"""WS5 cost model — pure arithmetic, no I/O, no guesses.

Every function here is fed numbers that come from a committed CSV (measured
token counts, measured footprints, measured skip rates) or from
`results/ws5/pricing.csv` (a fetched price with source URL + date). The
notebook is the only place those numbers are loaded; this module only
combines them. Nothing here may invent a token count.

Design notes
------------
* A "live" query sends a cacheable prefix (the corpus, history or stuffed
  repository), a small per-query uncached part (question + tool frames) and
  produces output tokens. With hit rate ``h`` the prefix is served from cache
  on a fraction ``h`` of queries and written on the rest. ``cached=False``
  prices the same tokens at list rate, which is the strawman the WS6 results
  warned against; both are computed so the gap is visible.
* Any request whose input exceeds the regime's context window is a hard
  capability boundary, returned as ``None`` — never an extrapolated cost
  (see results/code-retrieval.md).
* OpenAI bills the whole request at long-context rates once input exceeds a
  threshold; that is modelled explicitly because WS6b's M and L points cross
  it.
* Break-even is ``fixed_daily / (c_live - c_index_query)``. If the live arm
  is cheaper per query the index never repays and the break-even is
  ``math.inf`` (WS6a's agentic-vs-indexed result).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_M = 1_000_000
DAYS_PER_MONTH = 365.25 / 12  # 30.4375
HOURS_PER_MONTH = 24 * DAYS_PER_MONTH  # 730.5


@dataclass(frozen=True)
class Regime:
    """One provider's price list. All $/MTok. Multipliers are of input."""

    name: str
    model: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cache_write_multiplier: float
    cache_read_multiplier: float
    context_window_tokens: int
    # OpenAI-style long-context tier: whole request re-priced above threshold.
    long_context_threshold: int | None = None
    long_input_usd_per_mtok: float | None = None
    long_cached_usd_per_mtok: float | None = None
    long_output_usd_per_mtok: float | None = None

    def rates(self, total_input_tokens: int) -> tuple[float, float, float, float]:
        """(input, cache_write, cache_read, output) $/MTok for a request."""
        inp, out = self.input_usd_per_mtok, self.output_usd_per_mtok
        rd = inp * self.cache_read_multiplier
        if (self.long_context_threshold is not None
                and total_input_tokens > self.long_context_threshold):
            inp = self.long_input_usd_per_mtok
            rd = self.long_cached_usd_per_mtok
            out = self.long_output_usd_per_mtok
        wr = inp * self.cache_write_multiplier
        return inp, wr, rd, out


def live_query_cost(regime: Regime, prefix_tokens: float, dynamic_tokens: float,
                    output_tokens: float, hit_rate: float,
                    cached: bool = True) -> float | None:
    """Expected $ for one live query. ``None`` if it does not fit the window."""
    if not 0.0 <= hit_rate <= 1.0:
        raise ValueError(f"hit_rate must be in [0, 1], got {hit_rate}")
    total_in = prefix_tokens + dynamic_tokens
    if total_in > regime.context_window_tokens:
        return None
    inp, wr, rd, out = regime.rates(total_in)
    if not cached:
        return (total_in * inp + output_tokens * out) / _M
    prefix = prefix_tokens * (hit_rate * rd + (1.0 - hit_rate) * wr)
    return (prefix + dynamic_tokens * inp + output_tokens * out) / _M


def index_query_cost(regime: Regime, prompt_tokens: float,
                     output_tokens: float) -> float | None:
    """$ for one index-arm query: retrieved chunks + question in, answer out.

    No caching: WS6 measured cache_read == 0 on every index-arm row because
    the retrieved passages differ per question. Query-embedding cost is
    ~20 tokens at $0.02/MTok and is dropped (< $1e-6).
    """
    if prompt_tokens > regime.context_window_tokens:
        return None
    inp, _, _, out = regime.rates(prompt_tokens)
    return (prompt_tokens * inp + output_tokens * out) / _M


# ------------------------------------------------------------- index arm ----

def raw_fp32_bytes(n_vectors: float, dims: int) -> float:
    return n_vectors * dims * 4.0


def index_footprint_bytes(n_vectors: float, dims: int,
                          footprint_multiplier: float) -> float:
    """Loaded bytes = raw fp32 / MEASURED footprint multiplier (WS2/WS3).

    The multiplier is the measured ratio raw-fp32 : loaded-index at 10M on
    1024d — 3.54x for IVF_SQ8, 3.11x for RaBitQ+SQ8-refine, 8.66x for
    pca_uc_384 × SQ8. It is a ratio, so it is applied to whatever raw size the
    deployed embedding has (1536d for text-embedding-3-small in WS6).
    """
    if footprint_multiplier <= 0:
        raise ValueError("footprint multiplier must be positive")
    return raw_fp32_bytes(n_vectors, dims) / footprint_multiplier


def infra_monthly_usd(footprint_bytes: float, mode: str, *,
                      ram_headroom: float, instance_gib: float,
                      instance_usd_per_hour: float,
                      usd_per_gib_month: float) -> float:
    """Monthly serving cost of holding ``footprint_bytes`` in RAM.

    mode="dedicated": whole instances (the Milvus cost-optimisation guide's
    sizing approach): ceil(need / instance_gib) × instance price. The
    smallest instance is a FLOOR that a 2 MB index pays in full.
    mode="marginal": you already run a cluster and pay only for the GiB you
    add — the per-GiB rate derived from the same instance price.
    """
    need_gib = footprint_bytes * ram_headroom / 2**30
    if mode == "dedicated":
        n = max(1, math.ceil(need_gib / instance_gib))
        return n * instance_usd_per_hour * HOURS_PER_MONTH
    if mode == "marginal":
        return need_gib * usd_per_gib_month
    raise ValueError(f"unknown infra mode {mode!r}")


def embed_cost_usd(tokens: float, usd_per_mtok: float) -> float:
    return tokens / _M * usd_per_mtok


def churn_daily_usd(corpus_embed_tokens: float, churn_frac_per_day: float,
                    embed_usd_per_mtok: float, reembed_factor: float) -> float:
    """Re-embedding $/day when ``churn_frac_per_day`` of the corpus changes.

    ``reembed_factor`` is chunks re-embedded per chunk changed, from WS6b's
    measured skip rates: 1.0 for appends/new files (1 of 2,907 re-embedded),
    5.0 for a mid-file edit (5 of 2,906 — line numbers are inside the chunk
    id, so every chunk after the edit in that file is re-embedded). An
    embedding-model migration is a one-off at factor 1.0 over the WHOLE
    corpus (0% skip) and is not a rate; see the notebook.
    """
    if not 0.0 <= churn_frac_per_day <= 1.0:
        raise ValueError("churn fraction must be in [0, 1]")
    return embed_cost_usd(corpus_embed_tokens * churn_frac_per_day
                          * reembed_factor, embed_usd_per_mtok)


def fixed_daily_usd(infra_monthly: float, embed_once: float,
                    amortisation_days: float, churn_daily: float) -> float:
    """$/day the index costs before a single query is asked."""
    return infra_monthly / DAYS_PER_MONTH + embed_once / amortisation_days + churn_daily


def break_even_qpd(c_live: float | None, c_index_query: float | None,
                   fixed_daily: float) -> float | None:
    """Queries/day at which the index's daily bill equals live search's.

    None   → the live arm cannot run in this regime (window exceeded).
    inf    → live is cheaper per query; the index never repays its floor.
    """
    if c_live is None or c_index_query is None:
        return None
    saving = c_live - c_index_query
    if saving <= 0:
        return math.inf
    return fixed_daily / saving


def payback_days(fixed_one_off: float, fixed_daily: float, queries_per_day: float,
                 c_live: float, c_index_query: float) -> float:
    """Days until cumulative index spend drops below cumulative live spend."""
    saving_per_day = queries_per_day * (c_live - c_index_query) - fixed_daily
    if saving_per_day <= 0:
        return math.inf
    return fixed_one_off / saving_per_day


# ------------------------------------------------------------ gate helper ----

def postdiction_error(modelled: float | None, measured: float) -> float | None:
    """Signed relative error (modelled - measured) / measured."""
    if modelled is None:
        return None
    return (modelled - measured) / measured


def gate_passes(errors: list[float | None], tol: float = 0.10) -> bool:
    return all(e is not None and abs(e) <= tol for e in errors)


def stretch_hit_rate(hit_rate: float, factor: float) -> float:
    """'x2 / ÷2 on the cache hit rate' applied to the MISS rate.

    A hit rate of 0.97 cannot be doubled; what can be doubled or halved is
    the miss rate. factor=2 doubles the misses, factor=0.5 halves them.
    """
    miss = (1.0 - hit_rate) * factor
    return max(0.0, min(1.0, 1.0 - miss))
