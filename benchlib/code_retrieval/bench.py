"""WS6a pure functions: corpus enumeration, tiering, prefix construction,
scoring, cost arithmetic, run ordering.

Everything here is deterministic and offline -- unit-tested in
tests/code_retrieval/test_bench.py. Anything that calls an API lives in agent_harness.py and was
exercised by a pre-flight smoke script against live services (since removed;
see git history for scripts/smoke_ws6a_arms.py).
"""

import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..config import (
    WS6A_PREFIX_BUDGET_TOKENS,
    WS6A_SUBSETS,
    WS6A_TEXT_EXTENSIONS,
    WS6A_TIER_ORDER,
)


# ----------------------------------------------------- corpus enumeration


def is_text_file(path: str) -> bool:
    """True if `path` has an allowlisted text extension.

    Extension-based, not content-sniffing: the allowlist is a pinned constant
    so the corpus definition is reproducible from config alone.
    """
    return os.path.splitext(path)[1] in WS6A_TEXT_EXTENSIONS


def filter_text_files(paths) -> list[str]:
    return [p for p in paths if is_text_file(p)]


def subset_files(paths, subset: str) -> list[str]:
    """Restrict `paths` to one of the nested scaling subsets (S / M / L).

    L has an empty prefix tuple and means the whole repo.
    """
    if subset not in WS6A_SUBSETS:
        raise ValueError(f"unknown subset {subset!r}; expected one of {sorted(WS6A_SUBSETS)}")
    prefixes = WS6A_SUBSETS[subset]
    if not prefixes:
        return list(paths)
    return [p for p in paths if p.startswith(prefixes)]


def git_ls_files(repo_root: Path) -> list[str]:
    """Tracked paths at the checked-out commit, in git's own order.

    NUL-separated (`-z`) and split on `\\0`, not on whitespace: a plain
    `.split()` shreds any path containing a space into two bogus entries.
    """
    out = subprocess.check_output(
        ["git", "-C", str(repo_root), "ls-files", "-z"], text=True
    )
    return [p for p in out.split("\0") if p]


def git_last_modified(repo_root: Path, path: str) -> str:
    """Committer date (ISO-8601) of the last commit touching `path`."""
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "log", "-1", "--format=%cI", "--", path],
        text=True,
    ).strip()


# ------------------------------------------------- tiering and prefix build


def tier_rank(path: str) -> int:
    """Index of `path`'s tier in WS6A_TIER_ORDER; len(tiers) if it matches none.

    Lower sorts earlier. Everything unmatched shares the last rank and is
    ordered lexicographically among itself.
    """
    for i, prefix in enumerate(WS6A_TIER_ORDER):
        if path.startswith(prefix):
            return i
    return len(WS6A_TIER_ORDER)


def tiered_order(paths) -> list[str]:
    """Deterministic relevance ordering for the arm (c) prefix.

    Sort key is (tier rank, path) so the ordering is total and reproducible
    from the pinned commit alone.
    """
    return sorted(paths, key=lambda p: (tier_rank(p), p))


@dataclass(frozen=True)
class PrefixResult:
    included: list[str]
    tokens: int
    total_tokens: int
    fit_fraction: float


def build_prefix(paths, token_counts: dict, budget: int = WS6A_PREFIX_BUDGET_TOKENS
                 ) -> PrefixResult:
    """Greedily fill `budget` tokens from `paths` in tiered order.

    A file bigger than the remaining budget is skipped rather than ending the
    build, so a single oversized file cannot truncate the prefix early. The
    ordering is still strictly tiered -- skipping changes what fits, never the
    order things are considered in.
    """
    ordered = tiered_order(paths)
    total = sum(token_counts[p] for p in ordered)
    included: list[str] = []
    used = 0
    for p in ordered:
        n = token_counts[p]
        if used + n > budget:
            continue
        included.append(p)
        used += n
    fit = (used / total) if total else 0.0
    return PrefixResult(included=included, tokens=used,
                        total_tokens=total, fit_fraction=fit)


def classify_prefix_membership(expected_files, included) -> str:
    """`all` / `partial` / `none` -- three-state by design.

    A multi-hop question's expected files can straddle the prefix boundary. A
    boolean would silently call that in-prefix, conflating "long-context QA
    ability" with "the answer was only half there".
    """
    expected = list(expected_files)
    if not expected:
        raise ValueError("expected_files must be non-empty to classify membership")
    inc = set(included)
    hits = sum(1 for f in expected if f in inc)
    if hits == len(expected):
        return "all"
    if hits == 0:
        return "none"
    return "partial"


# ------------------------------------------------------ cost and governor


@dataclass
class Usage:
    """Token counts straight from API `usage` fields. Never estimated."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens

    @classmethod
    def from_api(cls, usage) -> "Usage":
        """Build from an SDK usage object. Absent fields are 0, never guessed."""
        return cls(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(
                usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(
                usage, "cache_read_input_tokens", 0) or 0,
        )

    def as_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }


def load_pricing(path, regime: str = "standard") -> dict:
    """Read results/ws6a/pricing.csv into {model: {rate fields}} for one regime."""
    import csv
    prices = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["regime"] != regime:
                continue
            prices[row["model"]] = {
                "input_usd_per_mtok": float(row["input_usd_per_mtok"]),
                "output_usd_per_mtok": float(row["output_usd_per_mtok"]),
                "cache_write_1h_multiplier": float(row["cache_write_1h_multiplier"]),
                "cache_read_multiplier": float(row["cache_read_multiplier"]),
                "source_url": row["source_url"],
                "fetched_date": row["fetched_date"],
            }
    if not prices:
        raise ValueError(f"no pricing rows for regime {regime!r} in {path}")
    return prices


_M = 1_000_000


def cost_billed(usage: Usage, price: dict) -> float:
    """What the run actually costs, honouring cache read/write rates."""
    inp = price["input_usd_per_mtok"]
    return round(
        usage.input_tokens / _M * inp
        + usage.output_tokens / _M * price["output_usd_per_mtok"]
        + usage.cache_creation_input_tokens / _M * inp
          * price["cache_write_1h_multiplier"]
        + usage.cache_read_input_tokens / _M * inp
          * price["cache_read_multiplier"],
        6,
    )


def cost_uncached_list(usage: Usage, price: dict) -> float:
    """What a cost model that ignores caching would predict.

    The gap against cost_billed is the C9 story: report only the cached number
    and you understate live search; report only this one and you strawman it.
    """
    inp = price["input_usd_per_mtok"]
    all_input = (usage.input_tokens
                 + usage.cache_creation_input_tokens
                 + usage.cache_read_input_tokens)
    return round(
        all_input / _M * inp
        + usage.output_tokens / _M * price["output_usd_per_mtok"],
        6,
    )


class BudgetExceeded(RuntimeError):
    """Raised when a charge would push cumulative spend past the governor."""


class CostGovernor:
    """Hard abort on cumulative billed spend across every phase.

    A rejected charge is not recorded, so a resumed run never double-counts
    spend that did not happen.
    """

    def __init__(self, limit: float, spent: float = 0.0):
        self.limit = limit
        self.spent = spent

    @property
    def remaining(self) -> float:
        return self.limit - self.spent

    def charge(self, usd: float) -> float:
        if self.spent + usd > self.limit:
            raise BudgetExceeded(
                f"charge ${usd:.4f} would exceed the ${self.limit:.2f} governor "
                f"(spent ${self.spent:.2f}); run is checkpointed and resumable"
            )
        self.spent += usd
        return self.spent


# --------------------------------------------------------- run ordering


def interleaved_order(qids, arms, seed: int) -> list[tuple[str, str]]:
    """Round-robin (qid, arm) execution order.

    Each arm gets its OWN shuffled question order (derived from `seed` plus the
    arm name) so no arm sees questions in authoring order, and arms are
    round-robined so time-of-day variance in API latency cannot bias one arm.
    """
    arms = list(arms)
    per_arm = {}
    for i, arm in enumerate(arms):
        rng = random.Random(f"{seed}:{arm}:{i}")
        shuffled = list(qids)
        rng.shuffle(shuffled)
        per_arm[arm] = shuffled
    order: list[tuple[str, str]] = []
    for round_i in range(len(qids)):
        for arm in arms:
            order.append((per_arm[arm][round_i], arm))
    return order


# ------------------------------------------------------ programmatic scoring


def normalise_path(p: str) -> str:
    """Strip decoration so `/fastapi/a.py`, `./fastapi/a.py` and `fastapi/a.py`
    all compare equal. Backslashes are normalised for Windows-style mentions.

    Dot-prefixed paths (e.g. `.github/workflows/`) are preserved; only the
    literal `./` prefix is removed, then leading `/` is stripped.
    """
    p = p.strip().strip("`'\"").replace("\\", "/")
    # Remove literal "./" prefix (may appear multiple times)
    while p.startswith("./"):
        p = p[2:]
    # Remove leading "/" only
    p = p.lstrip("/")
    return p


def score_programmatic(answer: str, expected_files, expected_symbols) -> dict:
    """Match expected repo-relative paths and symbols against the answer text.

    Full relative paths only -- a bare basename does NOT count. `utils.py`
    appears in dozens of places in a real repo, and crediting it would
    over-score every arm equally and wash out the deltas.

    File matching uses segment boundaries to prevent false positives:
    `fastapi/routing.py` will not match `vendor/fastapi/routing.py` or
    `notfastapi/routing.py`. A trailing `.` is explicitly permitted
    (sentence-final period: "The handler lives in fastapi/routing.py.")
    because sentence punctuation is far more common than filenames ending
    in `.orig` or `.bak`, which do not exist in the target repo.

    Case-sensitive matching is deliberate: under-crediting a differently-cased
    path is the safe direction for the mechanical layer; the LLM judge
    added in a later task is the intended backstop for variants.
    """
    import re

    hay = answer.replace("\\", "/")
    matched = []
    for f in expected_files:
        target = normalise_path(f)
        if target:
            # Match with segment boundaries. Reject if preceded by:
            # - word char, dot, or hyphen (would form joined names/extensions)
            # - word-or-hyphen followed by / (would form a path segment like vendor/)
            # Allow if preceded by standalone / (e.g. /fastapi/routing.py).
            # Reject if followed by word char or / (would form joined names/paths).
            pattern = (
                r"(?<![a-zA-Z0-9_.\-])"
                r"(?<![a-zA-Z0-9_-]/)"
                + re.escape(target)
                + r"(?![a-zA-Z0-9_/])"
            )
            if re.search(pattern, hay):
                matched.append(f)

    symbols = [s for s in expected_symbols if s]
    symbol_hit = any(
        re.search(rf"\b{re.escape(s)}\b", answer) for s in symbols
    ) if symbols else False

    expected = list(expected_files)
    return {
        "any_file_hit": len(matched) > 0,
        "all_files_hit": bool(expected) and len(matched) == len(expected),
        "symbol_hit": symbol_hit,
        "matched_files": matched,
    }


# ------------------------------------------------------ question validation

VALID_DIFFICULTIES = ("locate", "trace", "multi_hop")


def split_field(value: str) -> list[str]:
    """`;`-separated multi-value cell -> list, blanks dropped."""
    return [v.strip() for v in (value or "").split(";") if v.strip()]


def validate_questions(rows, repo_files, check_cardinality: bool = True,
                       scaling_prefix: str = "fastapi/") -> list[str]:
    """Return a list of problems; an empty list means the set is valid.

    Enforces the two constraints that would silently corrupt results if
    violated: every expected file must exist at the pinned commit, and every
    scaling_subset question must be answerable inside `scaling_prefix` alone.
    That prefix defaults to WS6a's own smallest tier, `fastapi/`; WS6c passes
    its corpus's most-central source directory instead.

    With `check_cardinality` (the default), also enforces the pinned
    cardinality rules from config.py: exactly WS6A_N_QUESTIONS rows, exactly
    WS6A_N_SCALING_QUESTIONS marked `scaling_subset`, the WS6A_DIFFICULTY_MIX
    split, and non-empty `expected_symbols` on every row. Set it to False for
    single-row unit tests that exercise one per-row rule in isolation.
    """
    from collections import Counter

    from ..config import WS6A_DIFFICULTY_MIX, WS6A_N_QUESTIONS, WS6A_N_SCALING_QUESTIONS

    rows = list(rows)
    problems: list[str] = []
    seen: set[str] = set()
    repo_files = set(repo_files)

    for i, row in enumerate(rows):
        qid_raw = row.get("qid")
        qid_is_missing = not qid_raw or (isinstance(qid_raw, str) and not qid_raw.strip())

        # Use unique placeholder for duplicate detection
        qid = qid_raw if (qid_raw and (not isinstance(qid_raw, str) or qid_raw.strip())) else f"<missing qid at row {i}>"

        # Report missing qid as a problem
        if qid_is_missing:
            problems.append(f"row {i}: qid is missing")

        # Detect duplicate qids (only if qid is present and non-empty)
        if not qid_is_missing and qid in seen:
            problems.append(f"{qid}: duplicate qid")
        seen.add(qid)

        files = split_field(row.get("expected_files", ""))
        if not files:
            problems.append(f"{qid}: expected_files must be non-empty")
        for f in files:
            if f not in repo_files:
                problems.append(f"{qid}: expected file {f!r} is not in the repo "
                                f"at the pinned commit")

        if row.get("difficulty") not in VALID_DIFFICULTIES:
            problems.append(
                f"{qid}: difficulty {row.get('difficulty')!r} not in "
                f"{VALID_DIFFICULTIES}")

        if str(row.get("scaling_subset", "")).lower() == "true":
            outside = [f for f in files if not f.startswith(scaling_prefix)]
            if outside:
                problems.append(
                    f"{qid}: scaling_subset question has expected_files outside "
                    f"{scaling_prefix} ({outside}); the S subset is "
                    f"{scaling_prefix} only, so it would be unanswerable at S")

        if check_cardinality and not split_field(row.get("expected_symbols", "")):
            problems.append(f"{qid}: expected_symbols must be non-empty")

    if check_cardinality:
        if len(rows) != WS6A_N_QUESTIONS:
            problems.append(
                f"expected exactly {WS6A_N_QUESTIONS} questions, got {len(rows)}")

        n_scaling = sum(1 for r in rows
                        if str(r.get("scaling_subset", "")).lower() == "true")
        if n_scaling != WS6A_N_SCALING_QUESTIONS:
            problems.append(
                f"expected exactly {WS6A_N_SCALING_QUESTIONS} scaling_subset "
                f"questions, got {n_scaling}")

        counts = Counter(r.get("difficulty") for r in rows)
        for difficulty, expected_n in WS6A_DIFFICULTY_MIX.items():
            got = counts.get(difficulty, 0)
            if got != expected_n:
                problems.append(
                    f"expected {expected_n} {difficulty!r} questions, got {got}")

    return problems


def assert_no_errors(df) -> None:
    """Raise if any row in `df` has a non-empty `error`.

    Callers gate `runs.csv` (or any other committed CSV) writing on this: a
    run with any errored pair -- e.g. every agentic-arm row failing because
    ripgrep went missing mid-run -- must never produce a clean-looking
    committed CSV. See C-2.
    """
    bad = [r for r in df.to_dict("records") if r.get("error")]
    if bad:
        detail = "; ".join(f"{r['qid']}/{r['arm']}: {r['error']}" for r in bad)
        raise RuntimeError(
            f"{len(bad)} run(s) errored and must not be committed: {detail}")


def assert_run_complete(df, expected_qids, arms) -> None:
    """Raise unless every (qid, arm) pair is present exactly once.

    This is the gate that assert_no_errors alone can no longer be: since C-2,
    a failed pair is skipped rather than checkpointed, so a broken arm shows up
    as MISSING ROWS, not as rows carrying an error. A run that lost all 40
    agentic pairs therefore passes assert_no_errors cleanly and writes a
    120-row runs.csv that looks like a finished experiment.

    Unequal n across arms is also the specific failure the $75 governor exists
    to prevent (spec section 8): every delta this workstream reports is
    between arms, and comparing 40 questions against 20 silently changes what
    those deltas mean.
    """
    want = {(q, a) for q in expected_qids for a in arms}
    have = [(r["qid"], r["arm"]) for r in df.to_dict("records")]
    missing = want - set(have)
    extra = set(have) - want
    dupes = {p for p in have if have.count(p) > 1}

    problems = []
    if missing:
        by_arm: dict[str, int] = {}
        for _, a in missing:
            by_arm[a] = by_arm.get(a, 0) + 1
        problems.append(f"{len(missing)} pair(s) missing {by_arm}: "
                        f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
    if extra:
        problems.append(f"{len(extra)} unexpected pair(s): {sorted(extra)[:10]}")
    if dupes:
        problems.append(f"{len(dupes)} duplicated pair(s): {sorted(dupes)[:10]}")
    if problems:
        raise RuntimeError(
            "run is incomplete and must not be committed -- re-run to resume: "
            + " | ".join(problems))


# -------------------------------------------------------------- aggregation


def summarise_runs(df):
    """Per-arm aggregates, with every accuracy expressed as lift over the
    parametric floor.

    Lift is NOT a decomposition: a retrieving arm still uses parametric memory,
    and retrieval can also correct a wrong prior. These columns bound
    contamination; they do not isolate retrieval-only performance.
    """
    import pandas as pd

    # THE PROMPT-TOKEN TRAP. `input_tokens` is the UNCACHED REMAINDER only --
    # the API reports cached content under cache_read/cache_creation instead.
    # Arm (c) sends ~960k tokens per question and reports a median
    # `input_tokens` of 28, so charting median_input_tokens would rank the
    # stuffed arm as by far the CHEAPEST. prompt_tokens is the honest
    # per-question prompt size and is what every token comparison must use.
    df = df.copy()
    df["prompt_tokens"] = (df["input_tokens"]
                           + df["cache_read_input_tokens"]
                           + df["cache_creation_input_tokens"])

    g = df.groupby("arm")
    out = pd.DataFrame({
        "n": g["qid"].count(),
        "median_prompt_tokens": g["prompt_tokens"].median(),
        "total_prompt_tokens": g["prompt_tokens"].sum(),
        "median_input_tokens": g["input_tokens"].median(),
        "total_input_tokens": g["input_tokens"].sum(),
        "median_output_tokens": g["output_tokens"].median(),
        "total_cache_write_tokens": g["cache_creation_input_tokens"].sum(),
        "total_cache_read_tokens": g["cache_read_input_tokens"].sum(),
        "median_turns": g["turns"].median(),
        "turn_cap_rate": g["hit_turn_cap"].mean(),
        "median_latency_s": g["wall_clock_s"].median(),
        "p95_latency_s": g["wall_clock_s"].quantile(0.95),
        "any_file_hit_rate": g["any_file_hit"].mean(),
        "all_files_hit_rate": g["all_files_hit"].mean(),
        "judge_accuracy": g["judge_correct"].mean(),
        # C-1: three distinct cost columns, and they are NOT interchangeable.
        #   cost_agent_billed  -- agent spend, cache rates applied
        #   cost_uncached_list -- the SAME agent spend priced with no caching
        #   cost_total_billed  -- agent + judge; this is what the governor charges
        # Only the first two are comparable. Dividing a total that includes the
        # Opus judge by an agent-only figure reports the parametric arm -- which
        # uses no cache at all -- as made ~160% more expensive BY caching.
        "total_cost_agent_billed": g["cost_agent_billed"].sum(),
        "total_cost_uncached_list": g["cost_uncached_list"].sum(),
        "total_cost_billed": g["cost_total_billed"].sum(),
        "total_cost_judge_billed": g["cost_judge_billed"].sum(),
    }).reset_index()

    floor = out.loc[out["arm"] == "parametric", "judge_accuracy"]
    base = float(floor.iloc[0]) if len(floor) else 0.0
    out["judge_lift_over_parametric"] = (out["judge_accuracy"] - base).round(6)

    pf = out.loc[out["arm"] == "parametric", "any_file_hit_rate"]
    pbase = float(pf.iloc[0]) if len(pf) else 0.0
    out["any_file_lift_over_parametric"] = (
        out["any_file_hit_rate"] - pbase).round(6)
    # Agent-only on BOTH sides. Using total_cost_billed here is the C-1 bug.
    out["cache_savings_ratio"] = (
        1 - out["total_cost_agent_billed"] / out["total_cost_uncached_list"]
    ).round(4)
    return out
