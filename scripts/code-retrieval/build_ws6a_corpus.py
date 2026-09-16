#!/usr/bin/env python
"""Clone fastapi at the pinned commit, measure every scope, materialise the
arm (c) prefix.

Outputs:
  results/ws6a/repo_stats.csv        one row per scope (L / S / M)
  data/ws6a_prefix_manifest.json     the ordered prefix file list + fit fraction

Token accounting (amended 2026-08-18, issues I-5 and I-6):

  I-6  messages.count_tokens prices a whole request, not a bare string. The
       per-request framing constant is measured once and subtracted from every
       batch, so a corpus counted file-by-file and the same corpus counted in
       one call agree instead of differing by ~2,900 x F.
  I-5  the string arm (c) actually sends carries a "===== path =====" header
       per file and a blank line between files. Selection and reporting are
       therefore done over prefix UNITS (header + body + separator), not bare
       bodies, so the published fit_fraction describes the string in the
       context window rather than a different one.

  repo_stats.csv keeps `count_tokens` as the bare-body corpus total (the
  figure WS5 amortises embedding cost over, and the scaling curve's x-axis)
  and adds `count_tokens_with_headers` as the stuffed-format total. Both are
  framing-corrected, so they are two honest measurements of two different
  strings rather than two disagreeing measurements of one.

Run:  python scripts/code-retrieval/build_ws6a_corpus.py
"""

import json
import os
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import pandas as pd
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.code_retrieval import bench as ws6a
from benchlib.config import (
    DATA_DIR, WS6A_AGENT_MODEL, WS6A_COMMIT, WS6A_PREFIX_BUDGET_TOKENS, WS6A_REPO_URL,
    WS6A_SUBSETS, ws6a_checkout, ws6a_dir,
)
from benchlib.agent_harness import (
    count_tokens_batched, measure_request_overhead, prefix_unit_text,
)

# Bumped whenever the per-file counting BASIS changes, so a cache written under
# the old basis is discarded rather than silently mixed with the new one. A
# stale-basis cache would produce plausible numbers for the wrong string --
# the worst failure mode here, because it is invisible.
CACHE_BASIS = "prefix_unit_framing_corrected_v1"

# Concurrency for the per-file count only. count_tokens is unbilled, so this
# buys wall-clock, not money.
COUNT_WORKERS = 4

# The organisation's count_tokens limit is 100 requests/minute. Retrying into a
# sustained 429 burns the backoff schedule and then raises, so the request rate
# is capped BELOW the limit rather than discovered by colliding with it.
COUNT_REQUESTS_PER_MINUTE = 90


class _RateLimiter:
    """Minimum interval between request starts, shared across worker threads."""

    def __init__(self, per_minute: int):
        self._interval = 60.0 / per_minute
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                time.sleep(wait)
                now = self._next
            self._next = now + self._interval


_limiter = _RateLimiter(COUNT_REQUESTS_PER_MINUTE)


def load_cache(commit: str) -> dict:
    """Load cached per-file token counts, invalidating if commit or basis differs.

    Corrupt caches (truncated JSON, read errors) are treated as empty rather
    than fatal: they cost a recount but do not crash.
    """
    cache_dir = DATA_DIR / "ws6a_cache"
    cache_file = cache_dir / "per_file_tokens.json"
    if not cache_file.exists():
        return {}
    try:
        data = json.loads(cache_file.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"warning: corrupt cache {cache_file}; ignoring: {e}")
        return {}
    if data.get("commit") != commit:
        print(f"cache commit {data.get('commit')} != {commit}; recount all")
        return {}
    if data.get("basis") != CACHE_BASIS:
        print(f"cache basis {data.get('basis')!r} != {CACHE_BASIS!r}; recount all")
        return {}
    return data.get("tokens", {})


def save_cache(commit: str, tokens: dict, overhead: int) -> None:
    """Write per-file cache atomically with commit SHA and counting basis.

    Writes to a temp file in the same directory, then os.replace()s it over
    the target. os.replace is atomic on POSIX, so a reader never observes a
    partial file. Temp file stays in cache_dir (same filesystem) for atomicity.
    """
    cache_dir = DATA_DIR / "ws6a_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "per_file_tokens.json"
    temp_file = cache_dir / f".per_file_tokens.tmp.{os.getpid()}"
    temp_file.write_text(json.dumps({
        "commit": commit,
        "basis": CACHE_BASIS,
        "request_overhead_tokens": overhead,
        "tokens": tokens,
    }, indent=2))
    os.replace(str(temp_file), str(cache_file))


def count_with_retry(client, model: str, texts: list, overhead: int = 0,
                     max_attempts: int = 6) -> int:
    """Count tokens with exponential backoff retry (2s / 4s / 8s / 16s / 32s).

    Six total attempts: five waits, then raise. Rate-limited before each
    attempt so retries guard genuine transients rather than a rate ceiling the
    caller is deliberately exceeding.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            _limiter.acquire()
            return count_tokens_batched(client, model, texts, overhead=overhead)
        except Exception as e:
            if attempt == max_attempts:
                raise
            wait = 2 ** attempt
            print(f"  retry {attempt} failed: {e}; waiting {wait}s")
            time.sleep(wait)


def ensure_checkout() -> Path:
    root = ws6a_checkout()
    if not (root / ".git").exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", WS6A_REPO_URL, str(root)])
    subprocess.check_call(["git", "-C", str(root), "fetch", "--all", "--tags"])
    subprocess.check_call(["git", "-C", str(root), "checkout", "--quiet", WS6A_COMMIT])
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != WS6A_COMMIT:
        raise SystemExit(f"checkout is at {head}, expected pinned {WS6A_COMMIT}")
    print(f"checkout verified at {head}")
    return root


def main() -> None:
    root = ensure_checkout()
    client = anthropic.Anthropic()
    enc = tiktoken.get_encoding("cl100k_base")

    overhead = measure_request_overhead(client, WS6A_AGENT_MODEL)
    print(f"count_tokens request framing overhead: {overhead} tokens/request "
          f"(subtracted from every figure below)")

    all_paths = ws6a.filter_text_files(ws6a.git_ls_files(root))
    body = {p: (root / p).read_text(encoding="utf-8", errors="replace")
            for p in all_paths}
    unit = {p: prefix_unit_text(root, p) for p in all_paths}

    rows = []
    for scope in ("L", "S", "M"):
        paths = ws6a.subset_files(all_paths, scope)
        ordered = ws6a.tiered_order(paths)
        bodies = [body[p] for p in ordered]
        units = [unit[p] for p in ordered]
        ct = count_tokens_batched(client, WS6A_AGENT_MODEL, bodies, overhead=overhead)
        ct_h = count_tokens_batched(client, WS6A_AGENT_MODEL, units, overhead=overhead)
        tk = sum(len(enc.encode(b)) for b in bodies)
        rows.append({
            "scope": scope,
            "scope_prefixes": "|".join(WS6A_SUBSETS[scope]) or "(whole repo)",
            "files": len(ordered),
            "chars": sum(len(b) for b in bodies),
            "lines": sum(b.count("\n") for b in bodies),
            "count_tokens": ct,
            "count_tokens_with_headers": ct_h,
            "tiktoken_cl100k": tk,
            "tiktoken_undercount_pct": round(100 * (ct - tk) / ct, 2) if ct else 0.0,
            "request_overhead_tokens": overhead,
        })
        print(f"{scope}: files={len(ordered)} count_tokens={ct:,} "
              f"with_headers={ct_h:,} tiktoken={tk:,}")

    out = ws6a_dir()
    out.mkdir(parents=True, exist_ok=True)
    stats_path = out / "repo_stats.csv"
    pd.DataFrame(rows).to_csv(stats_path, index=False)

    # Per-file counts drive prefix SELECTION only -- they never reach the cost
    # model. They are counted over prefix units (I-5) and framing-corrected
    # (I-6) so the greedy boundary is set against the real string.
    # Cached and resumable, with progress and retry.
    per_file = load_cache(WS6A_COMMIT)
    cached_count = len(per_file)
    remaining = [p for p in all_paths if p not in per_file]
    print(f"\nper-file tokens: {cached_count} cached, {len(remaining)} to count")

    # Counted concurrently: ~2,900 independent unbilled requests, serially
    # ~50/min. Concurrency changes the wall-clock only -- each file is still
    # its own request, so the per-file figure is identical either way. The
    # cache dict is written only from the main thread as futures complete, so
    # save_cache still sees a consistent snapshot.
    if remaining:
        with ThreadPoolExecutor(max_workers=COUNT_WORKERS) as pool:
            futures = {
                pool.submit(count_with_retry, client, WS6A_AGENT_MODEL,
                            [unit[p]], overhead): p
                for p in remaining
            }
            for i, fut in enumerate(as_completed(futures), start=1):
                path = futures[fut]
                per_file[path] = fut.result()
                if i % 100 == 0 or i == len(remaining):
                    print(f"  {cached_count + i} / {len(all_paths)} files counted",
                          flush=True)
                    save_cache(WS6A_COMMIT, per_file, overhead)
    save_cache(WS6A_COMMIT, per_file, overhead)

    res = ws6a.build_prefix(all_paths, per_file, WS6A_PREFIX_BUDGET_TOKENS)

    # Authoritative: one batched count of exactly the units that will be sent.
    # Over-counts the real string by one trailing separator (~1 token).
    prefix_tokens = count_tokens_batched(
        client, WS6A_AGENT_MODEL, [unit[p] for p in res.included],
        overhead=overhead)

    l_row = next(r for r in rows if r["scope"] == "L")
    repo_with_headers = l_row["count_tokens_with_headers"]
    fit_fraction = prefix_tokens / repo_with_headers if repo_with_headers else 0.0

    # Sum-of-per-file vs one-call over the same units: with every unit ending
    # in a blank line there is no token merge at a batch seam, so these should
    # agree to within a handful of tokens. A large gap means the framing
    # correction is wrong and every derived figure is suspect.
    drift = res.total_tokens - repo_with_headers
    print(f"\nsum-of-per-file vs single-call over all units: "
          f"{res.total_tokens:,} vs {repo_with_headers:,} "
          f"(drift {drift:+,} tokens, {100 * drift / repo_with_headers:+.3f}%)")

    prefix_path = DATA_DIR / "ws6a_prefix_manifest.json"
    prefix_path.write_text(json.dumps({
        "commit": WS6A_COMMIT,
        "budget_tokens": WS6A_PREFIX_BUDGET_TOKENS,
        "counting_basis": CACHE_BASIS,
        "request_overhead_tokens": overhead,
        "files": res.included,
        "n_files": len(res.included),
        # Selection-time figures: sums of per-file unit counts.
        "tokens_sum_per_file": res.tokens,
        "total_tokens_sum_per_file": res.total_tokens,
        # Authoritative figures: batched counts of the concatenated units.
        "tokens_concatenated": prefix_tokens,
        "repo_total_tokens_with_headers": repo_with_headers,
        "repo_total_tokens_bodies": l_row["count_tokens"],
        "fit_fraction": fit_fraction,
        "fit_fraction_basis":
            "tokens_concatenated / repo_total_tokens_with_headers -- both are "
            "the stuffed format (===== path ===== headers, blank-line "
            "separated) so the ratio compares like with like",
    }, indent=2))

    manifest.record(stats_path)
    manifest.record(prefix_path)
    print(f"\nprefix: {len(res.included)} files, {prefix_tokens:,} tokens, "
          f"fit_fraction={fit_fraction:.4f}")


if __name__ == "__main__":
    main()
