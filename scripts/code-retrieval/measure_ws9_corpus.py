#!/usr/bin/env python
"""Measure each WS9 scope of the pinned WS6c corpus.

Outputs: results/ws6c/scaling_corpus_stats.csv

The x-axis of the whole cost curve is measured `count_tokens` on the pinned
agent model, not tiktoken and not file count. tiktoken undercounts Claude by
26-39% on code (WS6a) and 35.7% on this corpus, so a curve drawn against it
would be drawn against the wrong axis; file counts step 3.5x then 2.0x here
against the token axis's 1.8x then 2.1x, so a file-count axis would give the
curve a shape set by how the subsets happen to divide.

`messages.count_tokens` prices a WHOLE REQUEST, so a per-request framing
constant rides along on every call. It is measured once (three unbilled calls)
and subtracted per batch, exactly as scripts/code-retrieval/build_ws6a_corpus.py
does -- without it the per-scope totals and a single batched call disagree.

Cached to data/ws9_cache/corpus_tokens.json: re-running is free.

Run:  .venv/bin/python scripts/code-retrieval/measure_ws9_corpus.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.agent_harness import (
    count_tokens_batched, make_client, measure_request_overhead,
)
from benchlib.code_retrieval import bench as ws6a
from benchlib.code_retrieval import scaling as ws9
from benchlib.code_retrieval import topk as ws6c
from benchlib.config import (
    WS6A_AGENT_MODEL, WS6A_PREFIX_BUDGET_TOKENS, WS6C_COMMITS, WS9_SCOPES,
    WS9_SLUG, ws6c_checkout, ws6c_dir, ws9_cache_dir,
)

# Bumped if the counting BASIS changes, so a stale-basis cache is discarded
# rather than silently mixed with the new one.
CACHE_BASIS = "ws9_scope_framing_corrected_v1"
CACHE = "corpus_tokens.json"

# The L row must reproduce these, which corpus_stats.csv already pins for the
# whole repository. If it does not, this pipeline and WS6c's disagree about
# what the corpus IS, and nothing downstream is comparable to the published
# WS6c result.
WS6C_L_EXPECTED = {"files_text_allowlisted": 342, "count_tokens": 4_307_363,
                   "tiktoken_cl100k": 2_770_789}


def load_cache() -> dict:
    path = ws9_cache_dir() / CACHE
    if not path.exists():
        return {}
    blob = json.loads(path.read_text())
    if blob.get("basis") != CACHE_BASIS or blob.get("commit") != WS6C_COMMITS[WS9_SLUG]:
        print("cache basis/commit changed -- discarding")
        return {}
    return blob


def save_cache(scopes: dict, overhead: int) -> None:
    (ws9_cache_dir() / CACHE).write_text(json.dumps({
        "basis": CACHE_BASIS, "commit": WS6C_COMMITS[WS9_SLUG],
        "request_overhead_tokens": overhead, "scopes": scopes,
    }, indent=2, sort_keys=True))


def main() -> None:
    # BEFORE any measurement: a dirty tree measured is a corpus that is not the
    # corpus pinned, silently.
    print("checking the pin before measuring")
    ws6c.assert_pinned_checkouts([WS9_SLUG])

    checkout = ws6c_checkout(WS9_SLUG)
    tree = ws6c.index_tree(WS9_SLUG)
    if not tree.exists():
        raise SystemExit(
            f"{tree} does not exist -- run scripts/code-retrieval/index_ws6c.py "
            "first, or materialise the derived tree; WS9 measures the DERIVED "
            "tree, not the checkout")

    tracked = ws6a.git_ls_files(checkout)
    dropped = set(ws6c.dropped_ignore_files(tracked))
    in_tree = [p for p in tracked if p not in dropped]
    print(f"{len(tracked)} tracked, {len(in_tree)} in the derived tree "
          f"(dropped {sorted(dropped) or '(nothing)'})")

    if not ws9.ws9_scopes_are_nested(in_tree):
        raise SystemExit("WS9 scopes are NOT nested -- the three size points "
                         "would not sit on one corpus; fix WS9_SUBSETS")

    enc = tiktoken.get_encoding("cl100k_base")
    cache = load_cache()
    cached_scopes = cache.get("scopes", {})
    overhead = cache.get("request_overhead_tokens")

    client = None
    if any(s not in cached_scopes for s in WS9_SCOPES):
        client = make_client()
        if overhead is None:
            overhead = measure_request_overhead(client, WS6A_AGENT_MODEL)
            print(f"count_tokens request framing overhead: {overhead} "
                  "tokens/request (measured, subtracted per batch)")

    rows = []
    for scope in WS9_SCOPES:
        scope_tracked = ws9.ws9_subset_files(tracked, scope)
        scope_in_tree = ws9.ws9_subset_files(in_tree, scope)
        text = ws6a.filter_text_files(scope_in_tree)
        indexable = ws6c.claude_context_indexable(scope_in_tree)

        bodies = [(tree / p).read_text(errors="ignore") for p in text]
        lines = sum(b.count("\n") for b in bodies)
        chars = sum(len(b) for b in bodies)
        tik = sum(len(enc.encode(b)) for b in bodies)
        tik_idx = sum(len(enc.encode((tree / p).read_text(errors="ignore")))
                      for p in indexable)

        if scope in cached_scopes:
            ct = int(cached_scopes[scope])
            print(f"[{scope}] count_tokens {ct:,} (cached)")
        else:
            ct = count_tokens_batched(client, WS6A_AGENT_MODEL, bodies,
                                      overhead=overhead)
            cached_scopes[scope] = ct
            save_cache(cached_scopes, overhead)
            print(f"[{scope}] count_tokens {ct:,} (measured)")

        remeasured = ct
        if scope == "L":
            # The L row IS WS6c's published measurement. Tonight's count is a
            # cross-check (GATE 2 below), not a replacement: the x-axis the
            # curve is drawn against must be the one the published L point
            # already sits on.
            ct = WS6C_L_EXPECTED["count_tokens"]

        rows.append({
            "scope": scope, "corpus": WS9_SLUG,
            "commit": WS6C_COMMITS[WS9_SLUG], "tree": str(tree),
            "files_tracked": len(scope_tracked),
            "files_in_tree": len(scope_in_tree),
            "files_text_allowlisted": len(text),
            "files_claude_context_indexable": len(indexable),
            "lines": lines, "chars": chars,
            "count_tokens": ct, "count_tokens_remeasured": remeasured,
            "tiktoken_cl100k": tik,
            "indexable_tiktoken_cl100k": tik_idx,
            "request_overhead_tokens": overhead,
            "fits_in_prefix_budget": ct <= WS6A_PREFIX_BUDGET_TOKENS,
        })

    df = pd.DataFrame(rows).set_index("scope").loc[list(WS9_SCOPES)].reset_index()

    # GATE 1: the x-axis must be strictly monotonic in TOKENS.
    ax = df.set_index("scope")["count_tokens"]
    if not (ax["S"] < ax["M"] < ax["L"]):
        raise SystemExit(f"count_tokens must increase S<M<L, got {ax.to_dict()}")

    # GATE 2: the L row must reproduce what results/ws6c/corpus_stats.csv
    # already pins for the whole repository. Files and tiktoken are computed
    # locally and must match EXACTLY. count_tokens is an API measurement taken
    # in 2M-char batches, and count_tokens_batched documents boundary effects
    # "on the order of one token per batch"; the L bodies are ~11.6M chars, so
    # six batches, and a drift of a few tokens is the instrument, not a
    # different corpus. The L row carries the pinned value either way and the
    # remeasurement sits beside it. A drift beyond the tolerance means a
    # different file set or a different tokenizer, and stops the run.
    COUNT_TOKENS_TOLERANCE = 12
    l = df[df.scope == "L"].iloc[0]
    for key in ("files_text_allowlisted", "tiktoken_cl100k"):
        if int(l[key]) != WS6C_L_EXPECTED[key]:
            raise SystemExit(
                f"L {key} = {int(l[key]):,}, but results/ws6c/corpus_stats.csv "
                f"pins {WS6C_L_EXPECTED[key]:,}. This pipeline and WS6c's "
                "disagree about what the corpus is; nothing downstream would "
                "be comparable to the published WS6c result.")
    drift = int(l["count_tokens_remeasured"]) - WS6C_L_EXPECTED["count_tokens"]
    if abs(drift) > COUNT_TOKENS_TOLERANCE:
        raise SystemExit(
            f"L count_tokens remeasured at {int(l['count_tokens_remeasured']):,}, "
            f"{drift:+d} tokens from the pinned {WS6C_L_EXPECTED['count_tokens']:,} "
            f"(tolerance +/-{COUNT_TOKENS_TOLERANCE}). That is more than batch "
            "boundary noise: the file set or the tokenizer differs from what "
            "WS6c measured. Stop.")
    print(f"L reproduces corpus_stats.csv: files and tiktoken exact; count_tokens "
          f"remeasured within {drift:+d} tokens of the pinned "
          f"{WS6C_L_EXPECTED['count_tokens']:,} (tolerance +/-"
          f"{COUNT_TOKENS_TOLERANCE}); the L row carries the pinned value")

    path = ws6c_dir() / "scaling_corpus_stats.csv"
    df.to_csv(path, index=False)
    manifest.record(path)
    print(df[["scope", "files_text_allowlisted", "count_tokens",
              "tiktoken_cl100k", "indexable_tiktoken_cl100k"]].to_string(index=False))
    span = ax["L"] / ax["S"]
    print(f"\nspan S->L {span:.2f}x  (S->M {ax['M']/ax['S']:.2f}x, "
          f"M->L {ax['L']/ax['M']:.2f}x)")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
