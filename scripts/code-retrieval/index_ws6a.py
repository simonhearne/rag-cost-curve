#!/usr/bin/env python
"""Index a scope of the pinned fastapi checkout with claude-context and record
the one-time cost. WS5 needs this for amortisation.

Outputs: results/ws6a/index_cost.csv

Indexing is ASYNCHRONOUS in @zilliz/claude-context-mcp@0.1.15: index_codebase
returns as soon as the background task is queued. Taking its return as
completion would record a few seconds for a job that takes many minutes and
hand WS5 an amortisation cost near zero.

get_indexing_status is NOT a usable completion signal either, and this was
established by observation, not by reading. On the first attempt it reported
"✅ fully indexed ... 1100 files, 1100 chunks" 19 seconds in, while the
server's own log showed 330/2825 files and Milvus held 1,400 chunks. The
status handler calls syncIndexedCodebasesFromCloud() first, which treats the
mere EXISTENCE of a cloud collection carrying this codebase's path as proof
the codebase is indexed -- true from the moment the first batch is inserted.
Trusting it exited the script, which closed stdio, which killed the server
mid-index and left a silently partial collection that would have answered
every arm (a) query from 21% of the repo.

The authoritative signal is the server's own log line for THIS process,
"[BACKGROUND-INDEX] ✅ Indexing completed successfully! Files: N, Chunks: M",
cross-checked against a Milvus count(*).

Run:  python scripts/code-retrieval/index_ws6a.py --scope L [--force]
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.code_retrieval import bench as ws6a
from benchlib.config import (
    DATA_DIR, WS6A_EMBED_MODEL, ws6a_checkout, ws6a_dir,
)
from benchlib.agent_harness import claude_context_session

POLL_SECONDS = 15
POLL_TIMEOUT_SECONDS = 7200

# The server's own log for THIS process -- the only trustworthy signals.
COMPLETE_RE = re.compile(
    r"Indexing completed successfully! Files: (\d+), Chunks: (\d+)")
FAILED_RE = re.compile(r"Indexing failed for .*|Indexing for .* was cancelled")
PROGRESS_RE = re.compile(r"Processed (\d+)/(\d+) files")


def collection_prefix(scope: str) -> str:
    return f"ws6a_{scope.lower()}"


def materialise_subset(root: Path, scope: str, dest: Path) -> Path:
    """Copy a scope's files into their own tree.

    claude-context indexes a directory, and the agentic arm greps one, so a
    subset has to exist on disk as a real tree or S and M would both silently
    be L.
    """
    paths = ws6a.subset_files(ws6a.filter_text_files(ws6a.git_ls_files(root)), scope)
    for p in paths:
        target = dest / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / p, target)
    return dest


def _text(result) -> str:
    return "\n".join(getattr(b, "text", "") for b in result.content)


async def run(scope: str, tree: Path, force: bool) -> dict:
    prefix = collection_prefix(scope)
    log = DATA_DIR / "ws6a_cache" / f"mcp_index_{prefix}.stderr.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    async with claude_context_session(tree, prefix, stderr_log=log) as (
            session, tools, raw_tools):
        tool_names = [t.name for t in raw_tools]
        print("claude-context tools:", tool_names)

        started = time.time()
        result = await session.call_tool(
            "index_codebase",
            {"path": str(tree), "splitter": "ast", "force": force})
        first = _text(result)
        print("index_codebase ->", first[:400])
        if getattr(result, "isError", False):
            raise SystemExit(f"index_codebase failed: {first}")

        reported_files = reported_chunks = -1
        while True:
            elapsed = time.time() - started
            if elapsed > POLL_TIMEOUT_SECONDS:
                raise SystemExit(f"indexing still not complete after {elapsed:.0f}s")
            await asyncio.sleep(POLL_SECONDS)

            text = log.read_text(encoding="utf-8", errors="replace")
            done = COMPLETE_RE.search(text)
            if done:
                reported_files = int(done.group(1))
                reported_chunks = int(done.group(2))
                print(f"  [{elapsed:6.0f}s] COMPLETE: files={reported_files} "
                      f"chunks={reported_chunks}")
                break
            bad = FAILED_RE.search(text)
            if bad:
                raise SystemExit(f"indexing failed/cancelled: {bad.group(0)}")

            seen = PROGRESS_RE.findall(text)
            if seen:
                cur, total = seen[-1]
                print(f"  [{elapsed:6.0f}s] {cur}/{total} files "
                      f"({100 * int(cur) / max(int(total), 1):.1f}%)", flush=True)
            else:
                print(f"  [{elapsed:6.0f}s] starting...", flush=True)

        elapsed = time.time() - started
        status_text = _text(await session.call_tool("get_indexing_status",
                                                    {"path": str(tree)}))

    return {
        "scope": scope,
        "wall_clock_s": round(elapsed, 1),
        "tools": "|".join(tool_names),
        "indexed_files_reported": reported_files,
        "chunks_reported": reported_chunks,
        "status_text": status_text.replace("\n", " | "),
    }


def chunk_stats(prefix: str) -> dict:
    """Chunk count read from Milvus directly, as a cross-check on the server's
    own reported figure."""
    from pymilvus import MilvusClient

    client = MilvusClient(uri=os.environ["MILVUS_ADDRESS"],
                          token=os.environ["MILVUS_TOKEN"])
    names = [c for c in client.list_collections() if prefix in c]
    if not names:
        print(f"WARNING: no collection matching {prefix!r}; "
              f"saw {client.list_collections()}")
        return {"collection": "", "chunk_count": 0}
    name = sorted(names)[0]
    # count(*), NOT get_collection_stats: stats are computed from sealed
    # segments and lag recent inserts, reporting row_count 0 for a
    # freshly-indexed collection (observed: stats said 0 while count(*) said
    # 1,400). claude-context's own source carries the same warning.
    client.load_collection(name)
    rows = client.query(name, filter="", output_fields=["count(*)"])
    return {"collection": name, "chunk_count": int(rows[0]["count(*)"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="L", choices=["S", "M", "L"])
    ap.add_argument("--force", action="store_true",
                    help="clear any existing index for this scope and rebuild "
                         "-- required after a partial/killed index, and it "
                         "re-spends the embedding cost")
    args = ap.parse_args()
    scope = args.scope

    root = ws6a_checkout()
    if scope == "L":
        tree = root
    else:
        tree = DATA_DIR / f"ws6a_subset_{scope.lower()}"
        if not tree.exists():
            print(f"materialising {scope} subset at {tree}")
            materialise_subset(root, scope, tree)
        else:
            print(f"reusing {scope} subset at {tree}")

    info = asyncio.run(run(scope, tree, args.force))
    info["tree"] = str(tree)
    info.update(chunk_stats(collection_prefix(scope)))

    stats = pd.read_csv(ws6a_dir() / "repo_stats.csv")
    scope_row = stats[stats["scope"] == scope].iloc[0]
    # OpenAI bills embeddings on ITS OWN tokenizer, so the embedding cost is
    # priced over the tiktoken/cl100k figure, not the Claude count_tokens one.
    # count_tokens runs ~26% higher on this corpus; using it here would
    # overstate the one-time index cost that WS5 amortises by that much, in
    # the direction that flatters the live-search side of C9.
    embed_tokens = int(scope_row["tiktoken_cl100k"])
    claude_tokens = int(scope_row["count_tokens"])
    prices = ws6a.load_pricing(ws6a_dir() / "pricing.csv")
    rate = prices[WS6A_EMBED_MODEL]["input_usd_per_mtok"]

    info.update({
        "embed_model": WS6A_EMBED_MODEL,
        "corpus_tokens_tiktoken": embed_tokens,
        "corpus_tokens_claude": claude_tokens,
        "corpus_files_allowlisted": int(scope_row["files"]),
        "embed_cost_usd": round(embed_tokens / 1_000_000 * rate, 6),
        # NOT a usage field: @zilliz/claude-context-mcp@0.1.15 does not surface
        # embedding usage through MCP, so this is a derived figure over a
        # measured token count and a published rate. It is the only basis
        # available and is labelled as derived everywhere it is reported.
        "embed_cost_basis": "derived: tiktoken_cl100k x list rate; "
                            "claude-context exposes no embedding usage field",
        # claude-context indexes its own 25 allowlisted extensions minus 51
        # ignore patterns, which is NOT our text-file allowlist. Recording both
        # so the difference is visible rather than assumed away.
        "mean_chunk_tokens": (round(embed_tokens / info["chunk_count"], 1)
                              if info["chunk_count"] else 0),
    })

    path = ws6a_dir() / "index_cost.csv"
    rows = ([pd.read_csv(path)] if path.exists() else []) + [pd.DataFrame([info])]
    out = pd.concat(rows, ignore_index=True).drop_duplicates(
        subset=["scope"], keep="last").sort_values("scope")
    out.to_csv(path, index=False)
    manifest.record(path)
    print(out.drop(columns=["status_text"]).to_string(index=False))


if __name__ == "__main__":
    main()
