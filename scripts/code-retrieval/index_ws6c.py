#!/usr/bin/env python
"""Phase 3a: index the selected uncontaminated corpus with claude-context and
record the one-time cost.

Outputs: results/ws6c/index_cost.csv

Modelled directly on scripts/code-retrieval/index_ws6a.py, with --slug in place of --scope
and no subset materialisation (R7: the whole corpus is the comparison cell).

What is indexed is the DERIVED tree, not the checkout: the pinned package
indexes zero files of agentic_hil as-is, because it merges the repo's
`**`-style .dockerignore into its own ignore set. See benchlib.code_retrieval.topk's
"derived corpus tree" note for the reproduction and for why dropping that one
file leaves the measured corpus byte-identical.

WHY THE VERIFICATION IS NOT OPTIONAL. Indexing is ASYNCHRONOUS in
@zilliz/claude-context-mcp@0.1.15 and neither of the two obvious completion
signals is trustworthy -- both were caught lying during WS6a:

  * get_indexing_status reported "fully indexed ... 1100 files, 1100 chunks"
    19 seconds in, while the server's own log showed 330/2825. Its handler
    calls syncIndexedCodebasesFromCloud() first, which treats the mere
    EXISTENCE of a cloud collection for this path as proof of completion --
    true from the moment the first batch lands.
  * get_collection_stats returned row_count 0 for a collection that already
    held 1,400 rows: stats come from sealed segments and lag recent inserts.

A partially-indexed collection answers every query perfectly happily, so
trusting either would have benchmarked 21% of a repo and reported it as the
whole thing. Three independent sources must agree here before any run spends:

  1. the server's own stderr log for THIS process -- both the completion line
     and the "Processed N/T files" denominator, so a run that stopped early
     cannot look finished;
  2. an independent Milvus count(*) after an explicit flush;
  3. the file count WE handed it, computed from the pinned checkout with the
     package's own extension/ignore rules (benchlib.code_retrieval.topk.claude_context_
     indexable, calibrated to reproduce WS6a's 2,825 exactly).

Run:  python scripts/code-retrieval/index_ws6c.py --slug agentic_hil [--force]
      python scripts/code-retrieval/index_ws6c.py --slug agentic_hil --verify-only
"""

import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.code_retrieval import bench as ws6a
from benchlib.code_retrieval import topk as ws6c
from benchlib.config import (
    DATA_DIR, WS6A_EMBED_MODEL, WS6C_CANDIDATES, ws6a_dir, ws6c_checkout,
    ws6c_dir,
)
from benchlib.agent_harness import claude_context_session
# Shared with scripts/code-retrieval/gate_ws6c.py -- ONE implementation of the pin guard.
from benchlib.code_retrieval.topk import assert_pinned_checkouts

POLL_SECONDS = 15
POLL_TIMEOUT_SECONDS = 7200

# The server's own log for THIS process -- the only trustworthy signals.
COMPLETE_RE = re.compile(
    r"Indexing completed successfully! Files: (\d+), Chunks: (\d+)")
FAILED_RE = re.compile(r"Indexing failed for .*|Indexing for .* was cancelled")
PROGRESS_RE = re.compile(r"Processed (\d+)/(\d+) files")

SLUGS = [c["slug"] for c in WS6C_CANDIDATES]


def collection_prefix(slug: str) -> str:
    return f"ws6c_{slug}"


def stderr_log_path(slug: str) -> Path:
    return DATA_DIR / "ws6c_cache" / f"mcp_index_{collection_prefix(slug)}.stderr.log"


def _text(result) -> str:
    return "\n".join(getattr(b, "text", "") for b in result.content)


async def run_index(tree: Path, prefix: str, log: Path, force: bool) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        # Source 1 must describe THIS index and nothing else. Rotated rather
        # than truncated: the log of the zero-file attempt is the evidence
        # for the .dockerignore finding.
        rotated = log.with_suffix(f".{int(time.time())}.log")
        log.rename(rotated)
        print(f"rotated previous server log to {rotated.name}")

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

        while True:
            elapsed = time.time() - started
            if elapsed > POLL_TIMEOUT_SECONDS:
                raise SystemExit(f"indexing still not complete after {elapsed:.0f}s")
            await asyncio.sleep(POLL_SECONDS)

            text = log.read_text(encoding="utf-8", errors="replace")
            if COMPLETE_RE.search(text):
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

    return {"wall_clock_s": round(elapsed, 1), "tools": "|".join(tool_names),
            "status_text": status_text.replace("\n", " | ")}


def read_server_log(log: Path) -> dict:
    """SOURCE 1: the server's own completion line and progress denominator."""
    if not log.exists():
        raise SystemExit(f"no server log at {log}; run without --verify-only first")
    text = log.read_text(encoding="utf-8", errors="replace")
    done = COMPLETE_RE.findall(text)
    if not done:
        raise SystemExit(
            f"{log} carries no 'Indexing completed successfully!' line -- the "
            "index is unfinished or was killed mid-run")
    files, chunks = (int(x) for x in done[-1])
    progress = PROGRESS_RE.findall(text)
    # The denominator the server itself enumerated. Comparing it to the
    # completion line is what makes "330 of 2825" impossible to mistake for
    # a finished index.
    total = int(progress[-1][1]) if progress else -1
    return {"server_files": files, "server_chunks": chunks,
            "server_enumerated": total}


def milvus_rows(prefix: str) -> dict:
    """SOURCE 2: an independent count(*), after an explicit flush.

    NOT get_collection_stats: stats are computed from sealed segments and lag
    recent inserts (observed reporting row_count 0 against a live 1,400).
    claude-context's own source carries the same warning.
    """
    from pymilvus import MilvusClient

    client = MilvusClient(uri=os.environ["MILVUS_ADDRESS"],
                          token=os.environ["MILVUS_TOKEN"])
    names = [c for c in client.list_collections() if prefix in c]
    if not names:
        raise SystemExit(f"no collection matching {prefix!r}; "
                         f"saw {client.list_collections()}")
    if len(names) > 1:
        # The collection name carries a hash of the INDEXED PATH, so a
        # retargeted index leaves the old one behind. Picking one at random
        # (sorted()[0] is random with respect to which is current) would let
        # the cross-check pass against a collection nothing queries.
        raise SystemExit(
            f"{len(names)} collections match {prefix!r}: {sorted(names)}. "
            "Drop the stale one before verifying -- the name hashes the "
            "indexed path, so only one of these is the tree being measured.")
    name = names[0]
    client.flush(name)
    client.load_collection(name)
    rows = client.query(name, filter="", output_fields=["count(*)"])
    return {"collection": name, "chunk_count": int(rows[0]["count(*)"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, choices=SLUGS)
    ap.add_argument("--force", action="store_true",
                    help="clear any existing index and rebuild -- required "
                         "after a partial/killed index, and it re-spends the "
                         "embedding cost")
    ap.add_argument("--embedding-batch-size", type=int, default=None,
                    help="EMBEDDING_BATCH_SIZE for the server (package "
                         "default 100). Lowering it is a THROUGHPUT throttle, "
                         "not a content change: it alters how many chunks per "
                         "embedding request, never the chunks or the vectors. "
                         "Needed here because this corpus embeds ~2.9M tokens "
                         "and the account's text-embedding-3-small limit is "
                         "1M TPM -- at the package default the index 429s at "
                         "~60% and the package has no retry. Recorded in "
                         "index_cost.csv because it moves wall_clock_s, which "
                         "WS5 amortises.")
    ap.add_argument("--verify-only", action="store_true",
                    help="skip indexing; re-run the three-source cross-check "
                         "against the existing collection and log")
    args = ap.parse_args()
    slug = args.slug
    prefix = collection_prefix(slug)
    checkout = ws6c_checkout(slug)

    # BEFORE indexing, not only before the run: indexing a dirty tree means
    # the corpus indexed is not the corpus measured and pinned, silently.
    print("checking the pin before any embedding spend")
    assert_pinned_checkouts([slug])

    # The tree BOTH tool-using arms see, derived from the pinned checkout and
    # sha256-verified against it file by file. See benchlib.code_retrieval.topk for why the
    # checkout itself cannot be indexed: its `**`-style .dockerignore makes
    # the pinned package index zero files and call that success.
    tracked = ws6a.git_ls_files(checkout)
    derived = ws6c.materialise_index_tree(
        checkout, tracked, ws6c.index_tree(slug))
    tree = Path(derived["tree"])
    print(f"derived tree verified: {derived['files_copied']} files, dropped "
          f"{derived['dropped_ignore_files'] or '(nothing)'}")

    # SOURCE 3: what we handed it.
    corpus_paths = [p for p in tracked
                    if p not in set(derived["dropped_ignore_files"].split("|"))]
    text_files = ws6a.filter_text_files(corpus_paths)
    indexable = ws6c.claude_context_indexable(corpus_paths)
    print(f"corpus: {len(tracked)} tracked files, {len(text_files)} in OUR "
          f"text allowlist, {len(indexable)} claude-context will index")

    if args.embedding_batch_size:
        # mcp_env() copies os.environ, so setting it here reaches the server.
        os.environ["EMBEDDING_BATCH_SIZE"] = str(args.embedding_batch_size)
        print(f"EMBEDDING_BATCH_SIZE={args.embedding_batch_size} (throttle)")

    log = stderr_log_path(slug)
    path = ws6c_dir() / "index_cost.csv"
    if args.verify_only:
        # Carry the INDEX-TIME measurements forward from the recorded row.
        # Re-verifying must never blank wall_clock_s -- WS5 amortises it, and
        # a NaN there is a silently destroyed measurement, not a missing one.
        if not path.exists():
            raise SystemExit(f"--verify-only needs an existing {path}")
        prior = pd.read_csv(path).set_index("slug")
        if slug not in prior.index:
            raise SystemExit(f"{path} has no row for {slug!r} to verify")
        row0 = prior.loc[slug]
        info = {k: row0[k] for k in
                ("wall_clock_s", "tools", "status_text", "embedding_batch_size")
                if k in prior.columns}
        print(f"--verify-only: carrying forward wall_clock_s="
              f"{info.get('wall_clock_s')}")
    else:
        info = asyncio.run(run_index(tree, prefix, log, args.force))
        info["embedding_batch_size"] = args.embedding_batch_size or 100

    info.update(read_server_log(log))
    info.update(milvus_rows(prefix))

    # ------------------------------------------------- the three-way check
    server_files = info["server_files"]
    server_chunks = info["server_chunks"]
    enumerated = info["server_enumerated"]
    rows = info["chunk_count"]

    assert enumerated == server_files, (
        f"server completed at {server_files} files but enumerated "
        f"{enumerated} -- this is the 330-of-2825 partial index")
    assert rows == server_chunks, (
        f"Milvus holds {rows} rows, server claims {server_chunks} chunks")
    assert server_files == len(indexable), (
        f"server indexed {server_files} files, we handed it {len(indexable)} "
        f"indexable of {len(tracked)} tracked")
    print(f"INDEX VERIFIED: server log {server_files}/{enumerated} files, "
          f"{server_chunks} chunks; Milvus count(*) {rows}; "
          f"predicted indexable {len(indexable)} -- all three agree")

    # OpenAI bills embeddings on ITS OWN tokenizer, so the embedding cost is
    # priced over the tiktoken/cl100k figure, not the Claude count_tokens one
    # (count_tokens runs 26-39% higher on code). And it is priced over the
    # files claude-context ACTUALLY indexed, not our wider text allowlist --
    # WS6a priced the allowlist and thereby overstated its own index cost.
    enc = tiktoken.get_encoding("cl100k_base")
    indexed_tokens = sum(
        len(enc.encode((tree / p).read_text(errors="ignore"))) for p in indexable)

    stats = pd.read_csv(ws6c_dir() / "corpus_stats.csv").set_index("slug")
    row = stats.loc[slug]
    prices = ws6a.load_pricing(ws6a_dir() / "pricing.csv")
    rate = prices[WS6A_EMBED_MODEL]["input_usd_per_mtok"]

    info.update({
        "slug": slug,
        "repo": row["repo"],
        "commit": row["commit"],
        "checkout": str(checkout),
        "tree": str(tree),
        "dropped_ignore_files": derived["dropped_ignore_files"],
        # sha256 over "<path> <file sha256>\n" lines, sorted by path. Lets a
        # reader reproduce the derived tree from the pinned checkout and
        # confirm it in one comparison.
        "tree_sha256": derived["tree_sha256"],
        "embed_model": WS6A_EMBED_MODEL,
        "files_tracked": len(tracked),
        "files_text_allowlisted": len(text_files),
        "files_claude_context_indexable": len(indexable),
        "indexed_tokens_tiktoken": indexed_tokens,
        # Whole-allowlist figures, for comparability with WS6a's own row.
        "corpus_tokens_tiktoken": int(row["tiktoken_cl100k"]),
        "corpus_tokens_claude": int(row["count_tokens"]),
        # DERIVED, NOT MEASURED. @zilliz/claude-context-mcp@0.1.15 surfaces no
        # embedding usage through MCP, so this is a local tiktoken count over
        # the files the server actually indexed times a published list rate.
        # It is the only basis available and is labelled as derived
        # everywhere it is reported. It is NOT covered by the Anthropic-
        # denominated cost governor -- different provider.
        "embed_cost_usd": round(indexed_tokens / 1_000_000 * rate, 6),
        "embed_cost_basis": (
            "DERIVED: tiktoken_cl100k over the claude-context-indexable files "
            "x list rate; the package exposes no embedding usage field, and "
            "this spend is NOT under the Anthropic cost governor"),
        "mean_chunk_tokens": (round(indexed_tokens / rows, 1) if rows else 0),
    })

    prev = [pd.read_csv(path)] if path.exists() else []
    out = pd.concat(prev + [pd.DataFrame([info])], ignore_index=True)
    out = out.drop_duplicates(subset=["slug"], keep="last").sort_values("slug")
    out.to_csv(path, index=False)
    manifest.record(path)
    print(out.drop(columns=["status_text"]).to_string(index=False))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
