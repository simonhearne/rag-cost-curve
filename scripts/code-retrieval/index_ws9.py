#!/usr/bin/env python
"""Index one WS9 scope with claude-context and record its one-time cost.

Outputs: results/ws6c/index_cost.csv (per-scope rows; the existing WS6c row is
backfilled as scope=L with every other value byte-identical)

Modelled directly on scripts/code-retrieval/index_ws6c.py. Two differences:

  * --scope in place of --slug, with subset materialisation. The subset is cut
    from the DERIVED tree, not the git checkout, so it inherits the dropped
    root-level .dockerignore -- the file that makes the pinned package index
    ZERO files and call it success -- and so each subset is provably a subset
    of the exact corpus WS6c measured, byte for byte.
  * scope=L is a RECORD-ONLY path. L is already indexed and already measured;
    re-indexing it would re-spend the embedding cost and, worse, replace a
    measurement the published WS6c result rests on.

WHY THE VERIFICATION IS NOT OPTIONAL. Indexing is ASYNCHRONOUS and neither
obvious completion signal is trustworthy -- both were caught lying during WS6a.
get_indexing_status reported "fully indexed ... 1100 files" 19 seconds into a
2,825-file job because its handler treats the mere EXISTENCE of a cloud
collection as proof of completion; get_collection_stats returned row_count 0
for a collection holding 1,400 rows because stats lag unflushed inserts. A
partially-indexed collection answers every query perfectly happily. Three
independent sources must agree before any run spends.

Run:  .venv/bin/python scripts/code-retrieval/index_ws9.py --scope S
      .venv/bin/python scripts/code-retrieval/index_ws9.py --scope M --embedding-batch-size 20
      .venv/bin/python scripts/code-retrieval/index_ws9.py --scope L   # record-only
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
from benchlib.agent_harness import claude_context_session
from benchlib.code_retrieval import bench as ws6a
from benchlib.code_retrieval import scaling as ws9
from benchlib.code_retrieval import topk as ws6c
from benchlib.config import (
    WS6A_EMBED_MODEL, WS6C_COMMITS, WS9_SCOPES, WS9_SLUG, ws6a_dir,
    ws6c_checkout, ws6c_dir, ws9_cache_dir, ws9_subset_tree,
)

POLL_SECONDS = 15
POLL_TIMEOUT_SECONDS = 7200

COMPLETE_RE = re.compile(
    r"Indexing completed successfully! Files: (\d+), Chunks: (\d+)")
FAILED_RE = re.compile(r"Indexing failed for .*|Indexing for .* was cancelled")
PROGRESS_RE = re.compile(r"Processed (\d+)/(\d+) files")


def stderr_log_path(scope: str) -> Path:
    return ws9_cache_dir() / f"mcp_index_{ws9.ws9_collection_prefix(scope)}.stderr.log"


def _text(result) -> str:
    return "\n".join(getattr(b, "text", "") for b in result.content)


async def run_index(tree: Path, prefix: str, log: Path, force: bool) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        # Source 1 must describe THIS index and nothing else.
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
        raise SystemExit(f"no server log at {log}")
    text = log.read_text(encoding="utf-8", errors="replace")
    done = COMPLETE_RE.findall(text)
    if not done:
        raise SystemExit(
            f"{log} carries no 'Indexing completed successfully!' line -- the "
            "index is unfinished or was killed mid-run")
    files, chunks = (int(x) for x in done[-1])
    progress = PROGRESS_RE.findall(text)
    return {"server_files": files, "server_chunks": chunks,
            "server_enumerated": int(progress[-1][1]) if progress else -1}


def milvus_rows(prefix: str) -> dict:
    """SOURCE 2: an independent count(*), after an explicit flush."""
    from pymilvus import MilvusClient

    client = MilvusClient(uri=os.environ["MILVUS_ADDRESS"],
                          token=os.environ["MILVUS_TOKEN"])
    names = [c for c in client.list_collections() if prefix in c]
    if not names:
        raise SystemExit(f"no collection matching {prefix!r}; "
                         f"saw {client.list_collections()}")
    if len(names) > 1:
        # The collection name hashes the INDEXED PATH, so a retargeted index
        # leaves the old one behind and sorted()[0] is random with respect to
        # which is current.
        raise SystemExit(
            f"{len(names)} collections match {prefix!r}: {sorted(names)}. "
            "Drop the stale one before verifying.")
    name = names[0]
    client.flush(name)
    client.load_collection(name)
    rows = client.query(name, filter="", output_fields=["count(*)"])
    return {"collection": name, "chunk_count": int(rows[0]["count(*)"])}


def upsert(path: Path, info: dict) -> pd.DataFrame:
    """Write one (slug, scope) row, backfilling the pre-WS9 row as scope=L.

    The existing WS6c row has no `scope` column. Backfilling it must not change
    a single other value -- results/ws6c/index_cost.csv is a published WS6c
    output the deck copies.
    """
    prev = []
    if path.exists():
        old = pd.read_csv(path)
        if "scope" not in old.columns:
            before = old.copy()
            old.insert(1, "scope", "L")
            # Every other column, byte-identical.
            assert old.drop(columns=["scope"]).equals(before), (
                "backfilling `scope` changed another column -- refusing to "
                "write over a published WS6c output")
            print(f"backfilled {len(old)} pre-WS9 row(s) as scope=L")
        prev = [old]
    out = pd.concat(prev + [pd.DataFrame([info])], ignore_index=True)
    out = out.drop_duplicates(subset=["slug", "scope"], keep="last")
    out = out.sort_values(["slug", "scope"])
    out.to_csv(path, index=False)
    manifest.record(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", required=True, choices=list(WS9_SCOPES))
    ap.add_argument("--force", action="store_true",
                    help="clear any existing index for this scope and rebuild "
                         "-- required after a partial/killed index, and it "
                         "re-spends the embedding cost")
    ap.add_argument("--embedding-batch-size", type=int, default=None,
                    help="EMBEDDING_BATCH_SIZE for the server (package default "
                         "100). A THROUGHPUT throttle, not a content change: "
                         "it alters how many chunks per embedding request, "
                         "never the chunks or the vectors. M embeds ~1.14M "
                         "tokens against a 1M TPM account limit, so 20 is "
                         "needed there; at the default the index 429s and the "
                         "package has no retry.")
    ap.add_argument("--verify-only", action="store_true",
                    help="skip indexing; re-run the three-source cross-check")
    args = ap.parse_args()
    scope = args.scope
    path = ws6c_dir() / "index_cost.csv"

    if scope == "L":
        # RECORD-ONLY. L is already indexed and already measured; re-indexing
        # would re-spend the embedding cost and replace a measurement the
        # published WS6c result rests on.
        if not path.exists():
            raise SystemExit(f"{path} does not exist -- L has never been indexed")
        old = pd.read_csv(path)
        if "scope" in old.columns and "L" in set(old["scope"]):
            print("L row already carries scope=L; nothing to do")
            return
        old.insert(1, "scope", "L")
        old.to_csv(path, index=False)
        manifest.record(path)
        print(f"backfilled the existing WS6c row as scope=L in {path}")
        return

    print("checking the pin before any embedding spend")
    ws6c.assert_pinned_checkouts([WS9_SLUG])

    checkout = ws6c_checkout(WS9_SLUG)
    l_tree = ws6c.index_tree(WS9_SLUG)
    tracked = ws6a.git_ls_files(checkout)
    dropped = set(ws6c.dropped_ignore_files(tracked))
    in_tree = [p for p in tracked if p not in dropped]

    if not ws9.ws9_scopes_are_nested(in_tree):
        raise SystemExit("WS9 scopes are NOT nested -- fix WS9_SUBSETS")

    # Cut the subset from the DERIVED tree, and verify it file by file against
    # that tree rather than against the checkout: the derived tree is the
    # corpus WS6c measured, and a subset of it is provably a subset of that.
    wanted = ws9.ws9_subset_files(in_tree, scope)
    tree = ws9_subset_tree(scope)
    derived = ws6c.materialise_index_tree(l_tree, wanted, tree)
    print(f"subset tree verified: {derived['files_copied']} files at {tree}")

    # SOURCE 3: what we handed it.
    indexable = ws6c.claude_context_indexable(wanted)
    text_files = ws6a.filter_text_files(wanted)
    print(f"scope {scope}: {len(wanted)} files in tree, {len(text_files)} in "
          f"OUR text allowlist, {len(indexable)} claude-context will index")

    if args.embedding_batch_size:
        os.environ["EMBEDDING_BATCH_SIZE"] = str(args.embedding_batch_size)
        print(f"EMBEDDING_BATCH_SIZE={args.embedding_batch_size} (throttle)")

    prefix = ws9.ws9_collection_prefix(scope)
    log = stderr_log_path(scope)
    if args.verify_only:
        prior = pd.read_csv(path)
        row0 = prior[(prior.slug == WS9_SLUG) & (prior.scope == scope)]
        if row0.empty:
            raise SystemExit(f"{path} has no row for scope {scope!r} to verify")
        # Re-verifying must never blank wall_clock_s -- a NaN there is a
        # silently destroyed measurement, not a missing one.
        info = {k: row0.iloc[0][k] for k in
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
    if info["server_enumerated"] != info["server_files"]:
        raise SystemExit(
            f"server completed at {info['server_files']} files but enumerated "
            f"{info['server_enumerated']} -- this is the 330-of-2825 partial index")
    if info["chunk_count"] != info["server_chunks"]:
        raise SystemExit(
            f"Milvus holds {info['chunk_count']} rows, server claims "
            f"{info['server_chunks']} chunks")
    if info["server_files"] != len(indexable):
        raise SystemExit(
            f"server indexed {info['server_files']} files, we handed it "
            f"{len(indexable)} indexable of {len(wanted)} in tree")
    print(f"INDEX VERIFIED: server log {info['server_files']}/"
          f"{info['server_enumerated']} files, {info['server_chunks']} chunks; "
          f"Milvus count(*) {info['chunk_count']}; predicted indexable "
          f"{len(indexable)} -- all three agree")

    # OpenAI bills embeddings on ITS OWN tokenizer, so cost is priced over
    # tiktoken, not count_tokens (which runs 26-39% higher on code) -- and over
    # the files claude-context ACTUALLY indexed, not our wider text allowlist.
    enc = tiktoken.get_encoding("cl100k_base")
    indexed_tokens = sum(
        len(enc.encode((tree / p).read_text(errors="ignore"))) for p in indexable)
    stats = pd.read_csv(ws6c_dir() / "scaling_corpus_stats.csv").set_index("scope")
    row = stats.loc[scope]
    prices = ws6a.load_pricing(ws6a_dir() / "pricing.csv")
    rate = prices[WS6A_EMBED_MODEL]["input_usd_per_mtok"]

    # Five published columns below are COPIED out of the stats CSV rather than
    # measured here. The three-way check says nothing about them: a stale or
    # regenerated scaling_corpus_stats.csv would be laundered into
    # index_cost.csv under an INDEX VERIFIED line. Cross-check the two the run
    # holds live values for -- the file count we just cut, and the pin
    # assert_pinned_checkouts already verified -- before trusting the row.
    if len(text_files) != int(row["files_text_allowlisted"]):
        raise SystemExit(
            f"scaling_corpus_stats.csv disagrees with this run at scope "
            f"{scope}: it records files_text_allowlisted="
            f"{int(row['files_text_allowlisted'])}, we cut "
            f"{len(text_files)} text-allowlisted files -- re-measure the "
            "corpus before recording a cost row against it")
    if str(row["commit"]) != WS6C_COMMITS[WS9_SLUG]:
        raise SystemExit(
            f"scaling_corpus_stats.csv was measured at commit "
            f"{str(row['commit'])!r} but the pinned corpus is "
            f"{WS6C_COMMITS[WS9_SLUG]!r} -- re-measure the corpus before "
            "recording a cost row against it")

    info.update({
        "slug": WS9_SLUG, "scope": scope,
        "repo": "agentic-hil/agentic-hil", "commit": str(row["commit"]),
        "checkout": str(checkout), "tree": str(tree),
        "dropped_ignore_files": "|".join(sorted(dropped)),
        "tree_sha256": derived["tree_sha256"],
        "embed_model": WS6A_EMBED_MODEL,
        "files_tracked": int(row["files_tracked"]),
        "files_text_allowlisted": int(row["files_text_allowlisted"]),
        "files_claude_context_indexable": len(indexable),
        "indexed_tokens_tiktoken": indexed_tokens,
        "corpus_tokens_tiktoken": int(row["tiktoken_cl100k"]),
        "corpus_tokens_claude": int(row["count_tokens"]),
        "embed_cost_usd": round(indexed_tokens / 1_000_000 * rate, 6),
        "embed_cost_basis": (
            "DERIVED: tiktoken_cl100k over the claude-context-indexable files "
            "x list rate; the package exposes no embedding usage field, and "
            "this spend is NOT under the Anthropic cost governor"),
        "mean_chunk_tokens": (round(indexed_tokens / info["chunk_count"], 1)
                              if info["chunk_count"] else 0),
    })

    out = upsert(path, info)
    print(out[["slug", "scope", "chunk_count", "indexed_tokens_tiktoken",
               "embed_cost_usd", "wall_clock_s"]].to_string(index=False))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
