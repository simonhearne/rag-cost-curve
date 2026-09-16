#!/usr/bin/env python3
"""WS6b phase 4: build the S/M/L memsearch indexes in local Milvus.

Writes results/ws6b/index_cost.csv.

Two things here are defences learned from WS6a, not conveniences:

  * The collection is FLUSHED after indexing. memsearch's store.search()
    guards BM25 against empty collections with
    get_collection_stats()['row_count'], and Milvus derives that count from
    sealed segments only -- so a freshly indexed collection reads as 0 rows
    and every search returns [] with rc=0 and no error. Recorded in
    results/ws6b/setup_effort.csv as a silent failure; it is the same mistake
    WS6a found in claude-context.
  * The reported chunk count is cross-checked against an independent Milvus
    count(*). WS6a caught a status API reporting "fully indexed -- 1100
    files" at 330/2,825 actual, which left a collection holding 21% of the
    corpus that answered queries perfectly happily.

Design: results/code-retrieval.md
"""

import csv
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import config, manifest  # noqa: E402
from benchlib.code_retrieval import memory_runner as ws6b_runner  # noqa: E402
from benchlib.code_retrieval import bench as ws6a  # noqa: E402

OUT = config.ws6b_dir()


def embed_tokens_for(paths) -> int:
    """tiktoken/cl100k count of what memsearch actually embeds.

    DERIVED, not a usage field: memsearch surfaces no embedding usage, so
    this is a measured token count times a published list rate. cl100k is
    OpenAI's own tokenizer -- NOT Claude's, which ran 26% higher on WS6a's
    corpus. Using Claude's here would overstate the index's fixed cost in
    the direction that flatters live search in C9.

    HTML comments are stripped because memsearch's
    clean_content_for_embedding() strips them before embedding.
    """
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for p in paths:
        text = re.sub(r"<!--.*?-->", "", p.read_text(), flags=re.S)
        total += len(enc.encode(text))
    return total


def main() -> int:
    from pymilvus import MilvusClient

    OUT.mkdir(parents=True, exist_ok=True)
    if not ws6b_runner.memsearch_available():
        print("memsearch is not runnable; aborting")
        return 1

    day_files = sorted(config.WS6B_CORPUS_DIR.glob("*.md"))
    stats = {r["size"]: r for r in
             csv.DictReader((OUT / "corpus_stats.csv").open())}
    prices = ws6a.load_pricing(OUT / "pricing.csv")
    embed_rate = prices[config.WS6B_EMBED_MODEL]["input_usd_per_mtok"]
    mc = MilvusClient(uri=config.WS6B_MILVUS_URI)

    rows = []
    for size in ("s", "m", "l"):
        n = int(stats[size]["files"])
        files = day_files[:n]
        collection = config.WS6B_COLLECTIONS[size]
        if collection in mc.list_collections():
            mc.drop_collection(collection)

        t0 = time.time()
        proc = ws6b_runner.memsearch_cli(
            ["index", *[str(p.resolve()) for p in files],
             *ws6b_runner._ms_common(collection)],
            timeout=7200)
        wall = round(time.time() - t0, 1)
        if proc.returncode != 0:
            print(f"{size}: index FAILED rc={proc.returncode}\n{proc.stderr[-800:]}")
            return 1

        m = re.search(r"Indexed (\d+) chunks", proc.stdout)
        reported = int(m.group(1)) if m else -1

        # Flush, or every search silently returns nothing. See the docstring.
        mc.flush(collection)
        time.sleep(2)
        counted = mc.query(collection_name=collection, filter="",
                           output_fields=["count(*)"])[0]["count(*)"]
        sealed = int(mc.get_collection_stats(collection).get("row_count", 0))

        if reported != counted:
            print(f"{size}: FAIL memsearch reported {reported} chunks but "
                  f"Milvus counts {counted}; the two describe different "
                  "corpora and no number from this collection is trustworthy")
            return 1

        etok = embed_tokens_for(files)
        rows.append({
            "size": size, "collection": collection, "files_indexed": len(files),
            "chunks_reported": reported, "milvus_count": counted,
            "milvus_sealed_row_count": sealed,
            "corpus_count_tokens": int(stats[size]["count_tokens"]),
            "wall_clock_s": wall, "embed_tokens_cl100k": etok,
            "embed_cost_usd": round(etok / 1_000_000 * embed_rate, 6),
            "embed_model": config.WS6B_EMBED_MODEL, "derived": True,
        })
        print(f"{size}: {len(files)} files, {counted} chunks, {wall}s, "
              f"{etok:,} embed tok, ${rows[-1]['embed_cost_usd']:.4f}")

    with open(OUT / "index_cost.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    manifest.record(OUT / "index_cost.csv")
    print(f"\ntotal derived embedding cost: "
          f"${sum(r['embed_cost_usd'] for r in rows):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
