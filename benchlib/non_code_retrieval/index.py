"""WS8 Phase 2: chunking, embedding bookkeeping, Milvus, and the mandatory
three-way completeness assert.

Design: results/non-code-retrieval.md

The assert exists because three prior workstreams each found a tool
reporting success while having indexed nothing usable:

  * WS6a: get_indexing_status said "fully indexed ... 1100 files, 1100
    chunks" 19 seconds in, while the server's own log showed 330/2,825 files
    and Milvus held 1,400 chunks.
  * WS6b: memsearch printed "Indexed 52 chunks." with rc=0, and every
    subsequent search returned "No results found." -- the collection's
    row_count read 0 because get_collection_stats() only sees sealed
    segments, and nothing had been flushed.
  * WS6c: a vendor MCP tool indexed ZERO files and reported "✅ completed
    successfully".

A partially- or zero-indexed collection answers every query perfectly
happily. assert_index_complete cross-checks the writer's own count against
an INDEPENDENT Milvus count(*) (after an explicit flush -- unflushed rows
read as zero, which is exactly WS6b's bug) and against a prediction derived
from the corpus alone, computed with the identical chunking function used to
build the rows. All three must agree, or this raises rather than warns.

This module is pure except for the two functions that take a MilvusClient
(`assert_index_complete`, `make_search_tool`), and even those never touch
the network directly here -- they only call methods on the client/embed_fn
handed to them, so they are exercised in tests with tiny fakes. No network,
no API keys, importable from a plain `pytest tests/non_code_retrieval/test_bench.py`.
"""

from __future__ import annotations

import hashlib


class IndexIncomplete(RuntimeError):
    """Raised when the three completeness signals (writer, Milvus, corpus
    prediction) disagree. Never caught and downgraded to a warning -- see
    scripts/non-code-retrieval/index_ws8.py's module docstring for why."""


def chunk_document(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-width chunks with a fixed overlap. Deterministic, no tokeniser.

    Used identically by the indexer (what actually gets embedded, via
    build_rows) and by predict_chunk_count (the corpus-derived half of the
    three-way assert) -- they MUST be the same function, or "predicted"
    stops being an independent check on the ROW COUNT and starts being a
    tautology that can never disagree with itself.
    """
    if len(text) <= size:
        return [text]
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)
            if text[i:i + size]]


def predict_chunk_count(texts, size: int, overlap: int) -> int:
    """Independent prediction of the row count, from the corpus alone."""
    return sum(len(chunk_document(t, size, overlap)) for t in texts)


def build_rows(doc_records, size: int, overlap: int) -> list[dict]:
    """(id, document, text) rows in one fixed, deterministic order.

    `doc_records` is an iterable of (filename, text) pairs, already in the
    caller's chosen order (scripts/non-code-retrieval/index_ws8.py sorts by filename) -- this
    function does not re-sort, so the same input order always assigns the
    same sequential integer id to the same chunk across runs. That
    determinism is what makes resuming-by-row-index safe: a crash and a
    fresh process both compute the IDENTICAL row list, so re-embedding rows
    [i, j) and upserting them under the SAME ids as a prior attempt
    overwrites those rows in place rather than duplicating them.
    """
    rows = []
    rid = 0
    for name, text in doc_records:
        for chunk in chunk_document(text, size, overlap):
            rows.append({"id": rid, "document": name, "text": chunk})
            rid += 1
    return rows


def corpus_signature(filenames, size: int, overlap: int) -> str:
    """Stable fingerprint of (which documents, what chunk parameters).

    Guards a checkpoint against being trusted as "already done" progress for
    a DIFFERENT corpus or chunk configuration than the one it actually
    describes. Resuming across such a change would silently skip embedding
    rows that were never inserted for the CURRENT corpus, leaving Milvus
    short of the predicted count -- and the three-way assert would only
    catch that after the run finished and reported itself complete.
    """
    h = hashlib.sha256()
    h.update(f"{size}:{overlap}\n".encode())
    for name in sorted(filenames):
        h.update(name.encode())
        h.update(b"\n")
    return h.hexdigest()


def resume_from(checkpoint_lines, signature: str) -> int:
    """Row index to resume embedding from, given prior checkpoint lines.

    Only lines whose `signature` matches the CURRENT run's signature and
    whose `inserted` flag is True count as progress. Any line recorded
    against a DIFFERENT signature raises rather than being silently ignored:
    that checkpoint describes a different corpus or chunk configuration, and
    treating it as progress here would skip rows never actually written for
    this corpus (R18 -- archive the stale checkpoint to
    `<name>.voided-<ISO8601>.jsonl` and start a fresh one; never delete or
    silently override it).

    A line with `inserted: False` is a spend-only record of a batch whose
    embedding call was billed but whose Milvus write failed or never
    happened (see scripts/non-code-retrieval/index_ws8.py's judge-exception-pattern handling).
    It must count toward cumulative spend -- ws8.total_spent_across_checkpoints
    sums every `cost_total_billed` regardless of this flag -- but must NOT
    advance the resume point, since its rows are not actually in the
    collection.
    """
    mismatched = [l for l in checkpoint_lines if l.get("signature") != signature]
    if mismatched:
        raise ValueError(
            f"checkpoint has {len(mismatched)} line(s) recorded against a "
            "different corpus/chunk-config signature than the current run "
            f"({signature}) -- archive it to <name>.voided-<ISO8601>.jsonl "
            "before resuming (R18); resuming across a signature change "
            "would silently skip rows never inserted for this corpus.")
    done = [l for l in checkpoint_lines if l.get("inserted")]
    if not done:
        return 0
    return max(l["batch_end"] for l in done)


def assert_index_complete(mc, collection: str, reported: int,
                          predicted: int) -> None:
    """Three signals must agree: writer's count, Milvus count(*), prediction.

    The flush is load-bearing: get_collection_stats() reads sealed segments
    only, so unflushed rows read as zero -- WS6b's exact bug (memsearch
    reported "Indexed 52 chunks." with rc=0 and every search then returned
    "No results found." because row_count read 0). count(*) via query(),
    taken AFTER an explicit flush, is the only one of the three signals here
    that Milvus itself computes independently of anything this module wrote
    or predicted.
    """
    mc.flush(collection)
    counted = int(mc.query(collection_name=collection, filter="",
                           output_fields=["count(*)"])[0]["count(*)"])
    if counted != reported:
        raise IndexIncomplete(
            f"{collection}: writer reported {reported} rows, Milvus counted "
            f"{counted}")
    if counted != predicted:
        raise IndexIncomplete(
            f"{collection}: Milvus counted {counted}, corpus predicted "
            f"{predicted}")


def make_search_tool(mc, collection: str, embed_fn, top_k: int):
    """A `search_documents` tool for the indexed arm.

    Returns a (schema, callable) pair -- raw MCP-style tool definition plus
    its implementation -- the same shape agent_harness.claude_context_tools
    consumes to build a beta_tool-wrapped callable for tool_runner. The
    description is part of the measurement: it stays factual and neutral,
    naming no retrieval strategy and making no claim about result quality,
    so it does not tip the model toward or away from using it relative to
    the agentic arm's grep/glob/read tools.

    Each hit's `document` field is returned so answers can cite the source
    filename, matching WS8_SYSTEM_PROMPT's "Always cite the specific
    document filename or filenames your answer relies on" instruction.
    """
    schema = {
        "name": "search_documents",
        "description": "Search a collection of GitHub issue and pull-request "
                       "threads. Returns the most relevant passages along "
                       "with the filename of the document each passage came "
                       "from.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "The search query."},
            },
            "required": ["query"],
        },
    }

    def call(query: str) -> str:
        hits = mc.search(collection_name=collection, data=[embed_fn(query)],
                         limit=top_k, output_fields=["document", "text"])[0]
        if not hits:
            return "No results found."
        return "\n\n".join(
            f"--- {h['entity']['document']} ---\n{h['entity']['text']}"
            for h in hits)

    return schema, call
