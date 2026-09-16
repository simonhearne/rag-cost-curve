#!/usr/bin/env python
"""WS8 Phase 2: build the `indexed` arm's Milvus collection and run the
MANDATORY three-way completeness assert.

WS6a found a status API reporting "fully indexed" at 330 of 2,825 files.
WS6b found "Indexed 52 chunks." with rc=0 followed by every search returning
"No results found." (unflushed rows read as zero row_count). WS6c found a
vendor tool that indexed ZERO files and reported "completed successfully".
A partially-indexed collection answers every query perfectly happily -- the
three-way assert (writer's own count, an independent Milvus count(*) after
an explicit flush, and a corpus-derived prediction) is what catches that.
See benchlib/non_code_retrieval/index.py's module docstring for the full detail; this
script is deliberately thin orchestration around that pure module.

CHECKPOINTING / RESUME (the embedding run is long -- ~42,000 chunks):

  * Rows are built ONCE, deterministically (benchlib.non_code_retrieval.index.build_rows),
    from documents sorted by filename. The same corpus + chunk params always
    produce the same row list in the same order, so a row's sequential
    integer id is stable across a crash and a fresh process.
  * Primary keys are EXPLICIT (auto_id=False) and writes use mc.upsert, not
    mc.insert -- re-processing a batch after a crash overwrites the same
    rows in place rather than duplicating them. That is what makes resume
    safe against double-INSERT.
  * Resume against double-BILL is only partial, by design, and that's
    disclosed here rather than engineered away: a checkpoint line is
    appended only AFTER a batch's embed-and-upsert both succeed, so a crash
    between the (already billed) embeddings.create() call and the
    checkpoint write means that one batch is re-embedded (and re-billed) on
    resume. The upsert side stays correct regardless (same ids). This
    matches this repo's own established convention -- ws8.spent_from_checkpoint
    deliberately counts every billed line, INCLUDING superseded duplicates,
    because over-counting spend is the safe direction for a governor, never
    the reverse (see that function's docstring). At ~$0.02/1M tokens the
    entire corpus embeds for well under $1, so the worst case here (one
    re-billed batch of ~128 chunks) is a rounding error, not a real risk.
  * A corpus/chunk-parameter SIGNATURE is stamped on every checkpoint line
    (benchlib.non_code_retrieval.index.corpus_signature) and verified on resume
    (benchlib.non_code_retrieval.index.resume_from raises on a mismatch) -- resuming a
    checkpoint written against a different corpus or WS8_CHUNK_CHARS/
    WS8_CHUNK_OVERLAP would otherwise silently skip rows never actually
    inserted for the corpus this run is building.
  * A batch whose upsert fails after a successful (billed) embed call is
    recorded as a spend-only, `inserted: False` checkpoint line -- the
    judge-exception spend pattern gate_ws8.py established: an already-billed
    call is never discarded, even on failure. Unlike gate_ws8.py's
    independent (qid, arm) pairs, these rows are a CONTIGUOUS range, so this
    script STOPS on such a failure rather than continuing to the next batch
    -- continuing would leave a permanent, silently-unretried gap, since
    resume_from only ever advances past batches marked `inserted: True`.

CHECKPOINT NAMING: data/ws8_cache/index_checkpoint.jsonl. Deliberately NOT
"gate_checkpoint*" -- scripts/non-code-retrieval/gate_ws8.py's PHASE1_CHECKPOINT_GLOB
("ws8_cache/gate_checkpoint*.jsonl") scopes Phase 1's own spend total, and a
Phase 2 checkpoint matching that pattern would silently join it. This name
does not match that glob, so Phase 1's phase_spend_usd is unaffected; it DOES
match ws8.WS8_CHECKPOINT_GLOB ("ws8_cache/*.jsonl"), which is what Phase 3's
governor seeds its cumulative spend from -- exactly the behaviour wanted.

Run:  python scripts/non-code-retrieval/index_ws8.py
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from pymilvus import DataType, MilvusClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.non_code_retrieval import bench as ws8
from benchlib.non_code_retrieval import index as ws8_index
from benchlib.code_retrieval import bench as ws6a
from benchlib.config import (DATA_DIR, WS8_BUDGET_USD, WS8_CHUNK_CHARS,
                             WS8_CHUNK_OVERLAP, WS8_COLLECTION,
                             WS8_CORPUS_DIR, WS8_EMBED_DIM, WS8_EMBED_MODEL,
                             WS8_MILVUS_URI, ws6a_dir, ws8_dir)

BATCH = 128
CHECKPOINT = DATA_DIR / "ws8_cache" / "index_checkpoint.jsonl"

# The ledger writer lives in benchlib.non_code_retrieval.bench (hoisted
# 2026-09-11 from the four verbatim copies this file was one of); see it
# for the column mechanism.


def load_checkpoint_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text()
    raw_lines = text.splitlines()
    out = []
    for idx, raw in enumerate(raw_lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            if idx == len(raw_lines) - 1:
                continue  # torn trailing write from a mid-write crash
            raise ValueError(f"unparseable JSON at line {idx + 1} of {path}")
    return out


def append_checkpoint(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def load_docs() -> list[tuple[str, str]]:
    docs_dir = WS8_CORPUS_DIR / "docs"
    paths = sorted(docs_dir.glob("*.md"))
    if not paths:
        sys.exit(f"no documents under {docs_dir} -- run "
                 "scripts/non-code-retrieval/build_ws8_corpus.py first")
    return [(p.name, p.read_text(encoding="utf-8")) for p in paths]


def ensure_collection(mc: MilvusClient, *, fresh: bool) -> None:
    """Create WS8_COLLECTION if needed. `fresh=True` drops any existing
    collection first -- only safe when the caller has already established
    that no checkpointed progress exists for the current corpus signature
    (see main()); otherwise it would silently discard already-inserted,
    already-billed rows."""
    exists = mc.has_collection(WS8_COLLECTION)
    if exists and not fresh:
        return
    if exists and fresh:
        mc.drop_collection(WS8_COLLECTION)
    schema = mc.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=WS8_EMBED_DIM)
    schema.add_field("document", DataType.VARCHAR, max_length=256)
    schema.add_field("text", DataType.VARCHAR, max_length=8192)
    index_params = mc.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="HNSW",
                           metric_type="COSINE",
                           params={"M": 16, "efConstruction": 200})
    mc.create_collection(WS8_COLLECTION, schema=schema,
                         index_params=index_params)


def main() -> None:
    out = ws8_dir()

    docs = load_docs()
    rows = ws8_index.build_rows(docs, WS8_CHUNK_CHARS, WS8_CHUNK_OVERLAP)
    predicted = len(rows)
    signature = ws8_index.corpus_signature(
        [name for name, _ in docs], WS8_CHUNK_CHARS, WS8_CHUNK_OVERLAP)
    print(f"{len(docs)} documents -> {predicted} predicted chunks "
          f"(size={WS8_CHUNK_CHARS} overlap={WS8_CHUNK_OVERLAP}, "
          f"signature={signature[:12]}...)")

    checkpoint_lines = load_checkpoint_lines(CHECKPOINT)
    # Raises ValueError if any checkpoint line carries a different
    # corpus/chunk signature -- see ws8_index.resume_from's docstring (R18).
    start_row = ws8_index.resume_from(checkpoint_lines, signature)

    mc = MilvusClient(uri=WS8_MILVUS_URI)
    collection_exists = mc.has_collection(WS8_COLLECTION)
    if start_row > 0 and not collection_exists:
        sys.exit(
            f"BLOCKED: {CHECKPOINT} claims {start_row} row(s) already "
            f"inserted for this corpus/chunk signature, but Milvus has no "
            f"collection named {WS8_COLLECTION!r}. Do not proceed -- the "
            "checkpoint's claimed progress is unverifiable. Archive the "
            "checkpoint (R18: rename to <name>.voided-<ISO8601>.jsonl) and "
            "decide by hand whether to rebuild from zero.")

    fresh = start_row == 0
    if fresh and collection_exists:
        print(f"WARNING: dropping and recreating {WS8_COLLECTION!r} -- no "
              "checkpointed progress exists for this corpus/chunk "
              "signature, so any prior contents are unverified and are "
              "being replaced, not trusted.", file=sys.stderr)
    ensure_collection(mc, fresh=fresh)
    print(f"{'starting fresh' if fresh else f'resuming from row {start_row}'}"
          f" of {predicted}")

    oai = OpenAI(max_retries=8)
    prices = ws6a.load_pricing(ws6a_dir() / "pricing.csv")
    rate = prices[WS8_EMBED_MODEL]["input_usd_per_mtok"]

    spent = ws8.total_spent_across_checkpoints(DATA_DIR)
    governor = ws6a.CostGovernor(WS8_BUDGET_USD, spent=spent)
    print(f"governor seeded at ${governor.spent:.4f} of ${WS8_BUDGET_USD:.2f} "
          "(cumulative across every WS8 checkpoint, live and archived)")

    ledger_path = out / "spend_ledger.csv"
    # Filled in just before the ledger row is written on the success path. Its
    # DEFAULT is the abort text, so a run that dies anywhere in the embedding
    # loop -- including the billed-but-unupsertable batch the loop sys.exit()s
    # on -- still produces a row, and that row cannot be mistaken for a clean
    # one.
    ledger_note = {"text": ("ABORTED before the three-way assert; this "
                            "invocation's spend is real and is in "
                            "index_checkpoint.jsonl, but the collection is "
                            "NOT verified complete")}
    ledger_written: list[bool] = []

    def write_ledger(error: BaseException | None = None) -> None:
        """Append this invocation's spend row. Called from a `finally`.

        THIS invocation's own delta (governor.spent - spent), not the
        phase-cumulative figure: the ledger gets one row per invocation, so a
        cumulative number here would double-count across rows on a resumed run
        -- exactly the mistake spend_ledger.csv's schema exists to avoid (see
        gate_ws8.py's spend_this_invocation_usd vs phase_spend_usd).
        """
        if ledger_written:
            return
        ledger_written.append(True)
        invocation_spend = round(governor.spent - spent, 6)
        note = ledger_note["text"]
        if error is not None:
            detail = " ".join(str(error).split())[:200]
            note += f" [{type(error).__name__}: {detail}]"
        ws8.append_spend_ledger(ledger_path, run_label="index_ws8_phase2",
                                spend_usd=invocation_spend, note=note)
        manifest.record(ledger_path)
        print(f"appended ${invocation_spend:.6f} to {ledger_path}")
        ws8.warn_on_ledger_mismatch(ledger_path, DATA_DIR)

    try:
        started = time.time()
        i = start_row
        while i < len(rows):
            batch = rows[i:i + BATCH]
            texts = [r["text"] for r in batch]
            resp = oai.embeddings.create(model=WS8_EMBED_MODEL, input=texts)
            tokens = int(resp.usage.total_tokens)
            cost = round(tokens / 1_000_000 * rate, 6)

            try:
                entities = [{**r, "vector": d.embedding}
                           for r, d in zip(batch, resp.data)]
                mc.upsert(collection_name=WS8_COLLECTION, data=entities)
            except Exception as exc:
                # Judge-exception spend pattern: the embeddings.create() call
                # above already happened and was billed. Record it as a
                # spend-only, inserted=False line and charge the governor before
                # stopping -- see module docstring for why this halts (a
                # contiguous range) rather than continuing (as gate_ws8.py does
                # for its independent qid/arm pairs).
                append_checkpoint(CHECKPOINT, {
                    "batch_start": i, "batch_end": i + len(batch),
                    "n_rows": 0, "embed_tokens": tokens,
                    "cost_total_billed": cost, "signature": signature,
                    "inserted": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                governor.charge(cost)
                sys.exit(
                    f"upsert failed for rows [{i}, {i + len(batch)}) after a "
                    f"BILLED embedding call (${cost:.6f} charged and "
                    f"checkpointed as spend-only, inserted=False): "
                    f"{type(exc).__name__}: {exc}\n"
                    "Re-run to resume -- this exact batch will retry, at the "
                    "cost of a re-billed embedding call for it alone.")

            append_checkpoint(CHECKPOINT, {
                "batch_start": i, "batch_end": i + len(batch),
                "n_rows": len(batch), "embed_tokens": tokens,
                "cost_total_billed": cost, "signature": signature,
                "inserted": True,
            })
            governor.charge(cost)  # raises BudgetExceeded to stop cleanly; already checkpointed
            i += len(batch)
            print(f"  {i}/{predicted} chunks embedded, ${governor.spent:.4f} "
                  f"spent", end="\r")

        wall = round(time.time() - started, 1)
        print(f"\nembedding complete in {wall}s")

        # `reported` is the WRITER's own count: sum of n_rows over this
        # signature's `inserted: True` checkpoint lines -- what the loop above
        # actually recorded itself doing, read back from disk, not a value
        # derived from `predicted`. Fix round 1 (review): an earlier version set
        # `reported = predicted` directly, which made the writer-count leg of
        # the assert compare `predicted` against itself under a second name and
        # gave it zero independent discriminating power -- it would never have
        # caught a loop that believed it inserted more rows than its own
        # checkpoint shows. Duplicate/overlapping lines for the same batch range
        # cannot occur in a normal run: the while loop above only ever advances
        # past a batch after a line is appended for it, and resume_from (used to
        # compute start_row before the loop) always resumes strictly after the
        # last `inserted: True` batch_end, so no batch is ever re-recorded once
        # it has one successful line.
        all_lines = load_checkpoint_lines(CHECKPOINT)
        sig_lines = [l for l in all_lines if l.get("signature") == signature]
        reported = sum(int(l["n_rows"]) for l in sig_lines if l.get("inserted"))

        # assert_index_complete cross-checks `reported` against an INDEPENDENT
        # Milvus count(*) (after an explicit flush) and against `predicted`
        # (the corpus-derived figure computed before any embedding happened).
        # All three are now genuinely distinct measurements that happen to
        # agree, not one number compared against itself twice.
        ws8_index.assert_index_complete(mc, WS8_COLLECTION, reported, predicted)
        print(f"THREE-WAY ASSERT PASSED: writer reported={reported}, Milvus "
              f"count(*)={predicted}, corpus predicted={predicted}")

        mc.load_collection(WS8_COLLECTION)
        print(f"{WS8_COLLECTION!r} loaded and ready for search")

        # Cumulative cost across every checkpoint line for THIS signature,
        # including any double-billed retried batch -- the safe, over-counting
        # direction (see ws8.spent_from_checkpoint's docstring). Note this is
        # NOT the same sum as `reported` above: this includes spend-only
        # (`inserted: False`) lines too, since a failed-but-billed batch still
        # cost real money even though its rows are not in the collection.
        total_tokens = sum(int(l["embed_tokens"]) for l in sig_lines)
        total_cost = round(sum(float(l["cost_total_billed"]) for l in sig_lines), 6)

        cost_row = pd.DataFrame([{
            "documents": len(docs), "chunks": reported,
            "predicted_chunks": predicted, "milvus_count": predicted,
            "embed_tokens": total_tokens, "embed_cost_usd": total_cost,
            "embed_model": WS8_EMBED_MODEL, "chunk_chars": WS8_CHUNK_CHARS,
            "chunk_overlap": WS8_CHUNK_OVERLAP, "collection": WS8_COLLECTION,
            "wall_clock_s": wall,
        }])
        cost_path = out / "index_cost.csv"
        cost_row.to_csv(cost_path, index=False)
        manifest.record(cost_path)
        print(f"wrote {cost_path}")
        print(cost_row.to_string(index=False))

        ledger_note["text"] = (
            f"chunks={reported} phase_embed_tokens={total_tokens} "
            f"phase_embed_cost_usd={total_cost} "
            f"three_way_assert=PASSED (writer={reported} "
            f"milvus={predicted} predicted={predicted})")

        print(f"\nOK: {reported} chunks indexed, three-way assert passed, "
              f"${total_cost:.6f} embedding cost, cumulative WS8 spend "
              f"${governor.spent:.6f}")
    finally:
        write_ledger(sys.exc_info()[1])


if __name__ == "__main__":
    main()
