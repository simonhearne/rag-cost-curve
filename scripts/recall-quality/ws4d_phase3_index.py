#!/usr/bin/env python
"""WS4d Phase 3a (spec S5.2, corrected 2026-09-12): rebuild the IVF_SQ8 index
that was dropped from ws2_10m_shared, and sweep the four WS4B_EXTRA_NPROBES
gap-filling arms.

Design: results/recall-vs-quality.md

WS4's arm ladder has three gaps wider than 0.035 recall in exactly the band
where the changepoint sits; that geometry -- not question noise -- caps the
confidence interval. These four arms fill those gaps on the *existing*
shared SQ8 index (WS2_10M_SHARED / WS4_NLIST), so they are directly
comparable to the eleven already-measured arms.

Usage:
    python scripts/recall-quality/ws4d_phase3_index.py build   # ~1.8-2.7h; run under nohup
    python scripts/recall-quality/ws4d_phase3_index.py sweep   # fast; needs the index loaded
    python scripts/recall-quality/ws4d_phase3_index.py verify  # recompute recall from the npz files

`build` is idempotent: skips straight to load if an index already exists.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib.config import (  # noqa: E402
    DATA_DIR, EMBED_DIM, MILVUS_URI, MILVUS_VERSION, WS4B_EXTRA_NPROBES,
    WS4_NLIST, WS4_SHARED_COLLECTION, ws4_cache,
)
from benchlib.quantization import VECTOR_FIELD, recall_at_k  # noqa: E402

CACHE = ws4_cache()
RET = CACHE / "retrieval"
RET.mkdir(parents=True, exist_ok=True)
BUILD_LOG = CACHE / "phase3_build.json"
N_ROWS_EXPECTED = 10_000_000


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def make_client():
    from pymilvus import MilvusClient
    c = MilvusClient(uri=MILVUS_URI)
    ver = c.get_server_version()
    assert MILVUS_VERSION in ver, f"expected Milvus {MILVUS_VERSION}, got {ver}"
    return c


# --------------------------------------------------------------- build ---


def cmd_build() -> None:
    c = make_client()
    coll = WS4_SHARED_COLLECTION
    n = int(c.get_collection_stats(coll)["row_count"])
    if n != N_ROWS_EXPECTED:
        raise RuntimeError(f"{coll} has {n} rows, expected {N_ROWS_EXPECTED}")

    existing = c.list_indexes(coll)
    if existing:
        log(f"index already present ({existing}); skipping build, loading only")
    else:
        # verified parameters -- do not change: these must match the index
        # the eleven already-measured arms were swept on, or the new arms are
        # not comparable to them.
        ip = c.prepare_index_params()
        ip.add_index(field_name=VECTOR_FIELD, index_type="IVF_SQ8",
                     metric_type="IP", params={"nlist": WS4_NLIST})
        log(f"building IVF_SQ8 metric_type=IP nlist={WS4_NLIST} on "
            f"{coll}.{VECTOR_FIELD} ({n} rows) -- expect 1.8-2.7h")
        t0 = time.perf_counter()
        c.create_index(coll, ip)
        index_name = c.list_indexes(coll)[0]
        while True:
            d = c.describe_index(coll, index_name)
            pending = int(d.get("pending_index_rows", 0) or 0)
            total = int(d.get("total_rows", 0) or 0)
            indexed = int(d.get("indexed_rows", 0) or 0)
            state = str(d.get("state", ""))
            if state == "Failed":
                raise RuntimeError(f"index build failed for {coll}: {d}")
            if pending == 0 and total > 0 and indexed >= total:
                break
            time.sleep(5.0)
        build_s = time.perf_counter() - t0
        BUILD_LOG.write_text(json.dumps({
            "collection": coll, "index_type": "IVF_SQ8", "metric_type": "IP",
            "params": {"nlist": WS4_NLIST}, "build_seconds": build_s,
            "n_rows": n}, indent=1))
        log(f"build complete in {build_s:.1f}s ({build_s/3600:.2f}h)")

    log("loading collection")
    t0 = time.perf_counter()
    c.load_collection(coll)
    log(f"load complete in {time.perf_counter() - t0:.1f}s")
    log(f"load_state={c.get_load_state(coll)}")
    log("BUILD_DONE")


# --------------------------------------------------------------- sweep ---


def search_with_scores(client, coll, queries, limit, params, batch=1000):
    nq = len(queries)
    ids = np.full((nq, limit), -1, dtype=np.int64)
    sc = np.full((nq, limit), np.nan, dtype=np.float32)
    for s in range(0, nq, batch):
        block = np.asarray(queries[s:s + batch], dtype=np.float32)
        res = client.search(coll, data=block, anns_field=VECTOR_FIELD,
                            limit=limit, search_params={"params": params})
        for j, hits in enumerate(res):
            for r, h in enumerate(hits):
                ids[s + j, r] = h["id"]
                sc[s + j, r] = h["distance"]
    return ids, sc


def cmd_sweep() -> None:
    c = make_client()
    coll = WS4_SHARED_COLLECTION
    state = c.get_load_state(coll)
    log(f"load_state={state}")

    queries = np.load(DATA_DIR / "queries_ws4_emb.npy")
    assert queries.shape == (3610, EMBED_DIM), queries.shape
    # already L2-normalised per the brief -- do not re-embed; spot-check norms
    norms = np.linalg.norm(queries[:50], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), (
        f"queries_ws4_emb.npy does not look L2-normalised: {norms[:5]}")

    gt = np.load(DATA_DIR / "gt_10m_top100.npz")["ws4_indices"]
    assert gt.shape[0] == 3610

    for nprobe in WS4B_EXTRA_NPROBES:
        key = f"sq8_np{nprobe}"
        out = RET / f"{key}.npz"
        if out.exists():
            log(f"[{key}] already cached, skipping")
            continue
        t0 = time.perf_counter()
        ids, sc = search_with_scores(c, coll, queries, 10, {"nprobe": nprobe})
        seconds = time.perf_counter() - t0
        r10 = recall_at_k(ids, gt, 10)
        meta = {
            "arm": key, "family": "reference", "index": "IVF_SQ8",
            "collection": coll, "dim": EMBED_DIM,
            "params": {"nprobe": nprobe}, "seconds": round(seconds, 1),
            "ws4_recall_at_10_all3610": r10, "n_queries": int(ids.shape[0]),
        }
        np.savez(out, ids=ids, scores=sc, meta=json.dumps(meta))
        log(f"[{key}] saved; {seconds:.1f}s; r@10 over all 3610 = {r10:.4f}")


# -------------------------------------------------------------- verify ---


def cmd_verify() -> None:
    gt = np.load(DATA_DIR / "gt_10m_top100.npz")["ws4_indices"][:, :10]
    for nprobe in WS4B_EXTRA_NPROBES:
        f = RET / f"sq8_np{nprobe}.npz"
        z = np.load(f)
        ids = z["ids"]
        r = recall_at_k(ids, gt, 10)
        in_band = 0.80 <= r <= 0.97
        print(f"{f.name:24s} {ids.shape} recall@10={r:.4f} "
              f"{'OK' if in_band else 'OUT_OF_BAND[0.80,0.97]'}")


COMMANDS = {"build": cmd_build, "sweep": cmd_sweep, "verify": cmd_verify}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: python {sys.argv[0]} {{{'|'.join(COMMANDS)}}}")
    COMMANDS[sys.argv[1]]()
