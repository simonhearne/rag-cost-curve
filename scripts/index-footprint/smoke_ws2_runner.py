"""Live smoke test for benchlib.quantization Milvus runner helpers.

Run with Milvus v2.6.18 up: PYTHONPATH=. .venv/bin/python scripts/index-footprint/smoke_ws2_runner.py

Validates, on a 10k x 1024d micro-corpus, everything notebook 02 relies on:
ingest → build → load → measured-loaded-bytes API → search → recall vs
numpy brute force (FLAT must be exactly 1.0) → sequential latency →
concurrent QPS → IVF_RABITQ(+SQ8 refine) accepting refine_k at search time.
No CSVs, no manifest writes — this is a gate, not a benchmark.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pymilvus import MilvusClient

from benchlib.config import EMBED_DIM, MILVUS_URI, MILVUS_VERSION, SEED
from benchlib.quantization import (
    concurrent_qps,
    ensure_fresh_collection,
    ingest,
    build_index,
    latency_summary,
    measure_footprint,
    milvus_rss_bytes,
    recall_at_k,
    search_ids,
    timed_sequential,
)

N, NQ, K = 10_000, 200, 10
COLL = "ws2_smoke"


def main():
    client = MilvusClient(uri=MILVUS_URI)
    assert MILVUS_VERSION in client.get_server_version()

    # Real WS1 data (first N corpus rows + first NQ GT queries): RaBitQ-style
    # quantizers behave pathologically on random gaussians (near-tied
    # neighbors), so smoke on the actual embedding distribution.
    from benchlib.config import DATA_DIR, slice_path
    vecs = np.array(np.load(slice_path("1m"), mmap_mode="r")[:N])
    queries = np.load(DATA_DIR / "queries_gt_emb.npy")[:NQ]
    gt = np.argsort(-(queries @ vecs.T), axis=1)[:, :K]

    # ---- FLAT: exactness + all measurement paths -------------------------
    ensure_fresh_collection(client, COLL, EMBED_DIM)
    t_ing = ingest(client, COLL, vecs, batch=2000)
    t_build = build_index(client, COLL, "FLAT", {})
    rss0 = milvus_rss_bytes()
    client.load_collection(COLL)

    pred = search_ids(client, COLL, queries, K, {}, batch=100)
    r = recall_at_k(pred, gt, K)
    print(f"FLAT ingest {t_ing:.1f}s build {t_build:.1f}s "
          f"recall@{K} vs numpy brute force: {r}")
    assert r == 1.0, f"FLAT recall must be exactly 1.0 on 10k, got {r}"

    m = measure_footprint(client, COLL, rss0)  # after workload: lazy materialization
    print(f"FLAT loaded {m['loaded_bytes']/1e6:.1f}MB "
          f"(index {m['index_files_bytes']/1e6:.1f} + chunks "
          f"{m['chunk_cache_bytes']/1e6:.1f} + heap {m['segment_heap_bytes']/1e6:.1f})")
    loaded = m["loaded_bytes"]
    expected = N * EMBED_DIM * 4
    assert expected * 0.9 < loaded < expected * 1.5, f"loaded_bytes implausible: {m}"

    lat = timed_sequential(client, COLL, queries, {}, n_warmup=20, n_timed=50)
    s = latency_summary(lat)
    assert 0 < s["p50_ms"] < 1000 and s["p50_ms"] <= s["p99_ms"]
    qps = concurrent_qps(client, COLL, queries, {}, threads=4, n_queries=200)
    print(f"FLAT p50 {s['p50_ms']:.2f}ms p99 {s['p99_ms']:.2f}ms qps(4thr) {qps:.0f}")
    assert qps > 0

    # ---- IVF_RABITQ + SQ8 refine: accepts refine_k at search time --------
    client.release_collection(COLL)
    ensure_fresh_collection(client, COLL, EMBED_DIM)
    ingest(client, COLL, vecs, batch=2000)
    t_build = build_index(client, COLL, "IVF_RABITQ",
                          {"nlist": 64, "refine": True, "refine_type": "SQ8"})
    rss0 = milvus_rss_bytes()
    client.load_collection(COLL)
    pred = search_ids(client, COLL, queries, K,
                      {"nprobe": 64, "refine_k": 2}, batch=100)
    r = recall_at_k(pred, gt, K)
    mr = measure_footprint(client, COLL, rss0)
    print(f"RABITQ+refine build {t_build:.1f}s loaded "
          f"{mr['loaded_bytes']/1e6:.1f}MB "
          f"(index {mr['index_files_bytes']/1e6:.1f} + chunks "
          f"{mr['chunk_cache_bytes']/1e6:.1f} + heap {mr['segment_heap_bytes']/1e6:.1f}) "
          f"recall@{K} (nprobe=64,refine_k=2): {r}")
    assert r > 0.8, f"RaBitQ+refine full-scan recall suspiciously low: {r}"
    # binary codes + SQ8 refine copy must be visible (~N*D*1.125 = 11.5MB)
    expected = N * EMBED_DIM * 1.125
    assert expected * 0.8 < mr["loaded_bytes"] < expected * 2.0, mr

    client.drop_collection(COLL)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
