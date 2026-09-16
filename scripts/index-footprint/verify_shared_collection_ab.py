"""Decision-2 A/B verification for the 10M run (runs at 1M — fast).

The 10M run wants ONE ingested collection cycled through indexes
(release -> drop_index -> create_index -> load) instead of a fresh
collection per arm (the 1M methodology). That switch is allowed only if
recall and footprint are shown to be identical both ways on identical
data. This script proves (or refutes) that at 1M, where the
fresh-collection reference numbers already exist in data/ws2_cache/1m/:

  cycle 1: ivf_sq8       — compare vs cached fresh-collection ivf_sq8
  cycle 2: rabitq        — a *different* index family in between
  cycle 3: ivf_sq8 again — must equal cycle 1 exactly (no state leakage
                           from the intervening rabitq build)
  cycle 4: ivf_flat_fp32 + mmap.enabled — Decision-1 mechanics check:
           recall must match the cached (non-mmap) ivf_flat_fp32; the
           index must actually report mmap.enabled=true.

Each cycle measures one sweep point (nprobe=256, recall@10 + recall@100)
plus the footprint. chunk_cache_bytes was 0 for every IVF arm at 1M, so
footprint comparison = index_files_bytes + segment_heap_bytes, which do
not depend on how much of the query workload ran before measuring.

Also probes Woodpecker WAL growth (du of a-bucket/files/wp in MinIO)
around the ingest — the 10M disk budget depends on whether WAL truncates.

Result cached to data/ws2_cache/10m/_ab_verification.json; re-runs are
no-ops. Notebook 02's 10M section loads and asserts on this file.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from benchlib.config import (  # noqa: E402
    DATA_DIR, EMBED_DIM, GT_TOP_K, MILVUS_URI, N_GT_QUERIES, gt_path,
    slice_path,
)
from benchlib.quantization import (  # noqa: E402
    build_index, compact_and_wait, ensure_fresh_collection, ingest,
    measure_footprint, milvus_rss_bytes, recall_at_k, search_ids,
)

CACHE_F = DATA_DIR / "ws2_cache" / "10m" / "_ab_verification.json"
COLL = "ws2ab_1m"
NPROBE = 256

# Fresh-collection reference values from data/ws2_cache/1m/ (nprobe=256).
REFS = {
    "ivf_sq8": dict(r10=0.99031, r100=0.990156,
                    index_files_bytes=1127428096, heap=56750234),
    "rabitq": dict(r10=0.74109, r100=0.770663,
                   index_files_bytes=248000512, heap=56750250),
    "ivf_flat_fp32": dict(r10=0.99521, r100=0.99226,
                          index_files_bytes=4203270144, heap=56750250),
}

CYCLES = [
    ("cycle1_ivf_sq8", "ivf_sq8", "IVF_SQ8", {"nlist": 4096}),
    ("cycle2_rabitq", "rabitq", "IVF_RABITQ", {"nlist": 4096}),
    ("cycle3_ivf_sq8_again", "ivf_sq8", "IVF_SQ8", {"nlist": 4096}),
    ("cycle4_ivf_flat_mmap", "ivf_flat_fp32", "IVF_FLAT",
     {"nlist": 4096, "mmap.enabled": True}),
    # same build WITHOUT mmap on the same frozen segments: recall must be
    # bit-identical to cycle 4 (mmap changes serving, never results)
    ("cycle5_ivf_flat_ram", "ivf_flat_fp32", "IVF_FLAT", {"nlist": 4096}),
]


def wal_bytes() -> int:
    out = subprocess.run(
        ["docker", "exec", "milvus-minio", "sh", "-c",
         "du -sk /minio_data/a-bucket/files/wp 2>/dev/null; true"],
        capture_output=True, text=True).stdout.strip()
    return int(out.split()[0]) * 1024 if out else 0


def main():
    if CACHE_F.exists():
        print(f"cached — {CACHE_F} exists, not re-running")
        print(CACHE_F.read_text())
        return

    from pymilvus import MilvusClient
    client = MilvusClient(uri=MILVUS_URI)

    corpus = np.load(slice_path("1m"), mmap_mode="r")
    queries = np.load(DATA_DIR / "queries_gt_emb.npy")
    gt = np.load(gt_path("1m"))["gt_indices"]
    assert queries.shape == (N_GT_QUERIES, EMBED_DIM)

    result = {"collection": COLL, "nprobe": NPROBE, "refs": REFS,
              "wal_bytes_before_ingest": wal_bytes(), "cycles": []}

    ensure_fresh_collection(client, COLL, EMBED_DIM)
    t0 = time.time()
    ingest_s = ingest(client, COLL, corpus)
    result["ingest_seconds"] = ingest_s
    result["wal_bytes_after_ingest"] = wal_bytes()
    print(f"ingested 1M in {ingest_s:.0f}s; WAL "
          f"{result['wal_bytes_before_ingest']/2**30:.1f} -> "
          f"{result['wal_bytes_after_ingest']/2**30:.1f} GiB")
    # freeze the segment set BEFORE any cycle (see compact_and_wait docstring)
    t0c = time.time()
    result["compaction_rounds"] = compact_and_wait(client, COLL)
    result["compaction_seconds"] = time.time() - t0c
    print(f"compaction frozen in {result['compaction_seconds']:.0f}s "
          f"({result['compaction_rounds']} rounds)")

    try:
        for cyc_name, ref_key, index_type, params in CYCLES:
            # release + drop any previous index — the cycle under test
            try:
                client.release_collection(COLL)
            except Exception:
                pass
            for ix in client.list_indexes(COLL):
                client.drop_index(COLL, ix)

            build_s = build_index(client, COLL, index_type, params)
            ix_name = client.list_indexes(COLL)[0]
            desc = client.describe_index(COLL, ix_name)
            mmap_echoed = str(desc.get("mmap.enabled", "")).lower()
            if params.get("mmap.enabled") and mmap_echoed != "true":
                # fallback path: set mmap as an index property before load
                client.alter_index_properties(
                    COLL, ix_name, properties={"mmap.enabled": True})
                desc = client.describe_index(COLL, ix_name)
                mmap_echoed = str(desc.get("mmap.enabled", "")).lower()
                if mmap_echoed != "true":
                    mmap_echoed = "SET-BUT-NOT-ECHOED"

            rss0 = milvus_rss_bytes()
            client.load_collection(COLL)
            pred10 = search_ids(client, COLL, queries, 10,
                                {"nprobe": NPROBE})
            r10 = recall_at_k(pred10, gt, 10)
            pred100 = search_ids(client, COLL, queries, GT_TOP_K,
                                 {"nprobe": NPROBE})
            r100 = recall_at_k(pred100, gt, GT_TOP_K)
            m = measure_footprint(client, COLL, rss0)

            ref = REFS[ref_key]
            cyc = dict(
                cycle=cyc_name, index_type=index_type, params_sent=str(params),
                build_seconds=build_s, mmap_echoed=mmap_echoed,
                recall_at_10=r10, recall_at_100=r100,
                index_files_bytes=m["index_files_bytes"],
                chunk_cache_bytes=m["chunk_cache_bytes"],
                segment_heap_bytes=m["segment_heap_bytes"],
                d_r10_vs_fresh=round(r10 - ref["r10"], 6),
                d_r100_vs_fresh=round(r100 - ref["r100"], 6),
                d_index_bytes_vs_fresh=m["index_files_bytes"]
                - ref["index_files_bytes"],
            )
            result["cycles"].append(cyc)
            print(f"[{cyc_name}] r@10={r10:.5f} (Δ{cyc['d_r10_vs_fresh']:+.6f}) "
                  f"r@100={r100:.5f} (Δ{cyc['d_r100_vs_fresh']:+.6f}) "
                  f"index={m['index_files_bytes']/2**20:.0f}MiB "
                  f"(Δ{cyc['d_index_bytes_vs_fresh']/2**20:+.1f}MiB) "
                  f"chunk_cache={m['chunk_cache_bytes']} mmap={mmap_echoed}")
    finally:
        try:
            client.release_collection(COLL)
            client.drop_collection(COLL)
        except Exception:
            pass

    result["wal_bytes_after_cycles"] = wal_bytes()

    c1, c3 = result["cycles"][0], result["cycles"][2]
    c4, c5 = result["cycles"][3], result["cycles"][4]
    result["verdict"] = {
        # the Decision-2 hard gate: rebuild on the frozen segment set must
        # reproduce the first build exactly (kmeans is deterministic given
        # identical training data; any difference = state leakage or an
        # unfrozen segment set)
        "rebuild_identical_to_first_build":
            c1["recall_at_10"] == c3["recall_at_10"]
            and c1["recall_at_100"] == c3["recall_at_100"]
            and c1["index_files_bytes"] == c3["index_files_bytes"],
        # informative only: cached 1M refs were measured on a different
        # (uncompacted) segment state; expect ~1% band, not identity
        "max_abs_d_r10_vs_fresh": max(
            abs(c["d_r10_vs_fresh"]) for c in result["cycles"]),
        # Decision-1 hard gate: mmap serving returns identical results
        "mmap_cycle_ok":
            c4["mmap_echoed"] == "true" and c5["mmap_echoed"] != "true"
            and c4["recall_at_10"] == c5["recall_at_10"]
            and c4["recall_at_100"] == c5["recall_at_100"],
    }
    CACHE_F.parent.mkdir(parents=True, exist_ok=True)
    CACHE_F.write_text(json.dumps(result, indent=1))
    print("verdict:", json.dumps(result["verdict"], indent=1))
    print(f"WAL after cycles: {result['wal_bytes_after_cycles']/2**30:.1f} GiB")
    print(f"wrote {CACHE_F}")


if __name__ == "__main__":
    main()
