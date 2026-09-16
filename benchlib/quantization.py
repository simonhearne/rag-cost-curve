"""WS2 measurement functions: recall, latency, compression, sweep selection.

Pure functions here are unit-tested in tests/test_quantization.py. Milvus runner
helpers (below the pure section) are exercised by
scripts/index-footprint/smoke_ws2_runner.py against a live Milvus before any benchmark run.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd


# ----------------------------------------------------------- pure functions


def recall_at_k(pred_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """Mean per-query |pred[:k] ∩ gt[:k]| / k. Order within top-k ignored."""
    pred_ids = np.asarray(pred_ids)
    gt_ids = np.asarray(gt_ids)
    if pred_ids.shape[0] != gt_ids.shape[0]:
        raise ValueError(
            f"query count mismatch: pred {pred_ids.shape[0]} vs gt {gt_ids.shape[0]}")
    hits = 0
    for p, g in zip(pred_ids[:, :k], gt_ids[:, :k]):
        hits += len(np.intersect1d(p, g))
    return hits / (pred_ids.shape[0] * k)


def latency_summary(samples_ms) -> dict:
    """p50/p99 of per-query latencies (ms), linear interpolation."""
    a = np.asarray(list(samples_ms), dtype=np.float64)
    if a.size == 0:
        raise ValueError("no latency samples")
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p99_ms": float(np.percentile(a, 99)),
    }


def payload_compression(nominal_payload_bytes: int, dim: int) -> float:
    """Compression of the per-vector index payload vs fp32 (4*dim bytes)."""
    return (4 * dim) / nominal_payload_bytes


def pick_best_sweep_points(df: pd.DataFrame, targets) -> pd.DataFrame:
    """Best operating point per (arm, recall target at k=10).

    Qualifying rows have recall_at_10 >= target; among them pick max qps
    (tie: lowest sweep_value). If no row qualifies, emit the arm's
    max-recall row with met_target=False.
    """
    out = []
    for target in targets:
        for arm, g in df.groupby("arm", sort=False):
            ok = g[g.recall_at_10 >= target]
            if len(ok):
                best = ok.sort_values(
                    ["qps", "sweep_value"], ascending=[False, True]).iloc[0]
                met = True
            else:
                best = g.sort_values(
                    ["recall_at_10", "qps"], ascending=[False, False]).iloc[0]
                met = False
            row = best.to_dict()
            row["target"] = target
            row["met_target"] = met
            out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------ Milvus runner helpers
# Imported lazily so the pure functions stay usable without pymilvus.

VECTOR_FIELD = "emb"


def ensure_fresh_collection(client, name: str, dim: int) -> None:
    """Drop `name` if it exists, then create it: id INT64 pk + fp32 vector."""
    from pymilvus import DataType

    if client.has_collection(name):
        client.drop_collection(name)
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=dim)
    client.create_collection(name, schema=schema)


def ingest(client, name: str, vecs, batch: int = 5000) -> float:
    """Insert rows in memmap order (PK = row index), flush. Returns seconds."""
    t0 = time.perf_counter()
    n = len(vecs)
    for s in range(0, n, batch):
        block = np.asarray(vecs[s:s + batch], dtype=np.float32)
        rows = [{"id": s + j, VECTOR_FIELD: block[j]} for j in range(len(block))]
        client.insert(name, rows)
    client.flush(name)
    return time.perf_counter() - t0


def compact_and_wait(client, name: str, settle_s: float = 5.0) -> int:
    """Force compaction to completion so the sealed-segment set is frozen.

    Background compaction between index builds changes the segment set
    (fewer, larger segments -> different per-segment kmeans training),
    shifting recall at fixed nprobe by up to ~1% — observed directly when a
    shared-collection A/B's sq8 rebuild failed to reproduce its own first
    build (0.9825 vs 0.9927 r@10). Every arm must be built against one
    frozen segment set: call this once after ingest+flush, never write to
    the collection afterwards. ONE round only — measured at 10M, a manual
    compaction round is not a no-op even on an already-compacted set: it
    rewrites ~all segment binlogs (~41 GB) and pushes ~10 GB through the
    WAL, and the superseded generation lingers until GC (~1 h). The freeze
    property comes from "no writes after compaction", not from repetition.
    Returns rounds executed.
    """
    rounds = 0
    for _ in range(1):
        job = client.compact(name)
        rounds += 1
        if job in (None, -1):  # nothing to compact
            break
        while True:
            state = str(client.get_compaction_state(job))
            if "Completed" in state:
                break
            if "Failed" in state:
                raise RuntimeError(f"compaction failed: {state}")
            time.sleep(2)
        time.sleep(settle_s)
    return rounds


def build_index(client, name: str, index_type: str, params: dict,
                metric: str = "IP", poll_s: float = 2.0) -> float:
    """Create the index and block until fully built. Returns build seconds
    (create_index call through last pending row indexed, flush excluded)."""
    ip = client.prepare_index_params()
    ip.add_index(field_name=VECTOR_FIELD, index_type=index_type,
                 metric_type=metric, params=params)
    t0 = time.perf_counter()
    client.create_index(name, ip)
    index_name = client.list_indexes(name)[0]
    while True:
        d = client.describe_index(name, index_name)
        pending = int(d.get("pending_index_rows", 0) or 0)
        total = int(d.get("total_rows", 0) or 0)
        indexed = int(d.get("indexed_rows", 0) or 0)
        state = str(d.get("state", ""))
        if state == "Failed":
            raise RuntimeError(f"index build failed for {name}: {d}")
        if pending == 0 and total > 0 and indexed >= total:
            return time.perf_counter() - t0
        time.sleep(poll_s)


MEASURE_API = (
    "loaded_bytes = index_files_bytes + chunk_cache_bytes + "
    "segment_heap_bytes, all reported by Milvus itself, measured AFTER the "
    "arm's full query workload (v2.6 materializes serving state lazily). "
    "index_files_bytes: on-disk size (du inside the MinIO container) of "
    "files/index_files/{build_id} for exactly the build_ids referenced by "
    "this collection's currently loaded sealed segments (GET "
    ":9091/api/v1/_qn/segments) — i.e. the index files the query node "
    "deserializes to serve. NOT the datacoord stored_index_files_size "
    "metric, which also counts stale pre-compaction generations until GC "
    "(~2x overcount; kept as an audit field). 0 for FLAT, which has no "
    "index. chunk_cache_bytes: total size of this collection's segment "
    "chunk files under /var/lib/milvus/data/cache in the standalone "
    "container — raw columns mmapped for serving (how FLAT vectors are "
    "served). segment_heap_bytes: sum(mem_size) over the collection's "
    "sealed segments from GET :9091/api/v1/_qn/segments (pk/stats heap; "
    "v2.6 does not count index or vector bytes here). Process-RSS deltas "
    "are recorded for audit only — allocator retention makes them 2-3x "
    "noisy."
)


_RSS_CMD = (
    "for d in /proc/[0-9]*; do "
    '[ "$(cat $d/comm 2>/dev/null)" = milvus ] && grep VmRSS $d/status; '
    "done; true"
)


RSS_UNAVAILABLE = -1
DOCKER_ATTEMPTS = 3


def docker_exec(container: str, script: str, attempts: int = DOCKER_ATTEMPTS,
                retry_wait_s: float = 2.0) -> str:
    """`docker exec <container> sh -c <script>`, retried on transient failure.

    Docker Desktop intermittently kills exec processes under load — seen twice
    during the WS3 matrix as CalledProcessError(-5), each time discarding a
    fully-measured arm. Every read issued through here is deterministic and
    side-effect free, so retrying cannot change a result. Exhausting the
    attempts re-raises: these reads produce the measured footprint, and a
    default would be a fabricated number.
    """
    import subprocess

    for attempt in range(attempts):
        try:
            return subprocess.run(
                ["docker", "exec", container, "sh", "-c", script],
                capture_output=True, text=True, check=True).stdout
        except (subprocess.SubprocessError, OSError):
            if attempt == attempts - 1:
                raise
            time.sleep(retry_wait_s)
    raise RuntimeError("unreachable")


def milvus_rss_bytes(container: str = "milvus-standalone",
                     samples: int = 3, interval_s: float = 1.0,
                     attempts: int = 3) -> int:
    """Median VmRSS of the milvus server process inside the container
    (found by comm name — PID 1 is tini, not milvus).

    Returns RSS_UNAVAILABLE (-1) rather than raising if `docker exec` cannot
    be completed. RSS is an audit-only field (allocator retention makes it
    2-3x noisy, so no reported number depends on it) and a transient docker
    hiccup must never destroy an arm that has already run its workload —
    observed during the WS3 matrix, where a `docker exec` killed by signal 5
    lost a completed 20-minute arm.
    """
    import subprocess

    for attempt in range(attempts):
        try:
            vals = []
            for i in range(samples):
                if i:
                    time.sleep(interval_s)
                out = docker_exec(container, _RSS_CMD, attempts=1,
                                  retry_wait_s=0)
                vals.append(int(out.split()[1]) * 1024)  # kB -> bytes
            return int(np.median(vals))
        except (subprocess.SubprocessError, OSError, ValueError, IndexError):
            if attempt == attempts - 1:
                return RSS_UNAVAILABLE
            time.sleep(interval_s)
    return RSS_UNAVAILABLE


def _index_files_bytes(build_ids,
                       minio_container: str = "milvus-minio") -> int:
    """On-disk bytes of the index files for exactly these build ids
    (du inside the MinIO container)."""
    if not build_ids:
        return 0
    paths = " ".join(
        f"/minio_data/a-bucket/files/index_files/{b}" for b in sorted(build_ids))
    out = docker_exec(minio_container, f"du -sk {paths} 2>/dev/null; true")
    return sum(int(line.split()[0]) * 1024
               for line in out.splitlines() if line.strip())


def _stored_index_metric_bytes(cid: str) -> int:
    """milvus_datacoord_stored_index_files_size for this collection id.
    Audit only: includes stale pre-compaction index generations until GC."""
    import re
    import urllib.request

    metrics = urllib.request.urlopen(
        "http://localhost:9091/metrics").read().decode()
    pat = re.compile(
        r'^milvus_datacoord_stored_index_files_size\{[^}]*'
        r'collection_id="' + re.escape(cid) + r'"[^}]*\}\s+([0-9.e+]+)',
        re.M)
    m = pat.search(metrics)
    return int(float(m.group(1))) if m else 0


def measure_footprint(client, name: str, rss_baseline: int,
                      container: str = "milvus-standalone") -> dict:
    """Measure a loaded collection's serving footprint. Call AFTER running
    the arm's query workload (index memory and chunk pages materialize
    lazily, so measuring at load time undercounts). `rss_baseline` from
    milvus_rss_bytes() just before load_collection is recorded for audit.
    See MEASURE_API for exactly what loaded_bytes counts.
    """
    import json
    import subprocess
    import urllib.request

    rss_after = milvus_rss_bytes(container)
    cid = str(client.describe_collection(name)["collection_id"])
    segs = json.load(urllib.request.urlopen(
        "http://localhost:9091/api/v1/_qn/segments")) or []
    mine = [s for s in segs if str(s["collection_id"]) == cid]
    heap = sum(int(s["mem_size"]) for s in mine)
    seg_ids = {str(s["segment_id"]) for s in mine}
    build_ids = {str(f["build_id"]) for s in mine
                 for f in s.get("index_fields", [])}

    listing = docker_exec(
        container,
        r"find /var/lib/milvus/data/cache -type f -exec stat -c '%s %n' {} + "
        r"2>/dev/null; true")
    chunk_bytes = 0
    for line in listing.splitlines():
        size, _, path = line.partition(" ")
        if any(f"seg_{sid}_" in path for sid in seg_ids):
            chunk_bytes += int(size)
    index_bytes = _index_files_bytes(build_ids)
    return {
        "loaded_bytes": index_bytes + chunk_bytes + heap,
        "index_files_bytes": index_bytes,
        "chunk_cache_bytes": chunk_bytes,
        "segment_heap_bytes": heap,
        "index_build_ids": sorted(build_ids),
        "stored_index_metric_bytes": _stored_index_metric_bytes(cid),
        "rss_before_bytes": rss_baseline,
        "rss_after_bytes": rss_after,
        "api_used": MEASURE_API,
    }


def _hit_ids(hits, limit: int) -> list[int]:
    ids = [h["id"] for h in hits]
    return ids + [-1] * (limit - len(ids))


def search_ids(client, name: str, queries, limit: int, params: dict,
               batch: int = 1000) -> np.ndarray:
    """Batched search over all queries; returns (nq, limit) int64 ids,
    -1-padded if Milvus returns fewer than `limit` hits."""
    nq = len(queries)
    out = np.full((nq, limit), -1, dtype=np.int64)
    for s in range(0, nq, batch):
        block = np.asarray(queries[s:s + batch], dtype=np.float32)
        res = client.search(name, data=block, anns_field=VECTOR_FIELD,
                            limit=limit, search_params={"params": params})
        for j, hits in enumerate(res):
            out[s + j] = _hit_ids(hits, limit)
    return out


def timed_sequential(client, name: str, queries, params: dict,
                     n_warmup: int = 100, n_timed: int = 500,
                     limit: int = 10) -> list[float]:
    """Per-query wall-clock ms for single-threaded, one-query-per-request
    searches: n_warmup untimed, then n_timed timed (distinct queries)."""
    assert len(queries) >= n_warmup + n_timed
    sp = {"params": params}
    for q in queries[:n_warmup]:
        client.search(name, data=[np.asarray(q, dtype=np.float32)],
                      anns_field=VECTOR_FIELD, limit=limit, search_params=sp)
    out = []
    for q in queries[n_warmup:n_warmup + n_timed]:
        t0 = time.perf_counter()
        client.search(name, data=[np.asarray(q, dtype=np.float32)],
                      anns_field=VECTOR_FIELD, limit=limit, search_params=sp)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def concurrent_qps(client, name: str, queries, params: dict,
                   threads: int = 8, n_queries: int = 2000,
                   limit: int = 10) -> float:
    """Throughput from `threads` workers issuing one-query requests until
    n_queries total are done. Separate run from latency — never derived."""
    sp = {"params": params}
    qs = [np.asarray(queries[i % len(queries)], dtype=np.float32)
          for i in range(n_queries)]

    def one(q):
        client.search(name, data=[q], anns_field=VECTOR_FIELD,
                      limit=limit, search_params=sp)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(one, qs))
    return n_queries / (time.perf_counter() - t0)
