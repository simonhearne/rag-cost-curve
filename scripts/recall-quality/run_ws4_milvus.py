#!/usr/bin/env python
"""WS4 Milvus side: build every retrieval arm at 10M and retrieve top-10 for
the WS4 question set. Also produces the WS3 winning-combo 10M sweeps.

Design: results/recall-vs-quality.md
(sections 4.4-4.6). Every phase is cached and resumable; gates raise.

    python scripts/recall-quality/run_ws4_milvus.py [phase ...]

Phases, in the order the spec runs them:
    sq8              rebuild IVF_SQ8 on ws2_10m_shared, gate G3, extend the
                     GT sweep to nprobe 1-8, retrieve WS4 at every nprobe
    pca_uc_512_sq8   truncation arms: transform, ingest, compact, build,
    pca_uc_384_sq8   gates G1/G2, WS3 10M sweep, WS4 retrieval
    mrl_512_sq8
    refine           rabitq+sq8 refine (refine_k=2) rebuild on shared
    pq               ivf_pq m=256 rebuild on shared
    rabitq           ivf_rabitq rebuild on shared

No phase writes to results/; the notebook does that from the caches.
"""

import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest  # noqa: E402
from benchlib.config import (  # noqa: E402
    DATA_DIR, EMBED_DIM, GT_TOP_K, MILVUS_URI, MILVUS_VERSION, N_GT_QUERIES,
    REPO_ROOT, SLICES, WS4_CEILING_1M, WS4_DISK_FLOOR_GB, WS4_GATE_G1_TOL,
    WS4_GATE_G2_BAND, WS4_GATE_G3_EXPECTED, WS4_GATE_G3_TOL,
    WS4_GATE_N_QUERIES, WS4_LOW_NPROBES, WS4_NLIST, WS4_PCA_ROTATION,
    WS4_QUANT_NPROBE, WS4_REFINE_NPROBE, WS4_SCALE, WS4_SHARED_COLLECTION,
    WS4_SQ8_NPROBES, WS4_TRUNC_ARMS, WS4_TRUNC_NPROBE, WS4_WS3_SWEEP,
    gt_path, slice_path, ws4_cache,
)
from benchlib.quantization import (  # noqa: E402
    build_index, compact_and_wait, concurrent_qps, ensure_fresh_collection,
    ingest, latency_summary, measure_footprint, milvus_rss_bytes, recall_at_k,
    search_ids, timed_sequential,
)
from benchlib.dimensionality import (  # noqa: E402
    load_pca, mrl_truncate, pca_project, transform_corpus,
)

N_ROWS = SLICES[WS4_SCALE]
CACHE = ws4_cache()
RET = CACHE / "retrieval"
WS3_CACHE_10 = DATA_DIR / "ws3_cache" / WS4_SCALE
DERIVED = DATA_DIR / "ws3_derived"
WS2_CACHE_10 = DATA_DIR / "ws2_cache" / WS4_SCALE
for d in (CACHE, RET, WS3_CACHE_10, DERIVED):
    d.mkdir(parents=True, exist_ok=True)

N_WARMUP, N_TIMED = 100, 500
QPS_THREADS, N_QPS = 8, 2000

# WS2 10M rows to reproduce on rebuild (arm, nprobe, mode, n) -> expected r@10
WS2_REPRO = {
    "refine": dict(arm="rabitq_refine_sq8_k2", nprobe=WS4_REFINE_NPROBE,
                   mode="separate", refine_k=2),
    "pq": dict(arm="ivf_pq_m256", nprobe=WS4_QUANT_NPROBE, mode="merged"),
    "rabitq": dict(arm="rabitq", nprobe=WS4_QUANT_NPROBE, mode="merged"),
}
REPRO_TOL = 0.003

QUANT_SPECS = {
    "refine": dict(index_type="IVF_RABITQ",
                   params={"nlist": WS4_NLIST, "refine": True,
                           "refine_type": "SQ8"},
                   search_extra={"refine_k": 2},
                   ws4_nprobes=(32, 64, 128)),
    "pq": dict(index_type="IVF_PQ",
               params={"nlist": WS4_NLIST, "m": 256, "nbits": 8},
               search_extra={}, ws4_nprobes=(64, 128, 256)),
    "rabitq": dict(index_type="IVF_RABITQ", params={"nlist": WS4_NLIST},
                   search_extra={}, ws4_nprobes=(64, 128, 256)),
}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def free_gb() -> float:
    return shutil.disk_usage(REPO_ROOT).free / 2**30


def disk_guard(stage: str):
    if free_gb() < WS4_DISK_FLOOR_GB:
        raise RuntimeError(
            f"free disk {free_gb():.0f} GiB below the {WS4_DISK_FLOOR_GB} GiB "
            f"floor before {stage} -- stopping before Milvus wedges the VM")


# ------------------------------------------------------------ fixtures ---

from pymilvus import MilvusClient  # noqa: E402

client = MilvusClient(uri=MILVUS_URI)
_ver = client.get_server_version()
assert MILVUS_VERSION in _ver, f"expected Milvus {MILVUS_VERSION}, got {_ver}"

queries_gt = np.load(DATA_DIR / "queries_gt_emb.npy")
assert queries_gt.shape == (N_GT_QUERIES, EMBED_DIM)
gtz = np.load(gt_path(WS4_SCALE))
gt_10 = gtz["gt_indices"]
ws4_gt = gtz["ws4_indices"]
queries_ws4 = np.load(DATA_DIR / "queries_ws4_emb.npy")
assert queries_ws4.shape[0] == ws4_gt.shape[0] == 3610
assert gt_10.max() < N_ROWS and ws4_gt.max() < N_ROWS
corpus = np.load(slice_path(WS4_SCALE), mmap_mode="r")
assert corpus.shape == (N_ROWS, EMBED_DIM)


def save_retrieval(key: str, ids, scores, meta: dict):
    """One npz per (arm, operating point) over ALL 3,610 WS4 queries; the
    notebook selects the main/sanity subsets. Recall vs ws4_indices is
    recorded here for the log only -- the notebook recomputes it."""
    r10 = recall_at_k(ids, ws4_gt, 10)
    np.savez(RET / f"{key}.npz", ids=ids, scores=scores,
             meta=json.dumps({**meta, "ws4_recall_at_10_all3610": r10,
                              "n_queries": int(ids.shape[0])}))
    log(f"  [{key}] WS4 retrieval saved; r@10 over all 3610 = {r10:.4f}")


def search_with_scores(coll, queries, limit, params, batch=1000):
    nq = len(queries)
    ids = np.full((nq, limit), -1, dtype=np.int64)
    sc = np.full((nq, limit), np.nan, dtype=np.float32)
    for s in range(0, nq, batch):
        block = np.asarray(queries[s:s + batch], dtype=np.float32)
        res = client.search(coll, data=block, anns_field="emb", limit=limit,
                            search_params={"params": params})
        for j, hits in enumerate(res):
            for r, h in enumerate(hits):
                ids[s + j, r] = h["id"]
                sc[s + j, r] = h["distance"]
    return ids, sc


def retrieve_ws4(coll, key, queries, params, meta):
    if (RET / f"{key}.npz").exists():
        log(f"  [{key}] retrieval cached")
        return
    t0 = time.time()
    ids, sc = search_with_scores(coll, queries, 10, params)
    save_retrieval(key, ids, sc, {**meta, "params": params,
                                  "seconds": round(time.time() - t0, 1)})


def _drop_all_indexes(coll):
    try:
        client.release_collection(coll)
    except Exception:
        pass
    for ix in client.list_indexes(coll):
        client.drop_index(coll, ix)


# ---------------------------------------------------------- phase: sq8 ---


def phase_sq8():
    coll = WS4_SHARED_COLLECTION
    n = int(client.get_collection_stats(coll)["row_count"])
    if n != N_ROWS:
        raise RuntimeError(f"{coll} has {n} rows, expected {N_ROWS}")
    done = all((RET / f"sq8_np{p}.npz").exists() for p in WS4_SQ8_NPROBES)
    done &= (CACHE / "gate_g3.json").exists() and (
        CACHE / "sq8_low_nprobe_sweep.json").exists()
    if done:
        log("[sq8] phase complete (cached)")
        return
    disk_guard("sq8 build")
    build_f = CACHE / "shared_sq8_build.json"
    if not build_f.exists():
        _drop_all_indexes(coll)
        log("[sq8] building IVF_SQ8 on the shared collection (~2.7 h)")
        bs = build_index(client, coll, "IVF_SQ8", {"nlist": WS4_NLIST})
        build_f.write_text(json.dumps({"build_seconds": bs}))
        log(f"[sq8] built in {bs:.0f}s")
    client.load_collection(coll)

    # G3: reproduce the WS2 10M curve at nprobe=512 on all 10k GT queries.
    g3 = CACHE / "gate_g3.json"
    if not g3.exists():
        t0 = time.time()
        pred = search_ids(client, coll, queries_gt, 10, {"nprobe": 512})
        r10 = recall_at_k(pred, gt_10, 10)
        delta = r10 - WS4_GATE_G3_EXPECTED
        passed = abs(delta) <= WS4_GATE_G3_TOL
        g3.write_text(json.dumps(dict(
            measured_recall_at_10=r10, expected=WS4_GATE_G3_EXPECTED,
            delta=delta, tolerance=WS4_GATE_G3_TOL, nprobe=512,
            n_queries=N_GT_QUERIES, passed=passed,
            seconds=time.time() - t0), indent=1))
        log(f"[sq8] G3: r@10={r10:.5f} vs {WS4_GATE_G3_EXPECTED} "
            f"(delta {delta:+.5f}, tol {WS4_GATE_G3_TOL}) -> "
            f"{'PASS' if passed else 'FAIL'}")
        if not passed:
            raise RuntimeError("G3 FAILED: the shared collection does not "
                               "reproduce WS2's frozen-set SQ8 curve; no "
                               "1024d arm can be trusted. Stopping.")

    # Extend the GT sweep below nprobe=16 (merged limit=100 pass, 10k queries).
    low = CACHE / "sq8_low_nprobe_sweep.json"
    if not low.exists():
        rows = []
        for p in WS4_LOW_NPROBES:
            t0 = time.time()
            pred100 = search_ids(client, coll, queries_gt, GT_TOP_K,
                                 {"nprobe": p})
            rows.append(dict(scale=WS4_SCALE, arm="ivf_sq8",
                             sweep_param="nprobe", sweep_value=p,
                             recall_at_10=recall_at_k(pred100, gt_10, 10),
                             recall_at_100=recall_at_k(pred100, gt_10,
                                                       GT_TOP_K),
                             n_recall_queries=N_GT_QUERIES,
                             recall_pass_mode="merged",
                             seconds=round(time.time() - t0, 1)))
            log(f"[sq8] GT sweep nprobe={p}: r@10={rows[-1]['recall_at_10']:.4f}")
        low.write_text(json.dumps(rows, indent=1))

    for p in WS4_SQ8_NPROBES:
        retrieve_ws4(coll, f"sq8_np{p}", queries_ws4, {"nprobe": p},
                     dict(arm=f"sq8_np{p}", family="pruning" if p < 512
                          else "reference", index="IVF_SQ8",
                          collection=coll, dim=EMBED_DIM))
    _drop_all_indexes(coll)
    log("[sq8] phase complete; index dropped")


# ------------------------------------------------- phase: truncation arm ---

_pca_model = None


def transform_fn(transform: str, d: int):
    global _pca_model
    if transform == "mrl":
        return lambda b: mrl_truncate(b, d)
    if _pca_model is None:
        _pca_model = load_pca(DERIVED / WS4_PCA_ROTATION)
        assert not _pca_model["centered"], "WS4 pins the UNCENTERED rotation"
    return lambda b: pca_project(b, _pca_model, d)


def exact_topk_ip(q_path_or_arr, c_path, k=10, chunk=200_000, q_batch=1000):
    """Exact top-k inner product in the projected space (the WS3 ceiling
    probe, blocked on both axes so the score tile stays ~0.8 GB)."""
    import torch
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    src = np.load(c_path, mmap_mode="r")
    q_all = (np.load(q_path_or_arr) if isinstance(q_path_or_arr, (str, Path))
             else np.asarray(q_path_or_arr))
    out = np.empty((q_all.shape[0], k), dtype=np.int64)
    for q0 in range(0, q_all.shape[0], q_batch):
        qs = torch.from_numpy(np.ascontiguousarray(q_all[q0:q0 + q_batch])).to(dev)
        best_v = torch.zeros((qs.shape[0], 0), device=dev)
        best_i = torch.zeros((qs.shape[0], 0), dtype=torch.int64, device=dev)
        for s in range(0, src.shape[0], chunk):
            block = torch.from_numpy(np.ascontiguousarray(src[s:s + chunk])).to(dev)
            v, i = torch.topk(qs @ block.T, min(k, block.shape[0]), dim=1)
            best_v = torch.cat([best_v, v], dim=1)
            best_i = torch.cat([best_i, i + s], dim=1)
            v, sel = torch.topk(best_v, min(k, best_v.shape[1]), dim=1)
            best_v, best_i = v, torch.gather(best_i, 1, sel)
            del block
        out[q0:q0 + q_batch] = best_i.cpu().numpy()
    return out


def phase_trunc(arm: str):
    transform, d = WS4_TRUNC_ARMS[arm]
    cache_f = WS3_CACHE_10 / f"{arm}.json"
    ret_done = (RET / f"{arm}_np{WS4_TRUNC_NPROBE}.npz").exists()
    if cache_f.exists() and ret_done:
        log(f"[{arm}] phase complete (cached)")
        return
    if (WS3_CACHE_10 / f"{arm}.failed.json").exists():
        log(f"[{arm}] previously FAILED -- delete the .failed.json to retry")
        return
    fn = transform_fn(transform, d)

    # 1. derived corpus + query sets (identical transform function)
    c_path = DERIVED / f"corpus_{WS4_SCALE}_{transform}_{d}.npy"
    q_gt_path = DERIVED / f"queries_{transform}_{d}.npy"   # from WS3 (same fn)
    q_ws4_path = DERIVED / f"queries_ws4_{transform}_{d}.npy"
    disk_guard(f"{arm} transform")
    if not c_path.exists():
        t0 = time.time()
        log(f"[{arm}] transforming the 10M corpus -> {c_path.name}")
        transform_corpus(corpus, c_path, fn)
        log(f"[{arm}] corpus transformed in {time.time() - t0:.0f}s; hashing")
        manifest.record(c_path, {"kind": "ws3_corpus", "transform": transform,
                                 "dim": d, "scale": WS4_SCALE,
                                 "rotation": WS4_PCA_ROTATION
                                 if transform != "mrl" else None})
    if not q_gt_path.exists():
        np.save(q_gt_path, fn(queries_gt))
        manifest.record(q_gt_path, {"kind": "ws3_queries", "transform":
                                    transform, "dim": d})
    if not q_ws4_path.exists():
        np.save(q_ws4_path, fn(queries_ws4))
        manifest.record(q_ws4_path, {"kind": "ws4_queries", "transform":
                                     transform, "dim": d})
    q_gt_t = np.load(q_gt_path)
    q_ws4_t = np.load(q_ws4_path)
    # the WS3 GT-query file was written by the 1M notebook with the same fn;
    # assert that here rather than trust it
    chk = fn(queries_gt[:64])
    assert np.allclose(chk, q_gt_t[:64], atol=1e-5), (
        f"{q_gt_path.name} was not produced by the pinned transform")

    # 2. G1 offline ceiling on the first N GT queries against the full 10M
    g1_f = CACHE / f"gate_g1_{arm}.json"
    if not g1_f.exists():
        t0 = time.time()
        pred = exact_topk_ip(q_gt_t[:WS4_GATE_N_QUERIES], c_path, 10)
        ceil = recall_at_k(pred, gt_10[:WS4_GATE_N_QUERIES], 10)
        ref = WS4_CEILING_1M[arm]
        passed = abs(ceil - ref) <= WS4_GATE_G1_TOL
        g1_f.write_text(json.dumps(dict(
            arm=arm, ceiling_10m_first_n=ceil, ceiling_1m=ref,
            delta=ceil - ref, tolerance=WS4_GATE_G1_TOL,
            n_queries=WS4_GATE_N_QUERIES, passed=passed,
            seconds=time.time() - t0), indent=1))
        log(f"[{arm}] G1 ceiling@10M (n={WS4_GATE_N_QUERIES}) = {ceil:.4f} "
            f"vs 1M {ref:.4f} -> {'PASS' if passed else 'FAIL'}")
        if not passed:
            raise RuntimeError(f"G1 FAILED for {arm}: transform plumbing is "
                               "wrong at 10M. Stopping.")
    ceiling = json.loads(g1_f.read_text())["ceiling_10m_first_n"]

    # 3. fresh collection: ingest, compact once, build, load
    coll = f"ws3_{WS4_SCALE}_{arm}"
    t_arm = time.time()
    try:
        ing_f = CACHE / f"ingest_{arm}.json"
        if ing_f.exists() and client.has_collection(coll) and int(
                client.get_collection_stats(coll)["row_count"]) == N_ROWS:
            info = json.loads(ing_f.read_text())
            log(f"[{arm}] collection present ({N_ROWS} rows), reusing")
        else:
            disk_guard(f"{arm} ingest")
            vecs = np.load(c_path, mmap_mode="r")
            assert vecs.shape == (N_ROWS, d)
            ensure_fresh_collection(client, coll, d)
            log(f"[{arm}] ingesting {N_ROWS} rows at {d}d")
            ingest_s = ingest(client, coll, vecs)
            n = int(client.get_collection_stats(coll)["row_count"])
            assert n == N_ROWS, f"ingested {n} != {N_ROWS}"
            t0c = time.time()
            rounds = compact_and_wait(client, coll)
            info = dict(ingest_seconds=ingest_s, compaction_rounds=rounds,
                        compaction_seconds=time.time() - t0c, n_rows=n)
            ing_f.write_text(json.dumps(info, indent=1))
            log(f"[{arm}] ingested in {ingest_s:.0f}s, compacted in "
                f"{info['compaction_seconds']:.0f}s")
        build_f = CACHE / f"build_{arm}.json"
        if build_f.exists() and client.list_indexes(coll):
            build_s = json.loads(build_f.read_text())["build_seconds"]
            log(f"[{arm}] index present, reusing (build was {build_s:.0f}s)")
        else:
            disk_guard(f"{arm} build")
            _drop_all_indexes(coll)
            log(f"[{arm}] building IVF_SQ8 nlist={WS4_NLIST}")
            build_s = build_index(client, coll, "IVF_SQ8",
                                  {"nlist": WS4_NLIST})
            build_f.write_text(json.dumps({"build_seconds": build_s}))
            log(f"[{arm}] built in {build_s:.0f}s")
        rss0 = milvus_rss_bytes()
        client.load_collection(coll)

        # 4. G2: Milvus arm vs its own offline ceiling on the same queries
        g2_f = CACHE / f"gate_g2_{arm}.json"
        if not g2_f.exists():
            pred = search_ids(client, coll, q_gt_t[:WS4_GATE_N_QUERIES], 10,
                              {"nprobe": WS4_TRUNC_NPROBE})
            r10 = recall_at_k(pred, gt_10[:WS4_GATE_N_QUERIES], 10)
            lo, hi = WS4_GATE_G2_BAND
            passed = (ceiling + lo) <= r10 <= (ceiling + hi)
            g2_f.write_text(json.dumps(dict(
                arm=arm, milvus_recall_at_10=r10, ceiling=ceiling,
                delta=r10 - ceiling, band=WS4_GATE_G2_BAND,
                nprobe=WS4_TRUNC_NPROBE, n_queries=WS4_GATE_N_QUERIES,
                passed=passed), indent=1))
            log(f"[{arm}] G2 milvus@512 = {r10:.4f} vs ceiling {ceiling:.4f} "
                f"-> {'PASS' if passed else 'FAIL'}")
            if not passed:
                raise RuntimeError(f"G2 FAILED for {arm}. Stopping.")

        # 5. WS3 10M sweep (WS2-10M protocol, merged limit=100 pass)
        if not cache_f.exists():
            rows = []
            for nprobe in WS4_WS3_SWEEP:
                params = {"nprobe": nprobe}
                pred100 = search_ids(client, coll, q_gt_t, GT_TOP_K, params)
                r10 = recall_at_k(pred100, gt_10, 10)
                r100 = recall_at_k(pred100, gt_10, GT_TOP_K)
                lat = latency_summary(timed_sequential(
                    client, coll, q_gt_t, params, N_WARMUP, N_TIMED, limit=10))
                qps = concurrent_qps(client, coll, q_gt_t, params,
                                     QPS_THREADS, N_QPS, limit=10)
                rows.append(dict(
                    scale=WS4_SCALE, arm=arm, transform=transform, dim=d,
                    quantizer="sq8", index_type="IVF_SQ8",
                    sweep_param="nprobe", sweep_value=nprobe, refine_k=None,
                    recall_at_10=r10, recall_at_100=r100,
                    p50_ms=lat["p50_ms"], p99_ms=lat["p99_ms"], qps=qps,
                    qps_threads=QPS_THREADS, build_seconds=build_s,
                    ingest_seconds=info["ingest_seconds"],
                    compaction_seconds=info["compaction_seconds"],
                    nominal_payload_bytes=d, n_rows=N_ROWS,
                    n_recall_queries=N_GT_QUERIES, r100_n_queries=N_GT_QUERIES,
                    recall_pass_mode="merged", latency_measured=True,
                    qps_n_queries=N_QPS, mmap=False,
                    ingest_strategy="fresh_collection_compacted_once"))
                log(f"[{arm}] nprobe={nprobe}: r@10={r10:.4f} r@100={r100:.4f} "
                    f"p50={lat['p50_ms']:.1f}ms qps={qps:.0f}")
                if free_gb() < WS4_DISK_FLOOR_GB:
                    raise RuntimeError("disk below floor mid-sweep")
            m = measure_footprint(client, coll, rss0)
            for row in rows:
                row.update({k: m[k] for k in (
                    "loaded_bytes", "index_files_bytes", "chunk_cache_bytes",
                    "segment_heap_bytes")})
            cache_f.write_text(json.dumps({
                "arm": arm, "rows": rows, "measure": m,
                "arm_wall_seconds": time.time() - t_arm,
                "build_seconds": build_s, "ceiling_10m_first_n": ceiling,
                "disk_free_bytes": shutil.disk_usage(REPO_ROOT).free},
                indent=1))
            log(f"[{arm}] sweep cached; loaded {m['loaded_bytes']/2**20:.0f} MiB")

        # 6. WS4 retrieval at the ceiling point (+ a few cheaper points)
        for nprobe in (64, 128, 256, WS4_TRUNC_NPROBE):
            retrieve_ws4(coll, f"{arm}_np{nprobe}", q_ws4_t,
                         {"nprobe": nprobe},
                         dict(arm=arm, family="truncation", index="IVF_SQ8",
                              collection=coll, dim=d, transform=transform))
    except Exception as e:
        (WS3_CACHE_10 / f"{arm}.failed.json").write_text(json.dumps(
            {"arm": arm, "error": repr(e),
             "traceback": traceback.format_exc()}, indent=1))
        log(f"[{arm}] FAILED -- recorded: {e!r}")
        raise
    finally:
        try:
            client.release_collection(coll)  # collection kept for re-search
        except Exception:
            pass
        log(f"[{arm}] wall {time.time() - t_arm:.0f}s; disk free {free_gb():.0f} GiB")


# ------------------------------------------- phase: quantizer rebuilds ---


def phase_quant(name: str):
    spec = QUANT_SPECS[name]
    coll = WS4_SHARED_COLLECTION
    keys = [f"{name}_np{p}" for p in spec["ws4_nprobes"]]
    repro_f = CACHE / f"repro_{name}.json"
    if all((RET / f"{k}.npz").exists() for k in keys) and repro_f.exists():
        log(f"[{name}] phase complete (cached)")
        return
    disk_guard(f"{name} build")
    _drop_all_indexes(coll)
    log(f"[{name}] building {spec['index_type']} {spec['params']} on shared")
    bs = build_index(client, coll, spec["index_type"], spec["params"])
    log(f"[{name}] built in {bs:.0f}s")
    client.load_collection(coll)
    try:
        # reproduction gate against the committed WS2 10M row
        if not repro_f.exists():
            r = WS2_REPRO[name]
            curves = pd.read_csv(REPO_ROOT / "results/ws2/curves_10m.csv")
            row = curves[(curves.arm == r["arm"])
                         & (curves.sweep_value == r["nprobe"])].iloc[0]
            n = int(row.n_recall_queries)
            params = {"nprobe": r["nprobe"], **spec["search_extra"]}
            if r["mode"] == "merged":
                pred = search_ids(client, coll, queries_gt[:n], GT_TOP_K, params)
            else:
                pred = search_ids(client, coll, queries_gt[:n], 10, params)
            r10 = recall_at_k(pred, gt_10[:n], 10)
            delta = r10 - float(row.recall_at_10)
            passed = abs(delta) <= REPRO_TOL
            repro_f.write_text(json.dumps(dict(
                arm=r["arm"], nprobe=r["nprobe"], n_queries=n,
                measured=r10, expected=float(row.recall_at_10), delta=delta,
                tolerance=REPRO_TOL, passed=passed, build_seconds=bs),
                indent=1))
            log(f"[{name}] repro {r['arm']}@{r['nprobe']} (n={n}): "
                f"{r10:.5f} vs {row.recall_at_10:.5f} -> "
                f"{'PASS' if passed else 'FAIL'}")
            if not passed:
                raise RuntimeError(f"{name} rebuild does not reproduce WS2 "
                                   "10M; frozen set suspect. Stopping.")
        fam = "pruning" if name == "refine" else "quantization"
        for p in spec["ws4_nprobes"]:
            retrieve_ws4(coll, f"{name}_np{p}", queries_ws4,
                         {"nprobe": p, **spec["search_extra"]},
                         dict(arm=f"{name}_np{p}", family=fam,
                              index=spec["index_type"], collection=coll,
                              dim=EMBED_DIM, refine_k=spec["search_extra"]
                              .get("refine_k")))
    finally:
        _drop_all_indexes(coll)
        log(f"[{name}] index dropped; disk free {free_gb():.0f} GiB")


# ---------------------------------------------------------------- main ---

PHASES = {"sq8": phase_sq8, "refine": lambda: phase_quant("refine"),
          "pq": lambda: phase_quant("pq"),
          "rabitq": lambda: phase_quant("rabitq"),
          **{a: (lambda a=a: phase_trunc(a)) for a in WS4_TRUNC_ARMS}}
DEFAULT_ORDER = ["sq8", *WS4_TRUNC_ARMS, "refine", "pq", "rabitq"]

if __name__ == "__main__":
    order = sys.argv[1:] or DEFAULT_ORDER
    unknown = [p for p in order if p not in PHASES]
    if unknown:
        sys.exit(f"unknown phase(s) {unknown}; valid: {list(PHASES)}")
    log(f"WS4 Milvus run: phases {order}; disk free {free_gb():.0f} GiB")
    t_all = time.time()
    for p in order:
        t0 = time.time()
        log(f"=== phase {p} ===")
        PHASES[p]()
        log(f"=== phase {p} done in {(time.time() - t0)/3600:.2f} h ===")
    log(f"all phases done in {(time.time() - t_all)/3600:.2f} h")
