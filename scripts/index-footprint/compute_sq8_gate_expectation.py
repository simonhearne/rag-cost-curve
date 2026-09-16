"""Offline GT-based expectation for the 10M correctness gate.

At 1M the gate was a FLAT arm (recall 1.0 vs exact GT). At 10M, fp32 FLAT
does not fit in the Docker VM (40.96 GB raw vs 31.3 GiB), so the gate is:
run ivf_sq8 at nprobe = nlist = 4096 (full scan — IVF cluster pruning
disabled, so the only recall loss left is SQ8 quantization itself) on a
fixed query subset, and require agreement within ±0.002 of the expectation
computed HERE, offline, against the exact WS1 ground truth.

Expectation = recall@10 of exact fp32-query x SQ8-dequantized-corpus inner
product (asymmetric distance, like faiss/knowhere IVF_SQ8 QT_8bit):
  per-dim min/max over the full 10M slice, 256 buckets,
  code = clip(floor((x-min)/(max-min)*256), 0, 255)
  dequant = min + (code+0.5)*(max-min)/256
Caveat (documented in the notebook): knowhere trains SQ ranges per segment
on sampled data, so ranges differ microscopically from our global min/max;
top-10 recall is insensitive to this at far below the ±0.002 tolerance.

The comparison is PAIRED — the same N_GATE_QUERIES queries (the first
N_GATE_QUERIES of the 10k GT set, deterministic) are used offline and
against Milvus — so query-subset choice adds no sampling noise.

Runs on MPS (falls back to CPU). Cached to
data/ws2_cache/10m/_gate_expectation.json; re-runs are no-ops.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from benchlib.config import (  # noqa: E402
    DATA_DIR, EMBED_DIM, gt_path, slice_path,
)
from benchlib.quantization import recall_at_k  # noqa: E402

CACHE_F = DATA_DIR / "ws2_cache" / "10m" / "_gate_expectation.json"
N_GATE_QUERIES = 2000
CHUNK = 200_000
SCALE = "10m"


def main():
    if CACHE_F.exists():
        print(f"cached — {CACHE_F} exists, not re-running")
        print(CACHE_F.read_text())
        return

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    corpus = np.lib.format.open_memmap(slice_path(SCALE), mode="r")
    n_rows = corpus.shape[0]
    gt = np.load(gt_path(SCALE))["gt_indices"][:N_GATE_QUERIES]
    queries = np.load(DATA_DIR / "queries_gt_emb.npy")[:N_GATE_QUERIES]
    q = torch.from_numpy(np.ascontiguousarray(queries)).to(dev)

    # pass 1: per-dim min/max over the full slice
    t0 = time.time()
    dmin = np.full(EMBED_DIM, np.inf, dtype=np.float32)
    dmax = np.full(EMBED_DIM, -np.inf, dtype=np.float32)
    for s in range(0, n_rows, CHUNK):
        block = np.asarray(corpus[s:s + CHUNK])
        np.minimum(dmin, block.min(axis=0), out=dmin)
        np.maximum(dmax, block.max(axis=0), out=dmax)
    print(f"min/max pass done in {time.time()-t0:.0f}s")
    vmin = torch.from_numpy(dmin).to(dev)
    vdiff = torch.from_numpy(dmax - dmin).to(dev)
    vdiff = torch.where(vdiff > 0, vdiff, torch.ones_like(vdiff))

    # pass 2: quantize -> dequantize -> exact top-10 (running merge)
    t0 = time.time()
    best_scores = torch.full((N_GATE_QUERIES, 10), -1e30, device=dev)
    best_ids = torch.full((N_GATE_QUERIES, 10), -1, dtype=torch.int64,
                          device=dev)
    for s in range(0, n_rows, CHUNK):
        block = torch.from_numpy(np.asarray(corpus[s:s + CHUNK])).to(dev)
        codes = torch.clamp(
            torch.floor((block - vmin) / vdiff * 256.0), 0, 255)
        deq = vmin + (codes + 0.5) * vdiff / 256.0
        scores = q @ deq.T                       # (nq, chunk)
        top_s, top_i = torch.topk(scores, 10, dim=1)
        cat_s = torch.cat([best_scores, top_s], dim=1)
        cat_i = torch.cat([best_ids, top_i + s], dim=1)
        sel_s, sel_pos = torch.topk(cat_s, 10, dim=1)
        best_scores = sel_s
        best_ids = torch.gather(cat_i, 1, sel_pos)
        if (s // CHUNK) % 10 == 0:
            print(f"  chunk {s//CHUNK + 1}/{(n_rows+CHUNK-1)//CHUNK}", end="\r")
    pred = best_ids.cpu().numpy()
    print(f"\nscan+topk done in {time.time()-t0:.0f}s on {dev}")

    expected_r10 = recall_at_k(pred, gt, 10)
    # paired-prefix expectations: the Milvus gate may use fewer queries
    # (full-scan nprobe=4096 is slow); comparison stays paired either way.
    prefix_expectations = {
        str(n): recall_at_k(pred[:n], gt[:n], 10)
        for n in (500, 1000, 2000)}
    result = {
        "scale": SCALE, "n_gate_queries": N_GATE_QUERIES,
        "query_subset": f"first {N_GATE_QUERIES} of queries_gt_emb.npy",
        "expected_recall_at_10": expected_r10,
        "expected_recall_at_10_by_prefix": prefix_expectations,
        "tolerance": 0.002,
        "milvus_gate_point": "ivf_sq8, nprobe=4096 (= nlist, full scan)",
        "quant_scheme": ("per-dim global minmax, 256 buckets, "
                         "dequant = min+(code+0.5)*range/256, asymmetric IP"),
        "device": dev,
        "footprint_denominator_note": (
            "10M footprint ratios use analytic raw fp32 bytes "
            "(10M*1024*4 = 40.96 GB) as denominator; justified empirically "
            "by the 1M run where FLAT measured loaded bytes == raw vector "
            "bytes exactly (3906 MiB heap-excluded)."),
    }
    CACHE_F.parent.mkdir(parents=True, exist_ok=True)
    CACHE_F.write_text(json.dumps(result, indent=1))
    print(f"expected recall@10 (SQ8 full-scan, offline) = {expected_r10:.6f}")
    print(f"wrote {CACHE_F}")


if __name__ == "__main__":
    main()
