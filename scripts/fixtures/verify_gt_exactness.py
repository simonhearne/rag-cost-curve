#!/usr/bin/env python3
"""Independently re-verify committed ground truth via exact fp64 CPU search.

The committed gt_{scale}_top100.npz files were produced by notebook 01 using
torch fp32 matmul on MPS. This script recomputes exact inner-product top-100
for a small random probe of GT queries using plain numpy in float64 on CPU
(no torch, no GPU) and compares index sets + ordering against the committed
files, distinguishing genuine mismatches from score near-ties (gap < 1e-6).

Corpus memmaps are streamed in chunks (never loaded fully into RAM) so the
10M / ~41GB pass stays memory-bounded.

Usage:  python scripts/fixtures/verify_gt_exactness.py
Writes: results/ws1/gt_verification.csv
Exit code 0 always completes and writes the CSV; prints a warning and does
NOT touch the committed ground truth if mismatches beyond near-ties exist.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib.config import (
    DATA_DIR, EMBED_DIM, GT_CORPUS_CHUNK, GT_TOP_K, RESULTS_DIR, SEED,
    SLICES, gt_path, slice_path,
)

N_PROBE_QUERIES = 50
NEAR_TIE_GAP = 1e-6


def exact_topk_fp64(corpus_path: Path, n_rows: int, queries: np.ndarray,
                    k: int, chunk_rows: int = GT_CORPUS_CHUNK):
    """Exact float64 CPU inner-product top-k, streamed from the memmap.

    queries: (Q, D) float64, unit-norm. Returns (idx int64 (Q,k),
    score float64 (Q,k)), sorted descending.
    """
    corpus = np.lib.format.open_memmap(corpus_path, mode="r")
    assert corpus.shape[0] >= n_rows and corpus.shape[1] == queries.shape[1]
    Q = queries.shape[0]
    top_s = np.full((Q, k), -2.0, dtype=np.float64)
    top_i = np.full((Q, k), -1, dtype=np.int64)
    for lo in range(0, n_rows, chunk_rows):
        hi = min(lo + chunk_rows, n_rows)
        block = np.asarray(corpus[lo:hi], dtype=np.float64)  # (C, D)
        scores = queries @ block.T                            # (Q, C) fp64
        C = hi - lo
        keep = min(k, C)
        part = np.argpartition(-scores, keep - 1, axis=1)[:, :keep]
        part_s = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_s, axis=1)
        cand_i = np.take_along_axis(part, order, axis=1) + lo
        cand_s = np.take_along_axis(part_s, order, axis=1)
        merged_s = np.concatenate([top_s, cand_s], axis=1)
        merged_i = np.concatenate([top_i, cand_i], axis=1)
        keep_order = np.argsort(-merged_s, axis=1)[:, :k]
        top_s = np.take_along_axis(merged_s, keep_order, axis=1)
        top_i = np.take_along_axis(merged_i, keep_order, axis=1)
    return top_i, top_s


def true_score(corpus, q_j: np.ndarray, idx: int, lookup: dict) -> float:
    """fp64 IP score of a single corpus row, via lookup if already known."""
    if idx in lookup:
        return lookup[idx]
    return float(q_j @ np.asarray(corpus[idx], dtype=np.float64))


def verify_scale(scale: str, qe_gt: np.ndarray, probe_idx: np.ndarray) -> dict:
    n_rows = SLICES[scale]
    corpus_path = slice_path(scale)
    committed = np.load(gt_path(scale))
    committed_idx_all = committed["gt_indices"]
    committed_score_all = committed["gt_scores"]

    queries = qe_gt[probe_idx].astype(np.float64)
    t0 = time.time()
    ref_idx, ref_score = exact_topk_fp64(corpus_path, n_rows, queries, GT_TOP_K)
    elapsed = time.time() - t0

    corpus = np.lib.format.open_memmap(corpus_path, mode="r")

    n_full_set_match = 0
    n_full_exact_match = 0
    n_true_mismatches = 0
    n_near_tie_mismatches = 0
    max_true_gap = 0.0
    per_query_overlap = []

    for qi, j in enumerate(probe_idx):
        r_idx, r_score = ref_idx[qi], ref_score[qi]
        c_idx, c_score = committed_idx_all[j], committed_score_all[j]

        r_set, c_set = set(r_idx.tolist()), set(c_idx.tolist())
        overlap = len(r_set & c_set)
        per_query_overlap.append(overlap / GT_TOP_K)
        if overlap == GT_TOP_K:
            n_full_set_match += 1
        exact_order = bool(np.array_equal(r_idx, c_idx))
        if exact_order:
            n_full_exact_match += 1
            continue

        lookup = dict(zip(r_idx.tolist(), r_score.tolist()))
        q_j = queries[qi]
        for pos in range(GT_TOP_K):
            if r_idx[pos] == c_idx[pos]:
                continue
            true_ref = r_score[pos]
            true_committed = true_score(corpus, q_j, int(c_idx[pos]), lookup)
            gap = abs(true_ref - true_committed)
            if gap < NEAR_TIE_GAP:
                n_near_tie_mismatches += 1
            else:
                n_true_mismatches += 1
                max_true_gap = max(max_true_gap, gap)

    return {
        "scale": scale,
        "n_queries_checked": len(probe_idx),
        "k": GT_TOP_K,
        "n_queries_full_set_match": n_full_set_match,
        "n_queries_exact_order_match": n_full_exact_match,
        "mean_set_overlap_pct": round(float(np.mean(per_query_overlap)) * 100, 4),
        "min_set_overlap_pct": round(float(np.min(per_query_overlap)) * 100, 4),
        "n_position_mismatches_near_tie": n_near_tie_mismatches,
        "n_position_mismatches_true": n_true_mismatches,
        "max_true_mismatch_score_gap": round(max_true_gap, 9),
        "elapsed_seconds": round(elapsed, 1),
        "device": "cpu",
        "dtype": "float64",
        "seed": SEED,
    }


def main():
    qe_gt = np.load(DATA_DIR / "queries_gt_emb.npy")
    assert qe_gt.shape[1] == EMBED_DIM
    rng = np.random.default_rng(SEED)
    probe_idx = np.sort(rng.choice(len(qe_gt), size=N_PROBE_QUERIES, replace=False))

    rows = []
    for scale in ("1m", "10m"):
        if not gt_path(scale).exists():
            print(f"skip {scale}: {gt_path(scale).name} not found")
            continue
        print(f"verifying {scale} ({N_PROBE_QUERIES} probe queries, "
              f"fp64 CPU, streamed against {slice_path(scale).name})...")
        result = verify_scale(scale, qe_gt, probe_idx)
        rows.append(result)
        print(f"  {result}")

    out = RESULTS_DIR / "ws1"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "gt_verification.csv", index=False)
    print(f"-> {out / 'gt_verification.csv'}")

    beyond_near_ties = [r for r in rows if r["n_position_mismatches_true"] > 0]
    if beyond_near_ties:
        print("\nWARNING: true mismatches (beyond near-ties) found at scale(s): "
              f"{[r['scale'] for r in beyond_near_ties]}. "
              "Ground truth may need investigation/recomputation — "
              "do NOT recompute without checking in first.")
        sys.exit(1)
    print("\nAll rank differences (if any) are near-ties (score gap < 1e-6). "
          "Committed ground truth verified exact.")


if __name__ == "__main__":
    main()
