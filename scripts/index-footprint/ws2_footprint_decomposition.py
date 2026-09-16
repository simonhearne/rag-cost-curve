"""Decompose the WS2 measured loaded footprint into its physical parts.

Motivating question: `rabitq` (no refine) has a 32x nominal payload
(1024 dims x 1 bit = 128 B/vector) but measures only 13.6x at 1M and 14.1x
at 10M measured footprint (`results/ws2/c8_verdict_{1m,10m}.csv`). Where do
the other bytes go?

This script does NOT measure anything new. It is arithmetic over the
already-committed per-arm footprint measurements in `data/ws2_cache/`
(`measure.index_files_bytes`, `measure.segment_heap_bytes`,
`measure.chunk_cache_bytes`, `measure.index_build_ids`) plus the index
parameters from the notebook's INDEX_MATRIX / INDEX_MATRIX_10 (cells 3 and
15 of notebooks/02_quantization_curves.ipynb), which are mirrored below.

The model, per sealed segment, is the standard faiss/Knowhere IVF layout:

    index_files = codes                 n_rows x nominal_payload_bytes
                + coarse centroids      n_segments x nlist x dim x 4  (fp32!)
                + row ids               n_rows x 8                    (int64)
                + PQ codebook           n_segments x m x 256 x (dim/m) x 4
                + RaBitQ factors        n_rows x ~29                  (IVF_RABITQ only)

and the served footprint adds Milvus's per-row pk/stats segment heap
(`segment_heap_bytes`, ~57 B/row, independent of the index).

Two things make the centroid term much larger than a reader expects:

1. It is stored as **fp32 at full dimension** and is never quantized, so
   every centroid costs a whole uncompressed vector (4096 B at 1024d) no
   matter what the quantizer does to the payload.
2. It is **per segment**, and nlist is per segment too, so the cost scales
   with n_segments x nlist, not with nlist.

Identification (why 10M is the primary scale here): at 10M the collection
was force-compacted once and then frozen, so all 41 segments are uniform
(243,902 rows each) and every arm shares the identical layout. 243,902 >=
39 x 4096 = 159,744, so faiss's MIN_POINTS_PER_CENTROID clamp does not
bite and nlist is exactly the configured 4096 -- the centroid term is known,
not fitted. The residual is then ~0 B/vector for every non-RaBitQ IVF arm
(the model's error bar) and ~29 B/vector for both RaBitQ arms, which is
what identifies the RaBitQ per-vector estimator factors.

At 1M the collections were never compacted, segment sizes are uneven and
unrecorded, so nlist per segment is unknown (it may be clamped). There the
centroid term is taken as the residual instead, with the 10M-identified
per-vector constants applied -- labelled `basis=inferred_1m` in the CSV. The
falsification check is that the implied centroid count per segment must
come out at or below the configured nlist=4096; the script asserts it.

Writes results/ws2/footprint_decomposition_{1m,10m}.csv (long format, one
row per component per arm) and records both in data/MANIFEST.json.
Pure arithmetic over cached JSON: no Milvus, no GPU, runs in <1s.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from benchlib.config import (  # noqa: E402
    DATA_DIR, EMBED_DIM, RESULTS_DIR, SLICES,
)
from benchlib.manifest import record  # noqa: E402

D = EMBED_DIM                  # 1024
NLIST = 4096                   # configured for every IVF arm, both scales
MIN_POINTS_PER_CENTROID = 39   # faiss/Knowhere clamp on nlist at train time
ID_BYTES = 8                   # int64 row id per vector in the inverted list
FP32 = 4

# Mirrors INDEX_MATRIX (cell 3) / INDEX_MATRIX_10 (cell 15) of
# notebooks/02_quantization_curves.ipynb. `pq_m=None` means not an IVF_PQ arm;
# `ivf=False` means no coarse quantizer (FLAT/HNSW).
ARMS = {
    "flat_fp32":     dict(codes=FP32 * D, ivf=False, pq_m=None, rabitq=False),
    "hnsw_fp32":     dict(codes=FP32 * D, ivf=False, pq_m=None, rabitq=False),
    "ivf_flat_fp32": dict(codes=FP32 * D, ivf=True, pq_m=None, rabitq=False),
    "ivf_sq8":       dict(codes=D, ivf=True, pq_m=None, rabitq=False),
    "ivf_pq_m512":   dict(codes=512, ivf=True, pq_m=512, rabitq=False),
    "ivf_pq_m256":   dict(codes=256, ivf=True, pq_m=256, rabitq=False),
    "ivf_pq_m128":   dict(codes=128, ivf=True, pq_m=128, rabitq=False),
    "rabitq":        dict(codes=D // 8, ivf=True, pq_m=None, rabitq=True),
    **{f"rabitq_refine_sq8_k{k}":
       dict(codes=D // 8 + D, ivf=True, pq_m=None, rabitq=True)
       for k in (1, 2, 5, 10)},
}


def load_arms(scale: str) -> dict:
    """Per-arm measured footprint from the WS2 cache, newest fields only."""
    out = {}
    for f in sorted((DATA_DIR / "ws2_cache" / scale).glob("*.json")):
        if f.name.startswith("_"):
            continue
        d = json.loads(f.read_text())
        m = d.get("measure")
        if not m:
            continue
        out[d["arm"]] = dict(
            n_segments=len(m.get("index_build_ids") or []),
            index_files_bytes=m["index_files_bytes"],
            chunk_cache_bytes=m["chunk_cache_bytes"],
            segment_heap_bytes=m["segment_heap_bytes"],
            loaded_bytes=m["loaded_bytes"],
        )
    return out


def codebook_bytes(spec, n_segments):
    """IVF_PQ trains one codebook per segment: m x 256 centroids of dim/m."""
    if spec["pq_m"] is None:
        return 0
    m = spec["pq_m"]
    return n_segments * m * 256 * (D // m) * FP32


def identify_rabitq_factor_bytes(arms_10m, n_rows) -> tuple[float, dict]:
    """Solve for the RaBitQ per-vector factor cost at 10M, where the centroid
    term is known exactly (uniform compacted segments, nlist unclamped).

    Returns (bytes_per_vector, diagnostics). Diagnostics carry the residual
    for every non-RaBitQ IVF arm -- the model's error bar, and the reason the
    RaBitQ residual can be read as a real component rather than slop.
    """
    resid, rabitq_resid = {}, []
    for arm, meas in arms_10m.items():
        spec = ARMS[arm]
        if not spec["ivf"]:
            continue
        segs = meas["n_segments"]
        rows_per_seg = n_rows / segs
        nlist = min(NLIST, int(rows_per_seg // MIN_POINTS_PER_CENTROID))
        assert nlist == NLIST, (
            f"{arm}: 10M segments ({rows_per_seg:.0f} rows) were expected to "
            f"be large enough to leave nlist unclamped; got {nlist}. The "
            "identification below assumes an exactly known centroid term.")
        modelled = (n_rows * spec["codes"]
                    + segs * nlist * D * FP32
                    + n_rows * ID_BYTES
                    + codebook_bytes(spec, segs))
        r = (meas["index_files_bytes"] - modelled) / n_rows
        resid[arm] = r
        if spec["rabitq"]:
            rabitq_resid.append(r)
    non_rabitq = [v for a, v in resid.items() if not ARMS[a]["rabitq"]]
    worst = max(abs(v) for v in non_rabitq)
    assert worst < 2.0, (
        f"model error bar is {worst:.2f} B/vector on a non-RaBitQ arm -- too "
        "large to read the RaBitQ residual as a distinct component")
    factor = sum(rabitq_resid) / len(rabitq_resid)
    return factor, dict(residual_b_per_vector=resid,
                        non_rabitq_worst_b_per_vector=worst)


def decompose(scale: str, arms: dict, factor_b: float) -> list[dict]:
    """One row per (arm, component). Components sum exactly to loaded_bytes:
    whatever the model does not explain lands in `unexplained_residual`."""
    n_rows = SLICES[scale]
    rows = []
    for arm in sorted(arms):
        spec, meas = ARMS[arm], arms[arm]
        segs = meas["n_segments"]
        idx = meas["index_files_bytes"]
        parts = []

        # FLAT has no index: its vectors are served from the mmapped chunk
        # cache instead, so that is where its payload shows up.
        if meas["chunk_cache_bytes"]:
            parts.append(("vector_chunk_cache", meas["chunk_cache_bytes"],
                          "measured", "raw fp32 columns mmapped for serving"))
        codes = n_rows * spec["codes"] if idx else 0
        if codes:
            label = ("quantized_codes_plus_sq8_refine_copy"
                     if "refine" in arm else "quantized_codes")
            parts.append((label, codes, "derived",
                          f"{spec['codes']} B/vector x {n_rows:,} rows"))
        cb = codebook_bytes(spec, segs)
        if cb:
            parts.append(("pq_codebook", cb, "derived",
                          f"{segs} segments x m x 256 x (dim/m) x 4 B"))
        if idx:
            parts.append(("row_ids", n_rows * ID_BYTES, "derived",
                          "int64 row id per vector in the inverted list"))
        if spec["rabitq"]:
            parts.append(("rabitq_per_vector_factors",
                          round(n_rows * factor_b), "derived_10m",
                          "fp32 estimator factors; identified at 10M, "
                          "applied at both scales"))
        if spec["ivf"]:
            rows_per_seg = n_rows / segs
            nlist_cap = int(rows_per_seg // MIN_POINTS_PER_CENTROID)
            if nlist_cap >= NLIST and scale == "10m":
                cent = segs * NLIST * D * FP32
                basis, note = "derived", (
                    f"{segs} uniform segments x nlist={NLIST} x {D}d fp32; "
                    "unclamped (rows/segment >= 39 x nlist)")
            else:
                cent = idx - sum(p[1] for p in parts)
                implied = cent / segs / (D * FP32)
                assert 0 < implied <= NLIST, (
                    f"{scale}/{arm}: implied {implied:.0f} centroids per "
                    f"segment is outside (0, nlist={NLIST}] -- decomposition "
                    "does not hold")
                basis, note = "inferred_1m", (
                    f"residual after the 10M-identified constants; implies "
                    f"{implied:.0f} centroids/segment over {segs} uneven "
                    f"uncompacted segments (configured nlist={NLIST})")
            parts.append(("ivf_coarse_centroids", cent, basis, note))
        if idx:
            resid = idx - sum(p[1] for p in parts if p[0] != "vector_chunk_cache")
            parts.append(("unexplained_residual", resid, "residual",
                          "index_files_bytes minus the modelled components"))
        parts.append(("segment_heap", meas["segment_heap_bytes"], "measured",
                      "Milvus pk/stats heap per row; index-independent"))

        total = meas["loaded_bytes"]
        assert abs(sum(p[1] for p in parts) - total) <= 2, (
            f"{scale}/{arm}: components do not sum to loaded_bytes")
        for name, b, basis, note in parts:
            rows.append(dict(
                scale=scale, arm=arm, n_rows=n_rows, n_segments=segs,
                component=name, bytes=int(b),
                bytes_per_vector=round(b / n_rows, 2),
                share_of_loaded=round(b / total, 4),
                loaded_bytes=total, basis=basis, note=note))
    return rows


def main():
    arms_1m, arms_10m = load_arms("1m"), load_arms("10m")
    factor_b, diag = identify_rabitq_factor_bytes(arms_10m, SLICES["10m"])
    print(f"RaBitQ per-vector factors identified at 10M: {factor_b:.1f} "
          f"B/vector (model error bar on non-RaBitQ IVF arms: "
          f"{diag['non_rabitq_worst_b_per_vector']:.2f} B/vector)")
    for arm, r in sorted(diag["residual_b_per_vector"].items()):
        print(f"  10M residual  {arm:24s} {r:7.2f} B/vector")

    for scale, arms in (("1m", arms_1m), ("10m", arms_10m)):
        rows = decompose(scale, arms, factor_b)
        out = RESULTS_DIR / "ws2" / f"footprint_decomposition_{scale}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        record(out)
        print(f"wrote {out.relative_to(RESULTS_DIR.parent)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
