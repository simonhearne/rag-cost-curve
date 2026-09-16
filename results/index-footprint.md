# Index footprint — quantization and dimensionality

Covers `results/ws2/` (quantization, 1M and 10M) and `results/ws3/`
(dimensionality reduction, 1M and 10M). Claims C3/C4/C5/C8/C14. Every
number below is written by exactly one cell in `notebooks/02_quantization_curves.ipynb`,
`notebooks/03_dimensionality.ipynb`, or `notebooks/04_recall_vs_quality.ipynb`
(the dimensionality 10M addendum), into the CSVs cited inline. One
exception, flagged where it is used: the footprint decomposition in point 5
below comes from `scripts/index-footprint/ws2_footprint_decomposition.py`
(`results/ws2/footprint_decomposition_{1m,10m}.csv`), which is arithmetic
over the already-committed `data/ws2_cache/` measurements — it measures
nothing new. Charts:
`results/ws2/chart_recall_vs_compression_1m_10m.png`,
`results/ws2/chart_refine_ablation_1m.png`,
`results/ws3/chart_recall_vs_compression_1m.png`,
`results/ws3/chart_recall_vs_compression_1m_10m.png`.

## The question, and the answer

The premise on the slide: an index compressed 16–32x costs only a
single-digit percentage of recall. **That claim does not survive as a
measured-footprint number, at either scale.**

Two different things get compressed here, and they compress by very
different amounts, so every ratio below is labelled:

- **payload** — nominal bytes-per-vector versus uncompressed fp32.
- **measured footprint** (also "loaded footprint") — bytes Milvus actually
  loads to serve the collection: index files, mmapped chunk-cache, and
  segment heap, measured after a real query workload, not computed from a
  formula.

Quantization alone (`results/ws2/`) reaches **32x payload** with RaBitQ's
1-bit codes, but those codes need an SQ8 refine copy to hit usable recall,
and that copy dominates the served footprint: **measured footprint is
3.1x–3.5x**, not 32x, at ≤1 % recall loss (`results/ws2/c8_verdict_1m.csv`,
`c8_verdict_10m.csv`).

Stacking dimensionality reduction on top (`results/ws3/`) raises the
honest ceiling, but not to 16x. The best point that keeps recall loss to a
single digit against full-dimension ground truth is uncentered PCA to
384 dimensions plus SQ8: **8.66x measured footprint at recall@10 = 0.9089
(9.11 % loss)** at 1M scale (`results/ws3/dim_verdict_1m.csv`), confirmed
and nudged up to **8.92x at recall@10 = 0.9152 (8.5 % loss) at 10M**
(`results/ws3/dim_verdict_10m.csv`) — **10M is the citable scale for this
number and for C3/C8 generally.** Nothing measured between 9x and 16x
measured footprint holds single-digit recall loss, on any transform, at
either scale. Deep compression itself is trivial to reach (128d SQ8
measures 20.9x measured footprint) — recall is what runs out first.

The defensible framing, unchanged in kind from 1M to 10M: **32x on the
codes you scan; ~3–3.5x on the memory you provision if you need ≥99 %
recall; ~8.7–8.9x on the memory you provision if you accept 9 % recall
loss — measured, both scales, curve in the repo.**

A more conservative, still-comfortable operating point one rung up from
the 8.66x/8.92x edge case: **512d uncentered PCA × SQ8**, **6.71x measured
footprint at recall@10 = 0.9599 (1M, 4.0 % loss)** / **6.89x at
recall@10 = 0.9630 (10M)**. At 1M, at the 0.95 recall target it serves
**246 QPS versus the 1024d SQ8 arm's 184** — smaller *and* faster than
the uncompressed-dimension arm.

## Why the obvious reading is wrong

**1. The 32x number is a payload number, not a footprint number.**
RaBitQ's binary codes are 32x smaller than fp32 payload, but Milvus can't
serve usable recall from the codes alone — recall@10 without a refine step
is 0.7411–0.7776 depending on scale. Adding an SQ8 refine copy to rerank
candidates gets recall to 0.978–0.992, but the refine copy is itself
~3.5x payload, so the *served* footprint lands at 3.1x, not 32x. The two
numbers describe different things and both are true; a slide that quotes
one without the other is misleading either way.

**2. mxbai's Matryoshka (MRL) claim does not survive contact with
retrieval recall.** With the quantizer removed entirely and measured
against exact full-dimension ground truth, Matryoshka truncation to 512d
retains only 0.700 of the full-dimensional top-10; uncentered PCA at the
same 512d retains 0.964 — a 12–27 point gap across the dimension ladder,
far outside the ~0.01 tie band this evaluation otherwise works to. Even
truncating a mere 128 of 1024 dimensions (896d) already falls to 0.871,
failing the single-digit-loss bar at just 1.14x dimension reduction. PCA,
a generic transform with no training-time relationship to the embedding
model, beats the model's own claimed Matryoshka structure at every rung
(`results/ws3/transform_ceiling_1m.csv`). This result was independently
re-derived in a separate float64 numpy implementation sharing no code
with the notebook (0.6956 / 0.5194 / 0.3311 at 512/256/128d, n=1000) — it
is not an implementation bug.

The honest qualifier, worth stating plainly rather than softening: this is
a *stricter* test than most Matryoshka claims are checked against. MRL is
usually validated on task metrics (NDCG/MTEB against human relevance
labels), which tolerate a *different but equally good* neighbour set. This
evaluation measures neighbour *identity* against the full-dimension
index — "how much of what the full model considers relevant survives
compression?" A model can legitimately preserve task quality while doing
badly on this stricter test; whether that distinction matters for
downstream answer quality is a separate question this evidence does not
settle. So the defensible claim is **"Matryoshka truncation does not
preserve retrieval recall on this model,"** not "Matryoshka doesn't work."

**3. Compression multiplies; recall damage does not.** The intuitive
picture — 4x from dimension reduction times ~3.5x from the quantizer
gives ~12x of *bytes* — is correct for footprint but wrong for damage.
Split the two losses apart (project first with no quantizer, then
quantize what's left) and SQ8 costs almost nothing on top of a projection:
median gap below the exact-search ceiling is 0.0016 (range 0.0002–0.0092)
across all 13 SQ8 arms. RaBitQ-plus-refine costs an order of magnitude
more (median 0.0187, range 0.010–0.027) at every dimension tested. The
practical consequence: once a dimension is chosen, SQ8 is close to a free
3.5x, and the whole design problem collapses to picking the dimension —
**the projection is the loss.**

**4. A contrarian side-finding, confirmed at both scales: plain SQ8
quietly beats the fancy binary quantizer.** RaBitQ+SQ8-refine and plain
IVF_SQ8 land in the same measured-footprint class (~3.1–3.5x) and tie on
recall (0.9917 SQ8 vs 0.9920 refine-k2 at 10M — inside the ±0.0024
ingest-noise band) — but SQ8 wins on everything else: it reaches its
recall ceiling at a shallower search parameter, and at the 0.99 operating
point it serves 8.8 QPS versus refine's 0.88 QPS, a 10x gap (37.9 vs 12.1
QPS at the more relaxed 0.95 target). Refine's only remaining edge is
~12 % less footprint (3.11x vs 3.54x); it costs an order of magnitude of
throughput to get it. Caveat: single-node Docker on a laptop with client
and server sharing cores, so absolute throughput numbers are indicative —
but a 10x ordering at matched recall is far beyond that noise.

**5. Less than half of the deepest-compression arm's footprint is the
compressed vector — and the biggest other piece is an fp32 table the
quantizer never touches.** The no-refine `rabitq` arm is the cleanest case,
because it has no refine copy to blame: 32x nominal payload, but 14.1x
measured footprint at 10M (13.6x at 1M). Decomposing the measured bytes
(`results/ws2/footprint_decomposition_10m.csv`; 10M is the exact scale, see
below):

| component | B/vector | share of loaded |
|---|---:|---:|
| RaBitQ 1-bit codes — the part that *is* 32x | 128.0 | 44.1 % |
| **IVF coarse-quantizer centroids — fp32, full 1024d, unquantized** | **68.8** | **23.7 %** |
| RaBitQ per-vector estimator factors | 29.4 | 10.1 % |
| int64 row ids in the inverted lists | 8.0 | 2.8 % |
| Milvus pk/stats segment heap | 56.8 | 19.5 % |
| unexplained residual (the model's error bar) | −0.4 | −0.1 % |
| **total** | **290.5** | **= 14.10x vs 4096 B/vector raw** |

Three things follow, none of them RaBitQ's fault:

- **The coarse quantizer is stored as whole uncompressed vectors.** Every
  centroid is 4096 B at 1024d no matter what the payload quantizer does, and
  the table is *per segment* — its cost scales with `n_segments × nlist`,
  not with `nlist`. At 10M that is 41 uniform segments × nlist=4096 = one
  fp32 centroid per 59.5 rows, 68.8 B/vector. At 1M, segments are smaller
  and uncompacted, faiss's `MIN_POINTS_PER_CENTROID = 39` clamp bites on
  some of them, and the same term lands at 78–105 B/vector depending on the
  arm's segment layout (`footprint_decomposition_1m.csv`) — *larger* at the
  smaller scale.
- **RaBitQ's real payload is 26x, not 32x.** The 1-bit codes need 29.4
  B/vector of fp32 estimator factors alongside them, so the served payload
  is 157.4 B/vector ≈ 1.23 bits/dim. The 32x on the slide counts only the
  codes.
- **32x measured footprint was never reachable here by any quantizer.**
  Centroids + row ids + segment heap are 133.6 B/vector at 10M before a
  single payload byte, so an IVF collection at 1024d in this setup tops out
  at **~31x measured footprint even with zero-byte vectors** (~24–29x at
  1M, where the centroid term is bigger). The C8 result is bounded by index
  structure, not only by recall.

Provenance and error bar, because this is a derivation rather than a
measurement: the totals above reproduce the `c8_verdict` compression ratios
exactly (14.10x, 13.63x, 3.54x, 3.11x), and at 10M the centroid term is
*known*, not fitted — the collection was force-compacted into 41 uniform
243,902-row segments, comfortably above the `39 × nlist` clamp, so nlist is
exactly the configured 4096. The model then leaves a residual of ≤1.7
B/vector (≤0.04 % of index bytes) on every non-RaBitQ IVF arm, and 29.0–29.5
B/vector on both RaBitQ arms — which is what identifies the RaBitQ factor
term as a real component rather than slop. Two honest limits: at 1M,
per-segment row counts were never recorded, so there the centroid term is
taken as the residual after applying the 10M-identified constants (the
falsification check — implied centroids per segment must not exceed the
configured 4096 — is asserted in the script and pinned in
`tests/test_footprint_decomposition.py`); and the 29.4 B/vector RaBitQ term
is identified by subtraction, not read out of Knowhere's source, so its
internal composition is unverified.

## What was measured

**Quantization (`results/ws2/`).** Milvus v2.6.18 standalone, Docker on
Apple Silicon. A 12-arm index matrix (FLAT/HNSW/IVF_FLAT fp32, IVF_SQ8,
IVF_PQ at m=128/256/512, RaBitQ with and without SQ8 refine at
refine_k=1/2/5/10) was run at 1M vectors (`curves_1m.csv`, 97 rows; sum of
per-arm wall clock 7.9h; no arm failed) and again at 10M
(`curves_10m.csv`, 33 rows; total wall-clock ≈41h, of which 22.4h was an
aborted fp32-mmap reference arm — see caveats). Correctness gates passed
at both scales: at 1M, FLAT vs fp64-verified exact ground truth measured
recall@10 = 1.000000, recall@100 = 0.999994 (threshold ≥0.999); at 10M, an
offline SQ8 simulation predicted recall@10 = 0.9934 and Milvus measured
0.9930 (Δ −0.0004, tolerance ±0.002).

**Dimensionality reduction (`results/ws3/`).** Same stack and protocol as
the quantization 1M run (fresh collection per arm, `nlist=4096`, IP on
L2-normalized vectors, one deliberate deviation: nprobe sweep extended to
512). 21 arms (MRL truncation, centered PCA, uncentered PCA, each ×
quantizer) × 10 nprobe points, 8.2h total wall clock, no arm failed
(`curves_1m.csv`, 210 rows). Two correctness gates passed: `mrl_1024_sq8`
(identity transform) reproduced the quantization results' `ivf_sq8` 1M curve within a worst
Δ of 0.0060 (inside the 0.01 tie band) and matched its measured footprint
within 0.4 % (1124 vs 1129 MiB); `mrl_1024_refine_k2` reproduced
`rabitq_refine_sq8_k2` similarly (0.97849 vs 0.97861 r@10, 1273 vs 1272
MiB). A separate "transform ceiling" probe (`transform_ceiling_1m.csv`,
29 offline configs) measured exact brute-force search in the projected
space with no index and no quantizer — the upper bound each Milvus arm is
measured against, letting the two loss sources (projection, quantization)
be attributed separately.

Three winning-combination arms — `pca_uc_512_sq8`, `pca_uc_384_sq8`,
`mrl_512_sq8` — were re-run at 10M on 2026-09-07
(`results/ws3/curves_10m.csv`, 18 rows; `dim_verdict_10m.csv`;
`chart_recall_vs_compression_1m_10m.png`), using the quantization results' 10M protocol (one
forced compaction round, then frozen) rather than the dimensionality results' 1M protocol,
because that is production-realistic. Two gates per arm passed: exact
search over the full 10M corpus on the first 1,000 GT queries within
±0.05 of the 1M ceiling, and the Milvus arm at nprobe=512 on the same
1,000 queries within [ceiling − 0.03, ceiling + 0.005].

## Limits and caveats

- **1M and 10M are not directly comparable at fixed search parameters.**
  A forced-compaction step used only at 10M changes the sealed-segment set
  and shifts recall at fixed nprobe by up to ~1 %; the 1M numbers (measured
  on uncompacted, many-segment collections) are ~0.5–1 % *optimistic*
  relative to the production-realistic, frozen-compacted 10M numbers.
  Separately, repeat ingests of identical data differ by up to ±0.0024
  recall@10 from segment-boundary nondeterminism — read any cross-arm
  recall gap under ~0.01 as a tie.
- **Two Milvus v2.6 measurement pitfalls were found and fixed before any
  footprint number was trusted:** sealed-segment `mem_size` counts only
  pk/stats heap, not vector or index bytes, and serving state materializes
  lazily; the datacoord `stored_index_files_size` metric overcounts by
  ~2x from stale pre-compaction index generations until GC. The final
  accounting sums built-index file bytes for the loaded segments, the
  mmapped chunk-cache bytes, and segment heap, measured after a real query
  workload. Cross-checks **at 1M**: FLAT equals raw fp32 exactly (3960 vs
  3906+54 MiB), and HNSW costs ~1.04x raw (vectors + ~144 MiB graph at
  M=16). **At 10M no fp32 in-RAM arm was run** (pruned, below); the
  40.96 GB analytic denominator is validated against the measured 10M
  IVF_FLAT index files, 41.75 GB — agreement within ~2%, not exact.
  Process-RSS deltas were recorded as audit-only fields (2–3x noisy from
  allocator retention), never as a reported footprint number.
- **Latency and QPS are laptop numbers**, single-node Docker with client
  and server sharing cores; absolute values and adjacent-sweep-point
  orderings are indicative only (non-monotone p50 was observed across HNSW
  ef points). Recall values are deterministic and are the citable numbers.
  Latency is client-observed single-query gRPC (100 warm-ups, 500 timed);
  QPS comes from a separate 8-thread run and is never derived from
  latency. fp32 serving latency at 10M is deliberately unmeasured — the
  fp32 reference arm at 10M was mmap-served through a VirtioFS mount
  (2.4–9.2s p50), a serving-mode artifact of the Docker VM, not an fp32
  latency claim, and must not be quoted as one.
- **Recall protocol details that affect what a number means:** recall@10
  is measured at serving `limit=10`; refine arms rerank `refine_k × limit`
  candidates, so limit matters. recall@100 is undefined for HNSW below
  `ef=100` (Milvus requires ef ≥ limit). The recall denominator throughout
  is the fixed, full-dimension, fp64-verified ground truth — never
  regenerated at a truncated dimension, because the question being asked
  is "how much of what the full model considers relevant survives
  compression," not "how self-consistent is a compressed index with
  itself" (which would score a transform that destroys the embedding's
  semantics as perfect, by construction).
- **Some ceilings are still sweep caps, not necessarily quantizer limits.**
  At 1M, RaBitQ+refine plateaued at ≈0.981 recall@10 by nprobe=256 with
  recall still rising. Extending the sweep to nprobe=512/1024 (the
  dimensionality reduction run, and the 10M run) showed 0.99 recall *is* reachable for refine (0.9920 at
  nprobe=1024) — but the cost is brutal: the last ~0.003 of recall halves
  QPS to 0.88, and rerank depth saturates at k=2 (k=5/k=10 are
  recall-identical to k=2 at matched nprobe, confirming the 1M ablation
  behind C5/C14 at scale). Treat 0.99 recall with refine as reachable but
  operationally unattractive; SQ8 at nprobe=512 is the practical 0.99
  operating point.
- **Arms pruned, truncated, or aborted (no silent gaps):** at 10M,
  `flat_fp32`/`hnsw_fp32` were pruned (a ~41 GB in-RAM index does not fit
  the Docker VM); `ivf_pq_m512` was pruned (dominated at 1M, slowest
  build, projected ≥9h build bought no new information); the fp32-mmap
  reference arm completed 2 of 4 sweep points before hitting a 9h abort
  budget (its purpose — footprint and recall reference — was already met,
  and results were consistent with the 1M curve); refine k1/k10 were
  reduced to single anchor points once k-saturation was established. No
  arm silently failed at any scale (`data/ws2_cache/*/`,
  `data/ws3_cache/1m/` contain no `*.failed.json`).
- **A fixed per-collection cost caps how far footprint compression can
  go.** Every dimensionality-reduction arm carries the same ~54 MiB of Milvus pk/stats segment
  heap regardless of vector dimension. At 1024d that's 5 % of the
  footprint; at 128d it's 28 % (189 MiB total, only 135 MiB of it index).
  Reported ratios include this cost because it is really provisioned, but
  it means footprint compression flattens out at low dimension no matter
  how small the vectors get. The heap is not the only such cost: the IVF
  centroid table (point 5 above) adds 69–105 B/vector at 1024d, unquantized,
  and the two together put a hard ~31x (10M) / ~24–29x (1M) ceiling on
  measured footprint compression for any IVF arm at this dimension.
- **The transform ceiling is scale-dependent, and the lossiest transform
  gains the most from more data.** Measured offline on the identical first
  1,000 queries at both scales: `pca_uc_512`'s ceiling moved 0.9631 →
  0.9666 and `pca_uc_384`'s moved 0.9116 → 0.9158 (both +0.004), but
  `mrl_512`'s moved 0.6956 → 0.7503 (+0.055) — a larger corpus gives
  closer true nearest-neighbours, which survive a lossy projection more
  often. This does not rescue MRL (still 21 points below PCA at the same
  dimension at 10M) but it means the citable `mrl_512` figure is 0.75,
  not the 1M run's 0.70, and any comparison should say which scale it's
  from.
- **PCA is fitted on a sample, not refit per scale.** The rotation used at
  both 1M and 10M was fit once on 500k corpus-only rows (never queries,
  seed 42) and reused unchanged at 10M, matching how a deployed transform
  would actually be built; both rotations are cached with pinned
  eigenvector signs and checksummed in `data/MANIFEST.json` as a real
  reproducibility check. The 1M footprint ratios use measured-FLAT-1024
  as the denominator; the 10M-addendum ratios use the analytic 40.96 GB
  raw-fp32-1024d denominator instead (the two agree within 2–3 %, but they
  are not the identical calculation).
- **The centering trap — a methodology error, disclosed rather than
  smoothed over.** The first PCA implementation mean-centered the data
  (defended at the time as "the standard definition, what faiss/sklearn
  do"). That cost 0.17 recall@10 by itself: centered PCA sits at a flat
  ~0.83 from 896d down to 384d, a shape that is the signature of a fixed
  transform cost, not a dimensionality effect. The cause: mean-centering
  and then L2-renormalizing is not an orthogonal map on the sphere, so it
  reorders cosine neighbours — and text embeddings sit in a narrow cone
  with a large shared component, making that component's removal a major
  geometric change. Dropping the centering restores an orthogonal
  projection (the full-rank case then returns exactly 1.0, now pinned by
  a unit test, alongside a test pinning that centering *breaks* it). Both
  variants are kept and reported in the tables above and in the appendix.
- **A transient measurement-harness issue, disclosed:** Docker Desktop
  twice killed a footprint-measuring subprocess mid-matrix, destroying two
  already-complete arm sweeps (`mrl_256_sq8`, `mrl_128_sq8`); both were
  re-run cleanly and the published numbers are from the clean re-runs.
  Footprint reads themselves re-raise on failure rather than silently
  defaulting, so a transient failure can never fabricate a footprint
  number — only an audit-only, already-noisy RSS metric was allowed to
  degrade gracefully.
- **Reproduction budget, for anyone re-running this:** the 10M
  quantization run took ≈41h wall clock and needs disk headroom well
  beyond the ~41 GB corpus — Milvus's write-ahead log does not truncate in
  this version (grows ≈1x ingested bytes, survives drops/restarts) and
  each forced compaction round rewrites the full binlog again; peak disk
  during the 10M run reached ≈155 GB over baseline. A mid-run VM crash
  from disk pressure was recovered via verified top-up ingest to exactly
  10,000,000 rows, with integrity proven by the correctness gate passing
  afterward, not assumed.

## Appendix — arm by arm

**Quantization, 1M — best sweep point per arm** (payload = nominal bytes/vector vs
fp32; footprint = measured loaded bytes vs measured FLAT; from
`results/ws2/c8_verdict_1m.csv`):

| Arm | best r@10 | payload | measured footprint | meets 0.90 | meets 0.95 | meets 0.99 |
|---|---|---|---|---|---|---|
| hnsw_fp32 | 0.9994 | 1x | 0.97x (4104 MiB) | ✓ | ✓ | ✓ |
| ivf_flat_fp32 | 0.9952 | 1x | 0.98x (4063 MiB) | ✓ | ✓ | ✓ |
| ivf_sq8 | 0.9903 | 4x | 3.5x (1129 MiB) | ✓ | ✓ | ✓ |
| ivf_pq_m512 | 0.9440 | 8x | 6.0x (660 MiB) | ✓ | ✗ | ✗ |
| ivf_pq_m256 | 0.8195 | 16x | 9.2x (432 MiB) | ✗ | ✗ | ✗ |
| ivf_pq_m128 | 0.6018 | 32x | 12.6x (314 MiB) | ✗ | ✗ | ✗ |
| rabitq (no refine) | 0.7411 | 32x† | 13.6x (291 MiB) | ✗ | ✗ | ✗ |
| rabitq_refine_sq8_k1 | 0.9546 | 3.6x* | 3.1x (1273 MiB) | ✓ | ✓ | ✗ |
| rabitq_refine_sq8_k2 | 0.9786 | 3.6x* | 3.1x (1272 MiB) | ✓ | ✓ | ✗ |
| rabitq_refine_sq8_k5 | 0.9812 | 3.6x* | 3.1x (1262 MiB) | ✓ | ✓ | ✗ |
| rabitq_refine_sq8_k10 | 0.9813 | 3.6x* | 3.1x (1282 MiB) | ✓ | ✓ | ✗ |

\* the refine arms' nominal payload counts binary codes plus the SQ8
refine copy (1152 B/vector); the binary codes alone are 32x — the number a
bare "32x" on a slide evokes.

† codes only. RaBitQ also stores 29.4 B/vector of fp32 estimator factors
(point 5), so the served payload is 157.4 B/vector — **26x, not 32x** —
before any IVF or Milvus overhead.

**Quantization, 10M — best sweep point per arm** (footprint vs 40.96 GB analytic
raw fp32; from `results/ws2/c8_verdict_10m.csv`):

| Arm | best r@10 | at | measured footprint | QPS (8-thread) | meets 0.99 |
|---|---|---|---|---|---|
| ivf_sq8 | 0.9917 | nprobe=512 | 3.54x (10.8 GiB) | 8.8 | ✓ |
| rabitq_refine_sq8_k2 | 0.9920 | nprobe=1024 | 3.11x (12.2 GiB) | 0.88 | ✓ |
| rabitq_refine_sq8_k5 | 0.9921 | nprobe=1024 | 3.11x | 0.64 | ✓ |
| rabitq_refine_sq8_k10 | 0.9849 | nprobe=256 (anchor) | 3.11x | 1.9 | ✗ |
| rabitq_refine_sq8_k1 | 0.9793 | nprobe=256 (anchor) | 3.11x | 1.9 | ✗ |
| ivf_flat_fp32 (mmap)† | 0.9697 | nprobe=64 | 0.97x disk-resident | 0.11 | (aborted) |
| ivf_pq_m256 | 0.8194 | nprobe=256 | 10.4x (3.7 GiB) | 28 | ✗ |
| rabitq (no refine) | 0.7776 | nprobe=256 | 14.1x (2.7 GiB) | 2.7 | ✗ |
| ivf_pq_m128 | 0.6034 | nprobe=256 | 15.4x (2.5 GiB) | 57 | ✗ |

† mmap-served from disk through the Docker VM's VirtioFS mount — a
serving-mode artifact, not an fp32 latency claim.

**Dimensionality reduction, 1M — arms clearing single-digit recall loss** (footprint vs
measured FLAT-1024; from `results/ws3/dim_verdict_1m.csv`; deepest
compression first):

| arm | best r@10 | measured footprint | QPS @ best | meets ≥0.90 |
|---|---|---|---|---|
| pca_uc_384_sq8 | 0.9089 | 8.66x (457 MiB) | 191 | ✓ (by a hair) |
| pca_uc_448_sq8 | 0.9392 | 7.76x (510 MiB) | 333 | ✓ |
| pca_uc_512_sq8 | 0.9599 | 6.71x (591 MiB) | 119 | ✓ |
| pca_uc_512_refine_k2 | 0.9450 | 5.91x (670 MiB) | 16 | ✓ |
| pca_uc_768_sq8 | 0.9817 | 4.70x (842 MiB) | 139 | ✓ |
| mrl_1024_sq8 (= quantization's ivf_sq8) | 0.9914 | 3.52x (1124 MiB) | 57 | ✓ |
| mrl_1024_refine_k2 | 0.9863 | 3.11x (1273 MiB) | 17 | ✓ |

The other 14 of 21 arms (every MRL arm, every centered-PCA arm, and
uncentered PCA at 256d and below) fail the 0.90 line; full surface in
`results/ws3/dim_verdict_1m.csv`.

Cost curve at 1M — recall target, price, and how:

| you want r@10 ≥ | you pay (measured footprint) | at |
|---|---|---|
| 0.99 | 3.52x | 1024d SQ8 (no dimension reduction survives here) |
| 0.98 | 4.70x | 768d uncentered PCA × SQ8 |
| 0.96 | 6.71x | 512d |
| 0.94 | 7.76x | 448d |
| 0.91 | 8.66x | 384d |

Transform ceiling — exact search in the projected space, no quantizer, vs
full-dimension ground truth (`results/ws3/transform_ceiling_1m.csv`):

| dim | MRL truncation | PCA (mean-centered) | PCA (uncentered / SVD) |
|---:|---:|---:|---:|
| 1024 | 1.00000 | — | 0.99997 |
| 896 | 0.87143 | 0.83124 | 0.99558 |
| 768 | 0.81816 | 0.83079 | 0.99087 |
| 640 | 0.76223 | 0.82993 | 0.98261 |
| 512 | 0.70035 | 0.82710 | 0.96374 |
| 448 | 0.66279 | 0.82383 | 0.94432 |
| 384 | 0.62414 | 0.81349 | 0.91097 |
| 320 | 0.57487 | 0.78756 | 0.85960 |
| 256 | 0.51881 | 0.74336 | 0.78924 |
| 128 | 0.33228 | 0.55001 | 0.55354 |

**Dimensionality reduction, 10M addendum — the three winning-combo arms re-run at citable
scale** (footprint vs 40.96 GB raw fp32 at 1024d; from
`results/ws3/dim_verdict_10m.csv`):

| arm | r@10 @10M | r@10 @1M | Δ | measured footprint | loaded | QPS @512 | p50 @512 | single-digit loss |
|---|---:|---:|---:|---:|---:|---:|---:|:-:|
| pca_uc_512_sq8 | 0.9630 | 0.9599 | +0.003 | 6.89x | 5.54 GiB | 44 | 25.6 ms | ✓ |
| pca_uc_384_sq8 | 0.9152 | 0.9089 | +0.006 | 8.92x | 4.28 GiB | 66 | 18.8 ms | ✓ (8.5 % loss) |
| mrl_512_sq8 | 0.7539 | 0.6995 | +0.054 | 6.89x | 5.54 GiB | 42 | 26.9 ms | ✗ |

Throughput dividend at 10M: the quantization results' 1024d `ivf_sq8` serves recall@10 = 0.965
at nprobe=64 for 38 QPS (p50 24.6 ms); `pca_uc_512_sq8` serves 0.952 at
nprobe=128 for 155 QPS (p50 9.0 ms) and 0.959 at nprobe=256 for 88 QPS —
at the 0.95 target, roughly 4x the throughput at half the memory. At 0.99
recall, nothing below 1024d reaches the target, at either scale.
