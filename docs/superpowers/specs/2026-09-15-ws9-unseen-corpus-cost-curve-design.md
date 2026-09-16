# WS9 — a cost curve on the corpus the model has never seen

**Status: pre-registration. Written 2026-09-15, before any WS9 row exists.**
Branch `ws9`, off `main` at `f150446`. Nothing lands on `main`.

Every branch below is quoted **by name** in the outputs, so each verdict is a
lookup rather than a judgement made after seeing the data. The hypotheses in
§3, the subsets in §2, the question panel in §2.3, the budget in §7 and the
stop rules in §8 are fixed by this document. Anything this document does not
name is not a tested hypothesis and its result may not be reported as one.

---

## 1. Why

`results/cost-model.md`'s code-search cost curve is built on `results/ws6a/`,
which is `fastapi`. The model has memorised `fastapi`: the parametric arm —
no repository access at all — scores **0.625** judge accuracy
(`results/ws6a/summary.csv`). On that corpus agentic grep is cheaper per query
than the index at every measured size ($0.036 vs $0.038, $0.036 vs $0.040,
$0.054 vs $0.067), so `break_even_qpd` is infinite and **the only finite
break-even in the whole code workload comes from the `stuffed` arm**, which
pays for a 960k-token prefix on every call. That is a strawman comparison and
it is not going on a slide as *the* cost curve.

`results/ws6c/` already shows the honest comparison exists. On
`agentic-hil/agentic-hil` — parametric judge accuracy **0.000**,
`any_file_hit_rate` **0.000** (`nonmemorization_gate.csv`) — agentic search
costs **$0.1387** a query against the index's **$0.0907**, and stuffing is not
feasible at all (4.31M corpus tokens, `fits_in_prefix_budget = False`). But
WS6c measured **one corpus size**. That is a break-even point, not a curve.

**WS9 measures `live = agentic search` against `index = claude-context indexed
retrieval` across corpus size on `agentic-hil`, and carries it through the
same WS5 machinery that produced the `fastapi` curve.** No `stuffed` arm
appears anywhere in it.

---

## 2. The instrument

### 2.1 What is frozen and what is new

Everything that defines the measurement is **reused unchanged** from WS6a and
WS6c: agent `claude-sonnet-5`, judge `claude-opus-5`, judge prompt
`results/ws6a/judge_prompt.txt`, embedding `text-embedding-3-small`,
`@zilliz/claude-context-mcp@0.1.15`, `WS6A_TURN_CAP = 15` on both tool-using
arms, the byte-identical system prompt that names no tool and no retrieval
strategy, `SEED = 42` question shuffling, and the statistical pins
`WS6C_BOOT_N = 10_000` / `WS6C_BOOT_SEED = 42` / `WS6C_ALPHA = 0.05`.

**No existing pin changes.** `benchlib/config.py`'s `WS6A_*` and `WS6C_*`
constants stay exactly as they are; WS9 only adds `WS9_*`. Prices, instance
SKU, seed, amortisation days and the `r8g` re-pin all stay as they are.

The corpus is the one the WS6c Phase 2 gate already cleared. **WS9 does not
re-gate it** — the gate is a property of the repository and the model, not of
how much of the repository is indexed, and re-running it would spend money to
re-derive a committed `0.000`.

### 2.2 The subsets

Nested, chosen by directory tier the way `WS6A_SUBSETS` was, **decided from
the tree and never from results**:

```python
WS9_SUBSETS = {
    "S": {"include": ("src/",), "exclude": ()},
    "M": {"include": (),        "exclude": ("tests/",)},
    "L": {"include": (),        "exclude": ()},
}
```

Measured from the pinned tree before this document was written:

| scope | tier | git-tracked | in derived tree | text-allowlisted | cc-indexable | tiktoken (text) | tiktoken (indexable) |
|---|---|---|---|---|---|---|---|
| S | `src/` | 49 | 49 | 49 | 46 | 741,343 | 722,831 |
| M | whole repo minus `tests/` | 212 | 211 | 172 | 112 | 1,352,243 | 1,142,900 |
| L | whole repo | **382** | 381 | 342 | 279 | 2,770,789 | 2,548,757 |

Log spacing **1.82×** then **2.05×** on text-allowlisted tokens; total span
**3.74×**.

The git-tracked and derived-tree columns differ by exactly one file at M and L:
the root-level `.dockerignore` the derived tree drops (§2.5). 382 is the figure
`git ls-files` and `index_cost.csv`'s `files_tracked` both report at L. The
indexable column reproduces `index_cost.csv`'s recorded
`indexed_tokens_tiktoken = 2,548,757` at L exactly, which is what makes the S
and M figures in the same column trustworthy as predictions.

**Why `tests/` is its own tier, stated before the run.** `tests/` is 170 files
and **51.2%** of corpus tokens — the single largest directory in the tree by a
wide margin. Putting it in its own step is the only division of this repository
that gives near-even log spacing: the literal `WS6A_SUBSETS` analogue
(`src/` → `src/`+`tools/`+`tests/` → all) spaces 3.06× then 1.22×, which puts M
and L almost on top of each other. `scripts/code-retrieval/scale_ws6a.py`'s own
guard calls that kind of spacing "badly conditioned — it would make the curve's
shape an artifact of how the subsets happen to divide," and that objection
applies here with more force, because WS9 has only 3.74× of span to spend.
"Index the source and the docs, not the test suite" is also a scope a real user
picks. M lands within 6% of the geometric midpoint of S and L (1,352,243 against
1,433,215), which is a consequence of the tiering, not a target it was fitted to.

**The span is small and this is a stated limitation, not a discovery.** WS6a
spanned 24× and WS6b 12.5×; WS9 spans 3.74× because `agentic-hil` is a
342-file repository whose library core is a quarter of it. No larger span
exists inside this corpus. This is recorded here so it cannot later be
presented as a surprise.

### 2.3 The question panel — eligibility decided from gold paths alone

A question is **eligible** iff **every** path in its `expected_files` lies
inside the **smallest** subset, `src/`. Applied to
`results/ws6c/questions.csv`:

- **37 of 40 eligible.** Excluded: `agentic_hil_q09`
  (`examples/nucleo-f446re_demo/Src/main.c`), `agentic_hil_q30`
  (`tools/bench_battery.py`), `agentic_hil_q31`
  (`tests/test_bench_mutex.py`).
- Difficulty mix of the panel: **12 locate / 15 trace / 10 multi_hop**.
- Both agentic turn-capped rows (`q21`, `q40`) are eligible and stay in.

**The same 37 questions run at every size.** Eligibility is evaluated once, at
S, and applied to S, M and L alike — so no question enters or leaves the panel
as the corpus grows. A curve whose points are drawn over different question
populations has a shape that is partly a composition artifact; this project has
caught that error twice already (`results/ws4e/composition_check.csv`,
`conversion_composition_check.csv`), and the fixed nested panel is what
forecloses it.

The pre-registered 10-question `scaling_subset` flag in `questions.csv` is
**not** used to select the panel. It is a subset of the 37 and is carried
through as a column so a reader can compute the narrower panel if they want it.

**Consequence, declared here rather than found later.** Restricting L to the
37-question panel moves the published WS6c L figures:

| panel | Q\* at `indexed` (k=10) | Q\* at `indexed_topk3` (k=3) |
|---|---|---|
| n = 40, as published in `results/code-retrieval.md` | 58.94/day | 70.44/day |
| n = 37, the WS9 panel | **63.79**/day | **82.51**/day |

Both are computed from the same frozen `results/ws6c/runs.csv` against the
same fixed daily cost; the difference is composition alone. **WS9's L point is
the n = 37 one**, because the curve must be one population end to end. The
n = 40 figures remain the published WS6c headline and both appear in the
report. The deck's `data/ws5/break_even_ws6c.csv` currently carries the n = 40
pair; §9 says what that means for it.

### 2.4 The arms

`WS9_ARMS = ("agentic", "indexed", "indexed_topk3")`.

- `agentic` — grep/read/glob over the subset tree. The live arm.
- `indexed` — `search_code` at the **measured shipped default**,
  `limit = 10`, provenance `results/ws6c/topk_provenance.txt`. Not a number
  chosen here.
- `indexed_topk3` — `limit` **bound to 3 in the wrapper and removed from the
  schema the model sees**, exactly as WS6c did, because the byte-identical
  system prompt cannot name a tool or a parameter. Verified from the committed
  transcripts. It runs at **every** size: the deck quotes both break-evens and
  the pair must hold across the curve, not only at L.

⚠️ **`indexed_topk3`'s L rows come from `results/ws6c/runs.csv`, not from
`results/ws6c/topk_runs.csv`.** `topk_runs.csv` is WS6c **Phase 1** — 40
`indexed_topk3` rows on **`fastapi`**, with WS6a-style qids (`q02`, `q39`).
`agentic-hil`'s 40 `indexed_topk3` rows are part of the 160-row Phase 3
`runs.csv`, qids `agentic_hil_q01`..`q40`. Reading `topk_runs.csv` for WS9's L
point would silently substitute the contaminated corpus for the uncontaminated
one at the arm the whole top-k half of this design rests on. Every read of L
rows goes through `runs.csv` filtered to `arm` and to the panel.

**No `stuffed` arm at any size.** `agentic-hil` measures 4.31M `count_tokens`
against a 960,000-token prefix budget — 4.4× over — so the arm is infeasible,
not omitted. `prefix_text` is `None` and `in_stuffed_prefix` stays blank.

**`parametric` enters at L only**, reused from `results/ws6c/runs.csv`. It has
no corpus access, so it is size-independent by construction and re-running it
at S and M would buy nothing. It is therefore **not a row in
`results/ws6c/scaling.csv`** — that file carries the three sweep arms only, at
three sizes. It enters WS5 directly from `results/ws6c/runs.csv` as
`role = "floor"` at L, aggregated over **the same 37-question panel** as every
other WS9 point, and it is gated like every other point.

### 2.5 Trees and indexes

Subsets are materialised **from the derived L tree**
(`data/ws6c_tree_agentic_hil`), not from the git checkout, via
`benchlib.code_retrieval.topk.materialise_index_tree` — which sha256-verifies
every file both ways and compares the file set in both directions. Destinations
`data/ws9_subset_s` and `data/ws9_subset_m`.

Two consequences, both deliberate:

1. The root-level `.dockerignore` that makes the pinned package index **zero
   files while reporting success** is already absent from the L tree, so the
   bug cannot recur at S or M.
2. Each subset is **provably a subset of the exact corpus WS6c measured**, byte
   for byte, so the three size points sit on one corpus rather than three
   nearly-identical ones.

**claude-context is pointed at the same absolute path the agentic arm greps.**
The package keys an index by absolute codebase path; indexing one path and
searching another measures an empty index. This is the failure
`scripts/code-retrieval/scale_ws6a.py` documents and it is the reason the
subsets are persisted directories rather than temporaries.

**Completeness is asserted three ways per scope before any run spends**, as
`index_ws6c.py` does: the server's own stderr completion line *and* its
`Processed N/T` denominator; an independent Milvus `count(*)` after an explicit
flush; and `claude_context_indexable()` computed from the tree we handed it.
All three must agree. `get_indexing_status` and `get_collection_stats` are not
trusted — both were caught lying during WS6a.

M's indexable corpus is ~1.14M tiktoken, over the account's 1M TPM embedding
limit, so `--embedding-batch-size 20` stays and is recorded per scope. It is a
throughput throttle: it changes how many chunks per request, never the chunks
or the vectors.

**Collection pre-flight, before any embedding spend.** Collection names hash
the indexed path, so WS9 adds two (`ws9_s`, `ws9_m`) alongside whatever WS6a
and WS6c left behind. `index_ws6c.py`'s guard already refuses to verify when
more than one collection matches a prefix, because picking one at random would
let the cross-check pass against a collection nothing queries. WS9 keeps that
guard and adds an explicit pre-flight listing of existing collections and the
account's collection limit, so a limit is hit before money is spent rather than
at 60% of an index.

---

## 3. Hypotheses, stated before running

- **H1 — the saving widens with corpus size.** Agentic cost rises with how
  much tree it must search; indexed cost is set by a fixed batch of ranked
  chunks and should be flatter. So `c_live − c_idx` grows with size and Q\*
  falls with size.
- **H2 — at S the agentic arm may be *cheaper* than the index, making Q\*
  infinite there.** A 49-file tree is cheap to grep. **What H2 firing would
  establish is that size dependence exists on an uncontaminated corpus too** —
  that grep's advantage is partly a function of how much tree there is, on a
  corpus with a **0.000** parametric floor. That is a real and reportable
  finding about scale.
- **H3 — the fixed cost does not move.** See §3.1.

**H2 is the hypothesis this design exists to be able to lose to**, and the
report must state its outcome whichever way it falls.

**What H2 does *not* establish, stated now so it cannot be claimed later.** H2
firing does **not** make scale an alternative explanation for the `fastapi`
result, and the report may not say it does. Grep wins on `fastapi` at
**6,232,509** `count_tokens` and loses on `agentic-hil` at **4,307,363** — the
larger corpus is the one where grep wins. A monotone size effect cannot produce
that ordering, so the size-matched contrast survives H2 intact. The contamination
question is settled by the matched-size comparison in §3.2, not by H2.

### 3.1 A structural fact, declared rather than discovered (H3)

Under the headline infra mode (dedicated `r8g.large`, `pca_uc_384_sq8`), the
index's fixed daily cost is **flat across all three sizes**:

| scope | predicted chunks | index bytes @ 8.66× | instances | `fixed_daily` |
|---|---|---|---|---|
| S | ~2,110 | ~1.5 MB | 1 | $2.828 |
| M | ~3,337 | ~2.4 MB | 1 | $2.828 |
| L | 7,441 (measured) | 5.3 MB | 1 | $2.828 |

**Basis of the S and M chunk predictions:** the measured L chunk count (7,441)
scaled by each scope's share of L's indexable tiktoken — S 722,831/2,548,757 =
0.2836, M 1,142,900/2,548,757 = 0.4484. That assumes a constant mean chunk size
(L measured 342.5 tokens/chunk), which is an assumption about the AST splitter's
behaviour on a subset, not a measurement. It is used **only** to show in advance
that every scope lands on one instance; the numbers that reach WS5 are the
`count(*)` measurements in `index_cost.csv`.

A 16 GiB instance holds a 5 MB index as easily as a 1.5 MB one, and the embed
amortisation is $0.00014/day. **So the shape of this curve is set entirely by
the per-query saving; the footprint is load-bearing only in marginal mode.**
The report says this in these words. Chunk counts for S and M are predictions
here and are replaced by measurements in `index_cost.csv`; the L figure is
measured.

### 3.2 The contamination test — matched by corpus size (C-series)

The `fastapi`/`agentic-hil` difference is attributed to contamination only if a
comparison **at matched corpus size** shows it. Registered here, before the S
and M sizes are measured.

**Pairing rule, fixed in advance:** two scopes are *matched* iff their measured
`count_tokens` are within **2.0×** of each other. Pairing is by measured tokens,
never by file count — WS6a §7's rule, and the rule `run_ws6c.py`'s provenance
already applies. Predicted pairs, to be re-evaluated against measurements:

| pair | `fastapi` `count_tokens` | `agentic-hil` `count_tokens` | ratio | matched? |
|---|---|---|---|---|
| `fastapi` M vs `agentic-hil` S | 1,138,485 | ~1,152,464 | ~1.01 | yes — the tightest available |
| `fastapi` L vs `agentic-hil` L | 6,232,509 | 4,307,363 | 1.45 | yes |
| `fastapi` M vs `agentic-hil` M | 1,138,485 | ~2,102,145 | ~1.85 | yes |
| `fastapi` L vs `agentic-hil` M | 6,232,509 | ~2,102,145 | ~2.97 | no |
| `fastapi` S vs `agentic-hil` S | 259,429 | ~1,152,464 | ~4.44 | no |

`agentic-hil` S and M `count_tokens` are predictions here (tiktoken × the 1.5546
ratio measured at L) and the band is applied to the measurements in
`scaling_corpus_stats.csv`, not to these. A pair that leaves the band on
measurement drops out; none is added.

**Inputs.** `fastapi` S and M from `results/ws6a/scaling.csv` (n = 10 per
cell), `fastapi` L from `results/ws6a/summary.csv` (n = 40); `agentic-hil` from
`results/ws6c/scaling.csv` (n = 37 per cell). `fastapi` has no
`indexed_topk3` arm at S or M — Phase 1 ran the knob at L only — so the
C-series is defined over `agentic` vs `indexed` alone and makes no claim about
top-k across corpora.

**Branches.** Resolved over which arm is cheaper per query (`agentic` vs
`indexed`, billed, judge excluded) on each side of a matched pair:

- **C1** — the per-query winner **differs** across corpora at matched size
  (grep cheaper on `fastapi`, index cheaper on `agentic-hil`). Size alone does
  not explain the `fastapi` result; contamination remains the candidate
  explanation.
- **C2** — the per-query winner is the **same** on both corpora at matched
  size. The matched comparison does not support the corpus-identity
  explanation, and the report says so.
- **C3** — no pair falls inside the band on measurement, so the test does not
  run and nothing is concluded from it.

**This comparison is not paired and no interval is computed across it.** The
two corpora have different questions, different n (10 or 40 on `fastapi` against
37 here) and no correspondence between rows. WS6c is explicit that no
cross-corpus interval may be paired; C-series is a **described comparison of
point estimates**, exactly as WS6c's own sign reversal is, and it carries that
label in the output.

**The confound cuts against C1, which is the useful direction.** The matched
pairs are matched on tokens and badly mismatched on file count — `fastapi` M is
**1,112** files against `agentic-hil` S's **49**, a 23× gap; at L it is 2,888
against 342, the 8.4× gap WS6c already documents. Grep's cost scales with how
much tree it must search, so a small-file-count haystack **favours the agentic
arm on the `agentic-hil` side of every pair**. That is the arm C1 requires to
lose there. A C1 result is therefore obtained against the confound, not with
it; a C2 result is partly explicable by it and the report must say so.

---

## 4. Pre-registered branches

New series. **No WS6c letter is borrowed** — the A/T/S/P/K series are defined
over WS6c's comparisons and none of them covers a size sweep on this corpus.
Cells outside the series below emit `n/a (not pre-registered)`, exactly as
`results/ws6c/branches.csv` does, and no letter here refers to anything but the
quantity its series names.

**F — finiteness of break-even on the unseen corpus** (over the `agentic` vs
`indexed` pair at the headline infra mode and footprint, Claude regime):

- **F1** — `break_even_qpd` finite at S, M **and** L.
- **F2** — finite at some sizes and infinite at others; the report names which.
- **F3** — infinite at every size (live cheaper per query throughout).

**B — the abstract's band** (only over sizes where F gives a finite Q\*):

- **B1** — every finite Q\* inside [10, 1000) queries/day.
- **B2** — at least one finite Q\* outside it; the report names the size and
  the number.

**K — the top-k ordering across the curve**:

- **K1** — Q\*(`indexed`, k=10) < Q\*(`indexed_topk3`, k=3) at every size:
  the order the deck quotes at L is preserved across the curve.
- **K2** — the order reverses at some size: the two cross below L.
- **K3** — at least one size has an infinite Q\* on one arm, so the pair is
  not ordered there.

**U — per-size paired token deltas**, resolved independently for each
(contrast, size) cell over `prompt_tokens`. Contrasts: `indexed − agentic`,
`indexed_topk3 − agentic`, `indexed_topk3 − indexed`.

- **U1** — 95% CI excludes zero and is negative (the index arm costs fewer
  tokens).
- **U2** — CI excludes zero and is positive.
- **U3** — CI includes zero: report the point estimate with its interval and
  **claim neither direction**.

`judge_correct` and `turns` deltas are computed and published as supporting
mechanism and carry `n/a (not pre-registered)`. **No U-letter anywhere refers
to anything but a `prompt_tokens` delta.**

**The U-series is nine unadjusted cells at α = 0.05** — 3 contrasts × 3 sizes,
each resolved against its own 95% interval with **no multiplicity correction**.
Under a global null that is ~0.45 expected U1/U2 resolutions by chance across
the nine. The correction is deliberately not applied, because the alternative —
choosing a correction after seeing how many cells resolved — is a worse analyst
degree of freedom than declaring the unadjusted rate here. **A single isolated
U1 or U2 among nine cells is therefore not evidence of anything and the report
must not present one as a finding**; only a coherent pattern across sizes in one
contrast may be discussed, and even then as a pattern, not as nine independent
tests.

**The statistical knobs are not analyst choices on the day.** `WS6C_ALPHA`
(0.05), `WS6C_BOOT_SEED` (42) and `WS6C_BOOT_N` (10,000) are reused frozen. A
cell that resolves U3 may **not** be re-resolved at a looser alpha, a different
seed or a larger resample count. WS6c measured exactly this: its near-miss was
robust to every seed and resample count thrown at it and fragile to alpha and
to dropping one question in three -- and those last two are analyst choices,
which is why the branch was fixed in writing before the data existed. Seed and
alpha sensitivity may be *reported* as a sensitivity check, labelled as such,
and it never displaces the committed branch.

The per-arm size-span deltas (L−S, L−M within an arm) are computed in the
`paired_ci_scaling.csv` shape as a dilution curve. They are **exploratory** and
carry `n/a (not pre-registered)`: WS6a's equivalent found no detectable growth
over a 24× span at n=10, and WS9 has 3.74× of span, so nothing is expected and
nothing is claimed.

### 4.1 What is expected to be underpowered, said in advance

WS6c Phase 3 at n = 40 reached **P3** on its primary comparison — the interval
missed excluding zero by 85 tokens. WS9's per-size cells are n = 37 on a
smaller span. **U3 at every size is the expected outcome and is not a
disappointment.** The value of WS9 is the *break-even curve*, which is a ratio
of means and does not need a separating interval; the U-series exists so the
report cannot quietly upgrade a point estimate into a direction.

Corollary, binding: **the report may state the curve and its point estimates.
It may not say the index wins at a size whose U-cell resolved U3.**

---

## 5. Outputs

Every file is sha256'd into `data/MANIFEST.json` via `benchlib.manifest.record`
and `python scripts/verify_checksums.py` must run clean.

| file | what |
|---|---|
| `results/ws6c/scaling.csv` | 333 rows = 37 questions × 3 arms × 3 sizes, in `results/ws6a/scaling.csv`'s schema plus a `corpus` column. S and M are new runs; L rows are **reused verbatim** from the frozen `results/ws6c/runs.csv`, filtered to the panel — not re-run. |
| `results/ws6c/index_cost.csv` | gains a `scope` column. The existing row is backfilled as `L` with **every existing value byte-identical**; S and M rows are added. |
| `results/ws6c/scaling_corpus_stats.csv` | per-scope `files`, `lines`, `chars`, `count_tokens`, `tiktoken_cl100k`, `tree`, `tree_sha256`, `fits_in_prefix_budget`. `count_tokens` is measured against the pinned agent model with WS6a's measured per-request framing constant subtracted. |
| `results/ws6c/paired_ci_unseen_scaling.csv` | paired median deltas, 10,000-resample percentile bootstrap, exact two-sided sign test, in `paired_ci_scaling.csv`'s shape plus `corpus` and `size`. Carries the per-size arm contrasts (U-series) **and** the per-arm size spans. |
| `results/ws6c/matched_size_contamination.csv` | the C-series (§3.2): one row per candidate pair with both scopes' measured `count_tokens`, the ratio, `in_band`, both sides' file counts, each side's `agentic` and `indexed` mean billed $/query and n, the cheaper arm on each side, and the branch. Carries `unpaired: no interval is computed across corpora` in a column, not only in prose. |
| `results/ws6c/scaling_provenance.txt` | the run record plus a summary of this pre-registration, so the public tree still carries the fixed-in-advance statement after `docs/` is stripped at release. |
| `results/ws5/*` | `break_even.csv`, `churn.csv`, `sensitivity.csv`, `postdiction_gate.csv`, `chart_break_even.png` regenerated by papermill. |

`scaling.csv` keeps `scale_ws6a.py`'s guard that `corpus_tokens` must increase
strictly S < M < L, asserted against `scaling_corpus_stats.csv` before the file
is written. The x-axis is measured `count_tokens`, not file count: S, M and L
step 3.5x then 2.0x in files but 1.8x then 2.1x in tokens, and a curve drawn
against the badly-conditioned axis would have a shape set by how the subsets
happen to divide.

**`paired_ci_unseen_scaling.csv` is a new file, not an append.**
`results/ws6c/paired_ci_scaling.csv` holds WS6a's frozen **fastapi** rows;
appending `agentic-hil` rows to it would put two corpora in one file whose
schema has no corpus column, which is how a cross-corpus interval gets computed
by accident. WS6c's own report is explicit that no interval may be paired
across the two corpora.

---

## 6. Wiring into WS5

In `scripts/cost-model/build_ws5_notebook.py`:

1. **A `code_unseen` block in the `pts` table**, beside the existing `code`
   block, carrying **four** arms and their roles: `agentic → live`,
   `indexed → index`, **`indexed_topk3 → index`**, `parametric → floor`.
   `n_vectors` from the per-scope `chunk_count`, `embed_tokens` from the
   per-scope `corpus_tokens_tiktoken` — exactly as the code block does.
   `hit_rate` is computed the same way for every point; WS6c measured
   `cache_read = cache_creation = 0` on all 160 rows, so it evaluates to 0.

   **`indexed_topk3` must be a row in `pts` with `role = "index"`**, not a
   variant handled downstream. The postdiction gate iterates `points` and
   branches on `role`, so a `k = 3` arm that is not a `pts` row is **not
   gated** — and the brief requires the gate to cover every new point. Giving
   it `role = "index"` puts it through `cost_model.index_query_cost`, which is
   the right arithmetic for it (retrieved chunks and question uncached, no
   prefix).

2. **`(live, idx)` pairs at S, M and L, keyed by arm.** The existing code block
   pairs with `idx = C[C.arm == "indexed"].iloc[0]` — a single hardcoded index
   arm — so `code_unseen` cannot reuse that shape: it would silently drop the
   `k = 3` arm from `break_even.csv` while still gating it. The `code_unseen`
   pairing iterates **`for idx_arm in ("indexed", "indexed_topk3")`** against
   `agentic` as the only live arm, so `break_even.csv` carries a
   `break_even_qpd` per top-k at every size and `index_arm` distinguishes them.
   Two rows now share `(workload, size, live_arm)` and differ only in
   `index_arm`; every downstream filter in the notebook that selects code rows
   must key on `index_arm` too, and the chart's two markers come from that
   column.
3. **The postdiction gate covers every new point on both asserted bases**
   (`mean_billed`, `mean_uncached_list`) at the same **±10%**. The tolerance is
   not widened. **If it fails, the work stops and the failure is reported with
   its cause — no line is drawn.**
4. **The series is added to `chart_break_even.png`** so the unseen-corpus curve
   is visible next to the `fastapi` one, with the two top-k arms
   distinguishable by marker, and the dashed modelled line drawn only between
   measured sizes.
5. **`churn.csv` and `sensitivity.csv` carry it**, including the
   dedicated-vs-marginal infra knob, which is the dominant lever in the model.
   Both currently hardcode `mem_live`/`mem_idx`; both gain a `workload` column
   and the `code_unseen` pair threaded through. Row counts change.

Regenerated with papermill, notebook committed in place.

### 6.1 The index footprint is derived, not estimated

`agentic-hil`'s index footprint is **not a new estimate and must not be
presented as one.** It is derived the way WS5 already derives WS6a's: chunk
count × `EMBED_DIMS` through `cost_model.index_footprint_bytes` at the headline
(`pca_uc_384_sq8`, 8.66×) and conservative (`sq8`, 3.54×) footprints. The
chunk counts are measured per scope by `count(*)` against Milvus.

The report says this explicitly, because the deck currently **refuses the
marginal and serverless rows for this corpus on the grounds that no footprint
exists** (`data/SOURCES.md`). It exists by exactly the same derivation the
`fastapi` rows already use.

The footprint multipliers themselves carry WS5's standing caveat unchanged:
measured at 10M × 1024d on `mxbai-embed-large-v1` and applied as ratios to
1536d `text-embedding-3-small` vectors — a transfer, not a re-measurement.

---

## 7. Budget

`WS9_BUDGET_USD = 55.00`, hard.

Sized from **measured** WS6c per-run cost on this corpus, agent plus judge:
`agentic` $0.1489, `indexed` $0.1026, `indexed_topk3` $0.1110 — $0.3625 per
question per size across the three arms.

| item | runs | basis |
|---|---|---|
| sweep: 37 questions × 3 arms × 2 sizes | 222 | $26.83 |
| of which the smoke check re-uses, not re-bills | (12) | ($1.45) |
| **declared basis** | **222** | **$26.83** |
| **cap** | | **$55.00** (2.05× headroom) |

**The smoke check shares the sweep's checkpoint, so its rows are re-used and
not re-billed.** `run_ws6c.py` does the same — its `--smoke N` slices
`questions.head(N)` and writes into the same `run_<slug>.jsonl` the full run
reads back — and WS9 follows it rather than inventing a second convention. The
12 smoke runs are the first 2 panel questions × 3 arms × 2 sizes and are the
first 12 of the 222; there is no separate smoke line in the budget.

**Declared consequence: 12 panel rows are visible before the sweep runs.** That
is acceptable and it is bounded here rather than left implicit. The smoke check
is a **mechanism** check only — both indexed arms must show `turns > 1`, and
the bound arm's tool frames must read `Found 3 results` — and **its outcome may
not change the panel, the arms, the subsets, the branches or the budget.** The
only decisions it may drive are *stop* (a mechanism failure) or *continue*. Any
other use of those 12 rows is a post-hoc design change made with data in view.

The headroom is 2.05×, chosen against the precedent recorded in
`benchlib/config.py`: WS6c's own estimate came in **1.6–1.9× low**.

Embedding spend is ≈$0.04 (S ~0.72M + M ~1.14M indexable tiktoken at
$0.02/MTok). It is OpenAI-denominated and therefore **outside** the Anthropic
governor; it is recorded in `index_cost.csv` as `DERIVED` on the same basis
WS6c used, because the package surfaces no embedding usage field.

### 7.1 The governor is WS9-scoped, and this is a deliberate deviation

`agent_harness.total_spent_across_checkpoints` deliberately fails closed across
**both** WS6a and WS6c checkpoints. Seeding WS9's governor from it would start
WS9 at ~$70 already spent against a $55 cap and trip on the first call.

WS9 therefore gets its own accumulator over `data/ws9_cache/*.jsonl`,
**cumulative across WS9's own phases** (smoke, S, M) so that no script gets a
fresh budget — that per-script reset is the I-2 defect the original comment
describes and it is not being reintroduced. The justification is that WS9 is a
separately authorised budget, not a later phase of WS6a's $75 or WS6c's $45.
This is recorded in `scaling_provenance.txt` as well as here.

### 7.2 Stop rules

- **Stop at the cap.** Ask before raising it. Never raise it unilaterally.
- **No arm is dropped mid-run with data already visible.** That is a post-hoc
  design change made with data in view, which is exactly what pre-registration
  exists to prevent, and it already happened once in WS6c (which raised its cap
  from $40 to $45 rather than drop an arm with 65 of 160 rows visible). If the
  budget proves insufficient, the run stops and reports what it has; it does
  not quietly become a two-arm design.
- **No question is dropped after the panel is fixed by §2.3.**
- **The postdiction gate is not widened.** A failure stops the work.

---

## 8. Order of work

1. Commit this spec. **No spend.**
2. `WS9_*` constants, subset/eligibility helpers, unit tests. **No spend.**
3. Per-scope corpus measurement → `scaling_corpus_stats.csv`. Anthropic
   `count_tokens` only, pennies.
4. Index S and M; three-way completeness assert per scope; per-scope rows into
   `index_cost.csv`. OpenAI embedding, ≈$0.04.
5. **Smoke: 2 questions × 3 arms × 2 sizes, no CSV written**, into the sweep's
   own checkpoint (§7). Both indexed arms must show `turns > 1` — a 1-turn
   indexed row answered from parametric memory is the exact WS6a failure this
   check exists to catch — and the bound arm must show `Found 3 results` in its
   tool frames. Mechanism only: stop or continue, nothing else.
6. Full sweep, checkpointed and resumable, governor live. It resumes over the
   12 smoke rows rather than re-billing them.
7. Bootstrap → `paired_ci_unseen_scaling.csv`.
8. WS5 rewire; papermill; **postdiction gate must pass before any chart is
   read**.
9. Reports (§9); manifest; `scripts/verify_checksums.py` clean; full test suite.

---

## 9. Reporting

`results/cost-model.md` and the WS6c part of `results/code-retrieval.md` are
updated to state plainly:

1. **Whether break-even is finite at every measured size on the unseen
   corpus** (branch F), and where it lands against the abstract's "tens to
   hundreds a day" (branch B).
2. **The direction of the `fastapi`-vs-`agentic-hil` difference, and that
   contamination is the candidate explanation rather than a proven one** —
   settled by the **matched-size C-series** (§3.2), not by H2, and reported
   with the pairs, their token ratios, their file-count mismatch and the
   statement that the comparison is unpaired and carries no interval. The H2
   outcome is stated separately as a finding about **scale on an uncontaminated
   corpus**; if H2 fired, the report says so and also says why it does not make
   scale an alternative explanation for the `fastapi` result — grep wins there
   at 6.23M tokens and loses here at 4.31M, which no monotone size effect
   produces.
3. **The caveats that already exist at L and apply to any curve drawn through
   it**, carried forward verbatim rather than dropped:
   - the paired median interval for `indexed` vs `agentic` prompt tokens
     **straddles zero** (`sign_p` 0.081, `results/ws6c/branches.csv`, branch
     P3);
   - the agentic arm **hit the turn cap on 2 of 40 runs**, so the saving is the
     **conservative end**;
   - the per-arm medians in `results/ws6c/summary.csv` carry the **opposite
     sign** to the paired deltas on this same data — marginal medians are not
     tested comparisons.
4. **Whether the k=10 and k=3 break-evens keep their order across the curve, or
   cross somewhere below L** (branch K).
5. **n at each new size, and whether it can carry a claim at all** — with §4.1's
   pre-declared expectation that U3 is the likely outcome, so a U3 result reads
   as predicted rather than as a failure discovered late.
6. **That the index footprint is derived, not estimated** (§6.1), naming the
   deck's current refusal of the marginal and serverless rows as the thing it
   retires.
7. **That the panel is n = 37 and the published WS6c headline is n = 40**, with
   both pairs of L figures, so no reader finds the discrepancy unannounced.
8. **That the curve spans 3.74×** and that under the dedicated headline the
   fixed cost is flat across it (§3.1), so the shape is the per-query saving
   and nothing else.

**If the curve does not hold up, that is said prominently. The honest framing
beats the clean number.** A result that retires WS9's own premise — F3, or a
B2 far outside the band — is reported as the finding, not smoothed over.

### 9.1 Handoff to the deck

The talk's slide deck (a separate, unpublished repository)
currently **derives** `data/ws5/break_even_ws6c.csv` in `data/derive.js`, from
`results/ws6c/summary.csv` plus a fixed-cost model recovered out of
`results/ws5/break_even.csv`, because upstream never modelled this corpus. Once
`break_even.csv` carries real `code_unseen` rows the handoff names:

- which deck-side derivation that retires, so the deck can **copy instead of
  derive**;
- the branch and commit to pin in the deck's `data/SOURCES.md`;
- the row-count and column changes in `churn.csv` and `sensitivity.csv` that
  the deck copies;
- the n = 40 → n = 37 change in the L figures, since the deck currently shows
  58.94 and 70.44.
