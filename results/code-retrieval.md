# Agentic code and memory retrieval

Covers `results/ws6a/` — a code-search benchmark, claude-context's indexed
`search_code` against an agentic grep/read loop on `fastapi/fastapi`;
`results/ws6b/` — an agent-memory benchmark, memsearch against full-history
replay on a synthetic agent history; and `results/ws6c/` — the top-k knob, an
uncontaminated repository, and a paired re-analysis of `results/ws6a/`'s own
frozen rows. Claims C1, C10, C11, C12, C15. The per-query token traces from
`results/ws6a/` and `results/ws6b/` also feed `results/cost-model.md`.

Note: `results/ws6a/mcp_server_provenance.txt` and
`results/ws6c/run_provenance.txt` are committed provenance from the original
runs. Both name `scripts/` and `benchlib/` paths as they stood at run time,
before this repo's modules and scripts were regrouped into topic directories
(2026-09-13) — so `run_provenance.txt`'s "benchlib/ws6c.py", for instance, is
today `benchlib/code_retrieval/topk.py`. They are historical records and are
intentionally not rewritten: a reader should map the paths they name onto the
current tree rather than expect them to resolve.

Every number traces to a committed CSV in those three directories, cited inline.
The agent transcripts are committed too — `results/ws6a/transcripts/`,
`results/ws6c/transcripts_topk/`, `transcripts_agentic_hil/`,
`transcripts_gate_*/`, and the WS9 size sweep's `transcripts_ws9_s/` and
`transcripts_ws9_m/` — and they are **the evidence**, not decoration. They are
how a reader checks that `search_code` was really called at the top-k claimed
(the default arm's tool results read `Found 10 results`, the constrained arm's
read `Found 3 results`), that the model on the clean corpora declined rather than
guessed, and that the payload sizes below are measurements, not estimates.

**How to read this.** "The question, and the answer" states what the three arms
found. "The correction" is the most important section here: `results/ws6c/`
revised magnitudes `results/ws6a/` had already published, and both the original
and corrected figures are set out there. A reader who wants only the conclusion
can stop at the end of it. "What was measured" is the instrument, arm by arm,
and the gates each passed or failed; "Limits and caveats" is the audit trail and
is long on purpose; the appendix holds the per-arm tables. `results/ws6c/` now
also carries a **size sweep** on the unseen repository (`scaling.csv`,
`scaling_corpus_stats.csv`, `paired_ci_unseen_scaling.csv`,
`matched_size_contamination.csv`; WS9, run 2026-09-15) — the same three arms at
three nested corpus sizes on one fixed panel of 37 questions. It is the last
subsection of section C, and it is what `results/cost-model.md`'s "second curve"
is drawn from.

---

## The question, and the answer

**1. The vendor's ~40% token-reduction claim was not reproduced at any tested
configuration, on either repository.** On `fastapi` the indexed arm cost *more*
tokens than agentic grep/read — the direction inverted. That inversion is real
but thin, and it is **specific to the shipped default top-k on a greppable,
heavily-trained-on repository**: constrain the knob and the index's token penalty
does not survive — neither direction can then be claimed; move to a repository the
model has never seen and the point estimate changes sign.
**C11 can no longer be stated as a one-sided negative result.** The magnitudes
originally published for it were wrong — see the next section.

**2. Contamination is the largest single effect measured anywhere here.** A model
with **no repository access at all** answered **0.625** of the `fastapi`
questions correctly (`results/ws6a/summary.csv`). The three retrieval arms on
that corpus span 10pp — stuffed 0.900 to indexed 1.000 — sitting on a 62.5pp
floor: **the arms are separated by less than the floor they all stand on.** The
same probe scores **0.0000** on a synthetic agent history
(`results/ws6b/parametric_gate.csv`) and **0.000** on each of three post-cutoff
real repositories (`results/ws6c/nonmemorization_gate.csv`). On a corpus this
well represented in training the question is not "does retrieval work" but "does
retrieval beat what the model already knows" — and any benchmark on a famous
repository that omits a parametric arm is measuring something else.

**3. The shipped default is the configuration that loses.** The pinned package's
own `dist/handlers.js` carries a measured `limit = 10`
(`results/ws6c/topk_provenance.txt`). At that default the indexed arm costs more
tokens than grep — thinly, on the corrected paired figures below. Binding `limit` to **3** cut indexed prompt tokens by a paired
median **4,429.5**, CI **[−8,385.0, −1,890.0]** — the knob demonstrably works —
and left a gap to agentic of **+584.5**, CI **[−2,572.0, +2,405.0]**:
pre-registered branch **T3**, which forbids claiming *either* direction
(`results/ws6c/topk_summary.csv`, `branches.csv`). It costs **7.5pp** of accuracy
(1.000 → 0.925). The only indexed configuration that competes on tokens is one a
user must bind by hand, at a measurable accuracy price — and "low effort" (C1) is
precisely a claim about the default.

**4. Silent failure is a pattern, not an anecdote.** Two retrieval tools,
**both published by Zilliz but independently built**, read Milvus sealed-segment
statistics as a liveness signal, and both fail quietly
while still returning confident answers. The shared publisher makes the point
*stronger*, not weaker: one company made the same wrong assumption about its own
database twice, in two separate codebases.
claude-context's `get_indexing_status` reported **"✅ fully indexed — 1100 files,
1100 chunks" 19 seconds in** while the server's own log showed **330/2,825
files**, leaving a collection holding **21% of the repository that answers
queries perfectly happily**. memsearch, after `Indexed 52 chunks.` with `rc=0`,
returned **`No results found.` with `rc=0` and no error** on a collection Milvus
held **52 rows** for. A third instance, found later against the same pinned package,
indexed **0 files and reported completion**. The indexed path's real cost is not step count
but that it has more surfaces that fail quietly while continuing to return
confident answers.

**5. Where indexed memory wins it wins on cost, and the one quality separation is
a window cliff.** Against full-history replay, memsearch is cheaper at every
tested history size — **$0.01004 / $0.01020 / $0.02318** per question against
replay's **$0.01994 / $0.07890 / $0.19405**, ratios 2.0×/7.7×/8.4× — with judge
accuracy 1.000 for both arms at S and M, and 1.000 for memsearch at every size;
truncated replay scores 0.000 on the `scaling` set at L, which is arithmetic —
those facts sit outside its window — not a result. **No crossover exists
inside the tested range**, so it lies below the smallest history tested and this
evidence does not bracket it. Quality separates in one place: at a ~959k-token
context, truncated replay scores **10/12** on facts it *can* see while memsearch
scores **12/12**.

**6. Unfamiliar code costs more per question — a result, not noise.** Both
tool-using arms cost **1.3–2.3× more per question** on a repository the model has
never seen than on `fastapi`, at a 0.691 token scope ratio. ⚠️ **Do not conflate
this with the 1.6–1.9× recorded in `benchlib/config.py`'s budget comment** — that
is a different ratio on a different base: *measured* per-pair cost against the
*plan's estimate*. This one is measured-against-measured, across corpora. The
agentic arm is
the extreme: **$0.1488 against $0.0659**. Its median turns rise 3.5 → 4.0, its
turn-cap rate 0.025 → **0.05**, and its accuracy falls furthest, 0.975 →
**0.900**, while the indexed arm holds at 0.975. Grep is cheapest exactly where
the model already knows the code — which is where retrieval is least needed.

---

## The correction — `results/ws6c/` revises `results/ws6a/`'s published magnitudes

`results/ws6a/` is a **paired** design: the same 40 questions run in every arm.
Its headline deltas were published as **differences of medians**, which is not a
paired statistic. On right-skewed arms (agentic max 186,597, indexed max 234,923
prompt tokens) a difference of medians systematically overstates the typical
per-question effect. `results/ws6c/` Phase 0 re-analysed the frozen rows — zero
API spend, no re-run — with a paired median delta, a 10,000-resample percentile
bootstrap (seed 42, α = 0.05) and an exact two-sided sign test. The design
supported the paired number all along; the difference of medians is the one that
reached the slide.

`results/ws6a/` itself is **frozen and unchanged**; only its prose was corrected.
Both sets of figures appear below. The originals are historical and must not be
quoted as current.

**The token headline** (`results/ws6c/paired_ci.csv`):

| statistic | value |
|---|---|
| **Originally published** — difference of medians, `results/ws6a/summary.csv` | 16,452 − 9,485 = **+6,967 (+73.5%)** |
| **Corrected** — paired median delta | **+2,856.5** |
| 95% CI | **[+9.0, +7,691.5]** |
| Questions where indexed cost more | **26/40** |
| Exact two-sided sign test | p = **0.081** |
| Pre-registered branch | **A1** — the CI excludes zero |

**The published figure is 2.44× the paired one.** Branch **A1** holds — the
inversion is real and survives an interval — but it is **thin**: the lower bound
is **nine tokens** and the sign test does not reach 0.05. Quote "+2,857 tokens at
the median, CI [+9, +7,692]", never "+73.5%".

**The accuracy gap that was never a gap:**

| statistic | value |
|---|---|
| **Originally published** | 1.000 vs 0.975, "+2.5pp (1 question)" |
| **Corrected** — paired median delta | **0.0** |
| 95% CI | **[0.0, 0.0]** |
| Questions where the arms differed | **1/40** (`q40`, agentic, which hit the turn cap) |
| Sign test | p = **1.0** |
| Pre-registered branch | **A3** — accuracy is *not separable at n = 40* |

**The accuracy claim is withdrawn.** 1.000 vs 0.975 was never a difference and
must not be quoted as one.

**The within-repository dilution curve.** `results/ws6a/` also published
"**agentic grows 35%** (6,040 → 8,151)" across a 24× corpus increase — the same
defect, a difference of medians across size points where the *same 10 questions*
run at every point. Paired, over the same frozen `scaling.csv`
(`results/ws6c/paired_ci_scaling.csv`):

| arm | span | paired median | 95% CI | grew | branch |
|---|---|---|---|---|---|
| agentic | L − S | **+1,041.0** | **[−108.0, +3,889.5]** | 5/10 | **A3** |
| agentic | L − M | +638.0 | [−40.0, +3,021.0] | 6/10 | **A3** |
| indexed | L − S | −2,223.0 | [−6,529.0, +11,375.5] | 4/10 | **A3** |
| indexed | L − M | +646.5 | [−5,440.0, +3,895.5] | 6/10 | **A3** |

**The conclusion survives; the stated magnitude does not.** Only 5 of 10
questions grew at all from S to L, and every interval crosses zero. "Neither arm
scales meaningfully over a 24× corpus increase" becomes **"no growth is
detectable at n = 10 per cell, in either arm, at either span"** — weaker about
the data, a *stronger* negative result for C12's within-repository half. **Do not
quote "35%".** Judge accuracy is 1.000 in all six cells, so every accuracy delta
is 0.0 with a [0.0, 0.0] interval; there is no quality curve either.

**Why this is not pedantry: the same error, reproduced live, flipping a sign.**
On the uncontaminated repository, `indexed_topk3` against `indexed`, same 80 rows:

| statistic | `indexed_topk3` | `indexed` | conclusion |
|---|---|---|---|
| **Marginal** median prompt tokens (`results/ws6c/summary.csv`) | 25,977.5 | 18,078.5 | topk3 costs **+7,899 more** |
| **Paired** median delta (`results/ws6c/branches.csv`) | — | — | topk3 costs **−3,961.5 less** |

**Same rows. Opposite conclusions. Opposite signs.** On `fastapi` the error cost
a 2.44× overstatement; here it would have cost the sign. Every marginal per-arm
total in this document is labelled marginal for that reason; only the paired
tables are tested comparisons.

**A reporting defect found while correcting the statistics.** As first written,
`scripts/code-retrieval/bootstrap_ws6a.py` passed one hardcoded label tuple to `resolve_branch`
for every row, so the accuracy cell read `A2` and **A3 — the branch that weakens
our own claim — appeared nowhere in the results at all**, while secondary
diagnostics carried A-labels for a pre-registration that does not cover them.
`scripts/code-retrieval/summarise_ws6c.py` and `summarise_ws6c_topk.py` carried the identical
defect: `branches.csv`, the file cited for the P-verdict, had **P1** in its
`turns` row. All four now select labels per (comparison, metric); every
non-pre-registered cell emits `n/a (not pre-registered)`, and the slot for
accuracy *separating* — which the specification never named, because it never
expected separation at n = 40 — emits the deliberately non-pre-registered
`A3-REFUTED` rather than borrowing a letter. **Across all four regenerated CSVs
only the `branch` column changed; every statistic is byte-identical**, verified
column-wise and reproducible by re-running the scripts.

---

## What was measured

### A. Code search — `results/ws6a/`

**Instrument.** `fastapi/fastapi` @ `a1fa70d4237d50aae6586a0d9b229df583463d21`;
agent `claude-sonnet-5` in all four arms; judge `claude-opus-5`, deliberately ≠
the agent; embedding OpenAI `text-embedding-3-small` (claude-context's own
default); Zilliz Cloud serverless `eu-central-1`;
`@zilliz/claude-context-mcp@0.1.15` via `npx` over stdio MCP; 40 questions
(15 locate / 15 trace / 10 multi_hop); turn cap 15 on **both** tool-using arms;
pricing $2/$10 and $5/$25 per MTok re-verified live 2026-08-18; run 2026-08-18,
spend $25.14 of a $75 governor. All four arms share one **byte-identical system
prompt** naming no tool and no retrieval strategy, so they differ in exactly one
variable: the `tools` list. Question order is shuffled per arm from `SEED=42` and
the 160 pairs run round-robin, so time-of-day API variance cannot bias one arm —
`started_at` is on every row, so that is falsifiable rather than asserted.
Adaptive thinking is on in every arm: controlled across arms, but it inflates all
four arms' `output_tokens` relative to a thinking-disabled setup.

**Corpus** (`repo_stats.csv`; `count_tokens` against the pinned agent model is
authoritative, tiktoken recorded only for comparability with the vendor's eval):
whole repo 2,888 files / 381,873 lines / **6,232,509** tokens (tiktoken
4,597,203; −26.2%); `fastapi/`+`docs_src/`+`tests/` 1,112 files / 109,319 lines /
1,138,485 (tiktoken 700,334; −38.5%); `fastapi/` alone 55 files / 21,957 lines /
259,429 (tiktoken 157,818; −39.2%). **tiktoken undercounts Claude by 26% here and
39% on the code-only subsets**; any token figure reaching the cost model is the
`count_tokens` one. Two corrections were made *before* anything was built on
these numbers: `messages.count_tokens` prices a whole request, so a constant rode
along on ~2,900 per-file calls while the batched per-scope call carried it once —
measured at **5 tokens/request** and subtracted, after which sum-of-per-file and
one batched call agree to **exactly 0 tokens**, where before there were two
disagreeing repo totals; and the stuffed arm really sends a `===== path =====`
header per file plus blank-line separators, so selection and reporting now run
over those units and the published `fit_fraction` describes the string actually
in the context window.

**`prompt_tokens = input + cache_read + cache_creation`.** The API reports
`input_tokens` as the *uncached remainder only*, so the stuffed arm — sending
~960k tokens per question — has a median `input_tokens` of **28**. Charting that
column would rank whole-repository stuffing as by far the cheapest arm. A
regression test pins this.

**Mechanism, measured from the transcripts rather than inferred:**

| arm | tool calls | median payload | chars returned per question |
|---|---|---|---|
| indexed | 110 × `search_code` | 12,014 chars (max 25,012) | **33,703** |
| agentic | 176 (`grep` 75, `read` 87, `glob` 14) | grep 237 · read 3,130 · glob 166 | **12,538** |

The indexed arm makes **fewer** calls but each returns ~2.7× more text.
`search_code` returns a fixed batch of ranked chunks with surrounding context
whether the question needs them or not; `grep` returns a 237-character line and
the agent reads only the file it actually wants. On "where is X" questions
cheap-probe-then-targeted-read is simply more token-efficient — and `fastapi`'s
library core is 48 `.py` files, so this does not test the case where semantic
search should earn its tokens back.

**Contamination, corroborated from a second direction.** Conditioned on prefix
membership in the stuffed arm: `all` (n = 36) judge accuracy 0.972, `any_file_hit`
0.944; `none` (**n = 4**) judge accuracy 0.250, `any_file_hit` 1.000; `partial`
empty (n = 0). The out-of-prefix cell is **n = 4**, reported with its n as a
corroborating signal, not a measurement — the parametric arm across all 40
questions remains the primary floor. One of those four (`q30`,
`scripts/prepare_release.py`) was answered correctly from training memory alone
with the file absent from context.

**The programmatic scorer and the judge disagree — and the scorer is wrong.**
11 of 160 rows disagree (6.9%), concentrated in the parametric arm (15 of its 40
rows have `any_file_hit` ≠ `judge_correct` in one direction or the other). The
instructive case: **on all four out-of-prefix stuffed questions, `any_file_hit`
is True while `judge_correct` is False.** The model wrote sentences like *"I
cannot determine this; there is no `scripts/docs.py` in the provided content"* —
naming the expected file inside an explicit denial, which path matching credits
because the path is present in the text. This is a finding about the metric, not
noise: **`any_file_hit` is not a usable correctness measure on a contaminated
corpus**, because a model that knows the file names from training will emit them
whether or not it can answer. Both scores are published; where they disagree,
**the judge is the one to trust.**

**Setup effort (C1)**, `setup_effort.csv`, recorded contemporaneously including
failures: parametric 0 steps; stuffed 4; agentic 4 steps, **0** external accounts,
**0** credentials, 1 failed attempt; indexed 8 steps, **2** accounts, **3**
credentials, **4** failed attempts. The "low effort" claim for the live side is
substantially true — three local tool functions and one system binary — but "zero
setup" is not: `rg` answered at an interactive shell prompt while
`shutil.which("rg")` returned `None`, because what answered was a *shell
function*, not a binary; the arm was unrunnable for an entire implementation
session and the verification step meant to catch it (`rg --version`) was satisfied
by the same shim.

The sharper finding is the *character* of the indexed arm's failures: **three of
the four were silent.**

1. The collection-name environment variable in our own plan
   (`COLLECTION_NAME_PREFIX`) is **ignored** by claude-context 0.1.15 — the real
   one is `CODE_CHUNKS_COLLECTION_NAME_OVERRIDE`. Collections land under an
   auto-generated path hash, and a multi-scope sweep cannot identify its own
   indexes.
2. `get_indexing_status` reported **"✅ fully indexed — 1100 files, 1100 chunks"
   19 seconds in** while the server's own log showed **330/2,825 files**. It treats
   the mere existence of a cloud collection as proof of completion. Trusting it
   exited our script, which closed stdio, which killed the server mid-index —
   leaving a collection holding **21% of the repository that answers queries
   perfectly happily**.
3. `get_collection_stats` returns `row_count: 0` for a freshly indexed collection,
   because stats come from sealed segments and lag unflushed inserts, so the naive
   chunk count is zero.
4. Background sync and a filesystem trigger watcher are both **on by default** and
   would have re-embedded the corpus mid-benchmark, spending outside the governor.

Only (4) is documented. The rest were found by cross-checking against a second
source. **A user who did not cross-check would have benchmarked a 21%-indexed
corpus and never known**, because every surface reported success. One more, found
only because a smoke test existed: `search_code` requires an absolute `path`
argument, and a system prompt that cannot name a tool cannot supply it — the model
called **no tool at all**, answered from memory, and produced a plausible 1-turn
indexed row scoring 0/2. Binding the path took it to 2/2 with 3 tool calls.
**Nothing errored in either case.**

**Index cost** (`index_cost.csv`): whole repo 249.3s, 2,825 files, **13,900**
chunks, **$0.0919**; mid scope 64.3s / 1,103 files / 8,170 chunks / $0.0140; small
scope 19.0s / 48 files / 661 chunks / $0.0032. Server-reported chunk counts agree
exactly with an independent Milvus `count(*)` at all three scopes.
`embed_cost_usd` and `mean_chunk_tokens` both carry caveats — see "Limits".

**Caching, and what it does not rescue.** The stuffed arm writes its 960,124-token
prefix once at 2× rate with a 1h TTL and reads it 39 times at 0.1×: 38,404,994
total prompt tokens, 37,443,744 of them cache reads, billed $11.76 against $77.24
at list price — an **84.8% saving**. Reporting only the uncached number would
strawman live search by 6.6×. **And it still does not rescue whole-context
stuffing:** even fully cached the stuffed arm costs **5.4× agentic and 4.4×
indexed per question** while being the *least* accurate retrieval arm (0.900).
Also verified live 2026-08-18: the full 1M-token context window bills at standard
per-token rates with **no long-context premium** and needs no beta header — there
is no context-size tier term in the cost model.

### B. Agent memory — `results/ws6b/`

**Instrument.** memsearch `0.4.0` @ `3149dc3` (`zilliztech/memsearch`), shipped
CLI via `uv run`, against **local** Milvus standalone at
`http://localhost:19530`; embedding OpenAI `text-embedding-3-small` (memsearch's
shipped default); memsearch at defaults only — chunk 1500 chars / 2 overlap lines,
dense + BM25 + RRF, cross-encoder reranker off; agent `claude-sonnet-5`, judge
`claude-opus-5`; turn cap 15 on the memsearch arm only; byte-identical system
prompt again; run 2026-08-18. **Stated so nothing is charted across directories:**
`results/ws6a/` ran against Zilliz Cloud and this against local Milvus — retrieval
quality is unaffected (same index types, same algorithms) but latency and
infrastructure cost are **not comparable**.

**The corpus is synthetic, and that is the point.**
`scripts/code-retrieval/generate_ws6b_corpus.py`, one `random.Random(42)`, committed; the corpus
is committed at `data/ws6b_corpus/` under an explicit granted exception to the
no-corpus-data rule, whose stated rationale is licensing and cannot apply to text
our own generator produced; `tests/code_retrieval/test_memory.py` asserts regeneration reproduces
the committed bytes. Sizes: 11 files / **98,134** tokens (target 100,000; 246,653
chars), 48 / **392,116** (target 400,000; 989,397 chars), 145 / **1,221,906**
(target 1,200,000; 3,083,351 chars) — span **12.45×**. The x-axis everywhere is
the measured token count, never a file or session count and never the target. The
largest size **deliberately exceeds the 1M context window**, so replay cannot run
there; `fit_fraction` for truncated replay is **0.7851** (first retained day file:
index 31). Two authorship problems the code-search benchmark has are absent by
construction: facts are planted by the generator and probes derived mechanically
from `(subject, attribute, value)` triples, so **nobody chose which questions to
ask**, and the corpus is uncontaminated by construction. Fidelity check: the
generated corpus measures **2.5234 chars/token** against the real memsearch memory
in this repository at **2.3769** — very slightly *less* token-dense than the real
thing; anchor share 14.0% against a real 18.6%, mean bullet length 173 chars
against a real 189.

**Probe sets**, committed to `probes.csv` before any arm ran: `scaling` n = 12,
inside the smallest prefix at relative depth 0.1/0.5/0.9, run at all three sizes —
**the same 12 questions**, so size is the only variable; `depth` n = 18, at
large-corpus-relative depth 0.1/0.5/0.9, 6 per band, largest size only;
`superseded` n = 8, originals at relative 0.40–0.55 with revisions in the last
third. **Both instruments hold in the published data, not merely in a test:** the
`depth` set straddles the truncation boundary (its 0.1 band is
`fact_in_window = False`, its 0.5 and 0.9 bands `True`), so the window cliff is
measured across three bands rather than assumed; and all 8 `superseded` originals
are `original_in_window = True`, which the specification required in advance so a
zero stale rate could not be invisibility scored as recency reasoning.

**The non-memorization gate passed at 0.0000.** Thresholds pre-registered at
pass ≤ 0.05, fail > 0.10. Attempt 1, generator SHA `17463a16a4da`, n = 38, **0
correct, accuracy 0.0000, PASS** — `parametric_gate.csv` carries `attempt` and
`generator_git_sha` on every row, so this is auditable rather than asserted. A
model with no history access answered **none** of these questions; every answer
was an explicit refusal — *"I don't have any information about a project involving
`girthbudget`"*. Beside `fastapi`'s 0.625 this is the cleanest pair of
contamination numbers in the project: every point of accuracy measured here is
retrieval rather than recall of training data.

**Incremental sync** (`skip_rate.csv`), each variant applied to the largest
corpus, measured, then rolled back, with the collection rebuilt between variants
so they cannot contaminate one another: `new_file` 2,907 chunks / 1 re-embedded /
skip **0.9997** / 1.8s / $0.00000; `tail_append` 2,907 / 1 / **0.9997** / 2.0s /
$0.00000; `mid_edit` 2,906 / 5 / **0.9983** / 2.3s / $0.00002. Chunk identity is
`f(source, start_line, end_line, sha256(content), model)`, so line numbers are
*inside* the identity: a mid-file edit re-embeds **5×** what an append does,
because every chunk after the edit gets a new id even though its content did not
change. Blast radius is one day file, not the corpus, because identity is also
keyed on source path. For C10's "smart dedup": true for memsearch's actual usage
pattern — append-only daily logs, ~99.97% skip — but for an edit-heavy corpus the
cost scales with how much of the file follows the edit. **Re-embedding cost
depends on where a change lands, not only how much changed.** Recorded, no run
needed: the embedding model name is part of the chunk id, so **changing embedding
models is a full re-index at a 0% skip rate**, not an incremental migration.

**Index cost** (`index_cost.csv`): 236 / 936 / 2,906 chunks over 11 / 48 / 145
files, 13.3s / 44.3s / 126.5s, 49,984 / 201,608 / 630,055 cl100k tokens,
$0.0010 / $0.0040 / $0.0126 — total **$0.0176**. memsearch's reported chunk count
agrees exactly with an independent Milvus `count(*)` at all three sizes and the
build script aborts on mismatch. **The zero-cost alternative, stated next to the
number:** memsearch's local ONNX provider (`gpahal/bge-m3-onnx-int8`) has **zero
marginal API cost** — a laptop user pays compute, not dollars — which materially
strengthens the indexed side of C9 in a way the figure above does not capture. The
`openai` provider was pinned because it is the shipped default and because it
makes this number directly comparable to the $0.0919 above.

**Cost against replay** (appendix table B), all billed **with caching live on the
replay side**, which is the steelman: the cache saved **77.3%** against list price
at the two smaller sizes and **84.9%** at the largest, and reporting uncached
numbers would have inflated replay's cost by 4.4× and strawmanned it. The
pre-registered branch is **B1** and the data landed on it, not marginally — but
**no crossover exists inside the tested range**: memsearch is cheaper at every
size, so the crossover lies *below* 98,134 tokens and **this evidence does not
bracket it.** The bracketing rule is satisfied vacuously; nothing is interpolated.

**What that does not settle — read it next to the number.** Per-query saving at
the largest size is $0.1709 against a one-time index build of $0.0126, so the
build repays itself in **0.07 queries**. That number is real and it is **not the
break-even**: it amortises **embedding cost only** and excludes the vector store's
running cost entirely, because this ran against a local Milvus that costs nothing
per month. A hosted index has a monthly bill that appears nowhere in this
directory, and at low query volumes that bill, not the embedding, is what
break-even is actually about. The honest statement is "the index repays its
*build* almost immediately," which is much weaker than "the index is worth
running." **`results/ws5/` owns the real break-even** (`results/cost-model.md`);
this directory contributes measured per-query token and cost deltas and nothing
more. **Putting "break-even after 0.07 queries" on a slide would be the most
attackable number in the talk.**

**The C12 curve: branch D2, partially supported.** On the `scaling` set — the same
12 questions at all three sizes — memsearch's median prompt tokens ran
4,554 → 4,636 → 10,593, **2.33×** over a 12.45× corpus, against replay's
98,244 → 392,189 → 959,269 † (9.8×). **2.33× fails the pre-registered D1 bar of
< 2×**, so the verdict is **D2**, not D1. It would have been easy to call 2.33
"roughly flat"; the pre-registered threshold exists precisely to stop that.
Mechanism, from the transcripts: median turns rose 2.0 → 2.0 → 3.0 — the model
issues *more searches* on a larger history and each returns a similar payload, so
growth is in the number of retrievals, not the size of each one. Replay's ~linear
growth is arithmetic, not a discovery, and is not reported as a finding. Judge
accuracy is **1.000 for both arms at S and M, and 1.000 for memsearch at every
size; truncated replay scores 0.000 on the `scaling` set at L**, which is
arithmetic — those facts sit outside its window — not a result. So on this probe
set the C12 story is **cost-only** wherever both arms can actually see the facts,
said plainly.

† At L the replay arm is *truncated* replay (`replay_trunc@l`): plain replay
exceeds the window and cannot run at all. See the window cliff below and
Appendix B.

**The window cliff.** Plain replay could not run at the largest size. One attempt,
recorded verbatim:

```
BadRequestError: 400 invalid_request_error
prompt is too long: 1221932 tokens > 1000000 maximum
```

`runs.csv` carries `status = window_exceeded` with **null** cost and **null**
tokens — a hard capability boundary, never an infinite or extrapolated cost.
(Incidental corroboration: the API's own count, 1,221,932, matches our measured
corpus, 1,221,906, to 26 tokens of request framing — the corpus measurement is
confirmed by the service that rejected it.) Truncated replay retains the most
recent **78.51%**:

| depth band | inside the truncated window | `replay_trunc` | `memsearch` |
|---|---|---|---|
| 0.1 | **no** | **0.000** (0/6) | 1.000 (6/6) |
| 0.5 | yes | 0.833 (5/6) | 1.000 (6/6) |
| 0.9 | yes | 0.833 (5/6) | 1.000 (6/6) |

**The 0/6 at band 0.1 is arithmetic and is not presented as a finding** — those
facts are not in the arm's context, and the specification conceded this in advance
precisely so it could not be sold as a discovery. **The finding is the in-window
rows:** on facts it *can* see, truncated replay scores **10/12 (0.833)** while
memsearch scores **12/12**. That is lost-in-the-middle, measured, on a ~959k-token
context, and it is the only place in this directory where retrieval quality
separates the arms. `cost_per_correct_usd` at that size, defined before any data
existed so it cannot be a post-hoc rescue: **memsearch $0.0188 vs truncated replay
$0.6126 — 32.6×.**

**C10, one line per architectural claim.** "Indexes plain markdown instead of
replaying full transcripts" — **confirmed** (4,554 → 10,593 against replay's
98,244 → 959,269). "SHA-256 dedup" — **confirmed, with the qualification above**.
Dense + BM25 + RRF hybrid — **confirmed by code inspection** (`store.py`), cited
rather than benchmarked. "3-layer progressive recall" — **accurate as software,
UNEXERCISED in this workload, and the cause is partly our corpus.** Across all 62
memsearch runs, `memory_expand` (L2) and `memory_read_session` (L3) were called
**zero** times; every answer came from L1 `memory_search` alone, sometimes repeated
(26× one call, 10× two, 7× three). Two reasons, the second ours: L1 already returns
full chunk content (median payload **5,926 chars**, max 7,761 — roughly **half**
the 12,014 chars/call measured for `search_code`), so there is usually nothing for
L2 to expand *to*; and **every planted fact lives in one bullet, inside one turn,
which is one chunk**, so the corpus makes single-chunk retrieval sufficient by
construction. A benchmark testing progressive disclosure would need answers
spanning chunks. **This one does not, and its verdict on the third layer is
therefore "untested", not "works".**

**The superseded subset is a null result, reported as one.** Both arms: judge
accuracy **1.000**, `stale_rate` **0.000**, n = 8. **Neither arm committed a
recency error, so the subset found nothing.** It was included to test a real
agent-memory failure mode; on this corpus that mode did not occur, and inventing
significance for it would be dishonest. Two things it *did* establish. The
visibility guarantee held — all 8 originals are `original_in_window = True`,
verifiable in `probes.csv`, so both arms saw both statements and a zero stale rate
means reasoning, not invisibility. And **it caught a measurement bug in our own
scorer**, the more useful outcome: `stale_mentioned_rate` is **1.000** — every
answer contained the superseded value, because the model narrates the change
("changed from 3319 to 8123"). A substring-based `stale_rate` would have published
a **100% recency-failure rate for an arm that got all 8 right.** `stale_rate` is
now judge-gated — stale value present **and** judged incorrect — with the raw
substring rate published beside it so the gap is auditable. This is the
`any_file_hit` lesson in a new place: a string appearing in a semantic role the
matcher cannot see. **It is the second run in a row where the mechanical scorer
failed and the judge did not.**

**Setup effort** (`setup_effort.csv`): parametric 0/0/0/0/0; replay 1 step, 0
accounts, 1 credential, 0 failures, 0 silent; memsearch 6 steps, 1 account, 2
credentials, 2 failed attempts, **1 silent**; against claude-context's
8 / 2 / 3 / 4 / **3**. **The significant one is the silent failure, and it
reproduces the sharpest C1 finding in a second, independently built tool.** After
`Indexed 52 chunks.` with `rc=0`, **every search returned `No results found.` with
`rc=0` and no error** — on a collection Milvus held 52 rows for. memsearch's
`store.search()` guards BM25 against empty collections using
`get_collection_stats()['row_count']`, and Milvus derives that count from **sealed
segments only**, so freshly inserted, unflushed data reads as zero; an explicit
flush makes the identical query return results immediately. That is exactly the
shape recorded for claude-context above. **Two retrieval tools both published by
Zilliz but independently built, both treating Milvus
segment statistics as a liveness signal — the same wrong assumption about the
same database, made twice.** A user who indexed and searched in one sitting would conclude the tool is
broken — or worse, that their history contains nothing.

Spend: parametric gate $0.2514, index embedding $0.0176, main run $15.2448 —
**$15.51 of the $75 governor**, against a $25.9 estimate; agent-only billed spend
$14.70. The governor was seeded cumulatively across phases from the checkpoint,
never per-script, and was never breached; the run was killed once mid-flight and
resumed with **zero re-spend** on the 91 completed runs.

### C. The top-k knob and an uncontaminated repository — `results/ws6c/`

This re-opens two stated limitations of the code-search benchmark — the retrieval
knob was never turned, and one repository was tested, the one most favourable to
grep. **It does not re-run it.** Pre-registration was written before any of its
data existed; every branch is quoted **by name**, so each verdict is a lookup
rather than a judgement made after seeing the data. Run 2026-09-08; the size
sweep in the last subsection (WS9) ran 2026-09-15 against its own
pre-registration.

**Instrument.** Phase 0 input is `results/ws6a/runs.csv` and `scaling.csv`,
**frozen, read-only** — the correction above. Phase 1 reuses `fastapi` at the same
pinned commit and the same Zilliz collection, **no re-index**, since top-k is a
query-time parameter, so embed spend is $0 and the corpus is *provably* identical.
Phase 3 uses `agentic-hil/agentic-hil` @
`837f39813e1421c7b8efc6e621c1368507f2dbb5` (Apache-2.0) with 40 new questions
(15 locate / 15 trace / 10 multi_hop). Same agent, judge, embedding, package, turn
cap. Statistics: paired median delta, 10,000-resample percentile bootstrap
(`WS6C_BOOT_SEED = 42`, α = 0.05) and an exact two-sided sign test via `math.comb`
— `scipy` is a transitive pin and is never imported. The constrained arm **binds
rather than prompts**: `search_code`'s `limit` is bound to 3 in the wrapper and
removed from the schema the model sees, exactly as `path` already is, because the
byte-identical system prompt cannot name a tool or a parameter and a prompt-side
instruction would have broken the one-variable design. The bind is verified from
the committed transcripts.

**The shipped default is a measured value, not an assumption.**
`topk_provenance.txt` records the scan of the pinned package's own `dist/`:
`node_modules/@zilliz/claude-context-mcp/dist/handlers.js:limit = 10`. The scan
returns three hits and the other two are disqualified **on inspection, not by
preference** — `@zilliz/milvus2-sdk-node/dist/milvus/grpc/Data.js` (`limit: 10`)
is a transitive SDK dependency whose match is coincidental, in code `search_code`
does not route through, and `langsmith/dist/client.js` (`limit = 100`) is
unrelated tracing middleware. So `WS6C_TOPK = 3` is a **3.3× reduction from a
measured value**.

**Phase 1 — the knob on `fastapi`** (`topk_summary.csv`, `topk_runs.csv`, 40 rows,
0 errors; the frozen agentic rows are the comparison arm):

| comparison | median | 95% CI | positive | sign p | branch |
|---|---|---|---|---|---|
| `indexed_topk3` − `agentic` | **+584.5** | **[−2,572.0, +2,405.0]** | 21/40 | 0.875 | **T3** |
| `indexed_topk3` − `indexed` | **−4,429.5** | **[−8,385.0, −1,890.0]** | 9/40 | 0.00068 | CI excludes zero (`plan:K_LESS`) |

Marginal medians, not paired: agentic 9,485 · `indexed_topk3` **11,667** · indexed
16,452. **The knob demonstrably works** — the fixed-batch mechanism responding
exactly as predicted. **And it does not win**: branch **T3** means *neither* T1 nor
T2 may be claimed, so we may not say the constrained index still loses and we may
not say it wins. **It costs accuracy**: 1.000 → **0.925**, three questions flipping
from correct to incorrect (`q10`, `q27`, `q32`, verified row-by-row against
`results/ws6a/runs.csv`), with turns rising from a median 2.0 to 3.0 as the model
compensates for the smaller payload with more searches.

**The `K_MORE` / `K_LESS` / `K_FLAT` labels are not specification branches, and the
CSVs say so.** Unlike the A-, T-, S- and P-series they appear nowhere in the
pre-registration; they came from the implementation plan's summariser and were
committed at `5323b38`, **before any of this data existed**, so they are
fixed-in-advance in the weaker sense of the word. The `plan:` prefix carries that
distinction in `branches.csv` and `topk_summary.csv` themselves. The comparison is
real and correctly computed; only its authority is weaker.

**Phase 2 — gate before spend** (`nonmemorization_gate.csv`, `corpus_stats.csv`).
All three candidates were gated before any was chosen, and all three scores are
published whether or not the repository was used: `agentic-hil/agentic-hil`,
`oaslananka/kicad-mcp-pro` and `gmboquet/mixle` each scored **0.000** judge
accuracy and **0.000** `any_file_hit` at n = 25 — branch **S1** in all three cases.
**A genuinely uncontaminated corpus of this size exists, and there are three of
them**, which retires the pre-registered branch saying the effort would stop at
Phase 1 for want of one. **Why 0.000 is real and not an artefact of bad
questions:** the failure mode that would fake a clean score is questions so obscure
the model guesses wrong, and that is not what happened. `any_file_hit` is **0/75** —
a guessing model would still emit plausible file paths, and on `fastapi` the
identical pipeline hits `any_file_hit` **0.675**; zero of seventy-five is not wrong
guessing, it is not answering. The committed transcripts
(`transcripts_gate_*/parametric/`, 75 files) show the model **declining
outright** — *"I don't have access to the codebase in this session"* — and are
committed so a reader can check rather than take it on trust. And the questions are
concrete and resolve against the pinned checkouts: `OpenOCDBackend` in
`src/agentic_hil/backends/openocd.py`, `classify_drc_report` in
`src/kicad_mcp/validation/drc_runner.py`, `AnnealedEM`/`SquaremEM` in
`mixle/inference/em.py`.

**The declared selection effect is retired, and its honest counterpart stated.**
With all three tied at exactly 0.000 there is no selection *on the scores* —
nothing distinguishes them — and the choice falls to a candidate ordering fixed in
the specification before any measurement existed. But **the gate proved all three
clean and did not rank them**: "why `agentic_hil`?" is answered by a
stars-and-creation-date prior, not by a measurement. That is a weaker justification
than a measured one, and it is stated rather than dressed up.

Corpus measurement (`corpus_stats.csv`): `agentic_hil` 342 files / 215,187 lines /
**4,307,363** `count_tokens` (tiktoken 2,770,789 — an undercount of **35.7%**,
consistent with the 26–39% range above); `kicad_mcp_pro` 1,466 files / 276,362
lines / 4,247,639; `mixle` 2,396 files / 784,644 lines / 14,967,861. Against
`fastapi`'s whole-repo scope the selected corpus has ratio **0.691**. None of the
three fits a 960k prefix.

**A vendor bug: zero files indexed, reported as success.** Unplanned finding.
Pointed at the pinned `agentic_hil` checkout, `@zilliz/claude-context-mcp@0.1.15`
indexed **zero files and reported success** — *"✅ Codebase … completed
successfully! · 📊 Statistics: **0 files, 0 chunks** · 📅 Status: completed"*.
Mechanism, from the package's own source: `Context.findIgnoreFiles()` merges
**every root-level `.*ignore` file** into its ignore set, and this repository ships
the standard minimal-Docker-context idiom — a `**` catch-all followed by negations
— in `.dockerignore`. `IgnoreMatcher` tests the **non-slash form first**, so every
top-level directory is pruned before its negation is ever consulted. **No operator
setting fixes it**: `CUSTOM_IGNORE_PATTERNS`, the `index_codebase` argument and
`.contextignore` are all **additive**, so a user cannot un-ignore what a merged
`.dockerignore` removed. **Any repository carrying that idiom is silently
unindexable by the pinned package.** This is the same pattern as the 330/2,825
"fully indexed" report and memsearch's `No results found.` — a surface reporting
success while the underlying operation did nothing — and it is the first instance
where the operation did *literally nothing* while reporting completion. **The
mandatory three-way completeness assert caught it on the first run; without it,
Phase 3 would have benchmarked an empty index and reported the resulting numbers
as a retrieval result.** The workaround is disclosed with its provenance: both
tool-using arms run against `data/ws6c_tree_agentic_hil`, a sha256-verified copy of
the pinned checkout with that single root-level `.dockerignore` dropped. That file
is outside the text extension allowlist, so it is none of the 342 files or
4,307,363 `count_tokens` that `corpus_stats.csv` pins — the measured corpus is
byte-identical; it is dot-prefixed and would never have been indexed in any case;
no question references it; **every** arm points at the same derived tree, so there
is no cross-arm confound; and `tree_sha256 = 72362f5c…6beaac7` is recorded
identically in `index_cost.csv` and `run_provenance.txt`.

**The index** (`index_cost.csv`, one row): 174.3s wall clock **at
`embedding_batch_size = 20`, not the package default 100** — a throttle forced to
stay under a 1M tokens/minute embedding rate limit, so this is an **upper bound**
on indexing time and is **not comparable** to a default-configuration run. **279**
files indexed of 342 text-allowlisted (382 git-tracked), **7,441** chunks, embed
cost **$0.050975 — DERIVED**, and **not** under the Anthropic governor.
Completeness was asserted three ways before the run — server stderr (279 files /
7,441 chunks), an independent Milvus `count(*)` after flush (7,441), and a
predictor calibrated to reproduce the earlier 2,825 exactly. All three agree.

**Phase 3 — the paired result** (`branches.csv`, `runs.csv`, 160 rows, 4 arms × 40
questions, **0 errors**):

| comparison | median | 95% CI | direction | sign p | branch |
|---|---|---|---|---|---|
| `indexed` − `agentic` | **−5,217.0** | **[−16,019.5, +85.5]** | 26/40 negative | 0.081 | **P3** |
| `indexed_topk3` − `agentic` | **−4,717.0** | **[−13,066.0, +1,280.5]** | 24/40 negative | 0.268 | **P3** |
| `indexed` − `agentic` (turns) | −1.0 | [−2.0, −1.0] | 38/40 ≤ 0 | 8.7e-07 | *not pre-registered* |

The turns delta is **exploratory** and `branches.csv` says so: the P-series is
defined for **tokens only**, so a `turns` row may not carry a P-letter, and **no
P-letter anywhere refers to anything but a `prompt_tokens` delta.** The same
applies to the two secondary deltas over the frozen `fastapi` rows — turns −1.0,
CI [−2.0, −1.0], which separates cleanly (3 of 40 questions took more turns), and
`wall_clock_s` +1.496, CI [−0.551, +2.323], which does not. Both are supporting
mechanism, never tested hypotheses; the turns result in particular is the kind of
clean separation it would be easy to present as a finding, and nothing said in
advance that we would look at it.

**What the flip does and does not say.** On `fastapi` the indexed arm cost
**more** (paired median +2,857, over `fastapi`'s 40 questions); on a corpus the
model has never seen it costs **less** (paired median −5,217, over a *different*
40 questions). **The two estimates are each paired within their own corpus and are
unpaired with respect to each other**, so this is a described sign reversal, not a
computed cross-corpus interval: **no interval is computed across them and none
could be paired.** The reversal is real and reportable; it is not a significance
claim. For the first time **on a real repository** in this project the retrieval
arms are not sitting on a contamination floor larger than the spread between them —
the parametric floor here is 0.000, not 0.625.

**And the interval will not commit.** `[−16,019.5, +85.5]` misses excluding zero
**by 85 tokens** on a scale of 16,000. Branch **P3** — *report the point estimates
with their intervals and claim neither direction.* **How fragile the near-miss is
was measured, not asserted:**

| perturbation | result |
|---|---|
| Bootstrap seed ∈ {1, 7, 42, 999, 12345} × `n_boot` ∈ {10,000, 50,000} | upper bound is **+85.5 in all ten runs** — the bootstrap median distribution is *discrete* at n = 40, so resampling noise cannot move it |
| Drop the two turn-capped agentic rows (`agentic_hil_q21`, `agentic_hil_q40`) | n = 38, median **−3,895.5**, CI **[−13,856.0, +736.0]** — still includes zero |
| Single-question jackknife (drop one of 40, refit) | **14 of 40** drops flip it to excluding zero |
| α = 0.10 instead of 0.05 | **[−13,856.0, −692.0]** — excludes zero |

These four are sensitivity checks over `runs.csv` via `benchlib.code_retrieval.topk`, **not
committed results**: the committed branch resolution is the α = 0.05, seed 42,
`n_boot` = 10,000 row in `branches.csv`, and nothing here displaces it. **So the
near-miss is robust to the things that are noise and fragile to the things that are
choices.** It does not depend on the two capped rows — the objection a reader
reaches for first, which does not land — and it survives every seed and resample
count thrown at it. But one question in three would move it, and a looser α moves
it, and **those last two are analyst choices; the branch was fixed in writing
before the data existed precisely so they were not ours to make on the day.** **We
do not upgrade the near-miss. The talk may not say the index wins on an unfamiliar
corpus; it may say the point estimate points that way and the study was not powered
to confirm it.**

**The confound that cuts *against* the flip.** The two corpora are scope-matched
**by measured tokens** — 4,307,363 against 6,232,509, ratio 0.691 — but badly
mismatched **by file count**: **342 files against 2,888, an 8.4× gap**. Nothing in
the specification anticipated this, and it is a genuine confound: grep's cost
scales with how much tree it has to search, so a 342-file haystack **favours the
agentic arm**, and token-based scope matching does not neutralise it. It cuts
against the phase's own hypothesis, which makes the flip more notable rather than
less — the arm the asymmetry flatters is the one that lost ground. Stated here as
it would have been stated had the branch landed the other way.

**The arms really did run at the defaults claimed.** On `fastapi` the model
supplied `limit` itself on **1 of 110** `search_code` calls (value 5). On the
unseen corpus the default arm made **119** calls and supplied `limit` on **1**; the
constrained arm made **221** calls and supplied `limit` on **0**, which is what
removing the parameter from the schema is supposed to produce and is an independent
confirmation that the bind held for the whole run. One call in 119 is not a tuned
configuration.

**The top-k saving on the unseen corpus — read carefully.** Both rows are
`indexed_topk3` − `indexed`, same metric, same statistic, different corpus:
`fastapi` **−4,429.5**, CI [−8,385.0, −1,890.0] (`plan:K_LESS`); `agentic_hil`
**−3,961.5**, CI [−7,942.0, +5,502.0] (`plan:K_FLAT`). **"The top-k saving does not
replicate" is a true label lookup and a misleading sentence.** The point estimates
are nearly identical — a difference of **468 tokens**. What changed is the
**interval width**: 6,495 tokens against 13,444. The honest statement is
**"directionally consistent, but the unfamiliar corpus is far noisier"**, not "the
effect vanished."

**Phase 3b never became available — a limit, not an omission.** Arm (c), true
whole-repository stuffing, was gated on the selected corpus measuring under
**960,000 `count_tokens`**. All three candidates measured **4.4×, 4.4× and 15.6×
over that budget** (4,307,363 / 4,247,639 / 14,967,861). **The gate is a
measurement, not a choice, and it closed.** Arm (c) is therefore **absent** from
Phase 3 — `prefix_text` is `None` and `in_stuffed_prefix` is blank on all 160 rows —
and two limitations the plan hoped to retire **stay open**: the stuffed arm remains
prefix stuffing at 15.2% of `fastapi`, and its tiering remains a judgement call.
Finding a post-cutoff, permissively licensed, real-structure repository that is
*also* under 960k Claude tokens is a materially harder search than the one this
specification ran, and it was not attempted.

**Branches that did not fire, named so the set is complete.** The pre-registered
tree has eleven outcomes; six fired — **A1** (tokens), **A3** (accuracy and the
dilution curve), **T3**, **S1** (all three candidates) and **P3** (both token
comparisons). The five that did not: **A2** (the token CI including zero — it
excluded zero), **T1** and **T2** (the constrained arm separating from agentic in
either direction), **P1** and **P2** (the index or agentic separating on the unseen
corpus), and **S2**/**S3** (a partially contaminated or rejected candidate — no
candidate scored above 0.000). A claim that every branch is quoted by name is only
meaningful if the unfired ones are listed too.

#### The size sweep on the unseen repository — WS9 (`scaling.csv`, `paired_ci_unseen_scaling.csv`, `matched_size_contamination.csv`)

Phase 3 measured one size. This puts the same three arms on three nested sizes
of the same repository, so that `results/cost-model.md` can draw a break-even
*curve* on a corpus with a 0.000 parametric floor rather than a single point.
Pre-registration:
`docs/superpowers/specs/2026-09-15-ws9-unseen-corpus-cost-curve-design.md`,
written before any S or M row existed; new branch series (F, B, K, U, C) —
**no WS6c letter is borrowed**, and cells outside the series read
`n/a (not pre-registered)` exactly as `branches.csv` does. Run 2026-09-15.
Spend **$26.44** of a $55.00 cap over 222 runs (basis $26.83), embeddings
$0.037.

**Instrument** (`scaling_provenance.txt`, `scaling_corpus_stats.csv`,
`index_cost.csv`). Corpus `agentic-hil/agentic-hil` @ `837f398`, the same derived
tree as Phase 3 (pinned checkout minus its root `.dockerignore`, see the vendor
bug above), cut into **nested directory tiers decided from the tree before any
row existed**: **S = `src/`**, **M = the whole repository minus `tests/`**,
**L = the whole repository**. `tests/` is 51.2% of this corpus's tiktoken (53.2%
of `count_tokens`) in one directory, so splitting it out is the only division
of this tree with even log spacing. Each subset is sha256-verified file by file
against the derived tree; both tool-using arms point at the same absolute path.

| scope | `count_tokens` | files (text-allowlisted / tracked) | files indexed | chunks | embed | index wall-clock |
|---|---|---|---|---|---|---|
| S | **1,139,248** | 49 / 49 | 46 | **2,099** | $0.014457 | 64.1 s |
| M | **2,017,476** | 172 / 212 | 112 | **3,324** | $0.022858 | 93.9 s |
| L | **4,307,363** | 342 / 382 | 279 | **7,441** | $0.050975 | 174.3 s |

The span is **3.78×** (1.77× then 2.14×); the spec pre-registered 3.74× from
tiktoken predictions. Each index was asserted complete three ways before any
run, as at L. `embedding_batch_size = 20` throughout, so the wall-clock figures
are upper bounds, as Phase 3's is.

**The panel: 37 of 40, decided once, from gold paths alone.** A question is
eligible iff every path in its `expected_files` lies inside S (`src/`). `q09`
(`examples/`), `q30` (`tools/`) and `q31` (`tests/`) do not qualify. The same 37
run at every size, so no question enters or leaves the panel as the corpus grows
— a curve drawn over different question populations has a shape that is partly a
composition artefact. **Consequence, stated up front: the L cells below are n =
37 and differ from the published n = 40 figures above.** The L rows are **reused,
not re-run**: they are Phase 3's frozen `runs.csv` filtered to the panel and to
these three arms (and not `topk_runs.csv`, which is Phase 1 on `fastapi`). So the
sweep spans a week of wall-clock as well as 3.78× of corpus — L measured
2026-09-08, S and M 2026-09-15 (`started_at`) — under the same pinned model,
package, prompt, judge and turn cap. Nothing pre-registered that gap; it is
reported, not explained away.

**Arms**: `agentic`, `indexed` (shipped default k=10), `indexed_topk3` (bound to
3, verified as at L). No stuffed arm at any size — 4.4× over the prefix budget at
L, and the spec ran S and M under the same rule. The two indexed arms are not
interleaved with each other (claude-context builds its tool list at session
entry, so the bind needs a second MCP session); each is interleaved against the
same question order under the same seed against the same index. Smoke check
(first 12 rows of the checkpoint, re-used not re-billed): both indexed arms
show turns > 1 and the bound arm's transcripts read `Found 3 results`
(`transcripts_ws9_s/`, `transcripts_ws9_m/`). 222 new runs, **0 errors**.

**Per size, per arm** (`scaling.csv`; $/query is the mean `cost_agent_billed`,
judge excluded — the quantity `results/cost-model.md` divides the floor by):

| size | arm | n | mean $/query | judge accuracy | hit turn cap |
|---|---|---|---|---|---|
| S | agentic | 37 | 0.1442 | 0.892 | 1 |
| S | indexed | 37 | **0.0677** | 0.973 | 0 |
| S | indexed_topk3 | 37 | 0.0926 | 0.973 | 0 |
| M | agentic | 37 | 0.1897 | 0.919 | 2 |
| M | indexed | 37 | 0.0799 | 0.973 | 0 |
| M | indexed_topk3 | 37 | **0.0705** | 0.973 | 0 |
| L | agentic | 37 | 0.1392 | 0.919 | 2 |
| L | indexed | 37 | **0.0949** | 0.973 | 0 |
| L | indexed_topk3 | 37 | 0.1050 | 0.946 | 0 |

The index is the cheaper arm per query at every size, on both settings, so
the spec's **H2** (grep might be cheaper on the 49-file tree) **did not fire** —
a finding about scale on an uncontaminated corpus, and nothing more: it says
nothing about whether scale explains `fastapi`, which is the C-series below.
The grep arm's cost is **not monotone** in corpus size (0.144 → 0.190 → 0.139)
while the index's rises steadily (0.068 → 0.080 → 0.095); with every primary
paired cell at U3, whether the bump at M is the corpus or the panel is not
known. The cheapest index setting also changes with size — k=10 at S and L, k=3
at M — which is branch **K2** in `results/cost-model.md`: the order the deck
quotes at L ("58.94 at k=10, 70.44 at k=3") does not hold across the curve.
Marginal means, not tested comparisons: the tests are next.

**U-series — nine unadjusted cells, U3 in eight** (`paired_ci_unseen_scaling.csv`,
`contrast_kind == "arm_at_size"`, `metric == "prompt_tokens"`; paired median
delta, 10,000-resample percentile bootstrap, `WS6C_BOOT_SEED = 42`, α = 0.05,
exact sign test — the frozen WS6c knobs, reused; no cell may be re-resolved at
another α, seed or resample count):

| size | contrast | n | median | 95% CI | sign p | branch |
|---|---|---|---|---|---|---|
| S | `indexed` − `agentic` | 37 | −1,743 | [−13,337, +3,075] | 0.511 | **U3** |
| S | `indexed_topk3` − `agentic` | 37 | −2,314 | [−8,198, +3,644] | 0.324 | **U3** |
| S | `indexed_topk3` − `indexed` | 37 | +1,671 | [−4,851, +6,725] | 1.000 | **U3** |
| M | `indexed` − `agentic` | 37 | −3,467 | [−21,743, +846] | 0.324 | **U3** |
| M | `indexed_topk3` − `agentic` | 37 | −8,263 | [−23,324, −2,576] | 0.0026 | U1 — *isolated; not a finding, see below* |
| M | `indexed_topk3` − `indexed` | 37 | −4,575 | [−10,476, +407] | 0.099 | **U3** |
| L | `indexed` − `agentic` | 37 | −3,228 | [−12,884, +961] | 0.188 | **U3** |
| L | `indexed_topk3` − `agentic` | 37 | −2,019 | [−12,676, +1,992] | 0.511 | **U3** |
| L | `indexed_topk3` − `indexed` | 37 | −2,145 | [−7,675, +10,341] | 1.000 | **U3** |

**Multiplicity disclosure, carried from the spec and printed by
`scripts/code-retrieval/bootstrap_ws9.py` where it cannot be missed:** these are
3 contrasts × 3 sizes resolved against their own 95% intervals with **no
correction**; under a global null ~0.45 of the nine resolve by chance. The
correction is deliberately not applied — choosing one after seeing how many cells
resolved would be a worse analyst degree of freedom than declaring the unadjusted
rate in advance. **A single isolated U1 among nine cells is therefore not
evidence of anything and is not presented as a finding.** The M `indexed_topk3`
− `agentic` cell is exactly that: flanked by U3 for the same contrast at S and
L, it is not a pattern across sizes in one contrast, which is the only thing the
spec allows to be discussed. Spec §4.1 pre-declared U3 at every size as the
expected outcome — n = 37 on a 3.78× span, below the n = 40 at which Phase 3's L
cell missed excluding zero by 85 tokens — so eight U3 cells read as predicted, not
as a failure found late. **Corollary, binding on the talk: the break-even curve is
a ratio of means and may be stated as point estimates; the index may not be said
to "win" at any size, because the primary contrast resolved U3 at every size.**

The L cell at n = 37 (−3,228, [−12,884, +961]) against the published n = 40 cell
above (−5,217, [−16,019.5, +85.5], P3): same frozen runs, three questions fewer.
The difference is composition alone.

**Supporting mechanism, `n/a (not pre-registered)`.** The `turns` delta for
`indexed` − `agentic` is −1.0 at every size with an interval that excludes zero
(S [−3, −1], M [−2, −1], L [−2, −1]), and `indexed_topk3` − `indexed` is +1.0 at
every size — the bound arm compensates with more searches, as on `fastapi`. The
**dilution curve** (`contrast_kind == "size_span"`, within-arm L−S and L−M
paired deltas, exploratory): the index arm's prompt tokens grow with the corpus
(L−S median +2,002, [+482, +12,524]; L−M +905, [+205, +2,562]) while the grep
arm's do not detectably (L−S +35, [−1,927, +5,655]; L−M −1,387, [−9,497,
+1,164]). Neither is a U-letter, neither was pre-registered, and the spec said in
advance that nothing was expected of them over 3.78×; they are the mechanism the
non-monotone curve above would need explaining by, not an explanation of it.

**C-series — the matched-size contamination test: C1 on all three pairs**
(`matched_size_contamination.csv`). Two scopes are matched iff their measured
`count_tokens` are within 2.0× (`WS9_MATCH_BAND`, fixed in spec §3.2 before S
and M were measured); pairing is by tokens, never by file count. Three of nine
pairs are in band, the three the spec predicted, and no pair left or entered the
band on measurement. Per-query winner is the arm with the lower mean
`cost_agent_billed`, judge excluded; `fastapi` figures from `results/ws6a/scaling.csv`
(n = 10 at S and M) and `summary.csv` (n = 40 at L):

| pair | tokens `fastapi` / unseen | ratio | files `fastapi` / unseen | `fastapi`: agentic vs indexed | unseen: agentic vs indexed | branch |
|---|---|---|---|---|---|---|
| `fastapi` M vs `agentic-hil` S | 1,138,485 / 1,139,248 | **1.001** | 1,112 / 49 (22.7×) | **0.0355** vs 0.0400 → agentic | 0.1442 vs **0.0677** → indexed | **C1** |
| `fastapi` L vs `agentic-hil` L | 6,232,509 / 4,307,363 | 1.447 | 2,888 / 342 (8.4×) | **0.0543** vs 0.0666 → agentic | 0.1392 vs **0.0949** → indexed | **C1** |
| `fastapi` M vs `agentic-hil` M | 1,138,485 / 2,017,476 | 1.772 | 1,112 / 172 (6.5×) | **0.0355** vs 0.0400 → agentic | 0.1897 vs **0.0799** → indexed | **C1** |

**Overall C1.** At matched corpus size the per-query winner differs across the
two corpora, so size alone does not explain the `fastapi` result; contamination
stays the **candidate** explanation — this test rules size out, it does not rule
contamination in. The comparison is **unpaired and carries no interval**:
different questions, different n, no row correspondence, and the CSV's `pairing`
column says so on every row. It is a described comparison of point estimates,
exactly as the Phase 3 sign reversal above is. The file-count mismatch **cuts
against C1** — a small-file haystack favours grep, the arm C1 needs to lose on the
unseen side, and it lost there in every pair — so C1 was obtained against the
confound, not with it. The `file_ratio` column is **directional only**: it
divides WS6a's `files` (`results/ws6a/repo_stats.csv`) by
`files_text_allowlisted` here, and the two are not established to be the same
measure (the tracked counts here are 49 / 212 / 382). `fastapi` has no
`indexed_topk3` arm at S or M, so the C-series is defined over `agentic` vs
`indexed` only.

**The L caveats carry forward to every point drawn through L**, unchanged from
Phase 3: the paired median interval for `indexed` vs `agentic` prompt tokens
**straddles zero** (`sign_p` 0.081, `branches.csv`, branch P3); the agentic arm
**hit the turn cap on 2 of 40 runs**, so the saving is the **conservative end**;
and the per-arm medians in `summary.csv` carry the **opposite sign** to the paired
deltas on this same data — marginal medians are not tested comparisons (the
`indexed_topk3` contrasts: marginal +4,615.5 against `agentic` and +7,899 against
`indexed`, paired −4,717 and −3,961.5; `indexed − agentic` agrees in sign). On the
panel the first two recur at every size: U3 throughout, and the grep arm capped
on 1 / 2 / 2 of 37 runs at S / M / L.

**Branches that did not fire, named so the set is complete.** F2 and F3 (a
size with infinite break-even — none; grep was never the cheap arm); B2 (a finite
Q\* outside [10, 1000) — the six values run 23.7 to 82.5); K1 (the k=10 < k=3
order holding at every size — it reversed at M) and K3 (an infinite Q\* on one
arm); U2 anywhere (no index arm cost detectably *more* tokens); C2 (the same
winner on both corpora at matched size) and C3 (no pair in band). The one
exception to U3, the M `indexed_topk3` − `agentic` cell, fired U1 and is recorded
as such in the CSV; it is not a finding for the reason given above.

---

## Limits and caveats

**Statistical form.**

- **A difference of medians is not a paired statistic**, and on these
  right-skewed arms it either overstates a magnitude (2.44×, on `fastapi`) or
  reverses a sign (`indexed_topk3` − `indexed` on the unseen corpus). Every
  marginal per-arm total here is labelled marginal; only the paired tables are
  tested comparisons.
- **Phase 3's primary result is underpowered.** `[−16,019.5, +85.5]` at n = 40 is
  branch P3. n = 40 was inherited for comparability, not chosen for power, and
  **no power calculation was performed in advance**.
- **Two question sets, one comparison.** `fastapi`'s 40 questions and
  `agentic_hil`'s 40 are **different questions**, so the cross-repository
  comparison is **between-subject and unpaired**. Only within-repository deltas
  carry the paired intervals reported here.
- **n = 40 / 12 / 18 / 8 / 10 per cell.** Small. No confidence intervals across
  repositories; per-arm deltas are one observation, not a population estimate.

**What the corpora do and do not represent.**

- **`fastapi`'s library core is small** (48 `.py` files), so "where is X"
  questions are more greppable here than in a larger or less well-named codebase.
  The agentic arm won; that bounds how far it generalises. The honest claim is
  **"we did not reproduce the vendor's reduction on a small, well-named,
  heavily-documented Python library"**, not "the claim is wrong." The
  unseen-corpus phase was built to test exactly this and returns an underpowered
  answer, so the limitation is **narrowed, not retired**.
- **The corpora are matched on tokens and mismatched 8.4× on file count**, which
  favours the agentic arm. Not controlled away.
- **The within-repository curve measures dilution, not general scaling.** What
  grows across the three nested subsets is distractor composition — tests, doc
  examples, translations — while the answer always lives in the same 55 files. A
  faithful model of how real repositories grow; not a claim about arbitrary
  corpora.
- **The scaling subset is the most-contaminated slice**, `fastapi/`-confined by
  design, and the core library is the most heavily trained-on part of the repo.
  The parametric arm scores **0.600** on those same 10 questions against 0.625
  across all 40, so every 1.000 cell in that curve sits on a 60% floor and should
  be read as lift over it.
- **A synthetic agent history is not real agent history.** Right shape, controlled
  ground truth, generated language; token density is faithful (2.523 against a
  real 2.377) but *content* is not. **Distractor density is a parameter we
  chose** — ≥3 competing components per probe parameter, ≥5 conversational
  mentions per subject; more would make retrieval harder, fewer trivial. The
  corpus is English-only, single-project, single-voice. The `scaling` probes sit
  in the oldest ~8% of the largest history — deliberate, so size is the only
  variable, but it makes their results a statement about *old* facts specifically,
  and it is why truncated replay scores 0.000 there, which is arithmetic and not a
  result.
- **The prefix-stuffing arm is stuffing at 15.2% of the repository** (900 files,
  959,987 tokens of 6,306,239), not full-repository stuffing; the specification
  projected ~25% and the measured figure is lower because the repository is 6.23M
  tokens, not the 3.59M a chars/4 estimate suggested. Every result from that arm
  carries this qualifier. **Its file ordering is a judgement call** —
  deterministic and documented, but a different tiering would produce different
  numbers. `git ls-files` order is lexicographic, so a 960k-token lexicographic
  prefix would have contained thirteen documentation translations and **none of
  the library**, scoring near zero for a reason that is an artifact of alphabetical
  order; the committed ordering is a documented relevance tiering (`fastapi/` →
  `docs_src/` → `tests/` → `docs/en/` → rest, lexicographic within each tier). The
  rejected alternative is recorded so a hostile reader can audit the choice rather
  than take it on trust.
- **The out-of-prefix conditional is n = 4.** Reported with its n; a corroborating
  signal, not a measurement.

**What the arms are, and what tuning was not done.**

- **Lift over the parametric arm is a floor, not a decomposition.** A retrieving
  arm does not stop using parametric memory — it uses both, and retrieval can
  additionally *correct* a wrong prior, so lift can understate retrieval's
  contribution. No number here claims to isolate retrieval-only performance.
- **The agentic arm's tools are a reimplementation** of Claude Code's semantics,
  not Claude Code's actual tools. Their schemas and result formatting are
  committed; every claim about that arm is a claim about *those* tools.
- **The embedding model is claude-context's shipped default, not the best
  available**, on both corpora, so the indexed arm's quality is a floor;
  `voyage-code-3` is an unrun sensitivity. **memsearch likewise ran entirely at
  defaults** — `top_k`, chunk size and the shipped-disabled cross-encoder reranker
  are unrun sensitivities.
- **`WS6C_TOPK = 3` is a judgement call**, fixed before the run and recorded. A
  different value produces different numbers; the measured shipped default (10) is
  published alongside it so the pair can be audited, but **the curve between 3 and
  10 is unrun.**
- **A clean corpus removes the parametric floor but not the ceiling.** Accuracy on
  the unseen repository spans 0.900–0.975 across three retrieval arms at n = 40 —
  one to three questions apart. **That phase ranks cost, not quality**, and no
  quality claim should be drawn from those gaps. The same saturation holds for the
  memory benchmark: memsearch scored 1.000 in 62 of 62 runs, and **a benchmark its
  best arm never fails cannot rank retrieval quality, only cost.** The depth set at
  the largest history is the single place any quality signal appeared. If the talk
  wants a quality claim a harder probe set is needed; if not, the talk should be
  explicit that this is a cost result.
- **Progressive retrieval cannot be tested by this corpus.** Every planted fact is
  single-chunk, so L2 and L3 were never exercised.

**Measurement and scoring.**

- **`any_file_hit` is not a usable correctness measure on a contaminated corpus**,
  and a substring-based `stale_rate` would have published a 100% recency-failure
  rate for an arm that got all 8 right. Both mechanical scorers failed where the
  judge did not; both raw scores are published beside the judge-gated ones so the
  gap is auditable.
- **`embed_cost_usd` is derived**, not a usage field, in every directory here — a
  tiktoken/cl100k count times a published list rate, where cl100k runs ~26% below
  Claude's tokenizer. Neither package surfaces an embedding usage field. Using
  Claude's count would overstate the index's fixed cost in the direction that
  flatters live search.
- **`mean_chunk_tokens` in `results/ws6a/index_cost.csv` mixes two file sets** —
  our allowlisted-file token count over claude-context's chunk count on *its* file
  set (2,825 files vs our 2,888) — and should not reach a slide. Use the
  transcript-measured payload figure (median 12,014 chars per call) instead.
- **Latency is not quotable.** In the memory benchmark it is confounded with
  execution order — cells ran contiguously to keep the cache warm, on top of the
  existing laptop caveat — so **no latency number from that directory should reach
  a slide**; token and cost columns are unaffected. In the unseen-corpus phase
  `wall_clock_s` is rate-limit throttled and not comparable across directories, and
  the 174.3s index build ran at `embedding_batch_size = 20` against a package
  default of 100, so it is an upper bound, not a default-configuration measurement.
- **Local Milvus in the memory benchmark, Zilliz Cloud in the code-search
  benchmark.** Latency and infrastructure cost are **not comparable** between them
  and must not be charted together.
- **The index's running cost is absent.** Local Milvus is free; a hosted index is
  not. Every cost conclusion here is **per-query only**, and "repays its build in
  0.07 queries" amortises embedding cost alone.
- **Cache re-writes during interleaving** add cost variance to the stuffed arm —
  visible in the data (one 960,096-token write against 37,443,744 tokens of reads),
  not controlled away.
- **The turn cap truncates the tail** of both tool-using arms, so median latency
  and token figures are conditioned on it; the per-arm `hit_turn_cap` rate is
  published so the truncation is visible. The cap of 15 was applied to **both**
  tool-using arms deliberately — capping one alone would have made the arms differ
  in tool availability *and* turn budget, and part of the indexed arm's advantage
  would then be an artifact of being allowed to run longer. One question (`q40`,
  agentic) hit the cap and returned an empty answer; it is scored incorrect and
  reported, not dropped (cap rate: agentic 0.025, all others 0).
- **`q40`'s ground truth on the unseen corpus carries 5 `expected_files`** — one
  more than any question in the earlier set — so its `all_files_hit` is the hardest
  row in either set and is not directly comparable.
- **The Phase 3 question set is more directory-concentrated and *broader* in what
  it asks.** 37 of 40 rows resolve entirely inside `src/agentic_hil/` against 32 of
  40 inside `fastapi/`, but it names **31 distinct expected files against 26**, so
  the more concentrated set is the one that touches more of its repository. The
  concentration reflects repo shape — 49 core files of 342, against a flatter
  2,888-file tree. These four figures, and the four sensitivity rows in the
  fragility table, are **the only numbers here without a results CSV of their
  own**: both question sets are committed, so they are reproducible, but no
  committed CSV states them.

**Provenance, authorship, and spend.**

- **Question authorship bias.** The same author wrote the questions and built the
  arms on both real repositories. Mitigated by committing the question set *before*
  the corpus manifest / index existed, so blind authoring is provable from git
  history rather than asserted (gate questions committed 11:44:02, every gate run
  11:48–11:54, results 11:57:33), by blind judging, and by a judge model different
  from the agent model. **The synthetic benchmark's fix — mechanically derived
  probes from planted triples — is unavailable on a real repository.** Reduced, not
  eliminated.
- **The Phase 3 parametric arm re-asks 25 of the gate's own question ids**, so its
  0.000 is **25/40 pre-measured** and is **not an independent replication** of the
  gate result. Reported as corroboration, not confirmation.
- **Phase 1's deltas are cross-session AND cross-date, against an aliased model
  id.** The constrained arm ran 2026-09-08; its comparison rows were measured
  2026-08-18. Three weeks separate them, they were not interleaved, and the agent
  pin is an **alias rather than a dated snapshot**, so a served-model change inside
  that window is neither excluded nor detectable from our data. **This bears on a
  claimed result:** `indexed_topk3` − `indexed` = −4,429.5, CI [−8,385.0,
  −1,890.0] is the whole evidence for "the knob demonstrably works." The corpus,
  collection, question set, system prompt and harness are provably identical across
  the gap, and a 4,430-token median shift is large against plausible drift — but a
  same-session interleaved re-run is the experiment that would settle it, and it
  was not run.
- **The two indexed arms are not interleaved with each other.** claude-context
  builds its tool list at session entry, so binding `limit` requires a second MCP
  session; Phase 3 is two sequential runs against one checkpoint. Each arm is
  interleaved against the same question order under the same seed and the same
  index, so the two primary comparisons are protected against time-of-day API
  variance. **`indexed_topk3` − `indexed` is not.**
- **A 27-question duplicate execution sits behind Phase 1's 40 rows.** A
  backgrounded run overlapped a rerun, giving 67 checkpoint lines over 40 distinct
  `(qid, arm)` pairs. The committed `topk_runs.csv` is sound and was audited before
  acceptance — 40 rows, 40 unique qids, one arm label, 0 errors — and the dedup rule
  was verified **empirically, not assumed**: for all 26 pairs whose two executions
  disagree on token totals, the retained row is the **last** one positionally
  (26 last-wins, 0 first-wins). The selection is positional and therefore cannot see
  the measured values, so it cannot bias the metric. The checkpoint file is a local
  resume cache and is **not committed**, so this is the only public record of it.
  Anyone re-running Phase 1 will produce 40 lines, not 67.
- **A governor defect, found and fixed mid-run.** `total_spent_across_checkpoints`
  globbed only the earlier directory's paths, so every later checkpoint was
  invisible to it and each phase would have re-authorised nearly the full cap on
  top of the last. Fixed before the next dollar was spent. A pre-existing
  under-count remains: `checkpoint_spent` sums *after* last-wins dedup, so it
  recorded $2.72 for Phase 1 against $4.56 actually billed. **That under-counts,
  which is the unsafe direction for a budget guard**, and it is recorded rather than
  silently carried. Final Phase 3 spend: **$43.44 governor-reported, ≈$45.3 actually
  billed** — about $0.3 over the raised cap, for the same deduplication reason.
- **The budget cap was raised mid-run, on explicit user authorisation.** Phase 3
  halted at 65/160 pairs against a $40 governor once the governor was corrected and
  measured per-pair cost came in at **1.6–1.9× the plan's estimate**
  (`benchlib/config.py`'s budget comment; not the 1.3–2.3× measured-against-measured
  cross-corpus ratio above — a different ratio on a different base). Three options were put to the
  user — raise to $45 and finish all four arms, drop the constrained arm and finish
  inside $40, or stop and write up 65 rows. **The user chose to raise the cap**,
  because dropping an arm *after* 65 rows are visible is a post-hoc design change
  made with data in view. `WS6C_BUDGET_USD` 40.00 → 45.00 is the only pin changed.
- **Budget, not science, became the binding constraint.** No sensitivity variant —
  a scope-matched cut, a second top-k value, a re-run — could have been afforded.
  The scope-matching rule was in any case **unexecutable**: the selected corpus
  lands between two `fastapi` scopes, nearest the largest, and **a corpus cannot be
  cut *up* to a larger scope**. The alternatives were to cut down, discarding 74% of
  the corpus, or to run the whole repository and state the ratio; **the latter was
  taken** and the ratio 0.691 is stated everywhere the cross-repository comparison
  appears. Honest caveat: at $11.44 remaining against a ~$6.70 phase, the cut
  variant could not have been afforded as a sensitivity even if a reader demanded
  one.
- **Two runtime failures in the code-search harness, both fixed and both recorded**
  because they shaped it. The judge had `max_tokens=512` covering thinking *and*
  output; it occasionally spent the budget thinking and returned no text block, and
  `json.loads("")` killed the run 15 pairs in — now an escalating budget
  (4,096 → 16,384) with usage accumulated across attempts. A transient `529
  Overloaded` killed the run again at 32 pairs, because `run_arm` already swallowed
  its own exceptions, making the judge the only unguarded API call in the loop — now
  `max_retries=8`, and a judge failure skips the pair rather than aborting.
  Checkpoint-before-charge meant no completed pair was lost to either crash; one
  consequence recorded for traceability is that the pair in flight during the first
  crash was billed by the API but never charged to the governor, so recorded spend
  under-reports actual spend by roughly one pair (~$0.05). The code-search
  governor itself was seeded cumulatively from every phase's checkpoint rather than
  per-script — a per-script governor would have authorised roughly $300 against a
  $75 cap.
- **A derived corpus tree.** Both tool-using arms on the unseen repository ran
  against a copy of the pinned checkout with one file removed. The measured corpus
  is byte-identical and the tree is sha256-pinned, but **the indexed tree is not
  literally the pinned checkout.**

---

## Appendix — arm by arm

### A. Code search, `fastapi` — `results/ws6a/summary.csv`

n = 40 per arm; marginal per-arm figures, not paired deltas. "Lift" is over the
parametric floor; it is a floor, not a decomposition. **The `$/question` column
here is agent-billed only** — appendix C's cross-repository cost table is agent
**and** judge, which is why the same `agentic`/`fastapi` cell reads $0.0543 here
and $0.0659 there.

| arm | median prompt tok | median out | median turns | median / p95 latency | judge acc | lift | $/question (agent only) |
|---|---|---|---|---|---|---|---|
| parametric | 129 | 556 | 1.0 | 6.8s / 34.0s | 0.625 | — | $0.0090 |
| indexed | 16,452 | 696 | 2.0 | 12.2s / 46.0s | **1.000** | **+37.5pp** | $0.0666 |
| agentic | 9,485 | 638 | 3.5 | 11.3s / 31.9s | 0.975 | +35.0pp | $0.0543 |
| stuffed | 960,124 | 664 | 1.0 | 10.3s / 26.6s | 0.900 | +27.5pp | $0.2941 |

Per-query totals, marginal (not paired):

| arm | median prompt tok | total prompt tok | cache read tok | billed | at list price | saving |
|---|---|---|---|---|---|---|
| parametric | 129 | 5,194 | 0 | $0.36 | $0.36 | 0% |
| agentic | 9,485 | 898,513 | 0 | $2.17 | $2.17 | 0% |
| indexed | 16,452 | 1,127,617 | 0 | $2.66 | $2.66 | 0% |
| stuffed | 960,124 | 38,404,994 | 37,443,744 | $11.76 | $77.24 | **84.8%** |

The originally published C11 table, **historical — read the correction section
before quoting any row**: median prompt tokens 16,452 vs 9,485, published as
**+73.5%** (paired median **+2,856.5**, CI [+9.0, +7,691.5]); total prompt tokens
1,127,617 vs 898,513, +25.5% (marginal); agent cost over 40 questions $2.66 vs
$2.17, +23% (marginal); judge accuracy 1.000 vs 0.975, published as **+2.5pp**
(paired median **0.0**, CI [0.0, 0.0] — **withdrawn**).

Within-repository dilution, `results/ws6a/scaling.csv`, 10 pre-registered
questions at three nested subsets of the same repository at the same commit
(marginal medians; the paired re-analysis is in the correction section):

| size | corpus tokens | files | agentic median prompt | indexed median prompt | judge acc (both) |
|---|---|---|---|---|---|
| S | 259,429 | 55 | 6,040 | 10,615 | 1.000 |
| M | 1,138,485 | 1,112 | 7,873 | 16,155 | 1.000 |
| L | 6,232,509 | 2,888 | 8,151 | 13,298 | 1.000 |

The x-axis is **measured tokens, never file count**: file counts step 20× then
2.6×, which would make the curve's shape an artifact of how the subsets happen to
divide, while token steps are 4.4× then 5.5× — near-even in log space, span 24×.
Indexed is non-monotonic (10,615 → 16,155 → 13,298), which is what a fixed top-k
retrieval plus n = 10 noise should look like.

### B. Agent memory — `results/ws6b/summary_{s,m,l}.csv`

| size | corpus tokens | memsearch $/q | replay $/q | ratio | memsearch acc | replay acc |
|---|---|---|---|---|---|---|
| S | 98,134 | **$0.01004** | $0.01994 | 2.0× | 1.000 | 1.000 |
| M | 392,116 | **$0.01020** | $0.07890 | 7.7× | 1.000 | 1.000 |
| L | 1,221,906 | **$0.02318** | $0.19405 † | 8.4× | 1.000 | 0.474 |

† truncated replay — plain replay cannot run at L. Its `scaling` accuracy is 0.000
by construction, because those facts sit in the oldest 8% of the history, outside
its window; that is arithmetic, not a result.

| arm | S | M | L | growth over a 12.45× corpus |
|---|---|---|---|---|
| memsearch median prompt tokens | 4,554 | 4,636 | 10,593 | **2.33×** |
| replay median prompt tokens | 98,244 | 392,189 | 959,269 † | 9.8× |

### C. Top-k and the unseen repository — `results/ws6c/summary.csv`

Marginal medians and accuracy on `agentic-hil/agentic-hil`. ⚠️ **Marginal, not
paired** — the `indexed` column reads 15% below `agentic` here while the paired
delta is branch **P3, no separation**, and on the `indexed_topk3` row the two
statistics disagree in **sign**. Read the paired tables above for the verdict and
this one for magnitudes only:

| arm | median prompt tok | median turns | judge accuracy | turn-cap rate | $/question (billed) |
|---|---|---|---|---|---|
| parametric | 139 | 1.0 | **0.000** | 0 | $0.0096 |
| indexed | **18,078.5** | 2.5 | **0.975** | 0 | $0.1026 |
| indexed_topk3 | 25,977.5 | 4.0 | 0.950 | 0 | $0.1110 |
| agentic | 21,362.0 | 4.0 | 0.900 | 0.05 | $0.1488 |

Per-question cost across both repositories (`summary.csv`; marginal per-arm
totals, **not** paired deltas — nothing in this table is a tested comparison):

| | fastapi | agentic_hil | ratio |
|---|---|---|---|
| agentic | $0.0659 | $0.1488 | **2.26×** |
| indexed | $0.0780 | $0.1026 | 1.32× |
| indexed_topk3 | $0.0680 | $0.1110 | 1.63× |
| parametric | $0.0202 | $0.0096 | 0.48× |

Contamination across every corpus measured — the spectrum behind C15:

| corpus | parametric judge accuracy | source |
|---|---|---|
| `fastapi` | **0.625** | `results/ws6a/summary.csv` |
| synthetic agent history | **0.0000** | `results/ws6b/parametric_gate.csv` |
| `agentic-hil/agentic-hil` | **0.000** | `results/ws6c/nonmemorization_gate.csv` |
| `oaslananka/kicad-mcp-pro` | **0.000** | `results/ws6c/nonmemorization_gate.csv` |
| `gmboquet/mixle` | **0.000** | `results/ws6c/nonmemorization_gate.csv` |
