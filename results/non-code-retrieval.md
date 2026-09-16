# Non-code retrieval — does an index beat agentic grep on prose?

Covers `results/ws8/`. Claim C15.

**Source of truth.** Every number cites the committed CSV in `results/ws8/` it
comes from, named inline. All of them are written by `scripts/non-code-retrieval/summarise_ws8.py`,
deterministic at seed 42, which regenerates every figure here from the frozen
`runs.csv` (150 paired rows) at **$0**, byte-identically. The statistics live in
`benchlib/non_code_retrieval/bench.py` and `benchlib/non_code_retrieval/stats.py`.

**One declared exception.** The correlation figures in the warning under "A
killed framing" come from a $0 exercise whose inputs were never committed. They
are **prose-sourced, not CSV-sourced**, and are recorded only so the number can
be recognised and refused. Everything else has a CSV.

**How to read this.** "The question, and the answer" carries the result, which
is a negative; a reader who wants only the conclusion can stop there. "What was
measured" is the instrument and the gates it passed or failed; "Mechanism" shows
the effect the hypothesis proposed is absent. "Limits and caveats" is the audit
trail, long on purpose — the power limitation in it is the most important
qualification on everything above, and the transcript-deletion cost at the end
of it is a disclosed hole in this benchmark's auditability. The appendix holds
the corpus, secondary and spend tables.

---

## The question, and the answer

Does a retrieval index beat agentic grep on **prose**, as a function of how many
documents an answer spans? This is the first non-code evidence in the project.

**No effect was detected, and the result is confounded on one of its own branch
conditions. Branch `H4-confounded`, both sensitivity variants.** Separately, the
design's own power analysis was wrong by about 5×, which makes that negative
uninformative below ~100,000 tokens. Both facts are measured, so both are here.

**This gives the talk no closing claim.** There is no "the index wins when the
answer isn't in one place." The test built for that sentence returned a
confounded null, and the honest reading of the unconfounded version is also a
null. The negative *is* the finding — not a partial success, and not a result
that more questions **at this dispersion** would rescue (the limit is
dispersion, not n). ⚠️ Below ~100,000 tokens the power analysis puts the null
out of reach entirely, as stated above; the claim here is about the region the
design does reach.

### The instrument

| | |
|---|---|
| **Primary outcome** | paired `indexed − agentic` **prompt tokens** |
| **Primary test** | the **contrast** between the `hop3plus` and `hop1` strata |
| **Method** | unpaired bootstrap on the difference of two paired medians, seed 42, `n_boot=10,000`, α 0.05 |
| **Landed on** | **`H4-confounded`** (`results/ws8/branches.csv`) |

The primary is a *contrast*, not a cell: reporting one cell would reproduce the
difference-of-medians error the code-search benchmark (`results/ws6a/`) made and
its follow-up (`results/ws6c/`) had to correct after publication. The branch is
a **lookup, not a judgement** — `resolve_hop_branch` in `benchlib/non_code_retrieval/stats.py`
reads two booleans and one interval direction and returns one of four strings,
H1–H4.

### The primary contrast

`results/ws8/branches.csv`

| variant | n (hop3plus/hop1) | contrast median | 95% CI | excludes 0 | hop1 separates | **branch** | branch if hop1 were null |
|---|---|---|---|---|---|---|---|
| **all pairs** | 30/30 | **−42,616.5** | **[−162,312.5, +39,690.0]** | No | **Yes** | **`H4-confounded`** | `H2` |
| **excl. `ws8_q067`** | 29/30 | **−43,743.0** | **[−167,415.5, +37,616.5]** | No | **Yes** | **`H4-confounded`** | `H2` |

**The two sensitivity variants agree**, so the censored-row rule's "publish both
and prefer neither" clause is not triggered on the primary: dropping the
censored row moves the point estimate by 1,127 tokens (2.6%) and changes
nothing.

**Two independent reasons this is not a finding.**

1. **H4 fired.** The design predicted `hop1` would be null. It is not: it
   separates grep-favourably at **+82,678, CI [+17,402, +195,849]**, sign p =
   0.016, 22 of 30. The H4 rule — when hop=1 separates, the contrast cannot be
   read as a hop effect *regardless of what hop≥3 does* — is checked first.
2. **The contrast spans zero anyway.** Absent H4 the lookup returns **H2 — no
   effect detected**. That is why `branches.csv` carries
   `branch_if_hop1_were_null` as its own column: `H4-confounded` alone reads as
   *"there was a finding and a confound spoiled it."* **There is no positive hop
   finding being confounded away. There is no positive hop finding.**

Under H2 we do not go looking for a subgroup that would supply one, and **none
was computed** — no repo split, no outlier removal, no pooling of strata, no
re-reading of the branch definitions after seeing the numbers.

### The null is uninformative below ~100,000 tokens

> **Say "no effect detectable at this dispersion." Never "no effect exists."**

The design sized n=30 for a detectable floor of ~25,000 tokens, transferred from
**code** dispersion. Prose deltas turned out **2.5–13× more dispersed**, and
re-running the design's own simulation on the *observed* dispersion puts 80%
power around **125,000 tokens — five times that floor**. **The observed contrast
is −42,617: larger than the floor, still far below what this design could
see.** Full table and three honesty notes under "Limits and caveats"; this
qualification travels with the negative everywhere.

### What it does give

1. **A negative that survived a hostile review**, with its power limitation
   quantified rather than hand-waved.
2. **A 0.000 contamination gate on two of the most heavily-trained-on
   repositories in existence.** Post-cutoff *content* suffices even where the
   model knows the codebase — this relaxes a real constraint on every future
   harness (C15). The declining-rather-than-guessing evidence behind it is no
   longer independently checkable ("What the transcript removal costs").
3. **A corpus where accuracy did not saturate** — a harder probe set exists, and
   this is what one looks like.
4. **A secondary observation** that a naively-configured index (top-k 10 ×
   1,800-char chunks) costs ~4× the prompt tokens of agentic grep at every hop
   count while being no more accurate, the cost sitting in **payload per turn**,
   not turn count, and the payload being a **configuration constant**.
5. **A design lesson**: a contrast test licenses only a contrast claim (below).

**The one-line honest framing:** *"We built a test for when prose retrieval
beats grep, ran it on a corpus the model had never seen, and it came back null —
and our own power analysis says we could only have seen an effect five times
larger than the one we were looking for."*

### The most quotable thing here — and why it stays a secondary

The index cost **more** than grep in every stratum while being **no more
accurate**: the most slide-ready sentence available.

> ⚠️ **It is built entirely from secondaries and must be labelled a secondary
> observation every time it appears — never as this benchmark's finding.** The
> finding is the negative above. H2 says the talk does not get its close and we
> do not hunt a subgroup that would supply one; reaching for a different
> headline from the secondaries is the mirror image of that.

State the multiple honestly: the ratio of median prompt tokens is **4.0× at
`hop1` and 3.9× at `hop3plus`**, the two powered strata. The tempting "3–13×"
takes its upper bound (13.4×) from **`hop2`, the cell that cannot carry a
claim**. Say **~4×**, or "~4× in both powered strata, 13× in the unpowered one."

---

## What was measured

The corpus is GitHub issue and PR threads from three repositories, in a window
that **postdates the model's training cutoff**, so the *text* is unseen; it is
fetched and never redistributed. Full statistics in the appendix.

### A killed framing, and the number it left behind

The organising variable was almost **lexical overlap** — "grep wins when the
question hands the model a rare token that appears verbatim in the evidence."
Tested under a kill criterion against the code-search question sets on `fastapi`
and `agentic-hil`, it failed on both; the metric survived only as a
**covariate**, where it did its job (below).

> ⚠️ **A pooled correlation across both corpora reads ρ = +0.270, p = 0.016, CI
> [+0.041, +0.469]. It must never be quoted.** It is a Simpson's paradox
> artifact — `fastapi` has both the higher mean grep-handle and the positive
> median delta — and it exists within *neither* corpus. It is recorded because
> it is the number most likely to survive onto a slide.

### The parametric gate — the strongest single result here

`results/ws8/parametric_gate.csv`

| n | accuracy | `any_doc_hit` | branch |
|---|---|---|---|
| 75 | **0.000** | **0 / 75** | **G1 (pass)** |

A parametric arm with no corpus access — same questions, no tools — scored
**zero**, and did so **in every stratum separately** (`summary.csv`: `hop1`
0.000, `hop2` 0.000, `hop3plus` 0.000), not merely in aggregate.

**This is a genuine methodological result, and it relaxes a real constraint.**
The earlier code work (`results/ws6c/`) needed *obscure* post-cutoff
repositories to find an uncontaminated corpus. Here post-cutoff **content**
suffices **even where the model knows the codebase intimately**: a **0.625**
parametric floor was measured on `fastapi`'s own code (`results/ws6a/`), and
`kubernetes` and `rust-lang` are among the most heavily-trained-on repositories
in existence. The parametric transcripts showed the model **declining** — *"I
don't have access to the corpus"* — rather than guessing wrong, and that
contrast, not the score alone, was the evidence.

> ⚠️ **Those transcripts were deleted for the licensing reason below, so a
> reader can verify the 0.000 score from `parametric_gate.csv` and `summary.csv`
> but CANNOT independently confirm the declining-vs-guessing distinction.** The
> claim rests on `any_doc_hit = 0/75` plus this report. It feeds C15; weigh it
> accordingly.

**Caveat on `any_doc_hit = 0/75`**: it is partly an instruction-following
artifact. The first gate run used the shared code-search system prompt, which
instructs the model to cite *Python file paths and function names* — wrong three
ways for a corpus of Go/Rust issue threads named
`kubernetes__kubernetes__140406.md`. The gate was **re-run under a corrected
prompt at $0.844041** (the superseded run cost $0.783336 and is archived, not
deleted) so the two paired arms would not differ in two variables. Accuracy
**0.000** stands under both, but the model was never told to cite documents in
that form, so `any_doc_hit` reads weaker than the underlying signal.

### Per-stratum results

`results/ws8/paired_ci.csv` · paired `indexed − agentic` prompt tokens.
**Positive = the index spent MORE.**

| stratum | n | powered | median Δ | 95% CI | excludes 0 | sign p | index dearer |
|---|---|---|---|---|---|---|---|
| `hop1` | 30 | yes | **+82,678** | [+17,402, +195,849] | **Yes** | 0.016 | 22/30 |
| `hop2` | 15 | **NO — n=15 by design, cannot carry a claim** | +171,479 | [+8,402, +524,495] | Yes | 0.007 | 13/15 |
| `hop3plus` | 30 | yes | **+40,062** | [+5,297, +75,281] | **Yes** | 0.016 | 22/30 |

`hop2` carries `powered = False` and the string *"NOT POWERED: n=15 by design
(spec section 3); reported as-is, cannot carry a claim"* **in the CSV itself**,
not only in this prose.

**All three strata separate, and every one separates grep-favourably.** On this
corpus the agentic arm is cheaper in prompt tokens at every hop count — the same
direction the code-search benchmark found, now on prose.

### The separation of `hop1` is robust — we tried to break it

Three checks, all measured during review (`results/ws8/robustness.csv`):

- **Not a calibration artifact — but not perfectly calibrated either.** The
  paired test's false-positive rate at n=30, measured on `hop1`'s own centred
  (heavy-tailed) shape, is **0.056 ± 0.003** against a nominal 0.05 (5,000
  studies, converging to 0.054 at 20,000): **very slightly anti-conservative**.
  Within Monte Carlo error of nominal and nowhere near enough to manufacture the
  separation, but "0.056 against a nominal 0.05" is the defensible statement and
  "exactly calibrated" is not.
- **Not an outlier artifact.** Across **all 435 leave-2-out subsets** of `hop1`
  the minimum bootstrap CI lower bound is **+9,389.5**, still excluding zero:
  **no pair of observations can be removed to un-separate `hop1`.** The margin
  does erode — leave-**one**-out floors at **+17,402**, leave-**two** at
  **+9,389.5**, leave-**three** at **+1,377**, so three well-chosen removals
  would very nearly do it. **All 4,525 subsets across k = 1, 2, 3 still exclude
  zero**, without exception. Deltas run to +1.35M tokens, so "isn't this just
  outliers?" is the expected hostile question; it is not, and this is a harder
  answer than "it's a median and the sign test agrees."
- **The null contrast is not a rigged test.** Its false-positive rate under a
  true null at the observed dispersion is **0.034 ± 0.003** against a nominal
  0.05 — slightly conservative. The wide interval is genuinely wide. The null
  pool is a **methodological choice**, so `robustness.csv` publishes all three
  defensible ones rather than the flattering one: both strata pooled (the
  headline, 0.034), `hop1` alone (0.031), each cell keeping its own dispersion
  (0.041). The conclusion holds under all three.

### Who computed what, and what the review found

The person coordinating the runs was progressively exposed to outcome data
during the paired phase — per-stratum costs, turn distributions, which rows were
judged correct — through legitimate monitoring of stop conditions, cumulatively
amounting to a picture of the result before the analysis ran. **The analysis was
therefore run independently, by someone with no exposure to any of that data,
and independently reviewed.** The branch assignment is a table lookup over the
committed statistics; the exposure is recorded because it was real.

The review re-derived every statistic independently, without importing the
analysis module: **zero critical findings**, the verdict reproduced exactly, the
summariser regenerates byte-identical CSVs, and — **at the time of that review
(2026-09-11)** — 421 tests passed and 470 checksums verified. Both counts have
grown since; today's gate is `pytest tests/ -q` and
`scripts/fixtures/verify_checksums.py --strict`. Fixes that landed as a result (commits `737bec9`, `e5990fd`):

- The **H4 guard failed open** — `hop1.get("excludes_zero")` returns `None` for
  a missing key, so a wrong-shaped dict would have made H4 silently never fire.
  Now indexed with `[]` and fails closed. This was the one guard the whole
  result rests on.
- H1/H3 now key off the **CI direction**, as the branch definitions word it,
  rather than the point estimate's sign.
- `branches.csv` gained **`branch_if_hop1_were_null`**, an expected-n assertion,
  and the `declared_floor_tokens` / `observed_power_at_declared_floor` /
  `gradient_for_80pc_power` columns.
- The power simulation **shared one bootstrap index matrix across all studies**
  and used S=300/B=600 against the design's S=400/B=1,000. Both fixed; a **null
  calibration row** and the **MAD columns** were added so the CSV carries its
  own evidence.

### The most valuable finding was a defect in the design, not the code

The H1 branch **licenses an absolute claim from a relative test**: it would have
authorised the talk's closing line — *"the index wins when the answer isn't in
one place"* — on a separating index-favourable **contrast alone**. But a
contrast only establishes that the index does relatively *better* at hop≥3 than
at hop1, and **on this data those two things came apart**: `hop3plus` is
**+40,062, CI [+5,297, +75,281]**, i.e. the index is *dearer* at hop≥3 too. Had
the contrast separated with `hop1` quiet, the lookup would have returned **H1 —
correctly, per the branch definitions as written** — and the talk would have
claimed the index wins in a cell where it demonstrably loses. **That outcome was
one bootstrap interval away.**

The rule that should have been written, and that future designs here get: **a
contrast test licenses a contrast claim; any absolute "X wins" claim requires
the absolute cell to be tested separately, as an explicit conjunction.** Same
family as the code-search difference-of-medians error — a statistic that does
not measure the quantity the sentence claims — which `results/ws6c/` had to
correct after publication. It is recorded as a dated, post-data amendment in the
design document, marked as changing no result here; the ordering is verifiable
from git — the analysis landed in `987d223`, the amendment in `bdd77ef`.

---

## Mechanism — the proposed mechanism is absent

`results/ws8/summary.csv` · `results/ws8/paired_ci.csv` (metric `turns`)

The hypothesis proposed a mechanism: grep needs more round trips as evidence
spreads across documents. **No such effect is detectable.**

| stratum | median turns agentic | median turns indexed | paired Δ | 95% CI | excludes 0 |
|---|---|---|---|---|---|
| `hop1` | 5.0 | 4.5 | **0.0** | [−1.0, +1.5] | No |
| `hop2` | 5.0 | 6.0 | −1.0 | [−1.0, +5.0] | No |
| `hop3plus` | 4.0 | 4.0 | **0.0** | [−1.0, +1.0] | No |

**Turn counts are statistically indistinguishable between arms in every stratum,
and flat across strata**, so **no per-hop turn cost is detectable in the agentic
arm**. Whatever this benchmark measured, it is not the thing the hypothesis was
about.

**Where the cost actually comes from.** Median prompt tokens **per turn**:

| stratum | agentic | indexed | ratio |
|---|---|---|---|
| `hop1` | 4,171 | 21,006 | 5.0× |
| `hop2` *(not powered)* | 3,222 | 31,948 | 9.9× |
| `hop3plus` | 4,336 | 16,492 | 3.8× |

The gap is **payload per turn**, not turn count, and that payload is a
**configuration constant**: `WS8_TOP_K = 10` × `WS8_CHUNK_CHARS = 1,800` ≈
16–18K chars on **every** search call, fixed by construction.

> ⚠️ **Do not reuse the code-search headline "the indexed arm returns ~2.7× more
> text per call" here.** There, `claude-context` returned *variable-width code
> chunks* and the ratio was emergent. Here the chunks are **fixed width**, so
> the ratio is **structural, not emergent**. The observed driver is
> **search-call count**.

> ⚠️ **And "top-10" does not mean ten documents.** `WS8_TOP_K = 10` returns ten
> **chunks**, and a document long enough to be split contributes more than one.
> A verified sample query returned only **8 unique documents** in its top-10, so
> the indexed arm's effective *document* recall is below what `k` implies and
> **varies per query** — which matters directly for a benchmark whose whole
> question is how many documents an answer spans. It does not break
> comparability with `results/ws6a/` or `results/ws6c/` (`claude-context`'s
> `limit = 10` hits the same chunked-collection knob), but an audience will hear
> "top-10" as ten documents and it is not.

**The covariate — the one place the design comes out clean.**
`results/ws8/paired_ci.csv`, `mean_grep_handle` / `median_grep_handle`: `hop1`
6.231 / 6.306, `hop2` 6.438 / 6.745, `hop3plus` 6.081 / 6.269. Flat on means,
medians and ranges (3.49–8.25, overlapping). **No stratum was handed its answer
more readily than another**, so the contrast is not a lexical contrast in
disguise. This is convenient for us, which is why it was checked three ways —
and it is a property of the question set, fixed before any run, so it cannot
have been shaped by the outcome.

---

## Limits and caveats

### The threats named in the design, and what happened to each

1. **The hypothesis came from the data it was tested against.** Mitigated only
   by the corpus being new and the questions mechanically derived. —
   *Mitigation held.*
2. **Effect sizes were transferred from code to prose**, named in the design as
   **"the single largest risk to the design."** — **It materialised (below).**
3. **n=30 is sized for a ≥25,000-token effect; smaller effects read as H2
   nulls.** — *Worse than that (below).*
4. **Corpus quality varies.** Issue threads range from precise to noise; bot
   exclusion helps but does not make the corpus uniform. — *Unresolved by
   construction.*
5. **The index may lose on multi-hop too; H3 is a live branch.** — *It did lose,
   in every stratum.*
6. **Budget.** — *Came in at $50.85 against a $70 governor.*
7. **Single corpus.** This adds **one** non-code point. **It does not establish
   a law across prose corpora and nothing here may imply one.**

An S/M/L corpus **size sweep was removed from scope and never run**, so nothing
here speaks to how any of this scales with corpus size — untested, not null.

### The power analysis was wrong by ~5×

The most important limitation, and it cuts against us. The design states *"the
detectable floor at n=30 is ~25,000 tokens"*, transferred from `agentic-hil`
**code** dispersion because nothing better existed. Measured dispersion of the
actual paired deltas:

| sample | n | median | MAD | max |
|---|---|---|---|---|
| `agentic-hil`, code (`results/ws6c/`, the power-analysis basis) | 40 | −5,217 | 16,918 | +94,213 |
| `hop1` | 30 | +82,678 | **100,877** | **+1,348,708** |
| `hop2` | 15 | +171,479 | **215,767** | +1,754,076 |
| `hop3plus` | 30 | +40,062 | **42,159** | +1,605,269 |

**Prose deltas are 2.5–13× more dispersed than the code deltas the design was
sized on.** Re-running the design's own simulation (S=400 studies × B=1,000
resamples, seed 42) on the **observed** dispersion —
`results/ws8/mde_retrospective.csv`, post-hoc, verdict-neutral:

| injected gradient | power at n = 30/30 |
|---|---|
| **0** *(null calibration)* | **0.043** ← the false-positive rate; it must sit near α = 0.05, and it does |
| 25,000 *(the design's floor)* | **≈0.045–0.07** |
| 50,000 | ≈0.14 |
| 100,000 | **≈0.69–0.76** |
| 125,000 | ≈0.86 |
| 200,000 | ≈0.98 |

**The observed contrast is −42,617 — larger than the design's stated floor, and
still far below what this design could actually see.** 80% power arrives around
**125,000 tokens — five times that floor**. (That multiple is quoted from the
80%-power crossing throughout; the 100,000 row, where power first exceeds
two-thirds, sits at ~4× that floor — a multiple of the *detectable floor*, not
the cost ratio quoted earlier. One number, one meaning: **5×**, at 80% power.)

> **Say "no effect detectable at this dispersion." Never "no effect exists."**

Three honesty notes on that table:

- **The figures carry Monte Carlo noise and must not be quoted to three
  decimals.** Across four outer seeds the 25,000 row lands at 0.065 / 0.052 /
  0.060 / 0.045 and the 100,000 row at 0.728 / 0.693 / 0.725 / 0.755. Quote
  **ranges**.
- **The grid resolution is part of the measurement.** The 125,000 and 150,000
  points were added *after* seeing that the coarse grid reported the 80%
  crossing at 200,000 — overstating it by ~1.6×. The refinement moved the number
  **against** us, tightening the claimed floor.
- **Not an artifact of the contrast being a two-cell statistic.** Re-running the
  design's own simulation at `agentic-hil` dispersion gives **0.85** power at a
  25,000 *single-cell* shift (reproducing the design's stated floor) and **0.78**
  at a 25,000 *contrast* gradient. The ~5× miss is prose dispersion, not the
  arithmetic of differencing two cells.

**This is the second consecutive piece of work here to ship an interval wider
than its design assumed** (after `results/ws6c/`) — the first time from a
missing power calculation, this time from one that was done and still wrong.

### Confounds on the primary contrast

- **Repo mix differs between the contrasted strata**: `hop1` is 10 `kubernetes`
  / 20 `rust-lang`, `hop3plus` is 16 / 14 — so **the primary contrast is partly
  a repo contrast.** A disclosure, not an adjustment — no repo-adjusted analysis
  was part of the design, and running one after seeing the result is exactly the
  subgroup-hunting H2 forbids. The outcome split by repo was **not computed**, by
  either the analyst or the reviewer: untested, not null.
- **Hop count is confounded with gold-evidence-set size by construction** — a
  hop≥3 question necessarily has ≥3 gold documents. Inherent and unavoidable,
  but distinct from the repo confound.
- **`hop≥3` tilts to 3-hop chains**: 20 of 30 sit at exactly `hops = 3`, an
  artifact of the greedy pairwise-disjointness constraint in question selection,
  which disfavours larger gold sets — **a property of the selection machinery,
  not of the corpus.**
- **Questions are built from issue *titles*, not bodies** — an owner decision
  taken with measured gate counts under each option and no outcome measured (no
  gate, no index and no retrieval had run). Disclosed cost: titles likely share
  more vocabulary with resolutions than bodies do, **favouring grep**, i.e.
  biasing *against* the direction the design predicted, the conservative
  direction.
- **Chain heads must be issues**, which is why `fastapi/fastapi` contributed 0
  of 75 questions (appendix). **This is a three-repo corpus with a two-repo
  question set.**

### What is and is not reproducible, and what is and is not committed

**The thread set is reproducible; the comment set is not.** The fetch applies
`since = WS8_WINDOW_START` with **no upper bound** on the comment pages
(`scripts/non-code-retrieval/fetch_ws8_corpus.py:195-205`), so an in-window thread accumulates
comments created after the window closes and a later fetch returns more of them.
Consequently the three `data/ws8_corpus/raw/*.json` entries in
`data/MANIFEST.json` **can never be re-verified by a fresh fetch**, and
`scripts/fixtures/verify_checksums.py` reports them **absent** on a clean clone
(254 of the 470 manifest entries **as of 2026-09-11** were untracked-by-design,
so that part is the repo's normal pattern — but these three are additionally *unreproducible*, which the
others are not).

**Nothing of the corpus is committed, and the run transcripts were deleted to
keep it that way.** `data/ws8_corpus/` is gitignored and was never committed.
The transcripts *were* committed, and their `tool_result` blocks carried **~1.5M
characters of verbatim GitHub issue and PR text across 982 distinct corpus
documents** (12.8% of the corpus), including commenter usernames. That is the
same evidentiary trade-off `README.md`'s *"Second exception: run transcripts
embed source excerpts"* describes for the code-search benchmarks — **but not the
same licensing position**: those excerpted MIT and Apache-2.0 *code*, this
excerpted **user-generated comment prose, owned by the individual commenters and
not covered by `kubernetes/kubernetes`' or `rust-lang/rust`'s code licences.**
Disposition (owner decision): `results/ws8/transcripts/` and
`results/ws8/transcripts_parametric/` were removed — 225 files, 2.7 MB.

> ⚠️ **Deleting the files does not remove them from git history.** The blobs
> remain reachable in this branch's commits until the pre-public squash that
> this repo's release process already schedules. **That squash is load-bearing for the licensing
> position, not just for tidiness.** Verify before publishing.

### What the transcript removal costs, stated plainly

The deleted files covered all 150 paired runs and all 75 parametric gate runs,
and they are **not regenerable**: the checkpoints hold usage, cost, turn counts
and final answers, but no message history. Three things in this document become
**report-only** — true as far as the aggregate CSVs go, but no longer checkable
against raw evidence:

| claim | still verifiable from | no longer verifiable |
|---|---|---|
| the parametric arm **declined** rather than guessing wrong | the 0.000 score (`parametric_gate.csv`, `summary.csv`) and `any_doc_hit = 0/75` | the wording of the refusals, which was the actual evidence for C15 |
| payload per turn is the driver, at ~16–18K chars per search call | median tokens/turn, derivable from `runs.csv` | that the payload was in fact `top_k` chunks of fixed width, read off the calls |
| search-call count, not turn count, drives the cost gap | turns and tokens per arm (`summary.csv`, `paired_ci.csv`) | the per-call breakdown inside each run |

**This is a real reduction in auditability, and it is the honest cost of not
redistributing other people's writing.** The alternative considered and not
taken was redacting each `tool_result` payload down to the document ids it
returned plus a character count, which would have preserved the second and third
rows. Anyone rebuilding on this should assume those three claims rest on this
report rather than on inspectable data, and **re-run the paired phase if they
need them independently confirmed** — the harness is committed and
deterministic, and the cost is ~$48. **The aggregate results in
`results/ws8/*.csv` are unaffected.**

### Measurement residuals

- **Extended/adaptive thinking is ON by default** for `claude-sonnet-5`, and
  thinking tokens bill as output tokens. **The recall-vs-quality runs explicitly
  disable it; these do not.** It applies identically to both arms here, so it
  cannot bias the paired contrast — but it inflates the absolute output-token
  and dollar figures relative to `results/ws4/`, and it directly caused the
  `ws8_q065` truncation at `max_tokens` that had to be retried
  (`scripts/non-code-retrieval/gate_ws8.py:69-78`). **Do not compare these dollar figures to
  `results/ws4/`'s without accounting for it.**
- **Neither arm used prompt caching**: `cache_creation_input_tokens` and
  `cache_read_input_tokens` are **zero on all 150 rows**, so `prompt_tokens` is
  exactly `input_tokens` — no double-count, no 10%-discounted tokens weighted at
  par. `results/ws6c/runs.csv` is likewise 100% uncached, so the cross-benchmark
  dispersion comparison above is metric-compatible.
- The `build_question` postcondition assert **reuses `REFERENCE_RE` for both the
  redaction and its verification**, so it is tautological against a bug in the
  regex itself: it catches *"redaction skipped or misapplied"*, **not** *"the
  regex misses a reference form."* Weaker coverage than it looks.
- The inline-code-span regex can consume one backtick of a triple fence in
  adversarial single-field input. Present and untested; matters only if this
  ever needs a faithful markdown parser rather than a regex approximation.
- `first_sentence` strips quotes line-by-line, so a markdown lazy-continuation
  blockquote leaks its continuation line.
- GitHub's `since` parameter filters `updated_at`, not `created_at`; the corpus
  window is enforced **client-side**.
- **Inline PR *review* comments are not fetched** — only issue-level comments.
- **7 of 62,518 comments** showed ±1–3 count mismatches between the listing and
  the fetched thread (`comment_count_mismatches` in `corpus_stats.csv`).
- `requirements.lock` was frozen during this work and **backfilled a pre-existing
  repo gap**: `anthropic`, `mcp`, `papermill` and `tiktoken` had never been
  locked.
- **This changed shared code that three frozen benchmarks call.** The
  `system_prompt` / `max_tokens` overrides on `ws6a_runner.run_arm` are
  keyword-only and default to the frozen constants, so `results/ws6a/`,
  `results/ws6b/` and `results/ws6c/` are byte-identical — but two other
  `execute_run` changes apply to **every** caller: a `truncated_empty` skip and
  a spend-only `<arm>_judge_error` checkpoint line. **No published CSV moves**
  (those directories are untouched, and `run_ws8.py` filters `_judge_error` rows
  via `arm.isin(ARMS)`), but a *re-run* of a frozen benchmark would now produce
  a different checkpoint than the one on disk.

---

## Appendix — corpus, strata, spend

### Corpus

`results/ws8/corpus_stats.csv` · GitHub issue and PR threads from three
repositories, window **2026-06-01 → 2026-08-31** — after the model's training
cutoff, so the *text* is unseen.

| repo | documents | chars | PRs | issues | bot comments dropped |
|---|---|---|---|---|---|
| `fastapi/fastapi` | 475 | 1,280,026 | 453 | 22 | 520 |
| `kubernetes/kubernetes` | 2,312 | 10,672,546 | 1,776 | 536 | 6,682 |
| `rust-lang/rust` | 4,860 | 55,929,828 | 3,698 | 1,162 | 11,800 |
| **all** | **7,647** | **67,882,400** | 5,927 | 1,720 | **19,002** |

**31,704,356 tokens**, measured with the API's `count_tokens` endpoint, not
estimated. Fetch is pinned to `sort=created`, `direction=asc`, **≤8,000 threads
per repo** (`WS8_MAX_THREADS_PER_REPO`).

**Pull requests are kept, deliberately**: "issue → the PR that fixed it →
follow-up issue" is the canonical multi-hop chain this benchmark exists to
measure. Consequence to state with it: **77.5% of the documents are PRs** (5,927
of 7,647; 84% of the *characters*), so the corpus is mostly PR-discussion prose,
arguably a different genre from issue-report prose.

**Composition is lopsided, and the question set only partly matches it.** The
corpus is **82% `rust-lang` by volume** (55.9M of 67.9M chars); the question set
leans the same way but less steeply, **30 `kubernetes` / 45 `rust-lang`**, i.e.
60% `rust-lang` (`questions_by_repo.csv`). `rust-lang` is the majority on
**both** sides, so the mismatch is real but mild — the question set
**under**-represents `rust-lang` relative to its share of the text.
**`fastapi/fastapi` contributed 0 of 75 questions**: 453 of its 475 threads are
PRs, leaving only 22 issues, and a chain head must be an issue. Its documents
remain in the searchable corpus for both arms.

### Secondaries — labelled secondary in the design, not claimable alone

The design lists per-cell medians, turns, cost and latency as **secondary and
not claimable alone**, and `branches.csv` marks every one of them in its
`branch` column as a secondary with no branch of its own.

| stratum | indexed total / mean / max | agentic total / mean / max | paired Δ median | 95% CI | excl. 0 |
|---|---|---|---|---|---|
| `hop1` | $16.51 / $0.550 / $2.82 | $3.26 / $0.109 / $0.44 | +$0.1897 | [+0.027, +0.387] | Yes |
| `hop2` *(not powered)* | $12.93 / $0.862 / $3.62 | $1.79 / $0.120 / $0.80 | +$0.3492 | [+0.021, +1.085] | Yes |
| `hop3plus` | $11.46 / $0.382 / $3.54 | $2.80 / $0.094 / $0.33 | +$0.0766 | [+0.005, +0.155] | Yes (sign p 0.099) |

Parametric: **$0.81** total across all 75.

**Accuracy — recorded, gated on, never claimed.** The design forbids claiming
accuracy as a finding; it is reported for context.

| stratum | indexed | agentic | parametric |
|---|---|---|---|
| `hop1` | 0.500 (30 judged) | 0.600 | **0.000** |
| `hop2` *(not powered)* | 0.200 (15) | 0.533 | **0.000** |
| `hop3plus` | 0.586 (29 judged, **1 censored**) | 0.667 | **0.000** |

Unlike the code-search benchmark (1.000 in all six scaling cells) and the
agent-memory one (62 of 62), **accuracy did not saturate**. This corpus is
genuinely hard — a harder probe set exists, and this is what one looks like.

### `ws8_q067` — the censored row

`ws8_q067`, **indexed**, `hop3plus`: 15 of 15 turns, `stop_reason = tool_use`,
**empty answer**, **$3.5387** — the second-largest single cost in the paired run
and the **only** empty answer in it. Disposition ruled at 126/150 pairs, before
any delta, contrast or branch existed: **cost kept and counted** (real, billed,
and it measures the indexed arm exhausting the shared turn budget with nothing
to show — dropping it would bias the cost comparison *toward* the index by
discarding its worst case); **accuracy censored, not scored incorrect** (the
judge short-circuits an empty answer to `False` without evaluating content, so
scoring it wrong would assert a measurement never taken); **not retried** (the
15-turn cap applies identically to both arms so they differ in one variable;
re-running this row alone would make it differ in two).

**A secondary does flip under this sensitivity.** The `hop3plus` *dollar* delta
separates with 30 pairs (+$0.0766, CI [+0.005, +0.155]) and stops separating at
29 (+$0.0748, CI [−0.0004, +0.138]). It was already marginal (sign p 0.099). The
all-pairs row is the prescribed reading, but **a separation resting on one
censored observation is worth naming out loud.** The primary is unaffected.

Two rows reached 15 turns; only one was cut off by the cap. `summary.csv`
reports `n_hit_turn_cap` (the measured flag) and `n_turns_at_cap` (arithmetic)
**separately**, because they disagree: `ws8_q035` reached 15 turns and still
terminated at `end_turn`.

### Spend

**$50.854398 of a $70 governor**, from
`ws8.total_spent_across_checkpoints(DATA_DIR)`, which sweeps every checkpoint,
live and archived. It decomposes **exactly**:

| checkpoint | billed |
|---|---|
| `run_checkpoint.jsonl` — the paired run, 150 runs | $48.758008 |
| `gate_checkpoint.jsonl` — the gate, valid | $0.851598 |
| `gate_checkpoint.voided-2026-09-11.jsonl` — the superseded gate, archived not deleted | $0.783336 |
| `index_checkpoint.jsonl` — embeddings, 23,072,705 tokens → 45,888 chunks | $0.461456 |
| **total** | **$50.854398** |

The governor was raised **$55 → $70 before the paired run**, not during it.

**Reconciliation gap 1: `spend_ledger.csv` sums to $50.603840 — $0.250558 short
of the governor.** The cause is **a missing ledger row, not a mis-count**: an
earlier invocation of `run_ws8.py` checkpointed two runs summing to exactly
$0.250558 and **died before reaching `append_spend_ledger`**, and the
successor's `spent_before` seed already included them, so its `governor.spent −
spent_before` legitimately excluded them. **The governor was never wrong; the
ledger is missing a row for an invocation that never wrote one.** That matters
for the fix: comparing `governor.spent` against the checkpoint sum *within* an
invocation would **not** catch it — inside the surviving invocation those
figures agree perfectly. What reveals it is the ledger **sum** against the
checkpoint **sweep**. The remedy now implemented
writes the ledger row from a `finally`, plus a phase-end reconciliation of
exactly those two totals (`scripts/non-code-retrieval/run_ws8.py`, `benchlib/non_code_retrieval/bench.py`). ⚠️ **That
reconciliation fires on the committed data right now** — it reports the
$0.250558 delta on every future invocation and will keep doing so until someone
reconstructs the missing row by hand. Correct behaviour, not a bug. **The ledger
numbers have deliberately not been patched**: they are the invocations' own
contemporaneous self-reports, and rewriting them would destroy the evidence of
the defect. **It fails safe** — the governor seeds by sweeping every checkpoint,
so budget enforcement always used the full, correct figure, and **no dollar is
missing from the measurement**: `runs.csv` sums to $48.758008, exactly
`run_checkpoint.jsonl`.


**Reconciliation gap 2: `runs_parametric.csv` ($0.810326) is $0.041272 below
`gate_checkpoint.jsonl` ($0.851598).** That is the superseded truncated
`ws8_q065` line — rightly **excluded** from the published CSV (its retry
replaces it) and rightly **counted** by the governor (it was really billed).

**An operator error that cost real spend record.** Earlier on, a checkpoint was
**deleted**, destroying $0.78 of spend record — the exact `results/ws6c/`
failure the tooling exists to prevent. It was caused by an **operator
instruction, not a code bug**, and the amount was reconstructed from the
committed `parametric_gate.csv`. The standing rule is now **archive, never
delete**, and the governor sweeps archived checkpoints so voided runs still
count.

### Reproducing

```bash
make setup
python scripts/non-code-retrieval/fetch_ws8_corpus.py     # corpus is fetched, not redistributed, $0
python scripts/non-code-retrieval/build_ws8_corpus.py
python scripts/non-code-retrieval/build_ws8_questions.py
python scripts/non-code-retrieval/gate_ws8.py             # parametric gate, ~$1.6
make up && python scripts/non-code-retrieval/index_ws8.py # index build, ~$0.46
python scripts/non-code-retrieval/run_ws8.py              # paired runs, ~$48
python scripts/non-code-retrieval/summarise_ws8.py        # analysis, $0 -- deterministic, seed 42
python scripts/fixtures/verify_checksums.py
```

`summarise_ws8.py` alone reproduces every number in this document from the
committed `results/ws8/runs.csv` at **$0**, byte-identically.
