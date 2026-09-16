# Recall vs end-task answer quality

Covers `results/ws4/`, `results/ws4b/`, `results/ws4c/`, `results/ws4d/` and
`results/ws4e/`. Claim C6.

**Source of truth.** `results/ws4/` — the original run: `runs.csv` (6,000 rows,
500 questions × 12 arms), `summary.csv`, `pairs.csv`, `residuals.csv`,
`recall_on_ws4_questions.csv`, `summary_{sanity,strata,leak_adjusted}.csv`,
`judge_reliability.csv`, each written by exactly one cell of
`notebooks/04_recall_vs_quality.ipynb`, plus `rstar_sensitivity.csv`,
`arm_tests_corrected.csv` and `mechanism_pairs_all.csv`, written after the fact
by `scripts/recall-quality/ws4_rstar_sensitivity.py` from the frozen `runs.csv`.
`results/ws4b/` — re-scoring on corrected metrics and the audits (`tier0_*`,
`tier0b_*`, `tier0d_*`, `decline_*`). `results/ws4c/` — the coverage gate and
the generated-question build. `results/ws4d/` — blind eligibility labelling,
the cleaned curve, the denser ladder. `results/ws4e/` — the depth sweep.
Charts: `results/ws4/chart_quality_vs_recall.png`,
`results/ws4/chart_mediator.png`, `results/ws4e/chart_mediator_by_depth.png`.

**How to read this.** "The question, and the answer" states the finding; "Why
the obvious reading is wrong" shows why the two natural readings of a flat curve
— that recall stops mattering above some point, and that shallow retrieval
discriminates better — are both wrong. Those two sections carry the result, and a
reader who wants only the conclusion can stop at the end of the second. "What was
measured" is the instrument and every gate it passed or failed; "Limits, caveats,
and what we got wrong" is the audit trail; the appendices hold the per-arm tables
and the two audit rubrics near-verbatim.

---

## The question, and the answer

The slide promised a threshold: *"where recall stops predicting end-task answer
quality"* — recall is a proxy, and here is where it breaks.

**There is no such threshold to find on this task, and the C6 headline is
retired.** Not unresolved pending more data: retired. Four independent attempts
at the number — the four rows below — across three of the five result
directories (`ws4d` contributes two of the four) and up to fifteen retrieval
arms, never produced an interval narrow enough to quote, and the last two
measured *why* rather than assuming it. `results/ws4e/rstar_final.csv` is the
record:

| stage | arms | questions | CI | CI width | status |
|---|---:|---:|---|---:|---|
| plateau rule, 450-question set | 11 | 450 | [0.8096, 0.9724]¹ | 0.1629¹ | **WITHDRAWN** |
| changepoint fit, 11 arms | 11 | 264 | [0.7569, 0.9699] | 0.2130 | **SUPERSEDED** |
| changepoint fit, densest ladder | 15 | 264 | [0.7573, 0.9698] | 0.2126 | **UNRESOLVED** |
| 4-arm depth sweep | 4 | 264 | — | — | **NOT_FITTED** |
| **verdict** | | | | **0.2126** | **RETIRED** |

¹ The plateau rule is a rule, not a fit: it has no interval of its own, and it
can only return the measured recall of *some arm*, so its resolution is bounded
by arm spacing. Shown is `rstar_sensitivity.csv`'s `median_fixed_width` row —
the 95 % spread of the point estimate over 10,000 paired resamples, the closest
thing to an interval this estimator has, not an interval. **The rule's published
point estimate is deliberately not reproduced anywhere in this document,
including as a historical aside.** Appendix A does print the measured recall of
the arm the rule landed on, because that is the ladder's own x-axis; read it as a
grid coordinate bounded by arm spacing, never as a threshold estimate. Re-applying the whole rule inside the same paired bootstrap
reproduces its own branch in **0.6749** of resamples; **0.2787** find no plateau
at all, in which case the threshold does not exist under the rule's own
definition; only **0.1904** land within ±0.01 of the published point. Three
failure rates out of one estimator is not "noisy but usable".

The bar for quoting a threshold (`N1_MAX_CI_WIDTH = 0.05`) was set once and
never met. The narrowest width achieved, on the densest ladder buildable, is
**0.2126** — **4.25×** too wide.

**What replaces it is a slope, and the slope is quotable.** The 15-arm refit
(`results/ws4d/solvable_changepoint.csv`, `all_eligible` row) is a
two-parameter fit with real residual degrees of freedom: judge accuracy rises
**0.235 [0.122, 0.353]** points per unit of recall@10. There is a line, not a
knee. Recall never *stops* predicting quality over the tested range; it predicts
it weakly and monotonically throughout — **as a trend, not arm by arm**: below
the plateau edge the ordering is *not* monotone in recall (`pq_m256` at 0.817 is
indistinguishable from the reference while `sq8@16` at 0.918 is not), which is
the first hint that *how* recall was lost matters, and the reason the mechanism
claim ends up scoped rather than general. On the original 450-question run the
same shape appears as a 24-point fall in recall@10 (0.99 → 0.76) costing
**6.2 points** of judge accuracy, with Spearman(recall, judge) across arms
**0.706 [0.032, 0.845]**.

Two things make that line shallow, and neither is about retrieval:

1. **The benchmark is diluted.** Only **81 of 264 questions (30.7 %)**
   discriminate between arms at all.
2. **The generator, not the retriever, is where answers are lost** — but that
   effect shrank by roughly a factor of three once the scoring was audited, and
   the audited numbers remain provisional.

The honest line for a stage: *recall is a shallow, honest, monotone proxy for
retrieval; we went looking for the cliff with a dense ladder and a cleaned
benchmark, and there isn't one.*

---

## Why the obvious reading is wrong — dilution, then composition

### 1. The curve is not flat. It is diluted.

A flat curve invites "above some recall, retrieval stops mattering". That is
wrong about the mechanism. Of the 264 eligible questions
(`results/ws4d/solvable_discrimination.csv`): **122 (46.2 %) correct on all 15
arms**, **61 (23.1 %) wrong on all 15**, **81 (30.7 %) discriminating.** Nearly
70 % of the stratum contributes denominator and noise, and nothing else.

Excluding the 61 never-correct questions changes nothing and *cannot*: it
removes zero correct and exactly one wrong answer from every arm, rescaling
accuracy uniformly by n/(n−k) — asserted rather than claimed, measured factor
**1.300493** on all 15 arms — and a positive rescale leaves the changepoint
unchanged.

| stratum | n | τ̂ | width | slope |
|---|---:|---:|---:|---|
| all eligible | 264 | 0.7894 | 0.2126 | 0.235 |
| solvable only | 203 | 0.7868 | 0.2059 | 0.336 |
| ⚠️ discriminating only *(diagnostic)* | 81 | 0.8101 | 0.2258 | 0.774 |

**On the 81 discriminating questions the curve *is* knee-shaped in C6's
direction** — b₁ = **+2.244** below τ, b₂ = **+0.252** above, two segments
beating one by **42.6 %** of SSE against 8.8 % on the full stratum — but the
interval is still 0.2258 wide, because isolating the signal collapses n to 81.

⚠️ **That row is doubly outcome-conditioned** — both tails defined by the arms'
own answers, unlike the eligibility labels, which never see an arm's answer — so
it is exploratory, never a primary, and **must never be quoted as a result.** It
is recorded to identify the mechanism: **the signal exists, lives in under a
third of the questions, and disappears into noise the moment it is isolated.**
That is dilution, a property of the *benchmark*, not of retrieval.

⚠️ **30.7 % is an upper bound on arm-driven discrimination, not a measurement**:
it has no internal null control, since across 15 arms no question receives
identical passages everywhere. The depth sweep measures that floor on its own
four-arm set — within-prompt variance on a byte-identical prompt is
R = **0.0341 [0.0152, 0.0568]**, accounting for **2 of 76** discriminating
questions at k = 10 (`results/ws4e/noise_share_at_k10.csv`). ⚠️ A better-powered
**12 of 57** figure sits in the same CSV but was measured on a *different,
better-powered four-arm set*, and carries that caveat wherever quoted.

### 2. Depth does not change the rate. It changes the composition.

If 70 % of questions carry no retrieval information at top-10 then the obvious
move is to vary depth k and find the depth that discriminates best. The sweep
(k ∈ {1, 3, 5, 10}, four SQ8 arms, the same 264 questions) found no such depth,
and found something more useful on the way.

Its headline statistic `signal(k)` — the discriminating rate among separable
questions, minus a measured noise floor — falls **0.4073 → 0.2201** across
k = 1 → 10 (Appendix A), reading as "shallow retrieval discriminates better".

⚠️ **That fall is not a depth effect.** `separable(k)` is **nested**: if two
arms' top-k lists differ, their top-k′ lists differ for every k′ > k — verified
on the committed data (98 ⊂ 172 ⊂ 206 ⊂ 241) — so the denominator is a
population that **grows with the treatment**. Holding it fixed
(`results/ws4e/composition_check.csv`):

| k | fixed 98 always-separable | fixed 23 always-identical | fixed signal | newcomers vs k=1 | newcomer rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.4796 | 0.0870 | 0.3926 | 0 | — |
| 3 | 0.4592 | 0.0870 | 0.3722 | 74 | 0.1757 |
| 5 | 0.4694 | 0.0870 | 0.3824 | 108 | 0.1296 |
| 10 | **0.4796** | 0.0870 | 0.3926 | 143 | 0.1888 |

Max pairwise difference **0.0204**, CI **[−0.2223, +0.1815]** — includes zero,
nearly five times below the 0.10 non-flatness floor, ten times smaller than the
committed statistic's own max pairwise difference of 0.2022. This is Simpson's
paradox with the sweep's own headline estimator as the collapsing variable:
`separable_rate(k)` averages a flat ~0.48 and a flat ~0.16 whose weights shift
as newcomers enter. **On no fixed population does discrimination fall with
depth** — all 264: 0.2235 → 0.2879; the 241 separable at k = 10:
0.2365 → 0.3071; the 143 newcomers: 0.0699 → 0.1888; the fixed 98 and fixed 23:
flat.

**The finding: depth does not change the *rate* at which a fixed population
discriminates. It changes how many questions can discriminate at all.**
k = 1 → 10 lifts the share whose arms even see different passages from 0.37 to
0.91, adding 143 questions that discriminate at ~0.19 against the original 98's
~0.48.

⚠️ **A flat rate is not a fixed set of questions.** 47 of the fixed 98
discriminate at k = 1 and 47 at k = 10, but only **32 are the same questions**;
the fixed 23 has two discriminators at each end with **zero overlap**. A *rate*
claim, never a per-question one.

⚠️ Four things travel with it:

- **Only one pairwise contrast survives multiplicity.** The flatness test picks
  the largest of six pairs; corrected (99.17 % intervals), **k = 1 vs k = 5** is
  the only survivor (−0.2022, [−0.3739, −0.0333]). **k = 1 vs k = 10 does not
  survive**, so "signal falls across the tested range" is unsupported. Quote
  that one contrast or none.
- **Counting questions instead of rates reverses the conclusion.** Excess
  questions, `signal(k) × n_separable(k)`: 39.9 / 50.5 / 42.2 / **53.0** —
  **minimum at k = 1, maximum at k = 10**, an equally natural reading with the
  opposite answer.
- **A second committed table points the other way.** `slope_by_k.csv`'s
  one-basis slope (judge accuracy on recall@10, x-axis fixed across depths):
  0.1495 / 0.2802 / 0.3630 / **0.3751**, arm accuracy spread 0.064 at k = 1
  against 0.144 at k = 10 — **2.5× better separation at k = 10**. The two tables
  answer different questions and **neither may be quoted without the other.**
- **A quarter of the k = 1 signal runs backwards**: among discriminating
  separable questions the fraction where the *worst* arm is right and the *best*
  wrong is **0.2553** at k = 1 against 0.1081 at k = 10, and a sign-aware variant
  peaks at k = 5 — a third answer. The headline is not robust to a defensible
  re-specification.

⚠️ **A denominator that is a function of the treatment is a design flaw in this
run's own instrument**, found by adversarial review after the data existed. The
headline statistic is reported unchanged, but always beside the diagnostic that
undermines it. **Not a
one-off: the same conditioning trap fired twice in the same sweep.**

⚠️ **Two numbers are not robust to the deduplication rule**
(`dedup_sensitivity.csv`). Under first-wins rather than last-wins, `share(1)`
moves 0.2235 → 0.2462 and the raw-share sequence **stops being monotone** —
never say "raw share rises monotonically with depth" without this — while the
fixed-98 rate at k = 1 becomes **0.5000**, the max pairwise difference
**0.0843**, and the margin against the 0.10 floor shrinks from 4.9× to
**1.19×**. `signal(1)` is robust (0.4073 → 0.4036). The conclusion survives
either way; the rhetoric about the margin does not.

### 3. The same trap, again — the mediator

`mediator_by_k.csv` extends the single k = 10 mediator slice across every swept
depth. On the reference arm, k = 1 → 10, presence@k **rises** 0.5303 → 1.0000 —
a direct count, no conditioning. Conversion, P(judge correct | gold present),
**appears** to fall 0.7714 → 0.6742.

⚠️ **That fall is a composition artifact too**: presence@k is nested
(140 ⊂ 213 ⊂ 236 ⊂ 264), and fixing the presence set to the 140 questions
already present at k = 1 removes almost all of it — 0.7714 → 0.7571, paired
difference **−0.0143**, CI including zero — as for every arm
(`conversion_composition_check.csv`: fixed-set differences −0.0085, −0.0226,
**0.0000**, −0.0143, all CIs spanning zero, against moving-set falls
0.7458 → 0.6343, 0.7669 → 0.6429, 0.7714 → 0.6756, 0.7714 → 0.6742). The chart
draws both series so it cannot be misread.

The fixed-set view exposes something real: golds surfaced at deeper ranks
convert progressively worse. On the reference arm, questions already present at
the previous swept depth convert at 0.7500 / 0.6854 / 0.7119 at k = 3 / 5 / 10;
newly-present ones at 0.5342 / 0.5217 / **0.3571** (n = 73 / 23 / 28);
`sq8_np48` **0.3704** on 27 newly-present at k = 10, `sq8_np4` 0.4211 on 19.
**Depth changes which questions qualify, not how a given question behaves once
it qualifies** — the same lesson as the discriminating rate, confirmed twice on
two independently built estimators. "More depth is strictly better" needs a
qualifier, but not the obvious one: presence rises unconditionally, conversion
does not fall once composition is fixed, and what degrades is the *marginal*
gold's chance of being used. **Do not claim low k is a better operating point**
— excess-count, slope, accuracy spread and gold-presence gap all favour k = 10.
⚠️ Read all of this next to the decline caveat below: declines are scored
**incorrect**, so a large part of what moves across depths is the generator
refusing less often, not answering better.

### 4. And the mechanism behind the flatness is smaller than it first looked

The original explanation for the flat curve was the generator: on the reference
arm the gold was inside the ten passages for 368 of 450 questions, of which 214
were answered correctly, **106 (29 %) declined**, 48 (13 %) answered wrongly.
That dwarfs anything the retriever gives up between 0.76 and 0.99 recall, and it
is why the answer-level curve is flatter than the retrieval-level one:
answer-presence@10 moves 0.74 → 0.82 across the ladder and *is* classified DROP
for most degraded arms, while judge accuracy barely moves.

⚠️ **Both failure numbers were largely artifacts of how answer presence was
scored, and both are withdrawn.** Re-scoring on corrected metrics moved
conversion only 0.5815 → **0.6037** and the decline rate 0.2880 → **0.2848**;
the prediction behind that pass — that the decline rate was a denominator
artifact — **was wrong**, since the denominator *was* inflated, by 9.3 %, but
declines fell across answerable and spuriously-answerable questions in almost
the same proportion. Auditing what the passages said then withdrew the numbers
properly: only 23.4 % of the error bucket is genuine model error, putting the
generator's real error rate at about **2.7 %**, and **0.9 %** after a further
audit; and **54.8 % of declines were correct behaviour** — gold string present
with word boundaries, passage still not answering — leaving an unjustified
decline rate of **36/325 = 11.1 %**, not 28.6 %. So *"declines 29 %, errs
13 %"*, and the intermediate restatement to "28 % / 11 %", **are both withdrawn
entirely**; the mediator argument survives in direction and **shrinks by roughly
a factor of three in size.**

⚠️ **The audited replacements are provisional**: one auditor with a disclosed
conflict of interest, 93 rows, 20 spot-checked **non-blind** by one human at
18/20, and a *blind* check on the same auditor's other bucket that **failed** at
10/23. **Do not quote 11.1 % / 0.9 %, or any absolute accuracy from this eval,
without that qualifier. If it will not fit on the slide, the number does not go
on the slide.**

**The metric underneath cannot be repaired by better string matching.** Word
boundaries fixed `"12"`-inside-`"2012"` and left "a gold string in a passage
about something else" untouched — `"After World War I"` matching a passage about
London housing; `1997` matching Russia joining the G8; `2002` matching a
hand-checking rule change. The failure is semantic. ⚠️ But the global claim that
answer-presence "does not measure what it claims" **overreached**: against an
external annotation on 124 questions it agrees on ~80 % of rows, upper-bound
false-positive rate ~10 %, not ~40 %. The honest statement is decline-specific:
*gold-string presence over-counts context by up to ~10 % of rows overall, and
badly inside the decline bucket.*

---

## What was measured, and the gates each arm had to clear

**Setup.** 10M-vector slice of `mixedbread-ai/wikipedia-embed-en-2023-11` @
`395eebc`, 1024d, L2-normalised; ground truth `gt_10m_top100.npz` (exact fp32
inner-product top-100); questions NQ-Open **validation** (3,610), reserved
untouched; Milvus v2.6.18 standalone, `limit=10`, IP on normalised vectors;
generator `claude-sonnet-5`, thinking disabled, `max_tokens=256`, the committed
grounded prompt, byte-identical for every arm; judge `claude-opus-5`, blind to
arm and to passages, structured JSON, prompt committed before any generation;
bootstrap 10,000 resamples, one index draw shared across every arm and metric
(paired), percentile CIs, seed 42. Prompt caching was not used, and the reason
is stated rather than hidden: the only shared prefix is the ~150-token system
prompt, below Sonnet 5's 1,024-token cacheable minimum, and every passage set is
per-question — the notebook asserts `cache_read_input_tokens == 0` on all 6,000
rows.

**The question set.** 2,597 of 3,610 validation questions carry the gold string
somewhere in their exact top-100 — **filter rate 0.7194**, i.e. **28 % of
validation questions have no gold-answer string anywhere in their exact top-100**
and were removed *before* any arm ran. That is a 2018 benchmark against a
2023-11 snapshot. The filter does not make the top-10 answerable: the exact
top-10 carries the gold for 82.2 % of the drawn main set, so the reference arm
has real headroom to lose. Draw (seed 42, committed before any retrieval):
**450 main**, plus **50 sanity** drawn without the filter.

**The x-axis rule.** Every x is the arm's mean recall@10 over the 450 main
questions, not the sweep recall that chose its operating point. Low-nprobe
pruning scores **1.7–2.6 points higher** on validation questions than on the 10k
train-sampled sweep queries, while truncation and quantization arms move < 0.8
points; had sweep numbers been plotted, two planned mechanism pairs would have
been reported as matched when on the measured axis they are not.

### Gates

| gate | check | result |
|---|---|---|
| repro | SQ8 / refine / PQ / RaBitQ rebuilds reproduce the frozen 10M rows (10k queries) | 0.99166 vs 0.99166; Δ 0.00000 on all four |
| G1/G2 (truncation) | exact ceiling at 10M vs 1M (±0.05); Milvus recall vs own ceiling | +0.003 / +0.005 / **+0.050** (MRL — a genuine scale effect); −0.0035 / −0.0034 / −0.0006 |
| G4 | reference recall@10 on the 450 main questions ≥ 0.97 | 0.9920 |
| judge reliability | 60-row seeded sample re-judged, threshold 0.95 | **60 of 60 agree** |
| G0/G0b/G0c | re-derived passages reproduce the published presence bit under the old rule; corrected rule only removes matches; row count | 5,500/5,500; 0 gains; 6,000 ✓ |
| K0/K1/K2 | external-annotation join reproduces its inputs and the corrected metric | 124/124; 148 titles, 0 mismatches; 1,364/1,364 |
| E1–E4 | blindness by prompt reconstruction; coverage; κ ≥ 0.60; blind human spot-check n = 20 | **PASS** — 896 compared, 0 mismatched; 450/450; κ_binary **0.708875** (A called 249 eligible, B 209); **2** disagree |
| **F1** | is the filtered curve still an answer-quality curve? decline share < 0.80 | ⚠️ **FIRED** — 0.858398 (879 declines / 145 errors) |
| L1/L2 | max adjacent gap in [0.80, 0.97] ≤ 0.035; 60 rows re-judged reproduce | **PASS** — **0.031439**, 11 arms in band; **60/60** |
| K1–K6 | one basis per cell; cache ordering; judge drift; no depth at the noise floor; governor; replicate CI excludes zero | **PASS** — 264 qids × 4 arms; 0 of 1,350 mismatched; **60/60**; R = 0.0341, 9 disagreements |
| P1/P2 | always-correct shrinks and never-correct grows as k falls | **HOLD** — 123 → 116 → 111 → 104; 65 → 83 → 91 → 101 |
| **C1** | every supporting sentence locatable in exactly one chunk of its gold page | ⚠️ **FAILS at 0.074** against 0.90 |
| **G-A** | a generated multi-hop set reaching 600 questions | ⚠️ **FAILS at 2.01 %** — 5 accepted |
| **G1** (gold refresh) | concordance with the provisional `stale_gold` labels ≥ 0.60 | ⚠️ **FAILS at 0.435** — 10/23 |

⚠️ One depth-sweep gate passed **by construction and verified nothing** — its
first row used the same question-set object for every cell; only the later row,
reading each cell's question and arm sets back from its checkpoint, is evidence.

### The three failures that shaped the result

**C1 — a 2017 benchmark cannot be sentence-matched against a 2023-11 corpus.**
The slice *can* host a multi-hop set: 3,421 HotpotQA distractor-train questions
have both gold pages inside it, six times the 600 needed, top page 0.4 %. What
it cannot do is locate the supporting sentence: only **31 %** of supporting
sentences appear verbatim in any chunk of their own gold page, so **C1 fails at
0.074** against 0.90 (0.0558 strict — a deliberately narrow quote-normalisation
accounts for the difference and nothing else, since indiscriminate punctuation
stripping is what broke the presence metric). The content has not vanished — a
token-containment locator finds a unique chunk at median containment **0.9651** —
but reaches only **0.5618** of questions. Both remedies were blocked:
page-level provenance reintroduces the very error the metric existed to remove,
and the locator does not clear the gate. Why not simply use HotpotQA: the slice
holds ~23 % of English Wikipedia pages, so **~96 % of a 2-hop benchmark is
unusable against it.**

**G-A — the generated-question route yields at 2 %.** Building the set as entity
bridges between corpus chunks accepted **5 of 249 candidates (2.01 %)** against a
target of 600. The build stopped on host memory pressure, not budget — **92.8 %
of the $14.00 governor unspent** — and a projection ended it: at
$0.004045/candidate the governor buys ≈3,461.06 candidates, projecting **≈69
accepted questions** at the measured rate and **≈160** at the top of the Wilson
95 % interval for 5/249 ([0.8607 %, 4.6137 %]), against the **17.34 %** G-A
needs — **8.6x** short at the point estimate, **3.8x** short even at the Wilson
upper bound. The governor is not raised to reach the number. The 5 accepted
questions are **not** a partial success: they show the filter chain runs end to
end, nothing more.

⚠️ **Phases 1 and 2 never ran, so the floor gate G-D — the one that amendment
existed to add — was never reached. Whether single-shot dense retrieval can
reach a hidden second hop on this corpus is an open question, not a negative
result**, and must never be reported as "retrieval fails on multi-hop": it was
never measured.

**G1 — the gold-refresh pass stopped before re-scoring.** A second, blind,
independent pass over the 23 rows the error audit labelled `stale_gold` upheld a
change on only **10 — concordance 0.4348 against a 0.60 floor** — and per that
design's fail action the pass **stopped before re-scoring**: no re-judge, no
slope recomputation, no corrected accuracy, no bound, because the step that
would have produced one never ran. That is the design working: a rescore on top
of a labelling disagreement this large would have produced a number dressed as
evidence about gold quality that was really evidence about which pass to trust.
⚠️ **The two passes disagree and this design cannot say which is right** — the
labels may be too liberal, the refresher too conservative, or both — but **the
earlier labels are impeached**, and the drop-78 analysis below rests on labels a
second pass would have applied to fewer than half of those rows.

### What the re-scoring and the audits produced

**Corrected metrics** (spend $0.00 — no API calls, no Milvus, no re-retrieval).
458 of 4,950 arm-rows (**9.3 %**) lose answer presence under word-boundary
matching; **14 of 500** questions have no usable gold alias at all, every one a
bare number or punctuation — one (`["---"]`) normalises to the empty string, a
substring of every passage, so that question was scored "answerable"
unconditionally under every arm for the whole run. Decline detection required
exact equality with `"I don't know"`, so answers that decline and then explain
were scored as errors: 47 rows recovered.

| reference arm | n | conversion | decline rate | error rate |
|---|---:|---:|---:|---:|
| as published | 368 | 0.5815 | 0.2880 | 0.1304 |
| **corrected** | **323** | **0.6037** | **0.2848** | **0.1146** |

Across all eleven arms, corrected conversion 0.577–0.621, decline rate
0.276–0.306. Absolute quality rises on the primary stratum (reference arm
0.498 → 0.600) only because ~18 % of the denominator — questions whose gold sits
at ranks 11–100, unreachable by any top-10 retriever — is removed by
construction: **a cleaner measurement, not an improvement.** One genuine win:
the stratum raises the quality span from 0.0622 to **0.0892**, 43 % more dynamic
range against a 28 % loss of questions, for free.

**The error-bucket audit** (rubric at Appendix B; 107 candidates; auditor blind
to arm, recall, presence and accuracy; share not computed until every row was
labelled): `ambiguous_question` **49 (45.8 %)**, `model_error` 25 (23.4 %),
`stale_gold` **23 (21.5 %)**, `unresolved` 6 (5.6 %), `judge_strict` 4 (3.7 %).
**72/107 = 0.673** against a 0.40 threshold, so the error rate is withdrawn as a
generator statistic and reported as a gold-quality statistic. **The finding is
not the one the audit was proposed on:** staleness is the smaller half, and the
dominant category is `ambiguous_question` — questions not stale but
*underspecified* (no year, no adaptation, no sense, or two valid answers with
one gold) — which **no corpus snapshot, prompt or stronger generator fixes**,
because it is a property of NQ-Open as an end-task benchmark. ⚠️ Six
`unresolved` rows are all the one thing the rubric has no category for — **the
gold is factually wrong** (78 % for the water share of the earth's surface where
71 % is right; the butler named for a question about the under-butler; Einstein
for de Broglie's matter waves). Adding a category mid-audit would have been
fitting the instrument to the data, so they were labelled `unresolved`, counting
*against* the threshold; including them would raise the share to 72.9 %.
Recorded as a rubric defect for any redo.

**Does bad gold quality change the shape?** Dropping all 78 artifact questions
and re-running the classification: the corrected primary stratum (n = 325) gives
quality span 0.0892, slope 0.221, branch **K1**; minus the 78 (n = 271), span
0.0959, slope 0.246, branch **K3**. **The flatness survives; the plateau does
not** — span and slope barely move, so the shape is not an artefact of bad
golds, but the branch flips to no-plateau-at-all because at n = 271 the
intervals widen until the top degraded arm can no longer rule out a 5-point
loss. **Power loss, not a stronger effect**, and a second independent route to
the same conclusion about the threshold. ⚠️ Also qualified by the failed
concordance above: 23 of those 78 rows carry labels a blind second pass agreed
with on fewer than half.

**The decline-bucket audit** (rubric at Appendix C; 93 rows; evidence is the
question, the golds, and ±250-char windows around every word-boundary gold match
in the ten passages that arm actually retrieved): `not_supported` **51
(54.8 %)**, `answerable_declined` 36 (38.7 %), `ambiguous_question` 5 (5.4 %),
`stale_gold` 1 (1.1 %). **R = 36/93 = 0.387 → branch D2**: the 0.70 conversion
target is reachable only by converting essentially *every* recoverable decline.
Both ceilings land in the same place — (195+36)/325 = **0.711**, or dropping the
51 unsupported declines from the denominator, 195/274 = **0.712**. ⚠️ The second
is **biased upward and must not be quoted as a corrected conversion**: only the
93 declines were audited, so unsupported questions among the 195 correct and 38
error rows were never removed.

⚠️ **The human spot-check passed and did not promote these numbers.** On 20 rows
(a seeded sample, re-derived and asserted by a test rather than trusted), **18 of
20 agree with the auditor's label** against a >3-of-20 redo threshold, so no redo
is forced — which removes the forced-redo trigger and nothing else. **Status:
"spot-checked, one auditor, non-blind"**, and that is how these numbers must be
labelled anywhere they appear. The one real correction pushes *away* from the
gate: in-sample it gives R = 0.376 and a ceiling of 0.7077, and if the sample's
demotion rate held, **R ≈ 0.332 and the ceiling 0.695 — below the gate**, only
0.012 above the D3 cut at which the follow-up does not run at all; two demotions
instead of one would give R ≈ 0.277, inside D3. **D2 still holds on the evidence
that exists** — not a branch change, and must not be reported as one — but D2 is
no longer comfortably held.

**Blindness check:** 18 of the 93 decline questions had appeared in the error
sheet, so their questions and golds had been seen; the seen group labels as
**less** recoverable (0.222 vs 0.427), the opposite direction from the auditor's
interest — likely selection, not bias.

**External calibration.** Joining an external provenance annotation to the 124 of
450 questions that appear in it *and* whose gold page landed inside the pinned
slice: presence-true/provenance-false **118/1,364 = 0.0865** joint (0.1047
conditional), presence-false/provenance-true 0.1100, raw agreement **0.8035** —
narrowing the withdrawal above to the decline bucket. ⚠️ That false-positive
figure is **an upper bound, not a measurement**: it mixes "the string matched
something irrelevant" with "the annotation never covered the page the answer was
on", so **~8.7 % must not be reported as the error rate of answer-presence**.
⚠️ And **page-level provenance is looser than chunk-level presence**: the corpus
averages 6.55 chunks per page, on all 150 provenance-only rows the gold string
is absent from the retrieved gold-page chunk, and counting coincidental matches
the string matched off the gold page on **197/1,127 = 17.5 %** of presence-true
rows.

⚠️ **The one external probe of the decline audit is underpowered and leans
against it.** The overlap is **n = 18**, below the floor set for that test, so it
resolves nothing — but the one statistic that exists is **0.611 agreement against
a 0.75 threshold**, with the failure concentrated where it matters: **6 of the 7
questions the auditor labelled `not_supported` had the annotated gold page in the
top-10**, and on all 6 the gold string matched on that gold page. At full power
that would say the auditor read the evidence to suit the hypothesis it had
proposed. **It is not softened here and must not be softened downstream.** Nor is
it resolved: one row moves agreement by 5.6 points, and the page-vs-chunk
granularity confound cuts *toward* the auditor on exactly these rows.

### The cleaned curve, and the three levers

Two model labellers independently labelled all 450 questions from byte-identical
prompts, seeing only the question, its golds and the **arm-independent**
ground-truth top-10 — no arm's answer, recall, presence flag or judge verdict
reached a prompt, asserted structurally by rebuilding every rendered prompt and
comparing byte-for-byte. Classes: `eligible` **262 (58.2 %)**, `unanswerable` 56
(12.4 %), `ambiguous` 53 (11.8 %), `ineligible_class_disputed` 49 (10.9 %),
`stale_gold` 16 (3.6 %), `wrong_gold` 14 (3.1 %). ⚠️ **The labellers differ
systematically**: A called 249 questions eligible and B 209 — mostly via a
higher `ambiguous` rate (103 vs 65). The adjudication rule resolves a binary split to
**eligible** — the conservative direction, since it minimises the drop and
dropping is what inflates accuracy — so the rule biases *against* this document's
own hypothesis by design. Primary drop set (`ambiguous` +
`wrong_gold` + `stale_gold`) = **83 of 450 (18.4 %)**; 61 fall inside the
325-question primary stratum, giving **n = 264**. The human spot-check here **was
blind by construction** — the deliberate correction of the decline spot-check's
weakness: the sheet carried question, golds and passages and **none** of the
model columns, with n, seed, threshold and blinding fixed before any label
existed. 18 of 20 agree.

| stratum | n | τ̂ | 95 % CI | width | slope |
|---|---:|---:|---|---:|---|
| unfiltered 325 (bridge) | 325 | 0.7968 | [0.7648, 0.9742] | **0.2094** | 0.221 [0.112, 0.335] |
| **primary** | **264** | **0.7894** | [0.7569, 0.9699] | **0.2130** | 0.212 [0.097, 0.332] |
| plus unanswerable | 246 | 0.7908 | [0.7612, 0.9712] | 0.2099 | 0.238 [0.111, 0.373] |
| disputed included | 241 | 0.7860 | [0.7560, 0.9743] | 0.2183 | 0.211 [0.091, 0.333] |
| secondary A | 136 | 0.8069 | [0.7842, 0.9695] | 0.1854 | 0.094 [−0.013, 0.217] |

**Cleaning the questions made the interval wider, not narrower.** All three
levers are individually measured and individually futile: **more questions** —
width ∝ **n^−0.065**, so 4× the questions buys 9 %, futile by arithmetic;
**cleaner questions** — 0.2094 → **0.2130**; **denser arms** — four new arms,
leave-one-arm-out spread 0.1754 → **0.0715**, width 0.2130 → **0.2126**. τ is
**arm-limited**: because the width scales as n^−0.065, **more questions can never
produce a usable threshold**, and adding arms inside the measured recall range
does not move it either. ⚠️ The honest conclusion is not "we lack resolution":
the ladder's fragility genuinely improved — no single arm can swing the fit by
0.175 any more — and the answer did not change.

⚠️ **Two claims must not be made from that table.** The 136-question row's
single-slope interval includes zero, but n = 136 is 42 % of the stratum and its
b₁ of +2.05 on 11 points is a wild fit — not distinguishable from low power, and
**not independent evidence that recall does not matter.** And an earlier scoping
pass, using the *error-bucket* labels as a stand-in, reported b₁ = +0.497,
knee-shaped; with the real blind labels the primary stratum gives
**b₁ = −0.5999, b₂ = +0.2672**, the backwards shape, so the scoping result **did
not replicate**, exactly as its own warning said it might.

⚠️ **Gate F1 fired.** 879 of the 1,024 remaining wrong rows are **declines**, 145
errors — decline share **0.8584** against 0.80. The filtering removes *errors*
well and does nothing to *declines*, and more than half of all declines are
correct behaviour, so the filtered curve plots substantially **when the generator
refuses to answer**, not answer quality. It is labelled a decline-behaviour
curve, and C6's primary remains the 325-question curve.

---

## Limits, caveats, and what we got wrong

**Measurement**

- **One generation per row.** Sonnet 5 accepts no temperature parameter (HTTP 400
  for `temperature`, `top_p`, `top_k`), so every CI covers question sampling
  only, not generator nondeterminism. A temperature-0 validation cell was planned
  and is **impossible**; a replicate cell measures the same floor differently,
  and *"why not just set temperature to 0?"* is now unanswerable by anyone.
- **The same 450 questions serve every arm and every test** — what makes the
  comparisons paired also means one draw. The 50-question unfiltered sanity
  sample is the only independent check: judge accuracy 0.34–0.38 for every
  retrieval arm (CIs ±0.14) against 0.14 parametric, ordering flat within noise
  (Spearman 0.54 on a 4-point spread with 14-point CIs). Nothing in it
  contradicts the main set; **it also cannot confirm the ordering, and is not
  claimed to.**
- **Grounded prompt** — the steelman for recall mattering; the lenient deployment
  prompt was not run and would flatten the curve further. The parametric arm
  measures the leak at 0.216, and re-running the classification on the 353
  questions it got wrong changes nothing.
- **NQ golds are 2018-era, the corpus 2023-11**, so absolute accuracies are lower
  bounds while paired differences are unaffected. **The talk should probably not
  use NQ-Open absolute accuracy at all** — 67 % of the error bucket is benchmark
  artifact, and no absolute number here means what an audience will assume.
- **EM and F1 undercount** multi-entity and reformatted answers (110 of the
  reference arm's judge-correct rows have EM = 0), while the judge never marks an
  EM hit wrong. Both published, judge primary, disagreements shown.
- **Ten arms were classified against the reference with no multiplicity
  control.** Under Holm only `sq8@4` separates (McNemar p 0.00013, Holm
  **0.0013**); the two arms setting the plateau edge sit at p ≈ 0.043 and 0.047
  uncorrected — the yield of ten tests at α = 0.05 under the null.
- **The equivalence margin is as wide as the data**: PLATEAU rules out a 5-point
  loss, while the whole spread of judge accuracy is 6.2 points (2.9 without
  `sq8@4`) against a median CI half-width of 2.7. Read PLATEAU as *not a
  catastrophe*, not as *no effect*.
- **The threshold was pinned by one arm's classification.** Flip distance — how
  many discordant questions must become ties before an arm leaves DROP — is **2**
  for `sq8@16`, 1 for `mrl_512`, **16** for `sq8@4`: two judge verdicts move the
  headline by 10.6 recall points, and a real effect needs sixteen.
- **The bootstrap covers question sampling only**, so judge nondeterminism is in
  no interval — which matters for a verdict that turned on two questions.
- **Latency and cost per query are recorded but are not results here.**

**Mechanism pairs**

- **Two planned pairs were unmatched** once the x-axis rule moved the pruning
  arms, and **the consequence was missed at the time**: that lost the design's
  only coverage below 0.87 recall. Applying the matching rule *exhaustively*
  rather than to a hand-picked list finds exactly six matched cross-family pairs,
  and **the two never named are the only two with an effect** — `pq_m256` vs
  `sq8@4`, at 0.817/0.801, gap 0.016, is **+5.8 points**, its CI excludes zero,
  it partly exceeds the ±0.05 band the no-effect reading asserts, and it survives
  Bonferroni over all six (Appendix A).
- **So the mechanism claim is scoped, not withdrawn.** Supported: *at matched
  recall between 0.92 and 0.97, mechanism does not measurably change answer
  quality.* Not supported: the same at 0.80. **How you lose recall does matter;
  it stops mattering only once you have enough of it.**

**The audits**

- **Both audits were applied by the same model that proposed the hypothesis they
  test** — a real conflict of interest, disclosed rather than managed away;
  mitigations in force are in Appendices B and C.
- **The decline audit is "spot-checked, one auditor, non-blind" and stays
  provisional**: 20 of 93 rows re-read (95 % CI on 18/20 roughly **0.70–0.97**),
  one reviewer, and the sheet carried the auditor's label and justification —
  confirmatory review, systematically more agreeable than blind re-derivation.
  **18/20 is therefore not evidence against the blind 10/23 concordance failure
  on the same auditor's other bucket, and must not be quoted as though the
  auditor has been vindicated.** ⚠️ **The error-bucket audit has still not been
  spot-checked at all.**
- ⚠️ **Of the 20 spot-checked rows**: one is a real label error; one is a rubric
  precedence call that moves nothing numerically but shows the rubric applied
  inconsistently on five rows; and one is a **right label reached by a wrong
  reading**, counted as agreement because the threshold counts label
  disagreements — **a distinct defect the threshold cannot see**, and the reason
  the illustrative decline examples above are illustrative rather than audited
  evidence.
- **The gold-refresh pass produced no corrected accuracy, slope or bound**, and
  none exists to quote: of 107 rows, 30 keep, 9 augment, 36 replace, 32
  unanswerable from corpus; of 45 proposed changes a blind verifier **upheld 26
  and rejected 19** (0.578), so gold changed on 26 rows. **Do not present this as
  a partial success** — its most important output is that its precondition
  failed, and it never reached the step that would say whether it rescued or
  worsened anything.
- **Row selection there was one-sided**: the 107 rows are exactly the rows where
  some arm erred, so any accuracy figure could only ever have been an upper
  bound. Only 72 of the 107 are in the primary stratum, so the other 35 could
  never have moved the slope — ⚠️ numerically identical to, but causally
  unrelated to, the 72/107 audit-label share above: this one counts stratum
  membership, that one counts labels, and the coincidence is not evidence of
  anything; and a `keep` verdict is not evidence a gold is
  right, only that two passes found no corpus evidence in the top-10 to change
  it.
- ⚠️ **A blindness gate is reported as failed, and the failure is a gate defect,
  not a leak.** Its structural check passed perfectly — **0 mismatches across all
  152 rendered prompts**, each rebuilt from only the allowed inputs and compared
  byte-for-byte — and that is the quotable result. The FAIL is entirely a
  secondary forbidden-term scan with **5 hits**, every one a row's own recorded
  gold colliding with a *different* row's audited wrong answer, because the term
  list was built globally rather than per row. **A term-presence scan cannot work
  here at all**: of the audited rows' pooled wrong answers, 105 are testable and
  **51 (48.6 %) appear verbatim in that row's own top-10 passage text** — what a
  retrieval-grounded wrong answer looks like. The gate was deliberately **not**
  amended a second time after its failure was known, so it stands as failed, and
  **its FAIL is not quotable as "the run leaked."** ⚠️ An earlier internal read
  of it was wrong and is corrected rather than silently fixed: the fifth hit was
  described as a question containing that row's own audited answer, but the scan
  is case-sensitive and that row has the term lowercased, so the fifth hit is a
  second scan of the same colliding gold field.
- ⚠️ **A licensing defect.** Free-text audit columns quoted corpus passage text
  verbatim — 129 fields across 94 of 107 rows, longest span 170 characters. Fixed
  structurally, not by a one-off scrub: a redaction pass strips any run of ≥ 20
  characters shared with that row's own passages, enforced by a test; no gate
  result, verdict or count changed. ⚠️ The guard was case-sensitive and a
  re-typed lower-case quote defeated it, so it gained a case-insensitive option —
  under which 24 of 93 justification fields in an earlier committed audit file
  flag, the longest 57 characters.

**The depth sweep**

- ⚠️ **Much of what moves across depths is decline behaviour, not answer
  quality.** Declines are scored **incorrect**, and the decline rate falls
  steadily with depth: on `sq8_np512`, **0.383 at k = 1 → 0.273 at k = 10**
  (`results/ws4e/depth_points.csv`), against an accuracy rise of 0.538 → 0.674
  over the same span. Gate F1 fired on this stratum for exactly this reason —
  declines are **0.8584** of the remaining wrong rows against a 0.80 bar — and
  more than half of all audited declines were correct behaviour. So the depth
  curve is substantially a curve in *when the generator refuses to answer*.
  Anywhere this document says accuracy rises with k, that clause applies.
- **Depths above 10 are out of scope by construction** — the stratum is "gold in
  the ground-truth top-10", so gold presence is 1.0000 at every k ≥ 10; nothing
  here speaks to k = 20, 50 or 100.
- ⚠️ **The k = 10 cell mixes generation vintages** (two arms from the original
  run, two later) and the judge-drift gate tests **judge** reproduction, not
  **generator** drift — while k = 10 is the endpoint of both monotonicity
  predictions and of the raw-share claim.
- **`null(10)` rests on 23 questions** and `signal(10)` inherits it (widest CI,
  0.261) — not load-bearing, since dropping k = 10 leaves the surviving contrast
  unchanged. The fixed-composition diagnostic now carrying the finding has the
  same exposure: n = 98 and n = 23, with **the 23-question term setting the CI's
  width**, and that CI is a **non-rejection of a flat curve, not a demonstration
  that the curve is exactly flat** — absence of evidence for a depth effect, not
  evidence of its absence.
- **Four arms is enough for a decomposition, not a changepoint**: four points
  against a four-parameter model is zero residual degrees of freedom, so **none
  was fitted** (`slope_by_k.csv` records `residual_df = 2` for the two-parameter
  slope actually fitted); reporting one would have been numerology. Two
  withdrawal branches were never exercised on real data and are unit-tested
  against synthetic fixtures only.
- **The replicate validates at one depth on one arm**, so transfer is assumed,
  not tested, and it passes on **4 and 5 disagreements** — read it at that power.
  The subsets' point estimates differ by 35 % (0.0408 separable vs 0.0301
  identical-passage) in the direction that matters: if the true
  separable-question floor is the higher, the subtracted floor
  **under-estimates** it and biases `signal(k)` **upward** at every depth, which
  a passing overlap test does not rule out. One consistency check agrees
  (R = 0.0341 implies 0.0676 against an observed 0.0723); the better-powered but
  non-independent one **over-predicts by 47 %** (0.1066 against the same 0.0723).
  Report both.
- ⚠️ **Two operational defects in that run, reported not dropped.** **555 of 4,251
  raw rows are duplicates (~$3.05)**: the harness can leave a background
  command's Python child orphaned, so a resume without checking produced two
  concurrent writers (549 rows) and a stale orphaned process later started a competing
  run (6 rows). Unique coverage is nevertheless **exactly 3,696 of 3,696**, and
  the analysis is unaffected for two independent reasons — the loader dedups by
  `(qid, arm)`, and the governor sums every *raw* line, so it saw the real money.
  The second defect is the dedup-rule sensitivity above. The supplementary
  replicate row **came from this defect, not a design**: never gated on, never
  touching a branch, **not independent** of the primary cell, and R = 0.0541 on
  the 555 accidental pairs is **1.6×** the primary cell's 0.0341. For 139 of 264
  questions the primary replicate row is itself one of a duplicated pair, so R is
  computed on an arbitrary pairing.
- ⚠️ **Reproducing these numbers is not free**: the analysis reads three
  gitignored artifacts (retrieval caches, per-cell checkpoints, and the earlier
  phase's run checkpoint — the only source for two arms' committed k = 10 rows),
  so a third party must re-spend roughly the **$24.68** this pass spent *and*
  rebuild the underlying 10M SQ8 index. Reproducible in principle, not for free.

**The generated-question build**

- **Its own filters are not clean.** F3 dropped only 1.61 % of candidates seen —
  low for heavily-memorised Wikipedia, though n is too small to call it broken.
  F4 dropped **32 of the 39 candidates that reached it (82 %)**, the dominant
  contributor to the 2.01 % yield, and a properly powered canary of 29
  single-hop-by-construction questions shows it **under-drops**: 21/29 =
  **0.7241** against 0.80, FAIL. Two mechanisms, not one: 6 of the 8 it missed
  are long or list-shaped spans where exact-match scoring would reject a correct
  restatement, but **2 — a clean proper noun and a clean date — are spans
  exact-match should have caught**, so the single-passage answerer sometimes
  fails to extract an available, cleanly-stated fact. ⚠️ **F4 also over-rejects
  genuine bridge questions** for reasons unrelated to scoring brittleness, and
  **the two mechanisms push in opposite directions with the net effect
  unmeasured** — the build recorded only that a drop occurred, not which arm
  fired. **We do not claim the true 2-hop rate is bounded above or below 2.01 %
  by this argument.**
- **Why the yield is 2 %, from reading all 195 decline notes** (not a sample):
  108 (55.4 %) an incidental mention rather than a genuine reference, 72 (36.9 %)
  a chunk B holding no connectable fact, 15 of 195 a chunk A naming the bridge
  too directly to hide it — so category A **clearly dominates** B by 18.5 points,
  correcting an earlier rounded pass that had them at ~51 % vs ~42 % and read as
  "narrowly dominates". ⚠️ **A weaker evidential basis than the yield table**: an
  LLM reading free-text notes against three definitions, source notes gitignored
  and not committed, and a different classifier could draw the A/B boundary
  differently. The earlier version printed rounded percentages that back-computed
  to integers summing to 194, one short of 195 — then re-divided them to false
  four-decimal precision; superseded by the exact tally.
- ⚠️ **One reported gate failure is not evaluable and must not be read at face
  value**: the page-concentration check on the accepted set reports FAIL at 0.200
  against a 0.05 ceiling, but with 5 accepted questions any single page supplying
  one is 20 % by arithmetic. The cap that matters is enforced on the candidate
  pool, where across all 9,000 pairs the most-concentrated title supplies
  **exactly 12 pairs, a 0.1333 % share** — identical to the cap's theoretical
  ceiling, so binding and saturated, not merely satisfied by chance.
- **One genuine success:** provenance is **chunk-level by construction**, 5/5
  accepted questions with 0 bad — known before the question exists, which is what
  C1 spent 3,421 questions failing to recover after the fact.
- **Selection effects recorded rather than smoothed over:** ASCII-only title
  normalisation means a wholly non-Latin-script title can never be a bridge; F3
  selects on the generator's ignorance; and the same model writes and answers the
  questions, so **absolute accuracy would not have been quotable even had the
  build succeeded** (common-mode across arms, so a relative measurement would
  have been unaffected).
- ⚠️ **Any comparison against the single-hop run would span two axes** — task
  (single-hop → multi-hop) *and* question provenance (human-written →
  machine-written) — and the separating control costs ~$60 and was not run, so
  **"multi-hop is more retrieval-bound" must never be stated bare.** The
  single-hop slope for any such comparison is **b = 0.2209, 95 % CI [0.1121,
  0.3352]** (`results/ws4b/tier0_slope.csv`), spanning a factor of three, so both
  intervals are reported side by side or neither is. Never exercised, because
  there is no comparison to confound.

**Scope, and two things that nearly went wrong**

- **This is still NQ-Open** — single-hop factoid QA over a 10M-passage Wikipedia
  slice at top-10. Removing drifted and ambiguous items does not make the
  remainder representative of production RAG traffic.
- **Two labellers from one provider**: correlated failure modes surface as
  *agreement*, not disagreement, so the κ gate can pass on labels that are
  jointly wrong; the blind human check is the only guard, at n = 20.
- ⚠️ **The drop set is probably too small** — both human disagreements ran the
  same direction (the human was *stricter*) and the adjudication rule already
  resolves splits toward eligible, so the filtered stratum still carries
  defective items. The safe direction for our own claim, stated rather than
  fixed. The 0.80 decline-share threshold is likewise a judgement, pinned before
  the share was known.
- ⚠️ **The index the arms were served from had been dropped.** The denser-ladder
  pass was scoped "retrieval only, no index build"; the collection still held all
  10,000,000 rows but returned no index, so it began with a **2.20-hour rebuild**
  (7,903.1 s) plus a 74.5 s load of ~10.8 GiB, parameters read off the repo
  rather than guessed — `IVF_SQ8`, `metric_type="IP"`, `nlist=4096`, on
  L2-normalised vectors. ⚠️ `COSINE` would have been wrong and would **not** have
  errored: on normalised vectors it returns plausible results, so the four new
  arms would have looked fine and been silently incomparable to the existing
  eleven.
- ⚠️ **A ladder-density gate was first reported as failing at 0.0416, and that
  was an error**, corrected here rather than quietly: it compared the new arms'
  recall over all 3,610 sweep queries against the existing arms' recall on the
  325-question stratum. On the basis the curve is actually fitted on it passes at
  0.031439; no committed artifact was written from the wrong figure, and
  `ladder_gates.csv` now records the basis explicitly.
- **Cosmetic, recorded rather than re-run:** ~5 label-reason rows carry a literal
  `—` escape instead of an em dash — a model output quirk, not a parsing or
  redaction defect.

**Spend.** Original run **$50.45** of $75.00 (6,000 generation+judgement pairs at
$0.0084 each, plus a reliability re-run and a smoke, under its ~$55 estimate);
re-scoring and provenance calibration **$0.00**; gold refresh **$2.104116** of a
$12.00 tier cap; coverage gate and generated-question build **$1.103157** of
$14.00 — $1.007207 of that is the build loop and **$0.095950** the follow-up
canary, and ⚠️ **that canary spend was never written to the build checkpoint**,
so a freshly-resumed governor would sum only the build and silently make it
re-spendable (fixed for any future canary run, and deliberately *not*
retro-appended: fabricating a checkpoint line to make a historical total look
complete would be worse than documenting the gap); eligibility labelling plus denser ladder **$25.2075** ($9.4316 +
$15.7759); depth sweep **$24.6845** of $35.00.

---

## Appendix A — arm by arm

**Families and operating points.** `sq8@512` is the IVF_SQ8 1024d reference;
`pca_uc_512` / `pca_uc_384` / `mrl_512` are **truncation** (uncentered PCA to
512d and 384d, Matryoshka to 512d, each on IVF_SQ8 at nprobe 512); `pq_m256@256`
and `rabitq@256` are **quantization** at the quantizer ceiling; `refine_k2@64`
and `sq8@64` / `sq8@16` / `sq8@8` / `sq8@4` are **IVF pruning** on the reference
index — clusters never
visited. That third family exists because reading the quantization curves showed
every mid-recall "quantizer" arm is really losing recall to pruning, not to its
quantizer (`refine_k2` reaches 0.992 at nprobe=1024 and 0.960 at nprobe=64), so a
pruning ladder costs no builds and gives every truncation arm a matched partner.

**The eleven-arm ladder, re-scored on corrected metrics**
(`results/ws4b/tier0_classification.csv`; quality is judge accuracy on the
corrected primary stratum):

| arm | recall | quality | Δ vs ref [95 % CI] | class | as published |
|---|---:|---:|---|---|---|
| `sq8@512` (ref) | 0.9905 | 0.600 | — | reference | reference |
| `refine@64` | 0.9677 | 0.609 | −0.009 [−0.037, +0.018] | PLATEAU | PLATEAU |
| `sq8@64` | 0.9674 | 0.578 | +0.022 [0.000, +0.046] | PLATEAU | PLATEAU |
| `pca_uc_512` | 0.9634 | 0.582 | +0.018 [−0.012, +0.049] | PLATEAU | PLATEAU |
| `pca_uc_384` | 0.9246 | 0.585 | +0.015 [−0.015, +0.046] | PLATEAU ‡ | PLATEAU ‡ |
| `sq8@16` | 0.9138 | 0.563 | +0.037 [+0.009, +0.065] | DROP | DROP |
| `sq8@8` | 0.8625 | 0.575 | +0.025 [−0.009, +0.058] | **INCONCLUSIVE** | PLATEAU |
| `pq_m256` | 0.8209 | 0.585 | +0.015 [−0.018, +0.049] | PLATEAU | PLATEAU |
| `sq8@4` | 0.7975 | 0.520 | +0.080 [+0.043, +0.120] | DROP | DROP |
| `rabitq` | 0.7837 | 0.554 | +0.046 [+0.012, +0.083] | **DROP** | PLATEAU |
| `mrl_512` | 0.7625 | 0.551 | +0.049 [+0.018, +0.083] | DROP | DROP |
| `parametric` (no passages) | — | 0.2156 ⁂ | — | leak floor | leak floor |

‡ The arm whose measured recall the plateau rule returned as its edge, in both
scorings. That coordinate is a grid point bounded by arm spacing, not a threshold
estimate — see the opening section. Two arms below it changed class on
re-scoring; neither is above it, so neither moves anything.
⁂ On the original 450-question basis, alongside a reference arm at 0.4978 — the
corrected-stratum accuracies in this table sit on a different denominator and are
not comparable to it.

**All six matched cross-family pairs** — the matching rule (|Δrecall| ≤ 0.03 on
the measured axis) applied *exhaustively* rather than to a hand-picked list
(`results/ws4/mechanism_pairs_all.csv`). The two in bold were never in the
original list of six, and they are the only two with an effect:

| pair | families | recall gap | Δ judge | 95 % CI | McNemar p | Bonf(6) | named? |
|---|---|---:|---:|---|---:|---:|:-:|
| **`pq_m256` vs `sq8@4`** | quant / pruning | 0.016 | **+0.0578** | **[+0.024, +0.093]** | 0.0016 | **0.0094** | **no** |
| `sq8@4` vs `rabitq` | pruning / quant | 0.020 | −0.0422 | [−0.078, −0.009] | 0.0248 | 0.149 | **no** |
| `pca_uc_384` vs `sq8@16` | trunc / pruning | 0.005 | +0.0133 | [−0.016, +0.042] | 0.451 | 1.00 | yes |
| `refine` vs `pca_uc_512` | pruning / trunc | 0.003 | +0.0089 | [−0.018, +0.036] | 0.618 | 1.00 | yes |
| `rabitq` vs `mrl_512` | quant / trunc | 0.024 | +0.0089 | [−0.020, +0.038] | 0.644 | 1.00 | yes |
| `sq8@64` vs `pca_uc_512` | pruning / trunc | 0.005 | −0.0067 | [−0.031, +0.018] | 0.720 | 1.00 | yes |

Not a fished pair: the same criterion applied to every candidate. The gap is
structural — the design *intended* to cover that region, and both pairs built for
it fell out as unmatched when the x-axis rule moved the pruning arms. Secondary
tests agree within the band that *was* covered: no truncation arm sits off the
non-truncation curve (residuals −0.007 [−0.030, +0.025]; +0.011 [−0.016, +0.039];
−0.009 [−0.036, +0.029] — the last **extrapolated: `mrl_512` sits at 0.758
recall, 2.4 points below the lowest non-truncation arm**, so the curve is
evaluated outside the data that defines it; `results/ws4/residuals.csv` flags it
`extrapolated = True`). ⚠️ So **two of the three residuals sit inside the covered
band; MRL's is extrapolated** — and 0.758 is well outside the 0.92–0.97 region
this same section concedes is the only one the mechanism claim covers. The
mediator pair tests include zero for every matched pair. ⚠️ There *is* a family pattern in the per-arm mediator
classification — every pruning arm is DROP on answer-presence, including `sq8@64`
at 0.969 recall, while truncation, `pq_m256` and `refine` at similar recall are
PLATEAU — but the instrument for the mechanism claim is the pair test, and it
does not reach significance. **Reported as a pattern worth a follow-up, not as a
finding.**

**The fifteen-arm ladder, on the filtered 264-question stratum**
(`results/ws4d/curve_points_15arm.csv`; ★ = added to close the ladder's gaps):

| arm | recall@10 | accuracy | | arm | recall@10 | accuracy |
|---|---:|---:|---|---|---:|---:|
| `sq8_np512` | 0.9913 | 0.6742 | | `sq8_np12` ★ | 0.8867 | 0.6667 |
| `refine_np64` | 0.9667 | 0.6818 | | `sq8_np8` | 0.8553 | 0.6326 |
| `sq8_np64` | 0.9655 | 0.6515 | | `sq8_np6` ★ | 0.8295 | 0.6174 |
| `pca_uc_512_sq8_np512` | 0.9621 | 0.6591 | | `pq_np256` | 0.8170 | 0.6667 |
| `sq8_np48` ★ | 0.9587 | 0.6705 | | `sq8_np4` | 0.7902 | 0.5795 |
| `sq8_np24` ★ | 0.9348 | 0.6667 | | `rabitq_np256` | 0.7803 | 0.6402 |
| `pca_uc_384_sq8_np512` | 0.9242 | 0.6705 | | `mrl_512_sq8_np512` | 0.7591 | 0.6326 |
| `sq8_np16` | 0.9095 | 0.6326 | | | | |

**The depth sweep, four SQ8 arms on the same stratum**
(`results/ws4e/depth_points.csv`; `separable` counts questions whose arms saw
different passages at that depth, written and committed before any row was
generated):

| k | always | never | discriminating | share | separable | separable_rate | null | **signal** | signal 95 % CI |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 104 | 101 | 59 | 0.2235 | 98 | 0.4796 | 0.0723 | **0.4073** | [0.2961, 0.5135] |
| 3 | 111 | 91 | 62 | 0.2348 | 172 | 0.3372 | 0.0435 | **0.2937** | [0.2110, 0.3743] |
| 5 | 116 | 83 | 65 | 0.2462 | 206 | 0.2913 | 0.0862 | **0.2051** | [0.1053, 0.2978] |
| 10 | 123 | 65 | 76 | 0.2879 | 241 | 0.3071 | 0.0870 | **0.2201** | [0.0750, 0.3360] |

⚠️ The individual `signal` CIs overlap while the paired difference is
significant. That is correct rather than contradictory: the bootstrap draws one
shared question index per replicate, so the difference is computed within-draw.
Raw `share` **rises** with depth; corrected `signal` **falls**; the second is the
composition artifact treated above. The separability ceiling behind the
`separable` column is 98 / 172 / 206 / 241 out of 264 — rates 0.3712 / 0.6515 /
0.7803 / 0.9129.

**Strata.** Splitting the 450 by whether the *exact* top-10 contains the gold: in
the 370 questions where it does, arm accuracies run 0.50–0.58 and the
DROP/PLATEAU pattern reappears; in the 80 where the gold sits only at ranks
11–100, every arm scores 0.11–0.18 — the leak floor — because no top-10 retriever
can reach it. **The recall-vs-quality question is entirely a question about the
first stratum**, and the depth-10 ceiling visible in those 80 questions is what a
k-sweep, not an nprobe sweep, probes.

---

## Appendix B — the Tier 0 audit rubric

*Reproduced substantially verbatim from the audit rubric fixed before the 107
candidates were read in bulk; the original is no longer a separate file. It
fixes what the labels mean so the threshold cannot be met by loosening a
definition after seeing the tally. Cross-references repointed to this document.*

### What is being decided

T0-C fires if **≥ 40 %** (`WS4B_T0C_STALE_SHARE`) of the error bucket labels as
`stale_gold` or `ambiguous_question`. If it fires, the 13 % (corrected: 11 %)
error rate is withdrawn as a *generator* statistic and reported as a
*gold-quality* statistic.

### What the auditor sees

`qid`, `question`, `golds`, every **distinct** answer given by any arm that erred
on that question, and the judge's reason. **No arm label, no recall, no presence
flag, no accuracy.** Answers are pooled across arms so the label is a property of
the question and its gold, not of one arm's draw.

### Labels, with precedence

Applied **in this order**; the first that matches wins.

1. **`judge_strict`** — the answer states the gold fact, and the judge marked it
   incorrect anyway (added detail, different formatting, a hedge that still
   commits, or a decline phrased so it was not caught as one).
   *Test:* would a careful reader say the candidate asserts the gold fact?
2. **`stale_gold`** — the gold was correct as of NQ's 2018 vintage, the 2023-11
   corpus documents a later fact, and the answer matches the **later** fact.
   *Test:* is the answer the post-2018 update of the gold? Not merely "different"
   — it must be the successor value.
3. **`ambiguous_question`** — the question underdetermines which entity, event,
   edition or sense is meant, and the answer is correct under a different
   reasonable reading.
   *Test:* can the question be read so the answer is right? A question with one
   natural reading is not ambiguous just because the answer is wrong.
4. **`model_error`** — everything else: the answer states a fact wrong under both
   the 2018 and the 2023 reading, or unsupported by any reading.

`unresolved` is available and **counts against T0-C** (toward the denominator,
not the numerator). It is used when the rubric genuinely cannot be applied from
the evidence shown, and must be justified per row.

### Counting

- numerator: `stale_gold` + `ambiguous_question`
- denominator: all 107 candidates, including `judge_strict`, `model_error` and
  `unresolved`

`judge_strict` is deliberately **not** in the numerator. It is a judge property,
not a gold property, and T0-C is a claim about gold quality.

### Auditor and its conflict of interest

⚠️ The auditor is **Claude Opus 5** — the same system that proposed this pass and
formulated the hypothesis T0-C tests. A real conflict of interest, disclosed
rather than managed away. Mitigations actually in force:

- this rubric is committed before the bulk read;
- the input is blind by construction (the CSV carries no arm or recall);
- the running share is **not** computed until every row is labelled;
- every row carries a one-line justification, so any label can be disputed
  individually;
- the result is reported as **provisional pending human spot-check**, and this
  document says so wherever T0-C is quoted.

A human spot-check of a random 20 rows is the recommended follow-up. If it
disagrees on more than 3 of 20, the audit should be redone by a human.
**That spot-check has still not happened for this bucket.**

---

## Appendix C — the decline audit rubric

*Reproduced substantially verbatim from the audit rubric fixed before the
decline evidence sheet was built or read; the original is no longer a separate
file. This audit was
added when the error audit above turned out to cover only error rows, leaving the
decline bucket — the larger of the two — unmeasured. Cross-references repointed
to this document.*

### What is being decided

The conversion target is ≥ 0.70; the corrected measurement is 0.600. Of the ~10
points needed, at most **+0.009** can come from the error bucket (only 3 of 38
reference-arm error rows are recoverable by a generator change), so **31.7 % of
all 93 declines must convert to correct answers**.

This audit measures the ceiling on that: what share of declines were on questions
the passages genuinely answered. It is an **upper bound** on what a generator
change can win — a recoverable decline is one a better generator *could* answer,
not one it *will*.

### Why the evidence differs from the error audit

A decline is the string "I don't know". It carries no signal about why. The only
way to classify one is to read what the model was given. So the sheet shows, per
question: the question, the golds, and the **context windows around every
word-boundary gold match** in the ten retrieved passages (±250 chars).

This also puts `presence_wb` itself on trial. Word-boundary matching fixed the
`"12"`-inside-`"2012"` pathology, but a gold string can still appear in a passage
that does not answer the question. If that is common, the corrected conversion
denominator is *still* inflated and the 28 % decline rate *still* overstated.

### What the auditor sees

Question, golds, and gold-match context windows. **No arm label, no recall, no
presence flag, no accuracy, no other arm's answer.**

⚠️ **Partial blindness compromise, disclosed:** 18 of the 93 decline questions
also appear in the error-audit candidate set, so their questions and golds were
seen during that audit. They are flagged `seen_in_error_audit` in the output so
their labels can be checked separately, and the write-up reports whether the
label distribution differs between the 18 and the other 75.

### Labels, with precedence

Applied in this order; first match wins.

1. **`answerable_declined`** — the retrieved context **states the answer** to the
   question as asked. The decline is a generator failure and is **recoverable**.
   *Test:* could a careful reader answer the question from the shown context
   alone, and would that answer match a gold?
2. **`not_supported`** — a gold string appears, but the context does **not**
   answer the question (incidental mention, wrong entity, wrong sense, gold
   appears only as a list item or navigation fragment). The decline is **correct
   behaviour**, and the row is evidence that `presence_wb` over-counts.
3. **`ambiguous_question`** — the question underdetermines entity, year, sense or
   edition, so no single answer is retrievable. Declining is defensible.
4. **`stale_gold`** — the context answers the question but with a *later* fact
   than the gold, so an answer would have been judged incorrect anyway.
5. **`unresolved`** — the rubric cannot be applied from the shown context. Counts
   as **not recoverable**.

Only **`answerable_declined`** counts as recoverable.

### Branches

Let **R** = `answerable_declined` / 93. A generator-improvement pass needs 31.7 %
of declines to convert, and can only convert recoverable ones.

| branch | condition | consequence |
|---|---|---|
| **D1** | R ≥ 0.50 | the conversion gate is plausibly reachable — 63 % of recoverable declines must convert. The pass runs as specified. |
| **D2** | 0.32 ≤ R < 0.50 | the gate requires converting essentially **every** recoverable decline. The pass runs only with the gate re-derived first, and is expected to fail it. |
| **D3** | R < 0.32 | **the gate is unreachable by arithmetic.** The pass does not run in its current form; the changes it proposes are aimed at a bucket too small to matter. |

Secondary, and reported regardless: if `not_supported` ≥ 0.20, the corrected
conversion is restated again with that share removed from the denominator, and
the "declines 28 %" figure is amended.

### Auditor and its conflict of interest

⚠️ Auditor: **Claude Opus 5** — again the same system that proposed the
hypothesis. Same mitigations as Appendix B: rubric committed first, blind input,
share not computed until every row is labelled, one justification per row, result
**provisional pending human spot-check**.

Note the incentive here runs *against* the auditor's earlier position: D3 kills
the follow-up spend this line of work was built to justify.
