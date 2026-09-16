# The RAG Cost Curve — benchmark repo

Simon Hearne, Zilliz — see [Disclosure](#disclosure)

Reproducible benchmarks behind the Haystack EU 2026 talk **"The RAG Cost
Curve: When a Retrieval Index Beats Live Search"** (Simon Hearne). Every
number in the talk traces back to a committed CSV in `results/`, produced by
a notebook in `notebooks/`, over pinned public data. No private data, fixed
seeds, checksums throughout.

## What's measured

Recall / latency / memory across quantization schemes (SQ8, PQ, RaBitQ ±
refine) and Matryoshka dimensionality reduction on Milvus, at 1M and 10M
vectors; whether recall ever stops predicting end-task answer quality; and a cost
model that finds the break-even query volume between maintaining a vector
index and "just search live / stuff the context" under Claude, OpenAI, and
open-weight pricing — with prompt caching steelmanned on the live side.

## The claims, and how they held up

Every statement in the talk that needs data behind it, and where the
evidence lives. A verdict of "retired" or "not reproduced" is a result, not
a gap — three of the fifteen came back flatly against the abstract and four
more are only partly for it, and they are stated here the way they were
measured.

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| C1 | Agentic retrieval with caching and hybrid tools is genuinely low-effort for small, fast-changing corpora | Half answered — the opposing side is largely right on effort, and the shipped default is the configuration that loses | [code-retrieval.md](results/code-retrieval.md) |
| C2 | "Just search live" has a cost curve of a very different shape | Supported — live search pays for the history, the index pays a floor plus the answer | [cost-model.md](results/cost-model.md) |
| C3 | Recall / latency / memory measured across quantization schemes | Supported at 1M and 10M — and plain SQ8 quietly beats the fancy binary quantizer | [index-footprint.md](results/index-footprint.md) |
| C4 | Dimensionality reduction — how far you can compress without trashing recall | Supported, but the Matryoshka premise is withdrawn **for this model**: uncentered PCA is what makes it work | [index-footprint.md](results/index-footprint.md) |
| C5 | Where the refinement step is doing the real work | Supported — `refine_k` = 1–2 does all of it, confirmed at 10M | [index-footprint.md](results/index-footprint.md) |
| C6 | Where recall stops predicting end-task answer quality | **Retired** — the threshold is not a locatable quantity on this task | [recall-vs-quality.md](results/recall-vs-quality.md) |
| C7 | Benchmarks run on public datasets, methodology and numbers open source | Supported by this repository — pins, checksums, CSVs, harnesses and run transcripts are all here, under Apache-2.0 | [results/](results/) |
| C8 | A 16–32x compressed vector index costs single-digit percentage recall loss | **Not at ≥16x** — the honest measured-footprint ceiling is 8.92x at 10M (8.66x at 1M); 32x is the scanned payload, ~3.1–3.5x the memory you provision | [index-footprint.md](results/index-footprint.md) |
| C9 | Break-even against live search lands in tens-to-hundreds of queries per day | Supported — 334 / 15 / 10 queries per day, but only against a dedicated-instance floor, and on the code-search workload the index never repays at any volume | [cost-model.md](results/cost-model.md) |
| C10 | memsearch indexes plain markdown into a vector store instead of replaying full transcripts | Three of four architecture claims confirmed, one of those by code inspection only; the fourth — progressive recall — was never exercised, so it is untested, not "works" | [code-retrieval.md](results/code-retrieval.md) |
| C11 | claude-context avoids loading whole repositories into context on every query | **Not reproduced at any tested configuration, on either repository** — and inverted at the shipped default only, thinly: paired median +2,857 tokens, CI [+9, +7,692] | [code-retrieval.md](results/code-retrieval.md) |
| C12 | Live re-search over large private history or repos gets expensive fast, with measured deltas | Split — no within-repository growth is detectable at this sample size; across an agent history the index grows 2.33x against replay's 9.8x | [code-retrieval.md](results/code-retrieval.md) |
| C13 | Frame the corpus-size / update-frequency / query-volume variables that decide it | Supported, and ranked — deployment mode (69,000x spread) dwarfs every other knob | [cost-model.md](results/cost-model.md) |
| C14 | Honest assessment of the refinement requirement, not just the compression headline | Supported — the refine copy **dominates** the gap between the 32x payload headline and the measured footprint (32x → 13.6x from fixed index overhead, 13.6x → 3.1x from the refine copy) | [index-footprint.md](results/index-footprint.md) |
| C15 | When an index genuinely doesn't pay — small or volatile corpora, low query volume, public data already in training | Supported, and now a measured spectrum — 0.625 answered with no corpus access on famous code, 0.000 on post-cutoff GitHub-issue prose | [code-retrieval.md](results/code-retrieval.md), [non-code-retrieval.md](results/non-code-retrieval.md), [cost-model.md](results/cost-model.md) |

## Pinned foundations

| Thing | Pin |
|---|---|
| Vector DB | Milvus standalone `v2.6.18` ([docker-compose.yml](docker-compose.yml), official release compose) |
| Client | `pymilvus==2.6.17` |
| Corpus | [`mixedbread-ai/wikipedia-embed-en-2023-11`](https://huggingface.co/datasets/mixedbread-ai/wikipedia-embed-en-2023-11) @ `395eebc` — 41.49M English Wikipedia chunks, 1024-d |
| Embedding model | [`mixedbread-ai/mxbai-embed-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) (Matryoshka-trained, Apache-2.0) |
| QA set | [`google-research-datasets/nq_open`](https://huggingface.co/datasets/google-research-datasets/nq_open) @ `5dd9790` (CC-BY-SA-3.0) |
| Seed | `42` everywhere (`benchlib/config.py` is the single source of truth) |

## Reproduction path

```bash
# 0. Python 3.11+, Docker, ~80 GB free disk for the 10M scale (25 GB for 1M)
make setup            # venv + pinned requirements
source .venv/bin/activate

# 1. Start Milvus (needed from notebook 02 onward, not for 01)
make up               # docker compose up -d + health wait

# 2. Fixtures: download pinned data, build slices + brute-force ground truth
make data-1m          # ~2.3 GB download   (make data-10m: ~23 GB)
jupyter lab           # run notebooks/01_datasets.ipynb top to bottom

# 3. Benchmarks — run notebooks in numeric order; each writes CSVs to results/
#    01 fixtures -> 02 quantization -> 03 dimensionality -> 04 recall-vs-quality
#    -> 05 cost model.  06a/06b (case studies) are independent.

# 4. Verify you rebuilt the same fixtures we used
make verify           # re-hashes everything against data/MANIFEST.json
```

`make verify` reports three counts — **verified / absent / corrupt** — and
fails only on *corrupt*. **Absent is expected and is not an error**: `data/` is
gitignored, so on a fresh clone most manifest entries simply have not been
generated yet, and they appear as you work through the steps above. Only a
*corrupt* file — present but hashing differently — means what you have is not
what produced the committed results. Once you have generated everything, run
`python scripts/fixtures/verify_checksums.py --strict`, which additionally
fails on absent entries; that is the gate this repo runs against a complete
checkout.

Expected runtimes and RAM are flagged at the top of each notebook and at
each heavy cell. The heavy step is one-time: exact top-100 ground truth at
10M scale is an overnight run on an Apple-Silicon GPU (hours on CUDA); it is
chunked and resumable, cached forever after.

## Repo layout

```
LICENSE              Apache-2.0 — code only; see "Licenses" below for the data
pyproject.toml       pytest configuration
docker-compose.yml   Milvus v2.6.18 standalone (official release compose, pinned)
requirements.txt     pinned Python deps (+ requirements.lock, the resolved set)
benchlib/            shared config/constants + checksum manifest helpers
scripts/             dataset download (pinned revisions, sha256 manifest),
                     verification, and the benchmark harnesses
notebooks/           01 fixtures · 02 quantization · 03 dims · 04 quality ·
                     05 cost model · 06a/06b case-study harnesses
tests/               unit tests pinning the arithmetic and the fixtures
results/             committed CSVs — the numbers the slides are built from —
                     and the five documents that read them:
                       index-footprint.md     quantization + dimensionality
                       recall-vs-quality.md   recall vs answer quality
                       cost-model.md          the break-even model
                       code-retrieval.md      agentic code + memory retrieval
                       non-code-retrieval.md  prose retrieval
data/                local fixtures (gitignored; MANIFEST.json committed)
```

### Which `results/ws*/` directory belongs to which document

The `ws*` directories are the raw CSV outputs, named after the benchmark
stage that produced them. Each topic document above reads one or more of
them:

| CSV directories | Document | What it covers |
|---|---|---|
| `ws1/`, `ws2/`, `ws3/` | [index-footprint.md](results/index-footprint.md) | quantization, dimensionality, measured footprint |
| `ws4/`, `ws4b/`, `ws4c/`, `ws4d/`, `ws4e/` | [recall-vs-quality.md](results/recall-vs-quality.md) | recall vs end-task answer quality, and the depth sweep |
| `ws5/` | [cost-model.md](results/cost-model.md) | the break-even model |
| `ws6a/`, `ws6b/`, `ws6c/` | [code-retrieval.md](results/code-retrieval.md) | agentic code retrieval and agent memory |
| `ws8/` | [non-code-retrieval.md](results/non-code-retrieval.md) | prose retrieval |

### A note on `spec N.N` references in comments

Many code comments and docstrings cite a `spec N.N` section. Those refer to
per-benchmark design records that were **consolidated into the `results/*.md`
documents** before release; the section numbering did not survive the merge.
The citations are kept because they mark *which* decisions were pre-registered
rather than chosen after seeing data — which is the load-bearing fact — but do
not expect to resolve a section number to a heading. The corresponding topic
document named in the table above is the authority.

## Honesty notes

Compression ratios are reported two ways — vector-payload and total loaded
footprint (refine structures included), and the talk says which is which.
The live-search baseline gets prompt caching at optimistic hit rates before
any break-even is claimed. Corpus/QA caveats (answer drift between the NQ
collection date and the 2023-11 Wikipedia snapshot) are handled in notebook
01 with an answer-presence filter and reported, not hidden.

## Disclosure

No part of this work was commissioned, reviewed, or approved by Zilliz before
publication, and no result was changed on Zilliz's behalf. Where an auditor with
their own conflict is involved, that is disclosed separately at the point of use
in [recall-vs-quality.md](results/recall-vs-quality.md).

## Licenses

Code: Apache-2.0. Wikipedia text: CC-BY-SA. NQ-Open: CC-BY-SA-3.0.
mxbai-embed-large-v1: Apache-2.0. The corpus embeddings dataset carries no
explicit license tag; this repo therefore redistributes **no corpus data** —
only download scripts, checksums, and derived aggregate results.

### Exception: the agent-memory synthetic corpus **is** committed

`data/ws6b_corpus/` is the one corpus this repo redistributes, and the
exception is deliberate. The no-corpus-data rule above exists because the
embeddings dataset carries no licence tag. That cannot apply to text produced
by `scripts/code-retrieval/generate_ws6b_corpus.py` at seed 42 — there is no upstream rights
holder. Committing it alongside the generator that re-derives it serves the
rule's deeper purpose, which is reproducibility: a reader diffs bytes instead
of trusting a generator.

Every file is sha256'd into `data/MANIFEST.json`, and
`tests/code_retrieval/test_memory.py::test_generator_is_byte_exact_across_runs` asserts that
regenerating reproduces those bytes. Exception granted by Simon Hearne,
2026-08-18.

### Second exception: run transcripts embed source excerpts

The "no corpus data" rule above is about *corpora*; it is not a claim that no
upstream source text appears anywhere in this repo, and stating it that way
would be inaccurate. **The committed run transcripts under `results/ws6a/` and
`results/ws6c/` contain `tool_result` blocks carrying verbatim excerpts** —
grep hits, file reads, and retrieved code chunks — **from the repositories the
benchmarks were run against.** Both upstream projects are permissively
licensed:

| Benchmarked repository | Licence | Where its excerpts appear |
|---|---|---|
| `fastapi/fastapi` @ `a1fa70d4` | **MIT** | `results/ws6a/transcripts/`, `results/ws6a/transcripts_smoke/`, `results/ws6c/transcripts_topk/` |
| `agentic-hil/agentic-hil` @ `837f3981` | **Apache-2.0** | `results/ws6c/transcripts_agentic_hil/`, `results/ws6c/transcripts_ws9_s/`, `results/ws6c/transcripts_ws9_m/` |

Two further directories carry no upstream excerpts and are listed for
completeness. `results/ws6c/transcripts_gate_*/` are the non-memorisation gate
runs over `agentic-hil/agentic-hil`, `oaslananka/kicad-mcp-pro` (MIT) and
`gmboquet/mixle` (MIT): the model answered **without repository access**, so
those transcripts contain its questions and refusals, not code. `results/ws6b/transcripts/`
are runs over the synthetic agent-memory corpus committed under the first
exception above.

> **The prose-retrieval benchmark is the case where this exception was
> declined — 2026-09-11.**
>
> It benchmarked GitHub **issue and PR comment threads** from
> `kubernetes/kubernetes` and `rust-lang/rust`. Its committed run transcripts
> carried ~1.5M characters of that text across 982 documents, including
> commenter usernames — and the reasoning below **does not extend to them.** The
> rows above are *code* under MIT and Apache-2.0. Comment prose is not code: a
> repository's licence does not cover what its users write on its issue
> threads, which belongs to the individual commenters. There is no permissive
> upstream licence to point at, so listing those runs in the table above would
> have been a fiction.
>
> **`results/ws8/transcripts/` and `results/ws8/transcripts_parametric/` were
> therefore deleted** (225 files, 2.7 MB) rather than disclosed and kept. The
> cost is real and is stated in
> [non-code-retrieval.md](results/non-code-retrieval.md) under "What the
> transcript removal costs, stated plainly": three of that benchmark's
> claims — the parametric arm *declining* rather than guessing, the per-call
> payload, and the search-call-count driver — are now report-only, checkable
> against the aggregate CSVs but not against raw evidence. **The aggregate
> results in `results/ws8/*.csv` are unaffected**, and the paired phase is
> deterministic and re-runnable (~$48) for anyone who needs those three
> independently confirmed.
>
> **Deleting the files does not remove them from git history, so this
> repository was published from a single squashed commit.** The development
> history that carried those transcripts is not part of the public history, and
> the published tree contains neither directory. That squash is load-bearing
> for this licensing position, which is why the public history is one commit.

These are excerpts an agent retrieved while answering a question, not a
redistribution of either codebase: no repository is reconstructible from them,
and neither is checked out into this repo (both are cloned at a pinned commit
by script). They are kept because **they are the evidence**. The transcripts are
what let a reader verify that `search_code` was called with the top-k we say it
was, that the parametric arm *declined* rather than guessed wrong, and that the
payload sizes behind every mechanism claim are real. Deleting them would leave
those findings resting on our summary of them. Licence and attribution for the
excerpted code remain with the upstream projects above.

