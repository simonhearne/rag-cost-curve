"""Single source of truth for every pinned constant in the benchmark.

Every notebook imports from here. If a number in the talk can't be traced
back to a constant in this file plus a committed CSV in results/, it
doesn't go on a slide.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths ----
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

# ------------------------------------------------------------ the corpus ----
# English Wikipedia (2023-11 snapshot, Cohere chunking) embedded with
# mixedbread-ai/mxbai-embed-large-v1 (1024-d, Matryoshka-trained, Apache-2.0).
# 41,488,110 chunks in 721 parquet shards (~130 MB each, ~57.5k rows/shard).
CORPUS_REPO = "mixedbread-ai/wikipedia-embed-en-2023-11"
CORPUS_REVISION = "395eebc2bdd5d7b79a5c6d04d7fc4e66d1113494"  # pinned commit
CORPUS_TOTAL_ROWS = 41_488_110
CORPUS_N_SHARDS = 721
CORPUS_SHARD_TEMPLATE = "data/train-{i:05d}-of-00721.parquet"

EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
EMBED_MODEL_REVISION = "b33106f585b9ce46904ad7443a3b52b7a63e231c"  # pinned commit
EMBED_DIM = 1024
# mxbai retrieval convention: queries get this prefix, documents get none.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Similarity: cosine. We L2-normalize everything once at ingest so that
# inner product == cosine everywhere downstream (Milvus metric_type="IP").

# ---------------------------------------------------------------- slices ----
# The 1M slice is a strict prefix of the 10M slice (both are prefixes of the
# shard order). Deterministic: no shuffling, no sampling.
SLICES = {"1m": 1_000_000, "10m": 10_000_000}

# ----------------------------------------------------------------- QA set ----
# NQ-Open: open-domain questions over English Wikipedia with short gold
# answers. train=87,925 / validation=3,610. CC-BY-SA-3.0.
NQ_REPO = "google-research-datasets/nq_open"
NQ_REVISION = "5dd9790a83002ad084ddeb7c420dc716852c6f28"  # pinned commit
NQ_TRAIN_FILE = "nq_open/train-00000-of-00001.parquet"
NQ_VAL_FILE = "nq_open/validation-00000-of-00001.parquet"

# Ground-truth query set: N_GT_QUERIES sampled (SEED) from NQ-Open *train*.
# NQ-Open *validation* (3,610 q) is reserved untouched for WS4 end-task eval.
SEED = 42
N_GT_QUERIES = 10_000
GT_TOP_K = 100

# ------------------------------------------------------------ ground truth ----
# Exact brute-force search parameters (notebook 01, step 5).
GT_CORPUS_CHUNK = 200_000   # corpus rows per matmul chunk (~0.8 GB fp32)
GT_QUERY_BLOCK = 2_048      # query columns per matmul chunk
# fp16 prefilter: compute scores in fp16, keep top GT_PREFILTER_K candidates,
# exact-rescore those in fp32. ~2x faster on MPS/CUDA. Verified against pure
# fp32 on a subsample inside the notebook before being trusted.
GT_USE_FP16_PREFILTER = False
GT_PREFILTER_K = 500

# ----------------------------------------------------------------- Milvus ----
MILVUS_URI = "http://localhost:19530"
MILVUS_VERSION = "2.6.18"  # must match docker-compose.yml


def slice_path(scale: str) -> Path:
    """Normalized fp32 memmap of the corpus slice, shape (N, EMBED_DIM)."""
    return DATA_DIR / f"corpus_{scale}_fp32_norm.npy"


def ids_path(scale: str) -> Path:
    return DATA_DIR / f"corpus_{scale}_ids.parquet"


def gt_path(scale: str) -> Path:
    return DATA_DIR / f"gt_{scale}_top{GT_TOP_K}.npz"


# ------------------------------------------------------------------ WS6a ----
# Code-search case study: claude-context MCP vs agentic grep vs context
# stuffing vs parametric memory. Feeds C1/C11/C12/C15 and the WS5 cost model.
# Design: results/code-retrieval.md

WS6A_REPO = "fastapi/fastapi"
WS6A_COMMIT = "a1fa70d4237d50aae6586a0d9b229df583463d21"
WS6A_REPO_URL = "https://github.com/fastapi/fastapi.git"

# Agent model is identical across all four arms; judge must differ from it.
WS6A_AGENT_MODEL = "claude-sonnet-5"
WS6A_JUDGE_MODEL = "claude-opus-5"

# claude-context, pinned exactly. Invoked as an stdio MCP server via npx.
WS6A_MCP_PACKAGE = "@zilliz/claude-context-mcp@0.1.15"
WS6A_EMBED_PROVIDER = "OpenAI"
WS6A_EMBED_MODEL = "text-embedding-3-small"

WS6A_ARMS = ("indexed", "agentic", "stuffed", "parametric")
# Applies to BOTH tool-using arms. Capping one and not the other would make the
# arms differ in tool availability AND turn budget -- two variables, one
# measurement. Single-turn arms are capped at 1 by construction.
WS6A_TURN_CAP = 15
WS6A_SINGLE_TURN_ARMS = ("stuffed", "parametric")

# Hard abort on cumulative billed spend, across every phase. See spec section 8.
WS6A_BUDGET_USD = 75.00

# Arm (c) prefix: context window minus a reserve for question + answer.
WS6A_CONTEXT_WINDOW_TOKENS = 1_000_000
WS6A_PREFIX_RESERVE_TOKENS = 40_000
WS6A_PREFIX_BUDGET_TOKENS = (
    WS6A_CONTEXT_WINDOW_TOKENS - WS6A_PREFIX_RESERVE_TOKENS
)

# Relevance tiering for the arm (c) prefix. Lexicographic ORDER within a tier.
# Pure-lexicographic ordering was rejected: docs/de/... sorts before fastapi/,
# so the prefix would hold 13 doc translations and none of the library --
# a strawman. See spec section 4.
WS6A_TIER_ORDER = ("fastapi/", "docs_src/", "tests/", "docs/en/")

# Nested scaling subsets, same repo, same commit (spec section 12).
WS6A_SUBSETS = {
    "S": ("fastapi/",),
    "M": ("fastapi/", "docs_src/", "tests/"),
    "L": (),  # empty prefix tuple == whole repo
}

WS6A_N_QUESTIONS = 40
WS6A_N_SCALING_QUESTIONS = 10
# Difficulty mix for the 40-question set (spec section 5). Sums to
# WS6A_N_QUESTIONS; keys match VALID_DIFFICULTIES in code_retrieval/bench.py.
WS6A_DIFFICULTY_MIX = {"locate": 15, "trace": 15, "multi_hop": 10}

WS6A_TEXT_EXTENSIONS = frozenset({
    ".py", ".md", ".mdx", ".yaml", ".yml", ".json", ".js", ".ts", ".tsx",
    ".txt", ".sh", ".toml", ".cfg", ".ini", ".html", ".css",
})


def ws6a_dir() -> Path:
    """results/ws6a/ -- every committed WS6a output lands here."""
    return RESULTS_DIR / "ws6a"


def ws6a_checkout() -> Path:
    """Local checkout of the repo under test, pinned to WS6A_COMMIT."""
    return DATA_DIR / "ws6a_fastapi"


# ---------------------------------------------------------------- WS6b ----
# memsearch agent-memory benchmark.
# Design: results/code-retrieval.md
# Nothing here may change during a run.

WS6B_MEMSEARCH_VERSION = "0.4.0"
WS6B_MEMSEARCH_COMMIT = "3149dc3"
WS6B_MEMSEARCH_REPO = "https://github.com/zilliztech/memsearch"
# The shipped CLI is invoked from the plugin marketplace checkout, so the
# pinned commit is the one actually executed rather than one asserted.
WS6B_MEMSEARCH_ROOT = Path(
    "~/.claude/plugins/marketplaces/memsearch-plugins").expanduser()

WS6B_EMBED_PROVIDER = "openai"
WS6B_EMBED_MODEL = "text-embedding-3-small"
WS6B_MILVUS_URI = "http://localhost:19530"

WS6B_AGENT_MODEL = "claude-sonnet-5"
WS6B_JUDGE_MODEL = "claude-opus-5"

WS6B_ARMS = ("memsearch", "replay", "parametric")
WS6B_TURN_CAP = 15
WS6B_BUDGET_USD = 75.00

WS6B_CONTEXT_WINDOW_TOKENS = 1_000_000
WS6B_RESERVE_TOKENS = 40_000
WS6B_TRUNC_BUDGET_TOKENS = WS6B_CONTEXT_WINDOW_TOKENS - WS6B_RESERVE_TOKENS

# Targets only. The published x-axis is the MEASURED count_tokens total.
WS6B_SIZES = {"s": 100_000, "m": 400_000, "l": 1_200_000}
WS6B_COLLECTIONS = {"s": "ws6b_s", "m": "ws6b_m", "l": "ws6b_l"}

WS6B_CORPUS_DIR = DATA_DIR / "ws6b_corpus"
WS6B_CORPUS_START_DATE = "2025-01-06"  # a Monday; weekdays only

WS6B_N_PROBES = {"scaling": 12, "depth": 18, "superseded": 8}
WS6B_DEPTH_BANDS = (0.10, 0.50, 0.90)
WS6B_DEPTH_TOLERANCE = 0.03
WS6B_SUPERSEDED_ORIGINAL_BAND = (0.40, 0.55)
WS6B_SUPERSEDED_WINDOW_MARGIN = 0.10

WS6B_MIN_DISTRACTOR_COMPONENTS = 3
WS6B_MIN_COMPONENT_MENTIONS = 5

WS6B_GATE_PASS = 0.05
WS6B_GATE_FAIL = 0.10


def ws6b_dir() -> Path:
    """results/ws6b/ -- every committed WS6b output lands here."""
    return RESULTS_DIR / "ws6b"


# ----------------------------------------------------------------- WS4 ----
# Recall vs end-task answer quality (C6).
# Design: results/recall-vs-quality.md
# Nothing here may change during a run.

WS4_SCALE = "10m"
WS4_TOP_K = 10                     # passages handed to the generator
WS4_GEN_MODEL = "claude-sonnet-5"  # thinking disabled, identical across arms
WS4_JUDGE_MODEL = "claude-opus-5"  # blind; must differ from the generator
WS4_GEN_MAX_TOKENS = 256
WS4_N_MAIN = 450                   # spec 3.2: drawn from the filter pass set
WS4_N_SANITY = 50                  # spec 3.3: drawn unfiltered, disjoint
WS4_N_BOOTSTRAP = 10_000
WS4_EQUIVALENCE_MARGIN = 0.05      # spec 7.1: PLATEAU if CI upper < this
WS4_MATCH_TOLERANCE = 0.03         # spec 7.2: |r(T) - r(P)| <= this is matched
WS4_PAIR_ALPHA = 0.05              # face-value CI for pair A
WS4_PAIR_BONFERRONI_N = 5          # pairs B-F evaluated at alpha / 5
WS4_JUDGE_RELIABILITY_N = 60
WS4_BUDGET_USD = 75.00
WS4_SHARED_COLLECTION = "ws2_10m_shared"   # frozen in WS2; never re-ingested
WS4_NLIST = 4096
WS4_TRUNC_NPROBE = 512             # every truncation arm served at its ceiling
WS4_SQ8_NPROBES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
WS4_LOW_NPROBES = (1, 2, 4, 8)     # the WS2 10M SQ8 sweep stopped at 16
WS4_WS3_SWEEP = (16, 32, 64, 128, 256, 512)  # WS3 10M sweep points
WS4_REFINE_NPROBE = 64             # the brief's ~0.95 refine_k2 point
WS4_QUANT_NPROBE = 256             # pq_m256 / rabitq at their quantizer ceiling
WS4_GATE_G3_EXPECTED = 0.99166     # results/ws2/curves_10m.csv ivf_sq8 @512
WS4_GATE_G3_TOL = 0.002
WS4_GATE_G1_TOL = 0.05             # offline ceiling vs the 1M ceiling
WS4_GATE_G2_BAND = (-0.03, 0.005)  # Milvus arm relative to its own ceiling
WS4_GATE_G4_MIN_RECALL = 0.97      # reference arm on the WS4 main questions
WS4_GATE_N_QUERIES = 1000          # G1/G2 paired subset: first N GT queries
WS4_CEILING_1M = {                 # results/ws3/transform_ceiling_1m.csv
    "pca_uc_512_sq8": 0.96374, "pca_uc_384_sq8": 0.91097,
    "mrl_512_sq8": 0.70035,
}
WS4_TRUNC_ARMS = {                 # arm -> (transform, dim)
    "pca_uc_512_sq8": ("pca_uc", 512),
    "pca_uc_384_sq8": ("pca_uc", 384),
    "mrl_512_sq8": ("mrl", 512),
}
WS4_PCA_ROTATION = "pca_rotation_1m_500000_pca_uc.npz"  # in data/ws3_derived
WS4_DISK_FLOOR_GB = 50


def ws4_dir() -> Path:
    """results/ws4/ -- every committed WS4 output lands here."""
    return RESULTS_DIR / "ws4"


def ws4_cache() -> Path:
    return DATA_DIR / "ws4_cache"


# ---------------------------------------------------------------- WS4b ----
# Generator headroom and a knee with a confidence interval.
# Design: results/recall-vs-quality.md
# These ADD to WS4; no WS4_* constant above may be changed by WS4b.

WS4B_PRESENCE_WORD_BOUNDARY = True   # spec 3.1: \b-delimited alias matching
WS4B_MIN_GOLD_CHARS = 3              # spec 3.1: shorter normalised aliases are
                                     # unmatchable by substring and are dropped
WS4B_IDK_PREFIX_MATCH = True         # spec 3.2: a leading decline line counts
WS4B_PRIMARY_STRATUM = "gold_in_exact_top10"   # spec 3.4
WS4B_CONVERSION_GATE = 0.70          # spec 7.2 G2
WS4B_LEAK_CEILING = 0.30             # spec 7.2 G1; WS4 spec 7.5 tier boundary
WS4B_GEN_THINKING = True             # spec 4.1
WS4B_GEN_MAX_TOKENS = 2048           # spec 4.1; WS4 hit max_tokens on 0/6000
WS4B_EXTRA_NPROBES = (6, 12, 24, 48)  # spec 5.1, on the existing SQ8 index
WS4B_CHANGEPOINT_BOOT_N = 10_000     # spec 6.1
WS4B_LADDER_CORRECTION = "holm"      # spec 6.2
WS4B_BUDGET_USD = 90.00              # spec 8
WS4B_T0C_STALE_SHARE = 0.40          # spec 7.1 T0-C threshold
WS4B_KILT_REPO = "facebook/kilt_tasks"          # spec 3A
WS4B_KILT_NQ_DEV = "nq/validation-00000-of-00001.parquet"
WS4B_P1_DISCORD = 0.40               # spec 3A.4 P1 boundary
WS4B_P3_DISCORD = 0.15               # spec 3A.4 P3 boundary
WS4B_AVAL_AGREEMENT = 0.75           # spec 3A.4 V1/V2 boundary
WS4B_T0_CONVERSION_DELTA = 0.05      # spec 7.1 T0-A/T0-B boundary
WS4B_WS4_CONVERSION = 0.581522       # WS4 measured P(correct | gold present)


def ws4b_dir() -> Path:
    """results/ws4b/ -- every committed WS4b output lands here. WS4b never
    writes into results/ws4/."""
    return RESULTS_DIR / "ws4b"


# ----------------------------------------------------------- WS4b Tier 0d ----
# Gold refresh on the audited error bucket.
# Design: results/recall-vs-quality.md
# These ADD to WS4b; no constant above may be changed.

WS4B_T0D_N_ROWS = 107               # spec 2: the audited error bucket, exactly
WS4B_T0D_PROPOSE_MODEL = "claude-sonnet-5"   # spec 3.2 stage 1
WS4B_T0D_VERIFY_MODEL = "claude-opus-5"      # spec 3.3 stage 2
WS4B_T0D_TIER_CAP_USD = 12.00       # spec 6; inside WS4B_BUDGET_USD, never raised
WS4B_T0D_UPHOLD_CEILING = 0.90      # spec 5 G2: over PROPOSED CHANGES only
WS4B_T0D_CONCORDANCE_MIN = 0.60     # spec 5 G1
WS4B_T0D_MIN_CHANGES = 10           # spec 5 G2 evaluability floor
WS4B_T0D_FALSE_CORRECT_MIN = 0.05   # spec 5 R3
WS4B_T0D_REDACT_MIN_CHARS = 20      # fix round 2 review: a licensing pin, not
                                     # a tuning knob -- the floor above which a
                                     # verbatim span shared with a row's own
                                     # passages is redacted before
                                     # propose_reason/verify_reason are
                                     # committed (the embeddings dataset has
                                     # no license tag). Chosen with margin
                                     # below the shortest leak found in review
                                     # (30 chars).


def ws4b_cache() -> Path:
    """data/ws4b_cache/ -- Tier 0d's checkpoint. NEVER data/ws4_cache/, which
    holds passages.parquet, frozen input to Tier 0/0b. WS4b has had no cache
    until now because no WS4b script had ever spent money."""
    return DATA_DIR / "ws4b_cache"


# ---------------------------------------------------------------- WS4c ----
# Does retrieval bind on a multi-hop task? (C6, third attempt)
# Design: results/recall-vs-quality.md
# These ADD to WS4; no WS4_* or WS4B_* constant above may be changed by WS4c.
# Every value below is copied from spec 2A.4 (amendment, 2026-09-11).

WS4C_BUDGET_USD = 75.00        # spec 7, total governor -- UNCHANGED by 2A
WS4C_BUILD_BUDGET_USD = 14.00  # spec 7, Phase 0b governor, SEPARATE from the
                               # eval so a low filter yield halts the build.
                               # Per G-A this is never raised to reach 600.
WS4C_N_QUESTIONS = 600         # spec 3A G-A
WS4C_QGEN_MODEL = "claude-sonnet-5"   # spec 2A.2 question generator
WS4C_QJUDGE_MODEL = "claude-opus-5"   # spec 2A.2 F5 validity judge
WS4C_MIN_BRIDGE_CHARS = 10     # spec 2A.1 bridge distinctiveness; a short
                               # title is not a real hop and matches spuriously
WS4C_MAX_PAGE_SHARE = 0.02     # spec 2A.1 concentration cap, enforced at
                               # construction so G-C cannot be discovered late
WS4C_FLOOR_MIN_SHARE = 0.30    # spec 3A G-D floor gate
WS4C_CANDIDATE_TARGET = 3_000  # spec 7 build sizing -- AN ESTIMATE, NOT A GATE


def ws4c_dir() -> Path:
    """results/ws4c/ -- every committed WS4c output lands here."""
    return RESULTS_DIR / "ws4c"


def ws4c_cache() -> Path:
    """data/ws4c_cache/ -- WS4c's own cache. NEVER data/ws4_cache/, which
    holds passages.parquet, frozen input to WS4b Tier 0/0b."""
    return DATA_DIR / "ws4c_cache"


# ---------------------------------------------------------------- WS4d ----
# A legitimate r*, or a pre-registered negative.
# Design: results/recall-vs-quality.md
# Spec §7 is the source of truth; none of these is a tuning knob.
WS4D_LABEL_MODEL_A = "claude-sonnet-5"   # labeller A (spec §3.3)
WS4D_LABEL_MODEL_B = "claude-opus-5"     # labeller B; independent, not a verifier
WS4D_N_QUESTIONS = 450                   # every WS4 question is labelled
WS4D_KAPPA_MIN = 0.60                    # E3 floor, BINARY eligible/ineligible kappa
WS4D_SPOTCHECK_N = 20                    # E4 sample size
WS4D_SPOTCHECK_MAX_DISAGREE = 3          # E4 redo threshold, "more than"
WS4D_DECLINE_SHARE_MAX = 0.80            # F1: is the filtered curve still a
                                         # quality curve? Expected to FIRE --
                                         # the §5-labelled analogue is 0.946
WS4D_LADDER_MAX_GAP = 0.035              # L1 spacing through [0.80, 0.97]
WS4D_PHASE1_CAP_USD = 25.00              # est. $15; inside WS4B_BUDGET_USD
WS4D_PHASE3_CAP_USD = 25.00              # est. $17; inside WS4B_BUDGET_USD


def ws4d_dir() -> Path:
    """results/ws4d/ -- every committed WS4d output lands here. WS4d never
    writes into results/ws4/ or results/ws4b/."""
    return RESULTS_DIR / "ws4d"


def ws4d_cache() -> Path:
    """data/ws4d_cache/ -- gitignored; checkpoints and corpus-derived sheets."""
    p = DATA_DIR / "ws4d_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------- WS4e ----
# Does the discriminating share depend on retrieval depth k?
# Design: results/recall-vs-quality.md
# Spec 10 is the source of truth; none of these is a tuning knob. WS4e ADDS
# to WS4/WS4b/WS4d; no constant above may be changed by it -- in particular
# WS4_TOP_K stays 10 and WS4E_K_VALUES is a separate pin (spec 3.3).

WS4E_K_VALUES = (1, 3, 5, 10)        # swept depths; DOWNWARD only (spec 3.1)
WS4E_ARMS = ("sq8_np1", "sq8_np4", "sq8_np48", "sq8_np512")
                                     # spec 3.2; sq8_np1 sets the separability
                                     # ceiling -- arms between 0.79 and 0.99
                                     # recall move it by exactly zero
WS4E_STRATUM = "ws4d_primary_264"    # spec 3.1
WS4E_FLAT_MIN_DIFF = 0.10            # spec 4.1 clause 2
WS4E_REJUDGE_N = 60                  # K3 sample size, inherited from WS4d L2
WS4E_REJUDGE_MIN = 58                # K3 threshold; loosens L2's 60/60 because
                                     # the judge is sampled too (spec 7)
WS4E_TEMPERATURE = None              # spec 3.5: INHERITED UNSET for the main
                                     # sweep (API default 1.0). Recorded as a
                                     # decision, not an oversight -- setting it
                                     # to 0 would vary temperature alongside
                                     # depth, which gate K1 exists to prevent.
                                     # Amendment 1 (2026-09-13): WS4_GEN_MODEL
                                     # has REMOVED temperature/top_p/top_k --
                                     # sending any value is a 400. This is no
                                     # longer merely unset by choice; it is
                                     # unsettable on the pinned model.
# --- RETIRED by Amendment 1 (2026-09-13), spec 3.5 & 10. The temperature-0
# validation cell they described is impossible on WS4_GEN_MODEL (see
# WS4E_TEMPERATURE above) and has been replaced by the WS4E_REPLICATE_* cell
# below. Kept, not deleted, so the withdrawn design stays legible. NEVER
# reused for anything else.
WS4E_T0_CELL_K = 1                   # RETIRED -- spec 5.3 (pre-Amendment 1):
                                     # the withdrawn validation cell's depth
WS4E_T0_NULL_MAX = 0.02              # RETIRED -- spec 5.3 V1 / K6 (pre-
                                     # Amendment 1)
# --- Amendment 1 replacement cell (spec 5.3, 10): re-run WS4E_REPLICATE_ARM
# at WS4E_REPLICATE_K a second, independent time and measure the disagreement
# rate R between the two runs -- within-prompt generator variance, directly,
# with no temperature parameter involved.
WS4E_REPLICATE_ARM = "sq8_np512"     # spec 5.3: the arm re-run a second time
WS4E_REPLICATE_K = 1                 # spec 5.3: chosen for the largest
                                     # identical-passage subset (166 of 264)
WS4E_REPLICATE_MIN_N = 5             # spec 5.3 V1 / K6 floor: a CI excluding
                                     # zero on fewer than 5 disagreeing
                                     # questions is one flaky row, not a
                                     # measurement
WS4E_CAP_USD = 35.00                 # est. $22.65 (Amendment 1, was $26.69);
                                     # inside WS4B_BUDGET_USD. Cap NOT lowered
                                     # with the estimate (spec 11) and NEVER
                                     # raised mid-run.


def ws4e_dir() -> Path:
    """results/ws4e/ -- every committed WS4e output lands here. WS4e never
    writes into results/ws4/, results/ws4b/ or results/ws4d/."""
    return RESULTS_DIR / "ws4e"


def ws4e_cache() -> Path:
    """data/ws4e_cache/ -- gitignored by data/*; per-cell run checkpoints."""
    p = DATA_DIR / "ws4e_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------- WS5 ----
# The cost model (C2/C9/C13/C15). Every price and every assumption lives in
# ONE cell of notebooks/05_cost_model.ipynb and is written to
# results/ws5/pricing.csv with source URL + fetch date. Nothing is pinned
# here because nothing here is a measurement: the inputs are WS2/WS3
# footprints, WS4's knee, WS6a/WS6b token counts, and fetched prices.


def ws5_dir() -> Path:
    """results/ws5/ -- every committed WS5 output lands here."""
    return RESULTS_DIR / "ws5"


# --------------------------------------------------------------- WS6c ----
# Top-k sensitivity on the frozen WS6a corpus, plus an uncontaminated corpus.
# Every WS6A_* constant above is frozen; WS6c only adds.

WS6C_TOPK = 3                  # bound search_code limit; default is MEASURED, see
                               # results/ws6c/topk_provenance.txt
WS6C_BOOT_N = 10_000
WS6C_BOOT_SEED = 42
WS6C_ALPHA = 0.05
# Raised from 40.00 to 45.00 mid-Phase-3, on explicit user authorisation.
# Measured per-pair cost on the uncontaminated corpus ran 1.6-1.9x the
# plan's estimate ($0.125/pair on the tool-using arms against WS6a's
# $0.066-0.078), so the pre-registered 160-pair design did not fit the
# original cap. The alternative was dropping an arm with 65 of 160 rows
# already visible -- a post-hoc design change made with data in view,
# which is exactly what pre-registration exists to prevent.
WS6C_BUDGET_USD = 45.00
WS6C_GATE_QUESTIONS = 25
WS6C_N_QUESTIONS = 40

# Ranked candidate corpora. Selection is decided by the Phase 2 gate, not by
# this order; the order only sets which S1 candidate is taken first.
WS6C_CANDIDATES = (
    {"slug": "agentic_hil", "repo": "agentic-hil/agentic-hil",
     "url": "https://github.com/agentic-hil/agentic-hil.git", "license": "Apache-2.0"},
    {"slug": "kicad_mcp_pro", "repo": "oaslananka/kicad-mcp-pro",
     "url": "https://github.com/oaslananka/kicad-mcp-pro.git", "license": "MIT"},
    {"slug": "mixle", "repo": "gmboquet/mixle",
     "url": "https://github.com/gmboquet/mixle.git", "license": "MIT"},
)


def ws6c_dir() -> Path:
    d = RESULTS_DIR / "ws6c"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ws6c_checkout(slug: str) -> Path:
    return DATA_DIR / f"ws6c_{slug}"


# Measured by scripts/code-retrieval/select_ws6c_corpus.py on 2026-09-08; copied verbatim
# out of results/ws6c/corpus_stats.csv.
WS6C_COMMITS = {
    "agentic_hil": "837f39813e1421c7b8efc6e621c1368507f2dbb5",
    "kicad_mcp_pro": "f36c7a2cbfd66b0812209305058e4f5634dd924d",
    "mixle": "7322ca27a2f9e959c628e884416415802d98edd2",
}


# ----------------------------------------------------------------- WS8 ----
# Non-code retrieval: does the index win when the answer spans documents?
# Design: results/non-code-retrieval.md
# Every WS6A_*/WS6B_*/WS6C_* constant above is frozen; WS8 only adds.

# Strictly after the agent model's ~2026-05 training cutoff. Widening this
# window to recover chains is forbidden by spec section 7 Phase 0 -- it would
# reach back toward the cutoff and reintroduce contamination.
WS8_WINDOW_START = "2026-06-01"
WS8_WINDOW_END = "2026-09-01"

# Pinned by the repo owner on 2026-09-10, BEFORE any fetch. Chosen for issue
# volume and cross-referencing culture; the subject-matter spread (web
# framework / orchestration / systems language) keeps the corpus from being one
# vocabulary. fastapi also bridges to WS6a/WS6c, which benchmarked its CODE.
# NOTE these are heavily-trained-on projects: the post-cutoff window makes the
# issue TEXT unseen, but the model still knows these codebases well (WS6a
# measured a 0.625 parametric floor on fastapi's code). The Phase 1 gate is
# therefore a real experiment here, not a formality -- see spec section 4.
WS8_REPOS: tuple[str, ...] = (
    "fastapi/fastapi",
    "kubernetes/kubernetes",
    "rust-lang/rust",
)

# Deterministic fetch. Without an explicit sort the GitHub default could change
# under us and pagination is not guaranteed stable, so the corpus would not be
# reproducible -- which is the whole point of pinning it.
WS8_FETCH_SORT = "created"
WS8_FETCH_DIRECTION = "asc"

# Per-repo cap, oldest-first within the window. RAISED 1000 -> 8000 on
# 2026-09-11 after the first fetch proved the cap, not the window, was defining
# the corpus: kubernetes stopped at 2026-07-10 and rust-lang at 2026-06-21,
# against a pre-registered window running to 2026-09-01. The gate then failed on
# ~30% of the intended corpus. This is a SAFETY VALVE, not a sampling rule -- at
# 8000 it should not bind, and the created_at window defines the corpus.
WS8_MAX_THREADS_PER_REPO = 8000

# Comments are fetched from the REPO-level endpoint (/repos/{repo}/issues/comments)
# in pages, not one request per thread. The per-thread route needed ~1 call per
# thread (~10,000 calls at full-window size, over two hours against a 5,000/hour
# limit) and silently truncated at 100 comments per thread with no page loop --
# five rust-lang threads hit exactly 100. Lost comments are lost `#N` references,
# hence lost graph edges: that corrupts the very link structure WS8 measures.
# Known gap, disclosed: this endpoint returns issue comments (including those on
# PRs) but NOT inline PR review comments. Those were absent before too.
WS8_COMMENTS_BULK = True

# The /issues endpoint returns pull requests as well. They are KEPT, deliberately:
# "issue -> the PR that fixed it -> follow-up issue" is the canonical multi-hop
# chain this workstream exists to measure, and dropping PRs would remove the
# most common real 2- and 3-hop structures. Recorded as a decision rather than
# left as an accident of which endpoint was called.
WS8_INCLUDE_PULL_REQUESTS = True

WS8_AGENT_MODEL = "claude-sonnet-5"
WS8_JUDGE_MODEL = "claude-opus-5"

# Structurally parallel to agent_harness.SYSTEM_PROMPT (same shape, same
# closing "say so plainly rather than guessing" clause, same deliberate
# silence about tools and retrieval strategy -- every WS8 arm shares this
# string, so it must not hint at one). agent_harness.SYSTEM_PROMPT itself is
# frozen and wrong for WS8: it names "a Python codebase" and asks for a
# "repository-relative file path" and "functions or classes", none of which
# exist in WS8's corpus of GitHub issue/PR threads (kubernetes is Go,
# rust-lang is Rust; the evidence is markdown documents, not source files).
# Passed explicitly via run_arm(..., system_prompt=WS8_SYSTEM_PROMPT) so
# every WS8 arm -- parametric, agentic, indexed -- gets the same corrected
# framing and the arms differ in exactly one variable (tools), never two.
WS8_SYSTEM_PROMPT = (
    "You are answering questions about GitHub issue and pull-request "
    "threads.\n\n"
    "Answer the question directly and concisely. Always cite the specific "
    "document filename or filenames your answer relies on (for example "
    "`owner__repo__1234.md`).\n\n"
    "If you cannot determine the answer, say so plainly rather than guessing."
)

WS8_EMBED_PROVIDER = "openai"
WS8_EMBED_MODEL = "text-embedding-3-small"   # same as WS6a/WS6b, for comparability
WS8_EMBED_DIM = 1536
WS8_MILVUS_URI = "http://localhost:19530"
WS8_COLLECTION = "ws8_issues"

# WS8's indexed arm is FIRST-PARTY, so there is no vendor default to measure.
# 10 is the claude-context default WS6c measured (results/ws6c/topk_provenance.txt),
# so this arm is configured like the one WS6a and WS6c benchmarked. Not swept.
WS8_TOP_K = 10
WS8_CHUNK_CHARS = 1_800
WS8_CHUNK_OVERLAP = 200

# WS8_ARMS documents the design's three arms: indexed, agentic, parametric.
# Phase 1's parametric gate rows are the parametric arm, reused verbatim rather
# than re-run in Phase 3 — Phase 3 itself runs only indexed and agentic.
WS8_ARMS = ("indexed", "agentic", "parametric")
WS8_TURN_CAP = 15                      # identical for both tool-using arms
WS8_SINGLE_TURN_ARMS = ("parametric",)
# RAISED 55.00 -> 70.00 on 2026-09-11, BEFORE the Phase 3 run, with 10 of 150
# pairs complete and every decision branch already fixed in writing. The timing
# is the point. WS6c raised its governor 40 -> 45 MID-RUN with 65 of 160 pairs
# visible -- a budget decision taken with results in view, which is what
# pre-registration exists to prevent (see results/code-retrieval.md and README.md).
# Here the stratified smoke put the full run at a $38.19 point estimate
# against a $20-$55+ plausible range, 59% of the estimate resting on a single
# n=1 hop2 indexed outlier ($1.498216, 10/15 turns, judged incorrect). Raising
# now costs nothing if the run lands at $38 -- a governor is a cap, not a budget --
# and it removes the need to make the same call mid-flight at ~85% completion.
WS8_BUDGET_USD = 70.00

# Sized by the power analysis in spec section 3: n=30 powers a ~25,000-token
# effect at 80%; hop2 is deliberately NOT powered and is reported as-is.
WS8_STRATA = {"hop1": 30, "hop2": 15, "hop3plus": 30}
WS8_N_QUESTIONS = 75

# WITHDRAWN 2026-09-11 -- KEPT AS A PIN, BUT NOTHING READS IT.
# These were the S/M/L corpus-size targets for a sweep that was withdrawn: WS8
# measures ONE corpus at its full measured size, and no script, notebook or
# results CSV consumes this dict (`grep -rn WS8_SIZES` finds only this line and
# the pin test). It is left in place because this repo's hard rule forbids
# changing a pin without being asked, and because a reader of the public repo
# comparing the design doc's three sizes against the code should find the
# constant AND the reason it is inert -- not conclude that a three-size sweep
# was run and its results dropped. If the sweep is ever revived, the published
# x-axis is the MEASURED count_tokens total, never these targets.
WS8_SIZES = {"s": 100_000, "m": 400_000, "l": 1_200_000}

# The gate must be exactly zero. WS6a's 0.625 floor is what made its other
# arms uninterpretable; a "small" floor is not harmless.
WS8_GATE_PASS = 0.0

WS8_BOT_DENYLIST = frozenset({
    "dependabot", "dependabot[bot]", "github-actions[bot]", "codecov[bot]",
    "renovate[bot]", "stale[bot]", "mergify[bot]", "sonarcloud[bot]",
    "netlify[bot]", "vercel[bot]", "pre-commit-ci[bot]", "allcontributors[bot]",
})

WS8_CORPUS_DIR = DATA_DIR / "ws8_corpus"


def ws8_dir() -> Path:
    """results/ws8/ -- every committed WS8 output lands here."""
    d = RESULTS_DIR / "ws8"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------- WS9 ----
# The unseen-corpus cost curve: agentic search vs claude-context indexed
# retrieval, priced across corpus size on the corpus WS6c's Phase 2 gate
# already cleared. Feeds C2/C9/C13/C15 through results/ws5/.
# Pre-registration:
#   docs/superpowers/specs/2026-09-15-ws9-unseen-corpus-cost-curve-design.md
# Every WS6A_*/WS6B_*/WS6C_*/WS8_* constant above is frozen; WS9 only adds.

WS9_SLUG = "agentic_hil"
# NOT re-gated. The non-memorisation gate is a property of the repository and
# the model, not of how much of the repository is indexed, and re-running it
# would spend money to re-derive a committed 0.000.

WS9_SCOPES = ("S", "M", "L")

# Nested size tiers, decided from the tree before any WS9 row existed (spec
# section 2.2). Measured at the pin: S 49 files / 741,343 tiktoken, M 172 /
# 1,352,243, L 342 / 2,770,789 -- log spacing 1.82x then 2.05x.
#
# M is an EXCLUDE tier, unlike WS6A_SUBSETS' include-only prefix tuples, for
# two reasons. tests/ is 170 files and 51.2% of this corpus's tokens in ONE
# directory, so splitting it out is the only division of this tree that gives
# even log spacing -- the literal WS6A_SUBSETS analogue (src/ ->
# src/+tools/+tests/ -> all) spaces 3.06x then 1.22x, which puts M and L almost
# on top of each other, the "badly conditioned" case
# scripts/code-retrieval/scale_ws6a.py's own guard warns about. And "all but
# tests/" cannot be written as an include list without naming every
# root-level file, which would be a brittle spelling of a simple tier.
WS9_SUBSETS = {
    "S": {"include": ("src/",), "exclude": ()},
    "M": {"include": (),        "exclude": ("tests/",)},
    "L": {"include": (),        "exclude": ()},
}

# L is REUSED from results/ws6c/runs.csv, never re-run.
WS9_SWEEP_SIZES = ("S", "M")

# No stuffed arm at any size: this corpus measures 4,307,363 count_tokens
# against a 960,000-token prefix budget, 4.4x over, so the arm is INFEASIBLE
# rather than omitted. parametric enters at L only -- it has no corpus access,
# so it is size-independent by construction.
WS9_ARMS = ("agentic", "indexed", f"indexed_topk{WS6C_TOPK}")

# Eligibility is evaluated ONCE, at the smallest scope, and the resulting panel
# runs at every size -- so no question enters or leaves the panel as the corpus
# grows. A curve whose points are drawn over different question populations has
# a shape that is partly a composition artifact; this project has caught that
# error twice already (results/ws4e/composition_check.csv). 37 of WS6c's 40
# qualify; q09 (examples/), q30 (tools/) and q31 (tests/) do not.
WS9_ELIGIBILITY_SCOPE = "S"
WS9_PANEL_N = 37

# Hard cap across every WS9 phase. Basis $26.83 = 37 questions x 3 arms x 2
# sizes at WS6c's MEASURED per-run cost on this corpus, agent plus judge
# (agentic $0.1489, indexed $0.1026, indexed_topk3 $0.1110). Headroom 2.05x,
# against the 1.6-1.9x by which WS6c's own estimate came in low. The smoke
# check shares the sweep's checkpoint, so it is re-used rather than re-billed
# and is not a separate line.
WS9_BUDGET_USD = 55.00
WS9_SMOKE_QUESTIONS = 2

# Matched-size contamination test (spec section 3.2). Two scopes are MATCHED
# iff their measured count_tokens are within this factor of each other.
# Pairing is by measured tokens, never by file count.
WS9_MATCH_BAND = 2.0


def ws9_subset_tree(scope: str) -> Path:
    """Persisted tree for one scope.

    claude-context keys an index by ABSOLUTE codebase path, so the directory
    the index is built over must be the same directory the agentic arm greps.
    Indexing one path and searching another measures an EMPTY index and
    reports nothing wrong.
    """
    return DATA_DIR / f"ws9_subset_{scope.lower()}"


def ws9_cache_dir() -> Path:
    """Checkpoints and stderr logs. The WS9 cost governor sums exactly this
    directory -- see benchlib.agent_harness.WS9_SPEND_CHECKPOINT_GLOBS."""
    d = DATA_DIR / "ws9_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d
