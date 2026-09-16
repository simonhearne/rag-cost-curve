Data fixtures live here and are **not committed** (tens of GB). Everything is
re-derivable from pinned upstream sources: run `scripts/fixtures/download_corpus.py`,
`scripts/fixtures/download_nq.py`, then notebook `01_datasets.ipynb`. `MANIFEST.json`
(committed) records the sha256 of every file so a third party can verify they
rebuilt the exact same fixtures.

### Exception: the agent-memory synthetic corpus **is** committed

`data/ws6b_corpus/` is the one corpus this repo redistributes, and the
exception is deliberate. The no-corpus-data rule above exists because the
embeddings dataset carries no licence tag. That cannot apply to text produced
by `scripts/code-retrieval/generate_ws6b_corpus.py` at seed 42 — there is no
upstream rights holder. Committing it alongside the generator that re-derives
it serves the rule's deeper purpose, which is reproducibility: a reader diffs
bytes instead of trusting a generator.

Every file is sha256'd into `data/MANIFEST.json`, and
`tests/code_retrieval/test_memory.py::test_generator_is_byte_exact_across_runs`
asserts that regenerating reproduces those bytes. Exception granted by
Simon Hearne, 2026-08-18.
