#!/usr/bin/env bash
# WS9 overnight driver: smoke -> sweep -> bootstrap -> matched-size -> build ->
# papermill -> tests -> checksums. Logs to data/ws9_cache/overnight.log with
# timestamps; writes the step it is on to data/ws9_cache/overnight.status.
#
# Start (from the repo root, in a shell that will be closed):
#   caffeinate -i nohup scripts/code-retrieval/run_ws9_overnight.sh >/dev/null 2>&1 &
# Watch:
#   tail -f data/ws9_cache/overnight.log
# Resume after a failure: fix the cause, then start it again -- every step
# short-circuits on what is already done (checkpoints, caches).
#
# It never commits. The morning checklist in the plan does that.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
LOG=data/ws9_cache/overnight.log
STATUS=data/ws9_cache/overnight.status
LOCK=data/ws9_cache/overnight.lock
mkdir -p data/ws9_cache
exec > >(while IFS= read -r line; do printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line"; done >> "$LOG") 2>&1

step() { echo "=== STEP $1 ==="; echo "RUNNING $1 pid $$ $(date '+%H:%M:%S')" > "$STATUS"; }
fail() { echo "FAILED $1 $(date '+%H:%M:%S')" > "$STATUS"; echo "!!! FAILED at: $1"; exit 1; }

echo "=== WS9 overnight driver starting, pid $$ ==="
step preflight

# One instance only: two drivers, or a driver next to a hand-started sweep,
# would bill the same pairs twice and duplicate the checkpoint rows.
mkdir "$LOCK" 2>/dev/null || fail "preflight: $LOCK exists -- another driver is running; if none is, rmdir it"
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
pgrep -f 'scripts/code-retrieval/scale_ws9\.py' >/dev/null \
  && fail "preflight: a scale_ws9.py is already running (pid $(pgrep -f 'scripts/code-retrieval/scale_ws9\.py' | head -1)); wait for it or stop it first"

# Credentials. NOTHING in this repo reads .env automatically.
if [ -f .env ]; then set -a; . ./.env; set +a; fi
for v in ANTHROPIC_API_KEY OPENAI_API_KEY MILVUS_ADDRESS MILVUS_TOKEN; do
  [ -n "${!v:-}" ] || fail "preflight: $v is not set (source .env)"
done
command -v rg >/dev/null || fail "preflight: ripgrep (rg) not on PATH; the agentic arm needs it"
command -v caffeinate >/dev/null && pgrep -x caffeinate >/dev/null || echo "WARNING: caffeinate not running; the machine may sleep"

# Pre-conditions from the attended phase.
for f in results/ws6c/scaling_corpus_stats.csv \
         data/ws9_subset_s data/ws9_subset_m \
         scripts/code-retrieval/scale_ws9.py \
         scripts/code-retrieval/bootstrap_ws9.py \
         scripts/code-retrieval/matched_size_ws9.py \
         tests/test_ws9_cost_model.py \
         data/ws9_cache/ws5_before/break_even.csv; do
  [ -e "$f" ] || fail "preflight: $f missing -- the attended phase is not done"
done
grep -q code_unseen scripts/cost-model/build_ws5_notebook.py \
  || fail "preflight: Task 8 builder edits not applied"
$PY - <<'EOF' || fail "preflight: index_cost.csv lacks verified S/M/L rows (Task 4)"
import pandas as pd
d = pd.read_csv("results/ws6c/index_cost.csv").set_index("scope")
assert {"S", "M", "L"} <= set(d.index), d.index.tolist()
assert d.loc["S", "chunk_count"] < d.loc["M", "chunk_count"] < d.loc["L", "chunk_count"]
assert int(d.loc["L", "chunk_count"]) == 7441
EOF
$PY - <<'EOF' || fail "preflight: the Milvus collections for S and M are not both present"
import os
from pymilvus import MilvusClient
c = MilvusClient(uri=os.environ["MILVUS_ADDRESS"], token=os.environ["MILVUS_TOKEN"])
names = c.list_collections()
for p in ("ws9_s", "ws9_m"):
    hits = [n for n in names if p in n]
    assert len(hits) == 1, (p, hits)
print("collections present:", names)
EOF
$PY -c "import anthropic, os; anthropic.Anthropic().messages.count_tokens(model='claude-sonnet-5', messages=[{'role':'user','content':'ping'}]); print('anthropic OK')" \
  || fail "preflight: Anthropic API not reachable"
# The suite is the last pre-flight: it proves the pure half (pins, panel,
# governor globs) before a dollar is spent. -x so a broken test is the
# first line of the failure, not the four-hundredth.
# test_ws9_cost_model.py is excluded here -- it can only pass after this driver's own papermill step; the post-papermill `tests` step runs the full suite including it.
$PY -m pytest tests/ -q -x --ignore=tests/test_ws9_cost_model.py || fail "preflight: test suite (WS9 cost-model tests are excluded here; the post-papermill tests step runs them)"

step smoke;      $PY scripts/code-retrieval/scale_ws9.py --smoke || fail "smoke (stop rule 5: read the log; do not change the design)"
step sweep;      $PY scripts/code-retrieval/scale_ws9.py         || fail "sweep (governor, an exhausted session retry, or a completeness gate: read the log; re-running resumes)"
step bootstrap;  $PY scripts/code-retrieval/bootstrap_ws9.py     || fail bootstrap
step matched;    $PY scripts/code-retrieval/matched_size_ws9.py  || fail matched
step build;      $PY scripts/cost-model/build_ws5_notebook.py    || fail build
step papermill;  .venv/bin/papermill notebooks/05_cost_model.ipynb notebooks/05_cost_model.ipynb \
  || fail "papermill (if the log says POSTDICTION GATE FAILED: stop rule 2, do NOT widen GATE_TOL)"
step tests;      $PY -m pytest tests/ -q                          || fail tests
step checksums;  $PY scripts/fixtures/verify_checksums.py         || fail checksums

echo "DONE $(date '+%H:%M:%S')" > "$STATUS"
echo "=== ALL STEPS COMPLETE ==="
# Guarded: an unreadable checkpoint must not pass for a silent empty total.
billed=$($PY -c "from benchlib import agent_harness as a; from benchlib.config import DATA_DIR; print(f'{a.total_spent_across_checkpoints(DATA_DIR, globs=a.WS9_SPEND_CHECKPOINT_GLOBS):.2f}')") || fail "billed total"
echo "WS9 billed: $billed of 55.00"
