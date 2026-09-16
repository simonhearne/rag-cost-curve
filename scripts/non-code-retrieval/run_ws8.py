#!/usr/bin/env python
"""WS8 Phase 3: indexed + agentic over the same pinned tree.

TWO arms, not three: Phase 1's gate rows ARE the parametric arm and are
reused verbatim (spec section 7) -- re-running it would spend money on a
number we already have and would leave WS6c's "corroboration, not
independent replication" ambiguity. This script runs only `indexed` and
`agentic`, 75 questions each = 150 pairs at full scale.

Corrections applied here (one known defect found in review, 2026-09-11,
R21):

R21 -- the brief handed `mcp_tools=[(search_schema, search_call)]` straight
    to execute_run. run_arm's "indexed" branch does `tools = list(mcp_tools
    or [])` and passes that list DIRECTLY to
    `client.beta.messages.tool_runner`, which needs RUNNABLE `beta_tool`-
    wrapped callables -- the shape `agent_harness.claude_context_tools`
    produces, not a bare (schema, callable) tuple. Fixed in build_mcp_tools()
    below: `beta_tool(call, name=..., description=..., input_schema=...)`.

R17 -- WS8_SYSTEM_PROMPT, not agent_harness.SYSTEM_PROMPT's frozen "Python
    codebase" framing, for every arm. Phase 1's parametric rows (reused
    verbatim as the parametric arm here) already ran under it
    (scripts/non-code-retrieval/gate_ws8.py); running Phase 3's two arms under a different
    prompt would make the arms differ in two variables (tools AND prompt)
    instead of one. agent_harness.execute_run did not previously accept a
    system_prompt to forward to run_arm -- gate_ws8.py sidesteps this by
    calling run_arm directly in its own single-arm loop -- so execute_run
    was extended with an optional system_prompt passthrough (same
    default-None, resolved-inside-run_arm pattern R17 already established
    on run_arm itself). See benchlib/agent_harness.py.

R19 -- extended/adaptive thinking is ON BY DEFAULT for claude-sonnet-5 (the
    WS8 agent model) and counts against max_tokens. Phase 1 lost ws8_q065 to
    exactly this at the 4,096 default: the whole budget went to an unbilled
    thinking block with no text emitted. Tool-using arms produce longer
    final answers than Phase 1's single-turn parametric arm, so the risk is
    higher here, not lower. RUN_MAX_TOKENS below is passed identically to
    BOTH arms (execute_run's new max_tokens passthrough, same mechanism as
    system_prompt) so the arms never differ in a second variable besides the
    tools list. Chosen as 16384: 4x headroom over the 4096 default that
    failed, and the exact value already proven sufficient for the q065 retry
    (scripts/non-code-retrieval/retry_ws8_q065.py, which ran under real, billed API load).

Truncation handling (Task 6, standing). Originally a gap here too:
execute_run did not special-case a max_tokens stop with an EMPTY answer the
way gate_ws8.py's own hand-rolled loop does -- it would have silently
checkpointed such a row as a normal completed pair (judge_answer
short-circuits an empty answer to judge_correct=False WITHOUT even calling
the judge, so this reads as a confident, free, wrong answer rather than an
unmeasured question -- exactly Phase 1's ws8_q065 failure, this time with
no signal that anything went wrong). Found and fixed the same way as the
judge-exception gap: execute_run now treats a truncated-empty result
identically to an arm error -- NOT checkpointed, retried on resume -- so
this can no longer happen structurally. The post-run scan below is a
defensive, redundant check (also catches a max_tokens stop with a
NON-empty but possibly cut-off answer, which the guard deliberately leaves
alone since that is a milder, still-scoreable case, not an unmeasured
question).

R18/R20 (standing, spend integrity) -- the governor is seeded from
    ws8.total_spent_across_checkpoints (every WS8 checkpoint, live AND
    archived, under ws8_cache/*.jsonl), never a hand-picked subset of files.
    Every billed invocation of this script is appended to the git-tracked
    results/ws8/spend_ledger.csv; scripts/fixtures/stamp_ledger_commit.py fills in
    committed_in afterwards, same as Phase 1/2.

Judge-exception spend pattern -- agent_harness.execute_run previously
    discarded an already-billed agent call's cost on a judge exception
    (skipped the checkpoint entirely, charged nothing). Fixed generically in
    execute_run itself: a judge exception now appends a spend-only
    checkpoint line under `<arm>_judge_error` and charges the governor
    before continuing, matching the pattern gate_ws8.py already used in its
    own hand-rolled loop. See benchlib/agent_harness.py.

Checkpoint naming -- data/ws8_cache/run_checkpoint.jsonl. Distinct from
    Phase 1's `gate_checkpoint*.jsonl` (gate_ws8.py's PHASE1_CHECKPOINT_GLOB
    scopes on that prefix) and Phase 2's `index_checkpoint.jsonl`; it still
    matches ws8.WS8_CHECKPOINT_GLOB ("ws8_cache/*.jsonl"), so
    total_spent_across_checkpoints sweeps it automatically.

expected_files/expected_symbols mapping -- agent_harness.execute_run's
    programmatic scorer reads row["expected_files"] and
    row["expected_symbols"] (WS6a's column names). WS8's questions.csv has
    "expected_documents" instead (";"-separated, same format) and no symbol
    analogue -- gate_ws8.py passes expected_symbols="" explicitly for the
    same reason (its own R2 note). Mapped here via a column assignment
    rather than touching the committed questions.csv.

THE SMOKE GATE: set WS8_SMOKE=<n> to run only the first n questions (by
qid, deterministic) across both arms and stop, or WS8_QIDS=<qid,qid,...>
for an explicit, targeted set (e.g. a stratified pilot picked by the
pre-registered seeded order rather than by eye) -- this script does NOT
loop past either. Running the full 150 is a separate, unguarded invocation
with both unset, authorised only after reviewing the smoke numbers. Smoke
reporting is cumulative: it reads every pair checkpointed so far (across
however many WS8_SMOKE/WS8_QIDS invocations have run) and breaks cost down
per (stratum, arm), so a second targeted smoke adds to the first rather
than replacing it.

Run (smoke, first n):     WS8_SMOKE=2 python scripts/non-code-retrieval/run_ws8.py
Run (smoke, targeted):    WS8_QIDS=ws8_q060,ws8_q075,ws8_q037 python scripts/non-code-retrieval/run_ws8.py
Run (full):                python scripts/non-code-retrieval/run_ws8.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
from anthropic import beta_tool
from openai import OpenAI
from pymilvus import MilvusClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest, agent_harness
from benchlib.non_code_retrieval import bench as ws8
from benchlib.non_code_retrieval import index as ws8_index
from benchlib.code_retrieval import bench as ws6a
from benchlib.config import (DATA_DIR, SEED, WS8_BUDGET_USD, WS8_COLLECTION,
                             WS8_CORPUS_DIR, WS8_EMBED_MODEL, WS8_MILVUS_URI,
                             WS8_SYSTEM_PROMPT, WS8_TOP_K, ws6a_dir, ws8_dir)

ARMS = ("indexed", "agentic")

CHECKPOINT = DATA_DIR / "ws8_cache" / "run_checkpoint.jsonl"

# R19 -- see module docstring. Same value for both arms.
RUN_MAX_TOKENS = 16384

# ws8.append_spend_ledger / ws8.warn_on_ledger_mismatch were four verbatim
# copies (this file, gate_ws8.py, index_ws8.py, retry_ws8_q065.py) until
# 2026-09-11; see benchlib/non_code_retrieval/bench.py for the schema and
# the incident that moved them there.


def scope_counts(checkpoint_path, target_qids) -> dict:
    """What the checkpoint currently holds for this invocation's scope.

    Read back from the checkpoint rather than accumulated in memory, so it
    answers the same way whether the run finished or died: the checkpoint is
    the record of spend, and the ledger note must describe THAT, not whatever
    a half-built in-process list happened to contain when the exception hit.

    `pairs` counts every checkpointed pair in scope, including ones written by
    an EARLIER invocation -- the same semantics the ledger's existing rows
    already carry (the final Phase 3 row reads pairs_checkpointed=150 for an
    invocation that added 4), so hoisting this changes no committed number.
    """
    ckpt = (agent_harness.load_checkpoint(checkpoint_path)
            if Path(checkpoint_path).exists() else {})
    pairs = [r for (qid, arm), r in ckpt.items()
             if arm in ARMS and qid in target_qids]
    return {
        "ckpt": ckpt,
        "pairs": pairs,
        "truncated": [r for r in pairs
                      if r.get("stop_reason") == "max_tokens"
                      and not (r.get("answer") or "").strip()],
        "capped": [r for r in pairs if r.get("hit_turn_cap")],
    }


def build_mcp_tools(mc, embed_fn):
    """R21: wrap ws8_index.make_search_tool's (schema, callable) pair as a
    RUNNABLE beta_tool, the shape client.beta.messages.tool_runner requires.
    """
    search_schema, search_call = ws8_index.make_search_tool(
        mc, WS8_COLLECTION, embed_fn, WS8_TOP_K)
    return [beta_tool(search_call, name=search_schema["name"],
                      description=search_schema["description"],
                      input_schema=search_schema["input_schema"])]


def main() -> None:
    out = ws8_dir()
    smoke_n = int(os.environ.get("WS8_SMOKE", "0") or 0)
    # WS8_QIDS: explicit comma-separated qid list, for a targeted (e.g.
    # stratified) smoke rather than the first-n-by-qid pilot WS8_SMOKE gives.
    # Selection of WHICH qids to pass is the caller's responsibility and
    # should be made by a deterministic, pre-registered rule (e.g. the same
    # seeded benchlib.code_retrieval.bench.interleaved_order the real run uses), not by eye.
    qids_env = os.environ.get("WS8_QIDS", "").strip()
    smoke_qids = [q.strip() for q in qids_env.split(",") if q.strip()]

    qs_all = pd.read_csv(out / "questions.csv")
    qs_all = qs_all[~qs_all["excluded"].astype(bool)].reset_index(drop=True)

    gate = pd.read_csv(out / "parametric_gate.csv")
    if gate.loc[0, "branch"] != "G1":
        sys.exit("Phase 1 did not return G1; Phase 3 must not run.")

    # expected_files/expected_symbols mapping -- see module docstring.
    qs_all = qs_all.assign(expected_files=qs_all["expected_documents"],
                           expected_symbols="")
    qs = qs_all

    is_smoke = bool(smoke_n) or bool(smoke_qids)
    if smoke_qids:
        qs = qs_all[qs_all["qid"].isin(smoke_qids)].reset_index(drop=True)
        missing = set(smoke_qids) - set(qs["qid"])
        if missing:
            sys.exit(f"WS8_QIDS names qid(s) not found in questions.csv "
                     f"(or excluded): {sorted(missing)}")
        print(f"TARGETED SMOKE: {len(qs)} question(s) x {len(ARMS)} arm(s) "
              f"= {len(qs) * len(ARMS)} run(s), qids={smoke_qids}. Full run "
              "is a SEPARATE, unguarded invocation with WS8_QIDS/WS8_SMOKE "
              "unset.")
    elif smoke_n:
        qs = qs_all.sort_values("qid").head(smoke_n).reset_index(drop=True)
        print(f"SMOKE MODE: {smoke_n} question(s) x {len(ARMS)} arm(s) = "
              f"{smoke_n * len(ARMS)} run(s). Full run is a SEPARATE, "
              "unguarded invocation with WS8_SMOKE unset.")

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    # R18: seed from EVERY WS8 checkpoint, live and archived (gate + index +
    # this script's own), never a hand-picked subset -- see
    # ws8.total_spent_across_checkpoints's docstring for the incident this
    # prevents.
    spent_before = ws8.total_spent_across_checkpoints(DATA_DIR)
    governor = ws6a.CostGovernor(WS8_BUDGET_USD, spent=spent_before)
    print(f"governor: ${spent_before:.6f} already billed of "
          f"${WS8_BUDGET_USD:.2f}")

    mc = MilvusClient(uri=WS8_MILVUS_URI)
    oai = OpenAI()

    def embed(text: str):
        return oai.embeddings.create(model=WS8_EMBED_MODEL,
                                     input=[text]).data[0].embedding

    mcp_tools = build_mcp_tools(mc, embed)

    client = agent_harness.make_client()
    prices = ws6a.load_pricing(ws6a_dir() / "pricing.csv", regime="standard")

    ledger_path = out / "spend_ledger.csv"
    scope = (f"qids={smoke_qids}" if smoke_qids else
             f"smoke_n={smoke_n}" if smoke_n else "full")
    # A list, not a bool, so the closure can flip it without `nonlocal`. One
    # shot: the finally below is the ONLY caller on the normal path, but a
    # future edit that also calls write_ledger() explicitly must not be able
    # to append the same invocation's spend twice.
    ledger_written: list[bool] = []

    def write_ledger(error: BaseException | None = None) -> None:
        """Append this invocation's spend row. Called from a `finally`.

        THE POINT OF THE finally: an invocation of this script checkpointed
        ws8_q028/indexed ($0.176158) and ws8_q011/agentic ($0.074400) and then
        died before it reached the append that used to sit at the end of
        main(). The $0.250558 was really billed and really recorded in the
        checkpoint, and it has no row in the committed ledger to this day. The
        money is gone the moment the API answers; the only thing an exception
        can still destroy is the record of it.

        Every figure is re-derived from the governor and the checkpoint at
        write time, so a row is produced whether the body finished, raised, or
        was interrupted.
        """
        if ledger_written:
            return
        ledger_written.append(True)
        invocation_spend = round(governor.spent - spent_before, 6)
        facts = scope_counts(CHECKPOINT, set(qs["qid"]))
        note = (f"scope={scope} pairs_checkpointed={len(facts['pairs'])} "
                f"truncated_empty={len(facts['truncated'])} "
                f"turn_capped={len(facts['capped'])}")
        if error is not None:
            # Named in the ledger itself: a row whose spend is real but whose
            # run did not finish must not read like a clean invocation.
            detail = " ".join(str(error).split())[:200]
            note += (f" ABORTED before completion: "
                     f"{type(error).__name__}: {detail}")
        ws8.append_spend_ledger(
            ledger_path,
            run_label=f"run_ws8_phase3{'_smoke' if is_smoke else ''}",
            spend_usd=invocation_spend, note=note)
        manifest.record(ledger_path)
        print(f"\nappended ${invocation_spend:.6f} to {ledger_path}")
        print(f"cumulative WS8 spend: ${governor.spent:.6f} of "
              f"${WS8_BUDGET_USD:.2f}")
        # Phase-end reconciliation, AFTER the row is written. Warns, never
        # raises -- see ws8.warn_on_ledger_mismatch.
        ws8.warn_on_ledger_mismatch(ledger_path, DATA_DIR)

    try:
        agent_harness.execute_run(
            client, qs, ARMS,
            root=WS8_CORPUS_DIR / "docs",
            prefix_text=None,
            mcp_tools=mcp_tools,
            prices=prices,
            checkpoint_path=CHECKPOINT,
            judge_prompt=agent_harness.load_judge_prompt(out / "judge_prompt.txt"),
            governor=governor,
            transcript_dir=out / "transcripts",
            seed=SEED,
            system_prompt=WS8_SYSTEM_PROMPT,
            max_tokens=RUN_MAX_TOKENS,
        )

        # The SAME derivation write_ledger() uses, so the printed counts and
        # the ledger note can never disagree about what this scope contains.
        facts = scope_counts(CHECKPOINT, set(qs["qid"]))
        all_ckpt = facts["ckpt"]
        pair_rows, truncated, capped = (
            facts["pairs"], facts["truncated"], facts["capped"])

        print(f"\n{len(pair_rows)} of {len(qs) * len(ARMS)} targeted pair(s) "
              f"checkpointed this invocation's scope.")

        if truncated:
            print(f"\nWARNING: {len(truncated)} pair(s) hit max_tokens with an "
                  "EMPTY answer -- Phase 1's ws8_q065 failure mode. These "
                  "should NOT be trusted as genuine declines:", file=sys.stderr)
            for r in truncated:
                print(f"  {r['arm']:8s} {r['qid']}", file=sys.stderr)

        if capped:
            print(f"\n{len(capped)} pair(s) hit the turn cap "
                  f"(WS8_TURN_CAP): {[(r['arm'], r['qid']) for r in capped]}")

        if is_smoke:
            # Cumulative across EVERY smoke invocation so far (not just this
            # one's target qids): each WS8_SMOKE/WS8_QIDS run adds pairs to the
            # same checkpoint, and a per-stratum estimate needs whatever strata
            # have been sampled across all of them, not just today's.
            stratum_of = dict(zip(qs_all["qid"], qs_all["stratum"]))
            stratum_size = qs_all.groupby("stratum")["qid"].count().to_dict()
            all_pair_rows = [r for (qid, arm), r in all_ckpt.items()
                             if arm in ARMS and qid in stratum_of]

            print("\n--- SMOKE RESULTS (per stratum x arm, cumulative across "
                  "every smoke invocation) ---")
            per_cell = {}
            for r in all_pair_rows:
                key = (stratum_of[r["qid"]], r["arm"])
                per_cell.setdefault(key, []).append(r)
            for stratum in ("hop1", "hop2", "hop3plus"):
                for arm in ARMS:
                    rows = per_cell.get((stratum, arm), [])
                    if not rows:
                        print(f"{stratum:9s} {arm:8s} n=0 (not yet sampled)")
                        continue
                    costs = [r["cost_total_billed"] for r in rows]
                    turns = [r["turns"] for r in rows]
                    n_capped = sum(1 for r in rows if r.get("hit_turn_cap"))
                    mean_cost = sum(costs) / len(costs)
                    print(f"{stratum:9s} {arm:8s} n={len(rows)} "
                          f"mean_cost=${mean_cost:.6f} "
                          f"costs={[round(c, 6) for c in costs]} "
                          f"mean_turns={sum(turns) / len(turns):.1f} "
                          f"turns={turns} turn_capped={n_capped}")

            # Retrieved-payload size per search call, indexed arm, by stratum:
            # tool_result content_length (the UNTRUNCATED length _compact_blocks
            # records) is exactly what ws8_index.make_search_tool's
            # search_documents returned that turn, in characters. Every
            # tool_result in an `indexed`-arm transcript is a search_documents
            # call -- it is the only tool exposed to that arm.
            print("\n--- indexed arm: retrieved payload size per search call, "
                  "by stratum ---")
            for stratum in ("hop1", "hop2", "hop3plus"):
                rows = per_cell.get((stratum, "indexed"), [])
                sizes = []
                n_no_hit = 0
                for r in rows:
                    tpath = out / "transcripts" / "indexed" / f"{r['qid']}.jsonl"
                    if not tpath.exists():
                        continue
                    for line in tpath.read_text().splitlines():
                        turn = json.loads(line)
                        for block in turn.get("content", []):
                            if block.get("type") == "tool_result":
                                sizes.append(block.get("content_length", 0))
                                if block.get("content", "").strip() == (
                                        "No results found."):
                                    n_no_hit += 1
                if not sizes:
                    print(f"{stratum:9s} n=0 (not yet sampled)")
                    continue
                print(f"{stratum:9s} n_calls={len(sizes)} "
                      f"mean_chars={sum(sizes) / len(sizes):.0f} "
                      f"min={min(sizes)} max={max(sizes)} "
                      f"no_hit_calls={n_no_hit}/{len(sizes)}")

            # Weighted full-run estimate from real stratum sizes (30/15/30),
            # using whichever per-stratum x arm means have been measured so far.
            # A cell with no data yet makes the total "incomplete", printed
            # loudly rather than silently guessed.
            print("\n--- weighted full-run estimate (30 hop1 / 15 hop2 / "
                  "30 hop3plus) ---")
            total = 0.0
            incomplete = False
            for stratum in ("hop1", "hop2", "hop3plus"):
                n_q = stratum_size.get(stratum, 0)
                for arm in ARMS:
                    rows = per_cell.get((stratum, arm), [])
                    if not rows:
                        incomplete = True
                        print(f"  {stratum:9s} {arm:8s}: NO DATA -- estimate "
                              "below excludes this cell")
                        continue
                    costs = [r["cost_total_billed"] for r in rows]
                    mean_cost = sum(costs) / len(costs)
                    subtotal = mean_cost * n_q
                    total += subtotal
                    print(f"  {stratum:9s} {arm:8s}: mean=${mean_cost:.6f} x "
                          f"{n_q} questions = ${subtotal:.4f}")
            print(f"\n{'INCOMPLETE ' if incomplete else ''}"
                  f"weighted full-run estimate: ${total:.4f}")
        else:
            # Full run: write the audited runs.csv -- WS6a's columns plus
            # stratum and hops, so the two benchmarks stay directly comparable.
            rows = pd.read_json(CHECKPOINT, lines=True).drop_duplicates(
                subset=["qid", "arm"], keep="last")
            rows = rows[rows["arm"].isin(ARMS)]
            rows = rows.merge(
                qs[["qid", "stratum", "hops", "grep_handle"]], on="qid")
            runs_path = out / "runs.csv"
            rows.to_csv(runs_path, index=False)
            manifest.record(runs_path)
            print(f"{len(rows)} rows -> {runs_path}")

    finally:
        # sys.exc_info()[1] is the exception still propagating, if any, so the
        # ledger note can name it without an except/raise pair that could
        # itself swallow the traceback.
        write_ledger(sys.exc_info()[1])


if __name__ == "__main__":
    main()
