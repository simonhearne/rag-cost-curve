#!/usr/bin/env python
"""WS8 Phase 1: the parametric gate. Runs BEFORE anything else is spent.

All 75 questions, no corpus access. Publishes its score whether or not the
corpus is used -- WS6c published all three candidates' gate scores.

Corrections applied here (two defects found in review, 2026-09-10):

R2 -- agent_harness.judge_answer takes SIX positionals:
    (client, prompt_template, question, expected_files, expected_symbols,
     answer, model=None) -> {judge_correct, judge_reason, usage}.
    WS8 has no symbol analogue, so expected_symbols="" is passed through.

R11/R12 -- the governor is seeded from ws8.total_spent_across_checkpoints,
    which sums benchlib.non_code_retrieval.bench.spent_from_checkpoint (counts every billed line,
    including superseded duplicates) over every WS8 checkpoint file, never
    from agent_harness.checkpoint_spent (last-wins dedup) or
    agent_harness.total_spent_across_checkpoints (built on top of the same
    dedup). The latter is how WS6c finished over its cap on retried pairs
    (see results/code-retrieval.md).

Fix round 1 (2026-09-11) -- R17: run_arm was called under
agent_harness.SYSTEM_PROMPT by default, WS6a's frozen "Python codebase" /
"file path" / "functions or classes" framing -- wrong for WS8's GitHub
issue/PR threads on Go and Rust repos. Spec section 7 reuses Phase 1's gate
rows AS the parametric arm in Phase 3 verbatim, so a Phase 3 run under a
corrected prompt would have left the parametric arm alone under the wrong
one -- two variables (tools AND system prompt) differing between arms
instead of one, which agent_harness.run_arm's own docstring forbids. Fixed by
passing config.WS8_SYSTEM_PROMPT explicitly (run_arm's system_prompt
parameter, which defaults to the unchanged SYSTEM_PROMPT for every frozen
WS6a/WS6b/WS6c call site that does not pass it). The first gate run
(accuracy 0.000, any_doc_hit 0/75, $0.783336 billed) is VOID.

R18 (2026-09-11) -- ARCHIVE, NEVER DELETE a checkpoint that recorded real
billed spend. The fix-round-1 instruction to delete the voided run's
checkpoint before re-running was itself a defect: it destroyed the record of
$0.783336 in real spend and made it invisible to the governor -- the exact
WS6c failure mode ws8.spent_from_checkpoint exists to prevent. The voided
run's checkpoint is reconstructed as
`data/ws8_cache/gate_checkpoint.voided-2026-09-11.jsonl` (gitignored, like
every other checkpoint, but present locally and its cost reflected in every
downstream spend figure). Going forward, a superseded checkpoint is renamed
to `<name>.voided-<ISO8601>.jsonl`, never removed -- ws8.
total_spent_across_checkpoints globs every `ws8_cache/*.jsonl` file, live or
archived, so this is automatic. The live checkpoint path below
(`gate_checkpoint.jsonl`) is still the ONLY file read for RESUME purposes,
which is what keeps a fresh run from mixing with a superseded one -- the
legitimate half of the original instruction.

Fix round 2 (2026-09-11) -- five findings from review of the corrected run:

R19 -- ws8_q065 (hop3plus) came back stop_reason="max_tokens" with an EMPTY
    answer: the entire 4,096-token budget was consumed by an unbilled
    `thinking` block with no text emitted. agent_harness.judge_answer
    short-circuits an empty answer to judge_correct=False WITHOUT calling
    the judge (see judge_answer's own docstring/code), so this row was an
    *unmeasured* question scored as a zero -- the direction that makes the
    gate easier to pass -- not a genuine decline. Fixed by giving run_arm an
    optional `max_tokens` parameter (same default-None, resolve-internally
    pattern as `system_prompt` from R17) and retrying JUST this question at
    a larger budget via scripts/non-code-retrieval/retry_ws8_q065.py, appending the corrected
    row to the SAME live checkpoint (never deleting the truncated one --
    R18's rule applies to a single row exactly as it does to a whole file:
    both lines are real billed spend and spent_from_checkpoint counts both;
    load_checkpoint's last-wins dedup makes the corrected row the one that
    reaches runs_parametric.csv).

    Extended/adaptive thinking is ON BY DEFAULT for claude-sonnet-5 (the WS8
    agent model) in this API, exactly as agent_harness.py's own comment
    already documents for claude-opus-5 (the judge model) -- confirmed by
    inspecting the committed transcripts: some parametric rows show
    ['thinking', 'text'] content blocks (e.g. ws8_q001), some show only
    ['text'] (no thinking triggered), and ws8_q065 showed only ['thinking']
    with no text at all (the whole budget spent thinking). Nothing in
    agent_harness.py disables it for the agent arms, unlike WS4
    (benchlib/recall_quality/runner.py passes thinking={"type": "disabled"}
    explicitly). It applies uniformly to every WS8 arm sharing run_arm's
    _single_turn/tool_runner calls, so it does not break comparability
    within WS8, but it was previously undocumented for the agent model and
    is the direct mechanism behind this truncation.

Truncated-empty rows are now treated like arm errors (guard below): NOT
checkpointed, so a resume retries them, rather than being silently recorded
as a permanent (and free) zero.

R20 -- results/ws8/spend_ledger.csv (committed, unlike data/ws8_cache/,
    which is gitignored) carries one row per invocation of this script, so
    the cumulative spend figure can be reconstructed from committed state
    alone even if data/ is ever lost or reset. The script appends to it
    every run and reconciles its running total against
    ws8.total_spent_across_checkpoints -- a mismatch means spend has gone
    invisible (the R18 failure class) and is printed loudly rather than
    silently ignored.

    Fix round 4 (2026-09-11): the ledger's first version stamped a single
    `source_commit` column with `git rev-parse HEAD` AT RUN TIME. Two of its
    first four rows then named a commit that did not contain their
    evidence -- a run cannot know the commit that will eventually contain
    its own outputs. Replaced with three columns: `ran_at_head` (HEAD at
    run time, honest about a possibly-stale value), `dirty` (whether the
    working tree had uncommitted changes when it ran -- true for every row
    so far, since every fix in this phase ran before its own commit landed),
    and `committed_in` (the commit that actually contains the row's code and
    outputs; written empty at append time and filled in AFTER that commit
    exists -- see scripts/fixtures/stamp_ledger_commit.py, the mechanical fill-in
    step Phase 2/3 should reuse rather than re-deriving this).

The committed summary CSV distinguishes `spend_this_invocation_usd` (what
THIS run of the script billed -- often $0.00 on a pure resume/regenerate
pass) from `phase_spend_usd` (every gate_checkpoint* file's total, live and
archived) and `cumulative_ws8_spend_usd` (every WS8 checkpoint of any
phase) -- the first fix round's committed CSV read "spend_this_run_usd:
0.0" with nothing explaining that the phase had actually billed $1.627377
across three invocations, which is exactly the kind of gap this repo's
"every number is a committed CSV" rule exists to close.

A successful-but-judge-failed pair no longer discards its already-billed
agent cost: judge_answer raising is recorded as a spend-only checkpoint line
(distinct arm label, so it is summed by spent_from_checkpoint but never
mistaken for a completed pair) and charged to the in-memory governor before
the loop continues, rather than silently dropping real spend on the floor.

Run:  python scripts/non-code-retrieval/gate_ws8.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest, agent_harness
from benchlib.non_code_retrieval import bench as ws8
from benchlib.code_retrieval import bench as ws6a
from benchlib.config import (DATA_DIR, WS8_AGENT_MODEL, WS8_BUDGET_USD,
                             WS8_JUDGE_MODEL, WS8_N_QUESTIONS,
                             WS8_SYSTEM_PROMPT, ws6a_dir, ws8_dir)

# Phase 1's own checkpoints only (live + any archived .voided-* siblings),
# a narrower glob than ws8.WS8_CHECKPOINT_GLOB -- which will also catch
# Phase 2/3's checkpoints once they exist under ws8_cache/. This is what
# lets phase_spend_usd (this phase alone) and cumulative_ws8_spend_usd
# (every WS8 phase) read differently once Phase 2/3 land, rather than
# silently aliasing to the same number forever.
#
# ASSUMPTION FOR PHASE 2/3's AUTHOR: this glob matches "gate_checkpoint*"
# specifically because that is Phase 1's own checkpoint naming. If a later
# phase's checkpoint file is ever named to also match "gate_checkpoint*"
# (under ws8_cache/), its spend silently joins Phase 1's phase_spend_usd
# instead of its own phase's total. Give Phase 2/3 checkpoints a
# distinctly-prefixed name (e.g. "index_checkpoint*.jsonl",
# "retrieval_checkpoint*.jsonl") and this glob does not need to change.
PHASE1_CHECKPOINT_GLOB = "ws8_cache/gate_checkpoint*.jsonl"


def phase1_spent(data_dir) -> float:
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob(PHASE1_CHECKPOINT_GLOB))
    return round(sum(ws8.spent_from_checkpoint(p) for p in paths), 6)


# The ledger writer -- append_spend_ledger plus the git HEAD/dirty helpers it
# stamps rows with -- lives in benchlib.non_code_retrieval.bench, hoisted
# there on 2026-09-11 from the four verbatim copies this file was the
# original of. Its docstring carries
# the ran_at_head / dirty / committed_in reasoning.


def main() -> None:
    out = ws8_dir()
    qs = pd.read_csv(out / "questions.csv")
    qs = qs[~qs["excluded"].astype(bool)].reset_index(drop=True)
    # If `excluded` ever parsed as strings instead of bool, every row would
    # silently look truthy/falsy in a way that could drop the whole set --
    # fail loudly against the pre-registered n instead.
    assert len(qs) == WS8_N_QUESTIONS, (
        f"expected {WS8_N_QUESTIONS} non-excluded questions, got {len(qs)} "
        f"-- check questions.csv's `excluded` column dtype")
    print(f"{len(qs)} questions loaded (excluded rows dropped)")

    client = agent_harness.make_client()
    prices = ws6a.load_pricing(ws6a_dir() / "pricing.csv", regime="standard")
    judge_prompt_path = out / "judge_prompt.txt"
    judge_prompt = agent_harness.load_judge_prompt(judge_prompt_path)
    manifest.record(judge_prompt_path)  # a graded input to the gate; cheap

    ckpt = DATA_DIR / "ws8_cache" / "gate_checkpoint.jsonl"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    transcript_dir = out / "transcripts_parametric"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    spent = ws8.total_spent_across_checkpoints(DATA_DIR)
    governor = ws6a.CostGovernor(WS8_BUDGET_USD, spent=spent)
    print(f"governor seeded at ${governor.spent:.4f} of "
          f"${WS8_BUDGET_USD:.2f} (cumulative across every WS8 checkpoint, "
          f"live and archived -- R18)")

    # R20: one row per invocation in the committed spend ledger, written from
    # a `finally` so an invocation that dies mid-loop still records what it
    # billed. The label and note default to the abort form and are replaced
    # only once the gate has actually produced a branch.
    ledger_path = out / "spend_ledger.csv"
    ledger_row = {"run_label": "gate_ws8_aborted",
                  "note": ("ABORTED before the gate summary was computed; "
                           "this invocation's spend is real and is in "
                           "gate_checkpoint.jsonl, but no branch was decided")}
    ledger_written: list[bool] = []

    def write_ledger(error: BaseException | None = None) -> None:
        if ledger_written:
            return
        ledger_written.append(True)
        invocation_spend = round(governor.spent - spent, 6)
        note = ledger_row["note"]
        if error is not None and not isinstance(error, SystemExit):
            # A G2 exit is an ORDERLY halt with a branch already decided, not
            # a crash -- annotating it as aborted would misdescribe the one
            # outcome the gate exists to produce.
            detail = " ".join(str(error).split())[:200]
            note += f" [{type(error).__name__}: {detail}]"
        ws8.append_spend_ledger(ledger_path, run_label=ledger_row["run_label"],
                                spend_usd=invocation_spend, note=note)
        manifest.record(ledger_path)
        print(f"appended ${invocation_spend:.6f} to {ledger_path}")
        # The ledger SUM against a fresh sweep of every checkpoint on disk.
        # The check this replaces compared the ledger against `governor.spent`
        # -- an in-process accumulator seeded once at startup, which agrees
        # with itself by construction inside a surviving invocation and so
        # cannot see a row that a DEAD invocation never wrote.
        ws8.warn_on_ledger_mismatch(ledger_path, DATA_DIR)

    try:
        done = agent_harness.load_checkpoint(ckpt)
        todo = sum(1 for r in qs.to_dict("records")
                  if (r["qid"], "parametric") not in done)
        print(f"{len(done)} pair(s) already in the checkpoint; {todo} to run")

        errors: list[tuple[str, str]] = []

        for r in qs.to_dict("records"):
            qid = r["qid"]
            if (qid, "parametric") in done:
                continue

            res = agent_harness.run_arm(client, "parametric", qid, r["question"],
                                      root=None, model=WS8_AGENT_MODEL,
                                      system_prompt=WS8_SYSTEM_PROMPT)

            # R19: a truncated, empty answer is an UNMEASURED question, not a
            # genuine decline -- agent_harness.judge_answer would short-circuit it
            # to judge_correct=False without ever calling the judge, recording a
            # free zero in the direction that makes the gate easier to pass.
            # Treat it exactly like an arm error: never checkpoint it, so a
            # resume retries it (see scripts/non-code-retrieval/retry_ws8_q065.py for the one-off
            # correction already applied to this phase's own truncated row).
            truncated_empty = (res.stop_reason == "max_tokens"
                               and not (res.answer or "").strip())

            if res.error or truncated_empty:
                reason = res.error or (
                    "truncated: max_tokens hit with an empty answer (thinking "
                    "consumed the whole budget) -- not a genuine decline")
                errors.append((qid, reason))
                print(f"  WARNING: {qid}  NOT checkpointed (will retry on "
                      f"resume): {reason}", file=sys.stderr)
                continue

            expected_docs = str(r["expected_documents"]).split(";")

            try:
                # R2: six positionals; "" for expected_symbols (WS8 has no
                # symbol analogue); the key is judge_correct, not correct.
                verdict = agent_harness.judge_answer(
                    client, judge_prompt, r["question"], expected_docs, "",
                    res.answer, model=WS8_JUDGE_MODEL)
            except Exception as exc:
                # The agent call already happened and was billed -- discarding
                # it here without charging the governor or recording it
                # anywhere is exactly the invisible-spend class R11/R12/R18
                # exist to close. Record a spend-only line under a DISTINCT arm
                # label (never "parametric"), so spent_from_checkpoint sums it
                # but load_checkpoint's resume-skip logic never mistakes it for
                # a completed (qid, "parametric") pair.
                agent_price = prices[WS8_AGENT_MODEL]
                agent_cost = ws6a.cost_billed(res.usage, agent_price)
                agent_harness.append_checkpoint(ckpt, {
                    "qid": qid, "arm": "parametric_judge_error",
                    "cost_total_billed": agent_cost,
                    "error": f"{type(exc).__name__}: {exc}",
                    "note": "agent call billed and charged; judge raised, pair "
                            "not recorded as done, resume will retry it",
                })
                governor.charge(agent_cost)
                errors.append((qid, f"{type(exc).__name__}: {exc}"))
                print(f"  WARNING: {qid}  judge failed, agent cost "
                      f"${agent_cost:.6f} charged and checkpointed as spend-only "
                      f"(will retry on resume): {type(exc).__name__}: {exc}",
                      file=sys.stderr)
                continue

            agent_price = prices[WS8_AGENT_MODEL]
            judge_price = prices[WS8_JUDGE_MODEL]
            agent_cost = ws6a.cost_billed(res.usage, agent_price)
            judge_cost = ws6a.cost_billed(verdict["usage"], judge_price)
            total_cost = round(agent_cost + judge_cost, 6)

            # any_doc_hit: does the answer text name any expected document?
            # WS6a found this substring signal unreliable on a contaminated
            # corpus -- the model can name a file inside a sentence DENYING it
            # can answer -- so it is published alongside judge_correct, never
            # relied on alone.
            any_doc_hit = int(any(d in res.answer for d in expected_docs))

            record = {
                "qid": qid, "arm": "parametric",
                "stratum": r["stratum"], "hops": r["hops"], "repo": r["repo"],
                "question": r["question"],
                "expected_documents": r["expected_documents"],
                "answer": res.answer,
                "judge_correct": bool(verdict["judge_correct"]),
                "judge_reason": verdict["judge_reason"],
                "any_doc_hit": any_doc_hit,
                "turns": res.turns,
                "stop_reason": res.stop_reason,
                "wall_clock_s": res.wall_clock_s,
                "started_at": res.started_at,
                "cost_total_billed": total_cost,
                "cost_agent_billed": agent_cost,
                "cost_judge_billed": judge_cost,
            }
            agent_harness.append_checkpoint(ckpt, record)

            try:
                with open(transcript_dir / f"{qid}.jsonl", "w") as fh:
                    for turn in res.turns_detail:
                        fh.write(json.dumps(turn) + "\n")
            except Exception as exc:
                print(f"Warning: transcript write failed for {qid}: {exc}",
                      file=sys.stderr)

            governor.charge(total_cost)  # raises BudgetExceeded to stop cleanly
            print(f"  {qid}  ${governor.spent:6.4f} spent  "
                  f"judge={record['judge_correct']}  any_doc_hit={any_doc_hit}")

        if errors:
            print(f"\n=== {len(errors)} pair(s) FAILED and were NOT recorded "
                  f"(retry by re-running -- resume will pick them up) ===",
                  file=sys.stderr)
            for qid, err in errors:
                print(f"  {qid}: {err}", file=sys.stderr)

        all_done = agent_harness.load_checkpoint(ckpt)
        rows = [all_done[(r["qid"], "parametric")] for r in qs.to_dict("records")
                if (r["qid"], "parametric") in all_done]
        df = pd.DataFrame(rows)

        if len(df) < len(qs):
            print(f"INCOMPLETE: {len(df)} of {len(qs)} questions have a "
                  f"checkpointed result; re-run to resume.", file=sys.stderr)
            sys.exit(1)

        # manifest.record runs only once the CSV is known-complete -- recording
        # an incomplete resume's partial sha256 and then exiting 1 would commit
        # a checksum for data nobody should treat as final.
        runs_path = out / "runs_parametric.csv"
        df.to_csv(runs_path, index=False)
        manifest.record(runs_path)
        print(f"\nwrote {runs_path} ({len(df)} row(s))")

        acc = float(df["judge_correct"].astype(bool).mean())
        hits = int(df["any_doc_hit"].astype(bool).sum())
        branch = ws8.gate_branch(acc, hits)

        invocation_spend = round(governor.spent - spent, 6)
        phase_spend = phase1_spent(DATA_DIR)

        summary = pd.DataFrame([{
            "n": len(df), "accuracy": acc, "any_doc_hit": hits,
            "any_doc_hit_rate": f"{hits}/{len(df)}",
            "branch": branch,
            "spend_this_invocation_usd": invocation_spend,
            "phase_spend_usd": phase_spend,
            "cumulative_ws8_spend_usd": governor.spent,
        }])
        gate_path = out / "parametric_gate.csv"
        summary.to_csv(gate_path, index=False)
        manifest.record(gate_path)
        print(f"wrote {gate_path}")
        print(summary.to_string(index=False))

        ledger_row["run_label"] = f"gate_ws8_{branch.lower()}"
        ledger_row["note"] = (
            f"n={len(df)} accuracy={acc:.3f} "
            f"any_doc_hit={hits}/{len(df)} branch={branch}")

        if branch == "G2":
            sys.exit(
                "\nGATE FAILED (G2): parametric accuracy or any_doc_hit is "
                "non-zero on this corpus. This is a legitimate outcome, not a "
                "bug -- record the score, commit it, and halt WS8. Do not "
                "re-run hoping for a different number and do not tune the "
                "judge prompt or exclude questions to force a pass.")

        print("\nGATE PASSED (G1): parametric accuracy 0.000, any_doc_hit "
              f"{hits}/{len(df)}. Proceed to Phase 2.")
    finally:
        write_ledger(sys.exc_info()[1])


if __name__ == "__main__":
    main()
