#!/usr/bin/env python
"""One-off correction (R19): re-run ws8_q065 alone at a larger max_tokens.

ws8_q065 (hop3plus, rust-lang/rust) truncated on the Phase 1 gate's first
corrected-prompt (R17) run: stop_reason="max_tokens", empty answer, the
whole 4,096-token budget consumed by an unbilled `thinking` block with no
text emitted. That is an UNMEASURED question, not a genuine decline --
agent_harness.judge_answer short-circuits an empty answer to
judge_correct=False WITHOUT ever calling the judge, so this row would have
recorded a free zero in the direction that makes the gate easier to pass.
Its cost ($0.041272) was also 5.2x the median row -- a cost outlier driven
by truncation, not by the arm.

Extended/adaptive thinking is on by default for claude-sonnet-5 (the WS8
agent model) in this API, exactly as agent_harness.py's own comment already
documents for claude-opus-5 (the judge model). Nothing disables it for the
agent arms here (contrast benchlib/recall_quality/runner.py, which passes
thinking={"type": "disabled"} explicitly), and it applies uniformly to
every WS8 arm, so it does not break comparability -- but it is the direct
mechanism behind this truncation and was previously undocumented for the
agent model.

Rather than raise the whole gate's MAX_TOKENS default (which would change
every other row's comparability retroactively -- see run_arm's system_prompt
precedent, R17), this retries JUST ws8_q065 at max_tokens=16384 (the same
ceiling agent_harness.JUDGE_MAX_TOKENS_SCHEDULE already escalates to for the
analogous judge-side problem) via run_arm's new `max_tokens` parameter.

The corrected row is APPENDED to the same live checkpoint
(data/ws8_cache/gate_checkpoint.jsonl) -- never overwriting or deleting the
original truncated line. Per R18, both calls were real, billed API spend, so
ws8.spent_from_checkpoint must count both; load_checkpoint's last-wins dedup
makes this corrected row the one scripts/non-code-retrieval/gate_ws8.py uses when it next
regenerates runs_parametric.csv (a $0 pass, since every question is then
already checkpointed).

Run once: python scripts/non-code-retrieval/retry_ws8_q065.py
Then:      python scripts/non-code-retrieval/gate_ws8.py   # regenerates the CSVs, $0 new spend
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
                             WS8_JUDGE_MODEL, WS8_SYSTEM_PROMPT, ws6a_dir,
                             ws8_dir)

QID = "ws8_q065"
RETRY_MAX_TOKENS = 16384


# The ledger writer lives in benchlib.non_code_retrieval.bench (hoisted
# 2026-09-11 from the four verbatim copies this file was one of); see it
# for the column mechanism.


def billed_or_zero(usage, price) -> float:
    """Cost of a call that has already happened, or 0.0 if its usage cannot be
    read. Used on the abort path, where the alternative to a best-effort figure
    is no ledger row at all."""
    try:
        return ws6a.cost_billed(usage, price)
    except Exception:
        return 0.0


def main() -> None:
    out = ws8_dir()
    qs = pd.read_csv(out / "questions.csv")
    row = qs[qs["qid"] == QID]
    if row.empty:
        sys.exit(f"{QID} not found in questions.csv")
    r = row.iloc[0].to_dict()

    client = agent_harness.make_client()
    prices = ws6a.load_pricing(ws6a_dir() / "pricing.csv", regime="standard")
    judge_prompt = agent_harness.load_judge_prompt(out / "judge_prompt.txt")
    ckpt = DATA_DIR / "ws8_cache" / "gate_checkpoint.jsonl"

    spent = ws8.total_spent_across_checkpoints(DATA_DIR)
    governor = ws6a.CostGovernor(WS8_BUDGET_USD, spent=spent)
    print(f"governor seeded at ${governor.spent:.4f} of ${WS8_BUDGET_USD:.2f}")

    agent_price = prices[WS8_AGENT_MODEL]
    judge_price = prices[WS8_JUDGE_MODEL]

    ledger_path = out / "spend_ledger.csv"
    # Charged as each call RETURNS, not once at the end. Both guards below
    # sys.exit() after a real, billed agent call, and the old end-of-main
    # append meant either exit threw away the record of that spend entirely --
    # the same failure that cost run_ws8.py a $0.250558 ledger row, except
    # here it was reachable on a deliberate, non-exceptional path.
    billed = {"agent": 0.0, "judge": 0.0}
    ledger_note = {"text": (f"ABORTED: retry of ws8_q065 at "
                            f"max_tokens={RETRY_MAX_TOKENS} did not complete; "
                            "spend below is what had already been billed")}
    ledger_written: list[bool] = []

    def write_ledger(error: BaseException | None = None) -> None:
        if ledger_written:
            return
        ledger_written.append(True)
        spend = round(billed["agent"] + billed["judge"], 6)
        note = ledger_note["text"]
        if error is not None:
            detail = " ".join(str(error).split())[:200]
            note += f" [{type(error).__name__}: {detail}]"
        ws8.append_spend_ledger(ledger_path, run_label="retry_ws8_q065_r19",
                                spend_usd=spend, note=note)
        manifest.record(ledger_path)
        print(f"appended ${spend:.6f} to {ledger_path}")
        ws8.warn_on_ledger_mismatch(ledger_path, DATA_DIR)

    try:
        res = agent_harness.run_arm(client, "parametric", QID, r["question"],
                                  root=None, model=WS8_AGENT_MODEL,
                                  system_prompt=WS8_SYSTEM_PROMPT,
                                  max_tokens=RETRY_MAX_TOKENS)
        billed["agent"] = billed_or_zero(res.usage, agent_price)
        if res.error:
            sys.exit(f"run_arm failed: {res.error}")
        if not res.answer.strip():
            sys.exit(f"still empty at max_tokens={RETRY_MAX_TOKENS} "
                     f"(stop_reason={res.stop_reason!r}) -- escalate further "
                     f"before appending a checkpoint row; do NOT record a zero")

        expected_docs = str(r["expected_documents"]).split(";")
        verdict = agent_harness.judge_answer(
            client, judge_prompt, r["question"], expected_docs, "",
            res.answer, model=WS8_JUDGE_MODEL)
        billed["judge"] = billed_or_zero(verdict["usage"], judge_price)

        agent_cost, judge_cost = billed["agent"], billed["judge"]
        total_cost = round(agent_cost + judge_cost, 6)
        any_doc_hit = int(any(d in res.answer for d in expected_docs))

        record = {
            "qid": QID, "arm": "parametric",
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
            "note": (f"R19 retry at max_tokens={RETRY_MAX_TOKENS}; supersedes "
                     f"the truncated max_tokens=4096 line for this (qid, arm) "
                     f"earlier in this checkpoint file, which is left in place "
                     f"per R18 (both calls were real, billed spend)"),
        }
        agent_harness.append_checkpoint(ckpt, record)

        transcript_dir = out / "transcripts_parametric"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        with open(transcript_dir / f"{QID}.jsonl", "w") as fh:
            for turn in res.turns_detail:
                fh.write(json.dumps(turn) + "\n")

        governor.charge(total_cost)
        print(f"{QID}  stop_reason={res.stop_reason}  judge_correct="
              f"{verdict['judge_correct']}  any_doc_hit={any_doc_hit}  "
              f"cost=${total_cost:.6f}  cumulative=${governor.spent:.6f}")
        print(f"answer: {res.answer[:300]!r}")

        ledger_note["text"] = (f"R19: retry of the truncated ws8_q065 at "
                               f"max_tokens={RETRY_MAX_TOKENS}; judge_correct="
                               f"{verdict['judge_correct']}")
    finally:
        write_ledger(sys.exc_info()[1])


if __name__ == "__main__":
    main()
