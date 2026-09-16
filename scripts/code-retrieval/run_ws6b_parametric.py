#!/usr/bin/env python3
"""WS6b phase 3: the parametric arm, and the non-memorization gate.

HARD PHASE BOUNDARY. No retrieval or replay arm may be run until this gate
passes (spec 5.5). If a model with NO history access answers materially above
zero, the answers are leaking from question phrasing or world knowledge and
every downstream number is contaminated by an unknown amount. WS6a discovered
a 62.5% contamination floor after spending; this gates on it before.

results/ws6b/parametric_gate.csv records EVERY attempt with the generator's
git SHA. A gate that passed on attempt 3 reported as passing on attempt 1 is
the dishonesty the pre-registration exists to prevent.

Design: results/code-retrieval.md
"""

import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import config, manifest  # noqa: E402
from benchlib.code_retrieval import memory_runner as ws6b_runner  # noqa: E402
from benchlib.code_retrieval import memory as ws6b  # noqa: E402
from benchlib.code_retrieval import bench as ws6a  # noqa: E402
from benchlib.agent_harness import (append_checkpoint, checkpoint_spent,  # noqa: E402
                                  load_checkpoint, load_judge_prompt,
                                  make_client)

OUT = config.ws6b_dir()
CACHE = config.DATA_DIR / "ws6b_cache"
CHECKPOINT = CACHE / "parametric_checkpoint.jsonl"
GATE_CSV = OUT / "parametric_gate.csv"


def generator_sha() -> str:
    """The commit of the generator that produced the corpus under test."""
    p = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--",
         "scripts/code-retrieval/generate_ws6b_corpus.py"],
        capture_output=True, text=True, cwd=config.REPO_ROOT)
    return p.stdout.strip()[:12] or "uncommitted"


def next_attempt() -> int:
    if not GATE_CSV.exists():
        return 1
    rows = list(csv.DictReader(GATE_CSV.open()))
    return max((int(r["attempt"]) for r in rows), default=0) + 1


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    probes = list(csv.DictReader((OUT / "probes.csv").open()))
    prices = ws6a.load_pricing(OUT / "pricing.csv")
    judge_prompt = load_judge_prompt(OUT / "judge_prompt.txt")
    client = make_client()

    # Cumulative across every phase, seeded from the checkpoint -- never
    # per-script. WS6a recorded that a per-script governor would have
    # authorised roughly $300 against a $75 cap.
    governor = ws6a.CostGovernor(config.WS6B_BUDGET_USD,
                                 spent=checkpoint_spent(CHECKPOINT))
    done = load_checkpoint(CHECKPOINT)
    print(f"governor: ${governor.spent:.4f} spent of ${governor.limit:.2f}; "
          f"{len(done)} probe(s) already checkpointed")

    agent_price = prices[config.WS6B_AGENT_MODEL]
    judge_price = prices[config.WS6B_JUDGE_MODEL]

    for probe in probes:
        pid = probe["probe_id"]
        if (pid, "parametric") in done:
            continue
        r = ws6b_runner.run_arm_ws6b(
            client, "parametric", pid, probe["question"])
        if r.error:
            print(f"  {pid}: ERROR {r.error} -- not checkpointed, will retry")
            continue
        try:
            verdict = ws6b_runner.judge_ws6b(
                client, judge_prompt, probe["question"],
                probe["expected_answer"], probe["stale_answer"], r.answer)
        except Exception as exc:
            # Never record a judge failure as a wrong answer: that fabricates
            # a result. Un-checkpointed, so a resumed run retries it.
            print(f"  {pid}: JUDGE FAILED ({exc}) -- not checkpointed")
            continue

        agent_billed = ws6a.cost_billed(r.usage, agent_price)
        judge_billed = ws6a.cost_billed(verdict["usage"], judge_price)
        row = {
            "qid": pid, "arm": "parametric", "probe_set": probe["probe_set"],
            "answer": r.answer, "judge_correct": verdict["judge_correct"],
            "judge_reason": verdict["judge_reason"],
            "mechanical_correct": ws6b.score_mechanical(
                r.answer, probe["expected_answer"]),
            "prompt_tokens": ws6b.prompt_tokens(r.usage),
            "output_tokens": r.usage.output_tokens,
            "cost_billed_usd": agent_billed,
            "cost_list_usd": ws6a.cost_uncached_list(r.usage, agent_price),
            "cost_total_billed": round(agent_billed + judge_billed, 6),
            "wall_clock_s": r.wall_clock_s, "started_at": r.started_at,
        }
        # Checkpoint BEFORE charging, so a crash never loses a completed pair.
        append_checkpoint(CHECKPOINT, row)
        governor.charge(row["cost_total_billed"])
        mark = "OK " if verdict["judge_correct"] else "   "
        print(f"  {mark}{pid} [{probe['probe_set']:>10}] "
              f"{verdict['judge_correct']!s:>5}  {r.answer.strip()[:70]!r}")

    rows = list(load_checkpoint(CHECKPOINT).values())
    correct = sum(1 for r in rows if r["judge_correct"])
    accuracy = correct / len(rows) if rows else 0.0
    verdict = ws6b_runner.gate_verdict(accuracy)
    sha, attempt = generator_sha(), next_attempt()

    header = not GATE_CSV.exists()
    with open(GATE_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "attempt", "generator_git_sha", "probe_id", "probe_set", "answer",
            "judge_correct", "judge_reason", "mechanical_correct",
            "accuracy", "verdict", "run_date"])
        if header:
            w.writeheader()
        for r in rows:
            w.writerow({
                "attempt": attempt, "generator_git_sha": sha,
                "probe_id": r["qid"], "probe_set": r["probe_set"],
                "answer": r["answer"], "judge_correct": r["judge_correct"],
                "judge_reason": r["judge_reason"],
                "mechanical_correct": r["mechanical_correct"],
                "accuracy": round(accuracy, 4), "verdict": verdict,
                "run_date": r["started_at"][:10]})
    manifest.record(GATE_CSV)

    print(f"\n{'=' * 62}")
    print(f"PARAMETRIC GATE  attempt={attempt}  generator={sha}")
    print(f"  n={len(rows)}  correct={correct}  accuracy={accuracy:.4f}")
    print(f"  thresholds: pass <= {config.WS6B_GATE_PASS}, "
          f"fail > {config.WS6B_GATE_FAIL}")
    print(f"  VERDICT: {verdict.upper()}")
    print(f"  spend this phase: ${governor.spent:.4f} of "
          f"${governor.limit:.2f}")
    if correct:
        print("\n  Probes answered WITHOUT history access -- inspect each:")
        for r in rows:
            if r["judge_correct"]:
                print(f"    {r['qid']}: {r['answer'].strip()[:100]!r}")
                print(f"      judge: {r['judge_reason']}")
    print("=" * 62)
    return 0 if verdict == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
