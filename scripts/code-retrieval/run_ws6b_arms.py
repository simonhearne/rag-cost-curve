#!/usr/bin/env python3
"""WS6b phase 5: the full run across three arms and three history sizes.

Execution is CONTIGUOUS per (arm, size) cell, not interleaved. That is a
deliberate departure from WS6a, which round-robined arms to control
time-of-day API variance: contiguous grouping is what lets a replay prefix be
written to cache once and read n-1 times, and the cache is C9's entire
steelman. The price is that latency is confounded with execution order and
cannot be compared across arms (spec 5.6). Token and cost columns are immune,
and started_at is recorded on every row so the confound is falsifiable.

Resumable: re-run after any interruption.

Design: results/code-retrieval.md
"""

import csv
import json
import random
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
CHECKPOINT = CACHE / "arms_checkpoint.jsonl"
PARAMETRIC_CHECKPOINT = CACHE / "parametric_checkpoint.jsonl"
TRANSCRIPTS = OUT / "transcripts"


def cell_plan():
    """The seven (arm, size) cells, in execution order.

    Probe order is shuffled from SEED within each cell; the cells themselves
    are fixed so the replay cache is written once per size.
    """
    probes = list(csv.DictReader((OUT / "probes.csv").open()))
    by_set = {}
    for p in probes:
        by_set.setdefault(p["probe_set"], []).append(p["probe_id"])
    scaling = by_set["scaling"]
    all_l = scaling + by_set["depth"] + by_set["superseded"]

    def shuffled(ids, tag):
        out = list(ids)
        random.Random(f"{config.SEED}:{tag}").shuffle(out)
        return out

    return [
        {"arm": "memsearch", "size": "s", "probe_ids": shuffled(scaling, "ms")},
        {"arm": "replay", "size": "s", "probe_ids": shuffled(scaling, "rs")},
        {"arm": "memsearch", "size": "m", "probe_ids": shuffled(scaling, "mm")},
        {"arm": "replay", "size": "m", "probe_ids": shuffled(scaling, "rm")},
        {"arm": "memsearch", "size": "l", "probe_ids": shuffled(all_l, "ml")},
        # One attempt only, to capture the API's verbatim error (spec 8.4).
        {"arm": "replay", "size": "l", "probe_ids": all_l[:1]},
        {"arm": "replay_trunc", "size": "l", "probe_ids": shuffled(all_l, "tl")},
    ]


def retrieved_chars(turns_detail) -> int:
    """Untruncated size of every tool_result payload, summed.

    The analogue of WS6a's median-12,014-chars-per-search_code finding, which
    was its sharpest mechanistic result -- so this column has to be the real
    length, not the transcript's truncated copy.

    The key is `content_length`. _compact_blocks truncates tool_result content
    to 2,000 chars for the transcript and records the true length beside it;
    reading len(content) here would silently cap every retrieval at 2,000 and
    make a 12,000-char payload indistinguishable from a 2,000-char one.
    """
    total = 0
    for turn in turns_detail or []:
        for block in (turn.get("content") or []):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                total += int(block.get("content_length",
                                       len(str(block.get("content", "")))))
    return total


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    probes = {p["probe_id"]: p for p in
              csv.DictReader((OUT / "probes.csv").open())}
    stats = {r["size"]: r for r in
             csv.DictReader((OUT / "corpus_stats.csv").open())}
    prices = ws6a.load_pricing(OUT / "pricing.csv")
    judge_prompt = load_judge_prompt(OUT / "judge_prompt.txt")
    agent_price = prices[config.WS6B_AGENT_MODEL]
    judge_price = prices[config.WS6B_JUDGE_MODEL]
    client = make_client()

    # Cumulative across EVERY phase, including the parametric gate.
    spent = checkpoint_spent(CHECKPOINT) + checkpoint_spent(PARAMETRIC_CHECKPOINT)
    governor = ws6a.CostGovernor(config.WS6B_BUDGET_USD, spent=spent)
    done = load_checkpoint(CHECKPOINT)
    print(f"governor: ${governor.spent:.4f} of ${governor.limit:.2f}; "
          f"{len(done)} run(s) checkpointed")

    n_l = int(stats["l"]["files"])
    first_kept = int(stats["l"]["first_kept_day"])
    fit_l = float(stats["l"]["fit_fraction"])

    for cell in cell_plan():
        arm, size = cell["arm"], cell["size"]
        key = f"{arm}@{size}"
        n = int(stats[size]["files"])

        corpus_text, tools, fit = "", None, 1.0
        if arm == "replay":
            corpus_text = ws6b_runner.load_corpus_text(
                config.WS6B_CORPUS_DIR, 0, n)
        elif arm == "replay_trunc":
            corpus_text = ws6b_runner.load_corpus_text(
                config.WS6B_CORPUS_DIR, first_kept, n_l)
            fit = fit_l
        else:
            tools = ws6b_runner.make_memsearch_tools(
                config.WS6B_COLLECTIONS[size], config.WS6B_CORPUS_DIR)

        todo = [p for p in cell["probe_ids"] if (p, key) not in done]
        print(f"\n=== {key}  {len(todo)}/{len(cell['probe_ids'])} to run "
              f"({len(corpus_text):,} corpus chars) ===")

        for pid in todo:
            probe = probes[pid]
            r = ws6b_runner.run_arm_ws6b(
                client, arm, pid, probe["question"],
                corpus_text=corpus_text, tools=tools)

            if r.error == ws6b_runner.WINDOW_EXCEEDED:
                # Spec 8.4: a hard capability boundary with a NULL cost, never
                # an extrapolated one.
                row = {"qid": pid, "arm": key, "probe_set": probe["probe_set"],
                       "status": ws6b_runner.WINDOW_EXCEEDED,
                       "api_error": r.stop_reason, "answer": "",
                       "cost_billed_usd": None, "cost_list_usd": None,
                       "cost_total_billed": 0.0, "prompt_tokens": None,
                       "judge_correct": None, "started_at": r.started_at}
                append_checkpoint(CHECKPOINT, row)
                print(f"  {pid}: WINDOW EXCEEDED -- {r.stop_reason[:140]}")
                continue
            if r.error:
                print(f"  {pid}: ERROR {r.error[:120]} -- retried on resume")
                continue

            try:
                verdict = ws6b_runner.judge_ws6b(
                    client, judge_prompt, probe["question"],
                    probe["expected_answer"], probe["stale_answer"], r.answer)
            except Exception as exc:
                # Never record a judge failure as a wrong answer -- that
                # fabricates a result. Un-checkpointed, so a resume retries.
                print(f"  {pid}: JUDGE FAILED ({exc}) -- not checkpointed")
                continue

            agent_billed = ws6a.cost_billed(r.usage, agent_price)
            judge_billed = ws6a.cost_billed(verdict["usage"], judge_price)
            row = {
                "qid": pid, "arm": key, "arm_base": arm, "size": size,
                "probe_set": probe["probe_set"], "status": "ok",
                "answer": r.answer, "turns": r.turns,
                "hit_turn_cap": r.hit_turn_cap, "stop_reason": r.stop_reason,
                "wall_clock_s": r.wall_clock_s, "started_at": r.started_at,
                "input_tokens": r.usage.input_tokens,
                "output_tokens": r.usage.output_tokens,
                "cache_creation_input_tokens": r.usage.cache_creation_input_tokens,
                "cache_read_input_tokens": r.usage.cache_read_input_tokens,
                "prompt_tokens": ws6b.prompt_tokens(r.usage),
                "cost_billed_usd": agent_billed,
                "cost_list_usd": ws6a.cost_uncached_list(r.usage, agent_price),
                "cost_total_billed": round(agent_billed + judge_billed, 6),
                "judge_correct": verdict["judge_correct"],
                "judge_reason": verdict["judge_reason"],
                "mechanical_correct": ws6b.score_mechanical(
                    r.answer, probe["expected_answer"]),
                "stale": ws6b.score_stale(r.answer, probe["stale_answer"]),
                "retrieved_chars": retrieved_chars(r.turns_detail),
                "fit_fraction": fit,
                "fact_in_window": probe["fact_in_window"],
                "depth_l": probe["depth_l"], "error": "",
            }
            # Checkpoint BEFORE charging: a crash never loses a completed pair.
            append_checkpoint(CHECKPOINT, row)
            governor.charge(row["cost_total_billed"])
            (TRANSCRIPTS / f"{pid}__{key}.json").write_text(
                json.dumps({"probe_id": pid, "arm": key,
                            "question": probe["question"],
                            "expected": probe["expected_answer"],
                            "turns_detail": r.turns_detail}, indent=2,
                           default=str))
            mark = "OK " if verdict["judge_correct"] else "   "
            print(f"  {mark}{pid} tok={row['prompt_tokens']:>9,} "
                  f"turns={r.turns:>2} ${row['cost_billed_usd']:.4f} "
                  f"{r.answer.strip()[:50]!r}")

    rows = list(load_checkpoint(CHECKPOINT).values())
    fields = sorted({k for r in rows for k in r})
    with open(OUT / "runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    manifest.record(OUT / "runs.csv")
    print(f"\n{len(rows)} rows written. governor: ${governor.spent:.4f} "
          f"of ${governor.limit:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
