"""WS4 live half: passage loading, the generator call, the blind judge, and
the checkpointed run loop under the budget governor.

Everything here touches an API or the filesystem. Pure logic lives in
benchlib/recall_quality/core.py and is unit-tested. WS6a's judge schema,
retry schedule, checkpoint helpers and CostGovernor are imported, never
modified.

Design: results/recall-vs-quality.md
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import core as _pure
from ..code_retrieval import bench as _ws6a
from ..config import (WS4_BUDGET_USD, WS4_GEN_MAX_TOKENS, WS4_GEN_MODEL,
                     WS4_JUDGE_MODEL, WS4_TOP_K, ids_path, ws4_cache, ws4_dir)
from ..agent_harness import (JUDGE_MAX_TOKENS_SCHEDULE, JUDGE_SCHEMA, JudgeError,
                          _text_of, append_checkpoint, checkpoint_spent,
                          load_checkpoint, make_client)

CHECKPOINT = ws4_cache() / "runs_checkpoint.jsonl"
RELIABILITY_CHECKPOINT = ws4_cache() / "judge_reliability_checkpoint.jsonl"
PASSAGES = ws4_cache() / "passages.parquet"


# ------------------------------------------------------------ passages ----


def load_passages(global_rows, scale: str = "10m") -> pd.DataFrame:
    """title/text for corpus row indices, read from the shards on demand and
    cached in PASSAGES. Rows already cached are not re-read."""
    import pyarrow.parquet as pq

    from ..download import shard_local_path

    want = np.unique(np.asarray(global_rows, dtype=np.int64))
    have = pd.read_parquet(PASSAGES) if PASSAGES.exists() else pd.DataFrame(
        columns=["row", "title", "text"]).astype({"row": np.int64})
    missing = np.setdiff1d(want, have["row"].to_numpy())
    if len(missing):
        ids = pd.read_parquet(ids_path(scale), columns=["title", "shard",
                                                         "row_in_shard"])
        sub = ids.iloc[missing]
        out = []
        for shard, grp in sub.groupby("shard"):
            tbl = pq.read_table(shard_local_path(int(shard)), columns=["text"])
            texts = tbl.column("text").to_pylist()
            for row, title, ris in zip(grp.index, grp["title"], grp["row_in_shard"]):
                out.append((int(row), title, texts[int(ris)]))
        new = pd.DataFrame(out, columns=["row", "title", "text"])
        have = pd.concat([have, new], ignore_index=True)
        have = have.drop_duplicates("row").sort_values("row").reset_index(drop=True)
        PASSAGES.parent.mkdir(parents=True, exist_ok=True)
        have.to_parquet(PASSAGES, index=False)
    return have.set_index("row").loc[want]


def passages_for(retrieved_ids, table: pd.DataFrame) -> list[dict]:
    """Rank-ordered [{title, text}] for one question's top-k ids (-1 padded
    ids are skipped)."""
    return [{"title": table.at[int(i), "title"], "text": table.at[int(i), "text"]}
            for i in retrieved_ids[:WS4_TOP_K] if int(i) >= 0]


# ---------------------------------------------------------- generator ----


def generate(client, system_prompt: str, context: str, question: str,
             model: str = WS4_GEN_MODEL, temperature=None) -> dict:
    """One fixed-pipeline generation. Thinking disabled, no tools, no cache
    marker (the prefix is below Sonnet 5's cacheable minimum; spec 5.2).

    `temperature` is OMITTED from the request when None, so the default path
    is byte-identical to every call that produced a committed WS4 number.
    WS4e spec 3.5 keeps the main sweep on that path and passes 0 only in its
    Phase 1b validation cell.
    """
    t0 = time.time()
    extra = {} if temperature is None else {"temperature": temperature}
    resp = client.messages.create(
        model=model, max_tokens=WS4_GEN_MAX_TOKENS,
        thinking={"type": "disabled"},
        system=system_prompt,
        messages=[{"role": "user",
                   "content": _pure.user_message(context, question)}],
        **extra)
    usage = _ws6a.Usage.from_api(resp.usage)
    return {"answer": _text_of(resp).strip(), "stop_reason": resp.stop_reason,
            "usage": usage, "wall_clock_s": round(time.time() - t0, 3),
            "request_id": getattr(resp, "_request_id", None)}


# -------------------------------------------------------------- judge ----


def judge(client, prompt_template: str, question: str, golds, answer: str,
          model: str = WS4_JUDGE_MODEL) -> dict:
    """Blind judge: question, gold aliases, candidate. Never the arm, never
    the passages. Same retry schedule and usage accounting as WS6a/b."""
    if not (answer or "").strip():
        return {"judge_correct": False, "judge_reason": "empty answer",
                "usage": _ws6a.Usage()}
    filled = prompt_template.format(
        question=question,
        gold_answers="\n".join(f"- {g}" for g in golds),
        answer=answer)
    total = _ws6a.Usage()
    last = ""
    for budget in JUDGE_MAX_TOKENS_SCHEDULE:
        resp = client.messages.create(
            model=model, max_tokens=budget,
            messages=[{"role": "user", "content": filled}],
            output_config={"format": {"type": "json_schema",
                                      "schema": JUDGE_SCHEMA}})
        total.add(_ws6a.Usage.from_api(resp.usage))
        last = _text_of(resp)
        if last.strip():
            try:
                payload = json.loads(last)
            except json.JSONDecodeError:
                continue
            return {"judge_correct": bool(payload["correct"]),
                    "judge_reason": payload["reason"], "usage": total}
    raise JudgeError(f"judge produced no parseable verdict after "
                     f"{len(JUDGE_MAX_TOKENS_SCHEDULE)} attempts "
                     f"(last={last[:120]!r})")


# ---------------------------------------------------------- run loop ----


def governor_from_checkpoints() -> _ws6a.CostGovernor:
    """Seeded from EVERY WS4 checkpoint (main run + reliability re-judge +
    smoke), so no phase gets its own fresh $75."""
    spent = sum(checkpoint_spent(p) for p in ws4_cache().glob("*checkpoint*.jsonl"))
    return _ws6a.CostGovernor(WS4_BUDGET_USD, spent=round(spent, 6))


def run_pairs(client, pairs, *, questions: dict, retrieval: dict,
              passages: pd.DataFrame, system_prompt: str, judge_prompt: str,
              prices: dict, governor, checkpoint=CHECKPOINT, log=print,
              top_k: int = WS4_TOP_K, temperature=None):
    """pairs: iterable of (qid, arm). questions: qid -> {question, golds,
    subset}. retrieval: arm -> (nq_all, 10) ids indexed by validation row,
    or None for the parametric arm. Checkpoints BEFORE charging; a judge
    failure leaves the pair un-checkpointed so a resume retries it."""
    gen_price, judge_price = prices[WS4_GEN_MODEL], prices[WS4_JUDGE_MODEL]
    done = load_checkpoint(checkpoint)
    todo = [(q, a) for q, a in pairs if (str(q), a) not in done]
    log(f"{len(done)} pair(s) checkpointed; {len(todo)} to run; governor "
        f"${governor.spent:.2f} of ${governor.limit:.2f}")
    for qid, arm in todo:
        q = questions[qid]
        ids = retrieval.get(arm)
        if ids is None:
            ctx, ret_ids, ap_k, r10 = _pure.NO_PASSAGES, [], None, None
        else:
            ret_ids = [int(i) for i in ids[qid][:top_k] if int(i) >= 0]
            ps = passages_for(ret_ids, passages)
            ctx = _pure.render_context(ps)
            ap_k = _pure.answer_present(q["golds"], [p["text"] for p in ps])
            # recall@10 is the ARM's identity and is held fixed across every
            # depth (WS4e spec 6.4): the x-axis must not move with the
            # treatment. It is NOT recomputed at top_k.
            r10 = q["recall"].get(arm)
        g = generate(client, system_prompt, ctx, q["question"],
                     temperature=temperature)
        if g["usage"].cache_read_input_tokens or g["usage"].cache_creation_input_tokens:
            raise RuntimeError("cache tokens reported on a run the spec says "
                               "cannot cache (5.2) -- investigate before spending more")
        try:
            v = judge(client, judge_prompt, q["question"], q["golds"], g["answer"])
        except Exception as exc:  # noqa: BLE001 -- never fabricate a verdict
            log(f"  {qid} {arm}: JUDGE FAILED ({exc}) -- not checkpointed")
            continue
        gen_cost = _ws6a.cost_billed(g["usage"], gen_price)
        judge_cost = _ws6a.cost_billed(v["usage"], judge_price)
        row = {
            "qid": str(qid), "arm": arm, "subset": q["subset"],
            "question": q["question"], "golds": json.dumps(q["golds"]),
            "answer": g["answer"], "stop_reason": g["stop_reason"],
            "retrieved_ids": json.dumps(ret_ids),
            "recall_at_10": r10,
            "answer_presence_at_10": ap_k if top_k == WS4_TOP_K else None,
            "answer_presence_at_k": ap_k,
            "k": top_k,
            "temperature": temperature,
            "em": _pure.exact_match(g["answer"], q["golds"]),
            "f1": round(_pure.f1(g["answer"], q["golds"]), 4),
            "idk": _pure.is_idk(g["answer"]),
            "judge_correct": v["judge_correct"], "judge_reason": v["judge_reason"],
            "input_tokens": g["usage"].input_tokens,
            "output_tokens": g["usage"].output_tokens,
            "cache_creation_input_tokens": g["usage"].cache_creation_input_tokens,
            "cache_read_input_tokens": g["usage"].cache_read_input_tokens,
            "judge_input_tokens": v["usage"].input_tokens,
            "judge_output_tokens": v["usage"].output_tokens,
            "cost_gen_usd": gen_cost, "cost_judge_usd": judge_cost,
            "cost_total_billed": round(gen_cost + judge_cost, 6),
            "wall_clock_s": g["wall_clock_s"], "request_id": g["request_id"],
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gen_model": WS4_GEN_MODEL, "judge_model": WS4_JUDGE_MODEL,
        }
        append_checkpoint(checkpoint, row)
        governor.charge(row["cost_total_billed"])  # raises BudgetExceeded
        mark = "OK " if v["judge_correct"] else "   "
        log(f"  {mark}{qid:>5} {arm:<18} r10={'-' if r10 is None else f'{r10:.1f}'} "
            f"${row['cost_total_billed']:.4f} {g['answer'][:48]!r}")
    return load_checkpoint(checkpoint)


def rejudge_sample(client, rows, judge_prompt: str, prices: dict, governor,
                   checkpoint=RELIABILITY_CHECKPOINT, log=print) -> dict:
    """Spec 5.4: judge a fixed sample a second time; returns {(qid, arm):
    row}. Checkpointed under its own file, charged to the same governor."""
    judge_price = prices[WS4_JUDGE_MODEL]
    done = load_checkpoint(checkpoint)
    for r in rows:
        key = (r["qid"], r["arm"])
        if key in done:
            continue
        v = judge(client, judge_prompt, r["question"], json.loads(r["golds"]),
                  r["answer"])
        cost = _ws6a.cost_billed(v["usage"], judge_price)
        row = {"qid": r["qid"], "arm": r["arm"],
               "first_verdict": bool(r["judge_correct"]),
               "second_verdict": v["judge_correct"],
               "second_reason": v["judge_reason"], "cost_total_billed": cost}
        append_checkpoint(checkpoint, row)
        governor.charge(cost)
        log(f"  rejudge {r['qid']} {r['arm']}: {row['first_verdict']} -> "
            f"{row['second_verdict']}")
    return load_checkpoint(checkpoint)


def load_prompts() -> tuple[str, str]:
    return ((ws4_dir() / "answer_prompt.txt").read_text().strip(),
            (ws4_dir() / "judge_prompt.txt").read_text())


__all__ = ["make_client", "generate", "judge", "run_pairs", "rejudge_sample",
           "load_passages", "passages_for", "governor_from_checkpoints",
           "load_prompts", "CHECKPOINT", "PASSAGES"]
