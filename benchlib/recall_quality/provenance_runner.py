"""WS4c live half: the build governor, question generation, the paid filters
F3/F4, and the F5 validity judge.

Everything here touches an API or the filesystem. Pure logic lives in
provenance.py and is unit-tested. WS6a's client, retry schedule and
CostGovernor are imported, never modified.

Design: results/recall-vs-quality.md
sections 2A.2 and 7.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from . import core as _pure4
from ..code_retrieval import bench as _ws6a
from ..config import WS4C_BUILD_BUDGET_USD, WS4C_QGEN_MODEL, WS4C_QJUDGE_MODEL, ws4c_cache
from .runner import generate as _ws4_generate
# Re-exported: Task 8's build loop calls provenance_runner.make_client() directly,
# so this is this module's documented entry point for building a client.
from ..agent_harness import _text_of, make_client

__all__ = [
    "BUILD_CHECKPOINT", "build_spent", "build_governor",
    "fill", "generate_question", "answer_with", "judge_validity",
    "QGEN_SCHEMA", "QVALID_SCHEMA", "QValidityError", "make_client",
]

BUILD_CHECKPOINT = ws4c_cache() / "build_checkpoint.jsonl"


def build_spent(path=BUILD_CHECKPOINT) -> float:
    """Billed spend from EVERY RAW LINE of the build checkpoint.

    Deliberately NOT agent_harness.checkpoint_spent, which sums
    load_checkpoint().values() -- a dict keyed by (qid, arm). A candidate
    retried after a crash writes a SECOND line with the same key, the dict
    keeps one, and its charge becomes invisible to the governor. WS6c
    finished about $0.30 over a raised cap for exactly this reason
    (see README.md and results/code-retrieval.md). Money spent twice is spent
    twice.
    """
    p = Path(path)
    if not p.exists():
        return 0.0
    total = 0.0
    with open(p) as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                total += float(json.loads(line).get("cost_total_billed", 0.0))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                print(f"Warning: {p}:{line_num} truncated or malformed, "
                      f"skipped: {exc}", file=sys.stderr)
    return round(total, 6)


def build_governor(path=BUILD_CHECKPOINT) -> _ws6a.CostGovernor:
    """Phase 0b's SEPARATE governor (spec 7). WS4C_BUDGET_USD (75.00) is the
    evaluation's; this one, WS4C_BUILD_BUDGET_USD (14.00), is the build's --
    a low filter yield must halt the build rather than quietly eat the
    eval's money. Per G-A this limit is never raised to reach 600."""
    return _ws6a.CostGovernor(WS4C_BUILD_BUDGET_USD, spent=build_spent(path))


# Question generation is a short structured emission; 512 caps it with room
# for a decline plus its reason. Lives here rather than in config.py because
# spec 2A.4 pins exactly nine constants and this is not one of them -- the
# same reason JUDGE_MAX_TOKENS_SCHEDULE lives in agent_harness.py.
QGEN_MAX_TOKENS = 512
QVALID_MAX_TOKENS_SCHEDULE = (4096, 16384)

QGEN_SCHEMA = {
    "type": "object",
    "properties": {
        "decline": {"type": "boolean"},
        "question": {"type": "string"},
        "gold_answer": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["decline", "question", "gold_answer", "reason"],
    "additionalProperties": False,
}

QVALID_SCHEMA = {
    "type": "object",
    "properties": {
        "valid": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["valid", "reason"],
    "additionalProperties": False,
}


def fill(template: str, **kw) -> str:
    """Placeholder substitution by str.replace, NOT str.format.

    Corpus chunks contain braces ('{{cite web}}' survives in some text).
    str.format would raise KeyError/IndexError on them and kill a paid build
    mid-run, or worse, silently reinterpret them.
    """
    out = template
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def generate_question(client, prompt_template: str, *, text_a: str, text_b: str,
                      page_title: str, bridge_title: str,
                      model: str = WS4C_QGEN_MODEL) -> dict:
    """One candidate question from one chunk pair (spec 2A.2).

    The model must emit a question answerable ONLY by combining A and B,
    never naming the bridge, with a gold answer that is a span of chunk B --
    or DECLINE. A decline is a result, not a retry: spec 2A.2 counts them and
    does not retry them into existence.
    """
    t0 = time.time()
    resp = client.messages.create(
        model=model, max_tokens=QGEN_MAX_TOKENS,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": fill(
            prompt_template, page_title=page_title, bridge_title=bridge_title,
            chunk_a=text_a, chunk_b=text_b)}],
        output_config={"format": {"type": "json_schema", "schema": QGEN_SCHEMA}})
    usage = _ws6a.Usage.from_api(resp.usage)
    raw = _text_of(resp)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"decline": True, "question": "", "gold_answer": "",
                "reason": f"unparseable: {raw[:120]!r}", "usage": usage,
                "stop_reason": resp.stop_reason,
                "wall_clock_s": round(time.time() - t0, 3)}
    return {"decline": bool(payload["decline"]),
            "question": (payload.get("question") or "").strip(),
            "gold_answer": (payload.get("gold_answer") or "").strip(),
            "reason": payload.get("reason", ""), "usage": usage,
            "stop_reason": resp.stop_reason,
            "wall_clock_s": round(time.time() - t0, 3)}


def answer_with(client, system_prompt: str, passages, question: str,
                model: str = WS4C_QGEN_MODEL) -> dict:
    """Answer `question` from `passages` under WS4's exact conditions.

    Delegates to ws4_runner.generate, so max_tokens, thinking-off and the
    message shape are WS4's by construction rather than by copy. That is what
    makes 'the model could not answer this unaided' mean the same thing at
    build time (F3/F4) and at eval time (spec 2A.2).

    passages: [] for F3 (render_context emits NO_PASSAGES), [A] or [B] for F4.
    """
    return _ws4_generate(client, system_prompt, _pure4.render_context(passages),
                         question, model=model)


class QValidityError(RuntimeError):
    """The F5 judge returned nothing parseable after every attempt.

    Carries the Usage already billed, so the caller can charge the governor
    for work that really happened. `ws4_runner.judge` and
    `agent_harness.judge_answer` raise without their usage and the caller
    writes the spend off; WS4c does not inherit that, because two exhausted
    Opus attempts bill about $0.52 -- 3.7% of WS4C_BUILD_BUDGET_USD -- and
    the governor must see it (see README.md for this failure class).

    Raised rather than defaulted to valid=False: recording a judge failure
    as an invalid question would fabricate a filter result.
    """

    def __init__(self, message: str, usage):
        super().__init__(message)
        self.usage = usage


def judge_validity(client, prompt_template: str, *, question: str,
                   gold_answer: str, text_a: str, text_b: str,
                   model: str = WS4C_QJUDGE_MODEL) -> dict:
    """F5: blind validity judge -- well-formed, unambiguous, genuinely 2-hop.

    The only LLM judge in the build (spec 2A.2): F3/F4 are mechanical
    exact_match filters. max_tokens escalates because Opus 5 runs adaptive
    thinking by default and can spend the whole budget thinking, returning no
    text block -- observed in WS6a, see JUDGE_MAX_TOKENS_SCHEDULE.
    """
    filled = fill(prompt_template, question=question, gold_answer=gold_answer,
                  chunk_a=text_a, chunk_b=text_b)
    total = _ws6a.Usage()
    last = ""
    for budget in QVALID_MAX_TOKENS_SCHEDULE:
        resp = client.messages.create(
            model=model, max_tokens=budget,
            messages=[{"role": "user", "content": filled}],
            output_config={"format": {"type": "json_schema",
                                      "schema": QVALID_SCHEMA}})
        total.add(_ws6a.Usage.from_api(resp.usage))
        last = _text_of(resp)
        if last.strip():
            try:
                payload = json.loads(last)
            except json.JSONDecodeError:
                continue
            return {"valid": bool(payload["valid"]),
                    "reason": payload["reason"], "usage": total}
    raise QValidityError(f"F5 judge produced no parseable verdict after "
                         f"{len(QVALID_MAX_TOKENS_SCHEDULE)} attempts "
                         f"(last={last[:120]!r})", total)
