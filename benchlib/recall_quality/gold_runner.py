"""WS4b Tier 0d live half: the tier governor, blind prompt rendering, and the
two model calls. Pure logic lives in gold.py.

Design: results/recall-vs-quality.md

The money machinery here is copied deliberately from WS4c, where both pieces
were added to fix real overspends. Neither is incidental.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..code_retrieval import bench as _ws6a
from ..config import (WS4B_T0D_PROPOSE_MODEL, WS4B_T0D_TIER_CAP_USD,
                      WS4B_T0D_VERIFY_MODEL, ws4b_cache)
from ..code_retrieval.bench import cost_billed
from ..agent_harness import _text_of, append_checkpoint, make_client

__all__ = [
    "CHECKPOINT", "tier_spent", "tier_governor", "charge", "fill",
    "append_checkpoint", "make_client",  # re-exported for Tasks 7 and 8
    "PROPOSE_SCHEMA", "VERIFY_SCHEMA", "PROPOSE_MAX_TOKENS",
    "VERIFY_MAX_TOKENS_SCHEDULE", "propose", "verify", "VerifyError",
    "VERIFIER_EXHAUSTED_REASON",
]

CHECKPOINT = ws4b_cache() / "tier0d_checkpoint.jsonl"

# Sentinel written to a row's verify_reason when VerifyError fires (the
# verifier returned nothing parseable after every retry) -- used at three
# sites: scripts/recall-quality/ws4b_tier0d_refresh.py writes it, and both
# scripts/recall-quality/ws4b_tier0d_refresh.py's resume check and
# scripts/recall-quality/ws4b_tier0d_g4_diagnosis.py's g4_rows reconstruction read it back
# to tell "verifier billed but never returned a verdict" apart from an
# ordinary keep.
VERIFIER_EXHAUSTED_REASON = "verifier exhausted"


def tier_spent(path=CHECKPOINT) -> float:
    """Billed spend from EVERY RAW LINE.

    Not agent_harness.checkpoint_spent, which sums a dict keyed by (qid, arm):
    a row retried after a crash writes a second line with the same key, the
    dict keeps one, and that real charge becomes invisible. WS6c finished
    about $0.30 over a raised cap this way (see README.md).
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
            except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
                print(f"Warning: {p}:{line_num} malformed, skipped: {exc}",
                      file=sys.stderr)
    return round(total, 6)


def tier_governor(path=CHECKPOINT) -> _ws6a.CostGovernor:
    """Capped at WS4B_T0D_TIER_CAP_USD, which sits INSIDE WS4B_BUDGET_USD so a
    runaway tier cannot consume the whole workstream's budget."""
    return _ws6a.CostGovernor(WS4B_T0D_TIER_CAP_USD, spent=tier_spent(path))


def charge(gov, usage, prices: dict, model: str, row: dict) -> float:
    """Record one call's billed cost on `row`, THEN ask the governor.

    CostGovernor.charge() raises before recording. WS6a survives a rejection
    because it checkpoints the row before charging, so a resumed run recovers
    the spend from the file. Tier 0d charges before the row is written, so it
    must keep the cost on the row and in gov.spent itself: record it and top
    up gov.spent before letting BudgetExceeded propagate.
    """
    usd = cost_billed(usage, prices[model])
    row["cost_total_billed"] = round(row.get("cost_total_billed", 0.0) + usd, 6)
    try:
        gov.charge(usd)
    except _ws6a.BudgetExceeded:
        gov.spent = round(gov.spent + usd, 6)
        raise
    return usd


def fill(template: str, **kw) -> str:
    """Placeholder substitution by str.replace, NOT str.format: corpus
    passages contain braces and .format would raise mid-run or silently
    reinterpret them."""
    out = template
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return out


PROPOSE_MAX_TOKENS = 512
VERIFY_MAX_TOKENS_SCHEDULE = (4096, 16384)

PROPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string",
                    "enum": ["keep", "augment", "replace",
                             "unanswerable_from_corpus"]},
        "new_golds": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "new_golds", "reason"],
    "additionalProperties": False,
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {"uphold": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["uphold", "reason"],
    "additionalProperties": False,
}


def propose(client, template: str, *, question: str, golds: str, passages: str,
            model: str = WS4B_T0D_PROPOSE_MODEL) -> dict:
    """Stage 1 (spec 3.2). Sees question, current golds and passages only --
    never an arm's answer, an arm name, or a prior audit label."""
    prompt = fill(template, question=question, golds=golds, passages=passages)
    resp = client.messages.create(
        model=model, max_tokens=PROPOSE_MAX_TOKENS,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema", "schema": PROPOSE_SCHEMA}})
    usage = _ws6a.Usage.from_api(resp.usage)
    raw = _text_of(resp)
    try:
        payload = json.loads(raw)
        verdict, new_golds, reason = (
            payload["verdict"], payload["new_golds"], payload["reason"])
    except (json.JSONDecodeError, KeyError):
        # A malformed-but-parseable payload (missing a required key) is
        # handled the same way as unparseable JSON -- both are a schema
        # violation the caller should treat as "propose could not be
        # trusted", not a KeyError that escapes uncaught and drops the
        # Usage already billed for this call.
        return {"verdict": "keep", "new_golds": [],
                "reason": f"unparseable: {raw[:120]!r}", "usage": usage,
                "prompt": prompt}
    return {"verdict": verdict, "new_golds": new_golds, "reason": reason,
            "usage": usage, "prompt": prompt}


class VerifyError(RuntimeError):
    """The verifier returned nothing parseable after every attempt. Carries the
    Usage already billed so the caller can charge the governor for work that
    really happened, and the last-attempted PROMPT so a caller can still hand
    it to gate G4 -- a billed call that never returned a verdict is still a
    call that was made, and its blindness still needs asserting."""

    def __init__(self, message: str, usage, prompt: str):
        super().__init__(message)
        self.usage = usage
        self.prompt = prompt


def verify(client, template: str, *, question: str, golds: str, passages: str,
           proposed, model: str = WS4B_T0D_VERIFY_MODEL) -> dict:
    """Stage 2 (spec 3.3). Sees the proposed gold list but NOT stage 1's
    reasoning, so it re-derives the judgement rather than grading an argument.
    Standing instruction in the prompt: `keep` is the default.

    max_tokens escalates because Opus runs adaptive thinking by default and can
    spend the whole budget thinking, returning no text block.
    """
    prompt = fill(template, question=question, golds=golds, passages=passages,
                  proposed=json.dumps(list(proposed)))
    total = _ws6a.Usage()
    last = ""
    for budget in VERIFY_MAX_TOKENS_SCHEDULE:
        resp = client.messages.create(
            model=model, max_tokens=budget,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": {"type": "json_schema",
                                      "schema": VERIFY_SCHEMA}})
        total.add(_ws6a.Usage.from_api(resp.usage))
        last = _text_of(resp)
        if last.strip():
            try:
                payload = json.loads(last)
                uphold, reason = bool(payload["uphold"]), payload["reason"]
            except (json.JSONDecodeError, KeyError):
                # Same treatment as an unparseable response: retry on the
                # next (larger) token budget rather than letting a KeyError
                # from a malformed-but-parseable payload escape uncaught.
                continue
            return {"uphold": uphold, "reason": reason,
                    "usage": total, "prompt": prompt}
    raise VerifyError(f"verifier produced no parseable verdict after "
                      f"{len(VERIFY_MAX_TOKENS_SCHEDULE)} attempts "
                      f"(last={last[:120]!r})", total, prompt)
