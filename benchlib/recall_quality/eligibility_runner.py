"""WS4d Phase 1 live half: the tier governor and the single blind label call.
Pure logic lives in eligibility.py.

Design: results/recall-vs-quality.md

There is deliberately NO propose/verify pair here. Tier 0d's verifier saw the
proposal, which measures an uphold rate rather than independent agreement --
results/recall-vs-quality.md is the record of why that distinction matters. Phase 1
runs two labellers who never see each other's output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import eligibility as ws4d
from ..code_retrieval import bench as _ws6a
from ..config import WS4D_PHASE1_CAP_USD, ws4d_cache
from .gold_runner import charge, fill
from ..agent_harness import _text_of, append_checkpoint, make_client

__all__ = ["CHECKPOINT", "LABEL_SCHEMA", "LABEL_MAX_TOKENS", "tier_spent",
           "tier_governor", "gt_passages", "label", "charge", "fill",
           "append_checkpoint", "make_client"]

CHECKPOINT = ws4d_cache() / "phase1_checkpoint.jsonl"
LABEL_MAX_TOKENS = 512

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(ws4d.LABELS)},
        "reason": {"type": "string"},
    },
    "required": ["label", "reason"],
    "additionalProperties": False,
}


def tier_spent(path=CHECKPOINT) -> float:
    """Billed spend from EVERY RAW LINE -- see ws4b_gold_runner.tier_spent for
    why this is not keyed by qid."""
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
    return _ws6a.CostGovernor(WS4D_PHASE1_CAP_USD, spent=tier_spent(path))


def gt_passages(qids) -> dict[int, str]:
    """qid -> the concatenated text of the ground-truth top-10 passages.

    ARM-INDEPENDENT by construction (spec §3.1): this is the best retrieval
    the corpus admits, which every arm meets identically. Using any arm's
    retrieved ids here would make eligibility a function of the independent
    variable, which is the one thing this design cannot do.

    Imported locally: pandas/numpy at module import would slow every caller
    of this module, and only the two Phase 1 scripts need this.
    """
    import numpy as np
    import pandas as pd

    from ..config import DATA_DIR, WS4_TOP_K

    tbl = pd.read_parquet(DATA_DIR / "ws4_cache" / "passages.parquet").set_index("row")
    idx = np.load(DATA_DIR / "gt_10m_top100.npz")["ws4_indices"]
    out = {}
    for q in qids:
        rows = [int(x) for x in idx[int(q)][:WS4_TOP_K] if int(x) >= 0]
        out[int(q)] = " ".join(str(tbl.at[x, "text"]) for x in rows
                               if x in tbl.index)
    return out


def label(client, template: str, *, question: str, golds: str, passages: str,
          model: str) -> dict:
    """One blind eligibility judgement (spec §3.1).

    Sees question, golds and the ground-truth top-10 passages. Never an arm's
    answer, an arm name, a recall, a judge verdict, a prior audit label, or
    the other labeller's output.

    An unparseable or off-vocabulary response falls back to `eligible`, never
    to an ineligible class: fabricating an ineligible label would silently
    remove a question from the benchmark, which is the one error this design
    cannot tolerate. The reason field records what actually happened so the
    fallback is visible in the committed CSV.
    """
    prompt = fill(template, question=question, golds=golds, passages=passages)
    resp = client.messages.create(
        model=model, max_tokens=LABEL_MAX_TOKENS,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
        output_config={"format": {"type": "json_schema",
                                  "schema": LABEL_SCHEMA}})
    usage = _ws6a.Usage.from_api(resp.usage)
    raw = _text_of(resp)
    try:
        payload = json.loads(raw)
        lab, reason = payload["label"], payload["reason"]
    except (json.JSONDecodeError, KeyError):
        return {"label": "eligible", "reason": f"unparseable: {raw[:120]!r}",
                "usage": usage, "prompt": prompt}
    err = ws4d.validate_label(lab)
    if err:
        return {"label": "eligible", "reason": f"{err}; raw={raw[:80]!r}",
                "usage": usage, "prompt": prompt}
    return {"label": lab, "reason": reason, "usage": usage, "prompt": prompt}
