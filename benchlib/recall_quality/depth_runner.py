"""WS4e live half: per-cell checkpoints and the tier governor.

Design: results/recall-vs-quality.md

Pure logic lives in depth.py. Generation and judging reuse
runner.run_pairs unchanged apart from its new top_k/temperature
arguments (Task 2).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..code_retrieval import bench as _ws6a
from ..config import WS4E_CAP_USD, ws4e_cache

__all__ = ["cell_checkpoint", "REJUDGE_NAME", "rejudge_checkpoint",
           "tier_spent", "tier_governor", "cell_rows", "cell_rows_first_wins"]

REJUDGE_NAME = "rejudge_checkpoint.jsonl"


def _cache_dir() -> Path:
    """Indirection so tests can redirect the whole tier to tmp_path."""
    return ws4e_cache()


def cell_checkpoint(k: int, temperature, replicate: bool = False) -> Path:
    """One file per (depth, temperature, replicate) cell.

    ⚠️ agent_harness.load_checkpoint keys rows by (qid, arm), and in WS4e the
    same (qid, arm) recurs at four depths. A single shared file would let a
    k=1 row satisfy the k=10 cell and silently skip thousands of rows. The
    cell IS the checkpoint.

    ⚠️ Amendment 1's replicate cell (spec 5.3) re-runs WS4E_REPLICATE_ARM at
    WS4E_REPLICATE_K = 1 with WS4E_TEMPERATURE = None -- the EXACT (k,
    temperature) of the primary k=1 sweep cell. Without `replicate` as a
    third key, this cell's checkpoint path would collide with the primary
    k=1 file: load_checkpoint's (qid, arm) dedup would treat every replicate
    row as already recorded, write zero new rows, make R trivially 0.0, and
    fire gate K6 for an entirely spurious reason. `replicate` breaks the
    collision.
    """
    t = "tdefault" if temperature is None else f"t{temperature}"
    name = f"runs_k{int(k)}_{t}"
    if replicate:
        name += "_rep"
    return _cache_dir() / f"{name}.jsonl"


def rejudge_checkpoint() -> Path:
    return _cache_dir() / REJUDGE_NAME


def tier_spent() -> float:
    """Billed spend over EVERY raw line of every WS4e checkpoint.

    Not deduped by (qid, arm): money billed on a line a later write
    superseded is still money spent. results/code-retrieval.md is the record
    of a governor that could not see ~$1.84 for exactly this reason.
    """
    total = 0.0
    for p in sorted(_cache_dir().glob("*.jsonl")):
        with open(p) as fh:
            for line_num, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    total += float(json.loads(line).get("cost_total_billed", 0.0))
                except (json.JSONDecodeError, AttributeError, TypeError,
                        ValueError) as exc:
                    print(f"Warning: {p}:{line_num} malformed, skipped: {exc}",
                          file=sys.stderr)
    return round(total, 6)


def tier_governor() -> _ws6a.CostGovernor:
    """WS4e's OWN budget line.

    ⚠️ Never ws4_runner.governor_from_checkpoints(): that reads WS4's separate
    $75 line and would conflate two budgets.
    """
    return _ws6a.CostGovernor(WS4E_CAP_USD, spent=tier_spent())


def cell_rows(k: int, temperature, replicate: bool = False) -> dict:
    """{(qid, arm): row} already recorded for one cell."""
    from ..agent_harness import load_checkpoint
    return load_checkpoint(cell_checkpoint(k, temperature, replicate=replicate))


def cell_rows_first_wins(k: int, temperature, replicate: bool = False) -> dict:
    """{(qid, arm): row}, but the FIRST recorded line wins rather than
    agent_harness.load_checkpoint's last-wins.

    Fix round 1, IMPORTANT 4: two concurrency incidents wrote duplicate raw
    lines for the same (qid, arm) key at k=1 (549) and k=5 (6) -- a
    concurrency accident, not a data-quality choice this project should be
    indifferent to. This is a SENSITIVITY CHECK on that dedup rule only:
    every committed number still uses `cell_rows` / `load_checkpoint`
    (last-wins) unchanged.
    """
    path = cell_checkpoint(k, temperature, replicate=replicate)
    done: dict = {}
    if not path.exists():
        return done
    with open(path) as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: {path}:{line_num} truncated or malformed, "
                      f"skipped: {exc}", file=sys.stderr)
                continue
            key = (row["qid"], row["arm"])
            if key not in done:          # FIRST occurrence wins
                done[key] = row
    return done
