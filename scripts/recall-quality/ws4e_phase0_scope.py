#!/usr/bin/env python
"""WS4e Phase 0 -- scoping, gate K1 and gate K2. Costs $0.

Reproduces every number in spec section 2 from committed artifacts, and
writes the separability denominators BEFORE any generation exists so they
are pre-registered by artifact rather than recomputed once verdicts are in.

Design: results/recall-vs-quality.md

Run:
    PYTHONPATH=. .venv/bin/python scripts/recall-quality/ws4e_phase0_scope.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest  # noqa: E402
from benchlib.recall_quality import depth as ws4e  # noqa: E402
from benchlib.config import (WS4_TOP_K, WS4E_ARMS, WS4E_K_VALUES,  # noqa: E402
                             WS4E_REPLICATE_K, ws4_cache, ws4b_dir,
                             ws4d_cache, ws4d_dir, ws4e_dir)

RET = ws4_cache() / "retrieval"


def load_ids():
    ids, scores = {}, {}
    for arm in WS4E_ARMS:
        f = RET / f"{arm}.npz"
        if not f.exists():
            sys.exit(f"missing retrieval cache for {arm}: {f}")
        d = np.load(f)
        ids[arm], scores[arm] = d["ids"], d["scores"]
    return ids, scores


def committed_ids():
    """(qid, arm) -> retrieved_ids as recorded on a committed row, for the
    WS4e arms that already have k=10 rows. sq8_np1 has none -- its k=10 cell
    is new (spec 3.2)."""
    out = {}
    runs = pd.read_csv("results/ws4/runs.csv")
    runs = runs[(runs.subset == "main") & runs.arm.isin(WS4E_ARMS)]
    for r in runs.itertuples():
        out[(int(r.qid), r.arm)] = json.loads(r.retrieved_ids)
    cp = ws4d_cache() / "phase3_runs_checkpoint.jsonl"
    if cp.exists():
        for line in cp.open():
            row = json.loads(line)
            if row["arm"] in WS4E_ARMS:
                out[(int(row["qid"]), row["arm"])] = json.loads(row["retrieved_ids"])
    return out


def main():
    out = ws4e_dir()
    out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(ws4d_dir() / "eligibility_labels.csv")
    tier0 = pd.read_csv(ws4b_dir() / "tier0_runs_corrected.csv")
    strata = ws4e.build_strata(labels, tier0)
    qids = strata["primary"]
    print(f"strata: " + ", ".join(f"{k}={len(v)}" for k, v in strata.items()))

    ids, scores = load_ids()

    rows = []
    for k in WS4E_K_VALUES:
        sep = ws4e.separable_at(ids, qids, k)
        n_sep = sum(sep.values())
        rows.append({"k": k, "n": len(qids), "n_separable": n_sep,
                     "n_identical": len(qids) - n_sep,
                     "separable_rate": round(n_sep / len(qids), 6),
                     "arms": "|".join(WS4E_ARMS),
                     "stratum": "primary",
                     # Fix round 1, Minor: this used to read WS4E_T0_CELL_K,
                     # a name describing the withdrawn temperature-0 design
                     # (Amendment 1). It happens to equal WS4E_REPLICATE_K,
                     # which is what this column actually flags now: the
                     # depth of the Amendment 1 replicate cell (spec 5.3).
                     "is_replicate_k": k == WS4E_REPLICATE_K})
    ceiling = pd.DataFrame(rows)
    ceiling.to_csv(out / "ceiling.csv", index=False)
    manifest.record(out / "ceiling.csv")
    print(ceiling.to_string(index=False))

    k1 = ws4e.gate_k1({(k, None): qids for k in WS4E_K_VALUES},
                      qids=qids, arms=WS4E_ARMS,
                      arms_by_cell={(k, None): WS4E_ARMS for k in WS4E_K_VALUES},
                      x_axis="recall_at_10")
    k2 = ws4e.gate_k2(ids, scores, qids=qids, committed=committed_ids(),
                      top_k=WS4_TOP_K)
    pd.DataFrame([k1, k2]).to_csv(out / "prefix_integrity.csv", index=False)
    manifest.record(out / "prefix_integrity.csv")
    print(json.dumps(k1, indent=2))
    print(json.dumps(k2, indent=2))

    if not (k1["passed"] and k2["passed"]):
        sys.exit("GATE FAILURE (K1/K2) -- nothing downstream is interpretable")
    print("Phase 0 OK -- K1 and K2 pass; ceiling.csv is now pre-registered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
