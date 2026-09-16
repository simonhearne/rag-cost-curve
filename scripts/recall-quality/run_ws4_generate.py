#!/usr/bin/env python
"""WS4 generation + judging for one or more arms, checkpointed and resumable.

    python scripts/recall-quality/run_ws4_generate.py sq8_np512 parametric
    python scripts/recall-quality/run_ws4_generate.py --arms-csv     # every arm in arms.csv

Arms are retrieval keys (npz stems under data/ws4_cache/retrieval/) or the
literal `parametric`. Runs contiguously per arm; question order is a seeded
shuffle within each arm so a partial run is not biased toward low row ids.
Requires results/ws4/questions.csv (the committed draw) and phase 0's
passage cache. Never writes to results/ -- the notebook does.

Design: results/recall-vs-quality.md
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib.code_retrieval import bench as ws6a  # noqa: E402
from benchlib.recall_quality import runner as ws4_runner  # noqa: E402
from benchlib.config import (SEED, WS4_SCALE, WS4_TOP_K, gt_path,  # noqa: E402
                             ws4_cache, ws4_dir)
from benchlib.quantization import recall_at_k  # noqa: E402

RET = ws4_cache() / "retrieval"


def main(argv):
    if "--arms-csv" in argv:
        arms = pd.read_csv(ws4_dir() / "arms.csv")["arm_key"].tolist()
        argv = [a for a in argv if a != "--arms-csv"]
    else:
        arms = []
    arms += argv
    if not arms:
        sys.exit("no arms given")
    if not (ws4_cache() / "question_flags.parquet").exists():
        sys.exit("phase 0 (scripts/recall-quality/build_ws4_filter.py) has not completed")

    qdf = pd.read_csv(ws4_dir() / "questions.csv")
    rows = qdf["row"].to_numpy()
    ws4_gt = np.load(gt_path(WS4_SCALE))["ws4_indices"]
    questions = {int(r.row): dict(question=r.question,
                                  golds=json.loads(r.golds),
                                  subset=r.subset, recall={})
                 for r in qdf.itertuples()}

    retrieval, needed, skipped = {}, set(), []
    for arm in arms:
        if arm == "parametric":
            retrieval[arm] = None
            continue
        f = RET / f"{arm}.npz"
        if not f.exists():
            print(f"SKIPPING {arm}: no retrieval yet ({f})", flush=True)
            skipped.append(arm)
            continue
        ids = np.load(f)["ids"]
        assert ids.shape == (ws4_gt.shape[0], WS4_TOP_K), ids.shape
        retrieval[arm] = ids
        for r in rows:
            questions[int(r)]["recall"][arm] = recall_at_k(
                ids[r:r + 1], ws4_gt[r:r + 1], WS4_TOP_K)
            needed.update(int(i) for i in ids[r] if i >= 0)
    passages = ws4_runner.load_passages(sorted(needed), WS4_SCALE) if needed \
        else pd.DataFrame(columns=["title", "text"])

    arms = [a for a in arms if a not in skipped]
    pairs = []
    for arm in arms:
        order = [int(r) for r in rows]
        random.Random(f"{SEED}:{arm}").shuffle(order)
        pairs += [(q, arm) for q in order]

    sys_p, judge_p = ws4_runner.load_prompts()
    prices = ws6a.load_pricing(Path("results/ws6a/pricing.csv"))
    gov = ws4_runner.governor_from_checkpoints()
    client = ws4_runner.make_client()
    print(f"arms: {arms}; {len(pairs)} pairs; governor ${gov.spent:.2f} "
          f"of ${gov.limit:.2f}", flush=True)
    try:
        ws4_runner.run_pairs(client, pairs, questions=questions,
                             retrieval=retrieval, passages=passages,
                             system_prompt=sys_p, judge_prompt=judge_p,
                             prices=prices, governor=gov,
                             log=lambda m: print(m, flush=True))
    except ws6a.BudgetExceeded as exc:
        print(f"GOVERNOR STOP: {exc}", flush=True)
        return 2
    print(f"done; governor ${gov.spent:.4f}; skipped {skipped}", flush=True)
    return 3 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
