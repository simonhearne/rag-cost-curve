#!/usr/bin/env python
"""WS4 phase 0: passage cache + answer-presence flags for every NQ-Open
validation question against its EXACT 10M top-100 (spec 3.1).

Writes data/ws4_cache/question_flags.parquet (one row per validation
question) and fills data/ws4_cache/passages.parquet with every GT top-100
passage. Needs no Milvus and no API. The notebook draws the samples and
writes the committed CSVs from these flags.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib.recall_quality import core  # noqa: E402
from benchlib.config import DATA_DIR, WS4_SCALE, gt_path, ws4_cache  # noqa: E402
from benchlib.recall_quality.runner import load_passages  # noqa: E402

OUT = ws4_cache() / "question_flags.parquet"


def main():
    q = pd.read_parquet(DATA_DIR / "queries_ws4.parquet")
    gt = np.load(gt_path(WS4_SCALE))["ws4_indices"]
    assert gt.shape[0] == len(q) == 3610
    t0 = time.time()
    print(f"fetching {gt.size:,} passage rows ({len(np.unique(gt)):,} unique)",
          flush=True)
    table = load_passages(gt.reshape(-1), WS4_SCALE)
    print(f"passages cached in {time.time() - t0:.0f}s", flush=True)
    texts = table["text"]
    rows = []
    for i in range(len(q)):
        golds = [str(a) for a in q["answer"].iloc[i]]
        t = [texts.at[int(r)] for r in gt[i]]
        rows.append(dict(
            row=i, question=q["question"].iloc[i], n_golds=len(golds),
            answer_presence_at_10=core.answer_present(golds, t[:10]),
            answer_presence_at_100=core.answer_present(golds, t)))
    flags = pd.DataFrame(rows)
    flags.to_parquet(OUT, index=False)
    print(f"ap@10={flags.answer_presence_at_10.mean():.4f} "
          f"ap@100={flags.answer_presence_at_100.mean():.4f} "
          f"n={len(flags)} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
