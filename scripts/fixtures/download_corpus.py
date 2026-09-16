#!/usr/bin/env python3
"""Download the pinned corpus shards needed for a given slice size.

Shards download in order (00000, 00001, ...) until cumulative rows cover the
requested slice, so the 1M slice is a strict prefix of the 10M slice. Every
shard is sha256-recorded in data/MANIFEST.json.

Usage:
    python scripts/fixtures/download_corpus.py --rows 1000000
    python scripts/fixtures/download_corpus.py --rows 10000000

Expected: ~57.5k rows / ~130 MB per shard.
  1M  slice  -> ~18 shards,  ~2.3 GB download
  10M slice  -> ~175 shards, ~23 GB download
Resumable: shards already on disk are skipped.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib.download import ensure_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    args = ap.parse_args()
    shards = ensure_rows(args.rows)
    print(f"{len(shards)} shards on disk, {shards[-1]['cum_rows']:,} rows "
          f"(target {args.rows:,})")


if __name__ == "__main__":
    main()
