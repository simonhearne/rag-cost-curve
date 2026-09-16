#!/usr/bin/env python3
"""Download NQ-Open (pinned revision) train + validation parquet files.

Usage:  python scripts/fixtures/download_nq.py
~15 MB total. Files land in data/nq/ and are recorded in MANIFEST.json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib.download import fetch_nq

if __name__ == "__main__":
    for split, p in fetch_nq().items():
        print("ok", split, p)
