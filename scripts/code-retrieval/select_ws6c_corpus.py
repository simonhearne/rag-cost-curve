#!/usr/bin/env python
"""Phase 2a: clone and measure every WS6c candidate corpus.

Measurement is in count_tokens against the AGENT model -- tiktoken undercounts
Claude by 26-39% on code and is recorded only for comparability. Scope matching
is by measured tokens, never file count (WS6a section 7's own rule).

Run:  python scripts/code-retrieval/select_ws6c_corpus.py
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest, agent_harness
from benchlib.code_retrieval import bench as ws6a
from benchlib.config import (
    WS6A_AGENT_MODEL, WS6A_PREFIX_BUDGET_TOKENS, WS6C_CANDIDATES,
    ws6c_checkout, ws6c_dir,
)

# results/ws6a/repo_stats.csv, frozen.
FASTAPI_SCOPES = {"S": 259_429, "M": 1_138_485, "L": 6_232_509}


def clone(url: str, dest: Path) -> str:
    if not dest.exists():
        subprocess.run(["git", "clone", "--quiet", url, str(dest)], check=True)
    sha = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return sha


def main() -> None:
    client = agent_harness.make_client()
    enc = tiktoken.get_encoding("cl100k_base")
    overhead = agent_harness.measure_request_overhead(client, WS6A_AGENT_MODEL)
    print(f"request overhead: {overhead} tokens/request (subtracted)")

    rows = []
    for cand in WS6C_CANDIDATES:
        root = ws6c_checkout(cand["slug"])
        sha = clone(cand["url"], root)
        paths = ws6a.filter_text_files(ws6a.git_ls_files(root))
        texts = [(root / p).read_text(errors="ignore") for p in paths]
        chars = sum(len(t) for t in texts)
        lines = sum(t.count("\n") for t in texts)
        tok = agent_harness.count_tokens_batched(client, WS6A_AGENT_MODEL, texts,
                                                overhead=overhead)
        tk = sum(len(enc.encode(t)) for t in texts)

        nearest = min(FASTAPI_SCOPES, key=lambda s: abs(FASTAPI_SCOPES[s] - tok))
        rows.append({
            "slug": cand["slug"], "repo": cand["repo"], "commit": sha,
            "license": cand["license"], "files": len(paths), "lines": lines,
            "chars": chars, "count_tokens": tok, "tiktoken_cl100k": tk,
            "nearest_fastapi_scope": nearest,
            "scope_ratio": round(tok / FASTAPI_SCOPES[nearest], 3),
            "fits_in_prefix_budget": bool(tok <= WS6A_PREFIX_BUDGET_TOKENS),
        })
        print(f"{cand['slug']:16s} {len(paths):5d} files  {tok:9,d} tokens  "
              f"nearest fastapi {nearest}  "
              f"fits-in-1M-prefix={tok <= WS6A_PREFIX_BUDGET_TOKENS}")

    out = ws6c_dir() / "corpus_stats.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    manifest.record(out)
    print(f"\nwrote {out}")
    print("If a candidate's fits_in_prefix_budget is True, Phase 3b (TRUE "
          "whole-repository stuffing) becomes available for the first time in "
          "this project -- it retires WS6a limitations 3 and 4.")


if __name__ == "__main__":
    main()
