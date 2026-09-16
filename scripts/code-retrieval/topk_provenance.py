#!/usr/bin/env python
"""Record the pinned MCP package's OWN default search_code limit.

"Default" must be a measured value in the results, not an assumption. Same
method that established mcp_env's settings: read the package's dist/.

Run:  python scripts/code-retrieval/topk_provenance.py
"""

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.config import DATA_DIR, WS6A_MCP_PACKAGE, ws6c_dir


def main() -> None:
    pkg = WS6A_MCP_PACKAGE
    root = Path(subprocess.run(
        ["npm", "root", "-g"], capture_output=True, text=True, check=True
    ).stdout.strip())
    # npx caches under ~/.npm/_npx; fall back to an explicit install so the
    # version we read is the version we ran.
    dest = DATA_DIR / "ws6c_cache" / "mcp_pkg"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["npm", "install", "--prefix", str(dest), pkg], check=True)

    hits = []
    for js in (dest / "node_modules").rglob("*.js"):
        if not js.is_file():
            continue  # e.g. node_modules/bignumber.js/ is a package DIRECTORY
        text = js.read_text(errors="ignore")
        for m in re.finditer(r"limit\s*[:=]\s*(\d+)", text):
            ctx = text[max(0, m.start() - 200):m.end() + 200]
            if "search" in ctx.lower():
                hits.append(f"{js.relative_to(dest)}:{m.group(0)}")

    out = ws6c_dir() / "topk_provenance.txt"
    out.write_text(
        f"# search_code default limit provenance\n"
        f"# package: {pkg}\n"
        f"# npm root -g: {root}\n"
        f"# extracted {len(hits)} candidate site(s) from the package's own dist/\n\n"
        + "\n".join(hits) + "\n")
    manifest.record(out)
    print(f"wrote {out} with {len(hits)} candidate site(s)")
    print("READ THIS FILE and record the single authoritative default in "
          "results/code-retrieval.md. If it is ambiguous, say so rather than picking one.")


if __name__ == "__main__":
    main()
