#!/usr/bin/env python
"""Fill in results/ws8/spend_ledger.csv's `committed_in` column (fix round 4).

A ledger row's `ran_at_head`/`dirty` are stamped at RUN time by
`benchlib.non_code_retrieval.bench.append_spend_ledger` and are
never wrong, but a run cannot know the commit that will eventually contain
its own outputs -- git commit hashes are content-addressed over content that
includes this very file. `committed_in` is therefore written empty at append
time and must be filled in AFTER the commit that lands the row's code and
outputs actually exists.

This is that fill-in step, made mechanical so Phase 2 and Phase 3 inherit a
repeatable process rather than a manual edit someone forgets. Run it once,
right after making the commit that lands one or more blank-`committed_in`
rows:

    git commit -m "..."
    python scripts/fixtures/stamp_ledger_commit.py          # stamps HEAD
    git add results/ws8/spend_ledger.csv data/MANIFEST.json
    git commit -m "chore(ws8): stamp spend_ledger.csv committed_in for <sha>"

It only ever fills BLANK committed_in cells (never overwrites an existing
value), and only with the CURRENT HEAD, so it is safe to run repeatedly and
idempotent once every blank row is stamped. It refuses to run against a
dirty working tree -- committed_in must name a real, existing commit, and a
dirty tree means HEAD is not yet the commit that will contain whatever is
about to be staged.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.config import ws8_dir

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# NOT ws8.git_short_sha / ws8.git_is_dirty, deliberately: those swallow a git
# failure and return "unknown"/True, which is the right bias when the job is to
# record a run that already spent money. Here the job is to write a sha into a
# committed file, so a git that cannot answer must stop the script, not supply
# a plausible-looking value -- hence check=True and no except.
def _git_short_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
        text=True, cwd=_REPO_ROOT, timeout=5, check=True,
    ).stdout.strip()


def _git_is_dirty() -> bool:
    proc = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True,
        cwd=_REPO_ROOT, timeout=5, check=True,
    )
    return bool(proc.stdout.strip())


def main() -> None:
    if _git_is_dirty():
        sys.exit(
            "refusing to stamp committed_in against a DIRTY working tree -- "
            "commit first, then re-run this script against the resulting "
            "HEAD")

    sha = _git_short_sha()
    path = ws8_dir() / "spend_ledger.csv"
    df = pd.read_csv(path, dtype={"committed_in": str}, keep_default_na=False)

    blank = df["committed_in"] == ""
    n_blank = int(blank.sum())
    if n_blank == 0:
        print(f"no blank committed_in rows in {path}; nothing to stamp")
        return

    df.loc[blank, "committed_in"] = sha
    df.to_csv(path, index=False)
    manifest.record(path)
    print(f"stamped {n_blank} row(s) with committed_in={sha}")
    print(df[df["committed_in"] == sha][
        ["run_label", "ran_at_head", "dirty", "committed_in"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
