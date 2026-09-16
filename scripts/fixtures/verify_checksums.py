#!/usr/bin/env python3
"""Re-hash every file recorded in data/MANIFEST.json and report the result.

Three outcomes are distinguished, because they mean very different things:

  verified  the file is present and its sha256 matches the manifest.
  absent    the file is not on disk at all. On a fresh clone this is the
            NORMAL state for most entries: data/ is gitignored and its
            fixtures are re-derived by the download scripts and notebook 01.
            An absent file is "not generated yet", not "wrong".
  corrupt   the file is present but its sha256 does NOT match. This is the
            only failure: it means what is on disk is not what produced the
            committed results.

Usage:
    python scripts/fixtures/verify_checksums.py             # absent is OK
    python scripts/fixtures/verify_checksums.py --strict    # absent fails too

Exit code 0 = nothing corrupt (and, under --strict, nothing absent either);
1 = at least one corrupt file, listed on stdout.

Use --strict for the full-repo gate, where every artifact is expected to have
been generated and any absence is itself a defect.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest

parser = argparse.ArgumentParser(
    description="Verify data/MANIFEST.json checksums.",
    epilog="Absent files are reported but do not fail the run unless --strict.",
)
parser.add_argument(
    "--strict",
    action="store_true",
    help="also fail if any manifest entry is absent from disk "
    "(the full-repo gate; the default is the fresh-clone posture)",
)
args = parser.parse_args()

bad = manifest.verify()

absent = sorted(line.removeprefix("MISSING ") for line in bad
                if line.startswith("MISSING "))
corrupt = sorted(line.removeprefix("HASH MISMATCH ") for line in bad
                 if line.startswith("HASH MISMATCH "))
total = len(manifest.load_manifest()["files"])
verified = total - len(absent) - len(corrupt)

for rel in corrupt:
    print(f"CORRUPT {rel}")
if absent and args.strict:
    for rel in absent:
        print(f"ABSENT  {rel}")

print(f"{verified} verified, {len(absent)} absent, {len(corrupt)} corrupt "
      f"({total} manifest entries)")

if corrupt:
    print("\nFAIL: files on disk do not match the manifest. These are not "
          "reproducible artifacts of the committed results.")
    sys.exit(1)

if absent and args.strict:
    print(f"\nFAIL (--strict): {len(absent)} manifest entries are absent.")
    sys.exit(1)

if absent:
    print(f"\nOK: nothing corrupt. {len(absent)} "
          f"{'entry is' if len(absent) == 1 else 'entries are'} not present on "
          "disk -- expected on a fresh clone, where data/ fixtures are "
          "re-derived by scripts/fixtures/ and notebook 01. Re-run with "
          "--strict once everything has been generated.")
else:
    print("\nOK: all manifest entries present and matching.")
