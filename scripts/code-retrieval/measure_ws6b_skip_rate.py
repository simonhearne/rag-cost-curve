#!/usr/bin/env python3
"""WS6b section 6: memsearch's incremental-sync skip rate under three kinds
of history growth. WS5's churn arm is blocked on this number.

Each variant is applied to the L corpus, re-indexed against the existing
ws6b_l collection, measured, then ROLLED BACK before the next.

  new_file     add one new day file containing one session
  tail_append  append one session to the END of the last existing day file
  mid_edit     change one bullet in the MIDDLE of a mid-history day file

mid_edit is the adversarial case and is why there are three. memsearch's
chunk identity is f(source, start_line, end_line, sha256(content), model), so
start_line and end_line are INSIDE the identity: an edit that changes a
section's line count shifts every downstream chunk's id in that file and
forces a re-embed regardless of whether that content changed. Whether it
actually does is a measurement, not a prediction.

Design: results/code-retrieval.md
"""

import csv
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import config, manifest  # noqa: E402
from benchlib.code_retrieval import memory_runner as ws6b_runner  # noqa: E402
from benchlib.code_retrieval import bench as ws6a  # noqa: E402

OUT = config.ws6b_dir()
CORPUS = config.WS6B_CORPUS_DIR

_NEW_SESSION = """
## Session 09:30

### 09:30
<!-- session:00000000-0000-4000-8000-000000000001 turn:00000000-0000-4000-8000-000000000002 transcript:/synthetic/ws6b/churn.jsonl -->
- Added a short follow-up session so the incremental-sync behaviour can be
  measured against a corpus that has genuinely grown by one unit of work.
- Nothing in this session is referenced by any probe; it exists only to make
  the index do work it did not previously have to do.
"""


def reset_corpus():
    """git checkout the corpus, so no variant leaks into the next."""
    subprocess.run(["git", "checkout", "--", str(CORPUS)],
                   cwd=config.REPO_ROOT, check=True)
    for extra in CORPUS.glob("2099-*.md"):
        extra.unlink()


def apply_variant(variant: str, day_files):
    if variant == "new_file":
        (CORPUS / "2099-01-01.md").write_text(_NEW_SESSION)
        return day_files + [CORPUS / "2099-01-01.md"]
    if variant == "tail_append":
        last = day_files[-1]
        last.write_text(last.read_text().rstrip() + "\n" + _NEW_SESSION)
        return day_files
    if variant == "mid_edit":
        mid = day_files[len(day_files) // 2]
        lines = mid.read_text().splitlines()
        idx = next(i for i, l in enumerate(lines[len(lines) // 2:],
                                           start=len(lines) // 2)
                   if l.startswith("- "))
        # Replace one bullet with a LONGER one, so the section's line count
        # changes and downstream start_line/end_line shift. A same-length
        # edit would understate the effect this variant exists to expose.
        lines[idx] = ("- Revised this note during the churn measurement so the "
                      "section's line count changes; the point is to move every "
                      "following chunk's start_line and end_line, which are part "
                      "of memsearch's chunk identity.\n- Second line added for "
                      "the same reason.")
        mid.write_text("\n".join(lines) + "\n")
        return day_files
    raise ValueError(variant)


def main() -> int:
    from pymilvus import MilvusClient

    if not ws6b_runner.memsearch_available():
        print("memsearch is not runnable; aborting")
        return 1
    collection = config.WS6B_COLLECTIONS["l"]
    mc = MilvusClient(uri=config.WS6B_MILVUS_URI)
    prices = ws6a.load_pricing(OUT / "pricing.csv")
    embed_rate = prices[config.WS6B_EMBED_MODEL]["input_usd_per_mtok"]

    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")

    reset_corpus()
    rows = []
    for variant in ("new_file", "tail_append", "mid_edit"):
        base_files = sorted(CORPUS.glob("*.md"))
        mc.flush(collection)
        time.sleep(1)
        before = mc.query(collection_name=collection, filter="",
                          output_fields=["count(*)"])[0]["count(*)"]

        files = apply_variant(variant, base_files)
        t0 = time.time()
        proc = ws6b_runner.memsearch_cli(
            ["index", *[str(p.resolve()) for p in files],
             *ws6b_runner._ms_common(collection)], timeout=7200)
        wall = round(time.time() - t0, 1)
        if proc.returncode != 0:
            print(f"{variant}: FAILED\n{proc.stderr[-600:]}")
            reset_corpus()
            return 1
        m = re.search(r"Indexed (\d+) chunks", proc.stdout)
        reembedded = int(m.group(1)) if m else -1

        mc.flush(collection)
        time.sleep(2)
        after = mc.query(collection_name=collection, filter="",
                         output_fields=["count(*)"])[0]["count(*)"]

        # Denominator is the number of chunks the scan CONSIDERED, i.e. the
        # collection as it stands after the change. skip_rate is the fraction
        # of those that did not need re-embedding.
        skip_rate = round(1 - (reembedded / after), 6) if after else 0.0
        etok = 0
        if reembedded > 0:
            # Derived: mean embedded tokens per chunk over the corpus, times
            # the chunks actually re-embedded. Labelled derived like every
            # other embedding cost in this workstream.
            text = re.sub(r"<!--.*?-->", "",
                          "".join(p.read_text() for p in files), flags=re.S)
            etok = int(len(enc.encode(text)) / after * reembedded)

        rows.append({
            "variant": variant, "files_scanned": len(files),
            "chunks_before": before, "chunks_total": after,
            "chunks_reembedded": reembedded,
            "chunks_skipped": after - reembedded, "skip_rate": skip_rate,
            "wall_clock_s": wall, "embed_tokens": etok,
            "embed_cost_usd": round(etok / 1_000_000 * embed_rate, 6),
            "derived": True,
        })
        print(f"{variant:12s} total={after:5d} reembedded={reembedded:5d} "
              f"skip_rate={skip_rate:.4f} wall={wall}s ${rows[-1]['embed_cost_usd']:.5f}")

        reset_corpus()
        # Restore the collection to the pristine L index before the next
        # variant, so variants cannot contaminate one another.
        ws6b_runner.memsearch_cli(
            ["index", *[str(p.resolve()) for p in sorted(CORPUS.glob("*.md"))],
             *ws6b_runner._ms_common(collection)], timeout=7200)

    with open(OUT / "skip_rate.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    manifest.record(OUT / "skip_rate.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
