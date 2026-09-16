#!/usr/bin/env python
"""WS8 Phase 0b: bot filter, document tree, corpus summary. NO model calls
except count_tokens -- a token-counting endpoint, not generation, and
unbilled.

count_tokens is MEASURED, never estimated: WS6a recorded a chars/4 estimate
understating the real count by 1.7x.

Per-document counting REMOVED (Opus review, 2026-09-11). R3 originally fixed
a defect where the brief called `count_tokens_batched(client, model, texts)`
and iterated the result as if it were per-document counts -- it returns a
single int, the token count of the whole batch. That fix (count each
document separately, cache the results) existed only to place the S/M/L
prefix boundaries. The size sweep is gone (see below), so per-document
counts have no remaining consumer: at 7,647 documents that was 7,647
`count_tokens` calls (~an hour) for a number nothing reads. This script now
calls `count_tokens_batched` the way it was actually designed to be called
-- ONCE (internally chunked at its own max_chars boundary) over the whole
corpus -- which is a handful of calls, seconds instead of an hour. The
former per-document cache (data/ws8_cache/per_doc_tokens.json) is simply no
longer written; nothing reads it either.

A per-repo size breakdown is still useful for the report, so `chars` and a
derived `estimated_tokens` (proportional to each repo's share of corpus
characters) are recorded per repo -- explicitly labeled an ESTIMATE, not a
measurement. The authoritative figure is `measured_tokens` on the `all` row,
the one and only count_tokens call result.

Size sweep REMOVED (Opus review, spec amended 547a53a): S/M/L prefixes were
filled repo-major, so S (47 docs) and M (298) were 100% fastapi -- the one
repo contributing zero questions. Every arm would have tied at zero recall.
WS8 runs at the full corpus and claims nothing about corpus size; WS6b
already covers that dimension. No prefixes.json, no per-size rows.

Dedupe (found live, 2026-09-11): kubernetes/kubernetes#140406 was fetched
twice by GitHub's pagination. fetch_ws8_corpus.py dedupes at fetch time;
this script ALSO dedupes defensively across every raw file it loads, so a
stale raw JSON written before that fix cannot silently overwrite one
document's rendered file with itself and double-count its tokens.

Staleness guard (Opus review, 2026-09-11): a raw file left over from an
interrupted or superseded fetch (old cap, old per-thread comment strategy --
this is exactly what happened live when rust-lang/rust's fetch crashed mid-
run and its stale 2026-09-10 file sat on disk) must never silently join a
corpus this script treats as current. ws8.assert_fresh() checks every raw
file's embedded `_fetch_basis` tag before any other processing and fails
loudly if it is not one of ws8.ACCEPTED_FETCH_BASES.

Superseded documents (Opus review, 2026-09-11): docs/ used to be written
into in place, never cleared. Every document this run produces is overwritten,
but one it NO LONGER produces -- because the window moved, a thread was
deleted upstream, or a repo left WS8_REPOS -- simply stayed on disk. Phase 2
indexes that directory and the agentic arm greps it, so a superseded document
would silently remain part of the measured corpus with nothing describing it.
This run now renders into a staging directory and swaps it in, so docs/ always
contains exactly what this run produced and nothing else. Swap, rather than
"clear then write": the live corpus is not committed (it is fetched, not
redistributed), so a crash between the clear and the last write would destroy
an artifact with no copy anywhere. The old tree is only unlinked once the new
one is complete.

Run: python scripts/non-code-retrieval/build_ws8_corpus.py
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest, agent_harness
from benchlib.non_code_retrieval import bench as ws8
from benchlib.config import DATA_DIR, WS8_AGENT_MODEL, WS8_CORPUS_DIR, ws8_dir
from benchlib.manifest import load_manifest


def swap_docs(staging: Path, docs_dir: Path) -> set[str]:
    """Make `staging` the new docs/ and return the filenames it dropped.

    The old tree is RENAMED aside first and unlinked only after the new one is
    in place, so there is no instant at which neither exists. A `.superseded`
    directory left behind by an interrupted swap is the previous corpus and is
    cleared on the next run rather than merged into anything.
    """
    before = ({p.name for p in docs_dir.iterdir() if p.is_file()}
              if docs_dir.exists() else set())
    after = {p.name for p in staging.iterdir() if p.is_file()}

    previous = docs_dir.with_name(docs_dir.name + ".superseded")
    if previous.exists():
        shutil.rmtree(previous)
    if docs_dir.exists():
        docs_dir.rename(previous)
    staging.rename(docs_dir)
    shutil.rmtree(previous, ignore_errors=True)
    return before - after


def main() -> None:
    raw = sorted((WS8_CORPUS_DIR / "raw").glob("*.json"))
    if not raw:
        sys.exit("no raw corpus -- run scripts/non-code-retrieval/fetch_ws8_corpus.py first")
    docs_dir = WS8_CORPUS_DIR / "docs"
    # Rendered into here, then swapped over docs/ once every repo is done.
    staging = WS8_CORPUS_DIR / "docs.building"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    prior_manifest = load_manifest().get("files", {})

    per_repo_rows = []
    all_texts: list[str] = []
    all_threads: list[dict] = []

    for f in raw:
        raw_threads = json.loads(f.read_text())

        # Staleness guard: fails loudly, before any other processing, if
        # this raw file wasn't written under the current fetch basis.
        if raw_threads:
            ws8.assert_fresh(raw_threads, source=str(f))

        # Defensive dedupe (R16 follow-up): fetch_ws8_corpus.py already
        # dedupes at fetch time, but this guards against a stale raw file
        # written before that fix.
        raw_threads, duplicate_count = ws8.dedupe_threads(raw_threads)

        manifest_key = str(f.resolve().relative_to(DATA_DIR.resolve()))
        fetch_entry = prior_manifest.get(manifest_key, {})
        # Folds in the fetch-time count so an already-fixed raw file's
        # historical drop stays visible even when this run's own defensive
        # pass finds nothing left on disk to drop.
        duplicate_count += fetch_entry.get("duplicate_threads_dropped", 0)
        comment_mismatches = fetch_entry.get("comment_count_mismatches", 0)

        threads, dropped_total = [], 0
        pr_count = issue_count = 0
        for t in raw_threads:
            if ws8.is_pull_request(t):
                pr_count += 1
            else:
                issue_count += 1
            clean, dropped = ws8.strip_bots(t)
            dropped_total += dropped
            threads.append(clean)

        if not threads:
            print(f"warning: {f} contained no threads; skipping")
            continue

        repo = threads[0]["repo"]
        created_ats = [t["created_at"] for t in raw_threads]

        texts, chars = [], 0
        for t in sorted(threads, key=lambda t: (t["repo"], t["number"])):
            text = ws8.render_document(t)
            name = ws8.document_filename(t)
            (staging / name).write_text(text)
            texts.append(text)
            chars += len(text)

        per_repo_rows.append({
            "repo": repo,
            "documents": len(texts),
            "chars": chars,
            "prs_fetched": pr_count,
            "issues_fetched": issue_count,
            "bot_comments_dropped": dropped_total,
            "duplicate_threads_dropped": duplicate_count,
            "comment_count_mismatches": comment_mismatches,
            "created_at_min": min(created_ats) if created_ats else "",
            "created_at_max": max(created_ats) if created_ats else "",
        })
        all_texts.extend(texts)
        all_threads.extend(threads)
        print(f"{repo}: {len(texts)} documents rendered, {chars:,} chars")

    if not all_threads:
        # staging is left in place, unswapped: docs/ is untouched and still
        # holds the last complete corpus, which is the right state after a run
        # that produced nothing.
        sys.exit("raw corpus is empty -- nothing to build")

    superseded = swap_docs(staging, docs_dir)
    if superseded:
        print(f"removed {len(superseded)} superseded document(s) no longer "
              f"produced by this build: {sorted(superseded)[:5]}"
              f"{' ...' if len(superseded) > 5 else ''}")

    # THE measurement: one batched count_tokens call (internally chunked)
    # over the whole corpus. This is the number every downstream cost claim
    # must trace back to -- never a chars/4 estimate.
    client = agent_harness.make_client()
    print(f"counting tokens for {len(all_texts)} documents across the whole "
          f"corpus, in batches...")
    measured_tokens = agent_harness.count_tokens_batched(client, WS8_AGENT_MODEL, all_texts)
    print(f"measured corpus total: {measured_tokens:,} tokens")

    # Per-repo split is NOT separately measured -- it is proportional to each
    # repo's share of corpus CHARACTERS, explicitly labeled an estimate.
    total_chars = sum(r["chars"] for r in per_repo_rows) or 1
    for r in per_repo_rows:
        r["estimated_tokens"] = round(measured_tokens * r["chars"] / total_chars)
        r["measured_tokens"] = ""

    total_row = {
        "repo": "all",
        "documents": sum(r["documents"] for r in per_repo_rows),
        "chars": total_chars,
        "estimated_tokens": "",
        "measured_tokens": measured_tokens,
        "prs_fetched": sum(r["prs_fetched"] for r in per_repo_rows),
        "issues_fetched": sum(r["issues_fetched"] for r in per_repo_rows),
        "bot_comments_dropped": sum(r["bot_comments_dropped"] for r in per_repo_rows),
        "duplicate_threads_dropped": sum(r["duplicate_threads_dropped"] for r in per_repo_rows),
        "comment_count_mismatches": sum(r["comment_count_mismatches"] for r in per_repo_rows),
        "created_at_min": min((r["created_at_min"] for r in per_repo_rows if r["created_at_min"]),
                              default=""),
        "created_at_max": max((r["created_at_max"] for r in per_repo_rows if r["created_at_max"]),
                              default=""),
    }

    cols = ["repo", "documents", "chars", "estimated_tokens", "measured_tokens",
            "prs_fetched", "issues_fetched", "bot_comments_dropped",
            "duplicate_threads_dropped", "comment_count_mismatches",
            "created_at_min", "created_at_max"]
    df = pd.DataFrame(per_repo_rows + [total_row])[cols]
    out = ws8_dir() / "corpus_stats.csv"
    df.to_csv(out, index=False)
    manifest.record(out, {"measured_tokens_total": measured_tokens})

    print(df.to_string(index=False))
    total_mismatches = total_row["comment_count_mismatches"]
    if total_mismatches:
        print(f"WARNING: {total_mismatches} thread(s) have a comment-count "
              f"mismatch; see data/MANIFEST.json for detail", file=sys.stderr)


if __name__ == "__main__":
    main()
