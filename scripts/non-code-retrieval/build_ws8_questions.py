#!/usr/bin/env python
"""WS8 Phase 0c: reference graph -> chains -> questions.csv. NO model calls.

Committed BEFORE any retrieval spend, with the generator's git SHA recorded,
so mechanical derivation is provable from git history rather than asserted.

R16: build_question is built from the head's TITLE, not its body -- 81% of
fetched threads are pull requests, and a body-based template pulled in Prow
commands and template boilerplate ("/kind flake"), while issue bodies were
themselves unreliable (automated non-English scan reports, raw <img> tags).
Titles are human-written problem summaries and need far fewer judgement
calls. R16 also requires a chain's head to be an issue, never a pull
request: PRs stay in the corpus as documents and as chain MEMBERS -- "issue
-> the PR that fixed it -> follow-up issue" is the chain WS8 measures -- but
the thread that POSES the question must be the report, not the fix.

R9-carried: build_question redacts inline #N references from the reporter's
first sentence, so a chain head whose (now title-based) first sentence is
ONLY a reference (e.g. "see #34") becomes a contentless question after
redaction.

One question per head (Opus review, spec amended 547a53a): extract_chains
enumerates every maximal simple path, so one head in a dense reference
cluster yielded many chains -- the first run's hop>=3 stratum was 30 rows
over only 16 distinct heads, 6 of them with more than one gold set for the
identical question text. ws8.one_question_per_head() reduces to one chain
per (repo, head) -- the longest -- immediately after chains are extracted,
before any other filtering, so a question and its evidence are always 1:1.

select_questions then excludes both remaining kinds of disqualified chain --
PR-headed (R16) and degenerate (R9-carried) -- via closures built here
(since select_questions itself never sees head data) BEFORE filling quotas,
so neither can reach questions.csv -- not even as an excluded=True row.

Pairwise-disjoint evidence per stratum (spec amended c0f4b59): the earlier
"prefer the longest chain within hop3plus" tie-break concentrated that
stratum into one dense, correlated cluster -- measured on the real corpus,
29 of 30 hop3plus questions came from a single repo, 25 of 30 shared a gold
document with another question, and three documents each appeared in 11-12
of the 30 gold sets. The paired bootstrap (spec section 7) treats questions
as exchangeable draws; overlapping evidence means the arms repeat the same
retrieval work across correlated questions, so the nominal n=30 was really a
much smaller effective n -- an underpowered interval that LOOKS precise,
exactly the WS6c failure this workstream exists to avoid repeating. The
tie-break is REMOVED; select_questions now enforces pairwise-disjoint gold
document sets WITHIN each stratum via a `gold_documents` closure, and every
stratum orders candidates identically, by the seeded shuffle alone.

This is also the Phase 0 gate: if a stratum can't reach its quota after
reduction, exclusion, and the disjointness constraint, select_questions
raises and this script halts WS8. Widening WS8_WINDOW_START/END to recover
chains is forbidden (spec section 7) -- it would reach back toward the
model's training cutoff.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.non_code_retrieval import bench as ws8
from benchlib.config import (REPO_ROOT, SEED, WS8_CORPUS_DIR, WS8_REPOS,
                             WS8_STRATA, ws8_dir)


# Both paths the generator's logic actually lives in: the orchestration here,
# and select_questions/build_question/extract_chains/one_question_per_head in
# benchlib/non_code_retrieval/bench.py. A SHA covering only this file would
# omit most of what determines questions.csv's content.
GENERATOR_PATHS = ("scripts/non-code-retrieval/build_ws8_questions.py",
                   "benchlib/non_code_retrieval/bench.py")


def generator_provenance() -> dict:
    """{generator_sha, generator_dirty} for GENERATOR_PATHS.

    House convention (scripts/code-retrieval/run_ws6b_parametric.py:generator_sha):
    `git rev-parse HEAD` would misattribute provenance here -- this script,
    its tests, and the CSV it produces are committed together in one commit,
    so HEAD at run time is still the PARENT commit. `git log -1 -- <paths>`
    (the most recent commit touching EITHER path) instead returns
    "uncommitted" until that commit lands -- the honest value rather than a
    plausible-looking wrong one. `generator_dirty` additionally flags an
    uncommitted working-tree change to either path, so a manifest entry never
    claims a SHA that does not actually match what ran.
    """
    sha_p = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *GENERATOR_PATHS],
        capture_output=True, text=True, cwd=REPO_ROOT)
    sha = sha_p.stdout.strip() or "uncommitted"

    dirty_p = subprocess.run(
        ["git", "status", "--porcelain", "--", *GENERATOR_PATHS],
        capture_output=True, text=True, cwd=REPO_ROOT)
    dirty = bool(dirty_p.stdout.strip())

    return {"generator_sha": sha, "generator_dirty": dirty}


def main() -> None:
    raw = sorted((WS8_CORPUS_DIR / "raw").glob("*.json"))
    if not raw:
        sys.exit("no raw corpus -- run scripts/non-code-retrieval/fetch_ws8_corpus.py first")

    raw_threads = []
    for f in raw:
        file_threads = json.loads(f.read_text())
        # Staleness guard, same reasoning as build_ws8_corpus.py: a raw file
        # from an interrupted or superseded fetch must never silently
        # contribute chains to questions.csv.
        if file_threads:
            ws8.assert_fresh(file_threads, source=str(f))
        raw_threads.extend(file_threads)

    # Defensive dedupe, same reasoning as build_ws8_corpus.py: this script
    # also reads raw JSON directly, so a stale duplicate (GitHub pagination
    # repeating an item, e.g. kubernetes/kubernetes#140406) would otherwise
    # walk() twice from the same node in extract_chains and inflate the
    # candidate chain pool with a redundant duplicate.
    raw_threads, _ = ws8.dedupe_threads(raw_threads)

    by_repo: dict[str, list] = {}
    for t in raw_threads:
        by_repo.setdefault(t["repo"], []).append(ws8.strip_bots(t)[0])

    chains, index = [], {}
    for repo, threads in sorted(by_repo.items()):
        g = ws8.build_graph(threads)
        for c in ws8.extract_chains(threads, g):
            chains.append((repo, c))
        for t in threads:
            index[(repo, t["number"])] = t

    if not chains:
        sys.exit("no chains extracted -- nothing to select questions from")

    n_chains_before_reduction = len(chains)
    chains = ws8.one_question_per_head(chains)
    print(f"one question per head: {n_chains_before_reduction} chains -> "
          f"{len(chains)} distinct heads")

    # Repo travels WITH every chain from here on (chains is a list of
    # (repo, chain) pairs) -- select_questions is repo-scoped natively, so
    # no id()-keyed side table is needed to recover it.
    def is_pr_headed(repo: str, chain: list[int]) -> bool:
        # R16: a chain's head must be an issue, never a pull request. PRs
        # stay in the corpus as documents and as chain MEMBERS (the tail or
        # a middle node); only the head is restricted.
        head = index[(repo, chain[0])]
        return ws8.is_pull_request(head)

    def is_degenerate(repo: str, chain: list[int]) -> bool:
        head = index[(repo, chain[0])]
        return ws8.is_degenerate_head(head)

    def gold_documents(repo: str, chain: list[int]) -> set:
        # Spec amended c0f4b59: pairwise-disjoint evidence WITHIN a stratum
        # replaces the old longest-chain tie-break, which concentrated
        # hop3plus into one dense, correlated cluster (measured: 29/30 from
        # one repo, 25/30 sharing a gold document with another question).
        return {ws8.document_filename(index[(repo, n)]) for n in chain}

    picked = ws8.select_questions(
        chains, WS8_STRATA, seed=SEED, is_pr_headed=is_pr_headed,
        is_degenerate=is_degenerate, gold_documents=gold_documents)

    docs = [ws8.render_document(t) for t in index.values()]
    df_counts, n_docs = ws8.document_frequencies(docs)

    rows = []
    for i, (repo, chain) in enumerate(picked, start=1):
        head = index[(repo, chain[0])]
        evidence = "\n".join(
            ws8.render_document(index[(repo, n)]) for n in chain)
        question = ws8.build_question(head, chain)
        rows.append({
            "qid": f"ws8_q{i:03d}",
            "question": question,
            "expected_documents": ";".join(
                ws8.document_filename(index[(repo, n)]) for n in chain),
            "hops": len(chain),
            "stratum": ws8.classify_hops(chain),
            "repo": repo,
            "head_number": chain[0],
            "tail_number": chain[-1],
            "grep_handle": round(
                ws8.grep_handle(question, evidence, df_counts, n_docs), 4),
            "excluded": False,
            "exclusion_reason": "",
        })

    out = ws8_dir() / "questions.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    provenance = generator_provenance()
    manifest.record(out, {**provenance, "seed": SEED})

    counts = {k: sum(1 for r in rows if r["stratum"] == k) for k in WS8_STRATA}
    print(f"{len(rows)} questions -> {out} "
          f"(generator {provenance['generator_sha'][:8]}, "
          f"dirty={provenance['generator_dirty']})")
    print(f"per-stratum counts: {counts}")

    # Per-repo question breakdown, ALL pinned repos listed (even ones that
    # contributed zero, e.g. fastapi -- too few issue-headed non-degenerate
    # chains to clear its share of quota). The strata are pre-registered and
    # seeded; a repo contributing 0 questions is a fact about the corpus to
    # report, not something to rebalance toward post hoc.
    breakdown_rows = []
    for repo in WS8_REPOS:
        repo_rows = [r for r in rows if r["repo"] == repo]
        breakdown_rows.append({
            "repo": repo,
            **{k: sum(1 for r in repo_rows if r["stratum"] == k) for k in WS8_STRATA},
            "total": len(repo_rows),
        })
    breakdown_out = ws8_dir() / "questions_by_repo.csv"
    pd.DataFrame(breakdown_rows).to_csv(breakdown_out, index=False)
    manifest.record(breakdown_out, {**provenance, "seed": SEED})
    print(pd.DataFrame(breakdown_rows).to_string(index=False))


if __name__ == "__main__":
    main()
