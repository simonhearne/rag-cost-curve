"""WS8 pure logic: bot filtering, document rendering, the reference graph,
chain extraction, question templating, and checkpoint-spend reading. No network,
no model calls.

The one exception to "pure" is the spend ledger at the bottom of the file: it
writes results/ws8/spend_ledger.csv and shells out to `git` for the run's HEAD
and dirty flag. It lives here, as ws6c's pin guard does, because FOUR scripts
need it and four hand-synced copies of the artifact that records spend is how
a row goes missing.

Design: results/non-code-retrieval.md
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
import subprocess
from pathlib import Path

import pandas as pd

from ..config import (WS8_BOT_DENYLIST, WS8_FETCH_DIRECTION, WS8_FETCH_SORT,
                     WS8_GATE_PASS, WS8_MAX_THREADS_PER_REPO,
                     WS8_WINDOW_END, WS8_WINDOW_START)

# EVERY pin that decides which threads a fetch returns, in one string. A raw
# file stamped with a different one is from a different collection run and must
# be redone, not trusted. Shared by fetch_ws8_corpus.py (which stamps it onto
# every thread it writes and uses it for per-repo resume) and both build
# scripts (which assert every raw file they load carries it, so a stale file
# left over from an interrupted or superseded fetch can never silently join a
# corpus described as current -- the "mixed collection run" failure class the
# duplicate-thread and overwritten-document bugs were also instances of).
#
# v2 (2026-09-11) ADDS the window and the sort order. v1 keyed on the cap and
# the comment strategy ONLY, so moving WS8_WINDOW_END -- the one pin whose
# whole purpose is to define which threads exist -- would have left every raw/
# file passing the freshness guard while describing a different corpus. The
# guard was blind to the very parameter it was built to protect.
FETCH_BASIS = (
    f"cap{WS8_MAX_THREADS_PER_REPO}"
    f"_win{WS8_WINDOW_START}..{WS8_WINDOW_END}"
    f"_{WS8_FETCH_SORT}-{WS8_FETCH_DIRECTION}"
    "_bulk_comments_v2"
)

# The v1 string, and the exact condition under which a file carrying it is
# still a file fetched under today's settings.
#
# This is NOT a grandfather clause. The corpus is gitignored and is NOT
# committed (README, "Licenses"), so invalidating the raw/ files on disk does
# not mean "re-derive from a stored artifact", it means "re-fetch from GitHub
# in 2026-09-11+", and GitHub's issue threads keep growing: a re-fetch today
# returns MORE comments on the same threads than the measured corpus has. A
# basis bump that forces that is not a stricter guard, it is a silent corpus
# swap -- the exact failure this constant exists to prevent, arrived at from
# the other direction.
#
# So v1 is accepted, but ONLY while every pin v1 left implicit still holds the
# value the one v1 fetch actually ran under (recorded independently in
# data/MANIFEST.json's window_pinned / sort / direction fields on each raw
# file). Move any one of them and this goes False in the same breath, every v1
# file correctly reads as stale, and a re-fetch is then the right answer.
_V1_BASIS = "cap8000_bulk_comments_v1"
_V1_PINS_UNCHANGED = (
    WS8_MAX_THREADS_PER_REPO == 8000
    and WS8_WINDOW_START == "2026-06-01"
    and WS8_WINDOW_END == "2026-09-01"
    and WS8_FETCH_SORT == "created"
    and WS8_FETCH_DIRECTION == "asc"
)
ACCEPTED_FETCH_BASES = frozenset(
    {FETCH_BASIS} | ({_V1_BASIS} if _V1_PINS_UNCHANGED else set()))


def assert_fresh(threads: list[dict], source: str) -> None:
    """Raise if any thread was not fetched under an accepted fetch basis.

    Live failure this guards against (2026-09-11): a fetch crashed partway
    through rust-lang/rust, leaving its raw file untouched from a PRIOR run
    (old cap, old per-thread comment strategy) while fastapi and kubernetes's
    files were already fresh -- a mixed corpus that a downstream script would
    otherwise build from without any signal that one repo's data described a
    different collection run than the other two. An empty `threads` list
    raises nothing (an empty file has no basis to be stale about; the
    caller's own "no threads" check handles that case).
    """
    tags = {t.get("_fetch_basis") for t in threads}
    bad = tags - ACCEPTED_FETCH_BASES
    if bad:
        raise ValueError(
            f"{source}: STALE raw file -- found fetch_basis {sorted(str(b) for b in bad)}, "
            f"expected one of {sorted(ACCEPTED_FETCH_BASES)!r}. Re-run "
            "scripts/non-code-retrieval/fetch_ws8_corpus.py before building the corpus; do not "
            "proceed on this file.")


def is_bot(author: dict) -> bool:
    """True for GitHub App accounts and for the pinned denylist.

    Both checks are needed: some bots post under `type == "User"`.
    """
    if str(author.get("type", "")).lower() == "bot":
        return True
    return str(author.get("login", "")).lower() in WS8_BOT_DENYLIST


def dedupe_threads(threads: list[dict]) -> tuple[list[dict], int]:
    """Drop duplicate threads by (repo, number), keeping the first occurrence.

    GitHub's issue-list pagination can repeat an item across a page boundary
    when two threads share `created_at` -- likely, not exotic, under
    sort=created (WS8_FETCH_SORT). Observed live: kubernetes/kubernetes
    #140406 was fetched twice, byte-identical both times, inflating the
    document count, the token total, and the S/M/L prefix boundaries by one
    silently-overwritten file. Returns (deduped, dropped_count).
    """
    seen: set[tuple] = set()
    out: list[dict] = []
    dropped = 0
    for t in threads:
        key = (t["repo"], t["number"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(t)
    return out, dropped


def strip_bots(thread: dict) -> tuple[dict, int]:
    """Return (thread without bot comments, number dropped). Does not mutate."""
    out = copy.deepcopy(thread)
    kept = [c for c in out.get("comments", []) if not is_bot(c.get("user", {}))]
    dropped = len(out.get("comments", [])) - len(kept)
    out["comments"] = kept
    return out, dropped


def render_document(thread: dict) -> str:
    """One thread as one markdown document -- what both arms actually see."""
    parts = [
        f"# {thread['repo']}#{thread['number']}: {thread['title']}",
        f"state: {thread.get('state', 'unknown')}  "
        f"opened: {thread.get('created_at', '')}  "
        f"by: {thread.get('user', {}).get('login', '')}",
        "",
        thread.get("body") or "",
    ]
    for c in thread.get("comments", []):
        parts += [
            "",
            f"## comment by {c.get('user', {}).get('login', '')} "
            f"({c.get('created_at', '')})",
            c.get("body") or "",
        ]
    return "\n".join(parts).rstrip() + "\n"


def document_filename(thread: dict) -> str:
    """Flat, sortable, greppable. `/` in the repo slug becomes `__`."""
    return f"{thread['repo'].replace('/', '__')}__{thread['number']:04d}.md"


def is_pull_request(thread: dict) -> bool:
    """True for a PR. The /issues endpoint returns PRs alongside issues.

    PRs are kept as corpus documents and as chain members -- "issue -> the PR
    that fixed it -> follow-up" is the chain WS8 measures -- but a PR never
    HEADS a chain: the head poses the question and a PR is a resolution, not a
    report (R16).

    Handles both key shapes present in the fetched JSON: our own
    fetch-time-computed boolean `is_pull_request` (on documents already run
    through fetch_ws8_corpus.py), and the raw GitHub payload's `pull_request`
    key, which is present only on PR items and absent on issues.
    """
    if "is_pull_request" in thread:
        return bool(thread["is_pull_request"])
    return "pull_request" in thread


# Same-repo `#123` only. Rejects `owner/repo#9`, `https://.../issues/9`,
# and markdown headings (`# `, `## `). Non-numeric hex colours (#fff) are
# rejected by the digit requirement; numeric ones (#123, #123456) are handled
# by code-span stripping. A numeric hex colour written in bare prose outside
# any code span remains indistinguishable from a reference — that residual
# risk is disclosed, not fixed. Spec section 5 pins this rule; relaxing it
# after seeing results is forbidden.
REFERENCE_RE = re.compile(r"(?<![\w/#])#(\d+)\b")

# Closed fence, unclosed fence, then inline spans. GitHub does not linkify #N
# inside code blocks, so stripping code does not under-count — it removes edges
# that were never valid. Unclosed fences are per-field (see _thread_text);
# they run to end-of-field exactly as markdown renders them. Closed-fence
# branch must come first to avoid the unclosed branch swallowing well-formed
# blocks. Hex colours are numeric and would otherwise read as issue refs
# (#123 -> 123); they live in code spans. Stripping can only DROP candidate
# edges, never add one, which preserves the declared under-counting bias.
_CODE_SPAN_RE = re.compile(
    r"```.*?```"     # a closed fenced block
    r"|```.*"        # an UNCLOSED fence: runs to the end of this field
    r"|`[^`\n]*`",   # an inline span
    re.DOTALL,
)


def _strip_code(text: str) -> str:
    """Blank out code spans so their contents cannot become graph edges."""
    return _CODE_SPAN_RE.sub(" ", text or "")


def extract_references(text: str) -> list[int]:
    """Referenced issue numbers, de-duplicated, in first-appearance order."""
    text = _strip_code(text)
    seen, out = set(), []
    for m in REFERENCE_RE.finditer(text or ""):
        n = int(m.group(1))
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _thread_text(thread: dict) -> str:
    """Thread as text, with code spans stripped per-field before joining.

    Stripping before join bounds any malformed code fence to its own field,
    preventing an unclosed fence in one comment from affecting another.
    """
    parts = [_strip_code(thread.get("body") or "")]
    parts += [_strip_code(c.get("body") or "") for c in thread.get("comments", [])]
    return "\n".join(parts)


def build_graph(threads: list[dict]) -> dict[int, set[int]]:
    """number -> set of numbers it references, restricted to the corpus.

    Edges pointing outside the fetched window are dropped: a chain must be
    fully answerable from the corpus both arms can see.
    """
    present = {t["number"] for t in threads}
    return {
        t["number"]: {n for n in extract_references(_thread_text(t))
                      if n in present and n != t["number"]}
        for t in threads
    }


def extract_chains(threads, graph, max_len: int = 6) -> list[list[int]]:
    """Every maximal simple path through the reference graph.

    A node with no outgoing edge yields a length-1 chain, which is the hop1
    stratum. Cycles terminate because a node already on the path is never
    revisited.
    """
    chains: list[list[int]] = []

    def walk(path: list[int]) -> None:
        if len(path) >= max_len:
            chains.append(list(path))
            return
        nxt = [n for n in sorted(graph.get(path[-1], ())) if n not in path]
        if not nxt:
            chains.append(list(path))
            return
        for n in nxt:
            walk(path + [n])

    for t in sorted(threads, key=lambda t: t["number"]):
        walk([t["number"]])
    return chains


def classify_hops(chain: list[int]) -> str:
    """Stratum key for a chain. Keys match config.WS8_STRATA."""
    n = len(chain)
    if n <= 1:
        return "hop1"
    if n == 2:
        return "hop2"
    return "hop3plus"


_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
_STOPWORDS = frozenset("""a an the is are was were be been being do does did of
in on at to for with by from as it its this that these those and or but if then
than so what which who where when why how not no can could should would will
have has had i you he she we they them us me my your our there here all any
both each few more most other some such only own same too very just about into
over after before between under again further once during above below up down
out off""".split())


def first_sentence(text: str, max_chars: int = 300) -> str:
    """The reporter's opening sentence, with markdown quoting stripped.

    Quoted lines are dropped because a quote is usually someone else's words
    -- often the resolution's -- which would leak the answer into the question.
    """
    lines = [ln for ln in (text or "").splitlines()
             if ln.strip() and not ln.lstrip().startswith((">", "#", "```"))]
    body = " ".join(lines).strip()
    if not body:
        return ""
    sentence = _SENTENCE_END.split(body)[0].strip()
    return sentence[:max_chars]


def build_question(head: dict, chain: list[int]) -> str:
    """A question in the REPORTER's vocabulary. The answer lives at the tail.

    Built from the head's TITLE, not its body (R16): 81% of fetched threads
    are pull requests, so a body-based template pulled in Prow commands and
    template boilerplate ("/kind flake"), and issue bodies were themselves
    unreliable (automated non-English scan reports, raw <img> tags,
    statements rather than reports). Titles are human-written problem
    summaries and need far fewer judgement calls to use safely. first_sentence
    still trims it, in case a title ever needs trimming.

    Inline issue references are redacted from the reporter's sentence before
    embedding: edges in the graph exist precisely because the head references
    the next node, so the head's title can contain the next hop by
    construction. Redacting prevents each multi-hop question from becoming a
    pointer-follow.
    """
    sentence = first_sentence(head.get("title"))
    # Redact inline references like #34 to #… so prose still reads naturally.
    redacted = REFERENCE_RE.sub("#…", sentence)
    question = (
        f"In {head['repo']}, someone reported: \"{redacted}\" "
        f"What was this eventually traced to, and what resolved it?"
    )
    assert not extract_references(question), (
        f"question leaks a chain reference: {question!r}")
    return question


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD.findall(text or "") if len(t) > 2}


def document_frequencies(documents: list[str]) -> tuple[dict[str, int], int]:
    """(token -> number of documents containing it, document count)."""
    df: dict[str, int] = {}
    for doc in documents:
        for t in _tokens(doc):
            df[t] = df.get(t, 0) + 1
    return df, len(documents)


def grep_handle(question: str, evidence: str, df: dict, n_docs: int) -> float:
    """Max corpus-IDF over question tokens that appear in the evidence.

    Reported as a COVARIATE, never as a predictor. The predictive version of
    this metric was tested against the frozen WS6a/WS6c rows on 2026-09-10 and
    FAILED its pre-declared kill criterion: opposite-sign correlations across
    two corpora (−0.110 on fastapi, +0.316 on agentic-hil), both confidence
    intervals spanning zero, against a pre-declared prediction of consistent
    positive correlation. It is kept here only so a stratum that accidentally
    hands grep the answer is visible.

    Note: the IDF formula log(n_docs / (1 + df[t])) can return negative values
    when a token appears in every document (discriminates nothing). This is
    intentional — negative values distinguish "shared but ubiquitous" from
    "no shared tokens at all," both critical to understand in the results.
    """
    if n_docs == 0:
        return 0.0
    shared = (_tokens(question) - _STOPWORDS) & _tokens(evidence)
    if not shared:
        return 0.0
    return max(math.log(n_docs / (1 + df.get(t, 0))) for t in shared)


# Minimum non-stopword tokens a redacted first sentence must carry to count
# as a real question rather than a bare pointer-follow. R9-carried: a head
# whose first sentence is only "see #34" becomes "see #…" once
# build_question redacts it -- one word of content, no question left. The rule
# is: a redacted head must carry at least 4 non-stopword tokens to count as a
# question. Real issue reports are prose, so this only trips on genuinely
# reference-only sentences.
WS8_MIN_HEAD_TOKENS = 4


def one_question_per_head(chains: list[tuple[str, list[int]]]) -> list[tuple[str, list[int]]]:
    """Reduce to one chain per (repo, head), keeping the LONGEST.

    extract_chains enumerates every MAXIMAL simple path, so one head sitting
    in a dense reference cluster yields many chains. Opus review of the first
    Phase 0 run found hop>=3's 30 rows came from only 16 distinct heads, 6 of
    them carrying MORE THAN ONE gold set for the identical question text --
    recall and cost-per-question are undefined when two rows share a prompt.
    One question per head restores a 1:1 mapping between a question and its
    evidence. The longest chain is kept because the hypothesis under test is
    that cost rises with hops: among a head's candidates, the one that pushes
    furthest from hop1 is the one worth keeping.

    Ties (same max length) are broken deterministically by the chain's own
    member numbers -- the lexicographically smaller chain wins -- so the
    result does not depend on graph-traversal or dict-iteration order.

    `chains` is a list of (repo, chain) pairs, matching how
    scripts/non-code-retrieval/build_ws8_questions.py collects them; the head is
    repo-scoped (chain[0] alone is not a stable key, since head numbers
    recur across fastapi/kubernetes/rust-lang).
    """
    best: dict[tuple[str, int], tuple[str, list[int]]] = {}
    for repo, chain in chains:
        key = (repo, chain[0])
        current = best.get(key)
        if current is None or len(chain) > len(current[1]) or (
                len(chain) == len(current[1]) and chain < current[1]):
            best[key] = (repo, chain)
    return list(best.values())


def is_degenerate_head(head: dict) -> bool:
    """True if `head`'s redacted title carries too little signal.

    Mirrors build_question's own redaction step exactly (first_sentence over
    the TITLE, then REFERENCE_RE -> "#…", per R16) so this function answers
    the question build_question will actually ask: is there still a real
    report left after redaction? Degenerate is defined mechanically: fewer
    than WS8_MIN_HEAD_TOKENS non-stopword tokens survive.
    """
    sentence = first_sentence(head.get("title"))
    redacted = REFERENCE_RE.sub("#…", sentence)
    tokens = [t for t in _WORD.findall(redacted.lower()) if t not in _STOPWORDS]
    return len(tokens) < WS8_MIN_HEAD_TOKENS


def select_questions(chains: list[tuple[str, list[int]]], strata: dict, seed: int,
                     is_degenerate=None, is_pr_headed=None,
                     gold_documents=None) -> list[tuple[str, list[int]]]:
    """Fill each stratum to quota, deterministically under `seed`.

    `chains` is a list of (repo, chain) pairs, repo-scoped throughout: a
    stratum draws candidates across every repo at once (hop3plus pulls from
    both kubernetes and rust-lang), and a bare chain-of-numbers is not a
    stable identity across repos -- the same issue number recurs in every
    pinned repo.

    `is_degenerate(repo, chain)` and `is_pr_headed(repo, chain)`, when
    given, are called on each candidate BEFORE it is bucketed by stratum: a
    candidate either flags never enters a bucket, so it can never be picked
    and never counts toward a quota. Both default to None (no filtering).
    `is_pr_headed` is R16: a chain's head must be an issue, never a pull
    request. PRs stay in the corpus as documents and as chain MEMBERS --
    "issue -> the PR that fixed it -> follow-up issue" is the chain WS8
    measures -- but the thread that POSES the question must be the report,
    not the fix.

    `gold_documents(repo, chain) -> set`, when given, enforces PAIRWISE
    DISJOINT evidence WITHIN each stratum: while filling a stratum's quota
    (in seeded-shuffle order), a candidate whose evidence intersects any
    ALREADY-SELECTED question's evidence in that SAME stratum is skipped.
    The constraint is per-stratum, not global -- a hop1 and a hop3plus
    question may legitimately share a document; only questions competing
    for the same stratum's quota are constrained against each other.

    This replaces an earlier "prefer the longest chain within hop3plus"
    tie-break (spec amended commit c0f4b59). That tie-break concentrated the
    stratum into one dense, highly cross-referenced cluster: measured on the
    real corpus, 29 of 30 hop3plus questions came from a single repo, 25 of
    30 shared a gold document with another question, and three documents
    each appeared in 11-12 of the 30 gold sets. The paired bootstrap (spec
    section 7) treats questions as exchangeable draws; overlapping evidence
    means the arms repeat the same retrieval work across correlated
    questions, so the nominal n=30 was actually a much smaller effective n
    -- an underpowered confidence interval that looks precise but is not,
    exactly the WS6c failure this workstream exists to avoid repeating.
    Dropping the length preference buys nothing the hop strata do not
    already encode; every stratum now orders candidates identically, by the
    seeded shuffle alone.

    Default None on `gold_documents` applies no disjointness constraint.
    """
    rng = random.Random(seed)
    buckets: dict[str, list] = {k: [] for k in strata}
    for repo, chain in chains:
        if is_pr_headed is not None and is_pr_headed(repo, chain):
            continue
        if is_degenerate is not None and is_degenerate(repo, chain):
            continue
        buckets.setdefault(classify_hops(chain), []).append((repo, chain))

    picked: list[tuple[str, list[int]]] = []
    for key, quota in strata.items():
        pool = list(buckets.get(key, []))
        rng.shuffle(pool)

        selected: list[tuple[str, list[int]]] = []
        seen_docs: set = set()
        for repo, chain in pool:
            if len(selected) >= quota:
                break
            if gold_documents is not None:
                docs = gold_documents(repo, chain)
                if docs & seen_docs:
                    continue
                seen_docs |= docs
            selected.append((repo, chain))

        if len(selected) < quota:
            raise ValueError(
                f"stratum {key} filled {len(selected)} of {quota} required "
                f"(pool of {len(pool)} candidates before the disjointness "
                "constraint). Spec section 7 Phase 0 forbids widening the "
                "window to recover chains; halt and report the corpus as "
                "unsuitable.")
        picked.extend(selected)
    return picked


def spent_from_checkpoint(path: Path) -> float:
    """Every billed line, INCLUDING superseded duplicates.

    agent_harness.checkpoint_spent() dedups last-wins, which is right for
    resuming a run and wrong for a governor: a retried pair was billed twice
    and the API charged for both. WS6c finished over its cap because of
    exactly this (see results/code-retrieval.md).

    Core principle: any cost that cannot be trusted must never be silently
    converted to a number the governor will act on. A loud failure on corrupt
    data is recoverable by hand; silent under-counting or cap bypass is not.

    Behavior:
    - File does not exist: return 0.0
    - Empty or whitespace-only lines: skipped silently
    - Unparseable JSON on the final line only: skipped (torn write)
    - Unparseable JSON elsewhere: raises (mid-file corruption)
    - cost_total_billed key absent: contributes 0.0 (non-billed row)
    - cost_total_billed present but non-numeric or non-finite: raises with line
      number. Non-numeric includes strings, null, booleans, and non-finite
      floats (NaN, Infinity). Non-finite values silently bypass the cap;
      booleans read as $0 or $1 and hide charges.
    """
    if not Path(path).exists():
        return 0.0

    text = Path(path).read_text()
    lines = text.splitlines()
    total = 0.0

    # Identify the last non-empty line for torn-write detection
    non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]
    if not non_empty_indices:
        return 0.0

    last_non_empty_idx = non_empty_indices[-1]

    for idx, line in enumerate(lines):
        # Skip empty or whitespace-only lines
        if not line.strip():
            continue

        is_final_content_line = (idx == last_non_empty_idx)

        # Parse JSON
        try:
            data = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            if is_final_content_line:
                # Torn write: skip silently
                continue
            else:
                # Mid-file corruption: raise loudly
                raise ValueError(
                    f"Unparseable JSON at line {idx + 1}: {line[:100]}"
                )

        # Ensure we have a dict
        if not isinstance(data, dict):
            # Non-dict JSON. This is data corruption.
            raise ValueError(
                f"Line {idx + 1}: JSON is not an object: {line[:100]}"
            )

        # Extract cost_total_billed
        if "cost_total_billed" not in data:
            # Key absent: contributes 0.0
            continue

        cost = data["cost_total_billed"]

        # Reject booleans explicitly (before numeric check, since isinstance(True, int) is True)
        if isinstance(cost, bool):
            raise ValueError(
                f"Line {idx + 1}: cost_total_billed is non-numeric (boolean): {cost!r}"
            )

        # Key is present: must be numeric and finite
        try:
            fval = float(cost)
        except (TypeError, ValueError):
            raise ValueError(
                f"Line {idx + 1}: cost_total_billed is non-numeric: {cost!r}"
            )

        # Reject non-finite values (NaN, Infinity, -Infinity)
        if not math.isfinite(fval):
            raise ValueError(
                f"Line {idx + 1}: cost_total_billed is non-finite: {cost!r}"
            )

        total += fval

    return round(total, 6)


def gate_branch(accuracy: float, any_doc_hit: int) -> str:
    """Spec section 7 Phase 1. G1 proceeds; G2 halts for a decision.

    Both signals must be zero. WS6a's 0.625 parametric floor is exactly what
    made its other arms uninterpretable, so a non-zero floor is never waved
    through as "small".
    """
    return "G1" if (accuracy <= WS8_GATE_PASS and any_doc_hit == 0) else "G2"


# Every WS8 checkpoint file, relative to a data directory -- live AND
# archived. Ruling R18 (2026-09-11): a checkpoint that recorded real billed
# spend is RENAMED to `<name>.voided-<ISO8601>.jsonl` when its run is
# superseded, never deleted. This glob picks up archived files
# automatically, so a voided run's spend is never silently invisible to the
# governor. The live gate script still reads only the canonical
# `gate_checkpoint.jsonl` for RESUME purposes -- that stays the ONLY file
# consulted to decide which pairs still need to run, which is what keeps a
# fresh run from mixing with a superseded one. total_spent_across_checkpoints
# is what keeps the governor's SPEND total cumulative across both.
WS8_CHECKPOINT_GLOB = "ws8_cache/*.jsonl"


def total_spent_across_checkpoints(data_dir: Path) -> float:
    """Cumulative billed spend across every WS8 checkpoint, live or archived.

    Sums spent_from_checkpoint() (every billed line, including superseded
    duplicates from a retried pair) over every file matching
    WS8_CHECKPOINT_GLOB under `data_dir`.

    Deleting a checkpoint instead of archiving it makes its spend invisible
    here -- exactly the failure this function exists to prevent. Live
    incident, 2026-09-11: Phase 1's gate was re-run under a corrected system
    prompt and the voided run's checkpoint was DELETED first ("no chance of
    mixing runs"), which destroyed the record of $0.7833 in real, billed API
    spend -- the governor could no longer see it, reproducing the exact WS6c
    failure mode (see results/code-retrieval.md) that spent_from_checkpoint
    was built to prevent, one task after it shipped. Fixed by reconstructing
    the lost row as an archived checkpoint and adopting "rename, never
    delete" as the standing rule for every remaining WS8 phase.
    """
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob(WS8_CHECKPOINT_GLOB))
    return round(sum(spent_from_checkpoint(p) for p in paths), 6)


# ------------------------------------------------------------ the spend ledger
#
# HOISTED HERE on 2026-09-11 from four verbatim copies (scripts/non-code-retrieval/gate_ws8.py,
# scripts/non-code-retrieval/index_ws8.py, scripts/non-code-retrieval/retry_ws8_q065.py, scripts/non-code-retrieval/run_ws8.py), each of
# whose docstrings acknowledged the duplication and promised to keep it in sync
# by hand. The promise was the problem: the ledger is the artifact that actually
# lost money, and a fix to it had to land in four places at once.
#
# This module's docstring promises "no network, no model calls" and keeps it --
# but not purity: `git_short_sha`/`git_is_dirty` shell out to git, exactly as
# ws6c.checkout_problems does and for the same reason. The alternative is a
# fifth copy.

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# A cent. Below this the two figures disagree only by rounding across a
# hundred-odd checkpoint lines; at or above it, a row is missing.
LEDGER_TOLERANCE_USD = 0.01


def git_short_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
            text=True, cwd=_REPO_ROOT, timeout=5,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_is_dirty() -> bool:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            cwd=_REPO_ROOT, timeout=5,
        )
        return bool(proc.stdout.strip())
    except Exception:
        return True  # fail toward the more-honest ("don't trust this") side


def append_spend_ledger(path, *, run_label: str, spend_usd: float,
                        note: str) -> None:
    """Append one row to the committed spend ledger (R20).

    Unlike data/ws8_cache/, results/ws8/spend_ledger.csv is git-tracked, so
    the cumulative spend figure survives even a total loss of data/.

    Column mechanism (fixed after a real incident, 2026-09-11): a run cannot
    know the commit that will eventually contain its own outputs, so a
    single `source_commit` column stamped with `git rev-parse HEAD` at
    run time is a category error -- two of this ledger's first four rows
    named a commit that contained neither their evidence nor, in one case,
    even the code that produced them (the max_tokens parameter was still
    uncommitted when the q065 retry ran). Two fields instead:

    - `ran_at_head`: HEAD's short sha AT RUN TIME. Honest about what it is
      -- the last commit, NOT necessarily the commit containing the code
      that actually executed (see `dirty`).
    - `dirty`: True if the working tree had uncommitted changes at run
      time. A `ran_at_head` value on a dirty run is a LOWER BOUND on what
      ran, not a description of it -- callers must not treat it as if it
      were.
    - `committed_in`: the commit that actually contains this row's code and
      outputs. NOT knowable at append time (git commit hashes are
      content-addressed over content that includes this very file), so it
      is written empty here and must be filled in AFTER the commit that
      lands this row lands -- by hand, or via a small follow-up commit, but
      never guessed at run time. A blank `committed_in` on a row older than
      the current commit means the fill-in step was skipped, not that none
      is needed.

    CALL THIS FROM A `finally`. The incident that motivated hoisting it: an
    invocation of run_ws8.py checkpointed two pairs (ws8_q028/indexed
    $0.176158 and ws8_q011/agentic $0.074400) and then died before reaching
    its append, so $0.250558 of real billed spend never reached the ledger
    at all. The money was already gone by the time the exception was raised;
    the only thing still recoverable at that point was the record of it.
    """
    path = Path(path)
    row = pd.DataFrame([{
        "run_label": run_label, "spend_usd": round(spend_usd, 6),
        "ran_at_head": git_short_sha(), "dirty": git_is_dirty(),
        "committed_in": "", "note": note,
    }])
    if path.exists():
        existing = pd.read_csv(path)
        out = pd.concat([existing, row], ignore_index=True)
    else:
        out = row
    out.to_csv(path, index=False)


def reconcile_spend(ledger_path, data_dir, *,
                    tolerance: float = LEDGER_TOLERANCE_USD) -> dict:
    """The ledger's SUM against the checkpoint SWEEP. Never raises.

    This specific comparison, and not the tempting cheaper one. Comparing a
    governor's `spent` against the checkpoint sum WITHIN one invocation would
    not have caught the missing-row incident: inside the surviving invocation
    those two agree perfectly, because the governor and the checkpoint were
    both written by the process that lived. It is the committed ledger's total
    -- across every invocation, including the ones that died -- measured
    against a sweep of every checkpoint on disk that shows a row is absent.

    Returns ledger_total / checkpoint_total / delta / agrees. `delta` is
    positive when the checkpoints know about spend the ledger does not, which
    is the direction a lost row produces.
    """
    ledger_path = Path(ledger_path)
    ledger_total = 0.0
    if ledger_path.exists():
        ledger_total = round(
            float(pd.read_csv(ledger_path)["spend_usd"].astype(float).sum()), 6)
    checkpoint_total = total_spent_across_checkpoints(Path(data_dir))
    delta = round(checkpoint_total - ledger_total, 6)
    return {
        "ledger_total": ledger_total,
        "checkpoint_total": checkpoint_total,
        "delta": delta,
        "tolerance": tolerance,
        "agrees": bool(abs(delta) <= tolerance),
    }


def warn_on_ledger_mismatch(ledger_path, data_dir, *,
                            tolerance: float = LEDGER_TOLERANCE_USD,
                            stream=None) -> dict | None:
    """Print a loud warning naming BOTH figures if they disagree. Never raises.

    Warn, never raise: this runs after the money is already spent, and an
    exception here would turn a successful, fully-checkpointed run into a
    failed one over a bookkeeping discrepancy that the run itself may not have
    caused. A reconciliation that can abort a run is a reconciliation people
    switch off.

    Returns the reconcile_spend dict, or None if reconciliation itself failed
    (which is also reported, for the same reason).
    """
    import sys as _sys

    stream = stream if stream is not None else _sys.stderr
    try:
        rec = reconcile_spend(ledger_path, data_dir, tolerance=tolerance)
    except Exception as exc:                             # pragma: no cover
        print(f"\nWARNING: spend reconciliation could not run ({exc!r}). "
              "The ledger and the checkpoints have NOT been compared.",
              file=stream)
        return None
    if rec["agrees"]:
        print(f"spend reconciled: ledger ${rec['ledger_total']:.6f} == "
              f"checkpoints ${rec['checkpoint_total']:.6f}", file=stream)
        return rec
    print(
        "\n" + "!" * 72 +
        f"\nWARNING: WS8 SPEND LEDGER DISAGREES WITH THE CHECKPOINTS"
        f"\n  ledger sum        ${rec['ledger_total']:.6f}  ({ledger_path})"
        f"\n  checkpoint sweep  ${rec['checkpoint_total']:.6f}  "
        f"({Path(data_dir) / WS8_CHECKPOINT_GLOB})"
        f"\n  difference        ${rec['delta']:.6f}  "
        f"(tolerance ${rec['tolerance']:.2f})"
        "\n  A positive difference means real billed spend is recorded in a "
        "checkpoint\n  but has NO row in the committed ledger -- most likely "
        "an invocation that\n  died before it wrote one. Reconstruct the row "
        "by hand; do NOT delete a\n  checkpoint to make the numbers agree."
        "\n" + "!" * 72, file=stream)
    return rec
