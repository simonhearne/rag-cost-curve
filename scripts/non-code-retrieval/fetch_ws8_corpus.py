#!/usr/bin/env python
"""WS8 Phase 0a: fetch post-cutoff issues/PRs. NO model calls, $0.

Reads GITHUB_TOKEN from the environment. Writes raw JSON per repo and records
sha256 into data/MANIFEST.json. The corpus itself is NEVER committed
(data/ws8_corpus/ is gitignored).

R13 (deterministic, capped fetch): sort and direction are passed EXPLICITLY
-- without them GitHub's default order could change under us and pagination
is not guaranteed stable, so the corpus would not be reproducible.
WS8_MAX_THREADS_PER_REPO is a SAFETY VALVE (raised 1000 -> 8000 on
2026-09-11), not a sampling rule -- the created_at window defines the
corpus; the cap should not bind at 8000. It bound at 1000: kubernetes
stopped at 2026-07-10 and rust-lang at 2026-06-21 against a window running
to 2026-09-01, truncating the corpus to ~30% of what the window intended.

R14 (pull requests are kept): the /issues endpoint returns PRs alongside
issues. WS8_INCLUDE_PULL_REQUESTS is True -- "issue -> the PR that fixed it
-> follow-up issue" is the canonical multi-hop chain this workstream exists
to measure. Each raw thread records is_pull_request (a PR has a
`pull_request` key on the GitHub payload) so the split is visible in
corpus_stats.csv rather than assumed.

Dedupe (found live, 2026-09-11): GitHub's issue-list pagination can repeat an
item across a page boundary when two threads share `created_at` -- observed
as kubernetes/kubernetes#140406 fetched twice, byte-identical both times.
Deduped by (repo, number) via ws8.dedupe_threads before the raw JSON is
written and before PR/issue counts are computed, so a repeated item cannot
inflate the document count, the token total, or the PR/issue split.

Comment collection (WS8_COMMENTS_BULK, fixed 2026-09-11): the first fetch
pulled each thread's comments from `it["comments_url"]` with a single
per_page=100 request and NO page loop -- five rust-lang threads had exactly
100 comments, so that call silently truncated. A lost comment is a lost `#N`
reference, i.e. a lost graph edge -- the ground truth this workstream
measures. Comments are now fetched from the REPO-level endpoint
(/repos/{repo}/issues/comments), paged fully, and joined back to threads by
the issue number parsed from each comment's `issue_url`. This also cuts the
call count from roughly one-per-thread (~10,000 calls at full-window size)
to a few hundred. KNOWN GAP, disclosed rather than hidden: this endpoint
returns issue comments (including comments on PRs) but NOT inline PR review
comments -- those were absent before this fix too, so nothing regresses.
Every thread's comment count is checked against the issue listing's own
`comments` field; a shortfall is recorded in the manifest, never swallowed.

Rate limits: a 403/429 response is treated as a pause, not a failure -- the
run sleeps until the limit resets (primary) or `Retry-After` elapses
(secondary) and retries the SAME request, so a multi-hour run across the
5,000/hour ceiling does not abort partway. Resume is per-repo: a repo whose
raw file already exists AND was written under the CURRENT cap and comment
strategy is skipped, so a second invocation after an interrupted run does
not redo completed repos (an incomplete repo is simply redone in full --
no intra-repo checkpoint).
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.non_code_retrieval import bench as ws8
from benchlib.manifest import load_manifest
from benchlib.non_code_retrieval.bench import ACCEPTED_FETCH_BASES, FETCH_BASIS
from benchlib.config import (DATA_DIR, WS8_CORPUS_DIR, WS8_FETCH_DIRECTION,
                             WS8_FETCH_SORT, WS8_INCLUDE_PULL_REQUESTS,
                             WS8_MAX_THREADS_PER_REPO, WS8_REPOS,
                             WS8_WINDOW_END, WS8_WINDOW_START)

API = "https://api.github.com"

# Between pages, well inside the secondary rate limit.
PAGE_SLEEP_S = 0.5


def _get_with_backoff(url: str, params: dict, headers: dict,
                      max_attempts: int = 30) -> requests.Response:
    """GET with rate-limit-aware backoff. Never lets a 403/429 abort the run.

    Primary rate limit (403, X-RateLimit-Remaining: 0): sleep until
    X-RateLimit-Reset. Secondary/abuse limit (403 or 429 with Retry-After):
    sleep that many seconds. Transient 5xx: capped exponential backoff.
    Anything else: raised immediately -- a real error, not a rate limit.
    """
    for attempt in range(1, max_attempts + 1):
        r = requests.get(url, params=params, headers=headers, timeout=60)
        if r.status_code == 200:
            return r

        if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - int(time.time()), 1) + 5
            print(f"  primary rate limit hit; sleeping {wait}s until reset "
                  f"(attempt {attempt}/{max_attempts})", flush=True)
            time.sleep(wait)
            continue

        if r.status_code in (403, 429) and "retry-after" in {
                k.lower() for k in r.headers}:
            wait = int(r.headers.get("Retry-After", 30)) + 2
            print(f"  secondary rate limit; sleeping {wait}s "
                  f"(attempt {attempt}/{max_attempts})", flush=True)
            time.sleep(wait)
            continue

        if r.status_code in (500, 502, 503, 504):
            wait = min(2 ** attempt, 60)
            print(f"  transient {r.status_code}; retrying in {wait}s "
                  f"(attempt {attempt}/{max_attempts})", flush=True)
            time.sleep(wait)
            continue

        r.raise_for_status()

    raise RuntimeError(f"exceeded {max_attempts} retry attempts for {url}")


def fetch_issues(repo: str, token: str, max_threads: int) -> tuple[list[dict], int, int, int]:
    """Fetch up to `max_threads` issues/PRs from `repo` inside the pinned window.

    `since` narrows the candidate set server-side (GitHub filters it on
    updated_at, a superset of created_at within the window since updated_at
    can never precede created_at); the created_at window itself is enforced
    explicitly below, so a stale item bumped by a bot after the window start
    cannot slip in just because it was recently touched.

    With sort=created&direction=asc (WS8_FETCH_SORT/WS8_FETCH_DIRECTION,
    pinned in config.py -- not chosen here), the filtered set is monotonic in
    created_at: once an item's created_at reaches WS8_WINDOW_END, every
    remaining item -- the rest of this page and every later page -- is also
    >= WS8_WINDOW_END, so the fetch stops there instead of paging to the end
    of the repo's history.

    Comments are NOT fetched here -- see fetch_all_comments(). Each thread
    carries `expected_comment_count` (from the issue listing's own
    `comments` field) so the caller can verify the join against it, and an
    empty `comments` list to be filled in by that join.

    Returns (threads, n_prs, n_issues, n_duplicates).
    """
    headers = {"Authorization": f"Bearer {token}",
               "Accept": "application/vnd.github+json"}
    threads: list[dict] = []
    page = 1
    window_start = f"{WS8_WINDOW_START}T00:00:00Z"
    window_end = f"{WS8_WINDOW_END}T00:00:00Z"

    while len(threads) < max_threads:
        r = _get_with_backoff(
            f"{API}/repos/{repo}/issues",
            params={"state": "all", "since": window_start,
                    "sort": WS8_FETCH_SORT, "direction": WS8_FETCH_DIRECTION,
                    "per_page": 100, "page": page},
            headers=headers)
        batch = r.json()
        if not batch:
            break

        stop = False
        for it in batch:
            if it["created_at"] >= window_end:
                stop = True
                break  # ascending order: nothing later in the fetch is earlier
            if it["created_at"] < window_start:
                continue

            is_pr = "pull_request" in it
            threads.append({
                "repo": repo, "number": it["number"], "title": it["title"],
                "body": it.get("body") or "", "user": it["user"],
                "created_at": it["created_at"], "state": it["state"],
                "is_pull_request": is_pr,
                "expected_comment_count": it.get("comments", 0),
                "comments": [],
            })
            if len(threads) >= max_threads:
                stop = True
                break

        if stop:
            break
        page += 1
        time.sleep(PAGE_SLEEP_S)

    threads, n_duplicates = ws8.dedupe_threads(threads)
    n_prs = sum(1 for t in threads if t["is_pull_request"])
    n_issues = len(threads) - n_prs
    return threads, n_prs, n_issues, n_duplicates


def fetch_all_comments(repo: str, token: str) -> dict[int, list[dict]]:
    """All issue comments in `repo` since WS8_WINDOW_START, grouped by issue number.

    WS8_COMMENTS_BULK: the repo-level endpoint, paged fully (sort=created,
    direction=asc, so a comment's page position is chronological and the
    per-issue groups it's split into stay chronological too). `since` is
    safe here even though it filters on the COMMENT's own created_at: a
    comment on an in-window issue always has created_at >= the issue's
    created_at >= WS8_WINDOW_START, so nothing in-window is excluded -- only
    comments on issues we don't keep anyway are filtered out early. No upper
    bound is applied (an in-window issue can receive a comment any time up
    to today), so this pages to "now", not to WS8_WINDOW_END.

    GitHub hard-caps pagination on this resource at 30,000 results (page 300
    at per_page=100): a 422 "pagination is limited for this resource",
    verified live against rust-lang/rust, whose repo-wide comment volume
    since WS8_WINDOW_START exceeds that depth. NOT a rate limit and NOT
    transient -- retrying the same request forever would never succeed. The
    fix advances `since` to the created_at of the LAST comment fetched
    before the 422 and restarts pagination from page 1 there: the new query
    covers a much smaller remaining time range, so it lands well under the
    depth cap. Comments are deduped by their own `id` across this restart
    (and any further ones), since a comment sharing the exact boundary
    timestamp could otherwise be re-counted.
    """
    headers = {"Authorization": f"Bearer {token}",
               "Accept": "application/vnd.github+json"}
    since = f"{WS8_WINDOW_START}T00:00:00Z"
    by_number: dict[int, list[dict]] = {}
    seen_ids: set[int] = set()
    total_fetched = 0

    while True:
        page = 1
        last_created_at = None
        hit_depth_limit = False

        while True:
            try:
                r = _get_with_backoff(
                    f"{API}/repos/{repo}/issues/comments",
                    params={"since": since, "sort": "created", "direction": "asc",
                            "per_page": 100, "page": page},
                    headers=headers)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 422:
                    hit_depth_limit = True
                    break
                raise

            batch = r.json()
            if not batch:
                return by_number  # truly done: no more comments at all

            for c in batch:
                cid = c["id"]
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                number = int(c["issue_url"].rsplit("/", 1)[-1])
                by_number.setdefault(number, []).append({
                    "user": c["user"], "body": c.get("body") or "",
                    "created_at": c["created_at"],
                })
                last_created_at = c["created_at"]
                total_fetched += 1

            page += 1
            if page % 20 == 0:
                print(f"  ...{repo}: {total_fetched} comments fetched so far "
                      f"(since {since})", flush=True)
            time.sleep(PAGE_SLEEP_S)

        if last_created_at is None:
            raise RuntimeError(
                f"{repo}: hit GitHub's pagination depth limit with zero new "
                f"comments fetched since {since} -- cannot advance further")
        print(f"  {repo}: pagination depth limit (422) reached; advancing "
              f"since {since} -> {last_created_at} and continuing", flush=True)
        since = last_created_at


def already_fetched(out: Path) -> bool:
    """True if `out` was already written under an accepted fetch basis.

    The per-repo resume check: a raw file from an old cap, an old window, an
    old sort order or the old per-thread comment strategy must be redone, not
    trusted.
    """
    if not out.exists():
        return False
    key = str(out.resolve().relative_to(DATA_DIR.resolve()))
    entry = load_manifest().get("files", {}).get(key)
    # ACCEPTED_FETCH_BASES, not `== FETCH_BASIS`: the v2 basis added the window
    # and sort order, and a v1 file is still current while the pins v1 left
    # implicit are unmoved. See benchlib/non_code_retrieval/bench.py for why
    # that equivalence is conditional rather than permanent.
    return bool(entry) and entry.get("fetch_basis") in ACCEPTED_FETCH_BASES


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("GITHUB_TOKEN is not set -- Phase 0 halts rather than guessing "
                  "a credential source")
    if not WS8_REPOS:
        sys.exit("config.WS8_REPOS is empty -- pin the repo list before fetching")
    if not WS8_INCLUDE_PULL_REQUESTS:
        sys.exit("WS8_INCLUDE_PULL_REQUESTS is False -- Phase 0 assumes PRs are "
                  "kept (R14); flip the pin deliberately if that changed")

    raw = WS8_CORPUS_DIR / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    for repo in WS8_REPOS:
        out = raw / f"{repo.replace('/', '__')}.json"

        if already_fetched(out):
            # Name the basis the FILE carries, not the one the code would
            # stamp: those differ for a v1 file, and printing the current
            # constant would assert something about the file that is not true.
            key = str(out.resolve().relative_to(DATA_DIR.resolve()))
            on_disk = load_manifest()["files"][key].get("fetch_basis")
            print(f"{repo}: already fetched under an accepted basis "
                  f"({on_disk}); skipping (resume)")
            continue

        print(f"{repo}: fetching issues/PRs...", flush=True)
        threads, n_prs, n_issues, n_dup = fetch_issues(repo, token, WS8_MAX_THREADS_PER_REPO)
        print(f"{repo}: {len(threads)} threads ({n_issues} issues, {n_prs} PRs); "
              f"fetching comments in bulk...", flush=True)

        comments_by_number = fetch_all_comments(repo, token)

        mismatches = []
        for t in threads:
            comments = comments_by_number.get(t["number"], [])
            expected = t.pop("expected_comment_count")
            if len(comments) != expected:
                mismatches.append({"number": t["number"], "expected": expected,
                                   "actual": len(comments)})
            t["comments"] = comments

        created_ats = [t["created_at"] for t in threads]
        actual_min = min(created_ats) if created_ats else None
        actual_max = max(created_ats) if created_ats else None

        # Staleness guard: stamped onto every thread, in the raw JSON itself
        # -- not only in data/MANIFEST.json -- so a stale file left over from
        # an interrupted or superseded fetch is self-describing and cannot
        # silently join a corpus a downstream script treats as current. Both
        # build scripts assert this before processing any raw file.
        fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for t in threads:
            t["_fetch_basis"] = FETCH_BASIS
            t["_fetched_at"] = fetched_at

        out.write_text(json.dumps(threads, indent=2))
        manifest.record(out, {
            "repo": repo, "threads": len(threads), "prs": n_prs, "issues": n_issues,
            "duplicate_threads_dropped": n_dup,
            "window_pinned": f"{WS8_WINDOW_START}..{WS8_WINDOW_END}",
            "created_at_min": actual_min, "created_at_max": actual_max,
            "sort": WS8_FETCH_SORT, "direction": WS8_FETCH_DIRECTION,
            "max_threads_per_repo": WS8_MAX_THREADS_PER_REPO,
            "comment_count_mismatches": len(mismatches),
            "comment_count_mismatch_detail": mismatches[:20],
            "fetch_basis": FETCH_BASIS,
            "fetched_at": fetched_at,
        })

        dup_note = f", {n_dup} duplicate(s) dropped" if n_dup else ""
        mismatch_note = f", {len(mismatches)} comment-count mismatch(es)" if mismatches else ""
        print(f"{repo}: {len(threads)} threads ({n_issues} issues, {n_prs} PRs){dup_note}"
              f"{mismatch_note} -> {out}")
        print(f"{repo}: actual created_at range {actual_min} .. {actual_max}")
        if mismatches:
            print(f"  WARNING: {len(mismatches)} thread(s) had a comment-count "
                  f"mismatch; see data/MANIFEST.json", file=sys.stderr)


if __name__ == "__main__":
    main()
