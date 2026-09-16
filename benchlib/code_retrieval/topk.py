"""Statistics, corpus-guard and selection helpers for WS6c. No network.

The design WS6a already shipped is PAIRED -- the same 40 questions run in
every arm -- so per-question differences carry far more information than two
medians. Everything above `checkout_problems` operates on those differences
and is pure.

The pin guard at the bottom is the one exception to the module's original
"no I/O" promise: it shells out to `git` to confirm a candidate checkout is
still the pinned corpus. It lives here rather than in a script because BOTH
scripts/code-retrieval/gate_ws6c.py (Phase 2b) and scripts/code-retrieval/index_ws6c.py (Phase 3) must run
it before spending, and a second verbatim copy is a guard that can drift out
of agreement with itself.
"""

from __future__ import annotations

import fnmatch
import math
import subprocess
from pathlib import Path

import numpy as np

from ..config import WS6C_CANDIDATES, WS6C_COMMITS, ws6c_checkout
from ..recall_quality.core import bootstrap_indices, percentile_ci

# prompt_tokens is the only defensible token column. The API reports
# input_tokens as the UNCACHED REMAINDER, so a 960k-token cached arm has a
# median input_tokens of 28 and would rank as the cheapest. WS6a pins this in
# a regression test; WS6c re-derives it rather than trusting a stored column.
_PROMPT_TOKEN_COLUMNS = (
    "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
)


def metric_series(df, metric: str):
    """qid -> float for one metric, computed from raw usage columns."""
    if metric == "prompt_tokens":
        vals = sum(df[c].astype(float) for c in _PROMPT_TOKEN_COLUMNS)
    elif metric == "judge_correct":
        vals = df[metric].astype(str).str.lower().isin(("true", "1")).astype(float)
    else:
        vals = df[metric].astype(float)
    return dict(zip(df["qid"].astype(str), vals))


def paired_deltas(df, metric: str, arm_a: str, arm_b: str):
    """(qids, arm_a - arm_b) over the questions BOTH arms answered.

    Raises on any qid present in one arm and missing from the other: a silent
    inner join would quietly redefine n, and n is the whole argument here.
    """
    a = metric_series(df[df["arm"] == arm_a], metric)
    b = metric_series(df[df["arm"] == arm_b], metric)
    only = (set(a) ^ set(b))
    if only:
        raise ValueError(f"unpaired qids between {arm_a!r} and {arm_b!r}: "
                         f"{sorted(only)}")
    qids = sorted(a)
    return qids, np.array([a[q] - b[q] for q in qids], dtype=float)


def paired_summary(deltas, n_boot: int, seed: int, alpha: float) -> dict:
    """Median paired delta with a percentile bootstrap CI and a sign test.

    The bootstrap resamples QUESTIONS, not arms, which is what keeps the
    interval paired. One draw, shared by every metric a caller asks for.
    """
    d = np.asarray(deltas, dtype=float)
    n = int(d.size)
    if n == 0:
        raise ValueError("no paired observations")
    idx = bootstrap_indices(n, n_boot, seed)
    boot = np.median(d[idx], axis=1)
    lo, hi = percentile_ci(boot, alpha)
    n_pos = int((d > 0).sum())
    return {
        "n": n,
        "median": float(np.median(d)),
        "mean": float(d.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "n_positive": n_pos,
        "sign_p": sign_test_p(n_pos, int((d != 0).sum())),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def sign_test_p(n_positive: int, n: int) -> float:
    """Exact two-sided sign test. math.comb only -- scipy is a transitive pin
    in requirements.lock and is deliberately not imported."""
    if n == 0:
        return 1.0
    k = min(n_positive, n - n_positive)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def resolve_branch(excludes_zero: bool, median: float, labels) -> str:
    """(positive-separation, negative-separation, no-separation) -> branch name.

    The branch is a LOOKUP against the pre-registered spec, never a judgement
    made after seeing the data.
    """
    positive, negative, none = labels
    if not excludes_zero:
        return none
    return positive if median > 0 else negative


# ------------------------------------------------------------- the pin guard
#
# Moved here from scripts/code-retrieval/gate_ws6c.py so Phase 2b and Phase 3 share ONE
# implementation. Phase 3 needs it strictly more than Phase 2b did: the gate
# only ran the parametric arm, which never opens the checkout, whereas Phase 3
# INDEXES this tree. A dirty tree indexed is a corpus that is not the corpus
# pinned and measured, and nothing downstream can see the difference.

DIRTY_PATHS_SHOWN = 10


def checkout_problems(slug: str) -> list[str]:
    """Every way one candidate checkout can fail to be the pinned corpus.

    Two conditions, because a pin is only half the guarantee:

    * HEAD must equal `config.WS6C_COMMITS[slug]`. A `git pull` or a stray
      checkout moves the corpus out from under a measurement that claims a
      pinned SHA, and nothing else enforces this -- the clones are gitignored,
      so they live outside this repo's own history.
    * The working tree must be clean, untracked files included. An edited file
      or a stray untracked one is indexed like any other, so the corpus
      retrieved from would silently stop being the corpus pinned and measured.
    """
    root = ws6c_checkout(slug)
    expected = WS6C_COMMITS[slug]
    if not (root / ".git").exists():
        return [f"{slug}: no checkout at {root}"]

    problems: list[str] = []
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != expected:
        problems.append(
            f"{slug}: HEAD is {head}, config.WS6C_COMMITS pins {expected}")

    # --porcelain includes untracked files by default; keep it that way.
    dirty = [line for line in subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"],
        text=True).splitlines() if line.strip()]
    if dirty:
        shown = dirty[:DIRTY_PATHS_SHOWN]
        more = len(dirty) - len(shown)
        # Name the paths: a guard that says only "dirty" wastes the next
        # person's time working out which file to deal with.
        detail = "; ".join(entry.strip() for entry in shown)
        if more:
            detail += f"; ... and {more} more"
        problems.append(
            f"{slug}: working tree is not clean ({len(dirty)} entr"
            f"{'y' if len(dirty) == 1 else 'ies'}): {detail}")

    if not problems:
        print(f"  pin ok  {slug:16s} {head}  (clean tree)")
    return problems


def assert_pinned_checkouts(slugs=None) -> None:
    """Refuse to spend unless every named checkout is pinned and clean.

    Collects problems across ALL slugs before raising, so one run reports
    every bad checkout rather than sending the operator round the loop once
    per repo. `slugs=None` checks every candidate (Phase 2b's behaviour);
    Phase 3 passes the single selected slug, since the two rejected clones
    are never opened again and a stray edit in one of them must not block a
    run that cannot touch it.
    """
    if slugs is None:
        slugs = [cand["slug"] for cand in WS6C_CANDIDATES]
    problems: list[str] = []
    for slug in slugs:
        problems.extend(checkout_problems(slug))
    if problems:
        raise SystemExit(
            "Refusing to spend: candidate checkouts are not the pinned "
            "corpus.\n  " + "\n  ".join(problems))


# ------------------------------------------- what claude-context will index
#
# Read verbatim out of the PINNED package's own dist/ -- the same method that
# established mcp_env's settings and search_code's default limit:
#   @zilliz/claude-context-mcp@0.1.15
#     -> node_modules/@zilliz/claude-context-core/dist/context.js
#        DEFAULT_SUPPORTED_EXTENSIONS (line 74), DEFAULT_IGNORE_PATTERNS (85)
#     -> dist/utils/ignore-matcher.js  (hasHiddenSegment)
#
# This is NOT our WS6A_TEXT_EXTENSIONS allowlist and the difference matters:
# .yaml/.yml/.json/.toml/.txt/.sh/.html/.css are commented OUT of the
# package's list, so the agentic arm can grep files the indexed arm was never
# given. Making the gap computable is what lets it be REPORTED rather than
# assumed away.
#
# CALIBRATION: applied to the frozen WS6a fastapi checkout this predicts
# exactly 2,825 files -- the number the server's own completion log reported
# for that index (results/ws6a/index_cost.csv, indexed_files_reported). It is
# therefore an independent check on index completeness, not a guess.

CC_SUPPORTED_EXTENSIONS = frozenset({
    ".ts", ".tsx", ".js", ".jsx", ".py", ".java", ".cpp", ".c", ".h", ".hpp",
    ".cs", ".go", ".rs", ".php", ".rb", ".swift", ".kt", ".scala", ".m", ".mm",
    ".dart", ".sol", ".md", ".markdown", ".ipynb",
})

# The directory names DEFAULT_IGNORE_PATTERNS excludes. Dot-prefixed entries
# (.git, .vscode, .idea, .cache, .nyc_output, .pytest_cache) are listed for
# completeness only -- hasHiddenSegment already drops every path with a
# dot-prefixed segment, which is also why fastapi's .github/*.md files are
# absent from its index.
CC_IGNORE_DIRS = frozenset({
    "node_modules", "dist", "build", "out", "target", "coverage",
    ".nyc_output", ".vscode", ".idea", ".git", ".svn", ".hg", ".cache",
    "__pycache__", ".pytest_cache", "logs", "tmp", "temp",
})

CC_IGNORE_GLOBS = (
    "*.swp", "*.swo", ".env", ".env.*", "*.local", "*.min.js", "*.min.css",
    "*.min.map", "*.bundle.js", "*.bundle.css", "*.chunk.js", "*.vendor.js",
    "*.polyfills.js", "*.runtime.js", "*.map", "*.log",
)


def claude_context_indexable(paths) -> list[str]:
    """The subset of `paths` the pinned claude-context build will index.

    Pure: takes repo-relative POSIX paths (as `ws6a.git_ls_files` returns)
    and applies the package's own three filters in its own order -- hidden
    segment, ignored directory, ignored basename glob, supported extension.

    Sound only for a CLEAN checkout, which is why the pin guard above runs
    first: the server walks the filesystem, not the index, so an untracked
    file it can see would not appear in `paths`.
    """
    out = []
    for p in paths:
        parts = p.split("/")
        if any(seg.startswith(".") for seg in parts):
            continue
        if any(seg in CC_IGNORE_DIRS for seg in parts[:-1]):
            continue
        if any(fnmatch.fnmatch(parts[-1], g) for g in CC_IGNORE_GLOBS):
            continue
        if Path(p).suffix in CC_SUPPORTED_EXTENSIONS:
            out.append(p)
    return out


# ------------------------------------------------- the derived corpus tree
#
# UNPLANNED, AND IT CHANGES PHASE 3. Pointed at the pinned agentic_hil
# checkout, @zilliz/claude-context-mcp@0.1.15 indexed ZERO files:
#
#   [Context] 📁 Found 0 code files
#   [BACKGROUND-INDEX] ✅ Indexing completed successfully! Files: 0, Chunks: 0
#
# Cause, reproduced against the package's own dist/ rather than inferred.
# Context.findIgnoreFiles() slurps EVERY root-level `.*ignore` file, not just
# `.gitignore`, and merges their patterns. agentic-hil ships the standard
# minimal-Docker-context idiom:
#
#   **            <- ignore everything
#   !evals/       <- then re-include only what the build needs
#   !src/ ...
#
# IgnoreMatcher.ignores(path, isDirectory) tests the NON-slash form first and
# returns on the first hit, so `src` matches `**` and is pruned before
# `!src/` is ever consulted. The recursive walk therefore never descends into
# any top-level directory. Verified directly: IgnoreMatcher(['**','!src/',...])
# reports ignores('src', true) == true.
#
# Nothing an operator can set fixes this: CUSTOM_IGNORE_PATTERNS, the
# index_codebase `ignorePatterns` argument and `.contextignore` are all
# ADDITIVE, and `.contextignore` sorts before `.dockerignore` so its
# negations would be overridden anyway. This is a reportable property of the
# pinned package: any repository carrying a `**`-style .dockerignore is
# silently unindexable by it, and `get_indexing_status` calls that success.
#
# THE FIX, and its cost. Both tool-using arms run against a tree derived from
# the pinned checkout by dropping the root-level non-.gitignore ignore files.
# What that costs is exactly nothing measurable:
#
#   * `.dockerignore` has no suffix, so it is not in WS6A_TEXT_EXTENSIONS. It
#     is NOT one of the 342 files or 4,307,363 count_tokens that
#     results/ws6c/corpus_stats.csv pins as the corpus. The measured corpus
#     is byte-identical either way.
#   * It is dot-prefixed, so hasHiddenSegment excludes it from any index even
#     when the walk works.
#   * No question in results/ws6c/questions.csv references it.
#
# BOTH tool-using arms use this one tree, so they still differ only in
# retrieval strategy. The pinned checkout itself is never modified -- the pin
# guard still runs against it, and every copied byte is sha256-verified
# against it.

INDEX_TREE_KEPT_IGNORE_FILE = ".gitignore"


def index_tree(slug: str) -> Path:
    """Derived tree both tool-using arms see. Never a git checkout."""
    from ..config import DATA_DIR

    return DATA_DIR / f"ws6c_tree_{slug}"


def dropped_ignore_files(paths) -> list[str]:
    """Root-level `.*ignore` files claude-context merges, minus `.gitignore`.

    Pure. Mirrors Context.findIgnoreFiles(), which reads the codebase ROOT
    only -- a nested `.gitignore` or `Dockerfile.dockerignore` is never
    consulted and is therefore never dropped.
    """
    return sorted(
        p for p in paths
        if "/" not in p and p.startswith(".") and p.endswith("ignore")
        and p != INDEX_TREE_KEPT_IGNORE_FILE
    )


def materialise_index_tree(root, paths, dest) -> dict:
    """Copy `paths` (minus the dropped ignore files) from `root` into `dest`.

    Idempotent, and verified rather than trusted: every file in `dest` is
    sha256-compared against its source and the file SET is compared both
    ways, so a stale or partial tree from an earlier run is a loud failure
    rather than a silently different corpus.
    """
    import hashlib
    import shutil

    root, dest = Path(root), Path(dest)
    dropped = dropped_ignore_files(paths)
    wanted = [p for p in paths if p not in set(dropped)]

    for p in wanted:
        target = dest / p
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / p, target)

    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    on_disk = sorted(
        str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
    if on_disk != sorted(wanted):
        extra = sorted(set(on_disk) - set(wanted))
        missing = sorted(set(wanted) - set(on_disk))
        raise SystemExit(
            f"derived tree {dest} does not match the pinned checkout: "
            f"{len(extra)} extra {extra[:5]}, {len(missing)} missing "
            f"{missing[:5]} -- delete it and re-run")
    mismatched = [p for p in wanted if _sha(dest / p) != _sha(root / p)]
    if mismatched:
        raise SystemExit(
            f"{len(mismatched)} file(s) in {dest} differ from the pinned "
            f"checkout: {mismatched[:5]} -- delete it and re-run")

    # One digest over the whole tree, so a reader can reproduce the derivation
    # and check it in a single comparison rather than 381 of them. Defined as
    # sha256 of "<path> <file sha256>\n" lines in sorted path order -- the same
    # value the loop above already verified file by file.
    tree_sha = hashlib.sha256(
        "".join(f"{p} {_sha(dest / p)}\n" for p in sorted(wanted))
        .encode()).hexdigest()

    return {"tree": str(dest), "files_copied": len(wanted),
            "dropped_ignore_files": "|".join(dropped),
            "tree_sha256": tree_sha}
