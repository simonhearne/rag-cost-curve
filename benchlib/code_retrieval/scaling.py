"""WS9: nested size subsets of the WS6c corpus, and the question panel.

Pure functions only -- no network, no API keys, no Milvus, no file I/O beyond
what the caller hands in. Everything here is decided from the tree and from
gold file paths, never from results.

Design: docs/superpowers/specs/2026-09-15-ws9-unseen-corpus-cost-curve-design.md
"""

from ..config import WS9_ELIGIBILITY_SCOPE, WS9_SCOPES, WS9_SUBSETS


def ws9_subset_files(paths, scope: str) -> list[str]:
    """Restrict `paths` to one WS9 scope, preserving input order.

    A scope is an (include, exclude) pair rather than WS6a's include-only
    prefix tuple: WS9's M tier is "the whole repository except tests/", which
    an include list cannot express without naming every root-level file.

    An empty `include` means "everything"; an empty `exclude` means "drop
    nothing". Prefixes carry their trailing slash, so "tests/" excludes
    tests/ and leaves testsuite/ alone.
    """
    if scope not in WS9_SUBSETS:
        raise ValueError(
            f"unknown WS9 scope {scope!r}; expected one of {list(WS9_SCOPES)}")
    spec = WS9_SUBSETS[scope]
    include, exclude = spec["include"], spec["exclude"]
    out = [p for p in paths if not include or p.startswith(include)]
    if exclude:
        out = [p for p in out if not p.startswith(exclude)]
    return out


def ws9_scopes_are_nested(paths) -> bool:
    """S strictly inside M strictly inside L over `paths`, and S non-empty.

    Asserted at run time before anything is indexed or billed. The whole
    design rests on the three size points sitting on ONE corpus; a tier
    definition that broke nesting would make the curve three nearly-identical
    corpora instead, and nothing downstream could see the difference.

    The test is STRICT, and requires S to be non-empty, because the subset
    relation alone passes trivially on degenerate input: if `paths` arrives in
    a spelling the include prefixes never match -- absolute paths, or
    "./src/..." -- then S is empty, the empty set is a subset of everything,
    and this guard would wave through an index built over an EMPTY S tier.
    Strict growth also catches the other degenerate shape, three tiers that
    are the same set, which would spend the sweep's budget measuring one
    corpus size three times.
    """
    s, m, l = (set(ws9_subset_files(paths, k)) for k in WS9_SCOPES)
    return bool(s) and s < m < l


def ws9_eligible_qids(questions, scope: str = WS9_ELIGIBILITY_SCOPE) -> list[str]:
    """qids whose EVERY `expected_files` path lies inside `scope`.

    `questions` is the WS6c questions.csv frame. Eligibility is decided from
    gold file paths alone -- never from a run, a score or a difficulty -- and
    it is evaluated once, at the smallest scope, so the panel is identical at
    every size.

    A question with no gold files is not eligible: there is nothing to check
    it against, so it cannot be shown to be answerable inside the subset.
    """
    out = []
    for _, row in questions.iterrows():
        gold = [p for p in str(row["expected_files"]).split(";") if p]
        if gold and set(ws9_subset_files(gold, scope)) == set(gold):
            out.append(str(row["qid"]))
    return sorted(out)


def ws9_collection_prefix(scope: str) -> str:
    """Milvus collection prefix for one scope.

    Distinct from WS6c's `ws6c_<slug>` so the L index WS6c already built is
    never retargeted, and so index_ws6c.py's "more than one collection matches
    this prefix" guard stays meaningful.
    """
    if scope not in WS9_SUBSETS:
        raise ValueError(
            f"unknown WS9 scope {scope!r}; expected one of {list(WS9_SCOPES)}")
    return f"ws9_{scope.lower()}"
