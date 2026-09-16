#!/usr/bin/env python
"""WS9: the unseen-corpus size sweep.

37-question panel x (agentic, indexed, indexed_topk3) x (S, M). The L rows are
REUSED from the frozen results/ws6c/runs.csv -- 74 new runs per size, not 111.

Outputs: results/ws6c/scaling.csv, results/ws6c/scaling_provenance.txt

THE TREE. Each size point runs against the SAME persisted directory
scripts/code-retrieval/index_ws9.py indexed (data/ws9_subset_{s,m}), never a
fresh temporary one: claude-context keys an index by ABSOLUTE codebase path, so
indexing one path and searching another silently returns nothing and the sweep
measures an EMPTY index rather than a small one. The agentic arm greps the same
tree, so both arms see exactly the same corpus at each size point.

TWO MCP SESSIONS PER SIZE, ONE CHECKPOINT. claude-context builds its tool list
at session entry, so binding search_code's `limit` needs its own session; the
two indexed arms therefore cannot be interleaved WITH EACH OTHER. They are each
interleaved against the same question order, under the same seed, against the
same index. Recorded in scaling_provenance.txt rather than left to be inferred.

SELF-RETRYING, NEVER RE-BILLING. execute_run does not raise on a failed pair:
an agent error after the SDK's 8 retries, a judge exception, or an MCP tool
failure leaves the pair OUT of the checkpoint and the loop moves on. Left
alone, that surfaces hours later as assert_run_complete with M never run. So
each MCP session is re-opened and resumed from the checkpoint until every one
of its pairs is present, up to SESSION_ATTEMPTS, with a pause between
attempts. Nothing already checkpointed is ever re-run. BudgetExceeded is never
retried.

THE GOVERNOR IS WS9-SCOPED. total_spent_across_checkpoints fails closed across
WS6a and WS6c by design; seeding from it would start this run at ~$70 against a
$55 cap. WS9_SPEND_CHECKPOINT_GLOBS covers data/ws9_cache/ only -- still
cumulative across WS9's own phases, so the smoke run and both sizes draw on one
budget.

Run:  .venv/bin/python scripts/code-retrieval/scale_ws9.py --smoke   # 12 pairs, no CSV
      .venv/bin/python scripts/code-retrieval/scale_ws9.py           # the full 222
"""

import argparse
import asyncio
import functools
import json
import re
import sys
from pathlib import Path

import anyio.to_thread
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import agent_harness, manifest
from benchlib.code_retrieval import bench as ws6a
from benchlib.code_retrieval import scaling as ws9
from benchlib.config import (
    DATA_DIR, WS6A_PREFIX_BUDGET_TOKENS, WS6C_ALPHA, WS6C_BOOT_N,
    WS6C_BOOT_SEED, WS6C_TOPK, WS9_ARMS, WS9_BUDGET_USD,
    WS9_ELIGIBILITY_SCOPE, WS9_MATCH_BAND, WS9_PANEL_N, WS9_SLUG,
    WS9_SMOKE_QUESTIONS, WS9_SWEEP_SIZES, ws6a_dir, ws6c_dir, ws9_cache_dir,
    ws9_subset_tree,
)

# The declared basis in spec section 7: 222 new runs (37 questions x 3 arms x
# 2 new sizes) at WS6c's measured per-question-per-size cost of $0.3625. Quoted
# in the provenance run record beside what was actually billed.
WS9_DECLARED_BASIS_USD = 26.83

TOPK_ARM = f"indexed_topk{WS6C_TOPK}"
# indexed_topk<K> runs the `indexed` code path under its own published label,
# so the two never collide in the shared checkpoint.
ARM_KINDS = {TOPK_ARM: "indexed"}


def new_governor() -> ws6a.CostGovernor:
    gov = ws6a.CostGovernor(
        limit=WS9_BUDGET_USD,
        spent=agent_harness.total_spent_across_checkpoints(
            DATA_DIR, globs=agent_harness.WS9_SPEND_CHECKPOINT_GLOBS))
    print(f"governor: ${gov.spent:.2f} spent across WS9 phases, "
          f"${gov.remaining:.2f} of ${WS9_BUDGET_USD:.2f} left")
    return gov


SESSION_ATTEMPTS = 4        # per (scope, session): re-open the MCP session and resume
SESSION_RETRY_PAUSE_S = 90  # long enough for an overloaded/429 window to clear
FOUND_RE = re.compile(r"Found (\d+) results")


def _contains(exc, types) -> bool:
    """`exc` is one of `types`, whether raised bare or wrapped by a group.

    anyio's task groups (stdio_client and ClientSession each open one) re-raise
    whatever escapes them inside a BaseExceptionGroup, so a bare isinstance
    test sees the group, not the cause. Every "is this a reason to STOP rather
    than retry?" test in this script goes through here: a wrapped
    BudgetExceeded that is treated as transient would sleep and resume
    SPENDING, and a wrapped Ctrl-C would be caught and ignored.
    """
    if isinstance(exc, types):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return exc.subgroup(types) is not None
    return False


# Never retried, never swallowed: a tripped governor, an operator interrupt, a
# SystemExit raised by one of the gates, or a cancellation from above.
STOP_EXCEPTIONS = (ws6a.BudgetExceeded, KeyboardInterrupt, SystemExit,
                   asyncio.CancelledError)


def _drop_spend_only(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Drop execute_run's spend-only `<arm>_judge_error` rows before the gates.

    When the judge raises, execute_run charges the already-billed agent call
    and appends a spend-only checkpoint line under a DISTINCT arm label,
    leaving the real pair to be retried on resume (agent_harness ~line 862).
    That row is not a run. Left in, it fails assert_no_errors on its `error`
    field, shows up in assert_run_complete as an unexpected pair, and makes the
    333-row check see 334 -- and because it is in the checkpoint for good,
    every resumed run would abort in the same place and scaling.csv would never
    be written without hand-editing a cache file.

    Dropped from the GATES, never hidden: the row stays in the checkpoint, the
    governor still counts its spend, and each one is printed here.
    """
    keep = df["arm"].isin(WS9_ARMS)
    dropped = df[~keep]
    kept = df[keep].copy()
    if not len(dropped):
        return kept
    print(f"[{scope}] {len(dropped)} spend-only row(s) dropped from the gates "
          "(billed, checkpointed, and retried on resume):", file=sys.stderr)
    for _, row in dropped.iterrows():
        print(f"    {row['qid']}  {row['arm']}  {row.get('error', '')}",
              file=sys.stderr)

    # Dropping the ROWS is not enough. A spend-only row is the only kind that
    # carries `error` and `note`, so the moment one joins the frame pandas
    # backfills NaN into those fields for every REAL row -- and float('nan')
    # is TRUTHY, so assert_no_errors would flag all 222 kept rows as errored
    # and the run would abort one line further down than before (verified: a
    # 4-row frame with one judge_error row still fails on the other 3). So
    # restore `error` to the None a clean row has -- never touching a real
    # error string, which must still stop the run -- and drop the columns that
    # ONLY the dropped rows populated, so scaling.csv's schema is the same
    # whether or not a judge ever raised.
    if "error" in kept.columns:
        # Element-wise, not .where(..., None): pandas turns that None straight
        # back into NaN, which is the value being got rid of. A clean run's
        # column is object dtype holding None (verified against
        # results/ws6c/runs.csv, where `error` is None for all 160 rows).
        kept["error"] = [None if pd.isna(v) else v for v in kept["error"]]
    introduced = [c for c in kept.columns
                  if c != "error" and kept[c].isna().all()
                  and dropped[c].notna().any()]
    if introduced:
        print(f"[{scope}] columns carried only by those rows, dropped: "
              f"{introduced}", file=sys.stderr)
        kept = kept.drop(columns=introduced)
    return kept


def _missing(checkpoint, qids, arms) -> list:
    done = agent_harness.load_checkpoint(checkpoint)
    return [(q, a) for q in qids for a in arms if (q, a) not in done]


async def run_session(client, scope, tree, topk, arms, tag, questions, prices,
                      judge_prompt, checkpoint) -> None:
    """One MCP session's pairs, retried until complete or out of attempts.

    Each attempt opens a FRESH claude-context session -- a server process that
    died mid-run gets replaced -- and execute_run resumes from the checkpoint,
    so a pair that was billed is never billed again. The stderr log is opened
    in append mode by the harness, so every attempt's server output is kept.
    """
    qids = list(questions["qid"])
    for attempt in range(1, SESSION_ATTEMPTS + 1):
        todo = _missing(checkpoint, qids, arms)
        if not todo:
            print(f"[{scope}/{tag}] complete: {len(qids) * len(arms)} pairs in the checkpoint")
            return
        print(f"\n=== [{scope}] MCP session {tag}: topk={topk} arms={arms} "
              f"attempt {attempt}/{SESSION_ATTEMPTS}, {len(todo)} pair(s) to run ===")
        try:
            async with agent_harness.claude_context_session(
                    tree, ws9.ws9_collection_prefix(scope), topk=topk,
                    stderr_log=ws9_cache_dir()
                    / f"mcp_run_{scope.lower()}_{tag}.stderr.log") as (
                    session, mcp_tools, raw):
                print(f"[{scope}] claude-context tools: {[t.name for t in raw]}")
                if not mcp_tools:
                    raise RuntimeError("claude-context exposed no tools; the "
                                       "indexed arms cannot run")
                # execute_run is synchronous by design (the arms must differ
                # only in the tools list). The MCP tools are sync wrappers
                # bridging back to THIS event loop via anyio.from_thread.run,
                # which requires the sync code to run inside
                # anyio.to_thread.run_sync -- calling execute_run directly
                # here would deadlock on the first tool call.
                await anyio.to_thread.run_sync(functools.partial(
                    agent_harness.execute_run,
                    client, questions, arms,
                    root=tree, prefix_text=None, mcp_tools=mcp_tools,
                    prices=prices, checkpoint_path=checkpoint,
                    judge_prompt=judge_prompt, governor=new_governor(),
                    transcript_dir=ws6c_dir() / f"transcripts_ws9_{scope.lower()}",
                    arm_kinds=ARM_KINDS,
                ))
        except BaseException as exc:
            if _contains(exc, STOP_EXCEPTIONS):
                raise
            # Anything else -- a dead server, a torn-down task group, a
            # transport error -- is a reason to re-open, not to stop. The
            # checkpoint is intact: execute_run appends per pair.
            print(f"[{scope}/{tag}] attempt {attempt} raised "
                  f"{type(exc).__name__}: {exc} -- checkpoint intact, "
                  "will re-open the session", file=sys.stderr)
        still = _missing(checkpoint, qids, arms)
        if still and attempt < SESSION_ATTEMPTS:
            print(f"[{scope}/{tag}] {len(still)} pair(s) still missing after "
                  f"attempt {attempt}; pausing {SESSION_RETRY_PAUSE_S}s before "
                  "resuming", file=sys.stderr)
            await asyncio.sleep(SESSION_RETRY_PAUSE_S)
    still = _missing(checkpoint, qids, arms)
    if still:
        raise RuntimeError(
            f"[{scope}/{tag}] {len(still)} pair(s) still missing after "
            f"{SESSION_ATTEMPTS} attempts: {still[:10]} -- read "
            f"{ws9_cache_dir()}/mcp_run_{scope.lower()}_{tag}.stderr.log; "
            "re-running this script resumes from the checkpoint")


async def sweep_one(client, scope, questions, prices, judge_prompt) -> pd.DataFrame:
    tree = ws9_subset_tree(scope)
    if not tree.exists():
        raise SystemExit(
            f"{tree} does not exist -- run "
            f"scripts/code-retrieval/index_ws9.py --scope {scope} first. The "
            "index is keyed by this absolute path; a tree created here would "
            "not be the tree that was indexed.")

    # PRE-FLIGHT before any spend: a missing ripgrep must fail here, not 100
    # pairs in with the agentic arm silently absent from the results.
    agentic = agent_harness.make_agentic_tools(tree)
    print(f"[{scope}] agentic toolchain OK: {[t.name for t in agentic]}")

    checkpoint = ws9_cache_dir() / f"scale_{scope.lower()}.jsonl"
    # topk=None first, so the default-top-k arm is measured against exactly the
    # frozen WS6c tool binding.
    sessions = ((None, ("indexed", "agentic"), "default"),
                (WS6C_TOPK, (TOPK_ARM,), f"topk{WS6C_TOPK}"))
    for topk, arms, tag in sessions:
        await run_session(client, scope, tree, topk, arms, tag, questions,
                          prices, judge_prompt, checkpoint)
    return pd.DataFrame(list(agent_harness.load_checkpoint(checkpoint).values()))


def _search_result_counts(path: Path) -> list[int]:
    """Every "Found N results" count in this transcript's TOOL RESULT frames.

    Scoped to tool_result blocks deliberately. Regexing the whole file also
    matched ASSISTANT PROSE, so an answer that happened to quote "Found 12
    results" -- a plausible thing for a model to write about a search -- would
    abort the sweep with a false "the bind did not take" AFTER every pair had
    been billed. The count that decides the check must come from the tool, not
    from the model talking about the tool.
    """
    counts = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        for block in json.loads(line).get("content", []):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            body = block.get("content")
            if isinstance(body, list):  # content blocks rather than plain text
                body = "".join(b.get("text", "") for b in body
                               if isinstance(b, dict))
            counts += [int(n) for n in FOUND_RE.findall(str(body))]
    return counts


def check_topk_bind(scope: str, qids) -> None:
    """The one check usage numbers cannot make: that the bound arm's tool
    frames never show more than WS6C_TOPK results.

    Asserted as `never more than K`, not `exactly K`: at L the default arm
    returned 7-19 results per call and the bound arm 2-3, so a small corpus
    can legitimately return fewer. A frame ABOVE K means the wrapper binding
    did not take and the k=3 half of the design is measuring the default.
    """
    tdir = ws6c_dir() / f"transcripts_ws9_{scope.lower()}" / TOPK_ARM
    for qid in qids:
        path = tdir / f"{qid}.jsonl"
        if not path.exists():
            raise SystemExit(f"[{scope}] {TOPK_ARM} {qid}: no transcript at {path}")
        counts = _search_result_counts(path)
        if not counts:
            raise SystemExit(
                f"[{scope}] {TOPK_ARM} {qid}: no 'Found N results' tool-result "
                "frame -- search_code was never called; the row was answered "
                "without the index")
        if max(counts) > WS6C_TOPK:
            raise SystemExit(
                f"[{scope}] {TOPK_ARM} {qid}: 'Found {max(counts)} results' -- "
                f"the limit={WS6C_TOPK} bind did not take. STOP: the k=3 half "
                "of the design would be measuring the default.")
    print(f"[{scope}] {TOPK_ARM} bind verified on {len(qids)} transcript(s): "
          f"every search_code frame returned <= {WS6C_TOPK} results")


def reuse_l_rows(panel) -> pd.DataFrame:
    """The L cell, read from the FROZEN WS6c Phase 3 rows. Never re-run.

    Read from runs.csv, NOT topk_runs.csv: topk_runs.csv is WS6c Phase 1, 40
    indexed_topk3 rows on FASTAPI with WS6a-style qids (q02, q39). Reading it
    here would silently substitute the contaminated corpus for the
    uncontaminated one at the very arm the top-k half of this design rests on.
    """
    runs = pd.read_csv(ws6c_dir() / "runs.csv")
    rows = runs[runs["arm"].isin(WS9_ARMS) & runs["qid"].isin(panel)].copy()
    want = len(panel) * len(WS9_ARMS)
    if len(rows) != want:
        raise SystemExit(
            f"expected {want} reusable L rows from results/ws6c/runs.csv "
            f"({len(panel)} panel questions x {len(WS9_ARMS)} arms), found "
            f"{len(rows)}")
    # Belt and braces: these must be agentic_hil qids, not fastapi ones.
    if not rows["qid"].astype(str).str.startswith(WS9_SLUG).all():
        raise SystemExit("reused L rows are not all agentic_hil qids -- the "
                         "wrong file was read")
    rows["size_point"] = "L"
    return rows


def _run_record(combined: pd.DataFrame) -> str:
    """Per-size pair counts and run dates, read off the frame being written.

    Not hardcoded: the dates are the min and max `started_at` of the rows that
    actually went into scaling.csv, so the week of wall-clock between the
    reused L rows and the new S and M rows cannot drift out of date here.
    """
    day = combined["started_at"].astype(str).str[:10]
    dates = combined.assign(day=day).groupby("size_point")["day"].agg(
        ["min", "max"])
    pairs = combined.groupby("size_point").size()
    lines = ""
    for scope in ("S", "M", "L"):
        lo, hi = dates.loc[scope, "min"], dates.loc[scope, "max"]
        when = lo if lo == hi else f"{lo} to {hi}"
        lines += f"  {scope}: {int(pairs[scope]):>3} pairs, run {when}\n"
    return lines


def write_provenance(path: Path, panel, stats, combined) -> None:
    tests_share_tiktoken = (stats.loc["L", "tiktoken_cl100k"]
                            - stats.loc["M", "tiktoken_cl100k"]) / stats.loc[
                                "L", "tiktoken_cl100k"]
    tests_share_count = (stats.loc["L", "count_tokens"]
                         - stats.loc["M", "count_tokens"]) / stats.loc[
                             "L", "count_tokens"]
    over_budget = stats.loc["L", "count_tokens"] / WS6A_PREFIX_BUDGET_TOKENS
    spent = agent_harness.total_spent_across_checkpoints(
        DATA_DIR, globs=agent_harness.WS9_SPEND_CHECKPOINT_GLOBS)
    path.write_text(
        "# results/ws6c/scaling.csv -- how it was produced\n"
        "# Pre-registration: docs/superpowers/specs/"
        "2026-09-15-ws9-unseen-corpus-cost-curve-design.md\n"
        f"corpus: agentic-hil/agentic-hil @ {stats.loc['L', 'commit']} "
        "(Apache-2.0), the corpus WS6c's Phase 2 gate cleared at 0.000\n"
        f"arms: {', '.join(WS9_ARMS)}   sizes: S, M (new) + L (reused)\n"
        f"panel: {len(panel)} questions\n"
        "\n"
        "PRE-REGISTRATION SUMMARY. Everything below was fixed in writing before\n"
        "any WS9 row existed, so every verdict is a lookup rather than a\n"
        "judgement made after seeing the data. This block exists because the\n"
        "spec under docs/ is stripped at public release and this file is then\n"
        "the only pre-registration artefact that survives.\n"
        "\n"
        "  HYPOTHESES (spec section 3).\n"
        "  H1  the saving widens with corpus size. Agentic cost rises with how\n"
        "      much tree it must search; indexed cost is set by a fixed batch\n"
        "      of ranked chunks and should be flatter. So c_live - c_idx grows\n"
        "      with size and Q* falls with size.\n"
        "  H2  at S the agentic arm may be CHEAPER than the index, making Q*\n"
        "      infinite there. A 49-file tree is cheap to grep. What H2 firing\n"
        "      would establish is that size dependence exists on an\n"
        "      uncontaminated corpus too -- that grep's advantage is partly a\n"
        "      function of how much tree there is, on a corpus with a 0.000\n"
        "      parametric floor. H2 is the hypothesis this design exists to be\n"
        "      able to lose to, and the report states its outcome whichever way\n"
        "      it falls. H2 firing does NOT make scale an alternative\n"
        "      explanation for the fastapi result: grep wins on fastapi at\n"
        "      6,232,509 count_tokens and loses on agentic-hil at 4,307,363, so\n"
        "      no monotone size effect produces that ordering. Contamination is\n"
        "      settled by the matched-size C-series, not by H2.\n"
        "  H3  the fixed cost does not move. Under the headline infra mode\n"
        "      (dedicated r8g.large, pca_uc_384_sq8) the index's fixed daily\n"
        "      cost is flat across all three sizes, so the shape of this curve\n"
        "      is set entirely by the per-query saving; the footprint is\n"
        "      load-bearing only in marginal mode.\n"
        "\n"
        "  BRANCHES (spec section 4). Cells outside these series emit\n"
        "  'n/a (not pre-registered)', and no letter refers to anything but the\n"
        "  quantity its series names.\n"
        "\n"
        "  F -- finiteness of break-even on the unseen corpus, over the agentic\n"
        "       vs indexed pair at the headline infra mode and footprint,\n"
        "       Claude regime.\n"
        "    F1  break_even_qpd finite at S, M AND L.\n"
        "    F2  finite at some sizes and infinite at others; the report names\n"
        "        which.\n"
        "    F3  infinite at every size (live cheaper per query throughout).\n"
        "\n"
        "  B -- the abstract's band, only over sizes where F gives a finite Q*.\n"
        "    B1  every finite Q* inside [10, 1000) queries/day.\n"
        "    B2  at least one finite Q* outside it; the report names the size\n"
        "        and the number.\n"
        "\n"
        "  K -- the top-k ordering across the curve.\n"
        "    K1  Q*(indexed, k=10) < Q*(indexed_topk3, k=3) at every size: the\n"
        "        order the deck quotes at L is preserved across the curve.\n"
        "    K2  the order reverses at some size: the two cross below L.\n"
        "    K3  at least one size has an infinite Q* on one arm, so the pair\n"
        "        is not ordered there.\n"
        "\n"
        "  U -- per-size paired token deltas, resolved independently for each\n"
        "       (contrast, size) cell over prompt_tokens ARM CONTRASTS ONLY.\n"
        "       Contrasts: indexed - agentic, indexed_topk3 - agentic,\n"
        "       indexed_topk3 - indexed.\n"
        "    U1  95% CI excludes zero and is negative (the index arm costs\n"
        "        fewer tokens).\n"
        "    U2  CI excludes zero and is positive.\n"
        "    U3  CI includes zero: report the point estimate with its interval\n"
        "        and claim NEITHER direction.\n"
        "    judge_correct and turns deltas are computed and published as\n"
        "    supporting mechanism and carry 'n/a (not pre-registered)'. No\n"
        "    U-letter anywhere refers to anything but a prompt_tokens delta.\n"
        "    The per-arm size-span deltas (L-S, L-M within an arm) are\n"
        "    exploratory and carry 'n/a (not pre-registered)' too.\n"
        "\n"
        "  C -- the matched-size contamination test, over agentic vs indexed\n"
        "       alone (results/ws6c/matched_size_contamination.csv). Two\n"
        "       scopes are matched iff their measured count_tokens are within\n"
        f"       {WS9_MATCH_BAND:.1f}x of each other; pairing is by measured "
        "tokens, never by\n"
        "       file count.\n"
        "    C1  the per-query winner DIFFERS across corpora at matched size.\n"
        "        Size alone does not explain the fastapi result; contamination\n"
        "        remains the candidate explanation.\n"
        "    C2  the per-query winner is the SAME on both corpora at matched\n"
        "        size. The matched comparison does not support the\n"
        "        corpus-identity explanation, and the report says so.\n"
        "    C3  no pair falls inside the band on measurement, so the test does\n"
        "        not run and nothing is concluded from it.\n"
        "    THE C-SERIES IS NOT PAIRED AND NO INTERVAL IS COMPUTED ACROSS IT:\n"
        "    the two corpora have different questions, different n (10 or 40 on\n"
        "    fastapi against 37 here) and no correspondence between rows. It is\n"
        "    a described comparison of point estimates, and the CSV carries\n"
        "    that in a column, not only in prose.\n"
        "\n"
        "  STATISTICAL PINS, NOT ANALYST CHOICES ON THE DAY. WS6C_BOOT_N =\n"
        f"  {WS6C_BOOT_N:,}, WS6C_BOOT_SEED = {WS6C_BOOT_SEED} and WS6C_ALPHA ="
        f" {WS6C_ALPHA} are\n"
        "  reused FROZEN from WS6c. A cell that resolves U3 may NOT be\n"
        "  re-resolved at a looser alpha, a different seed or a larger resample\n"
        "  count. Seed and alpha sensitivity may be REPORTED as a sensitivity\n"
        "  check, labelled as such, and it never displaces the committed\n"
        "  branch.\n"
        "\n"
        "  MULTIPLICITY, DISCLOSED RATHER THAN CORRECTED. The U-series is NINE\n"
        f"  UNADJUSTED CELLS at alpha = {WS6C_ALPHA} -- 3 contrasts x 3 sizes,\n"
        "  each resolved against its own 95% interval with no multiplicity\n"
        "  correction. Under a global null that is ~0.45 expected U1/U2\n"
        "  resolutions by chance across the nine. The correction is\n"
        "  deliberately not applied, because choosing one after seeing how many\n"
        "  cells resolved is a worse analyst degree of freedom than declaring\n"
        "  the unadjusted rate here. A SINGLE ISOLATED U1 OR U2 AMONG NINE\n"
        "  CELLS IS THEREFORE NOT EVIDENCE OF ANYTHING and the report must not\n"
        "  present one as a finding; only a coherent pattern across sizes in\n"
        "  one contrast may be discussed, and even then as a pattern, not as\n"
        "  nine independent tests.\n"
        "\n"
        "  UNDERPOWERED BY DESIGN, SAID IN ADVANCE (spec section 4.1). WS6c\n"
        "  Phase 3 at n = 40 reached P3 on its primary comparison -- the\n"
        "  interval missed excluding zero by 85 tokens. WS9's per-size cells\n"
        f"  are n = {len(panel)} on a smaller span, so U3 at every size is the "
        "EXPECTED\n"
        "  outcome and is not a disappointment. BINDING COROLLARY: the report\n"
        "  may state the curve and its point estimates; it may not say the\n"
        "  index wins at a size whose U-cell resolved U3.\n"
        "\n"
        "THE PANEL. Eligibility is decided ONCE, at the smallest scope, from\n"
        "gold file paths alone: a question is eligible iff EVERY path in its\n"
        "expected_files lies inside S (src/). 37 of WS6c's 40 qualify; q09\n"
        "(examples/), q30 (tools/) and q31 (tests/) do not. The same 37 run at\n"
        "every size, so no question enters or leaves the panel as the corpus\n"
        "grows -- a curve whose points are drawn over different question\n"
        "populations has a shape that is partly a composition artifact.\n"
        "\n"
        "THE SUBSETS. Nested directory tiers, decided from the tree before any\n"
        "WS9 row existed. S = src/; M = the whole repository MINUS tests/;\n"
        f"L = the whole repository. tests/ is {tests_share_tiktoken:.1%} of "
        "this corpus's\n"
        f"tiktoken_cl100k tokens ({tests_share_count:.1%} of its count_tokens) "
        "in ONE directory,\n"
        "so splitting it out is the only division of this tree that gives even\n"
        "log spacing. The spec's pre-registered tiering figure is the\n"
        "tiktoken_cl100k share; the count_tokens share is the one the x-axis\n"
        "below is drawn on. Measured count_tokens:\n"
        + "".join(f"  {s}: {int(stats.loc[s, 'count_tokens']):>10,} tokens, "
                  f"{int(stats.loc[s, 'files_text_allowlisted']):>4} files\n"
                  for s in ("S", "M", "L"))
        + "\n"
        "THE TREES. Each size point runs against data/ws9_subset_<scope>, cut\n"
        "from the DERIVED tree data/ws6c_tree_agentic_hil and sha256-verified\n"
        "against it file by file. The derived tree is the pinned checkout minus\n"
        "its root-level .dockerignore, which makes the pinned package index\n"
        "ZERO files while reporting success -- so each subset inherits that fix\n"
        "and is provably a subset of the exact corpus WS6c measured.\n"
        "claude-context indexes, and the agentic arm greps, that same absolute\n"
        "path.\n"
        "\n"
        "L IS REUSED, NOT RE-RUN. The L rows come from the frozen\n"
        "results/ws6c/runs.csv filtered to the panel and to these three arms.\n"
        "NOT from topk_runs.csv: that file is WS6c Phase 1, 40 indexed_topk3\n"
        "rows on FASTAPI with WS6a-style qids.\n"
        "\n"
        "INTERLEAVING. claude-context builds its tool list at session entry, so\n"
        "binding search_code's limit needs a second MCP session. Each size is\n"
        "therefore two sequential execute_run calls against ONE checkpoint:\n"
        f"('indexed', 'agentic') first, then ('{TOPK_ARM}',). The two indexed\n"
        "arms are NOT interleaved with each other. Within a session the arms\n"
        "are round-robined and EACH ARM GETS ITS OWN SHUFFLE of the panel,\n"
        "keyed on the seed AND the arm name -- see interleaved_order in\n"
        "benchlib/code_retrieval/bench.py -- so no arm sees questions in\n"
        "authoring order and time-of-day variance in API latency cannot bias\n"
        "one arm. Every session runs against the same index.\n"
        "\n"
        f"NO STUFFED ARM AT ANY SIZE. This corpus measures "
        f"{int(stats.loc['L', 'count_tokens']):,}\n"
        f"count_tokens against a {WS6A_PREFIX_BUDGET_TOKENS:,}-token prefix "
        f"budget -- {over_budget:.2f}x over --\n"
        "so the arm is INFEASIBLE, not omitted. prefix_text is None throughout.\n"
        "\n"
        "THE SMOKE CHECK SHARES THIS CHECKPOINT, so its rows are re-used rather\n"
        "than re-billed and are the first 12 of the 222. It is a MECHANISM\n"
        "check only -- both indexed arms must show turns > 1, and EVERY\n"
        f"search_code frame in the bound arm must return AT MOST {WS6C_TOPK} "
        "results\n"
        "(check_topk_bind asserts 'never more than K', not 'exactly K': a small\n"
        "corpus can legitimately return fewer, while a frame ABOVE K means the\n"
        "wrapper binding did not take) -- and its outcome may drive only stop\n"
        "or continue. It may not change the panel, the arms, the subsets, the\n"
        "branches or the budget.\n"
        "\n"
        "THE GOVERNOR IS WS9-SCOPED, AND THIS IS A DELIBERATE DEVIATION (spec\n"
        "section 7.1). agent_harness.total_spent_across_checkpoints deliberately\n"
        "fails closed across BOTH WS6a and WS6c checkpoints; seeding WS9's\n"
        "governor from it would start WS9 at ~$70 already spent against a $55\n"
        "cap and trip on the first call. WS9 therefore gets its own accumulator\n"
        "over data/ws9_cache/*.jsonl, CUMULATIVE ACROSS WS9'S OWN PHASES (smoke,\n"
        "S, M) so that no script gets a fresh budget -- that per-script reset is\n"
        "the I-2 defect the original comment describes and it is not being\n"
        "reintroduced. The justification is that WS9 is a separately authorised\n"
        f"${WS9_BUDGET_USD:.0f} budget, not a later phase of WS6a's $75 or "
        "WS6c's $45.\n"
        "\n"
        "RUN RECORD, computed from the rows of scaling.csv at write time.\n"
        + _run_record(combined)
        + "The L rows are the reused WS6c Phase 3 rows and carry their original\n"
        "2026-09-08 timestamps; S and M are the new runs. The sweep therefore\n"
        "spans a week of wall-clock, which nothing pre-registered; the same\n"
        "pinned model, package, prompt and corpus were used throughout.\n"
        f"WS9 billed spend: ${spent:.2f} against the ${WS9_BUDGET_USD:.2f} cap "
        "(WS9_BUDGET_USD) and the\n"
        f"declared basis of ${WS9_DECLARED_BASIS_USD:.2f} for the 222 new runs "
        "(37 questions x 3 arms\n"
        "x 2 new sizes). The 12 smoke pairs share this checkpoint and were\n"
        "re-used, not re-billed. Embedding spend is OpenAI-denominated and\n"
        "therefore OUTSIDE this governor; it is recorded in\n"
        "results/ws6c/index_cost.csv as DERIVED.\n"
        "The smoke check PASSED -- both indexed arms took more than one turn at\n"
        f"both sizes, and every bound-arm search_code frame returned at most "
        f"{WS6C_TOPK}\n"
        "results -- and so drove 'continue'.\n")
    manifest.record(path)
    print(f"wrote {path}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help=f"run only the first {WS9_SMOKE_QUESTIONS} panel "
                         "questions across all arms and both sizes, and write "
                         "no CSV. Shares the sweep's checkpoint, so these rows "
                         "are re-used by the full run rather than re-billed.")
    args = ap.parse_args()

    out = ws6c_dir()
    stats = pd.read_csv(out / "scaling_corpus_stats.csv").set_index("scope")
    all_q = pd.read_csv(out / "questions.csv", dtype=str, keep_default_na=False)
    panel = ws9.ws9_eligible_qids(all_q, WS9_ELIGIBILITY_SCOPE)
    if len(panel) != WS9_PANEL_N:
        raise SystemExit(
            f"panel is {len(panel)} questions, WS9_PANEL_N pins "
            f"{WS9_PANEL_N} -- the questions file or the eligibility rule moved")
    questions = all_q[all_q["qid"].isin(panel)].reset_index(drop=True)
    print(f"panel: {len(questions)} questions eligible at "
          f"{WS9_ELIGIBILITY_SCOPE}, running at every size")

    if args.smoke:
        questions = questions.head(WS9_SMOKE_QUESTIONS)
        print(f"SMOKE: {len(questions)} question(s) x {len(WS9_ARMS)} arms "
              f"x {len(WS9_SWEEP_SIZES)} sizes")

    client = agent_harness.make_client()
    prices = ws6a.load_pricing(ws6a_dir() / "pricing.csv", regime="standard")
    judge_prompt = agent_harness.load_judge_prompt(ws6a_dir() / "judge_prompt.txt")

    frames = []
    try:
        for scope in WS9_SWEEP_SIZES:
            df = await sweep_one(client, scope, questions, prices, judge_prompt)
            df = df[df["qid"].isin(questions["qid"])].copy()
            df = _drop_spend_only(df, scope)
            ws6a.assert_no_errors(df)
            ws6a.assert_run_complete(df, list(questions["qid"]), WS9_ARMS)
            check_topk_bind(scope, list(questions["qid"]))
            df["size_point"] = scope
            frames.append(df)
    except BaseException as exc:
        # BaseException, not BudgetExceeded: run_session re-raises a tripped
        # governor still WRAPPED in the anyio task group's
        # BaseExceptionGroup, which `except ws6a.BudgetExceeded` would not
        # catch -- the STOP guidance would be skipped at exactly the moment it
        # is needed. Everything else propagates untouched.
        if not _contains(exc, ws6a.BudgetExceeded):
            raise
        print(f"\nGOVERNOR TRIPPED: {exc}")
        print("STOP. Do not drop an arm and do not raise the cap here -- both "
              "are post-hoc design changes made with data in view. Report what "
              "was collected and ask.")
        raise

    if args.smoke:
        combined = pd.concat(frames, ignore_index=True)
        print("\nSMOKE ONLY -- no CSV written.")
        print(combined.groupby(["size_point", "arm"])["turns"]
              .agg(["min", "median", "max"]).to_string())
        bad = combined[(combined.arm.str.startswith("indexed"))
                       & (combined.turns <= 1)]
        if len(bad):
            raise SystemExit(
                f"{len(bad)} indexed row(s) took 1 turn: a 1-turn indexed row "
                "answered from parametric memory is the exact WS6a failure "
                f"this check exists to catch.\n{bad[['qid','arm','size_point']]}")
        print("\nSMOKE PASSED: both indexed arms took >1 turn at both sizes, "
              f"and every {TOPK_ARM} search frame returned <= {WS6C_TOPK} "
              "results (checked from the transcripts above).")
        return

    frames.append(reuse_l_rows(panel))
    combined = pd.concat(frames, ignore_index=True)
    combined["corpus"] = WS9_SLUG
    combined["corpus_tokens"] = combined["size_point"].map(stats["count_tokens"])
    combined["corpus_files"] = combined["size_point"].map(
        stats["files_text_allowlisted"])

    # The x-axis must be monotonic in TOKENS, not in file count -- files step
    # 3.5x then 2.0x here against tokens' 1.8x then 2.1x, and a curve drawn on
    # the file axis would have a shape set by how the subsets happen to divide.
    axis = combined.groupby("size_point")["corpus_tokens"].first()
    if not (axis["S"] < axis["M"] < axis["L"]):
        raise SystemExit(f"corpus_tokens must increase S<M<L, got {axis.to_dict()}")

    want = len(panel) * len(WS9_ARMS) * 3
    if len(combined) != want:
        raise SystemExit(f"expected {want} rows, got {len(combined)}")

    path = out / "scaling.csv"
    combined.to_csv(path, index=False)
    manifest.record(path)
    write_provenance(out / "scaling_provenance.txt", panel, stats, combined)

    print(combined.groupby(["size_point", "arm"]).agg(
        n=("qid", "count"),
        median_prompt=("input_tokens", "median"),
        mean_billed=("cost_agent_billed", "mean"),
        judge_acc=("judge_correct", "mean"),
        capped=("hit_turn_cap", "sum")).to_string())
    print(f"\nwrote {len(combined)} rows to {path}")
    print(f"WS9 cumulative billed: $"
          f"{agent_harness.total_spent_across_checkpoints(DATA_DIR, globs=agent_harness.WS9_SPEND_CHECKPOINT_GLOBS):.2f}"
          f" of ${WS9_BUDGET_USD:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
