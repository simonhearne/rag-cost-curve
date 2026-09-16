#!/usr/bin/env python
"""WS4c Phase 0b -- build the generated 2-hop question set.

Design: results/recall-vs-quality.md
sections 2A and 3A (amendment 2026-09-11).

WARNING: sections 2 and 3 of that document are SUPERSEDED. C0/C1/C2 ran; C1
failed at 0.074 and the download-a-dataset approach is abandoned
(see results/recall-vs-quality.md). This script implements 2A/3A and nothing else.

Two stages:

  --pairs   $0, CPU only. Deterministic (chunk A, bridge, chunk B) triples
            from the pinned corpus, seed 42. Cached, so a failed paid run is
            diagnosable and resumable without re-selecting.
  (default) the paid build: generate, then F1..F5 cheapest-first, under the
            SEPARATE WS4C_BUILD_BUDGET_USD = 14.00 governor.

Per G-A the build governor is NEVER raised to reach 600. A build that cannot
produce 600 questions for $14 is reporting something, and the thing to do is
report it.

Run:  .venv/bin/python scripts/recall-quality/build_ws4c_questions.py --pairs
      .venv/bin/python scripts/recall-quality/build_ws4c_questions.py
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.recall_quality import provenance_runner as ws4c_runner
from benchlib.recall_quality import core as pure4
from benchlib.recall_quality import provenance as ws4c
from benchlib.config import (
    RESULTS_DIR, SEED, WS4C_CANDIDATE_TARGET, WS4C_N_QUESTIONS, ids_path,
    ws4c_cache, ws4c_dir,
)
from benchlib.download import shard_local_path
from benchlib.recall_quality.runner import load_prompts
# Cost/governor helpers are shared machinery, not code-retrieval-specific.
from benchlib.code_retrieval.bench import BudgetExceeded, cost_billed, load_pricing
from benchlib.agent_harness import append_checkpoint

SCALE = "10m"
# WS4C_CANDIDATE_TARGET (3,000) is an ESTIMATE, not a gate (spec 2A.4). The
# pool is 3x it so the $14 governor is what stops the build, never an
# artificially short pool -- a pool-exhausted stop would look like a G-A
# failure while actually being a sizing mistake.
N_POOL = 3 * WS4C_CANDIDATE_TARGET
# Not every sampled chunk names an eligible bridge title, so oversample the A
# candidates. 4x is sized from the 23% slice coverage; the real yield is
# reported by --pairs and can be re-tuned at $0.
N_SAMPLE_A = 4 * N_POOL

PAIRS = ws4c_cache() / "pairs.parquet"
PAIR_CHUNKS = ws4c_cache() / "pair_chunks.parquet"


def log(msg):
    print(msg, flush=True)


def read_texts(ids: pd.DataFrame, rows: np.ndarray) -> dict:
    """row position -> chunk text, read from the local shards in one pass.

    ROW POSITION IS THE RETRIEVAL ID: ids_path() has a RangeIndex in corpus
    row order, so .iloc position == the id Milvus returns.
    """
    import pyarrow.parquet as pq

    sub = ids.iloc[rows]
    out = {}
    for shard, grp in sub.groupby("shard"):
        texts = pq.read_table(shard_local_path(int(shard)),
                              columns=["text"]).column("text").to_pylist()
        for row, ris in zip(grp.index, grp["row_in_shard"]):
            out[int(row)] = texts[int(ris)]
    return out


def build_pool() -> pd.DataFrame:
    """Stage 1. $0. Deterministic from SEED."""
    log(f"pairs  reading {ids_path(SCALE)} ...")
    ids = pd.read_parquet(ids_path(SCALE), columns=["title", "shard", "row_in_shard"])
    assert (isinstance(ids.index, pd.RangeIndex)
            and ids.index.start == 0 and ids.index.step == 1), (
        "row position must be the retrieval id -- label == position requires "
        "start=0, step=1, not just RangeIndex-typed")
    log(f"pairs  {len(ids):,} chunks, {ids.title.nunique():,} distinct titles")

    lookup = {}
    for t in ids.title.unique():
        if ws4c.title_eligible(t):
            lookup[ws4c.norm_title(t)] = t
    log(f"pairs  {len(lookup):,} titles are eligible bridges "
        f"(multi-word, >= WS4C_MIN_BRIDGE_CHARS)")

    chunks_of = {t: np.asarray(v) for t, v in ids.groupby("title").indices.items()}

    rng = np.random.default_rng(SEED)
    a_rows = np.sort(rng.choice(len(ids), size=N_SAMPLE_A, replace=False))
    log(f"pairs  reading text for {len(a_rows):,} candidate A chunks "
        f"across {ids.iloc[a_rows].shard.nunique()} shards ...")
    text_a = read_texts(ids, a_rows)

    order = list(a_rows)
    random.Random(f"{SEED}:ws4c:order").shuffle(order)
    candidates = [(int(r), ids.iat[int(r), 0], text_a[int(r)]) for r in order]

    cap = ws4c.page_cap()
    pairs = ws4c.select_pairs(candidates, chunks_of=chunks_of, lookup=lookup,
                              n_pairs=N_POOL, seed=SEED, cap=cap)
    log(f"pairs  selected {len(pairs):,} / {N_POOL:,} requested "
        f"(cap {cap} per title, either role)")
    if len(pairs) < N_POOL:
        log(f"pairs  WARNING: pool short of target. Raise N_SAMPLE_A and "
            f"re-run -- this costs $0. Do NOT let a short pool masquerade "
            f"as a G-A failure.")

    df = pd.DataFrame(pairs)
    PAIRS.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PAIRS, index=False)
    manifest.record(PAIRS, extra={"rows": len(df), "seed": SEED,
                                  "purpose": "WS4c 2A.1 chunk-pair pool"})

    need = np.unique(np.concatenate([df.row_a.to_numpy(), df.row_b.to_numpy()]))
    log(f"pairs  reading text for {len(need):,} pair chunks ...")
    texts = read_texts(ids, need)
    chunks = pd.DataFrame({"row": need,
                           "title": [ids.iat[int(r), 0] for r in need],
                           "text": [texts[int(r)] for r in need]})
    chunks.to_parquet(PAIR_CHUNKS, index=False)
    manifest.record(PAIR_CHUNKS, extra={
        "rows": len(chunks),
        "purpose": "WS4c 2A chunk text for the pair pool -- CACHE ONLY, "
                   "never committed (corpus licence, README Licenses)"})
    log(f"pairs  wrote {PAIRS} and {PAIR_CHUNKS}")

    # pool_concentration.csv -- the pool-level control for gate G-C (spec
    # 3A, see results/recall-vs-quality.md): the concentration cap is enforced
    # HERE, on the pool, counting a title in EITHER role (page_title or
    # bridge_title), which is why it is computed from `df`/`cap` already in
    # scope rather than from the accepted question set.
    title_counts = pd.concat([df.page_title, df.bridge_title]).value_counts()
    max_title = title_counts.idxmax()
    max_title_count = int(title_counts.iloc[0])
    conc = pd.DataFrame([{
        "n_pairs": len(df), "page_cap": cap, "cap_share": cap / len(df),
        "max_title": max_title, "max_title_count": max_title_count,
        "max_title_share": max_title_count / len(df),
        "distinct_pages": df.page_title.nunique(),
        "distinct_bridges": df.bridge_title.nunique(),
        "note": ("Pool-level control for gate G-C (spec 3A): the 2% "
                 "concentration cap (WS4C_MAX_PAGE_SHARE) is enforced at "
                 "construction on the candidate pool via "
                 "benchlib.recall_quality.provenance.page_cap(), counting a title in EITHER role "
                 "(page_title or bridge_title) -- the conservative "
                 "measurement, since filters only remove pairs so a "
                 "pool-level cap bounds the accepted set too. "
                 "max_title_share == cap_share exactly when the cap is "
                 "binding and saturated, not merely satisfied by chance."),
    }])
    conc_path = ws4c_dir() / "pool_concentration.csv"
    conc.to_csv(conc_path, index=False)
    manifest.record(conc_path, extra={
        "rows": 1, "purpose": "WS4c 2A.1 pool concentration control for G-C"})
    log(f"pairs  wrote {conc_path}")
    return df


CANARY_MIN_DROP_SHARE = 0.80   # F4 must drop at least this share of
                               # deliberately single-hop canaries, or F4 is
                               # not working and the build stops

CANARY_PROMPT = """You are writing a benchmark question from ONE Wikipedia passage.

Write ONE question that is fully answerable from the passage below ALONE, and
that NAMES "{bridge_title}" explicitly. The answer must be a short span copied
verbatim from the passage. Decline only if the passage states no fact at all.

Passage (from "{bridge_title}"):
{chunk_b}
"""


def f4_verdict(ans_a: str, ans_b: str, golds) -> bool:
    """F4 passes only when NEITHER single chunk answers the question.

    Scored by ws4.exact_match, NOT an LLM judge (spec 2A.2): these are cheap
    mechanical checks over short spans, and using the eval's judge here would
    cost more and entangle question-building with the eval's scoring.
    """
    return (pure4.exact_match(ans_a, golds) == 0
            and pure4.exact_match(ans_b, golds) == 0)


def canary_reading(n: int, n_dropped: int) -> dict:
    """Is F4 actually working?

    The canaries are single-hop by construction: they name the bridge and are
    answerable from chunk B alone. F4 must drop nearly all of them. A filter
    that drops few is not a filter, and a silently-failing F4 would let real
    single-hop questions through -- S1 would then withdraw B1 anyway, after
    the eval money is spent.
    """
    share = (n_dropped / n) if n else 0.0
    return {"gate": "F4-CANARY", "n": int(n), "n_dropped": int(n_dropped),
            "drop_share": share, "min_drop_share": CANARY_MIN_DROP_SHARE,
            "pass": bool(share >= CANARY_MIN_DROP_SHARE)}


def charge(gov, usage, prices, model, row=None):
    """Record one call's billed cost, THEN ask the governor -- and if the
    governor rejects it, top `gov.spent` back up before re-raising.

    CostGovernor.charge() raises before recording. WS6a survives a rejection
    because it checkpoints the row before charging, so a resumed run recovers
    it from disk. WS4c charges before the row is written, so the call is
    already made and already billed by the time we get here, and a raise must
    not erase it.

    Two callers, two ways that money survives a rejection:
    - `row` given (the build path): the cost is written onto `row` before
      `gov.charge()` is attempted, so it survives the raise on the row
      itself; the BudgetExceeded handler in run_build checkpoints that row
      (via the loop's `finally`), so a resumed run recovers it from disk.
    - `row` omitted (the canary, run_canary's `_bill`): there is no row and
      no checkpoint for canary spend (finding 7), so the only place this
      money can survive is `gov.spent` itself -- topped up here with exactly
      this call's cost before re-raising, so the governor's own in-process
      total still reflects money that was truly spent.
    """
    usd = cost_billed(usage, prices[model])
    if row is not None:
        row["cost_total_billed"] = round(row["cost_total_billed"] + usd, 6)
    try:
        gov.charge(usd)
    except BudgetExceeded:
        gov.spent = round(gov.spent + usd, 6)
        raise
    return usd


def run_build(limit_new=None):
    """Stage 2. Paid, resumable, cheapest-first (spec 2A.2)."""
    pairs = pd.read_parquet(PAIRS)
    chunks = pd.read_parquet(PAIR_CHUNKS).set_index("row")
    text_of = chunks.text.to_dict()

    answer_prompt, _judge_prompt = load_prompts()
    qgen_prompt = (ws4c_dir() / "qgen_prompt.txt").read_text()
    qvalid_prompt = (ws4c_dir() / "qvalidity_prompt.txt").read_text()

    prices = load_pricing(RESULTS_DIR / "ws6a" / "pricing.csv", regime="standard")
    gov = ws4c_runner.build_governor()
    client = ws4c_runner.make_client()
    log(f"build  governor ${gov.spent:.4f} spent of ${gov.limit:.2f} "
        f"(WS4C_BUILD_BUDGET_USD; per G-A this is never raised)")

    done = {}
    if ws4c_runner.BUILD_CHECKPOINT.exists():
        for line in ws4c_runner.BUILD_CHECKPOINT.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    done[r["qid"]] = r
                except json.JSONDecodeError:
                    pass
    log(f"build  {len(done):,} candidates already recorded")

    counts = {k: 0 for k in
              ("seen", "decline", "f1", "f2", "f3", "f4", "f5", "accepted")}
    for r in done.values():
        counts["seen"] += 1
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    accepted = counts["accepted"]
    n_new = 0

    for _, p in pairs.iterrows():
        if accepted >= WS4C_N_QUESTIONS:
            log(f"build  reached WS4C_N_QUESTIONS = {WS4C_N_QUESTIONS}")
            break
        if limit_new is not None and n_new >= limit_new:
            break
        cid = f"{int(p.row_a)}_{int(p.row_b)}"
        if cid in done:
            continue
        text_a, text_b = text_of[int(p.row_a)], text_of[int(p.row_b)]
        row = {"qid": cid, "arm": "build", "row_a": int(p.row_a),
               "row_b": int(p.row_b), "page_title": p.page_title,
               "bridge_title": p.bridge_title, "question": "", "gold_answer": "",
               "outcome": "", "cost_total_billed": 0.0, "note": ""}
        try:
            g = ws4c_runner.generate_question(
                client, qgen_prompt, text_a=text_a, text_b=text_b,
                page_title=p.page_title, bridge_title=p.bridge_title)
            charge(gov, g["usage"], prices, ws4c_runner.WS4C_QGEN_MODEL, row)
            if g["decline"] or not g["question"] or not g["gold_answer"]:
                row.update(outcome="decline", note=g.get("reason", "")[:200])
            else:
                row.update(question=g["question"], gold_answer=g["gold_answer"])
                golds = [g["gold_answer"]]
                if not ws4c.f1_gold_in_b(g["gold_answer"], text_b):
                    row["outcome"] = "f1"
                elif not ws4c.f2_bridge_absent(p.bridge_title, g["question"]):
                    row["outcome"] = "f2"
                else:
                    a3 = ws4c_runner.answer_with(client, answer_prompt, [],
                                                 g["question"])
                    charge(gov, a3["usage"], prices, ws4c_runner.WS4C_QGEN_MODEL, row)
                    if pure4.exact_match(a3["answer"], golds) == 1:
                        row.update(outcome="f3", note="answered unaided")
                    else:
                        pa = [{"title": p.page_title, "text": text_a}]
                        pb = [{"title": p.bridge_title, "text": text_b}]
                        aa = ws4c_runner.answer_with(client, answer_prompt, pa,
                                                     g["question"])
                        charge(gov, aa["usage"], prices, ws4c_runner.WS4C_QGEN_MODEL, row)
                        ab = ws4c_runner.answer_with(client, answer_prompt, pb,
                                                     g["question"])
                        charge(gov, ab["usage"], prices, ws4c_runner.WS4C_QGEN_MODEL, row)
                        if not f4_verdict(aa["answer"], ab["answer"], golds):
                            # results/recall-vs-quality.md: which arm (A-alone or
                            # B-alone) fired is unmeasured for every row
                            # already committed -- this is the one-line fix
                            # for FUTURE runs only; nothing is re-run to
                            # backfill it.
                            arm = "A" if pure4.exact_match(aa["answer"], golds) == 1 else "B"
                            row.update(outcome="f4", note=f"single chunk sufficed (arm={arm})")
                        else:
                            try:
                                v = ws4c_runner.judge_validity(
                                    client, qvalid_prompt, question=g["question"],
                                    gold_answer=g["gold_answer"], text_a=text_a,
                                    text_b=text_b)
                                charge(gov, v["usage"], prices,
                                       ws4c_runner.WS4C_QJUDGE_MODEL, row)
                                if not v["valid"]:
                                    row.update(outcome="f5", note=v["reason"][:200])
                                else:
                                    row["outcome"] = "accepted"
                            except ws4c_runner.QValidityError as qerr:
                                # judge_validity's escalation schedule is
                                # exhausted. Its two paid attempts (~$0.52)
                                # are real spend that already happened, so
                                # they are charged to the governor BEFORE the
                                # candidate is recorded -- otherwise this
                                # becomes spend the governor never sees
                                # (interface note). Continue to
                                # the next candidate rather than aborting.
                                charge(gov, qerr.usage, prices,
                                       ws4c_runner.WS4C_QJUDGE_MODEL, row)
                                row.update(outcome="f5_error", note=str(qerr)[:200])
        except BudgetExceeded as exc:
            log(f"build  STOP: {exc}")
            row["outcome"] = row["outcome"] or "aborted"
            done[cid] = row
            counts["seen"] += 1
            counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
            break
        finally:
            # Runs whether the candidate completed normally, was aborted by
            # BudgetExceeded above, or died to any OTHER exception mid-loop
            # (this candidate loop is the only exception handler in the
            # build, and a run here HAS been killed mid-flight by host
            # memory pressure -- an uncaught exception must not lose billed
            # spend the way a bare post-try append would). Guarded on
            # cost_total_billed so the common case (every branch above bills
            # the qgen call first) is unchanged in practice.
            if row["cost_total_billed"] > 0:
                append_checkpoint(ws4c_runner.BUILD_CHECKPOINT, row)

        done[cid] = row
        counts["seen"] += 1
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
        accepted = counts["accepted"]
        n_new += 1
        if n_new % 25 == 0:
            log(f"build  {n_new:,} new | accepted {accepted:,}/"
                f"{WS4C_N_QUESTIONS} | ${gov.spent:.3f}/${gov.limit:.2f}")

    return done, counts, gov, prices, client, text_of, answer_prompt


def run_canary(client, prices, gov, text_of, answer_prompt, pairs, n):
    """Is F4 working? Ask for questions that are single-hop BY CONSTRUCTION
    (they name the bridge and are answerable from chunk B alone) and check
    that F4 drops them. Charged to the same governor, so it cannot overrun.

    Each candidate's billed cost is also appended to BUILD_CHECKPOINT as a
    "canary" row (finding 7): previously canary spend was reflected in
    gov.spent for THIS process only and nothing was written to disk, so
    build_spent() on a later run silently forgot it and the governor would
    let it be spent again. One checkpoint line per candidate, keyed
    'canary_<n>', mirrors the build loop's own checkpoint-per-candidate
    shape; build_spent() sums every raw line regardless of qid, so nothing
    else needs to change for this to be seen on a resumed run.
    """
    rows = []

    for i, (_, p) in enumerate(pairs.head(n).iterrows()):
        text_a, text_b = text_of[int(p.row_a)], text_of[int(p.row_b)]
        crow = {"qid": f"canary_{i}", "arm": "canary", "outcome": "canary",
                "cost_total_billed": 0.0}
        try:
            g = ws4c_runner.generate_question(
                client, CANARY_PROMPT, text_a="", text_b=text_b,
                page_title=p.page_title, bridge_title=p.bridge_title)
            charge(gov, g["usage"], prices, ws4c_runner.WS4C_QGEN_MODEL, crow)
            if g["decline"] or not g["gold_answer"]:
                continue
            golds = [g["gold_answer"]]
            pa = [{"title": p.page_title, "text": text_a}]
            pb = [{"title": p.bridge_title, "text": text_b}]
            aa = ws4c_runner.answer_with(client, answer_prompt, pa, g["question"])
            charge(gov, aa["usage"], prices, ws4c_runner.WS4C_QGEN_MODEL, crow)
            ab = ws4c_runner.answer_with(client, answer_prompt, pb, g["question"])
            charge(gov, ab["usage"], prices, ws4c_runner.WS4C_QGEN_MODEL, crow)
            rows.append({"row_b": int(p.row_b), "question": g["question"],
                         "gold_answer": g["gold_answer"],
                         "dropped_by_f4": not f4_verdict(aa["answer"], ab["answer"],
                                                         golds)})
        except BudgetExceeded as exc:
            log(f"canary STOP: {exc}")
            break
        finally:
            # NOTE: this checkpoints only spend from THIS run onward. The
            # ~$0.095950 already spent by the committed f4_canary.csv run
            # predates this fix and is deliberately NOT retro-appended here
            # (see results/recall-vs-quality.md) -- see finding 7.
            if crow["cost_total_billed"] > 0:
                append_checkpoint(ws4c_runner.BUILD_CHECKPOINT, crow)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", action="store_true",
                    help="stage 1 only: build the $0 candidate pool")
    ap.add_argument("--limit-new", type=int, default=None,
                    help="stop after N newly generated candidates (smoke run)")
    ap.add_argument("--canary", type=int, default=15,
                    help="F4 canaries; 0 to skip (NOT recommended)")
    args = ap.parse_args()

    if args.pairs:
        build_pool()
        return
    if not PAIRS.exists():
        raise SystemExit("run --pairs first (it is free)")

    out = ws4c_dir()
    out.mkdir(parents=True, exist_ok=True)
    done, counts, gov, prices, client, text_of, answer_prompt = run_build(
        limit_new=args.limit_new)

    accepted = [r for r in done.values() if r["outcome"] == "accepted"]
    accepted.sort(key=lambda r: (r["row_a"], r["row_b"]))
    qs = [{"qid": f"ws4c_{i:04d}", "question": r["question"],
           "gold_answer": r["gold_answer"], "row_a": r["row_a"],
           "row_b": r["row_b"], "page_title": r["page_title"],
           "bridge_title": r["bridge_title"]}
          for i, r in enumerate(accepted[:WS4C_N_QUESTIONS], start=1)]
    qdf = pd.DataFrame(qs)
    # LICENSING: questions, gold answers and chunk ROW IDS only. Chunk text is
    # NEVER committed -- the embeddings dataset carries no licence tag.
    assert "text" not in qdf.columns
    qdf.to_csv(out / "questions.csv", index=False)
    manifest.record(out / "questions.csv", extra={
        "rows": len(qdf), "purpose": "WS4c 2A frozen question set",
        "note": "LLM generation is not reproducible; THIS FILE is the "
                "committed artifact (spec 2A.3)"})

    # f5_error and aborted are not in the brief's original order list, but
    # both are real outcomes now that judge_validity can raise
    # QValidityError (interface note) and a BudgetExceeded can
    # land mid-candidate -- hiding them from the yield table would make a
    # nonzero count of either invisible to the skeptical read.
    order = ["seen", "decline", "f1", "f2", "f3", "f4", "f5", "f5_error",
             "aborted", "accepted"]
    yield_rows = [{"stage": k, "n": counts.get(k, 0),
                   "share_of_seen": counts.get(k, 0) / max(counts["seen"], 1)}
                  for k in order]
    yield_rows.append({"stage": "spend_usd", "n": round(gov.spent, 6),
                       "share_of_seen": gov.spent / gov.limit})
    pd.DataFrame(yield_rows).to_csv(out / "build_yield.csv", index=False)
    # This repo's hard rule: every produced file under results/ is sha256'd into
    # the manifest. The brief's original main() recorded questions.csv and
    # build_gates.csv only; build_yield.csv and f4_canary.csv are added here
    # for the same reason, not a change to any filter/gate/generation logic.
    manifest.record(out / "build_yield.csv")

    canary_rows = (run_canary(client, prices, gov, text_of, answer_prompt,
                              pd.read_parquet(PAIRS), args.canary)
                   if args.canary else [])
    if canary_rows:
        pd.DataFrame(canary_rows).to_csv(out / "f4_canary.csv", index=False)
        manifest.record(out / "f4_canary.csv")
    gates = [ws4c.gate_a(len(qdf), gov.spent),
             ws4c.gate_b(qs, text_of),
             ws4c.gate_c(qs)]
    if canary_rows:
        # Only a canary that actually RAN is a gate. An empty list (--canary 0,
        # or a governor stop mid-canary) must not read as a 0.0 drop share and
        # fail the build that just succeeded.
        gates.append(canary_reading(
            len(canary_rows), sum(1 for r in canary_rows if r["dropped_by_f4"])))
    elif args.canary:
        log("  WARNING: canary requested but none ran -- F4 is UNVERIFIED.")
    pd.DataFrame(gates).to_csv(out / "build_gates.csv", index=False)
    manifest.record(out / "build_gates.csv")

    log("")
    for g in gates:
        log(f"  {g['gate']:<10} {'PASS' if g['pass'] else 'FAIL'}  {g}")
    log(f"\nbuild  accepted {len(qdf):,} questions for ${gov.spent:.4f} "
        f"of ${gov.limit:.2f}")
    if not all(g["pass"] for g in gates):
        log("\n  *** A GATE FAILED. Report the per-filter yield and STOP. ***")
        log("  *** Per G-A, do NOT raise WS4C_BUILD_BUDGET_USD to reach 600. ***")
        raise SystemExit(1)
    log("\n  Sanity, before trusting this: is the F3 drop share plausibly")
    log("  LARGE (Wikipedia is heavily memorised) and the F4 drop share")
    log("  substantial? A filter that drops almost nothing is a filter that")
    log("  is not working, not a set of excellent questions.")


if __name__ == "__main__":
    main()
