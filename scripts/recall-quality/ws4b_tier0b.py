#!/usr/bin/env python
"""WS4b Tier 0b: provenance calibration of the string presence metric. $0.

Design: results/recall-vs-quality.mdA.1 feasibility (already measured)   3A.3 gates K0 / K1 / K2
  3A.2 what is computed                 3A.4 branches P1-P3 and A-VAL V1/V2

WHY THIS EXISTS. results/recall-vs-quality.md withdrew WS4's "declines 29 %, errs 13 %"
decomposition on the strength of a blind decline audit that found 55 % of
declines were `not_supported` -- a gold string sitting in a passage about
something else. That audit rests on 93 rows and ONE auditor's judgement, and
the auditor was the same model that proposed the hypothesis. Tier 0b checks it
against an external signal that involved no judgement of ours: KILT-NQ
provenance annotations.

WHAT IT IS NOT. KILT provenance is STRICTER than answerability (KILT annotates
one page; NQ answers appear on others it never annotated) and string presence
is LOOSER (a gold string in an unrelated passage counts). They bracket the
truth from opposite sides. D+ is an UPPER BOUND on the string metric's
false-positive rate, not a measurement of it, and is never reported as one.

SCOPE. n = 124 of the 450 main questions (27.6 %), and they are not a random
sample: they are questions present in KILT dev whose gold page also landed in
the pinned 10M corpus prefix. Tier 0b CALIBRATES THE METRIC. Any
recall-vs-quality curve it emits is a labelled sensitivity with wide CIs, not
a headline.

No API calls, no Milvus, no regeneration. Reads results/ws4/runs.csv,
data/ws4_cache/passages.parquet, results/ws4b/{kilt_usable_questions,
tier0_runs_corrected,decline_labelled}.csv and KILT-NQ via huggingface_hub.
results/ws4/ is NEVER written to.

Run:  python scripts/recall-quality/ws4b_tier0b.py
"""

import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib.recall_quality import gold as ws4b
from benchlib.recall_quality import core
from benchlib.config import (
    DATA_DIR,
    SEED,
    WS4_N_BOOTSTRAP,
    WS4_TOP_K,
    WS4B_AVAL_AGREEMENT,
    WS4B_KILT_NQ_DEV,
    WS4B_KILT_REPO,
    WS4B_P1_DISCORD,
    WS4B_P3_DISCORD,
    ids_path,
    ws4_cache,
    ws4_dir,
    ws4b_dir,
)

REF = "sq8_np512"
PARAMETRIC = "parametric"

# Not from the spec: the run instruction for Tier 0b requires A-VAL to be
# declared UNDERPOWERED rather than resolved if the overlap between the 124
# provenance-usable questions and the 93 audited declines is under ~20.
AVAL_MIN_N = 20

# The 93 audited declines are the reference arm's declines inside the
# corrected primary stratum (results/recall-vs-quality.md: R = 36/93, denominator 325).
AVAL_ARM = REF


# ------------------------------------------------------------- loading ----


def norm_q(q: str) -> str:
    """Byte-for-byte the match key of the (now-removed) WS4b KILT
    feasibility probe -- see git history for scripts/ws4b_kilt_feasibility.py.
    K0 is what proves it has not drifted."""
    q = unicodedata.normalize("NFKD", str(q).lower())
    return re.sub(r"[^a-z0-9]+", " ", q).strip()


def provenance_titles(output) -> set:
    t = set()
    for o in list(output):
        pr = o.get("provenance")
        if pr is None:
            continue
        for p in list(pr):
            if p.get("title"):
                t.add(str(p["title"]))
    return t


def load_runs() -> pd.DataFrame:
    r = pd.read_csv(ws4_dir() / "runs.csv")
    r["ids"] = r.retrieved_ids.map(
        lambda s: [int(i) for i in ws4b.parse_golds(s)] if str(s).strip() != "[]" else []
    )
    return r


def load_passage_table() -> pd.DataFrame:
    p = ws4_cache() / "passages.parquet"
    if not p.exists():
        sys.exit(f"missing passage cache {p} -- WS4's cache is required; this "
                 "script does not re-read corpus shards")
    return pd.read_parquet(p).set_index("row")


def load_corpus_titles():
    """The pinned 10M id table. Returns (arrow title column, set of titles).
    Row position IS the retrieval id -- the same index space as runs.csv's
    retrieved_ids and the WS1 exact ground truth."""
    col = pq.read_table(ids_path("10m"), columns=["title"]).column("title").combine_chunks()
    return col, set(pc.unique(col).to_pylist())


def texts_for(ids, table) -> list[str]:
    """Identical to scripts/recall-quality/ws4b_tier0.py: TEXT only, top-10 only, cached rows
    only. K2 is what proves this reproduces Tier 0."""
    return [table.at[i, "text"] for i in ids[:WS4_TOP_K] if i >= 0 and i in table.index]


def titles_for(ids, table) -> set:
    """Titles of the chunks that arm actually retrieved into the top-10."""
    return {table.at[i, "title"] for i in ids[:WS4_TOP_K]
            if i >= 0 and i in table.index}


def gold_page_texts(ids, table, gold_titles) -> list[str]:
    """Texts of only those retrieved top-10 chunks that come FROM a KILT gold
    page. Used by the post-hoc granularity diagnostic below."""
    return [table.at[i, "text"] for i in ids[:WS4_TOP_K]
            if i >= 0 and i in table.index and table.at[i, "title"] in gold_titles]


# --------------------------------------------------------------- gates ----


def gate_k0(runs, corpus_titles) -> tuple[pd.DataFrame, dict]:
    """K0: recomputing the 124 from KILT + NQ + runs.csv must reproduce
    results/ws4b/kilt_usable_questions.csv EXACTLY -- both the qid set and the
    provenance title strings. Fail => the match key drifted and the 124 are
    not the pre-registered 124."""
    kilt = pd.read_parquet(
        hf_hub_download(WS4B_KILT_REPO, WS4B_KILT_NQ_DEV, repo_type="dataset")
    )
    kilt["qn"] = kilt.input.map(norm_q)
    kilt["pt"] = kilt.output.map(provenance_titles)
    kmap = dict(zip(kilt.qn, kilt.pt))

    main_q = (runs[(runs.subset == "main") & (runs.arm == REF)][["qid", "question"]]
              .copy())
    main_q["qn"] = main_q.question.map(norm_q)
    m = main_q[main_q.qn.isin(kmap)].copy()
    m["pt"] = m.qn.map(kmap)
    m["in_slice"] = m.pt.map(lambda t: len(t & corpus_titles) > 0)

    got = m[m.in_slice][["qid"]].copy()
    got["provenance_titles"] = m[m.in_slice].pt.map(lambda t: "|".join(sorted(t)))
    got = got.sort_values("qid").reset_index(drop=True)

    want = pd.read_csv(ws4b_dir() / "kilt_usable_questions.csv").sort_values("qid")
    want = want.reset_index(drop=True)

    same_ids = list(got.qid) == list(want.qid)
    same_titles = same_ids and list(got.provenance_titles) == list(want.provenance_titles)
    print(f"K0  124 reproduced from KILT: n={len(got)} (expected {len(want)})"
          f"  qids_match={same_ids}  titles_match={same_titles}")
    if not (same_ids and same_titles and len(got) == 124):
        sys.exit("GATE K0 FAILURE -- the match key drifted; these are not the "
                 "pre-registered 124 questions")
    return got, {"gate": "K0", "check": "124 qids+titles reproduce "
                 "kilt_usable_questions.csv", "value": len(got), "passed": True}


def gate_k1(usable, corpus_titles, table, title_col) -> tuple[dict, dict]:
    """K1: every KILT gold title USED for the provenance test must be present
    in corpus_10m_ids.parquet, and every one of the 124 must retain at least
    one such title -- otherwise the failure is the title join, not the metric.

    Titles annotated by KILT that fall OUTSIDE the pinned prefix are counted
    and reported, not failed: §3A.1 selected these questions on "at least one
    gold page in the slice", so some annotated pages are legitimately absent.
    They are excluded from the test; a question is provenance-positive only
    for a page the corpus could actually have returned.

    Also checks the two title sources agree: data/ws4_cache/passages.parquet
    (used to read retrieved titles) against corpus_10m_ids.parquet (used to
    build the gold title set). A mismatch would make every comparison vacuous.
    """
    used, dropped = {}, 0
    for qid, titles in usable.items():
        keep = titles & corpus_titles
        dropped += len(titles) - len(keep)
        used[qid] = keep
    empty = [q for q, t in used.items() if not t]

    rows = np.asarray(table.index, dtype=np.int64)
    ids_titles = title_col.take(pa.array(rows)).to_pylist()
    cache_titles = list(table.title)
    title_mismatch = int(sum(a != b for a, b in zip(ids_titles, cache_titles)))

    n_used = sum(len(t) for t in used.values())
    print(f"K1  gold titles inside the 10M prefix: {n_used} used, {dropped} "
          f"annotated-but-outside (excluded)")
    print(f"    questions left with no usable gold title: {len(empty)}")
    print(f"    passages.parquet title == corpus ids title on {len(rows)} cached "
          f"rows: mismatches={title_mismatch}")
    if empty or title_mismatch:
        sys.exit("GATE K1 FAILURE -- the title join is wrong, not the metric")
    return used, {"gate": "K1", "check": "gold titles in corpus ids; cache/ids "
                  "titles agree", "value": n_used, "passed": True}


def gate_k2(rows_df) -> dict:
    """K2: presence_wb recomputed here must equal tier0_runs_corrected.csv on
    all 124 x 11 rows. This is what makes Tier 0b comparable to Tier 0 at all
    -- without it, D+ and D- are computed against a different metric than the
    one WS4b §6 is stated in."""
    bad = int((rows_df.presence_wb != rows_df.presence_wb_tier0).sum())
    n = len(rows_df)
    print(f"K2  presence_wb == tier0_runs_corrected.csv: {n - bad}/{n}  "
          f"mismatches={bad}")
    if bad or n != 124 * 11:
        sys.exit(f"GATE K2 FAILURE -- Tier 0b is not comparable to Tier 0 "
                 f"(rows={n}, expected {124 * 11})")
    return {"gate": "K2", "check": "presence_wb reproduces Tier 0 on 124x11 rows",
            "value": n, "passed": True}


# ------------------------------------------------------- the two metrics ----


def build_rows(runs, table, used, tier0) -> pd.DataFrame:
    """124 questions x 11 retrieval arms, from frozen data only (spec 3A.2).

      presence_wb     the Tier 0 corrected string metric, recomputed (K2)
      provenance@10   is any retrieved top-10 chunk's title a KILT gold page

    POST-HOC (added after the pre-registered statistics were computed, and
    labelled as such wherever it appears): `presence_wb_on_gold_page` applies
    the SAME word-boundary rule to only those retrieved chunks that come from
    a KILT gold page. It exists because provenance is annotated per PAGE while
    presence is tested per CHUNK, and the 10M corpus averages 6.55 chunks per
    page -- so "the gold page was retrieved" is a much weaker statement than
    "the chunk that states the answer was retrieved". It resolves no branch.
    """
    qids = set(used)
    sub = runs[(runs.subset == "main") & runs.qid.isin(qids)
               & (runs.arm != PARAMETRIC)].copy()
    out = []
    for _, x in sub.iterrows():
        gold_titles = used[int(x.qid)]
        got = titles_for(x.ids, table)
        out.append({
            "qid": int(x.qid),
            "arm": x.arm,
            "recall_at_10": float(x.recall_at_10),
            "presence_wb": bool(ws4b.answer_present_wb(x.golds, texts_for(x.ids, table))),
            "provenance_at_10": bool(got & gold_titles),
            "presence_wb_on_gold_page": bool(ws4b.answer_present_wb(
                x.golds, gold_page_texts(x.ids, table, gold_titles))),
            "n_gold_titles": len(gold_titles),
            "n_gold_titles_hit": len(got & gold_titles),
            "judge_correct": bool(x.judge_correct),
        })
    df = pd.DataFrame(out)
    t0 = tier0[["qid", "arm", "presence_wb", "idk_corrected", "in_primary"]].rename(
        columns={"presence_wb": "presence_wb_tier0"})
    return df.merge(t0, on=["qid", "arm"], how="left").sort_values(["qid", "arm"])


def discord(df) -> dict:
    """spec 3A.4. D+ = presence_wb true, provenance false. D- = the reverse.

    ⚠️ SPEC AMBIGUITY, recorded rather than resolved silently. §3A.4 defines
    both as "share of ROWS with X and Y", which reads as a JOINT share over
    all rows -- and the symmetric phrasing of D+ and D- only makes them a
    decomposition of total disagreement under that reading. But the same
    sentence calls D+ "an upper bound on the string metric's FALSE-POSITIVE
    RATE", and a false-positive rate is conditioned on the metric firing.
    The two readings differ by a factor of P(presence_wb), which is large.

    The literal reading is taken as primary because it is what the
    pre-registered text says; the conditional is computed alongside and BOTH
    are reported. If they straddle a P-branch boundary the script says so
    loudly and refuses to pick the flattering one.
    """
    n = len(df)
    fp = int((df.presence_wb & ~df.provenance_at_10).sum())
    fn = int((~df.presence_wb & df.provenance_at_10).sum())
    n_pres = int(df.presence_wb.sum())
    n_abs = int((~df.presence_wb).sum())
    return {
        "n_rows": n,
        "n_presence_wb": n_pres,
        "n_provenance": int(df.provenance_at_10.sum()),
        "d_plus_joint": fp / n,
        "d_minus_joint": fn / n,
        "d_plus_conditional": fp / n_pres if n_pres else float("nan"),
        "d_minus_conditional": fn / n_abs if n_abs else float("nan"),
        "agreement": float((df.presence_wb == df.provenance_at_10).mean()),
        "n_fp": fp,
        "n_fn": fn,
    }


def granularity(df) -> pd.DataFrame:
    """POST-HOC (spec 3A has no such statistic). Tests the spec's own claim
    that D- is "an unambiguous false negative -- the gold page WAS retrieved
    and the string metric missed it".

    It is not unambiguous. Provenance is annotated per PAGE; the corpus is
    chunked at 6.55 chunks per page on average, so retrieving one chunk of the
    gold page is NOT retrieving the chunk that states the answer. On every D-
    row the gold string is absent from the retrieved gold-page chunk -- which
    is true by construction, and that is exactly the point: the string metric
    is not obviously wrong on those rows, the gold page simply came back at a
    chunk that does not state the answer.

    The informative split is on the AGREEING rows. Where presence and
    provenance both fire, the string may have matched on the gold page (real
    agreement) or somewhere else entirely while the gold page happened to be
    retrieved too (coincidental agreement). Reported; resolves no branch."""
    cells = []
    for pres in (True, False):
        for prov in (True, False):
            g = df[(df.presence_wb == pres) & (df.provenance_at_10 == prov)]
            cells.append({
                "presence_wb": pres,
                "provenance_at_10": prov,
                "cell": {(True, True): "agree (both true)",
                         (True, False): "D+ (presence only)",
                         (False, True): "D- (provenance only)",
                         (False, False): "agree (both false)"}[(pres, prov)],
                "n": len(g),
                "share_of_rows": round(len(g) / len(df), 6),
                "gold_string_on_gold_page": int(g.presence_wb_on_gold_page.sum()),
            })
    out = pd.DataFrame(cells)
    agree = df[df.presence_wb & df.provenance_at_10]
    coincidental = int((~agree.presence_wb_on_gold_page).sum())
    fp = int((df.presence_wb & ~df.provenance_at_10).sum())
    n_pres = int(df.presence_wb.sum())
    out.attrs["coincidental"] = coincidental
    out.attrs["off_page_matches"] = fp + coincidental
    out.attrs["off_page_share"] = (fp + coincidental) / n_pres if n_pres else float("nan")
    return out


def p_branch(d_plus: float) -> str:
    if d_plus >= WS4B_P1_DISCORD:
        return "P1"
    if d_plus >= WS4B_P3_DISCORD:
        return "P2"
    return "P3"


P_CONSEQUENCE = {
    "P1": "string presence is unusable as a mediator; WS4 §6's decomposition "
          "is withdrawn in full rather than restated, and no future workstream "
          "scores presence by string match",
    "P2": "presence over-counts materially; results/recall-vs-quality.md "
          "reports it as a BRACKET (provenance low, "
          "string high) with no point estimate",
    "P3": "the decline audit's 55 % is specific to declines and does not "
          "generalise; string presence is roughly sound elsewhere and §6's "
          "withdrawal is narrowed to the decline bucket",
}


# --------------------------------------------------------------- A-VAL ----


def aval(rows_df) -> tuple[pd.DataFrame, dict]:
    """spec 3A.4. On the overlap between the 124 and the 93 audited declines,
    `not_supported` should coincide with provenance@10 = false.

    The audited declines are reference-arm rows, so provenance is read on the
    reference arm. Agreement is computed over all overlap rows (the literal
    reading) and, separately, over the two labels the rubric actually contrasts
    -- `not_supported` vs `answerable_declined` -- since `ambiguous_question`
    and `stale_gold` carry no clear provenance expectation.
    """
    lab = pd.read_csv(ws4b_dir() / "decline_labelled.csv")
    ref = rows_df[rows_df.arm == AVAL_ARM]
    ov = lab.merge(ref[["qid", "provenance_at_10", "presence_wb",
                        "presence_wb_on_gold_page"]], on="qid", how="inner")
    ov["expect_provenance"] = ov.label != "not_supported"
    ov["agrees"] = ov.expect_provenance == ov.provenance_at_10
    core = ov[ov.label.isin(["not_supported", "answerable_declined"])]

    stats = {
        "n_overlap": len(ov),
        "agreement_all": float(ov.agrees.mean()) if len(ov) else float("nan"),
        "n_core": len(core),
        "agreement_core": float(core.agrees.mean()) if len(core) else float("nan"),
        "n_not_supported": int((ov.label == "not_supported").sum()),
        "not_supported_prov_false": int(((ov.label == "not_supported")
                                         & ~ov.provenance_at_10).sum()),
        "n_answerable": int((ov.label == "answerable_declined").sum()),
        "answerable_prov_true": int(((ov.label == "answerable_declined")
                                     & ov.provenance_at_10).sum()),
        "underpowered": len(ov) < AVAL_MIN_N,
        # POST-HOC, sharper operationalisation of what the auditor actually
        # claimed: `not_supported` means the gold string matched in a passage
        # ABOUT SOMETHING ELSE -- i.e. not on the gold page. Reported
        # separately, never substituted for the pre-registered statistic.
        "n_not_supported_str_off_gold_page": int(
            ((ov.label == "not_supported") & ~ov.presence_wb_on_gold_page).sum()),
        "agreement_posthoc_str": float(
            ((ov.label != "not_supported") == ov.presence_wb_on_gold_page).mean())
        if len(ov) else float("nan"),
    }
    cols = ["qid", "question", "label", "provenance_at_10", "presence_wb",
            "presence_wb_on_gold_page", "expect_provenance", "agrees",
            "justification"]
    return ov[cols].sort_values("qid").reset_index(drop=True), stats


# -------------------------------------------------- sensitivity curve ----


def sensitivity_curve(rows_df) -> pd.DataFrame:
    """⚠️ SENSITIVITY ONLY, n = 124 and not a random sample of the 450.
    Emitted so the bracket can be seen per arm; NOT a restatement of WS4's
    recall-vs-quality curve and never to be quoted as one. Same bootstrap
    protocol and seed as WS4; CIs are ~1.6x wider than the 325-question
    primary stratum by construction."""
    q_pres = rows_df.pivot(index="qid", columns="arm", values="presence_wb").astype(float)
    q_prov = rows_df.pivot(index="qid", columns="arm", values="provenance_at_10").astype(float)
    q_corr = rows_df.pivot(index="qid", columns="arm", values="judge_correct").astype(float)
    rec = rows_df.pivot(index="qid", columns="arm", values="recall_at_10").astype(float)
    idx = core.bootstrap_indices(len(q_pres), WS4_N_BOOTSTRAP, SEED)

    out = []
    for arm in q_pres.columns:
        row = {"arm": arm, "n_questions": len(q_pres),
               "recall_at_10": float(rec[arm].mean())}
        for name, q in (("presence_wb", q_pres), ("provenance_at_10", q_prov),
                        ("judge_correct", q_corr)):
            lo, hi = core.percentile_ci(core.boot_means(q[arm], idx), 0.05)
            row[name] = float(q[arm].mean())
            row[f"{name}_ci_lo"] = lo
            row[f"{name}_ci_hi"] = hi
        out.append(row)
    return (pd.DataFrame(out).sort_values("recall_at_10", ascending=False)
            .round(6).reset_index(drop=True))


# ---------------------------------------------------------------- main ----


def main() -> None:
    out = ws4b_dir()
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs()
    table = load_passage_table()
    title_col, corpus_titles = load_corpus_titles()
    tier0 = pd.read_csv(out / "tier0_runs_corrected.csv")
    print(f"loaded {len(runs)} run rows, {len(table)} cached passages, "
          f"{len(corpus_titles)} distinct corpus titles\n")

    print("--- gates (spec 3A.3) ---")
    k0_df, g_k0 = gate_k0(runs, corpus_titles)
    usable = {int(r.qid): set(str(r.provenance_titles).split("|"))
              for r in k0_df.itertuples()}
    used, g_k1 = gate_k1(usable, corpus_titles, table, title_col)

    rows_df = build_rows(runs, table, used, tier0)
    g_k2 = gate_k2(rows_df)
    pd.DataFrame([g_k0, g_k1, g_k2]).to_csv(out / "tier0b_gates.csv", index=False)

    keep = ["qid", "arm", "recall_at_10", "presence_wb", "provenance_at_10",
            "presence_wb_on_gold_page", "n_gold_titles", "n_gold_titles_hit",
            "judge_correct", "idk_corrected", "in_primary"]
    rows_df[keep].to_csv(out / "tier0b_rows.csv", index=False)

    # ---- spec 3A.4 P-branches
    pooled = discord(rows_df)
    per_arm = []
    for arm, g in rows_df.groupby("arm"):
        d = discord(g)
        d["arm"] = arm
        per_arm.append(d)
    agree = pd.DataFrame([{**pooled, "arm": "POOLED"}] + per_arm)
    agree = agree[["arm"] + [c for c in agree.columns if c != "arm"]].round(6)
    agree.to_csv(out / "tier0b_agreement.csv", index=False)

    b_lit = p_branch(pooled["d_plus_joint"])
    b_cond = p_branch(pooled["d_plus_conditional"])
    print("\n--- spec 3A.4 D+ / D- (pooled over 124 x 11 rows) ---")
    print(f"presence_wb true on {pooled['n_presence_wb']}/{pooled['n_rows']} rows; "
          f"provenance@10 true on {pooled['n_provenance']}")
    print(f"D+ joint       = {pooled['n_fp']}/{pooled['n_rows']} = "
          f"{pooled['d_plus_joint']:.4f}  -> {b_lit}   (pre-registered reading)")
    print(f"D+ conditional = {pooled['n_fp']}/{pooled['n_presence_wb']} = "
          f"{pooled['d_plus_conditional']:.4f}  -> {b_cond}   (\"false-positive rate\")")
    print(f"D- joint       = {pooled['n_fn']}/{pooled['n_rows']} = "
          f"{pooled['d_minus_joint']:.4f}")
    print(f"D- conditional = {pooled['n_fn']}/{pooled['n_rows'] - pooled['n_presence_wb']} = "
          f"{pooled['d_minus_conditional']:.4f}")
    if b_lit != b_cond:
        print(f"\n⚠️  THE TWO READINGS OF D+ RESOLVE TO DIFFERENT BRANCHES "
              f"({b_lit} vs {b_cond}). Both are reported; neither is suppressed.")

    gran = granularity(rows_df)
    gran.to_csv(out / "tier0b_granularity.csv", index=False)
    print("\n--- POST-HOC: page-level provenance vs chunk-level presence ---")
    print(gran.to_string(index=False))
    print(f"coincidental agreement (both fire, string matched OFF the gold page): "
          f"{gran.attrs['coincidental']}")
    print(f"string matched off the gold page, any cell: "
          f"{gran.attrs['off_page_matches']}/{pooled['n_presence_wb']} = "
          f"{gran.attrs['off_page_share']:.4f} of presence_wb-true rows")
    print("the 10M corpus averages 6.55 chunks per page, so page-level "
          "provenance is\nLOOSER than chunk-level answerability -- the opposite "
          "direction to spec 3A.2's\ncaveat. On every D- row the gold string is "
          "absent from the retrieved gold-page\nchunk, so D- is NOT the "
          "unambiguous false negative the spec calls it.")

    # ---- spec 3A.4 A-VAL
    ov, av = aval(rows_df)
    ov.to_csv(out / "tier0b_aval.csv", index=False)
    print(f"\n--- spec 3A.4 A-VAL (overlap of the 124 with the 93 declines) ---")
    print(f"overlap n = {av['n_overlap']}  (underpowered threshold {AVAL_MIN_N})")
    print(f"agreement, all labels     = {av['agreement_all']:.4f} "
          f"on n={av['n_overlap']}")
    print(f"agreement, core contrast  = {av['agreement_core']:.4f} "
          f"on n={av['n_core']}  (not_supported vs answerable_declined)")
    print(f"not_supported with provenance FALSE: {av['not_supported_prov_false']}"
          f"/{av['n_not_supported']}")
    print(f"answerable_declined with provenance TRUE: {av['answerable_prov_true']}"
          f"/{av['n_answerable']}")
    print(f"POST-HOC  not_supported whose gold string is NOT on a gold page: "
          f"{av['n_not_supported_str_off_gold_page']}/{av['n_not_supported']}"
          f"  (agreement {av['agreement_posthoc_str']:.4f}) -- resolves no branch")
    if av["underpowered"]:
        print(f"⚠️  n = {av['n_overlap']} < {AVAL_MIN_N}: A-VAL is UNDERPOWERED. "
              "V1/V2 are NOT resolved; the numbers above are indicative only "
              "and §6 keeps its provisional status.")
        v_branch, v_fires = "UNDERPOWERED", None
    else:
        v_branch = "V1" if av["agreement_all"] >= WS4B_AVAL_AGREEMENT else "V2"
        v_fires = True

    # ---- branch table
    branches = [
        {"branch": p, "reading": "D+ joint (pre-registered text)",
         "fires": p == b_lit,
         "evidence": f"D+ = {pooled['n_fp']}/{pooled['n_rows']} = "
                     f"{pooled['d_plus_joint']:.4f}",
         "consequence": P_CONSEQUENCE[p]}
        for p in ("P1", "P2", "P3")
    ] + [
        {"branch": p, "reading": "D+ conditional (\"false-positive rate\")",
         "fires": p == b_cond,
         "evidence": f"D+ = {pooled['n_fp']}/{pooled['n_presence_wb']} = "
                     f"{pooled['d_plus_conditional']:.4f}",
         "consequence": P_CONSEQUENCE[p]}
        for p in ("P1", "P2", "P3")
    ] + [
        {"branch": "V1", "reading": "A-VAL agreement >= 0.75",
         "fires": (v_branch == "V1") if v_fires else None,
         "evidence": f"agreement {av['agreement_all']:.4f} on n={av['n_overlap']}"
                     + ("; UNDERPOWERED, not resolved" if av["underpowered"] else ""),
         "consequence": "the decline audit is corroborated by a signal that "
                        "involved no judgement; provisional status lifted for "
                        "the not_supported label"},
        {"branch": "V2", "reading": "A-VAL agreement < 0.75",
         "fires": (v_branch == "V2") if v_fires else None,
         "evidence": f"agreement {av['agreement_all']:.4f} on n={av['n_overlap']}"
                     + ("; UNDERPOWERED, not resolved" if av["underpowered"] else ""),
         "consequence": "the audit's labels are NOT corroborated; §6's numbers "
                        "revert to provisional and the human spot-check becomes "
                        "required, not recommended"},
    ]
    bdf = pd.DataFrame(branches)
    bdf.to_csv(out / "tier0b_branches.csv", index=False)

    # ---- labelled sensitivity
    curve = sensitivity_curve(rows_df)
    curve.to_csv(out / "tier0b_sensitivity_curve.csv", index=False)

    print("\n--- branches ---")
    print(bdf[["branch", "reading", "fires", "evidence"]].to_string(index=False))
    print("\n--- sensitivity curve (n=124, NOT a headline) ---")
    print(curve[["arm", "recall_at_10", "presence_wb", "provenance_at_10",
                 "judge_correct"]].to_string(index=False))
    print(f"\nwrote 6 CSVs into {out}/")


if __name__ == "__main__":
    main()
