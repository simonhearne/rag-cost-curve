#!/usr/bin/env python
"""WS4c Phase 0 -- the coverage gate. $0, CPU only, no Milvus, no API.

Design: results/recall-vs-quality.md

The pinned 10M slice is a PREFIX of a 41,488,110-row corpus. A multi-hop
question is only usable if EVERY gold page is inside it. This script measures
that for each candidate dataset and writes results/ws4c/coverage.csv. Nothing
downstream may run until C0/C2 are recorded here.

C1 (added 2026-09-11) decides the UNIT of WS4c's provenance metric: can every
supporting sentence be located in exactly one chunk of its gold title, or must
provenance fall back to the title. That fallback is no longer cheap. WS4b Tier
0b (results/recall-vs-quality.md) measured page-level provenance at 6.55 chunks/page
and showed it cannot adjudicate the very question WS4c exists to settle -- it
fires on `Madden NFL 19` for a question that page does not answer. So a C1
failure is not a routine downgrade; it is a threat to WS4c's premise, and this
script reports enough detail to tell WHY it failed.

Run:  python scripts/recall-quality/ws4c_phase0.py
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib.config import DATA_DIR, RESULTS_DIR, ids_path

C0_MIN_QUESTIONS = 600
C2_MAX_PAGE_SHARE = 0.05
C1_MIN_SHARE = 0.90        # spec 3: >= 90% of surviving questions
C1_MIN_SENT_CHARS = 40     # below this a "supporting sentence" is not a
                           # distinctive string; WS4's metric died of exactly
                           # this and the count is reported, not swept up
# Token-containment diagnostic (NOT a C1 reading; see gate_c1). A chunk is a
# unique locator when it holds this share of the sentence's tokens and beats
# the runner-up chunk by this margin.
C1_TOKEN_CONTAINMENT = 0.80
C1_TOKEN_MARGIN = 0.05

# HotpotQA renders italicised work titles as quotes; the corpus strips the
# markup. Removing quote characters ONLY -- see normq.
QUOTE_CHARS = dict.fromkeys(map(ord, '"\u201c\u201d\u2018\u2019\u00ab\u00bb\u201e\u2032\u2033`'), None)


def slice_titles() -> set:
    return set(pd.read_parquet(ids_path("10m"), columns=["title"]).title.unique())


def hotpot(split: str, titles: set, keep: dict | None = None) -> dict:
    files = {
        "dev": ["distractor/validation-00000-of-00001.parquet"],
        "train": ["distractor/train-00000-of-00002.parquet",
                  "distractor/train-00001-of-00002.parquet"],
    }[split]
    h = pd.concat([pd.read_parquet(hf_hub_download("hotpotqa/hotpot_qa", f,
                                                   repo_type="dataset"))
                   for f in files])
    gold = h.supporting_facts.map(lambda x: set(map(str, x["title"])))
    all_in = gold.map(lambda t: len(t) > 0 and t <= titles)
    sub = h[all_in]
    yn = sub.answer.str.lower().isin(["yes", "no"])
    pages = pd.Series([t for s in gold[all_in] for t in s]).value_counts()
    if keep is not None:
        keep[f"hotpotqa_distractor_{split}"] = sub[~yn]
    return {
        "dataset": f"hotpotqa_distractor_{split}",
        "questions": len(h),
        "mean_gold_titles": round(float(gold.map(len).mean()), 3),
        "all_gold_in_slice": int(all_in.sum()),
        "coverage": round(float(all_in.mean()), 5),
        "usable_non_yesno": int((~yn).sum()),
        "top_page": str(pages.index[0]) if len(pages) else "",
        "top_page_share": round(float(pages.iloc[0] / max(len(sub), 1)), 5) if len(pages) else 0.0,
    }


def musique(split: str, titles: set) -> dict:
    fn = f"musique_ans_v1.0_{split}.jsonl"
    rows = [json.loads(l) for l in open(hf_hub_download("bdsaglam/musique", fn,
                                                        repo_type="dataset"))]
    gold = [{str(p["title"]) for p in r["paragraphs"] if p.get("is_supporting")}
            for r in rows]
    all_in = [len(s) > 0 and s <= titles for s in gold]
    pages = pd.Series([t for s, k in zip(gold, all_in) if k for t in s]).value_counts()
    return {
        "dataset": f"musique_ans_{split}",
        "questions": len(rows),
        "mean_gold_titles": round(float(np.mean([len(s) for s in gold])), 3),
        "all_gold_in_slice": int(sum(all_in)),
        "coverage": round(float(np.mean(all_in)), 5),
        "usable_non_yesno": int(sum(all_in)),  # MuSiQue-Ans has no yes/no
        "top_page": str(pages.index[0]) if len(pages) else "",
        "top_page_share": round(float(pages.iloc[0] / max(sum(all_in), 1)), 5) if len(pages) else 0.0,
    }


# ----------------------------------------------------------------- C1 ----


def norm(s: str) -> str:
    """Whitespace only. Deliberately NOT the WS4 normalisation: stripping
    punctuation is what turned gold `["---"]` into a universal match, and
    these strings are long enough not to need it."""
    return " ".join(str(s).split())


def normc(s: str) -> str:
    return norm(s).casefold()


def normq(s: str) -> str:
    """normc plus NFKC and the removal of quote characters ONLY.

    HotpotQA's sentences come from Wikipedia abstracts that render italicised
    work titles as double quotes; the pinned corpus strips the markup. So
    HotpotQA has `The tenth season of "South Park", an American animated ...`
    where the corpus has `The tenth season of South Park, an American
    animated ...` -- identical text, different rendering. That is a rendering
    artifact, not evidence about the corpus, and it is removed.

    The class removed is deliberately narrow -- quotes and nothing else. This
    is the exact slope WS4's metric fell down (see benchlib/recall_quality/gold.py), so the
    strict figure is reported alongside and neither is suppressed."""
    s = unicodedata.normalize("NFKC", str(s)).translate(QUOTE_CHARS)
    return " ".join(s.split()).casefold()


def tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", normq(s)))


def gold_chunk_texts(wanted: set, scale: str = "10m") -> pd.DataFrame:
    """title/text for every chunk of the wanted titles, read from the local
    shards. Cached under data/ws4c_cache/ -- NOT into WS4's passage cache,
    which WS4b Tier 0/0b read as frozen input."""
    import pyarrow.parquet as pq

    from benchlib import manifest
    from benchlib.download import shard_local_path

    cache = DATA_DIR / "ws4c_cache" / "gold_chunks.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        if set(df.title.unique()) >= wanted:
            return df

    ids = pd.read_parquet(ids_path(scale), columns=["title", "shard", "row_in_shard"])
    sub = ids[ids.title.isin(wanted)]
    print(f"C1  reading {len(sub):,} gold-title chunks from "
          f"{sub.shard.nunique()} shards ...")
    out = []
    for shard, grp in sub.groupby("shard"):
        texts = pq.read_table(shard_local_path(int(shard)),
                              columns=["text"]).column("text").to_pylist()
        out += [(int(row), title, texts[int(ris)]) for row, title, ris
                in zip(grp.index, grp["title"], grp["row_in_shard"])]
    df = pd.DataFrame(out, columns=["row", "title", "text"]).sort_values("row")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    manifest.record(cache, extra={"rows": len(df), "titles": int(df.title.nunique()),
                                  "purpose": "WS4c C1 gold-title chunk texts"})
    return df


def supporting_sentences(row) -> list[tuple]:
    """[(gold_title, sentence)] for one HotpotQA question."""
    ct = list(row.context["title"])
    cs = list(row.context["sentences"])
    out = []
    for t, sid in zip(row.supporting_facts["title"], row.supporting_facts["sent_id"]):
        t = str(t)
        if t not in ct:
            out.append((t, None))          # supporting fact with no context
            continue
        sents = list(cs[ct.index(t)])
        out.append((t, str(sents[int(sid)]) if 0 <= int(sid) < len(sents) else None))
    return out


def gate_c1(surviving: pd.DataFrame) -> tuple:
    """spec 3 C1: >= 90% of surviving questions have EVERY supporting sentence
    locatable in exactly one chunk of its gold title.

    Three locators are computed on the same sentences, and all three are
    reported:

      strict   exact substring after whitespace+case normalisation. This is
               C1 as literally pre-registered and it is the headline number.
      quoted   the same, after removing quote characters only (see normq).
      tokens   DIAGNOSTIC, not a C1 reading. Best-matching chunk by token
               containment |S & C| / |S|, with the margin over the runner-up.
               It answers the question the pass/fail cannot: when the exact
               string is not found, is the supporting CONTENT nevertheless
               present in one identifiable chunk, or not present at all? The
               two have opposite consequences for WS4c and spec 3 does not
               distinguish them.
    """
    wanted = {t for _, r in surviving.iterrows()
              for t, _ in supporting_sentences(r) if t}
    chunks = gold_chunk_texts(wanted)
    chunks["strict"] = chunks.text.map(normc)
    chunks["quoted"] = chunks.text.map(normq)
    chunks["tok"] = chunks.text.map(tokens)
    by_title = {t: list(zip(g.row, g.strict, g.quoted, g.tok))
                for t, g in chunks.groupby("title")}

    per_sent, per_q = [], []
    for _, r in surviving.iterrows():
        c_strict, c_quoted, tok_ok = [], [], []
        short = missing = 0
        for title, sent in supporting_sentences(r):
            if sent is None:
                missing += 1
                c_strict.append(-1)
                c_quoted.append(-1)
                tok_ok.append(False)
                continue
            cand = by_title.get(title, [])
            n_raw = len(norm(sent))
            short += n_raw < C1_MIN_SENT_CHARS
            ks, kq, S = normc(sent), normq(sent), tokens(sent)

            hits = [row for row, st, _, _ in cand if ks in st]
            n_s = len(hits)
            n_q = sum(1 for _, _, qt, _ in cand if kq in qt)

            sc = sorted((len(S & C) / len(S) if S else 0.0) for _, _, _, C in cand)
            best = sc[-1] if sc else 0.0
            margin = (sc[-1] - sc[-2]) if len(sc) > 1 else 1.0
            unique = best >= C1_TOKEN_CONTAINMENT and margin >= C1_TOKEN_MARGIN

            per_sent.append({
                "qid": r.id, "title": title, "sent_chars": n_raw,
                "n_chunks_strict": n_s, "n_chunks_quoted": n_q,
                "best_token_containment": round(float(best), 4),
                "token_margin": round(float(margin), 4),
                "token_unique": bool(unique),
                "short": n_raw < C1_MIN_SENT_CHARS,
                "adjacent_chunks": bool(n_s > 1 and sorted(hits) == list(
                    range(min(hits), min(hits) + n_s))),
                "n_title_chunks": len(cand)})
            c_strict.append(n_s)
            c_quoted.append(n_q)
            tok_ok.append(bool(unique))

        per_q.append({
            "qid": r.id, "n_sents": len(c_strict),
            "all_exactly_one_strict": bool(c_strict and all(c == 1 for c in c_strict)),
            "all_exactly_one_quoted": bool(c_quoted and all(c == 1 for c in c_quoted)),
            "all_token_unique": bool(tok_ok and all(tok_ok)),
            "any_zero_quoted": any(c == 0 for c in c_quoted),
            "n_short": short, "n_missing_context": missing})

    sents = pd.DataFrame(per_sent)
    qs = pd.DataFrame(per_q)
    sc = sents[sents.n_chunks_strict >= 0]
    share_strict = float(qs.all_exactly_one_strict.mean())
    share_quoted = float(qs.all_exactly_one_quoted.mean())
    stats = {
        "n_questions": len(qs),
        "n_sentences": len(sents),
        # --- C1 as pre-registered
        "c1_share_strict": round(share_strict, 5),
        "c1_share_quoted": round(share_quoted, 5),
        "C1_pass": bool(share_quoted >= C1_MIN_SHARE),
        # --- why
        "sent_share_zero_strict": round(float((sc.n_chunks_strict == 0).mean()), 5),
        "sent_share_zero_quoted": round(float((sc.n_chunks_quoted == 0).mean()), 5),
        "sent_share_one_quoted": round(float((sc.n_chunks_quoted == 1).mean()), 5),
        "sent_share_multi_quoted": round(float((sc.n_chunks_quoted > 1).mean()), 5),
        "sent_share_short": round(float(sc.short.mean()), 5),
        "n_missing_context": int(qs.n_missing_context.sum()),
        # --- token diagnostic
        "token_median_containment": round(float(sc.best_token_containment.median()), 5),
        "token_median_margin": round(float(sc.token_margin.median()), 5),
        "sent_share_token_unique": round(float(sc.token_unique.mean()), 5),
        "q_share_token_unique": round(float(qs.all_token_unique.mean()), 5),
    }
    return sents, qs, stats


def main() -> None:
    titles = slice_titles()
    print(f"10M slice distinct titles: {len(titles):,}\n")

    keep: dict = {}
    rows = [hotpot("dev", titles, keep), hotpot("train", titles, keep),
            musique("dev", titles), musique("train", titles)]
    df = pd.DataFrame(rows)
    df["C0_pass"] = df.usable_non_yesno >= C0_MIN_QUESTIONS
    df["C2_pass"] = df.top_page_share < C2_MAX_PAGE_SHARE

    out = RESULTS_DIR / "ws4c"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "coverage.csv", index=False)

    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    ok = df[df.C0_pass & df.C2_pass]
    print(f"\nC0 (>= {C0_MIN_QUESTIONS} usable) and C2 (top page < "
          f"{C2_MAX_PAGE_SHARE:.0%}) passed by: "
          f"{list(ok.dataset) if len(ok) else 'NOTHING -- WS4c stops'}")

    # ---- C1, on the dataset(s) that survived C0 and C2 (spec 3)
    for ds in ok.dataset:
        if ds not in keep:
            print(f"\nC1 skipped for {ds}: only implemented for HotpotQA "
                  "(MuSiQue has no sentence-level supporting facts here)")
            continue
        print(f"\n--- C1 on {ds} ({len(keep[ds]):,} surviving questions) ---")
        sents, qs, stats = gate_c1(keep[ds])
        sents.to_csv(out / "c1_sentences.csv", index=False)
        qs.to_csv(out / "c1_questions.csv", index=False)
        pd.DataFrame([{"dataset": ds, **stats}]).to_csv(out / "c1.csv", index=False)

        print(f"C1  questions with EVERY supporting sentence in exactly one chunk")
        print(f"      strict (as pre-registered): "
              f"{stats['c1_share_strict']:.4f}")
        print(f"      quote-normalised          : "
              f"{stats['c1_share_quoted']:.4f}   threshold {C1_MIN_SHARE}  -> "
              f"{'PASS' if stats['C1_pass'] else 'FAIL'}")
        print(f"    per sentence (quote-normalised): 0 chunks "
              f"{stats['sent_share_zero_quoted']:.4f} | 1 chunk "
              f"{stats['sent_share_one_quoted']:.4f} | >1 chunk "
              f"{stats['sent_share_multi_quoted']:.4f}")
        print(f"    sentences under {C1_MIN_SENT_CHARS} chars: "
              f"{stats['sent_share_short']:.4f}   "
              f"missing context: {stats['n_missing_context']}")
        print(f"\n    DIAGNOSTIC -- is the CONTENT there even when the string "
              f"is not?")
        print(f"      best-chunk token containment: median "
              f"{stats['token_median_containment']:.4f}, median margin over "
              f"runner-up {stats['token_median_margin']:.4f}")
        print(f"      sentences with a unique locating chunk "
              f"(>={C1_TOKEN_CONTAINMENT} containment, >={C1_TOKEN_MARGIN} "
              f"margin): {stats['sent_share_token_unique']:.4f}")
        print(f"      questions with ALL sentences uniquely located: "
              f"{stats['q_share_token_unique']:.4f}")
        if not stats["C1_pass"]:
            print("\n    ⚠️  C1 FAILS, and NEITHER remedy is available off the "
                  "shelf:")
            print("        - spec 3's fail action is 'fall back to title-level "
                  "provenance', but")
            print("          WS4b Tier 0b §9.3 showed title-level provenance "
                  "cannot adjudicate the")
            print("          question WS4c exists to settle (6.55 chunks/page).")
            print("        - a token-containment locator DOES find the content, "
                  "but at")
            print(f"          {stats['q_share_token_unique']:.4f} of questions "
                  f"it is still below the pre-registered {C1_MIN_SHARE}.")
            print("        This is a spec decision, not a script default. "
                  "Nothing downstream runs.")

    print(f"\nwrote {out}/coverage.csv and c1*.csv")


if __name__ == "__main__":
    main()
