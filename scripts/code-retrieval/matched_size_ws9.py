#!/usr/bin/env python
"""WS9 C-series: is the fastapi/agentic-hil difference explained by SIZE?

Outputs: results/ws6c/matched_size_contamination.csv. Zero API spend.

The fastapi/agentic-hil difference is attributed to contamination only if a
comparison AT MATCHED CORPUS SIZE shows it. Two scopes are matched iff their
measured count_tokens are within WS9_MATCH_BAND of each other. Pairing is by
measured tokens, never by file count -- WS6a section 7's rule.

  C1  the per-query winner DIFFERS across corpora at matched size. Size alone
      does not explain the fastapi result; contamination remains the candidate.
  C2  the per-query winner is the SAME on both. The matched comparison does not
      support the corpus-identity explanation.
  C3  no pair falls inside the band on measurement; the test does not run.

A C-letter is emitted only from two complete cells with distinct means. An arm
with no rows at a scope, or a mean that is not finite, is a HARD ERROR and not
a verdict -- a partial or mid-write scaling.csv must never be labelled. Exactly
equal means are a tie, labelled "tie (not evaluated)", and yield no C-letter
rather than being broken silently toward either arm.

THIS COMPARISON IS NOT PAIRED AND NO INTERVAL IS COMPUTED ACROSS IT. The two
corpora have different questions, different n (10 or 40 on fastapi against 37
here) and no correspondence between rows. WS6c is explicit that no cross-corpus
interval may be paired. This is a DESCRIBED COMPARISON OF POINT ESTIMATES, and
the CSV carries that in a column, not only in prose.

fastapi has no indexed_topk3 arm at S or M -- Phase 1 ran the knob at L only --
so the C-series is defined over agentic vs indexed alone and makes no claim
about top-k across corpora.

Run:  .venv/bin/python scripts/code-retrieval/matched_size_ws9.py
"""

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.config import WS9_MATCH_BAND, ws6a_dir, ws6c_dir

ARMS = ("agentic", "indexed")
INT_COLS = ["fastapi_tokens", "unseen_tokens", "fastapi_files",
            "unseen_files", "fastapi_n", "unseen_n"]
UNPAIRED = ("unpaired: different questions, different n, no row "
            "correspondence; no interval is computed across corpora")
STATISTIC = "mean cost_agent_billed per query (judge excluded)"


def guard(cell: dict, counts: dict, corpus: str, scope: str) -> dict:
    """Refuse to label a pair from an incomplete cell.

    A missing arm or a non-finite mean means the frame is partial or mid-write.
    n is reported from the agentic arm alone, so a missing indexed arm would
    otherwise pass unnoticed into a manifest-recorded CSV as a confident
    C-letter. Die naming the corpus, scope and arm instead.
    """
    for arm in ARMS:
        if counts[arm] == 0:
            raise SystemExit(
                f"{corpus} {scope}: no rows for arm '{arm}' -- the frame is "
                "incomplete; refusing to emit a C-letter from it")
        if not math.isfinite(cell[f"{arm}_usd"]):
            raise SystemExit(
                f"{corpus} {scope}: mean cost_agent_billed for arm '{arm}' is "
                "not finite -- refusing to emit a C-letter from it")
    return cell


def rows_cell(sub: pd.DataFrame, corpus: str, scope: str) -> dict:
    """Per-arm mean billed $/query and n from one scope's per-query rows."""
    counts = {a: int((sub.arm == a).sum()) for a in ARMS}
    cell = {f"{a}_usd": float(sub[sub.arm == a].cost_agent_billed.mean())
            for a in ARMS}
    cell["n"] = counts["agentic"]
    return guard(cell, counts, corpus, scope)


def fastapi_cell(scope: str) -> dict:
    """fastapi per-arm mean billed $/query and n at one scope.

    L comes from summary.csv (n=40, the main run); S and M from scaling.csv
    (n=10, the scaling subsets). Both are marginal per-arm means over their own
    frozen rows -- exactly what the unseen side supplies.
    """
    if scope == "L":
        s = pd.read_csv(ws6a_dir() / "summary.csv").set_index("arm")
        counts = {a: int(s.loc[a, "n"]) if a in s.index else 0 for a in ARMS}
        cell = {f"{a}_usd": (float(s.loc[a, "total_cost_agent_billed"])
                             / counts[a]) if counts[a] else float("nan")
                for a in ARMS}
        cell["n"] = counts["agentic"]
        return guard(cell, counts, "fastapi", scope)
    sc = pd.read_csv(ws6a_dir() / "scaling.csv")
    return rows_cell(sc[sc.size_point == scope], "fastapi", scope)


def unseen_cell(scaling: pd.DataFrame, scope: str) -> dict:
    return rows_cell(scaling[scaling.size_point == scope],
                     "agentic_hil", scope)


def cheaper(cell: dict) -> str:
    if cell["agentic_usd"] == cell["indexed_usd"]:
        return "tie"
    return "agentic" if cell["agentic_usd"] < cell["indexed_usd"] else "indexed"


def main() -> None:
    out = ws6c_dir()
    fa_stats = pd.read_csv(ws6a_dir() / "repo_stats.csv").set_index("scope")
    un_stats = pd.read_csv(out / "scaling_corpus_stats.csv").set_index("scope")
    scaling = pd.read_csv(out / "scaling.csv")

    rows = []
    for fa in ("S", "M", "L"):
        for un in ("S", "M", "L"):
            fa_tok = int(fa_stats.loc[fa, "count_tokens"])
            un_tok = int(un_stats.loc[un, "count_tokens"])
            ratio = max(fa_tok, un_tok) / min(fa_tok, un_tok)
            in_band = ratio <= WS9_MATCH_BAND
            fa_cell = fastapi_cell(fa)
            un_cell = unseen_cell(scaling, un)
            fa_files = int(fa_stats.loc[fa, "files"])
            un_files = int(un_stats.loc[un, "files_text_allowlisted"])
            fa_win, un_win = cheaper(fa_cell), cheaper(un_cell)
            if not in_band:
                branch = "out of band (not evaluated)"
            elif "tie" in (fa_win, un_win):
                branch = "tie (not evaluated)"
            else:
                branch = "C1" if fa_win != un_win else "C2"
            rows.append({
                "pair": f"fastapi_{fa}__vs__agentic_hil_{un}",
                "fastapi_scope": fa, "unseen_scope": un,
                "fastapi_tokens": fa_tok, "unseen_tokens": un_tok,
                "token_ratio": round(ratio, 3), "in_band": in_band,
                "fastapi_files": fa_files, "unseen_files": un_files,
                "file_ratio": round(max(fa_files, un_files)
                                    / min(fa_files, un_files), 3),
                "fastapi_n": fa_cell["n"], "unseen_n": un_cell["n"],
                "fastapi_agentic_usd": round(fa_cell["agentic_usd"], 6),
                "fastapi_indexed_usd": round(fa_cell["indexed_usd"], 6),
                "fastapi_cheaper_arm": fa_win,
                "unseen_agentic_usd": round(un_cell["agentic_usd"], 6),
                "unseen_indexed_usd": round(un_cell["indexed_usd"], 6),
                "unseen_cheaper_arm": un_win,
                "branch": branch,
                "pairing": UNPAIRED, "statistic": STATISTIC,
            })

    df = pd.DataFrame(rows).sort_values("token_ratio").reset_index(drop=True)
    banded = df[df.in_band]
    if banded.empty:
        # C3 is a real outcome and is written, not silently omitted.
        df = pd.concat([df, pd.DataFrame([{
            "pair": "OVERALL", "branch": "C3", "in_band": False,
            "pairing": UNPAIRED, "statistic": STATISTIC}])], ignore_index=True)
        print("C3: no pair falls inside the band; the test does not run.")
    else:
        verdicts = set(banded.branch) & {"C1", "C2"}
        overall = ("tie (not evaluated)" if not verdicts else
                   "C1" if verdicts == {"C1"} else
                   "C2" if verdicts == {"C2"} else
                   "C1/C2 split -- report per pair")
        df = pd.concat([df, pd.DataFrame([{
            "pair": "OVERALL", "branch": overall, "in_band": True,
            "pairing": UNPAIRED, "statistic": STATISTIC}])], ignore_index=True)
        print(f"\nOVERALL: {overall}")

    # The OVERALL row carries none of the counts; keep them integral rather
    # than letting that one NaN upcast six count columns to float.
    df[INT_COLS] = df[INT_COLS].astype("Int64")
    path = out / "matched_size_contamination.csv"
    df.to_csv(path, index=False)
    manifest.record(path)
    print(df[["pair", "token_ratio", "in_band", "file_ratio",
              "fastapi_cheaper_arm", "unseen_cheaper_arm",
              "branch"]].to_string(index=False))
    print("\nThe file_ratio column is the confound: grep's cost scales with "
          "how much tree it must search, so a small-file-count haystack "
          "FAVOURS the agentic arm on the agentic_hil side of every pair -- "
          "the arm C1 requires to lose there. A C1 result is obtained AGAINST "
          "the confound; a C2 result is partly explicable by it.")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
