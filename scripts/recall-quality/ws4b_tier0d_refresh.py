#!/usr/bin/env python
"""WS4b Tier 0d stage 1+2 -- re-derive gold answers for the audited error rows.

Design: results/recall-vs-quality.md

Scope is the 107 rows where SOME ARM ERRED. That selection is one-sided (spec
2): a row where every arm was correct against a bad gold is never examined.
Everything downstream is therefore an UPPER BOUND on corrected accuracy.

The pass sees question, current golds and the exact top-10 passages -- never an
arm's answer, an arm name, or a prior audit label. Gate G4 asserts that against
the rendered prompts.

Run:  set -a; source .env; set +a
      .venv/bin/python scripts/recall-quality/ws4b_tier0d_refresh.py --limit 5   # smoke
      .venv/bin/python scripts/recall-quality/ws4b_tier0d_refresh.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest
from benchlib.recall_quality import gold as ws4b_gold
from benchlib.recall_quality import gold_runner as t0d
from benchlib.recall_quality import core
from benchlib.config import (DATA_DIR, RESULTS_DIR, WS4_TOP_K, WS4B_T0D_N_ROWS,
                             WS4B_T0D_PROPOSE_MODEL, WS4B_T0D_VERIFY_MODEL,
                             ws4b_dir)
# Cost/governor helpers are shared machinery, not code-retrieval-specific.
from benchlib.code_retrieval.bench import BudgetExceeded, load_pricing

LABELLED = "results/ws4b/tier0_staleness_labelled.csv"
PASSAGES = DATA_DIR / "ws4_cache" / "passages.parquet"
GT = DATA_DIR / "gt_10m_top100.npz"


def log(m):
    print(m, flush=True)


def gt_passages(qids) -> dict:
    """qid -> the exact fp32 top-WS4_TOP_K passages, rendered.

    Read-only against the FROZEN passages.parquet. A missing row is a hard
    error, never a silent refetch: ws4_runner.load_passages would WRITE to
    that file, which is Tier 0/0b's frozen input.
    """
    if not PASSAGES.exists():
        raise SystemExit(f"{PASSAGES} missing -- WS4 must have run first")
    tbl = pd.read_parquet(PASSAGES).set_index("row")
    idx = np.load(GT)["ws4_indices"]
    out = {}
    for qid in qids:
        rows = [int(r) for r in idx[int(qid)][:WS4_TOP_K] if int(r) >= 0]
        missing = [r for r in rows if r not in tbl.index]
        if missing:
            raise SystemExit(
                f"qid {qid}: passages.parquet is missing rows {missing[:5]}. "
                f"That file is FROZEN input to Tier 0/0b -- do not regenerate "
                f"it; investigate why WS4's cache is incomplete.")
        out[qid] = core.render_context(
            [{"title": tbl.at[r, "title"], "text": tbl.at[r, "text"]} for r in rows])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N newly processed rows (smoke run)")
    args = ap.parse_args()

    lab = pd.read_csv(LABELLED)
    assert len(lab) == WS4B_T0D_N_ROWS, (
        f"expected {WS4B_T0D_N_ROWS} audited rows, found {len(lab)}")
    out = ws4b_dir()

    # Withhold everything spec 3.1 says to withhold. The pass gets three
    # columns and nothing else.
    work = lab[["qid", "question", "golds"]].copy()
    labels_by_qid = dict(zip(lab.qid, lab.label))     # G1 ONLY, never a prompt
    forbidden = sorted({str(a) for a in lab.answer.dropna()}
                       | set(lab.label.dropna().unique()))

    passages = gt_passages(work.qid.tolist())
    propose_t = (out / "tier0d_propose_prompt.txt").read_text()
    verify_t = (out / "tier0d_verify_prompt.txt").read_text()
    prices = load_pricing(RESULTS_DIR / "ws6a" / "pricing.csv", regime="standard")
    gov = t0d.tier_governor()
    client = t0d.make_client()
    log(f"tier0d  governor ${gov.spent:.4f} of ${gov.limit:.2f} "
        f"(WS4B_T0D_TIER_CAP_USD, inside WS4B_BUDGET_USD; never raised)")

    done = {}
    if t0d.CHECKPOINT.exists():
        for line in t0d.CHECKPOINT.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    done[r["qid"]] = r
                except json.JSONDecodeError:
                    pass
    log(f"tier0d  {len(done)} rows already recorded")

    # actual_by_qid holds the prompt(s) REALLY sent this invocation, keyed by
    # qid -- gate G4's structural check compares these against an independent
    # reconstruction. A row recovered from checkpoint makes no fresh call, so
    # it simply has no entry here; G4 treats that as "nothing to compare".
    actual_by_qid, rows, n_new = {}, [], 0
    for _, w in work.iterrows():
        qid = int(w.qid)
        if qid in done or str(qid) in done:
            prior = done.get(qid, done.get(str(qid)))
            # A row checkpointed mid-abort (BudgetExceeded raised between a
            # billed charge() and the row.update() that would have recorded
            # its result -- see t0d.charge()'s docstring) is INCOMPLETE, not
            # done: an empty verdict means the propose charge landed but the
            # verdict was never written, and an augment/replace verdict with
            # no verify_reason means the verify charge landed but its result
            # was never written. Both must be redone on resume rather than
            # silently trusted, or a fabricated "verifier rejected it" (or a
            # blank verdict) would re-enter diff_rows and contaminate G1/G2.
            incomplete = (not prior.get("verdict")
                          or (prior.get("verdict") in ("augment", "replace")
                              and not prior.get("verify_reason")))
            if not incomplete:
                rows.append(prior)
                continue
        if args.limit is not None and n_new >= args.limit:
            break
        row = {"qid": qid, "arm": "tier0d", "question": w.question,
               "old_golds": w.golds, "verdict": "", "new_golds": "[]",
               "operation": "unchanged", "propose_reason": "", "uphold": False,
               "verify_reason": "", "cost_total_billed": 0.0}
        try:
            p = t0d.propose(client, propose_t, question=w.question,
                            golds=w.golds, passages=passages[qid])
            actual_by_qid[qid] = {"propose": p["prompt"], "verify": None}
            # row.update() BEFORE charge(): charge() can raise BudgetExceeded
            # after the call is already billed, and the row must be complete
            # -- a real verdict, not the "" it was initialised with -- before
            # that can happen, or an aborted row checkpoints with a blank
            # verdict (see the resume check above).
            err = ws4b_gold.validate_verdict(p["verdict"], w.golds, p["new_golds"])
            if err:
                row.update(verdict="keep", propose_reason=f"rejected: {err}")
            else:
                row.update(verdict=p["verdict"],
                           new_golds=json.dumps(p["new_golds"]),
                           propose_reason=p["reason"][:300])
            t0d.charge(gov, p["usage"], prices, WS4B_T0D_PROPOSE_MODEL, row)
            if not err and p["verdict"] in ("augment", "replace"):
                v = t0d.verify(client, verify_t, question=w.question,
                               golds=w.golds, passages=passages[qid],
                               proposed=p["new_golds"])
                actual_by_qid[qid]["verify"] = v["prompt"]
                # Same ordering rule: the row must carry uphold/verify_reason
                # (and operation, if upheld) before charge() can raise, or an
                # aborted row checkpoints as augment/replace with uphold=False
                # and no verify_reason -- indistinguishable from a genuine
                # verifier rejection once it re-enters diff_rows.
                row.update(uphold=bool(v["uphold"]),
                           verify_reason=v["reason"][:300])
                if v["uphold"]:
                    row["operation"] = ws4b_gold.operation_of(
                        p["verdict"], w.golds, p["new_golds"])
                t0d.charge(gov, v["usage"], prices, WS4B_T0D_VERIFY_MODEL, row)
        except t0d.VerifyError as exc:
            # Populate the row BEFORE charging. charge() can raise
            # BudgetExceeded, and a BudgetExceeded raised inside an `except`
            # block is NOT caught by the sibling `except BudgetExceeded`
            # below -- it propagates out of the loop, skipping the
            # append_checkpoint() at the bottom, so the row is lost and
            # tier_spent() under-counts money already billed (spec §8.1).
            # VerifyError can only be raised after propose succeeded, so
            # actual_by_qid[qid] already exists -- record the real, billed
            # verify prompt so G4 can still assert its blindness below.
            actual_by_qid[qid]["verify"] = exc.prompt
            row.update(verdict="keep",
                       verify_reason=t0d.VERIFIER_EXHAUSTED_REASON)
            try:
                t0d.charge(gov, exc.usage, prices, WS4B_T0D_VERIFY_MODEL, row)
            except BudgetExceeded as budget_exc:
                log(f"tier0d  STOP (in VerifyError handler): {budget_exc}")
                t0d.append_checkpoint(t0d.CHECKPOINT, row)
                break
        except BudgetExceeded as exc:
            log(f"tier0d  STOP: {exc}")
            if row["cost_total_billed"] > 0:
                t0d.append_checkpoint(t0d.CHECKPOINT, row)
            break
        t0d.append_checkpoint(t0d.CHECKPOINT, row)
        rows.append(row)
        n_new += 1
        if n_new % 20 == 0:
            log(f"tier0d  {n_new} new | ${gov.spent:.3f}/${gov.limit:.2f}")

    df = pd.DataFrame([r for r in rows if r])

    # This repo's hard rule: never commit corpus data. propose_reason and
    # verify_reason are free-text model justifications that naturally quote
    # the passage they reason about; redact any run that appears verbatim in
    # THAT ROW'S OWN top-10 passages before this DataFrame is written to a
    # committed CSV. The checkpoint (gitignored) keeps the raw text for
    # local diagnosis -- only the committed artifact is sanitised.
    df_out = df.copy()
    for col in ("propose_reason", "verify_reason"):
        df_out[col] = [
            ws4b_gold.redact_verbatim_passage_text(text, passages[qid])
            if isinstance(text, str) else text
            for qid, text in zip(df_out["qid"], df_out[col])]
    df_out.to_csv(out / "tier0d_gold_diff.csv", index=False)
    manifest.record(out / "tier0d_gold_diff.csv")

    conc = (df.assign(prior_label=df.qid.map(
                lambda q: labels_by_qid.get(q, labels_by_qid.get(str(q), ""))))
              .groupby(["prior_label", "verdict", "operation"])
              .size().reset_index(name="n"))
    conc.to_csv(out / "tier0d_concordance.csv", index=False)
    manifest.record(out / "tier0d_concordance.csv")

    diff_rows = df.to_dict("records")

    # G4's structural check needs, for every row (checkpointed or fresh), an
    # independently rebuilt prompt to compare against whatever was actually
    # sent. Rebuilding costs nothing -- fill() is pure string substitution
    # and passages[qid] is already loaded read-only -- so this covers all 107
    # rows even on a resumed run that made zero fresh calls.
    g4_rows = []
    for r in diff_rows:
        qid = r["qid"]
        actual = actual_by_qid.get(qid, {})
        expected_propose = t0d.fill(propose_t, question=r["question"],
                                    golds=r["old_golds"], passages=passages[qid])
        g4_rows.append({"qid": qid, "question": r["question"],
                        "golds": r["old_golds"], "expected": expected_propose,
                        "actual": actual.get("propose")})
        # A verify call happened whenever the ORIGINAL stage-1 verdict was
        # augment/replace -- including a VerifyError row, where `verdict` is
        # overwritten to "keep" afterwards but `verify_reason` keeps the
        # VERIFIER_EXHAUSTED_REASON marker and `new_golds` still holds
        # the proposal that was actually sent. Without this second
        # condition, a VerifyError row's billed verify call would get no G4
        # entry at all.
        if (r["verdict"] in ("augment", "replace")
                or r.get("verify_reason") == t0d.VERIFIER_EXHAUSTED_REASON):
            proposed = json.loads(r["new_golds"]) if r["new_golds"] else []
            expected_verify = t0d.fill(verify_t, question=r["question"],
                                       golds=r["old_golds"], passages=passages[qid],
                                       proposed=json.dumps(proposed))
            g4_rows.append({"qid": qid, "question": r["question"],
                            "golds": r["old_golds"], "expected": expected_verify,
                            "actual": actual.get("verify")})

    # gate_g1 looks up by the integer qid, which already compares equal
    # (and hashes equal) to labels_by_qid's numpy int64 keys -- no string
    # variant is needed.
    gates = [ws4b_gold.gate_g1(diff_rows, labels_by_qid),
             ws4b_gold.gate_g2(diff_rows),
             ws4b_gold.gate_g4(g4_rows, forbidden)]
    pd.DataFrame(gates).to_csv(out / "tier0d_gates.csv", index=False)
    manifest.record(out / "tier0d_gates.csv")

    log("")
    log(df.verdict.value_counts().to_string())
    for g in gates:
        log(f"  {g['gate']:<4} {'PASS' if g['pass'] else 'FAIL'}  {g}")
    log(f"\ntier0d  ${gov.spent:.4f} of ${gov.limit:.2f}")
    if not all(g["pass"] for g in gates):
        log("\n  *** A GATE FAILED. Do not run the rescore. ***")
        raise SystemExit(1)
    log("\n  Sanity before trusting this: is the verdict mix plausible? A run")
    log("  that keeps everything has not checked anything, and a run that")
    log("  replaces most golds is far more likely to be broken than right.")


if __name__ == "__main__":
    main()
