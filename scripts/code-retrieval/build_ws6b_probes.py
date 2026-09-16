#!/usr/bin/env python3
"""Measure the WS6b corpus and derive its probe set.

`count_tokens` against the pinned agent model is authoritative; the
generator's chars/4 estimate is not. Everything published rests on the
numbers this script measures.

Writes results/ws6b/{corpus_manifest,corpus_stats,probes}.csv and records
every one, plus every corpus file, in data/MANIFEST.json.

Exits non-zero without writing partial CSVs if any structural validator
fails (spec 3.5, 3.6, 4.3). The fix is to nudge the generator's placement and
regenerate -- never to relax a band.

Design: results/code-retrieval.md
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import config, manifest  # noqa: E402
from benchlib.code_retrieval import memory as ws6b  # noqa: E402
from benchlib.agent_harness import (count_tokens_batched, make_client,  # noqa: E402
                                  measure_request_overhead)

CORPUS = config.WS6B_CORPUS_DIR
OUT = config.ws6b_dir()


def _count(client, overhead, text: str) -> int:
    return count_tokens_batched(client, config.WS6B_AGENT_MODEL, [text],
                                overhead=overhead)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    day_files = sorted(CORPUS.glob("*.md"))
    if not day_files:
        print(f"no corpus at {CORPUS}; run generate_ws6b_corpus.py first")
        return 1
    ledger = json.loads((CORPUS / "facts.json").read_text())

    client = make_client()
    # WS6a I-6: count_tokens prices a whole REQUEST, so the framing constant
    # rides along on every call. Measured once and subtracted, so a
    # sum-of-parts and a single batched call describe the same string.
    overhead = measure_request_overhead(client, config.WS6B_AGENT_MODEL)
    print(f"request overhead: {overhead} tokens/request")

    texts = [p.read_text() for p in day_files]
    day_tokens = [_count(client, overhead, t) for t in texts]
    total = sum(day_tokens)

    # The I-6 consistency check, asserted rather than assumed.
    batched = count_tokens_batched(client, config.WS6B_AGENT_MODEL, texts,
                                   overhead=overhead)
    drift = abs(batched - total)
    print(f"per-file sum={total} batched={batched} drift={drift}")
    if drift > len(texts):  # <= 1 token/batch boundary is documented, not fixed
        print(f"FAIL: token accounting drift {drift} exceeds batch-boundary "
              "tolerance; the two totals describe different strings")
        return 1

    n_s = ws6b.size_prefix(day_tokens, config.WS6B_SIZES["s"])
    n_m = ws6b.size_prefix(day_tokens, config.WS6B_SIZES["m"])
    n_l = len(day_files)
    first_kept, fit_fraction = ws6b.truncate_recent(
        day_tokens, config.WS6B_TRUNC_BUDGET_TOKENS)
    print(f"sizes: s={n_s} files/{sum(day_tokens[:n_s])} tok  "
          f"m={n_m}/{sum(day_tokens[:n_m])}  l={n_l}/{total}")
    print(f"truncation: first_kept_day={first_kept} "
          f"fit_fraction={fit_fraction:.4f}")

    cum_days = ws6b.cumulative_before(day_tokens)

    # Exact cumulative position of each fact: whole days before its day, plus
    # the measured prefix of its own day up to the fact's turn. ~500 calls
    # rather than one per turn.
    def fact_cum(day_index: int, session_index: int, turn_index: int) -> int:
        day = ws6b.render_day(_parse_day(day_files[day_index]))
        prefix = _day_prefix_text(day_files[day_index], session_index, turn_index)
        return cum_days[day_index] + _count(client, overhead, prefix)

    rows, depths_by_set, sup_original_depths = [], {}, {}
    for rec in ledger:
        di, si, ti = rec["day_index"], rec["session_index"], rec["turn_index"]
        cum = fact_cum(di, si, ti)
        d_l = ws6b.depth_of(cum, total)
        d_m = ws6b.depth_of(cum, sum(day_tokens[:n_m])) if di < n_m else ""
        d_s = ws6b.depth_of(cum, sum(day_tokens[:n_s])) if di < n_s else ""

        f = ws6b.PlantedFact(**{k: v for k, v in rec.items()
                                if k in ws6b.PlantedFact.__dataclass_fields__})
        row = {
            "probe_id": rec["fact_id"], "probe_set": rec["probe_set"],
            "fact_type": rec["fact_type"], "question": ws6b.derive_question(f),
            "expected_answer": f.value, "stale_answer": f.stale_value,
            "subject": f.subject, "attribute": f.attribute,
            "source_file": day_files[di].name, "source_line": "",
            "session_id": "", "turn_id": "",
            "depth_s": d_s, "depth_m": d_m, "depth_l": round(d_l, 6),
            "cumulative_tokens_before": cum,
            "original_source_line": "", "original_depth_l": "",
            "original_in_window": "",
            "fact_in_window": di >= first_kept,
        }
        if rec["fact_type"] == "superseded":
            # For superseded probes the RECORDED depth is the revision's; the
            # original is carried separately so 4.3's guarantee is visible in
            # the published data rather than only in a test.
            o_cum = cum
            o_depth = d_l
            row["original_depth_l"] = round(o_depth, 6)
            row["original_in_window"] = di >= first_kept
            row["original_source_line"] = day_files[di].name
            sup_original_depths[rec["fact_id"]] = o_depth
            rdi = rec["revision_day_index"]
            row["source_file"] = day_files[rdi].name
            row["fact_in_window"] = rdi >= first_kept
        rows.append(row)
        depths_by_set.setdefault(rec["probe_set"], {})[rec["fact_id"]] = (
            d_s if rec["probe_set"] == "scaling" and d_s != "" else d_l)

    # ------------------------------------------------ structural validators
    text = "\n".join(texts)
    facts = [ws6b.PlantedFact(**{k: v for k, v in r.items()
                                 if k in ws6b.PlantedFact.__dataclass_fields__})
             for r in ledger]
    violations = []
    violations += ws6b.check_answer_not_in_question(facts)
    violations += ws6b.check_answer_unique_in_corpus(text, facts)
    violations += ws6b.check_distractor_density(
        text, facts, config.WS6B_MIN_DISTRACTOR_COMPONENTS,
        config.WS6B_MIN_COMPONENT_MENTIONS)
    for pset in ("scaling", "depth"):
        violations += ws6b.check_depth_bands(
            depths_by_set.get(pset, {}), config.WS6B_DEPTH_BANDS,
            config.WS6B_DEPTH_TOLERANCE)
    violations += ws6b.check_superseded_originals_in_window(
        sup_original_depths, fit_fraction,
        config.WS6B_SUPERSEDED_WINDOW_MARGIN)

    if violations:
        print(f"\nFAIL: {len(violations)} structural violation(s); "
              "no CSVs written. Nudge the generator and regenerate.\n")
        for v in violations:
            print("  -", v)
        return 1

    # ------------------------------------------------------------- outputs
    with open(OUT / "corpus_manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "day_index", "sessions", "turns", "chars",
                    "count_tokens", "cumulative_tokens", "in_s", "in_m", "in_l"])
        for i, p in enumerate(day_files):
            day = _parse_day(p)
            w.writerow([p.name, i, len(day.sessions),
                        sum(len(s.turns) for s in day.sessions),
                        len(texts[i]), day_tokens[i], cum_days[i],
                        i < n_s, i < n_m, True])

    with open(OUT / "corpus_stats.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["size", "files", "count_tokens", "target_tokens",
                    "chars", "fit_fraction", "first_kept_day"])
        for size, n in (("s", n_s), ("m", n_m), ("l", n_l)):
            w.writerow([size, n, sum(day_tokens[:n]), config.WS6B_SIZES[size],
                        sum(len(t) for t in texts[:n]),
                        fit_fraction if size == "l" else 1.0,
                        first_kept if size == "l" else 0])

    with open(OUT / "probes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for p in (OUT / "corpus_manifest.csv", OUT / "corpus_stats.csv",
              OUT / "probes.csv"):
        manifest.record(p)
    for p in day_files + [CORPUS / "facts.json"]:
        manifest.record(p)

    measured_cpt = sum(len(t) for t in texts) / total
    print(f"\nOK. measured CHARS_PER_TOKEN = {measured_cpt:.4f}")
    return 0


# --------------------------------------------------------------- parsing
# The corpus is generated, so it is parsed back with the same structure the
# generator wrote rather than with a general markdown parser.

def _parse_day(path: Path) -> ws6b.Day:
    sessions, cur_turns, cur_time = [], [], None
    turn = None
    for line in path.read_text().splitlines():
        if line.startswith("## Session "):
            if cur_time is not None:
                if turn:
                    cur_turns.append(turn)
                sessions.append(ws6b.Session(cur_time, tuple(cur_turns)))
            cur_time, cur_turns, turn = line[len("## Session "):].strip(), [], None
        elif line.startswith("### "):
            if turn:
                cur_turns.append(turn)
            turn = ws6b.Turn(line[4:].strip(), "", "", ())
        elif line.startswith("- ") and turn is not None:
            turn = ws6b.Turn(turn.time_hhmm, turn.session_id, turn.turn_id,
                             turn.bullets + (line[2:],))
    if turn:
        cur_turns.append(turn)
    if cur_time is not None:
        sessions.append(ws6b.Session(cur_time, tuple(cur_turns)))
    return ws6b.Day(path.stem, tuple(sessions))


def _day_prefix_text(path: Path, session_index: int, turn_index: int) -> str:
    """The day file's text up to (not including) the given turn."""
    day = _parse_day(path)
    parts = [""]
    for si, session in enumerate(day.sessions):
        if si > session_index:
            break
        parts.append(f"## Session {session.time_hhmm}\n")
        for ti, t in enumerate(session.turns):
            if si == session_index and ti >= turn_index:
                break
            parts.append(ws6b.render_turn(t) + "\n")
    return "\n".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
