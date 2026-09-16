#!/usr/bin/env python3
"""Generate WS6b's synthetic agent history.

THIS SCRIPT SHIPS IN THE PUBLIC REPO. The corpus it produces is committed
alongside it, and tests/code_retrieval/test_memory.py asserts that re-running it reproduces
those bytes exactly. Three rules make that hold, and breaking any one of them
silently breaks reproduction:

  1. ONE explicit random.Random(SEED) instance, threaded through every call.
     Never module-level random, never uuid.uuid4(), never time-based values.
  2. Never iterate a set or an unordered dict.
  3. No dependence on filesystem ordering -- day files are generated from a
     date sequence, not from a directory listing.

Design: results/code-retrieval.md

Usage:
    python scripts/code-retrieval/generate_ws6b_corpus.py --out data/ws6b_corpus \
        --target-tokens 1200000
"""

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import config  # noqa: E402
from benchlib.code_retrieval import memory as ws6b  # noqa: E402

# MEASURED, not assumed. A chars/4 estimate was wrong by 1.6x here, which put
# the whole corpus at 2.30M tokens instead of 1.2M and moved every planted
# fact out of its depth band -- caught by the structural gates in
# build_ws6b_probes.py before any arm was run.
#
# 2.5265 chars/token is what count_tokens reports for this corpus. For
# comparison, the REAL memsearch memory in this repository's .memsearch/memory
# measures 2.3769 -- so the density is a property of the format (backticked
# identifiers, paths, and two UUID anchors per turn), not an artifact of the
# coined vocabulary. The generated corpus is very slightly LESS token-dense
# than the real thing, which is the conservative direction.
CHARS_PER_TOKEN = 2.5265

# Only 4-digit values are used as answers. A 4-digit decimal string cannot be
# a proper substring of another 4-digit string, so check_answer_unique_in_corpus
# cannot be defeated by "271" hiding inside "4271" -- which would let an arm
# retrieve the wrong chunk and still score.
_ANSWER_VALUES = tuple(v for v in ws6b.VALUE_POOL if v >= 1000)

# Filler sentence templates. These carry the conversational noise that
# check_distractor_density requires (>= WS6B_MIN_COMPONENT_MENTIONS mentions
# of every probe subject), and make the history read like work rather than
# like a list of facts.
_FILLER = (
    "Reviewed `{c}` behaviour under sustained load and confirmed the observed "
    "latency came from `{d}` rather than from `{c}` itself; no action taken "
    "beyond a note for the next planning round.",
    "Traced a slow path through `{c}` and found the delay was already present "
    "before the `{m}` rollout, so the rollout was cleared as a cause and the "
    "investigation moved to `{d}`.",
    "{p} walked the team through the `{c}` change set, flagged two places where "
    "error handling diverges from `{d}`, and agreed to leave both alone until "
    "the interface settles.",
    "Rolled `{c}` forward to the current build without incident; the smoke "
    "checks passed on the first attempt and the `{m}` integration stayed green "
    "throughout.",
    "Paged through several days of `{c}` logs looking for the earlier stall. "
    "Nothing matched the signature, and {p} suggested the report may have been "
    "about `{d}` instead.",
    "Deferred the `{c}` cleanup until after the current milestone. The work is "
    "understood and scoped, but it touches `{d}` in three places and nobody "
    "wanted that landing mid-cycle.",
    "Compared `{c}` against `{d}` on the same workload. Throughput was broadly "
    "similar; the difference showed up only in tail latency, and only when the "
    "`{m}` path was exercised.",
    "{p} reported that `{c}` looked healthy across the weekend window. One "
    "transient alert fired and cleared on its own, which matches what `{d}` "
    "did last month.",
    "Backed out an experimental tweak to `{c}` after the numbers moved in the "
    "wrong direction. The revert was clean and `{m}` was unaffected, so no "
    "follow-up is needed.",
    "Spent the morning reading `{c}` end to end and taking notes. No changes "
    "yet -- the goal was to understand how it hands off to `{d}` before "
    "touching either of them.",
    "Cleared a stale alert on `{c}` that had been firing since Tuesday. The "
    "underlying condition resolved itself when `{m}` was restarted, and {p} "
    "confirmed nothing downstream noticed.",
    "Discussed whether `{c}` should adopt the `{m}` pattern the way `{d}` "
    "already has. No decision -- the tradeoff depends on work that has not "
    "been scheduled yet.",
    "Checked `{c}` against the staging dataset and saw no drift from the "
    "previous run. {p} noted the same held for `{d}`, so the discrepancy is "
    "probably in the harness.",
    "Re-ran the `{c}` verification suite after the `{m}` rollout. Everything "
    "passed, though two cases took noticeably longer than they did before the "
    "`{d}` change landed.",
    "Skimmed the `{c}` change history looking for the regression window. The "
    "likely candidates are all in the same week, and all of them also touch "
    "`{d}`, which complicates bisecting.",
    "{p} asked whether `{c}` needed a second reviewer before merging. It did "
    "not -- the change is confined to one file and does not alter how `{d}` "
    "consumes it.",
    "Filed a note about `{c}` for the next planning round. The gist is that it "
    "and `{d}` have grown apart enough that sharing the `{m}` path is now more "
    "cost than benefit.",
    "Left `{c}` untouched this week; the queue was entirely on `{d}`. Worth "
    "revisiting once the `{m}` work is done, since the two share more state "
    "than they should.",
    "Confirmed `{c}` still passes its smoke checks after the `{d}` change. {p} "
    "double-checked the `{m}` interaction by hand because the automated "
    "coverage there is thin.",
    "Noted that `{c}` and `{d}` disagree about retry semantics: one gives up "
    "after the first failure, the other keeps going. Neither is obviously "
    "wrong, so it was written down rather than fixed.",
)


def _weekdays(start_iso: str):
    """Yield ISO dates, weekdays only, forever."""
    d = dt.date.fromisoformat(start_iso)
    while True:
        if d.weekday() < 5:
            yield d.isoformat()
        d += dt.timedelta(days=1)


def _partition(rng, lex):
    """Carve the lexicon into disjoint roles.

    Disjointness is load-bearing. If an answer mechanism could also appear as
    a rejected alternative or as filler, check_answer_unique_in_corpus would
    fail -- or worse, pass while an arm scored by retrieving the wrong chunk.
    """
    n_facts = sum(config.WS6B_N_PROBES.values())
    return {
        "probe_subjects": lex.components[:n_facts],
        "filler_components": lex.components[n_facts:n_facts + 60],
        "distractor_components": lex.components[n_facts + 60:n_facts + 90],
        "probe_params": lex.params[:32],
        "filler_params": lex.params[32:],
        "answer_mechanisms": lex.mechanisms[:12],
        "alt_mechanisms": lex.mechanisms[12:24],
        "filler_mechanisms": lex.mechanisms[24:],
        "answer_persons": lex.persons[:12],
        "filler_persons": lex.persons[12:],
    }


def build_facts(rng, parts):
    """The 38 planted facts, with their probe-set membership.

    Types per set: scaling 4/4/4 decision/value/owner, depth 6/6/6, and all
    eight superseded facts in the superseded set. Questions are derived
    mechanically from these triples (spec 3.4) -- nobody chooses which
    questions to ask, which is what makes WS6a's authorship bias absent
    rather than merely mitigated.
    """
    values = list(_ANSWER_VALUES)
    rng.shuffle(values)
    vi = iter(values)

    subjects = list(parts["probe_subjects"])
    params = list(parts["probe_params"])
    mechs = list(parts["answer_mechanisms"])
    alts = list(parts["alt_mechanisms"])
    persons = list(parts["answer_persons"])

    facts, si, pi, mi, ai, pei = [], 0, 0, 0, 0, 0
    plan = ([("scaling", t) for t in ("decision", "value", "owner") for _ in range(4)]
            + [("depth", t) for t in ("decision", "value", "owner") for _ in range(6)]
            + [("superseded", "superseded")] * config.WS6B_N_PROBES["superseded"])

    for i, (probe_set, ftype) in enumerate(plan):
        subject, si = subjects[si], si + 1
        if ftype == "decision":
            f = ws6b.PlantedFact(f"f{i:02d}", ftype, subject, "", mechs[mi],
                                 alternative=alts[ai])
            mi, ai = mi + 1, ai + 1
        elif ftype == "value":
            f = ws6b.PlantedFact(f"f{i:02d}", ftype, subject, params[pi],
                                 str(next(vi)))
            pi += 1
        elif ftype == "owner":
            f = ws6b.PlantedFact(f"f{i:02d}", ftype, subject, "", persons[pei])
            pei += 1
        else:
            f = ws6b.PlantedFact(f"f{i:02d}", ftype, subject, params[pi],
                                 str(next(vi)), stale_value=str(next(vi)))
            pi += 1
        facts.append((probe_set, f))
    return facts


def required_bullets(rng, facts, parts):
    """Bullets the structural gates require, before any filler is added.

    Two kinds:
      - distractors: every probe parameter must be used by >= 3 OTHER
        components, or BM25 on the parameter name alone returns the one
        chunk that contains it and the benchmark measures nothing (spec 3.5).
      - subject mentions: every probe subject must appear in >= 5 non-fact
        turns as ordinary conversational noise.

    Distractor values are drawn from the 3-digit remainder of VALUE_POOL, so
    they can never collide with a 4-digit answer.
    """
    distractor_values = [v for v in ws6b.VALUE_POOL if v < 1000]
    out = []
    others = list(parts["distractor_components"])
    for _, f in facts:
        if f.attribute:
            picks = rng.sample(others, config.WS6B_MIN_DISTRACTOR_COMPONENTS + 1)
            for comp in picks:
                v = rng.choice(distractor_values)
                out.append(f"We set `{f.attribute}` on `{comp}` to {v}.")
        for _ in range(config.WS6B_MIN_COMPONENT_MENTIONS + 1):
            out.append(rng.choice(_FILLER).format(
                c=f.subject,
                d=rng.choice(parts["filler_components"]),
                p=rng.choice(parts["filler_persons"]),
                m=rng.choice(parts["filler_mechanisms"])))
    rng.shuffle(out)
    return out


def filler_bullet(rng, parts):
    return rng.choice(_FILLER).format(
        c=rng.choice(parts["filler_components"]),
        d=rng.choice(parts["filler_components"]),
        p=rng.choice(parts["filler_persons"]),
        m=rng.choice(parts["filler_mechanisms"]))


def build_skeleton(rng, parts, required, target_tokens):
    """Pass 1: the whole history with no facts in it.

    Required bullets are spread evenly through the stream rather than
    clustered, so distractors sit at every depth and cannot be retrieved as a
    block.
    """
    budget_chars = target_tokens * CHARS_PER_TOKEN
    days, chars, req_i = [], 0, 0
    n_req = len(required)
    # Spread the required bullets across roughly the whole history.
    req_every = max(1, int(budget_chars / max(1, n_req) / 110))
    emitted = 0

    dates = _weekdays(config.WS6B_CORPUS_START_DATE)
    while chars < budget_chars:
        date_iso = next(dates)
        sessions = []
        hour = 9
        for _ in range(rng.randint(2, 5)):
            hhmm = f"{hour:02d}:{rng.randint(0, 59):02d}"
            hour = min(hour + rng.randint(1, 3), 18)
            session_id = ws6b.seeded_uuid(rng)
            turns = []
            for _ in range(rng.randint(3, 8)):
                bullets = []
                for _ in range(rng.randint(3, 7)):
                    if req_i < n_req and emitted % req_every == 0:
                        bullets.append(required[req_i])
                        req_i += 1
                    else:
                        bullets.append(filler_bullet(rng, parts))
                    emitted += 1
                t_hhmm = f"{hour:02d}:{rng.randint(0, 59):02d}"
                turns.append(ws6b.Turn(t_hhmm, session_id,
                                       ws6b.seeded_uuid(rng), tuple(bullets)))
            sessions.append(ws6b.Session(hhmm, tuple(turns)))
        day = ws6b.Day(date_iso, tuple(sessions))
        days.append(day)
        chars += len(ws6b.render_day(day))

    # Any required bullets not yet placed are appended to the final day, so a
    # short run can never silently drop a structural guarantee.
    if req_i < n_req:
        leftover = required[req_i:]
        last = days[-1]
        extra = ws6b.Turn("18:59", ws6b.seeded_uuid(rng),
                          ws6b.seeded_uuid(rng), tuple(leftover))
        sess = last.sessions[-1]
        days[-1] = ws6b.Day(last.date_iso, last.sessions[:-1] + (
            ws6b.Session(sess.time_hhmm, sess.turns + (extra,)),))
    return days


def _turn_index(days):
    """Flat [(day_i, session_i, turn_i, est_tokens)] in chronological order."""
    out = []
    for di, day in enumerate(days):
        for si, session in enumerate(day.sessions):
            for ti, turn in enumerate(session.turns):
                est = int(len(ws6b.render_turn(turn)) / CHARS_PER_TOKEN)
                out.append((di, si, ti, est))
    return out


def _pick_turn(index, depths, target, used):
    """Nearest unused turn to a target depth."""
    best, best_d = None, 1e9
    for k, (_, _, _, _) in enumerate(index):
        if k in used:
            continue
        d = abs(depths[k] - target)
        if d < best_d:
            best, best_d = k, d
    used.add(best)
    return best


def place_facts(days, facts, day_token_est):
    """Pass 2: choose a turn for every fact and inject its bullet.

    Depths here are ESTIMATED from characters; build_ws6b_probes.py re-measures
    them with count_tokens and records the measured values. The estimate only
    has to be good enough to land inside the +/- 0.03 band tolerance.
    """
    index = _turn_index(days)
    tokens = [t[3] for t in index]
    total = sum(tokens)
    cum = ws6b.cumulative_before(tokens)
    depths = [c / total for c in cum]

    n_s = ws6b.size_prefix(day_token_est, config.WS6B_SIZES["s"])
    s_total = sum(day_token_est[:n_s])
    s_cut = sum(1 for k, (di, _, _, _) in enumerate(index) if di < n_s)
    s_depths = [cum[k] / s_total for k in range(s_cut)]

    used, placements = set(), {}
    bands = config.WS6B_DEPTH_BANDS
    lo, hi = config.WS6B_SUPERSEDED_ORIGINAL_BAND

    scaling = [f for ps, f in facts if ps == "scaling"]
    depth_set = [f for ps, f in facts if ps == "depth"]
    superseded = [f for ps, f in facts if ps == "superseded"]

    for i, f in enumerate(scaling):
        target = bands[i // 4]
        k = _pick_turn(index[:s_cut], s_depths, target, used)
        placements[f.fact_id] = k
    for i, f in enumerate(depth_set):
        target = bands[i // 6]
        k = _pick_turn(index, depths, target, used)
        placements[f.fact_id] = k
    revisions = {}
    for i, f in enumerate(superseded):
        target = lo + (hi - lo) * (i / max(1, len(superseded) - 1))
        placements[f.fact_id] = _pick_turn(index, depths, target, used)
        # Revision in the last third, spread out.
        rev_target = 0.70 + 0.28 * (i / max(1, len(superseded) - 1))
        revisions[f.fact_id] = _pick_turn(index, depths, rev_target, used)

    # Inject. Rebuild the immutable dataclasses rather than mutating them.
    inject: dict[int, list[str]] = {}
    for ps, f in facts:
        inject.setdefault(placements[f.fact_id], []).append(
            ws6b.render_fact_bullet(f))
        if f.fact_type == "superseded":
            inject.setdefault(revisions[f.fact_id], []).append(
                ws6b.render_revision_bullet(f))

    new_days, k = [], 0
    for day in days:
        new_sessions = []
        for session in day.sessions:
            new_turns = []
            for turn in session.turns:
                bullets = turn.bullets
                if k in inject:
                    bullets = bullets + tuple(inject[k])
                new_turns.append(ws6b.Turn(turn.time_hhmm, turn.session_id,
                                           turn.turn_id, bullets))
                k += 1
            new_sessions.append(ws6b.Session(session.time_hhmm, tuple(new_turns)))
        new_days.append(ws6b.Day(day.date_iso, tuple(new_sessions)))

    ledger = []
    for ps, f in facts:
        di, si, ti, _ = index[placements[f.fact_id]]
        rec = {"fact_id": f.fact_id, "probe_set": ps, "fact_type": f.fact_type,
               "subject": f.subject, "attribute": f.attribute, "value": f.value,
               "alternative": f.alternative, "stale_value": f.stale_value,
               "day_index": di, "session_index": si, "turn_index": ti}
        if f.fact_type == "superseded":
            rdi, rsi, rti, _ = index[revisions[f.fact_id]]
            rec |= {"revision_day_index": rdi, "revision_session_index": rsi,
                    "revision_turn_index": rti}
        ledger.append(rec)
    return new_days, ledger


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.WS6B_CORPUS_DIR))
    ap.add_argument("--target-tokens", type=int,
                    default=config.WS6B_SIZES["l"])
    args = ap.parse_args()

    # ONE rng instance, threaded everywhere. See the module docstring.
    rng = random.Random(config.SEED)
    lex = ws6b.make_lexicon(rng)
    parts = _partition(rng, lex)
    facts = build_facts(rng, parts)
    required = required_bullets(rng, facts, parts)
    days = build_skeleton(rng, parts, required, args.target_tokens)

    day_est = [int(len(ws6b.render_day(d)) / CHARS_PER_TOKEN) for d in days]
    days, ledger = place_facts(days, facts, day_est)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.md"):
        stale.unlink()
    for day in days:
        (out / f"{day.date_iso}.md").write_text(ws6b.render_day(day))
    (out / "facts.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n")

    total_chars = sum(len(ws6b.render_day(d)) for d in days)
    print(f"days={len(days)} facts={len(ledger)} chars={total_chars} "
          f"est_tokens={int(total_chars / CHARS_PER_TOKEN)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
