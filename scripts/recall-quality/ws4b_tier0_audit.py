#!/usr/bin/env python
"""WS4b Tier 0 §3.3 -- apply the committed blind-audit labels and resolve T0-C.

Rubric: reproduced in results/recall-vs-quality.md, Appendix B -- fixed BEFORE
the candidates were read in bulk. Auditor: Claude Opus 5, in-session,
2026-09-11 -- the same model that formulated the hypothesis T0-C tests. That
conflict is disclosed in the rubric and in results/recall-vs-quality.md; the result is PROVISIONAL pending human
spot-check.

The labels below were assigned from a blind sheet carrying only question,
golds, the distinct answers pooled across every arm that erred, and the judge's
reason -- no arm label, no recall, no accuracy -- and the share was not
computed until every row was labelled.

Run:  python scripts/recall-quality/ws4b_tier0_audit.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib.config import WS4B_T0C_STALE_SHARE, ws4b_dir

AUDITOR = "claude-opus-5 (in-session, 2026-09-11); PROVISIONAL, human spot-check pending"

# qid -> (label, one-line justification)
# Labels: stale_gold | ambiguous_question | judge_strict | model_error | unresolved
LABELS = {
    71:   ("model_error", "natural reading is the 111-game overall streak; 126 is a different streak"),
    125:  ("model_error", "Rules of Engagement ran 7 seasons; answer says 6"),
    129:  ("stale_gold", "Patty Murray succeeded Hatch as Senate president pro tempore in 2023"),
    151:  ("ambiguous_question", "US declared war on both Germany (Apr 1917) and Austria-Hungary (Dec 1917); gold lists only one"),
    217:  ("ambiguous_question", "'most gold medals' underdetermines nation (Norway) vs individual (Daehlie)"),
    350:  ("model_error", "Sacrifice is on Sleeping with the Past; Two Rooms is a tribute album"),
    400:  ("ambiguous_question", "English Bill of Rights 1689 vs US Bill of Rights 1791"),
    453:  ("ambiguous_question", "'where located' admits origin (left coronary artery) or course (interventricular sulcus)"),
    488:  ("stale_gold", "Pandey (2021) and Goel (2022) succeeded Rawat as Indian election commissioner"),
    496:  ("ambiguous_question", "underdetermines sport and championship; gold is 1989 NCAA basketball"),
    559:  ("model_error", "the sweet leavened Romanian Easter bread is cozonac; pasca is the cheese-filled one"),
    614:  ("ambiguous_question", "no adaptation specified; gold is the 2010 film, answers cite 1985/1999 versions"),
    644:  ("ambiguous_question", "'where' read as distance from the lander rather than which landing site"),
    697:  ("stale_gold", "the Data Protection Act 2018 superseded the 1998 Act before the corpus snapshot"),
    715:  ("ambiguous_question", "gradual process; 1967 (first vernacular) vs 1969 (Novus Ordo) both defensible"),
    726:  ("ambiguous_question", "in-text name Scheria vs its conventional real-world identification as Corfu"),
    845:  ("ambiguous_question", "'first declaration of human rights' reads as Cyrus Cylinder or the 1948 UDHR"),
    868:  ("ambiguous_question", "'the new' is time-relative; answer names a later production than the 2017 film"),
    891:  ("ambiguous_question", "'where does power come from' admits the utility (PREPA) or the fuel source"),
    896:  ("model_error", "question names the F-150 specifically (1975); 1948 is the first F-Series"),
    901:  ("ambiguous_question", "'largest city park' admits single park vs park system; gold is a state park"),
    907:  ("model_error", "the wedding song is I Write Sins Not Tragedies; answers name unrelated tracks"),
    944:  ("stale_gold", "Vande Bharat (2019) superseded Gatimaan as India's fastest train"),
    966:  ("unresolved", "RUBRIC GAP: gold 'Bacon' appears incorrect; Rousseau is the standard answer"),
    1001: ("ambiguous_question", "'last drafted out of high school' admits pre-2005-rule (Amir Johnson) or Thon Maker 2016"),
    1005: ("stale_gold", "Stewart-Cousins succeeded Flanagan as NY Senate majority leader in 2019"),
    1028: ("stale_gold", "time-relative; corpus documents Beijing 2022, gold is PyeongChang 2018"),
    1032: ("unresolved", "RUBRIC GAP: gold 'continental drift' restates the question; mantle convection is the process"),
    1056: ("model_error", "R. Kelly is not the highest-selling R&B artist under any reading"),
    1065: ("stale_gold", "time-relative return date; answers name later seasons than the 2017 gold"),
    1103: ("ambiguous_question", "Cespedes led the 1868 Ten Years' War; gold names the 1895 leaders"),
    1106: ("model_error", "Colby cheese is named for Colby, Wisconsin"),
    1125: ("stale_gold", "'next two' is time-relative; corpus documents Milan-Cortina 2026 and 2030 TBD"),
    1148: ("stale_gold", "time-relative; McIntyre won the 2020 Rumble, after the 2018 gold"),
    1279: ("stale_gold", "Lillard/Brown extensions postdate the 2017 Curry contract in the gold"),
    1282: ("stale_gold", "Brady passed Manning for career passing yards in 2021"),
    1313: ("stale_gold", "Astros' most recent World Series at corpus time was 2021 vs Atlanta"),
    1322: ("ambiguous_question", "multiple songs share the title; one answer is the 2012 Guetta track"),
    1370: ("stale_gold", "time-relative season dates; answers give 2021, gold is 2018"),
    1376: ("model_error", "Daniel Manche is not among the many credited Tom actors"),
    1433: ("stale_gold", "'this year' is time-relative; gold is the 2018 season"),
    1480: ("ambiguous_question", "first T20 overall vs first men's/women's international; gold is Lord's"),
    1496: ("ambiguous_question", "gold names a region within Germany; 'Germanic culture' fits the natural reading"),
    1501: ("ambiguous_question", "genuinely disputed; 'Nearer, My God, to Thee' is the mainstream account"),
    1538: ("model_error", "the record is 126 consecutive 20-point games; answers give 20 and 31"),
    1542: ("model_error", "gives the season-15 premiere date, not the April 2016 end date"),
    1660: ("model_error", "LA Memorial Coliseum is not the largest NFL stadium under any reading"),
    1699: ("ambiguous_question", "no year given; answers correctly note multiple winners"),
    1770: ("model_error", "1890s precedes the post-WWI period the gold identifies"),
    1804: ("ambiguous_question", "'super bowl 2018' reads as calendar-year LII or the 2018-season LIII"),
    1811: ("stale_gold", "'right now' is time-relative; Wade retired in 2019, before the corpus snapshot"),
    1837: ("ambiguous_question", "many industrial cities sit on the canal; Herne is one and the gold list is partial"),
    1842: ("ambiguous_question", "'introduced to the public' admits 1942 military, 1945 US civilian, or 1946"),
    1916: ("ambiguous_question", "Ohio State was admitted in 1912 and began Big Ten play in 1913"),
    1940: ("judge_strict", "an uncaught decline: the answer explains the office then says it does not know"),
    1952: ("model_error", "the classification in use is the Henry system; Galton devised an earlier one"),
    1964: ("stale_gold", "the series had grown from 12 to 18 books by the corpus snapshot"),
    1999: ("judge_strict", "'cohort study' is a correct characterisation; gold says prospective/long-term"),
    2003: ("stale_gold", "Curry passed Ray Allen for career three-pointers in 2021"),
    2064: ("ambiguous_question", "city aggregate vs metropolitan-area population"),
    2070: ("model_error", "the song is Hank Williams'; Sonny Burgess is unrelated"),
    2083: ("ambiguous_question", "'result' admits military outcome (gold) or constitutional consequence (answer)"),
    2092: ("ambiguous_question", "'where does the term come from' admits etymology or historical referent"),
    2108: ("model_error", "2009 is not a post-2018 update of the 2007 gold; both predate NQ's vintage"),
    2133: ("ambiguous_question", "'oldest still wrestling' is time-relative and depends on who counts as active"),
    2135: ("stale_gold", "Dead Reckoning Part One (2023) succeeded Fallout as the latest film"),
    2145: ("stale_gold", "time-relative; Maroon 5 played the 2019 halftime show, after the 2018 gold"),
    2167: ("stale_gold", "'next deadpool movie' is time-relative; the 2024 film postdates the gold"),
    2170: ("unresolved", "RUBRIC GAP: gold 78% is factually wrong; ~71% is correct and the answers give it"),
    2211: ("ambiguous_question", "the 2017 Cricket World Cup was the Women's, won by England; gold names the Champions Trophy"),
    2213: ("ambiguous_question", "ultimate source (Sierra snowmelt) vs the aqueduct's intake (the Delta)"),
    2218: ("model_error", "question asks what leaders were called; the answer names a person"),
    2275: ("ambiguous_question", "1945 Ivy Group Agreement vs the mid-1950s formal league"),
    2277: ("unresolved", "RUBRIC GAP: gold 'Einstein' appears incorrect for electrons; de Broglie proposed matter waves"),
    2284: ("model_error", "per glucose the answer is 6; the answer gives 2 per acetyl-CoA"),
    2295: ("ambiguous_question", "contested textbook answer; wealth maximisation is the standard modern statement"),
    2310: ("unresolved", "RUBRIC GAP: gold names the butler (Carson); Thomas Barrow is the under-butler"),
    2404: ("model_error", "states the gold then retracts it twice; the judge's rule covers this explicitly"),
    2504: ("ambiguous_question", "the Page Act 1875 is commonly cited as the first restrictive federal immigration law"),
    2514: ("stale_gold", "time-relative; corpus documents Paris 2024 and Milan-Cortina 2026"),
    2531: ("ambiguous_question", "'early voyage' fits Cabot 1497 as readily as Cartier"),
    2571: ("ambiguous_question", "which Peter Pan film; gold is 2003, answers give the 1953 Disney version"),
    2576: ("model_error", "a dipeptide forms by condensation into a peptide bond, not esterification"),
    2611: ("ambiguous_question", "Devastator is Vader's Star Destroyer, Executor his Super Star Destroyer flagship"),
    2636: ("model_error", "Kansas leads conference championships; Purdue does not"),
    2788: ("ambiguous_question", "no year given; many players have won the title"),
    2851: ("ambiguous_question", "the World Cup Golden Boot tiebreaker is assists; gold says shared"),
    2873: ("ambiguous_question", "2009 and 2010 both had white Christmases; 2010 was the widespread one"),
    2921: ("stale_gold", "the Buccaneers' 2020 title under Brady postdates the 2002 Brad Johnson gold"),
    2948: ("ambiguous_question", "founding at Dhaka 1906 vs first annual session at Karachi 1907"),
    3081: ("ambiguous_question", "written 1936, published 1937; the question says 'write'"),
    3088: ("model_error", "Nov 2017 is the season-5 part-one premiere, not the second half"),
    3118: ("model_error", "answers the team event rather than the individual medals, and garbles it"),
    3119: ("ambiguous_question", "'the 1500s' reads as the decade (Bayezid II) or the 16th century (Selim I)"),
    3135: ("ambiguous_question", "the House was directly elected originally; senators only after the 17th Amendment"),
    3211: ("ambiguous_question", "which reign; the New Day first won in 2015, gold says 2016"),
    3224: ("judge_strict", "asserts the gold fact (Medicaid) with an incorrect addition (Medicare)"),
    3239: ("ambiguous_question", "1619 House of Burgesses is the standard answer for self-government's start"),
    3312: ("stale_gold", "American Dream opened 2019 and displaced the Florida mall in the ranking"),
    3417: ("ambiguous_question", "'on riverdale' reads as the series (Jughead) vs the comics pairing (Archie)"),
    3434: ("model_error", "Thomas Mundy Peterson is the documented first; Turnbow is a 1960s figure"),
    3453: ("ambiguous_question", "gold names one instance (Mariana Trench); the answer gives the general distribution"),
    3480: ("ambiguous_question", "transition begins in the season-3 finale and completes in season 4"),
    3497: ("judge_strict", "Allegiant Stadium is in Paradise NV, universally described as Las Vegas"),
    3552: ("model_error", "question asks the episode; the answer gives the series number"),
    3570: ("ambiguous_question", "'who is Rose' admits the character or the actress"),
    3595: ("unresolved", "RUBRIC GAP: gold names a village, not a city; Derby is the standard answer for a city"),
}

NUMERATOR = ("stale_gold", "ambiguous_question")


def main() -> None:
    out = ws4b_dir()
    cand = pd.read_csv(out / "tier0_staleness_candidates.csv")

    missing = set(cand.qid) - set(LABELS)
    extra = set(LABELS) - set(cand.qid)
    if missing or extra:
        sys.exit(f"label set does not match candidates: missing={sorted(missing)} "
                 f"extra={sorted(extra)}")

    cand["label"] = cand.qid.map(lambda q: LABELS[q][0])
    cand["justification"] = cand.qid.map(lambda q: LABELS[q][1])
    cand["auditor"] = AUDITOR
    cand.to_csv(out / "tier0_staleness_labelled.csv", index=False)

    counts = cand.label.value_counts()
    n = len(cand)
    num = int(counts.reindex(NUMERATOR).fillna(0).sum())
    share = num / n
    fires = share >= WS4B_T0C_STALE_SHARE

    summary = pd.DataFrame([
        {"label": k, "n": int(v), "share": round(v / n, 4),
         "in_numerator": k in NUMERATOR}
        for k, v in counts.items()
    ]).sort_values("n", ascending=False)
    summary.to_csv(out / "tier0_audit_summary.csv", index=False)

    branches = pd.read_csv(out / "tier0_branches.csv")
    branches.loc[branches.branch == "T0-C", "fires"] = bool(fires)
    branches.loc[branches.branch == "T0-C", "evidence"] = (
        f"{num}/{n} = {share:.3f} stale_gold+ambiguous_question vs threshold "
        f"{WS4B_T0C_STALE_SHARE}; auditor {AUDITOR}"
    )
    branches.to_csv(out / "tier0_branches.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nnumerator (stale_gold + ambiguous_question) = {num}/{n} = {share:.4f}")
    print(f"threshold WS4B_T0C_STALE_SHARE = {WS4B_T0C_STALE_SHARE}")
    print(f"\nT0-C {'FIRES' if fires else 'does not fire'}")
    print("\nwrote tier0_staleness_labelled.csv, tier0_audit_summary.csv; "
          "updated tier0_branches.csv")


if __name__ == "__main__":
    main()
