#!/usr/bin/env python
"""WS4b decline audit -- apply the committed blind labels and resolve D1/D2/D3.

Rubric: reproduced in results/recall-vs-quality.md, Appendix C -- fixed BEFORE
the evidence sheet was built or read. Auditor: Claude Opus 5, in-session, 2026-09-11 -- the same
model that proposed the hypothesis; PROVISIONAL pending human spot-check.

Evidence was blind: question, golds, and the +/-250-char context windows around
every word-boundary gold match in the ten passages that arm actually retrieved.
No arm label, no recall, no presence flag, no accuracy. The share was not
computed until every row was labelled.

Run:  python scripts/recall-quality/ws4b_decline_audit.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib.config import ws4b_dir

AUDITOR = "claude-opus-5 (in-session, 2026-09-11); PROVISIONAL, human spot-check pending"
RECOVERABLE = ("answerable_declined",)

# qid -> (label, justification)
L = {
    92:   ("not_supported", "gold string appears in a bibliography entry and a cascade description; no three phases given"),
    105:  ("not_supported", "lists of Spaniards naming Bernardo de Galvez; surname origin never stated"),
    206:  ("not_supported", "2018 appears in unrelated WWE contexts; the Miz & Mrs start date is absent"),
    217:  ("not_supported", "per-Games medal tables for 1994/1998/2002; no all-time total"),
    279:  ("not_supported", "Crowne Plaza GLASGOW franchise, not the Chicago hotel"),
    337:  ("answerable_declined", "the passage states the play was written prior to June 1592, matching the gold verbatim"),
    416:  ("not_supported", "Adrian Peterson passage; wrong person entirely"),
    482:  ("not_supported", "Buffon named as a goalkeeper-captain, never as the 2006 award winner"),
    484:  ("answerable_declined", "routing table article states a router builds and uses a routing table"),
    508:  ("answerable_declined", "a critic quote attributes 'Got My Mind Set on You' to George Harrison"),
    513:  ("not_supported", "covers the 2001 Finals vs the 76ers, not game 4 of the 2000 Finals"),
    559:  ("not_supported", "cozonac listed among holiday baked items; Easter specificity never established"),
    590:  ("answerable_declined", "Coroebus of Elis named as winner of the first recorded Olympic footrace"),
    621:  ("answerable_declined", "season cast lists name Meloni, Hargitay, Belzer and Florek, all golds"),
    754:  ("not_supported", "High Noon and Highway to Heaven; High Chaparral never mentioned"),
    896:  ("answerable_declined", "1975 emissions change and the F-150 response are both stated; 1975 is derivable"),
    931:  ("not_supported", "Great SANDY Desert, not the Great Victoria Desert"),
    972:  ("not_supported", "Jordan River geography; the districts east of it are never collected under a name"),
    992:  ("not_supported", "Diane Frolov on Northern Exposure; wrong show"),
    1020: ("answerable_declined", "attributes 'Let's get ready to rumble!' to Michael Buffer"),
    1056: ("not_supported", "R. Kelly as most successful of the 1990s; no all-time best-seller claim"),
    1085: ("answerable_declined", "multiple passages place the World Economic Forum at Davos"),
    1088: ("answerable_declined", "the passage describes Hook and Emma's relationship directly"),
    1089: ("not_supported", "2021 World Junior Championships; a different tournament"),
    1114: ("answerable_declined", "names Test of English as a Foreign Language (TOEFL) explicitly"),
    1148: ("ambiguous_question", "no year given; the retrieved Rumble is 2008, won by Cena"),
    1183: ("not_supported", "Johnny Lozada, not Johnny Bravo"),
    1244: ("not_supported", "Joplin listed as a Capitol Theatre performer; songwriting never mentioned"),
    1278: ("not_supported", "Dodger Stadium lists World Series years without naming opponents"),
    1293: ("ambiguous_question", "the question is barely formed; continent groupings cannot resolve it"),
    1320: ("answerable_declined", "Finnish and Hungarian are stated to be Uralic, matching the gold"),
    1370: ("ambiguous_question", "no season specified; only a start date appears, never a finish"),
    1372: ("answerable_declined", "names the famous quote by Andy Warhol about 15 minutes of fame"),
    1419: ("not_supported", "an alphabetical guest-cast list with no role mapping"),
    1433: ("not_supported", "UVA rivalry description; no losses listed"),
    1437: ("ambiguous_question", "asks about a continent; the passage describes the US federal republic"),
    1480: ("not_supported", "Lord's Test cricket firsts, nothing about T20"),
    1520: ("answerable_declined", "states Disney must pay royalties to the Slesinger family"),
    1594: ("answerable_declined", "names Hanno the Great and Hamilcar Barca as Carthage's leaders of the period"),
    1660: ("not_supported", "MetLife named as a shared stadium, not as the largest"),
    1770: ("not_supported", "'After World War I' applies to LONDON housing, not America"),
    1810: ("not_supported", "2011/2007 World Championships and 1984 Olympics; 2002 absent"),
    1874: ("answerable_declined", "states cocoa was discovered in regions of Mesoamerica"),
    1940: ("not_supported", "Naidu named as Vice President; the Rajya Sabha chairmanship link is never made"),
    1955: ("not_supported", "PRC1 localisation during interphase; no claim about time spent"),
    1960: ("not_supported", "consecutive-loss records; no all-time most-losses claim"),
    2021: ("not_supported", "Rowe, Osman and Ashley surnames; Edwards never discussed"),
    2051: ("answerable_declined", "states the first post-independence general election was 1951-52"),
    2099: ("answerable_declined", "attributes the hit 'One Bad Apple' to the Osmonds"),
    2144: ("answerable_declined", "states Phoenix and Witherspoon recorded for Walk the Line"),
    2184: ("answerable_declined", "NACCS formed 1972 and the 'By 1975' phrase both appear in context"),
    2318: ("not_supported", "mining in Canada generally; the Canadian Shield is never named"),
    2329: ("not_supported", "Filmfare Critics Award record, a different award from the National Film Awards"),
    2337: ("not_supported", "Michigan basketball article; the 1989 title is not stated in the window"),
    2379: ("not_supported", "the 2002 rule change shown is hand-checking, not zone defense"),
    2391: ("answerable_declined", "places the Temple and sanctuary of Zeus at Olympia"),
    2393: ("not_supported", "Premier League club revenues; 'the Championship' tier is not covered"),
    2401: ("not_supported", "prize bingo mechanics; the winning call is never given"),
    2430: ("answerable_declined", "states the United States topped the medal table fifteen times"),
    2458: ("answerable_declined", "states the Beatles have sold more records than any other artist"),
    2525: ("not_supported", "charbagh and Persian gardens discussed, but never Rashtrapati Bhavan"),
    2533: ("not_supported", "The Thin Red Line filming; wrong film"),
    2542: ("answerable_declined", "names pia mater as one of the two innermost meningeal layers"),
    2576: ("answerable_declined", "states a peptide bond is formed, matching the gold"),
    2587: ("not_supported", "Hubble's 1929 discovery stated; the 'father of modern cosmology' title is not"),
    2621: ("not_supported", "an Ampersand article on query strings; no general URL structure"),
    2636: ("not_supported", "Kansas's CONSECUTIVE title run, not most conference championships overall"),
    2649: ("not_supported", "A Very Supernatural Christmas filming; wrong production"),
    2665: ("not_supported", "Uruguay appears in a home-soil winners list; the inaugural World Cup is not identified"),
    2687: ("answerable_declined", "states the G-Class Wagon is by Mercedes-Benz"),
    2720: ("not_supported", "asterisk used for totals on adding machines; multiplication never stated"),
    2866: ("answerable_declined", "names the Idaho-Montana border"),
    2873: ("stale_gold", "context gives white Christmases in 2009 AND 2010; 'last time' would answer 2010, judged wrong"),
    2899: ("answerable_declined", "states Edison introduced the first working phonograph for reproducing sound"),
    2914: ("ambiguous_question", "'last 10 years' is time-relative and no per-season winners appear"),
    3095: ("answerable_declined", "states that in season 3 the family flees to Alaska"),
    3118: ("not_supported", "Mark Kondratiuk at the 2022 team event; wrong year and wrong event"),
    3135: ("answerable_declined", "quotes the constitutional text that senators are directly chosen by the people"),
    3137: ("not_supported", "Colva and Raia in South Goa; Baga never mentioned"),
    3155: ("not_supported", "NO gold match in the retrieved passages at all -- declining was correct"),
    3160: ("answerable_declined", "states citizens were called forth to vote in the comitia"),
    3191: ("not_supported", "De la Mora and Del Rosario surnames; de la Rosa never discussed"),
    3258: ("answerable_declined", "names Jonathan Cheban among the cast, matching the gold"),
    3310: ("answerable_declined", "names the physical layer or layer 1, matching the gold verbatim"),
    3317: ("answerable_declined", "names British punk poet John Cooper Clarke"),
    3320: ("not_supported", "Jenkins as all-time international scorer, not Wales v France specifically"),
    3345: ("answerable_declined", "places the crown-of-thorns starfish in the Indo-Pacific region"),
    3368: ("not_supported", "1997 refers to Russia joining the G8, not the World Economic Forum"),
    3386: ("answerable_declined", "cast list names Englund and Langenkamp, both golds"),
    3417: ("not_supported", "Archie named as a childhood crush; who Betty ends up with is not stated"),
    3480: ("answerable_declined", "the season-3 finale passage refers to the change setting up the fourth season"),
    3591: ("not_supported", "Terrell Owens named via a Madden cover; the 2018 class is never listed"),
    3606: ("not_supported", "Columbia used as a FILMING location for Yale; the characters' college is not given"),
}


def main() -> None:
    out = ws4b_dir()
    sheet = pd.read_csv(sys.argv[1]) if len(sys.argv) > 1 else None
    if sheet is None:
        sys.exit("pass the decline evidence sheet path")

    missing = set(sheet.qid) - set(L)
    extra = set(L) - set(sheet.qid)
    if missing or extra:
        sys.exit(f"label set mismatch: missing={sorted(missing)} extra={sorted(extra)}")

    sheet["label"] = sheet.qid.map(lambda q: L[q][0])
    sheet["justification"] = sheet.qid.map(lambda q: L[q][1])
    sheet["auditor"] = AUDITOR
    sheet.drop(columns=["ctx"]).to_csv(out / "decline_labelled.csv", index=False)

    n = len(sheet)
    counts = sheet.label.value_counts()
    rec = int(counts.reindex(RECOVERABLE).fillna(0).sum())
    R = rec / n
    ns = int(counts.get("not_supported", 0))

    branch = "D1" if R >= 0.50 else ("D2" if R >= 0.32 else "D3")
    summary = pd.DataFrame([
        {"label": k, "n": int(v), "share": round(v / n, 4), "recoverable": k in RECOVERABLE}
        for k, v in counts.items()
    ]).sort_values("n", ascending=False)
    summary.to_csv(out / "decline_audit_summary.csv", index=False)

    # blindness check: do the 18 seen in the error audit label differently?
    seen = sheet[sheet.seen_in_error_audit]
    unseen = sheet[~sheet.seen_in_error_audit]
    bias = pd.DataFrame([
        {"group": "seen_in_error_audit", "n": len(seen),
         "recoverable_share": round((seen.label.isin(RECOVERABLE)).mean(), 4)},
        {"group": "unseen", "n": len(unseen),
         "recoverable_share": round((unseen.label.isin(RECOVERABLE)).mean(), 4)},
    ])
    bias.to_csv(out / "decline_blindness_check.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nrecoverable R = {rec}/{n} = {R:.4f}   -> branch {branch}")
    print(f"not_supported = {ns}/{n} = {ns/n:.4f}   (secondary fires at >= 0.20: "
          f"{'YES' if ns/n >= 0.20 else 'no'})")
    print()
    print(bias.to_string(index=False))
    print("\nwrote decline_labelled.csv, decline_audit_summary.csv, "
          "decline_blindness_check.csv")


if __name__ == "__main__":
    main()
