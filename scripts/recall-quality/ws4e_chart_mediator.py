#!/usr/bin/env python
"""results/ws4e/chart_mediator_by_depth.png -- the mediation chain across
all four swept depths, in one figure. $0.

Follows results/ws4/chart_mediator.png's own house style (read from
notebooks/04_recall_vs_quality.ipynb, the chart_quality_vs_recall.png and
chart_mediator.png cells): dpi=150, one row of panels sharing an x-axis of
recall@10, INK/INK2/GRID colors, errorbar markers with a white edge, a
family-style connecting line at alpha 0.5, top/right spines removed, a
shared left-hand ylabel, and fig.tight_layout() before saving. Generated
from a script rather than a notebook cell so it is reproducible with one
command (task 10 asked for this explicitly).

results/ws4/chart_mediator.png has one point per ARM at a single fixed
k=10; this chart has one LINE per DEPTH k, each with one point per arm, on
an x-axis that is the arm's fixed recall@10 identity (spec 6.4's one-basis
rule -- it must not move with k). This is why the coloring axis here is
depth, not the family axis of the WS4 charts: the two charts encode
different treatments of an otherwise identical visual grammar.

⚠️ Composition-artifact correction (post-hoc, same review pass as
composition_check.csv's fix for signal(k)): the conversion panel's MOVING
series (`conversion` in mediator_by_k.csv) conditions on presence@k, and
presence@k is nested/monotone in k -- so part of the apparent conversion
fall is Simpson's paradox, not a real drop. The panel now also draws the
FIXED-set series (`fixed_conversion` in conversion_composition_check.csv,
fixed = present at k=1, held identical at every k), dashed with open
markers, so the panel cannot be read as "conversion falls with depth" on
its own -- see that CSV and results/recall-vs-quality.md.

Design: results/recall-vs-quality.md

Run:
    PYTHONPATH=. .venv/bin/python scripts/recall-quality/ws4e_chart_mediator.py
"""

import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchlib import manifest  # noqa: E402
from benchlib.config import WS4E_ARMS, WS4E_K_VALUES, ws4e_dir  # noqa: E402

INK, INK2, GRID = "#333333", "#666666", "#e5e5e5"

# Same categorical palette WS4's own charts use for family, reused here for
# DEPTH -- the axis this chart is actually about (house style, not a new
# palette invented for this figure).
K_COLOR = {1: "#eb6834", 3: "#2a78d6", 5: "#1baf7a", 10: "#333333"}
K_MARKER = {1: "D", 3: "o", 5: "s", 10: "*"}

PANELS = [
    ("presence_at_k", "gold present at k"),
    ("em", "exact match"),
    ("judge_accuracy", "judge accuracy"),
    ("conversion", "conversion  P(correct | present)"),
]

LABEL = lambda a: a.replace("sq8_np", "sq8@")  # noqa: E731 -- matches WS4's "sq8@N" naming


def main():
    out_dir = ws4e_dir()
    df = pd.read_csv(out_dir / "mediator_by_k.csv")
    assert set(df.arm) == set(WS4E_ARMS)
    assert set(df.k) == set(WS4E_K_VALUES)
    n = int(df.n.iloc[0])
    assert (df.n == n).all()

    # Composition-artifact correction: the fixed-set conversion series,
    # read from the diagnostic this fix added rather than recomputed here.
    comp = pd.read_csv(out_dir / "conversion_composition_check.csv")
    assert set(comp.arm) == set(WS4E_ARMS)
    assert set(comp.k) == set(WS4E_K_VALUES)
    df = df.merge(
        comp[["arm", "k", "fixed_conversion"]], on=["arm", "k"], how="left")

    # recall@10 is the arm's fixed identity (spec 6.4) -- assert it really
    # is the same x-position at every depth before trusting the chart.
    for arm in WS4E_ARMS:
        vals = df.loc[df.arm == arm, "recall_at_10"].round(9).unique()
        assert len(vals) == 1, f"{arm}: recall@10 moves with k: {vals}"

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), dpi=150, sharex=True)
    for ax, (metric, title) in zip(axes, PANELS):
        for k in WS4E_K_VALUES:
            g = df[df.k == k].sort_values("recall_at_10")
            g = g.dropna(subset=[metric, f"{metric}_ci_lo", f"{metric}_ci_hi"])
            if g.empty:
                continue
            c, mk = K_COLOR[k], K_MARKER[k]
            yerr = [g[metric] - g[f"{metric}_ci_lo"],
                    g[f"{metric}_ci_hi"] - g[metric]]
            ax.errorbar(g.recall_at_10, g[metric], yerr=yerr, fmt=mk, color=c,
                        ms=9 if mk != "*" else 14, mec="white", mew=1,
                        elinewidth=1, capsize=2, zorder=4, label=f"k = {k}")
            ax.plot(g.recall_at_10, g[metric], "-", color=c, lw=1, alpha=0.5,
                    zorder=2)
            if metric == "conversion":
                # Composition-artifact correction: the FIXED-set series
                # (presence held at k=1, per arm) alongside the moving one,
                # dashed with an open marker, same color per depth -- so
                # this panel cannot be read as "conversion falls with
                # depth" without also showing that holding the question
                # set fixed removes almost all of the fall (§10,
                # conversion_composition_check.csv).
                gf = g.dropna(subset=["fixed_conversion"])
                if not gf.empty:
                    ax.plot(gf.recall_at_10, gf.fixed_conversion, "--",
                           color=c, lw=1, alpha=0.6, zorder=3)
                    ax.scatter(gf.recall_at_10, gf.fixed_conversion,
                              marker=mk, s=60 if mk != "*" else 140,
                              facecolors="white", edgecolors=c, linewidths=1.3,
                              zorder=3)
                    if k == WS4E_K_VALUES[0]:
                        # moving/fixed key: only meaningful in this panel,
                        # so it lives here rather than stealing a row from
                        # the shared top legend (which now carries only the
                        # depth colour key).
                        mv_fx_handles = [
                            plt.Line2D([], [], marker="o", ls="-", color=INK,
                                       ms=6, mfc=INK, mec="white",
                                       label="moving"),
                            plt.Line2D([], [], marker="o", ls="--", color=INK,
                                       ms=6, mfc="white", mec=INK,
                                       label="fixed at k=1"),
                        ]
                        ax.legend(handles=mv_fx_handles, loc="lower left",
                                  frameon=False, fontsize=7, handlelength=1.8,
                                  borderaxespad=0.3, labelspacing=0.3)
            if metric == "judge_accuracy" and k == 10:
                # One arm-name annotation pass on the judge-accuracy panel,
                # off the k=10 line only (widest recall spread, so points
                # are least likely to collide) -- avoids stacking four
                # near-identical labels per arm across every depth's line.
                for i, (_, r) in enumerate(g.iterrows()):
                    ax.annotate(LABEL(r.arm), (r.recall_at_10, r[metric]),
                               textcoords="offset points",
                               xytext=(0, 11 if i % 2 == 0 else -15),
                               fontsize=7, color=INK2, ha="center", zorder=5)
        ax.set_title(title, color=INK, fontsize=10.5)
        ax.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_xlabel("recall@10 (arm identity, fixed across k)", color=INK,
                      fontsize=8.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)

    # ⚠️ Conversion is the panel carrying the MOVING fall (0.771 -> 0.674
    # on sq8_np512, k=1 -> k=10) -- a y-scale that autoscaled to the whole
    # 0-1 range would flatten it into illegibility. Fixed to bracket every
    # committed conversion value in mediator_by_k.csv AND every
    # fixed_conversion value in conversion_composition_check.csv with
    # headroom, not to the data's own tight range.
    conv_ax = axes[[m for m, _ in PANELS].index("conversion")]
    conv_ax.set_ylim(0.55, 0.85)

    axes[0].set_ylabel(f"mean over {n} questions, 95% CI", color=INK)
    # Depth colour key only -- the moving/fixed key now lives inside the
    # conversion panel (it's the only panel that distinction applies to),
    # so this legend is a single row and no longer needs to be wide enough
    # to hold six long-labelled handles.
    handles = [plt.Line2D([], [], marker=K_MARKER[k], ls="-", color=K_COLOR[k],
                          ms=8 if K_MARKER[k] != "*" else 12, label=f"k = {k}")
              for k in WS4E_K_VALUES]
    fig.legend(handles=handles, frameon=False, fontsize=9, loc="upper center",
              ncol=len(WS4E_K_VALUES), bbox_to_anchor=(0.5, 0.975))
    fig.suptitle("Depth surfaces more golds, and the ones it surfaces "
                 "convert worse — "
                 "sq8_np1 · sq8_np4 · sq8_np48 · sq8_np512",
                 color=INK, fontsize=11, y=1.03)
    caption = textwrap.fill(
        f"n = {n} questions (WS4d's primary eligible stratum), 4 SQ8 arms "
        "(sq8_np1/np4/np48/np512), k ∈ {1, 3, 5, 10}. CIs are "
        "paired-bootstrap over questions, one shared question-index draw "
        "per replicate (WS4B_CHANGEPOINT_BOOT_N = 10,000, seed 42). The "
        "conversion panel's dashed/open-marker series holds the presence "
        "set fixed at k=1 (per arm) -- on that fixed set conversion does "
        "not fall with depth (paired CIs include zero for every arm); the "
        "moving (solid) series falls only because presence@k is nested in "
        "k, the same composition artifact section 4 documents for "
        "signal(k). Sources: results/ws4e/mediator_by_k.csv, "
        "results/ws4e/conversion_composition_check.csv.",
        width=225)
    fig.text(0.5, 0.01, caption, ha="center", va="top", fontsize=8,
             color=INK2, linespacing=1.5)
    fig.tight_layout()
    fig.savefig(out_dir / "chart_mediator_by_depth.png", bbox_inches="tight")
    plt.close(fig)
    manifest.record(out_dir / "chart_mediator_by_depth.png")
    print("wrote", out_dir / "chart_mediator_by_depth.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
