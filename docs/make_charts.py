"""Regenerate the README's frontier charts from measured results.

Run this whenever a real (non-stub) benchmark produces new numbers, so the
charts in `docs/charts/` never drift from `README.md`'s tables the way a
hand-edited image would. Values below are transcribed from `bench_full.json`
(synthetic) and `HANDOVER.md`'s recorded Kaggle result (banking77) - this
script does not invent or re-measure anything, it only draws what was already
reported.

Palette: validated categorical slots 1/2/3 (blue/orange/aqua) from the
project's dataviz reference palette, `node scripts/validate_palette.js
"#2a78d6,#eb6834,#1baf7a" --mode light --pairs all` -> ALL CHECKS PASS. Baseline
uses neutral gray, outside the categorical set, since `rules` is a reference
floor rather than a competing "kind" of backend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a86"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dc"

OUT = Path(__file__).parent / "charts"
OUT.mkdir(exist_ok=True)

# (label, kind, accuracy, p50_ms, cost_per_million_usd, label_anchor)
# label_anchor = (ha, va, dx, dy) hand-placed per point so a 4-point chart
# never needs generic collision avoidance - there are only ever 4 of these.
SYNTHETIC = [
    ("rules (baseline)", "baseline", 0.450, 0.013, 0.0003, ("left", "center", 10, 0)),
    ("distilbert-mnli, ONNX fp32", "local-fp32", 0.383, 481.0, 12.92, ("right", "top", -12, -10)),
    ("distilbert-mnli, ONNX int8", "local-int8", 0.500, 249.1, 6.69, ("left", "bottom", 12, 12)),
    ("jev-latest (hosted)", "hosted", 0.950, 385.5, 3.59, ("left", "center", 12, 0)),
]
BANKING77 = [
    ("rules (baseline)", "baseline", 0.293, 0.037, 0.001, ("left", "center", 10, 0)),
    ("distilbert-mnli, torch/CUDA fp32", "local-fp32", 0.193, 62.7, 1.68, ("right", "bottom", -12, 12)),
    ("distilbert-mnli, ONNX int8/CPU", "local-int8", 0.167, 774.0, 20.8, ("left", "top", 12, -12)),
    ("jev-latest (hosted)", "hosted", 0.790, 391.0, 57.4, ("left", "center", 12, 0)),
]
COLOR = {"baseline": GRAY, "local-fp32": ORANGE, "local-int8": AQUA, "hosted": BLUE}
KIND_LABEL = {
    "baseline": "keyword baseline",
    "local-fp32": "local model, fp32",
    "local-int8": "local model, int8",
    "hosted": "hosted (Jev)",
}


def draw(points: list[tuple], title: str, subtitle: str, out_path: Path) -> None:
    fig = plt.figure(figsize=(8.2, 5.2), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    # Reserve the top 22% of the figure for title+subtitle as their own
    # fig.text calls, in figure coordinates - so they can never collide with
    # each other or with ax.set_title's internal padding.
    ax = fig.add_axes((0.09, 0.13, 0.87, 0.65))
    ax.set_facecolor("#fcfcfb")

    fig.text(0.045, 0.95, title, color=INK, fontsize=13, fontweight="bold", va="top")
    fig.text(0.045, 0.885, subtitle, color=MUTED, fontsize=9, va="top")

    # cost -> marker area, sqrt-scaled so area (not radius) tracks cost
    costs = [row[4] for row in points]
    lo, hi = min(costs), max(costs)

    def size(c: float) -> float:
        if hi <= lo:
            return 260.0
        t = ((c - lo) / (hi - lo)) ** 0.5
        return 90.0 + t * 520.0

    seen_kinds: list[str] = []
    for label, kind, acc, p50, cost, (ha, va, dx, dy) in points:
        color = COLOR[kind]
        ax.scatter(
            [p50], [acc], s=[size(cost)], color=color, alpha=0.85,
            edgecolors="white", linewidths=1.2, zorder=3,
        )
        ax.annotate(
            f"{label}\nacc {acc:.0%} · ${cost:g}/1M",
            (p50, acc), xytext=(dx, dy), textcoords="offset points",
            fontsize=8.3, color=INK, va=va, ha=ha, fontfamily="monospace",
        )
        if kind not in seen_kinds:
            seen_kinds.append(kind)

    ax.set_xscale("log")
    ax.set_xlabel("p50 latency, ms (log scale) — further right is slower", color=MUTED, fontsize=9)
    ax.set_ylabel("accuracy", color=MUTED, fontsize=9)
    ax.set_ylim(-0.02, 1.08)
    xs = [row[3] for row in points]
    ax.set_xlim(min(xs) * 0.2, max(xs) * 6)

    ax.grid(True, which="both", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=COLOR[k],
                   markersize=8, label=KIND_LABEL[k])
        for k in seen_kinds
    ]
    ax.legend(
        handles=handles, loc="upper left", frameon=False, fontsize=8.3,
        labelcolor=INK, handletextpad=0.5,
    )
    fig.text(
        0.955, 0.035,
        "marker area ∝ cost per million calls — bigger circle costs more",
        ha="right", fontsize=7.6, color=MUTED, style="italic",
    )

    fig.savefig(
        out_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15
    )
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    draw(
        SYNTHETIC,
        "synthetic task — 4 labels, 60 examples",
        "keyword-vocabulary-overlapping task; a weak discriminator on its own, see Known weaknesses",
        OUT / "synthetic_frontier.png",
    )
    draw(
        BANKING77,
        "banking77 — 77 labels, 300 examples",
        "real intent-classification benchmark, Kaggle T4 (local) + hosted (local machine)",
        OUT / "banking77_frontier.png",
    )
