"""Draw the EduLogGen architecture figure (fig_architecture.pdf/.png)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1f2933"
MAIN = "#dbe7f3"
EXP = "#f6e3c6"
BASE = "#e4e7eb"


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color, edgecolor=INK, linewidth=0.9,
        )
    )  # fmt: skip
    ax.text(x + w / 2, y + h - 0.17, title, ha="center", va="top",
            fontsize=9, fontweight="bold", color=INK)  # fmt: skip
    ax.text(x + w / 2, y + h - 0.47, body, ha="center", va="top",
            fontsize=7.4, color=INK, linespacing=1.3)  # fmt: skip


def arrow(ax, start, end, text=None, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=11, color=INK,
            linewidth=0.9, linestyle="--" if dashed else "-",
        )
    )  # fmt: skip
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.1, text,
                ha="center", va="bottom", fontsize=7, color=INK, style="italic")  # fmt: skip


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.75)
    ax.axis("off")

    # interfaces
    box(ax, 0.2, 3.95, 9.6, 0.7, "Interfaces", "Python API  ·  command-line interface  ·  plugins (readers, generators, metrics, anomalies)", BASE)  # fmt: skip

    # main pipeline
    y, h, w = 2.3, 1.35, 1.75
    xs = [0.2, 2.15, 4.1, 6.05, 8.0]
    items = [
        ("Ingestion", "CSV, JSONL,\nParquet\nYAML mapping\npseudonymised"),
        ("Analysis", "sessions\nn-grams\ntransitions\ntiming, groups"),
        ("Generators", "independent\nMarkov (k)\nsemi-Markov\nstart times"),
        ("Validation", "fidelity\nmetrics\nprivacy\nindicators"),
        ("Benchmark", "versioned\nprotocol\nheld-out\nlearners"),
    ]
    for x, (title, body) in zip(xs, items, strict=True):
        box(ax, x, y, w, h, title, body, MAIN)
    for a, b in zip(xs, xs[1:], strict=False):
        arrow(ax, (a + w, y + h / 2), (b, y + h / 2))

    # experimental layer
    box(ax, 2.15, 0.75, 3.35, 1.1, "Scenarios (experimental layer)", "anomalies · controls\ncalendars · profiles", EXP)  # fmt: skip
    box(ax, 6.45, 0.75, 3.35, 1.1, "Evaluation", "detectors: ROC-AUC, AP\nclustering: ARI, NMI", EXP)  # fmt: skip
    arrow(ax, (xs[2] + 0.5, y), (xs[2] + 0.5, 1.85))
    ax.text(xs[2] + 0.6, 2.07, "transforms models", ha="left", va="center", fontsize=7.2, color=INK, style="italic")  # fmt: skip
    arrow(ax, (5.5, 1.3), (6.45, 1.3))
    ax.text(5.975, 1.38, "labels", ha="center", va="bottom", fontsize=7.2, color=INK, style="italic")  # fmt: skip

    # foundation
    box(ax, 0.2, 0.05, 9.6, 0.55, "", "", BASE)
    ax.text(5.0, 0.32, "Core: immutable data models · seeds · fingerprints · JSON artifacts",
            ha="center", va="center", fontsize=7.4, color=INK)  # fmt: skip
    box(ax, 0.2, 0.75, 1.75, 1.1, "Real logs", "LMS exports", "white")
    arrow(ax, (1.075, 1.85), (1.075, y))

    out = Path(__file__).with_name("fig_architecture.pdf")
    fig.tight_layout()
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=170)


if __name__ == "__main__":
    main()
