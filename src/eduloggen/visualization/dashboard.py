"""Charts for validation and benchmark reports (PRD §13.2 item 6, SAD §29U.7)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from eduloggen.benchmark import BenchmarkReport
from eduloggen.validation import ValidationReport
from eduloggen.visualization.base import PALETTE, new_figure

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["plot_benchmark", "plot_validation_report"]

_STATUS_COLORS = {
    "pass": PALETTE[2],
    "fail": PALETTE[5],
    "warn": PALETTE[1],
    "info": PALETTE[4],
    "skip": "#BBBBBB",
}


def plot_validation_report(report: ValidationReport) -> Figure:
    """Horizontal bars of metric values coloured by status, with thresholds.

    Most metrics lie in ``[0, 1]`` and share that axis. Larger values (such as
    ``session_duration_w1``, in seconds) are drawn clipped at the edge with
    the true value printed. Skipped metrics have no bar.
    """
    from matplotlib.patches import Patch

    metrics = list(report.metrics)
    figure, ax = new_figure(width=7.0, height=max(2.5, 0.38 * len(metrics) + 1.2))
    for row, metric in enumerate(metrics):
        color = _STATUS_COLORS.get(metric.status, "#777777")
        if metric.value is None:
            ax.text(0.01, row, "skipped", va="center", fontsize=8, color="#777777")
            continue
        clipped = metric.value > 1.0
        ax.barh(
            row, min(metric.value, 1.0), color=color, hatch="//" if clipped else None
        )
        label = f"{metric.value:.3g}" + (" (off scale)" if clipped else "")
        ax.text(min(metric.value, 1.0) + 0.01, row, label, va="center", fontsize=8)
        if metric.threshold is not None and metric.threshold <= 1.0:
            ax.plot(
                [metric.threshold] * 2,
                [row - 0.4, row + 0.4],
                color="black",
                linewidth=1.5,
            )
    better = {"lower_better": "lower", "higher_better": "higher"}
    ax.set_yticks(
        range(len(metrics)),
        [f"{m.name} ({better[m.direction]} better)" for m in metrics],
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1.25)
    ax.set_xlabel("value (black line: threshold)")
    ax.set_title(f"Validation: {report.status.upper()}")
    present = [s for s in _STATUS_COLORS if any(m.status == s for m in metrics)]
    ax.legend(
        handles=[Patch(color=_STATUS_COLORS[s], label=s) for s in present],
        loc="lower right",
    )
    return figure


def plot_benchmark(report: BenchmarkReport) -> Figure:
    """One small panel per metric: generator means with min-max whiskers.

    Whiskers span the values observed across seeds, so they never leave the
    metric's range. A dashed line marks the real-data reference (train vs
    holdout).
    """
    metrics = list(report.protocol.get("metrics", []))
    ok = [r for r in report.results if not r.error]
    ncols = min(4, max(1, len(metrics)))
    nrows = max(1, math.ceil(len(metrics) / ncols))
    figure, axes = new_figure(ncols=ncols, nrows=nrows, width=3.2, height=2.6)
    flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    colors = {r.name: PALETTE[i % len(PALETTE)] for i, r in enumerate(ok)}
    for ax, metric in zip(flat, metrics, strict=False):
        means: list[float] = []
        low: list[float] = []
        high: list[float] = []
        names: list[str] = []
        for result in ok:
            summary = result.metrics.get(metric)
            if summary is not None and summary.mean is not None:
                names.append(result.name)
                means.append(summary.mean)
                low.append(summary.mean - (summary.min or summary.mean))
                high.append((summary.max or summary.mean) - summary.mean)
        ax.bar(
            range(len(names)),
            means,
            yerr=[low, high],
            capsize=3,
            color=[colors[n] for n in names],
        )
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        reference = report.reference.get(metric)
        if reference is not None:
            ax.axhline(reference, color="black", linestyle="--", linewidth=1)
        direction = next(
            (r.metrics[metric].direction for r in ok if metric in r.metrics),
            "lower_better",
        )
        arrow = "↓" if direction == "lower_better" else "↑"
        ax.set_title(f"{metric} {arrow}", fontsize=9)
    for ax in flat[len(metrics) :]:
        ax.set_visible(False)
    figure.suptitle(
        f"Benchmark {report.protocol.get('name')} (dashed: real-data reference)"
    )
    return figure
