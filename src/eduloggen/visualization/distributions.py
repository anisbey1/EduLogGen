"""Real-versus-synthetic distribution plots (PRD §13.2 items 1-3 and 5).

Each function takes a real dataset and optionally a synthetic one, overlays
them with shared bins, and returns the figure. Labels show event types,
activities, and counts only; never learner identifiers.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from eduloggen.analysis import interevent_times, session_durations, session_sequences
from eduloggen.models import Dataset
from eduloggen.visualization.base import REAL_COLOR, SYNTHETIC_COLOR, new_figure

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "plot_event_frequencies",
    "plot_interevent_times",
    "plot_session_durations",
    "plot_session_lengths",
]


def _hist(
    ax: Any, values: list[float], bins: list[float], label: str, color: str
) -> None:
    """Real data as filled bars, synthetic as an outline, so overlaps stay legible."""
    if label == "real":
        ax.hist(values, bins=bins, density=True, alpha=0.45, label=label, color=color)
    else:
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=label,
            color=color,
        )


def _series(real: Any, synthetic: Any) -> list[tuple[str, Any, str]]:
    series = [("real", real, REAL_COLOR)]
    if synthetic is not None:
        series.append(("synthetic", synthetic, SYNTHETIC_COLOR))
    return series


def plot_event_frequencies(
    real: Dataset,
    synthetic: Dataset | None = None,
    *,
    field: Literal["event_type", "activity_id"] = "event_type",
    top_n: int = 20,
) -> Figure:
    """Grouped bars of the share of events per event type (or activity).

    Categories are the ``top_n`` most frequent in the real data.
    """
    shares = []
    for _, dataset, _ in _series(real, synthetic):
        counts = Counter(getattr(e, field) for e in dataset.events)
        total = sum(counts.values()) or 1
        shares.append({key: value / total for key, value in counts.items()})
    ranked = sorted(shares[0], key=lambda key: (-shares[0][key], key))[:top_n]
    figure, ax = new_figure(width=max(6.0, 0.45 * len(ranked) + 2))
    width = 0.8 / len(shares)
    for index, (label, _, color) in enumerate(_series(real, synthetic)):
        positions = [
            i + (index - (len(shares) - 1) / 2) * width for i in range(len(ranked))
        ]
        ax.bar(
            positions,
            [shares[index].get(k, 0.0) for k in ranked],
            width,
            label=label,
            color=color,
        )
    ax.set_xticks(range(len(ranked)), ranked, rotation=45, ha="right")
    ax.set_ylabel("share of events")
    ax.set_title(f"Event frequencies by {field.replace('_', ' ')}")
    if len(shares) > 1:
        ax.legend()
    return figure


def plot_session_lengths(real: Dataset, synthetic: Dataset | None = None) -> Figure:
    """Overlaid histograms of events per session (integer bins)."""
    samples = [
        (label, [float(len(s)) for s in session_sequences(d)], color)
        for label, d, color in _series(real, synthetic)
    ]
    top = int(max((max(values, default=1.0) for _, values, _ in samples), default=1.0))
    bins = [b - 0.5 for b in range(1, top + 2)]
    return _histograms(samples, bins, "events per session", "Session length")


def plot_session_durations(real: Dataset, synthetic: Dataset | None = None) -> Figure:
    """Overlaid histograms of session durations in minutes."""
    samples = [
        (label, [v / 60 for v in session_durations(d)], color)
        for label, d, color in _series(real, synthetic)
    ]
    return _histograms(samples, _shared_bins(samples), "minutes", "Session duration")


def plot_interevent_times(real: Dataset, synthetic: Dataset | None = None) -> Figure:
    """Histograms (log-spaced) and ECDFs of seconds between events in a session."""
    samples = [
        (label, [max(v, 1e-3) for v in interevent_times(d)], color)
        for label, d, color in _series(real, synthetic)
    ]
    figure, (left, right) = new_figure(ncols=2, width=4.5)
    values = [v for _, sample, _ in samples for v in sample]
    if values:
        low, high = math.log10(min(values)), math.log10(max(values))
        if high <= low:
            high = low + 1
        bins = [10 ** (low + (high - low) * i / 40) for i in range(41)]
        for label, sample, color in samples:
            if sample:
                left.hist(
                    sample,
                    bins=bins,
                    density=True,
                    alpha=0.55,
                    label=label,
                    color=color,
                )
                ordered = sorted(sample)
                right.step(
                    ordered,
                    [(i + 1) / len(ordered) for i in range(len(ordered))],
                    where="post",
                    label=label,
                    color=color,
                )
        left.set_xscale("log")
        right.set_xscale("log")
    left.set_xlabel("seconds (log scale)")
    left.set_ylabel("density")
    left.set_title("Inter-event time")
    right.set_xlabel("seconds (log scale)")
    right.set_ylabel("cumulative share")
    right.set_title("Inter-event time ECDF")
    if len(samples) > 1:
        right.legend()
    return figure


def _shared_bins(
    samples: Sequence[tuple[str, list[float], str]], n: int = 30
) -> list[float]:
    values = [v for _, sample, _ in samples for v in sample]
    if not values:
        return [0.0, 1.0]
    low, high = min(values), max(values)
    if high <= low:
        high = low + 1
    return [low + (high - low) * i / n for i in range(n + 1)]


def _histograms(
    samples: Sequence[tuple[str, list[float], str]],
    bins: list[float],
    xlabel: str,
    title: str,
) -> Figure:
    figure, ax = new_figure()
    for label, values, color in samples:
        if values:
            _hist(ax, values, bins, label, color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title)
    if len(samples) > 1:
        ax.legend()
    return figure
