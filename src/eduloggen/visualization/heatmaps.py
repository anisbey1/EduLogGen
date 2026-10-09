"""Transition heatmaps (PRD §13.2 item 4, SAD §29U.4)."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from eduloggen.analysis import ngram_counts, session_sequences
from eduloggen.models import Dataset
from eduloggen.visualization.base import new_figure

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["plot_activity_heatmap", "plot_transition_heatmap", "transition_matrix"]


def transition_matrix(dataset: Dataset, tokens: list[str]) -> list[list[float]]:
    """Row-normalized ``P(next | current)`` restricted to ``tokens``.

    Rows of tokens never followed by another listed token are all zero.
    """
    bigrams = ngram_counts(session_sequences(dataset), 2)
    index = {token: i for i, token in enumerate(tokens)}
    matrix = [[0.0] * len(tokens) for _ in tokens]
    for (a, b), count in bigrams.items():
        if a in index and b in index:
            matrix[index[a]][index[b]] += count
    for row in matrix:
        total = sum(row)
        if total:
            row[:] = [value / total for value in row]
    return matrix


def plot_transition_heatmap(
    real: Dataset, synthetic: Dataset | None = None, *, top_n: int = 12
) -> Figure:
    """Next-token probabilities for the ``top_n`` most frequent real tokens.

    With a synthetic dataset, both panels share the token order and colour
    scale so they can be compared cell by cell.
    """
    counts = Counter(t for s in session_sequences(real) for t in s)
    tokens = sorted(counts, key=lambda t: (-counts[t], t))[:top_n]
    panels = [("real", real)] + ([("synthetic", synthetic)] if synthetic else [])
    size = max(4.0, 0.42 * len(tokens) + 2.0)
    figure, axes = new_figure(ncols=len(panels), width=size, height=size)
    axes_list = list(axes) if len(panels) > 1 else [axes]
    image = None
    for ax, (label, dataset) in zip(axes_list, panels, strict=True):
        matrix = transition_matrix(dataset, tokens)
        image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(tokens)), tokens, rotation=60, ha="right")
        ax.set_yticks(range(len(tokens)), tokens)
        ax.set_xlabel("next")
        ax.set_ylabel("current")
        ax.set_title(f"Transitions ({label})")
        ax.grid(False)
    if image is not None:
        figure.colorbar(image, ax=axes_list, shrink=0.8, label="P(next | current)")
    return figure


def plot_activity_heatmap(
    real: Dataset, synthetic: Dataset | None = None, *, timezone: str = "UTC"
) -> Figure:
    """Share of session starts per weekday and hour (shared colour scale).

    Args:
        real: Sessionized real dataset.
        synthetic: Optional synthetic dataset shown alongside.
        timezone: IANA zone in which hours and weekdays are read.
    """
    from eduloggen.analysis import temporal_profile
    from eduloggen.analysis.temporal import WEEKDAYS

    panels = [("real", real)] + ([("synthetic", synthetic)] if synthetic else [])
    matrices = []
    for _, dataset in panels:
        grid = temporal_profile(dataset, timezone=timezone).heatmap
        total = sum(sum(row) for row in grid) or 1
        matrices.append([[count / total for count in row] for row in grid])
    peak = max((max(row) for m in matrices for row in m), default=0.0) or 1.0
    figure, axes = new_figure(nrows=len(panels), width=8.0, height=2.6)
    axes_list = list(axes) if len(panels) > 1 else [axes]
    image = None
    for ax, (label, _), matrix in zip(axes_list, panels, matrices, strict=True):
        image = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=peak, aspect="auto")
        ax.set_yticks(range(7), WEEKDAYS)
        ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
        ax.set_xlabel(f"hour ({timezone})")
        ax.set_title(f"Session starts ({label})")
        ax.grid(False)
    if image is not None:
        figure.colorbar(image, ax=axes_list, shrink=0.8, label="share of sessions")
    return figure
