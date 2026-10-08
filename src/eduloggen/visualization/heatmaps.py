"""Transition heatmaps (PRD §13.2 item 4, SAD §29U.4)."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from eduloggen.analysis import ngram_counts, session_sequences
from eduloggen.models import Dataset
from eduloggen.visualization.base import new_figure

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["plot_transition_heatmap", "transition_matrix"]


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
