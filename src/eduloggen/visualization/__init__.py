"""Figures for datasets, validation reports, and benchmarks (PRD §13, SAD §15).

Requires the ``viz`` extra (``pip install 'eduloggen[viz]'``); every other
package works without it. Every function returns a matplotlib ``Figure``;
:func:`save_figure` writes PNG, SVG, or PDF and :func:`plot_datasets` writes
the standard set::

    from eduloggen.visualization import plot_datasets, plot_session_lengths

    plot_session_lengths(real, synthetic).savefig("lengths.png")
    plot_datasets(real, synthetic, "figures/", formats=["png", "svg"])

Figures use a colour-blind-friendly palette and never show learner ids.
"""

from __future__ import annotations

from eduloggen.visualization.base import (
    FIGURE_FORMATS,
    PALETTE,
    require_matplotlib,
    save_figure,
)
from eduloggen.visualization.dashboard import plot_benchmark, plot_validation_report
from eduloggen.visualization.distributions import (
    plot_event_frequencies,
    plot_interevent_times,
    plot_session_durations,
    plot_session_lengths,
)
from eduloggen.visualization.export import DATASET_PLOTS, plot_datasets
from eduloggen.visualization.graphs import plot_transition_graph
from eduloggen.visualization.heatmaps import plot_transition_heatmap, transition_matrix
from eduloggen.visualization.sankey import SankeyLayout, plot_sankey, sankey_flows
from eduloggen.visualization.timeline import plot_timeline

__all__ = [
    "DATASET_PLOTS",
    "FIGURE_FORMATS",
    "PALETTE",
    "SankeyLayout",
    "plot_benchmark",
    "plot_datasets",
    "plot_event_frequencies",
    "plot_interevent_times",
    "plot_sankey",
    "plot_session_durations",
    "plot_session_lengths",
    "plot_timeline",
    "plot_transition_graph",
    "plot_transition_heatmap",
    "plot_validation_report",
    "require_matplotlib",
    "sankey_flows",
    "save_figure",
    "transition_matrix",
]
