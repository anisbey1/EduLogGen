"""Write the standard figure set to a directory."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Final

from eduloggen.core import ConfigError, PathLike
from eduloggen.models import Dataset
from eduloggen.visualization.base import save_figure
from eduloggen.visualization.distributions import (
    plot_event_frequencies,
    plot_interevent_times,
    plot_session_durations,
    plot_session_lengths,
)
from eduloggen.visualization.heatmaps import plot_transition_heatmap
from eduloggen.visualization.timeline import plot_timeline

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["DATASET_PLOTS", "plot_datasets"]

DATASET_PLOTS: Final[dict[str, Callable[[Dataset, Dataset | None], Figure]]] = {
    "event_frequencies": lambda real, synthetic: plot_event_frequencies(
        real, synthetic
    ),
    "activity_frequencies": lambda real, synthetic: plot_event_frequencies(
        real, synthetic, field="activity_id"
    ),
    "session_lengths": plot_session_lengths,
    "session_durations": plot_session_durations,
    "interevent_times": plot_interevent_times,
    "transitions": lambda real, synthetic: plot_transition_heatmap(real, synthetic),
    "timeline": lambda real, _synthetic: plot_timeline(real),
}
"""Plot name to function of (real, synthetic or ``None``)."""


def plot_datasets(
    real: Dataset,
    synthetic: Dataset | None,
    output_dir: PathLike,
    *,
    plots: Iterable[str] | None = None,
    formats: Iterable[str] = ("png",),
    dpi: int = 150,
) -> dict[str, list[Path]]:
    """Render dataset plots and save them as ``<output_dir>/<name>.<format>``.

    Args:
        real: Sessionized real dataset.
        synthetic: Optional sessionized synthetic dataset to overlay.
        output_dir: Directory for the figures (created if needed).
        plots: Names from :data:`DATASET_PLOTS`; all by default.
        formats: Image formats.
        dpi: Raster resolution.

    Returns:
        Plot name to written file paths.

    Raises:
        ConfigError: If a plot name is unknown.
    """
    names = list(plots) if plots is not None else list(DATASET_PLOTS)
    unknown = sorted(set(names) - set(DATASET_PLOTS))
    if unknown:
        raise ConfigError(
            f"unknown plots: {', '.join(unknown)}",
            code="plot_unknown",
            context={"plots": unknown, "available": sorted(DATASET_PLOTS)},
        )
    directory = Path(output_dir).expanduser()
    formats = list(formats)
    return {
        name: save_figure(
            DATASET_PLOTS[name](real, synthetic),
            directory / name,
            formats=formats,
            dpi=dpi,
        )
        for name in names
    }
