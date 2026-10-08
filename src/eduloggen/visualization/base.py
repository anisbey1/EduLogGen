"""Shared plotting setup: lazy matplotlib import, palette, and export.

matplotlib is an optional dependency (``pip install 'eduloggen[viz]'``).
It is imported only when a plot is made, with the non-interactive ``Agg``
backend unless another backend is already active, so plots work on headless
CI machines.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from eduloggen.core import ExportError, IoError, PathLike

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "FIGURE_FORMATS",
    "PALETTE",
    "REAL_COLOR",
    "SYNTHETIC_COLOR",
    "new_figure",
    "require_matplotlib",
    "save_figure",
]

PALETTE: Final = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#000000",
)
"""Okabe-Ito colours, distinguishable with common colour-vision deficiencies."""

REAL_COLOR: Final = PALETTE[0]
SYNTHETIC_COLOR: Final = PALETTE[1]
FIGURE_FORMATS: Final = frozenset({"png", "svg", "pdf"})

_STYLE: Final[dict[str, Any]] = {
    "figure.dpi": 100,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": False,
}


def require_matplotlib() -> Any:
    """Import :mod:`matplotlib.pyplot` or explain how to install it.

    Raises:
        IoError: If matplotlib is not installed.
    """
    try:
        import matplotlib
    except ImportError:
        raise IoError(
            "plotting requires matplotlib; install it with: "
            "pip install 'eduloggen[viz]'",
            code="io_missing_dependency",
            context={"package": "matplotlib", "extra": "viz"},
        ) from None
    if "matplotlib.pyplot" not in __import__("sys").modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def new_figure(
    ncols: int = 1, nrows: int = 1, *, width: float = 6.0, height: float = 3.6
) -> tuple[Figure, Any]:
    """Create a styled figure and its axes (an array when there are several).

    Deterministic styling keeps figures reproducible for papers.
    """
    plt = require_matplotlib()
    with plt.rc_context(_STYLE):
        figure, axes = plt.subplots(
            nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=True
        )
    figure.set_layout_engine("constrained")
    return figure, axes


def save_figure(
    figure: Figure,
    path: PathLike,
    *,
    formats: Iterable[str] | None = None,
    dpi: int = 150,
    close: bool = True,
) -> list[Path]:
    """Save a figure in one or more formats.

    Args:
        figure: Figure to save.
        path: Target file. With ``formats``, its suffix is replaced by each
            format; without, the suffix chooses the format.
        formats: Any of ``png``, ``svg``, ``pdf``.
        dpi: Resolution for raster output.
        close: Close the figure afterwards to free memory.

    Returns:
        The written paths.

    Raises:
        ExportError: If a format is unsupported or writing fails.
    """
    target = Path(path).expanduser()
    chosen = list(formats) if formats is not None else [target.suffix.lstrip(".")]
    unknown = sorted(set(chosen) - FIGURE_FORMATS)
    if unknown or not chosen:
        raise ExportError(
            "unsupported figure format",
            code="export_unknown_format",
            context={"formats": unknown or chosen, "supported": sorted(FIGURE_FORMATS)},
        )
    written = []
    plt = require_matplotlib()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # A fixed SVG id salt and metadata make output byte-stable across runs.
        with plt.rc_context({"svg.hashsalt": "eduloggen"}):
            for fmt in chosen:
                out = target.with_suffix(f".{fmt}")
                figure.savefig(out, format=fmt, dpi=dpi, metadata=_metadata(fmt))
                written.append(out)
    except OSError as exc:
        raise ExportError(
            "could not write figure",
            code="export_write_failed",
            context={"path": str(target)},
        ) from exc
    finally:
        if close:
            plt.close(figure)
    return written


def _metadata(fmt: str) -> dict[str, Any]:
    if fmt == "svg":
        return {"Date": None, "Creator": "EduLogGen"}
    if fmt == "pdf":
        return {"CreationDate": None, "ModDate": None, "Creator": "EduLogGen"}
    return {"Software": "EduLogGen"}
