"""Multi-step Sankey diagram of session pathways (SAD §15.2, §29U.3).

Column ``k`` shows what learners did at step ``k`` of a session. Nodes are
the ``top_n`` most frequent tokens at that step (the rest grouped as
``other``); a grey ``(end)`` node collects sessions that stopped. Bands
between columns are proportional to the number of sessions taking that path.
Heights use one scale across columns, so the shrinking total shows sessions
ending.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from eduloggen.analysis import session_sequences
from eduloggen.models import Dataset
from eduloggen.visualization.base import PALETTE, new_figure

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["END", "OTHER", "SankeyLayout", "plot_sankey", "sankey_flows"]

END: Final = "(end)"
OTHER: Final = "other"
_END_COLOR: Final = "#BBBBBB"
_OTHER_COLOR: Final = "#888888"
_NODE_WIDTH: Final = 0.12
_GAP: Final = 0.02


@dataclass(frozen=True, slots=True)
class SankeyLayout:
    """Node counts per step and flows between consecutive steps.

    Attributes:
        nodes: For each step, label to number of sessions.
        flows: For each step ``k``, ``(label at k, label at k + 1)`` to count.
    """

    nodes: tuple[dict[str, int], ...]
    flows: tuple[dict[tuple[str, str], int], ...]


def sankey_flows(
    sequences: Sequence[Sequence[str]], *, steps: int = 4, top_n: int = 6
) -> SankeyLayout:
    """Count sessions per step and label, and transitions between steps.

    Args:
        sequences: Session token sequences.
        steps: Number of columns (session steps) to show.
        top_n: Tokens kept per step; others become ``other``.
    """
    labels_per_step: list[list[str | None]] = []
    for step in range(steps):
        raw = Counter(seq[step] for seq in sequences if len(seq) > step)
        keep = {
            t for t, _ in sorted(raw.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        }
        row: list[str | None] = []
        for seq in sequences:
            if len(seq) > step:
                row.append(seq[step] if seq[step] in keep else OTHER)
            elif len(seq) == step and step > 0:
                row.append(END)
            else:
                row.append(None)
        labels_per_step.append(row)
    nodes = tuple(
        dict(Counter(label for label in row if label is not None))
        for row in labels_per_step
    )
    flows = []
    for step in range(steps - 1):
        counts: Counter[tuple[str, str]] = Counter()
        for here, there in zip(
            labels_per_step[step], labels_per_step[step + 1], strict=True
        ):
            if here is not None and here != END and there is not None:
                counts[(here, there)] += 1
        flows.append(dict(counts))
    return SankeyLayout(nodes=nodes, flows=tuple(flows))


def plot_sankey(
    real: Dataset,
    synthetic: Dataset | None = None,
    *,
    steps: int = 4,
    top_n: int = 6,
) -> Figure:
    """Flows of sessions through their first ``steps`` events.

    With a synthetic dataset, a second panel uses the same colours and token
    order so pathways can be compared.
    """
    panels = [("real", real)] + ([("synthetic", synthetic)] if synthetic else [])
    real_counts = Counter(t for s in session_sequences(real) for t in s)
    order = sorted(real_counts, key=lambda t: (-real_counts[t], t))
    colors = {token: PALETTE[i % len(PALETTE)] for i, token in enumerate(order)}
    rank = {token: i for i, token in enumerate(order)}

    figure, axes = new_figure(
        nrows=len(panels), width=max(6.5, 2.2 * steps), height=4.2
    )
    for ax, (label, dataset) in zip(
        axes if len(panels) > 1 else [axes], panels, strict=True
    ):
        sequences = session_sequences(dataset)
        layout = sankey_flows(sequences, steps=steps, top_n=top_n)
        _draw(ax, layout, len(sequences), colors, rank)
        ax.set_title(f"Session pathways, first {steps} steps ({label})")
    return figure


def _sort_key(rank: dict[str, int]) -> Any:
    def key(label: str) -> tuple[int, int, str]:
        if label == END:
            return (2, 0, label)
        if label == OTHER:
            return (1, 0, label)
        return (0, rank.get(label, len(rank)), label)

    return key


def _color(label: str, colors: dict[str, str]) -> str:
    if label == END:
        return _END_COLOR
    return colors.get(label, _OTHER_COLOR)


def _draw(
    ax: Any,
    layout: SankeyLayout,
    total: int,
    colors: dict[str, str],
    rank: dict[str, int],
) -> None:
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path

    ax.axis("off")
    if total == 0:
        ax.text(
            0.5, 0.5, "no sessions", ha="center", va="center", transform=ax.transAxes
        )
        return
    key = _sort_key(rank)
    scale = 1.0 / total
    spans: list[dict[str, tuple[float, float]]] = []
    for step, nodes in enumerate(layout.nodes):
        top = 1.0
        placed = {}
        for label in sorted(nodes, key=key):
            height = nodes[label] * scale * (1 - _GAP * max(len(nodes) - 1, 0))
            placed[label] = (top - height, top)
            x = step
            ax.add_patch(
                Rectangle(
                    (x - _NODE_WIDTH / 2, top - height),
                    _NODE_WIDTH,
                    height,
                    color=_color(label, colors),
                    zorder=2,
                )
            )
            if height > 0.035:
                ax.text(
                    x + _NODE_WIDTH / 2 + 0.02,
                    top - height / 2,
                    f"{label} ({nodes[label]})",
                    va="center",
                    fontsize=7,
                    zorder=3,
                )
            top -= height + _GAP
        spans.append(placed)

    for step, flows in enumerate(layout.flows):
        out_cursor = {label: span[1] for label, span in spans[step].items()}
        in_cursor = {label: span[1] for label, span in spans[step + 1].items()}
        for (src, dst), count in sorted(
            flows.items(), key=lambda kv: (key(kv[0][0]), key(kv[0][1]))
        ):
            height = count * scale * (1 - _GAP * max(len(layout.nodes[step]) - 1, 0))
            in_height = (
                count * scale * (1 - _GAP * max(len(layout.nodes[step + 1]) - 1, 0))
            )
            a1 = out_cursor[src]
            a0 = a1 - height
            out_cursor[src] = a0
            b1 = in_cursor[dst]
            b0 = b1 - in_height
            in_cursor[dst] = b0
            x0, x1 = step + _NODE_WIDTH / 2, step + 1 - _NODE_WIDTH / 2
            xm = (x0 + x1) / 2
            path = Path(
                [
                    (x0, a1),
                    (xm, a1),
                    (xm, b1),
                    (x1, b1),
                    (x1, b0),
                    (xm, b0),
                    (xm, a0),
                    (x0, a0),
                    (x0, a1),
                ],
                [
                    Path.MOVETO,
                    Path.CURVE4,
                    Path.CURVE4,
                    Path.CURVE4,
                    Path.LINETO,
                    Path.CURVE4,
                    Path.CURVE4,
                    Path.CURVE4,
                    Path.CLOSEPOLY,
                ],
            )
            ax.add_patch(
                PathPatch(
                    path,
                    facecolor=_color(src, colors),
                    edgecolor="none",
                    alpha=0.35,
                    zorder=1,
                )
            )

    for step in range(len(layout.nodes)):
        ax.text(
            step, 1.04, f"step {step + 1}", ha="center", fontsize=8, color="#555555"
        )
    ax.set_xlim(-0.3, len(layout.nodes) - 1 + 0.9)
    ax.set_ylim(
        min((s[0] for placed in spans for s in placed.values()), default=0.0) - 0.02,
        1.08,
    )
