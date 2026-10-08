"""Node-link transition graph (SAD §15.2, §29U.2).

Tokens are nodes on a circle, ordered by frequency so layouts are stable;
node area shows how often a token occurs and arrow width shows the share of
all transitions an edge carries. Only the ``top_n`` most frequent real tokens
and the ``max_edges`` heaviest edges are drawn; the analysis keeps the full
graph. With a synthetic dataset, both panels share the layout and scales.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from eduloggen.analysis import NavigationGraph, navigation_graph, session_sequences
from eduloggen.models import Dataset
from eduloggen.visualization.base import PALETTE, new_figure

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["plot_transition_graph"]

_MAX_EDGE_WIDTH = 9.0


def plot_transition_graph(
    real: Dataset,
    synthetic: Dataset | None = None,
    *,
    top_n: int = 10,
    max_edges: int = 25,
) -> Figure:
    """Directed graph of token-to-token moves within sessions.

    Args:
        real: Sessionized real dataset; defines nodes and layout.
        synthetic: Optional synthetic dataset drawn with the same layout.
        top_n: Most frequent real tokens shown as nodes.
        max_edges: Heaviest edges drawn per panel.
    """
    graphs = [("real", navigation_graph(session_sequences(real)))]
    if synthetic is not None:
        graphs.append(("synthetic", navigation_graph(session_sequences(synthetic))))
    nodes = sorted(graphs[0][1].nodes, key=lambda t: (-graphs[0][1].nodes[t], t))[
        :top_n
    ]
    positions = _circle(nodes)
    colors = {token: PALETTE[i % len(PALETTE)] for i, token in enumerate(nodes)}
    node_max = (
        max(
            (
                g.nodes.get(t, 0) / max(sum(g.nodes.values()), 1)
                for _, g in graphs
                for t in nodes
            ),
            default=1.0,
        )
        or 1.0
    )
    edge_max = (
        max(
            (share for _, g in graphs for _, share in _edges(g, nodes, max_edges)),
            default=1.0,
        )
        or 1.0
    )

    figure, axes = new_figure(ncols=len(graphs), width=5.5, height=5.5)
    for ax, (label, graph) in zip(
        axes if len(graphs) > 1 else [axes], graphs, strict=True
    ):
        _draw(ax, graph, nodes, positions, colors, node_max, edge_max, max_edges)
        ax.set_title(f"Transition graph ({label})")
    figure.text(
        0.5,
        0.01,
        "node area: token frequency · arrow width: share of transitions",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    return figure


def _circle(nodes: list[str]) -> dict[str, tuple[float, float]]:
    count = max(len(nodes), 1)
    return {
        token: (
            math.cos(math.pi / 2 - 2 * math.pi * i / count),
            math.sin(math.pi / 2 - 2 * math.pi * i / count),
        )
        for i, token in enumerate(nodes)
    }


def _edges(
    graph: NavigationGraph, nodes: list[str], limit: int
) -> list[tuple[tuple[str, str], float]]:
    total = sum(graph.edges.values()) or 1
    kept = [
        (edge, count / total)
        for edge, count in graph.edges.items()
        if edge[0] in nodes and edge[1] in nodes
    ]
    return sorted(kept, key=lambda item: (-item[1], item[0]))[:limit]


def _draw(
    ax: Any,
    graph: NavigationGraph,
    nodes: list[str],
    positions: dict[str, tuple[float, float]],
    colors: dict[str, str],
    node_max: float,
    edge_max: float,
    max_edges: int,
) -> None:
    from matplotlib.patches import Circle, FancyArrowPatch

    total_nodes = max(sum(graph.nodes.values()), 1)
    radius = {
        t: 0.05 + 0.15 * math.sqrt(graph.nodes.get(t, 0) / total_nodes / node_max)
        for t in nodes
    }
    loops: dict[str, float] = {}
    for (a, b), share in _edges(graph, nodes, max_edges):
        width = max(0.4, _MAX_EDGE_WIDTH * share / edge_max)
        if a == b:
            loops[a] = width
            continue
        (x0, y0), (x1, y1) = positions[a], positions[b]
        length = math.hypot(x1 - x0, y1 - y0) or 1.0
        ux, uy = (x1 - x0) / length, (y1 - y0) / length
        start = (x0 + ux * (radius[a] + 0.02), y0 + uy * (radius[a] + 0.02))
        end = (x1 - ux * (radius[b] + 0.03), y1 - uy * (radius[b] + 0.03))
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                connectionstyle="arc3,rad=0.18",
                arrowstyle="-|>",
                mutation_scale=8 + width,
                linewidth=width,
                color=colors[a],
                alpha=0.55,
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            )
        )
    for token in nodes:
        x, y = positions[token]
        norm = math.hypot(x, y) or 1.0
        ox, oy = x / norm, y / norm
        r = radius[token]
        reach = r
        if token in loops:
            loop_r = 0.45 * r + 0.03
            centre = (x + ox * (r + loop_r * 0.6), y + oy * (r + loop_r * 0.6))
            ax.add_patch(
                Circle(
                    centre,
                    loop_r,
                    fill=False,
                    linewidth=loops[token],
                    edgecolor=colors[token],
                    alpha=0.6,
                    zorder=2,
                )
            )
            reach = r + loop_r * 1.6
        ax.add_patch(
            Circle((x, y), r, color=colors[token], ec="white", lw=1.2, zorder=3)
        )
        ax.text(
            x + ox * (reach + 0.1),
            y + oy * (reach + 0.1),
            token,
            ha="left" if ox > 0.3 else "right" if ox < -0.3 else "center",
            va="bottom" if oy > 0.3 else "top" if oy < -0.3 else "center",
            fontsize=8,
            zorder=4,
        )
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.75, 1.75)
    ax.set_aspect("equal")
    ax.axis("off")
