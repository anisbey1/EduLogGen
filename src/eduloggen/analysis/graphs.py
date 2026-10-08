"""Navigation graph of token-to-token moves (SAD §11.8, §29C.5)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from types import MappingProxyType
from typing import Any

__all__ = ["NavigationGraph", "navigation_graph"]

Edge = tuple[str, str]


@dataclass(frozen=True, slots=True)
class NavigationGraph:
    """Directed, weighted graph of consecutive tokens within sessions.

    Attributes:
        nodes: Token to number of occurrences.
        edges: ``(from, to)`` to number of transitions.
    """

    nodes: Mapping[str, int] = field(hash=False)
    edges: Mapping[Edge, int] = field(hash=False)

    def top_edges(self, n: int) -> list[tuple[Edge, int]]:
        """The ``n`` heaviest edges, ties broken alphabetically."""
        return sorted(self.edges.items(), key=lambda item: (-item[1], item[0]))[:n]

    def self_loop_share(self) -> float:
        """Share of transitions that repeat the same token (0 if none)."""
        total = sum(self.edges.values())
        loops = sum(count for (a, b), count in self.edges.items() if a == b)
        return loops / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "nodes": dict(sorted(self.nodes.items())),
            "edges": [
                {"from": a, "to": b, "count": count}
                for (a, b), count in sorted(self.edges.items())
            ],
        }


def navigation_graph(sequences: Iterable[Sequence[str]]) -> NavigationGraph:
    """Build the graph from token sequences; moves never cross sequences."""
    nodes: Counter[str] = Counter()
    edges: Counter[Edge] = Counter()
    for sequence in sequences:
        nodes.update(sequence)
        edges.update(pairwise(sequence))
    return NavigationGraph(
        nodes=MappingProxyType(dict(nodes)), edges=MappingProxyType(dict(edges))
    )
