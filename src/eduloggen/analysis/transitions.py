"""Order-k transition counts and probabilities (FR-A.4, SAD §11.5, §29C.3).

Each sequence is left-padded with ``order`` copies of :data:`START`, so the
first tokens are predicted from contexts like ``(START,)`` (order 1) or
``(START, "view")`` (order 2). The start distribution is therefore the row of
the all-``START`` context, and no separate structure is needed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from eduloggen.core import AnalysisError

__all__ = ["START", "TransitionCounts", "count_transitions"]

START: Final = "<start>"
"""Padding token before the first event of a sequence; reserved."""

Context = tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransitionCounts:
    """Next-token counts per context.

    Attributes:
        order: Context length ``k``.
        counts: Context to next token to count.
        vocabulary: All tokens observed (excluding :data:`START`).
    """

    order: int
    counts: Mapping[Context, Mapping[str, int]] = field(hash=False)
    vocabulary: frozenset[str] = frozenset()

    @property
    def start_context(self) -> Context:
        """Context that predicts the first token of a sequence."""
        return (START,) * self.order

    def start_counts(self) -> Mapping[str, int]:
        """Counts of first tokens."""
        return self.counts.get(self.start_context, MappingProxyType({}))

    def n_transitions(self) -> int:
        """Total observed transitions, including from the start context."""
        return sum(sum(row.values()) for row in self.counts.values())

    def density(self) -> float:
        """Share of possible token-to-token transitions observed.

        Possible transitions are ``|V|^order * |V|`` over real-token
        contexts; start-padded contexts are excluded. ``0.0`` if empty.
        """
        size = len(self.vocabulary)
        if size == 0:
            return 0.0
        observed = sum(
            len(row) for context, row in self.counts.items() if START not in context
        )
        return observed / float(size ** (self.order + 1))

    def probabilities(
        self, smoothing_alpha: float = 0.0
    ) -> dict[Context, dict[str, float]]:
        """Row-normalized transition probabilities.

        Args:
            smoothing_alpha: Additive (Laplace/Lidstone) smoothing. With
                ``alpha > 0`` every vocabulary token gets mass in every
                observed context.

        Returns:
            Context to next token to probability; rows sum to 1.

        Raises:
            AnalysisError: If ``smoothing_alpha`` is negative.
        """
        if smoothing_alpha < 0:
            raise AnalysisError(
                "smoothing_alpha must be non-negative",
                code="analysis_invalid_parameter",
            )
        tokens = sorted(self.vocabulary)
        result: dict[Context, dict[str, float]] = {}
        for context, row in self.counts.items():
            support = tokens if smoothing_alpha > 0 else sorted(row)
            total = sum(row.values()) + smoothing_alpha * len(support)
            result[context] = {
                token: (row.get(token, 0) + smoothing_alpha) / total
                for token in support
            }
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data with contexts as lists."""
        return {
            "order": self.order,
            "vocabulary": sorted(self.vocabulary),
            "rows": [
                {"context": list(context), "next": dict(sorted(row.items()))}
                for context, row in sorted(self.counts.items())
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TransitionCounts:
        """Restore counts serialized with :meth:`to_dict`.

        Raises:
            AnalysisError: If the data is malformed.
        """
        try:
            order = int(data["order"])
            counts = {
                tuple(str(t) for t in row["context"]): {
                    str(token): int(count) for token, count in row["next"].items()
                }
                for row in data["rows"]
            }
            vocabulary = frozenset(str(token) for token in data["vocabulary"])
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise AnalysisError(
                "malformed transition counts", code="analysis_invalid_data"
            ) from exc
        if any(len(context) != order for context in counts):
            raise AnalysisError(
                "context length does not match order", code="analysis_invalid_data"
            )
        return _freeze(order, counts, vocabulary)


def count_transitions(
    sequences: Iterable[Sequence[str]], order: int = 1
) -> TransitionCounts:
    """Count order-k transitions over token sequences.

    Args:
        sequences: Token sequences (e.g. session event sequences).
        order: Context length ``k`` (1 for a first-order Markov chain).

    Returns:
        The counts, including transitions out of start-padded contexts.

    Raises:
        AnalysisError: If ``order`` is less than 1 or a token equals the
            reserved :data:`START` token.
    """
    if order < 1:
        raise AnalysisError(
            "order must be at least 1", code="analysis_invalid_parameter"
        )
    counts: defaultdict[Context, Counter[str]] = defaultdict(Counter)
    vocabulary: set[str] = set()
    for sequence in sequences:
        padded = (START,) * order + tuple(sequence)
        for position in range(order, len(padded)):
            token = padded[position]
            if token == START:
                raise AnalysisError(
                    f"token {START!r} is reserved", code="analysis_reserved_token"
                )
            counts[padded[position - order : position]][token] += 1
            vocabulary.add(token)
    return _freeze(order, counts, frozenset(vocabulary))


def _freeze(
    order: int,
    counts: Mapping[Context, Mapping[str, int]],
    vocabulary: frozenset[str],
) -> TransitionCounts:
    frozen = {context: MappingProxyType(dict(row)) for context, row in counts.items()}
    return TransitionCounts(
        order=order, counts=MappingProxyType(frozen), vocabulary=vocabulary
    )
