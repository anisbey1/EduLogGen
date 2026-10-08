"""The :class:`AnalysisResult` produced by :func:`~eduloggen.analysis.analyze`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from eduloggen.analysis.graphs import NavigationGraph
from eduloggen.analysis.stats import Summary
from eduloggen.analysis.transitions import TransitionCounts
from eduloggen.models import Feature

__all__ = ["AnalysisResult"]

_TOP = 10


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisResult:
    """Descriptive and behavioural statistics of a sessionized dataset.

    Attributes:
        dataset_id: Analysed dataset.
        dataset_fingerprint: Content fingerprint of the analysed dataset.
        ngram_order: Longest n-gram counted.
        event_type_counts: Event type to number of events.
        token_counts: Session token to occurrences.
        transitions: First-order transition counts (with start context).
        ngram_counts: ``n`` to n-gram to count, for ``2 <= n <= ngram_order``.
        n_rare_ngrams: ``n`` to number of n-grams seen exactly once.
        session_length: Events per session.
        session_duration_s: Session durations in seconds.
        interevent_s: Seconds between consecutive events within sessions.
        sojourn_s: Token to time spent before the next event.
        corpus_features: Corpus-scope features.
        graph: Navigation graph of token moves.
    """

    dataset_id: str
    dataset_fingerprint: str
    ngram_order: int
    event_type_counts: Mapping[str, int] = field(hash=False)
    token_counts: Mapping[str, int] = field(hash=False)
    transitions: TransitionCounts
    ngram_counts: Mapping[int, Mapping[tuple[str, ...], int]] = field(hash=False)
    n_rare_ngrams: Mapping[int, int] = field(hash=False)
    session_length: Summary
    session_duration_s: Summary
    interevent_s: Summary
    sojourn_s: Mapping[str, Summary] = field(hash=False)
    corpus_features: tuple[Feature, ...]
    graph: NavigationGraph

    def feature(self, name: str) -> Feature:
        """Look up a corpus feature by name.

        Raises:
            KeyError: If no feature has that name.
        """
        for feature in self.corpus_features:
            if feature.name == name:
                return feature
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data (FR-A.6)."""
        return {
            "dataset_id": self.dataset_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "ngram_order": self.ngram_order,
            "event_type_counts": dict(sorted(self.event_type_counts.items())),
            "token_counts": dict(sorted(self.token_counts.items())),
            "transitions": self.transitions.to_dict(),
            "ngrams": {
                str(n): [
                    {"ngram": list(gram), "count": count}
                    for gram, count in sorted(grams.items())
                ]
                for n, grams in sorted(self.ngram_counts.items())
            },
            "n_rare_ngrams": {str(n): c for n, c in sorted(self.n_rare_ngrams.items())},
            "session_length": self.session_length.to_dict(),
            "session_duration_s": self.session_duration_s.to_dict(),
            "interevent_s": self.interevent_s.to_dict(),
            "sojourn_s": {
                token: summary.to_dict() for token, summary in self.sojourn_s.items()
            },
            "corpus_features": [feature.to_dict() for feature in self.corpus_features],
            "graph": self.graph.to_dict(),
        }

    def to_markdown(self) -> str:
        """Human-readable summary with Markdown tables (FR-A.6)."""
        lines = [
            f"# Analysis of `{self.dataset_id}`",
            "",
            f"Fingerprint: `{self.dataset_fingerprint}`",
            "",
            "## Corpus",
            "",
            "| Feature | Value |",
            "| ------- | ----- |",
        ]
        lines += [f"| {f.name} | {_fmt(f.value)} |" for f in self.corpus_features]
        lines += [
            "",
            "## Distributions",
            "",
            "| Measure | Count | Mean | Median | P25 | P75 | Max |",
            "| ------- | ----- | ---- | ------ | --- | --- | --- |",
        ]
        for label, summary in (
            ("Session length (events)", self.session_length),
            ("Session duration (s)", self.session_duration_s),
            ("Inter-event time (s)", self.interevent_s),
        ):
            lines.append(
                f"| {label} | {summary.count} | {_fmt(summary.mean)} | "
                f"{_fmt(summary.median)} | {_fmt(summary.p25)} | "
                f"{_fmt(summary.p75)} | {_fmt(summary.max)} |"
            )
        lines += ["", "## Event types", "", "| Event type | Events |", "| --- | --- |"]
        lines += [
            f"| {name} | {count} |"
            for name, count in sorted(
                self.event_type_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        lines += [
            "",
            f"## Top {_TOP} transitions",
            "",
            "| From | To | Count |",
            "| ---- | -- | ----- |",
        ]
        lines += [
            f"| {a} | {b} | {count} |" for (a, b), count in self.graph.top_edges(_TOP)
        ]
        return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3g}" if abs(value) < 1e6 else f"{value:.3e}"
    return str(value)
