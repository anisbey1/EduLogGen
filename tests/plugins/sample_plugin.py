"""A sample third-party plugin module, as an external package would ship it."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from eduloggen.benchmark import BenchmarkProtocol
from eduloggen.generators import IndependentGenerator
from eduloggen.io import BaseReader
from eduloggen.models import Dataset
from eduloggen.validation import BaseMetric, ValidationContext


class EchoGenerator(IndependentGenerator):
    """Baseline under another name (stands in for a real model)."""

    name: ClassVar[str] = "echo"
    tags: ClassVar[frozenset[str]] = frozenset({"baseline", "third_party"})


class SessionRatio(BaseMetric):
    """Synthetic sessions per real session."""

    name: ClassVar[str] = "session_ratio"
    category: ClassVar[Any] = "utility"
    direction: ClassVar[Any] = "higher_better"

    def _compute(
        self, real: Dataset, synthetic: Dataset, context: ValidationContext
    ) -> tuple[float | None, dict[str, Any]]:
        return synthetic.n_sessions / max(real.n_sessions, 1), {}


class PipeReader(BaseReader):
    """Reads ``key=value|key=value`` lines."""

    format: ClassVar[str] = "pipe"
    suffixes: ClassVar[tuple[str, ...]] = (".pipe",)

    def _iter_rows(self, path: Path) -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield dict(part.split("=", 1) for part in line.strip().split("|"))


def event_count_plot(real: Dataset, synthetic: Dataset | None) -> Any:
    from eduloggen.visualization.base import new_figure

    figure, ax = new_figure()
    datasets = [real] + ([synthetic] if synthetic else [])
    ax.bar(range(len(datasets)), [d.n_events for d in datasets])
    return figure


quick_check = BenchmarkProtocol(
    name="quick_check",
    description="One seed, two metrics.",
    repeats=1,
    metrics=("event_type_tvd", "bigram_tvd"),
)


# --- broken plugins -------------------------------------------------------


class NotAGenerator:
    name = "not_a_generator"


class Misnamed(IndependentGenerator):
    name: ClassVar[str] = "something_else"


def crashing_factory() -> Any:
    raise RuntimeError("boom")


bad_protocol = BenchmarkProtocol(
    name="bad_protocol", description="", metrics=("no_such_metric",)
)
