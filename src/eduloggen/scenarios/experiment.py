"""Experiment settings for ``generate --experiment`` (Level 2, M2).

One YAML/TOML/JSON file with optional ``controls``, ``calendar``, and
``anomalies`` sections. Full scenario files (profiles, outcomes) arrive in a
later milestone and will build on the same sections.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from eduloggen.config import load_file
from eduloggen.core import ConfigError, PathLike
from eduloggen.generators import BaseGenerator, SessionCalendar
from eduloggen.models import Annotations, Dataset, GeneratorModel, SyntheticDataset
from eduloggen.scenarios.check import ManipulationCheck, manipulation_check
from eduloggen.scenarios.controls import Controls, apply_controls
from eduloggen.scenarios.inject import AnomalySpec, inject_anomalies
from eduloggen.utils import derive_seed

__all__ = ["ExperimentResult", "ExperimentSettings", "run_experiment"]

_SECTIONS: Final = frozenset({"controls", "calendar", "anomalies"})


@dataclass(frozen=True, slots=True)
class ExperimentSettings:
    """Controls, calendar, and anomalies for one generation run."""

    controls: Controls
    calendar: SessionCalendar | None
    anomalies: tuple[AnomalySpec, ...]

    @property
    def is_empty(self) -> bool:
        """Whether nothing is requested."""
        return self.controls.is_empty and self.calendar is None and not self.anomalies

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentSettings:
        """Build from a mapping with optional ``controls``, ``calendar``, ``anomalies``.

        Raises:
            ConfigError: If a section is unknown or invalid, or all are empty.
        """
        unknown = sorted(set(data) - _SECTIONS)
        if unknown:
            raise ConfigError(
                f"unknown experiment sections: {', '.join(unknown)}",
                code="config_unknown_key",
                context={"keys": unknown, "allowed": sorted(_SECTIONS)},
            )
        anomalies = data.get("anomalies") or []
        if not isinstance(anomalies, list):
            raise ConfigError("anomalies must be a list", code="config_invalid_value")
        settings = cls(
            controls=Controls.from_dict(data.get("controls")),
            calendar=(
                SessionCalendar.from_dict(data["calendar"])
                if data.get("calendar")
                else None
            ),
            anomalies=tuple(AnomalySpec.from_dict(item) for item in anomalies),
        )
        if settings.is_empty:
            raise ConfigError(
                "the experiment file sets no controls, calendar, or anomalies",
                code="config_invalid_value",
            )
        return settings

    @classmethod
    def from_file(cls, path: PathLike) -> ExperimentSettings:
        """Load from a YAML, TOML, or JSON file."""
        return cls.from_dict(load_file(path))


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Output of :func:`run_experiment`.

    Attributes:
        dataset: Generated data (controlled, calendar applied, anomalies injected).
        annotations: Ground truth for injected anomalies (``None`` if none).
        check: Manipulation check (controls and calendar against a baseline).
        model: The controlled model that produced the data.
    """

    dataset: Dataset
    annotations: Annotations | None
    check: ManipulationCheck
    model: GeneratorModel


def run_experiment(
    generator: BaseGenerator,
    model: GeneratorModel,
    settings: ExperimentSettings,
    *,
    n_sessions: int,
    seed: int,
    id_strategy: Any = "remap",
) -> ExperimentResult:
    """Generate with controls, calendar, and anomalies, and check them.

    A baseline is drawn from the uncontrolled ``model`` with the same seed,
    size, and calendar; the manipulation check compares the two before
    anomalies are injected.
    """
    controlled_model = apply_controls(model, settings.controls)
    controlled: SyntheticDataset = generator.generate(
        controlled_model,
        n_sessions,
        seed,
        id_strategy=id_strategy,
        calendar=settings.calendar,
    )
    baseline = (
        controlled
        if settings.controls.is_empty
        else generator.generate(
            model, n_sessions, seed, id_strategy=id_strategy, calendar=settings.calendar
        )
    )
    dataset: Dataset = controlled
    annotations = None
    report = None
    if settings.anomalies:
        injected = inject_anomalies(
            controlled, settings.anomalies, seed=derive_seed(seed, "anomalies") or 0
        )
        dataset, annotations, report = (
            injected.dataset,
            injected.annotations,
            injected.report,
        )
    check = manipulation_check(
        baseline,
        controlled,
        controls=None if settings.controls.is_empty else settings.controls,
        calendar=settings.calendar,
        anomaly_report=report,
    )
    return ExperimentResult(
        dataset=dataset, annotations=annotations, check=check, model=controlled_model
    )
