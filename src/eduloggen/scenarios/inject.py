"""Inject labelled anomalies into a sessionized dataset (Level 2, M1)."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import timedelta
from statistics import median
from types import MappingProxyType
from typing import Any, TypeVar

from eduloggen.analysis import interevent_times, ngram_counts, session_sequences
from eduloggen.config import load_file
from eduloggen.core import ConfigError, PathLike
from eduloggen.generators.tokenization import (
    companion_counts,
    companion_field,
    detect_tokenization,
)
from eduloggen.models import Annotation, Annotations, Dataset, LogRecord, Session
from eduloggen.privacy import remap_ids_with_mapping
from eduloggen.scenarios.anomalies import InjectionContext, get_anomaly
from eduloggen.utils import derive_seed, make_rng

__all__ = [
    "AnomalySpec",
    "InjectionResult",
    "inject_anomalies",
    "load_anomaly_specs",
]

logger = logging.getLogger(__name__)

D = TypeVar("D", bound=Dataset)


@dataclass(frozen=True, slots=True)
class AnomalySpec:
    """Request to inject one anomaly type into a share of sessions.

    Attributes:
        type: Registered anomaly name.
        rate: Share of sessions to perturb, in ``(0, 1]``.
        params: Injector parameters (see each injector's ``defaults``).
    """

    type: str
    rate: float
    params: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), hash=False
    )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AnomalySpec:
        """Build from ``{"type": ..., "rate": ..., **params}``.

        Raises:
            ConfigError: If ``type`` or ``rate`` is missing or invalid.
        """
        if not isinstance(data, Mapping) or "type" not in data or "rate" not in data:
            raise ConfigError(
                "each anomaly needs 'type' and 'rate'", code="config_invalid_value"
            )
        rest = {k: v for k, v in data.items() if k not in ("type", "rate")}
        return cls(
            type=str(data["type"]), rate=data["rate"], params=MappingProxyType(rest)
        )


@dataclass(frozen=True, slots=True)
class InjectionResult:
    """Output of :func:`inject_anomalies`.

    Attributes:
        dataset: The perturbed dataset with freshly remapped ids.
        annotations: Ground truth for every injected anomaly.
        report: Per anomaly type: requested, injected, and not-applicable
            counts, plus resolved parameters.
    """

    dataset: Dataset
    annotations: Annotations
    report: Mapping[str, Any] = field(hash=False)


def load_anomaly_specs(path: PathLike) -> list[AnomalySpec]:
    """Read ``anomalies: [...]`` from a YAML/TOML/JSON file.

    Raises:
        ConfigError: If the file is invalid.
    """
    data = load_file(path)
    raw = data.get("anomalies")
    if not isinstance(raw, list) or not raw:
        raise ConfigError(
            "the file must contain a non-empty 'anomalies' list",
            code="config_invalid_value",
            context={"key": "anomalies"},
        )
    unknown = sorted(set(data) - {"anomalies"})
    if unknown:
        raise ConfigError(
            f"unknown keys: {', '.join(unknown)}",
            code="config_unknown_key",
            context={"keys": unknown},
        )
    return [AnomalySpec.from_dict(item) for item in raw]


def inject_anomalies(
    dataset: D,
    specs: Iterable[AnomalySpec | Mapping[str, Any]],
    *,
    seed: int,
    reference: Dataset | None = None,
) -> InjectionResult:
    """Perturb a share of sessions and record ground truth.

    Each selected session receives exactly one anomaly, so labels are
    unambiguous. Sessions are chosen at random (seeded) and are disjoint
    across anomaly types. When an injection makes a session longer, the
    learner's later sessions are shifted so sessions never overlap. All ids
    are remapped afterwards, so neither id formats nor ordering reveal which
    records were injected.

    Args:
        dataset: Sessionized dataset (synthetic or real).
        specs: Anomaly requests, as :class:`AnomalySpec` or dicts.
        seed: Seed for session choice, injection details, and id remapping.
        reference: Data defining "normal" transitions and timing; defaults
            to ``dataset`` itself.

    Returns:
        The perturbed dataset, its annotations, and an injection report.

    Raises:
        ConfigError: If a spec is invalid or rates add up to more than 1.
        AnalysisError: If the dataset is not sessionized.
    """
    session_sequences(dataset)
    requests = [
        s if isinstance(s, AnomalySpec) else AnomalySpec.from_dict(s) for s in specs
    ]
    if not requests:
        raise ConfigError("no anomalies requested", code="config_invalid_value")
    resolved = []
    for spec in requests:
        if (
            isinstance(spec.rate, bool)
            or not isinstance(spec.rate, int | float)
            or not 0 < spec.rate <= 1
        ):
            raise ConfigError(
                f"rate for {spec.type} must be in (0, 1]",
                code="config_invalid_value",
                context={"key": "rate", "anomaly": spec.type},
            )
        injector = get_anomaly(spec.type)
        resolved.append((spec, injector, injector.resolve(spec.params)))
    if sum(spec.rate for spec, _, _ in resolved) > 1 + 1e-9:
        raise ConfigError(
            "anomaly rates add up to more than 1", code="config_invalid_value"
        )

    context = _context(reference if reference is not None else dataset, dataset)
    rng = make_rng(seed, "anomalies", "inject")
    members = _members(dataset)
    pool = sorted(members)
    rng.shuffle(pool)
    replaced: dict[str, list[LogRecord]] = {}
    rows: list[Annotation] = []
    report: dict[str, Any] = {}
    cursor = 0
    for spec, injector, params in resolved:
        wanted = round(spec.rate * len(pool))
        done = skipped = 0
        while done < wanted and cursor < len(pool):
            session_id = pool[cursor]
            cursor += 1
            result = injector.inject(list(members[session_id]), params, context, rng)
            if result is None:
                skipped += 1
                continue
            done += 1
            replaced[session_id] = result.events
            details = {**result.parameters, "injected_events": len(result.affected)}
            rows.append(
                _row("session", session_id, injector.name, injector.category, details)
            )
            rows += [
                _row("event", event_id, injector.name, injector.category, {})
                for event_id in result.affected
            ]
        if done < wanted:
            logger.warning(
                "%s: injected %d of %d requested sessions", injector.name, done, wanted
            )
        report[injector.name] = {
            "category": injector.category,
            "rate": spec.rate,
            "requested": wanted,
            "injected": done,
            "not_applicable": skipped,
            "parameters": {k: v for k, v in params.items()},
        }

    perturbed = _rebuild(dataset, members, replaced, context)
    final, mapping = remap_ids_with_mapping(
        perturbed, derive_seed(seed, "anomalies", "ids")
    )
    annotations = Annotations(tuple(rows)).remap(
        {
            "learner": mapping.learners,
            "session": mapping.sessions,
            "event": mapping.events,
        }
    )
    report_view = {"sessions": len(pool), "anomalies": report}
    return InjectionResult(
        dataset=final, annotations=annotations, report=MappingProxyType(report_view)
    )


def _row(
    level: str, item_id: str, kind: str, category: str, parameters: dict[str, Any]
) -> Annotation:
    return Annotation(
        level=level,  # type: ignore[arg-type]
        id=item_id,
        annotation="anomaly",
        type=kind,
        category=category,  # type: ignore[arg-type]
        value=1,
        parameters=parameters,
    )


def _members(dataset: Dataset) -> dict[str, list[LogRecord]]:
    members: defaultdict[str, list[LogRecord]] = defaultdict(list)
    for event in dataset.sorted_events():
        if event.session_id is not None:
            members[event.session_id].append(event)
    return dict(members)


def _context(reference: Dataset, target: Dataset) -> InjectionContext:
    tokenization = detect_tokenization(target)
    other = companion_field(tokenization)
    companions = companion_counts(target, tokenization)
    sequences = session_sequences(reference)
    gaps = interevent_times(reference)
    vocabulary = sorted(
        {t for s in session_sequences(target) for t in s}
        | {t for s in sequences for t in s}
    )
    return InjectionContext(
        tokenization=tokenization,
        companion_field=other,
        companions=MappingProxyType(
            {k: tuple(sorted(v)) for k, v in companions.items()}
        ),
        vocabulary=tuple(vocabulary),
        observed_transitions=frozenset((a, b) for a, b in ngram_counts(sequences, 2)),
        typical_gap_s=median(gaps) if gaps else 30.0,
    )


def _rebuild(
    dataset: D,
    members: Mapping[str, list[LogRecord]],
    replaced: Mapping[str, list[LogRecord]],
    context: InjectionContext,
) -> D:
    """Swap in perturbed sessions and push later sessions of a learner forward."""
    sessions_by_learner: defaultdict[str, list[str]] = defaultdict(list)
    for session_id, events in members.items():
        sessions_by_learner[events[0].learner_id].append(session_id)
    new_events: list[LogRecord] = []
    new_sessions: list[Session] = []
    for session_ids in sessions_by_learner.values():
        session_ids.sort(key=lambda sid: (members[sid][0].timestamp, sid))
        offset = timedelta(0)
        previous_end = None
        for session_id in session_ids:
            events = replaced.get(session_id, members[session_id])
            events = [replace(e, timestamp=e.timestamp + offset) for e in events]
            if previous_end is not None and events[0].timestamp <= previous_end:
                push = previous_end - events[0].timestamp + timedelta(seconds=1)
                offset += push
                events = [replace(e, timestamp=e.timestamp + push) for e in events]
            previous_end = events[-1].timestamp
            new_events.extend(events)
            new_sessions.append(
                Session.from_events(
                    session_id, events, token_field=context.tokenization
                )
            )
    orphans = [e for e in dataset.events if e.session_id is None]
    return replace(
        dataset, events=tuple(new_events + orphans), sessions=tuple(new_sessions)
    )
