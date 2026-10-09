"""Anomaly injectors (Level 2, design §7).

Each injector perturbs one session and reports what it changed. Categories:

- ``invalid_workflow``: the result breaks the application's event model.
- ``unusual_valid``: the result is valid but behaviourally rare.

Anomalies describe data patterns, never intentions: nothing here labels a
learner as cheating or misbehaving.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import timedelta
from itertools import pairwise
from typing import Any, ClassVar

from eduloggen.core import ConfigError, PluginError
from eduloggen.models import AnomalyCategory, LogRecord, TokenField

__all__ = [
    "ANOMALIES",
    "AbnormalTiming",
    "BaseAnomaly",
    "EventFrequency",
    "Inactivity",
    "Injection",
    "InjectionContext",
    "Repetition",
    "UnexpectedTransition",
    "available_anomalies",
    "get_anomaly",
    "register_anomaly",
]


@dataclass(frozen=True, slots=True)
class InjectionContext:
    """What injectors know about the data they perturb.

    Attributes:
        tokenization: Event field the session tokens come from.
        companion_field: The other of ``event_type`` / ``activity_id``.
        companions: Token to observed companion values (for new events).
        vocabulary: Known tokens, sorted.
        observed_transitions: Token pairs seen in the reference data.
        typical_gap_s: Median gap between events in the reference data.
    """

    tokenization: TokenField
    companion_field: TokenField
    companions: Mapping[str, Sequence[str]] = field(hash=False)
    vocabulary: tuple[str, ...]
    observed_transitions: frozenset[tuple[str, str]]
    typical_gap_s: float

    def new_event(
        self, template: LogRecord, token: str, rng: random.Random, offset_s: float
    ) -> LogRecord:
        """An event like ``template`` but with ``token``, ``offset_s`` later."""
        companion = rng.choice(
            list(
                self.companions.get(token) or [getattr(template, self.companion_field)]
            )
        )
        return replace(
            template,
            event_id=f"injected-{rng.getrandbits(64):016x}",
            timestamp=template.timestamp + timedelta(seconds=offset_s),
            score=None,
            success=None,
            duration_ms=None,
            **{self.tokenization: token, self.companion_field: companion},
        )


@dataclass(frozen=True, slots=True)
class Injection:
    """Result of perturbing one session.

    Attributes:
        events: The session's new events, in time order (ids of injected
            events are temporary and replaced later).
        affected: Ids (in ``events``) of inserted or modified events.
        parameters: JSON-compatible description of what was done.
    """

    events: list[LogRecord]
    affected: list[str]
    parameters: dict[str, Any]


class BaseAnomaly(ABC):
    """An injector for one anomaly type.

    Subclasses set :attr:`name`, :attr:`category`, :attr:`defaults`, and
    implement :meth:`inject`, returning ``None`` when the session is not
    suitable (e.g. too short).
    """

    name: ClassVar[str]
    category: ClassVar[AnomalyCategory]
    description: ClassVar[str] = ""
    defaults: ClassVar[Mapping[str, Any]] = {}

    def resolve(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Merge parameters with defaults and validate them.

        Raises:
            ConfigError: If a key is unknown or a value is invalid.
        """
        unknown = sorted(set(params) - set(self.defaults))
        if unknown:
            raise ConfigError(
                f"unknown parameters for anomaly {self.name}: {', '.join(unknown)}",
                code="config_unknown_key",
                context={"anomaly": self.name, "allowed": sorted(self.defaults)},
            )
        merged = {**self.defaults, **params}
        self.validate(merged)
        return merged

    def validate(self, params: dict[str, Any]) -> None:  # noqa: B027 - optional hook
        """Check resolved parameters (override as needed)."""

    @abstractmethod
    def inject(
        self,
        events: list[LogRecord],
        params: Mapping[str, Any],
        context: InjectionContext,
        rng: random.Random,
    ) -> Injection | None:
        """Perturb one session's events (sorted by time)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _positive(
    params: Mapping[str, Any], key: str, *, integer: bool = False, minimum: float = 0.0
) -> None:
    value = params[key]
    ok = (
        isinstance(value, int) if integer else isinstance(value, int | float)
    ) and not isinstance(value, bool)
    if not ok or value <= minimum:
        kind = "an integer" if integer else "a number"
        raise ConfigError(
            f"anomalies.{key} must be {kind} > {minimum:g}",
            code="config_invalid_value",
            context={"key": key},
        )


def _token(token_field: TokenField, event: LogRecord) -> str:
    return str(getattr(event, token_field))


def _shift(events: Sequence[LogRecord], seconds: float) -> list[LogRecord]:
    return [
        replace(e, timestamp=e.timestamp + timedelta(seconds=seconds)) for e in events
    ]


def _insert(
    events: list[LogRecord],
    position: int,
    tokens: Sequence[str],
    gap_s: float,
    context: InjectionContext,
    rng: random.Random,
) -> tuple[list[LogRecord], list[str]]:
    """Insert ``tokens`` after ``events[position]``, ``gap_s`` apart; shift the rest."""
    anchor = events[position]
    inserted = [
        context.new_event(anchor, token, rng, gap_s * (i + 1))
        for i, token in enumerate(tokens)
    ]
    rest = _shift(events[position + 1 :], gap_s * len(tokens))
    return [*events[: position + 1], *inserted, *rest], [e.event_id for e in inserted]


# ---------------------------------------------------------------------------
# P0 anomalies
# ---------------------------------------------------------------------------


class EventFrequency(BaseAnomaly):
    """A burst of one event type in a short window (e.g. 20 AI requests in 3 min)."""

    name: ClassVar[str] = "event_frequency"
    category: ClassVar[AnomalyCategory] = "unusual_valid"
    description = "Burst of one token within a short time window"
    defaults: ClassVar[Mapping[str, Any]] = {
        "token": None,
        "count": 20,
        "window_s": 180.0,
    }

    def validate(self, params: dict[str, Any]) -> None:
        """Count must be at least 2 and the window positive."""
        _positive(params, "count", integer=True, minimum=1)
        _positive(params, "window_s")

    def inject(
        self,
        events: list[LogRecord],
        params: Mapping[str, Any],
        context: InjectionContext,
        rng: random.Random,
    ) -> Injection | None:
        """Insert ``count`` copies of a token spread over ``window_s``."""
        token = params["token"] or rng.choice(context.vocabulary)
        if token not in context.vocabulary:
            raise ConfigError(
                f"event_frequency token {token!r} is not in the vocabulary",
                code="config_invalid_value",
                context={"key": "token"},
            )
        count = int(params["count"])
        gap = float(params["window_s"]) / count
        position = rng.randrange(len(events))
        new, affected = _insert(events, position, [token] * count, gap, context, rng)
        return Injection(
            new,
            affected,
            {"token": token, "count": count, "window_s": float(params["window_s"])},
        )


class AbnormalTiming(BaseAnomaly):
    """A session completed implausibly fast (all gaps compressed)."""

    name: ClassVar[str] = "abnormal_timing"
    category: ClassVar[AnomalyCategory] = "unusual_valid"
    description = "All gaps in a session shrunk by a factor"
    defaults: ClassVar[Mapping[str, Any]] = {"factor": 0.05, "min_events": 3}

    def validate(self, params: dict[str, Any]) -> None:
        """Factor must be in (0, 1); min_events at least 2."""
        _positive(params, "factor")
        if params["factor"] >= 1:
            raise ConfigError(
                "anomalies.factor must be below 1",
                code="config_invalid_value",
                context={"key": "factor"},
            )
        _positive(params, "min_events", integer=True, minimum=1)

    def inject(
        self,
        events: list[LogRecord],
        params: Mapping[str, Any],
        context: InjectionContext,
        rng: random.Random,
    ) -> Injection | None:
        """Scale every gap by ``factor`` (at least 1 ms apart)."""
        if len(events) < int(params["min_events"]):
            return None
        factor = float(params["factor"])
        start = events[0].timestamp
        new = [events[0]]
        for previous, event in pairwise(events):
            gap = max(
                (event.timestamp - previous.timestamp).total_seconds() * factor, 0.001
            )
            new.append(
                replace(event, timestamp=new[-1].timestamp + timedelta(seconds=gap))
            )
        before = (events[-1].timestamp - start).total_seconds()
        after = (new[-1].timestamp - start).total_seconds()
        return Injection(
            new,
            [e.event_id for e in new[1:]],
            {"factor": factor, "duration_before_s": before, "duration_after_s": after},
        )


class UnexpectedTransition(BaseAnomaly):
    """A transition the event model does not allow (e.g. ``submit`` → ``start``)."""

    name: ClassVar[str] = "unexpected_transition"
    category: ClassVar[AnomalyCategory] = "invalid_workflow"
    description = "Insert a token pair never observed in the reference data"
    defaults: ClassVar[Mapping[str, Any]] = {"transitions": None}

    def validate(self, params: dict[str, Any]) -> None:
        """Explicit transitions must be a list of [from, to] pairs."""
        pairs = params["transitions"]
        if pairs is not None and (
            not isinstance(pairs, list | tuple)
            or not pairs
            or not all(
                isinstance(p, list | tuple)
                and len(p) == 2
                and all(isinstance(t, str) for t in p)
                for p in pairs
            )
        ):
            raise ConfigError(
                "anomalies.transitions must be a list of [from, to] token pairs",
                code="config_invalid_value",
                context={"key": "transitions"},
            )

    def inject(
        self,
        events: list[LogRecord],
        params: Mapping[str, Any],
        context: InjectionContext,
        rng: random.Random,
    ) -> Injection | None:
        """Insert the pair ``from, to`` at a random point."""
        if params["transitions"] is not None:
            candidates = [tuple(p) for p in params["transitions"]]
        else:
            candidates = [
                (a, b)
                for a in context.vocabulary
                for b in context.vocabulary
                if (a, b) not in context.observed_transitions
            ]
        if not candidates:
            return None
        first, second = rng.choice(sorted(candidates))
        position = rng.randrange(len(events))
        gap = max(context.typical_gap_s, 1.0)
        new, affected = _insert(events, position, [first, second], gap, context, rng)
        source = "configured" if params["transitions"] is not None else "unobserved"
        return Injection(new, affected, {"from": first, "to": second, "source": source})


class Repetition(BaseAnomaly):
    """Excessive repetition: one token repeated, or two tokens alternating."""

    name: ClassVar[str] = "repetition"
    category: ClassVar[AnomalyCategory] = "unusual_valid"
    description = "A loop of one token or ping-pong between two tokens"
    defaults: ClassVar[Mapping[str, Any]] = {
        "pattern": "ping_pong",
        "length": 12,
        "gap_s": 5.0,
    }

    def validate(self, params: dict[str, Any]) -> None:
        """Pattern must be loop or ping_pong; length at least 3."""
        if params["pattern"] not in ("loop", "ping_pong"):
            raise ConfigError(
                "anomalies.pattern must be loop or ping_pong",
                code="config_invalid_value",
                context={"key": "pattern"},
            )
        _positive(params, "length", integer=True, minimum=2)
        _positive(params, "gap_s")

    def inject(
        self,
        events: list[LogRecord],
        params: Mapping[str, Any],
        context: InjectionContext,
        rng: random.Random,
    ) -> Injection | None:
        """Insert the repeated pattern, preferring tokens from the session."""
        own = sorted({_token(context.tokenization, e) for e in events})
        pool = (
            own
            if (params["pattern"] == "loop" or len(own) >= 2)
            else list(context.vocabulary)
        )
        if params["pattern"] == "ping_pong" and len(pool) < 2:
            return None
        if params["pattern"] == "loop":
            tokens = [rng.choice(pool)] * int(params["length"])
        else:
            a, b = rng.sample(pool, 2)
            tokens = [a if i % 2 == 0 else b for i in range(int(params["length"]))]
        position = rng.randrange(len(events))
        new, affected = _insert(
            events, position, tokens, float(params["gap_s"]), context, rng
        )
        return Injection(
            new,
            affected,
            {
                "pattern": params["pattern"],
                "tokens": sorted(set(tokens)),
                "length": len(tokens),
            },
        )


class Inactivity(BaseAnomaly):
    """A long pause inside a session (e.g. 40 minutes without activity)."""

    name: ClassVar[str] = "inactivity"
    category: ClassVar[AnomalyCategory] = "unusual_valid"
    description = "A long gap between two consecutive events of a session"
    defaults: ClassVar[Mapping[str, Any]] = {"gap_s": 2400.0}

    def validate(self, params: dict[str, Any]) -> None:
        """Gap must be positive."""
        _positive(params, "gap_s")

    def inject(
        self,
        events: list[LogRecord],
        params: Mapping[str, Any],
        context: InjectionContext,
        rng: random.Random,
    ) -> Injection | None:
        """Delay everything after a random event by ``gap_s``."""
        if len(events) < 2:
            return None
        cut = rng.randrange(1, len(events))
        gap = float(params["gap_s"])
        new = [*events[:cut], *_shift(events[cut:], gap)]
        return Injection(
            new, [new[cut].event_id], {"gap_s": gap, "after_position": cut}
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ANOMALIES: dict[str, BaseAnomaly] = {
    cls.name: cls()
    for cls in (
        EventFrequency,
        AbnormalTiming,
        UnexpectedTransition,
        Repetition,
        Inactivity,
    )
}
BUILTIN_ANOMALIES: frozenset[str] = frozenset(ANOMALIES)


def register_anomaly(anomaly: BaseAnomaly, *, replace: bool = False) -> None:
    """Add an anomaly injector.

    Raises:
        PluginError: If it is not a :class:`BaseAnomaly`, has an invalid
            name or category, or the name is built in or taken.
    """
    if not isinstance(anomaly, BaseAnomaly):
        raise PluginError(
            "anomalies must be BaseAnomaly instances", code="plugin_contract_violation"
        )
    name = getattr(anomaly, "name", None)
    if not isinstance(name, str) or not name.isidentifier():
        raise PluginError(
            "anomaly names must be identifiers",
            code="plugin_invalid_name",
            context={"name": repr(name)},
        )
    if anomaly.category not in ("invalid_workflow", "unusual_valid"):
        raise PluginError(
            "anomaly category must be invalid_workflow or unusual_valid",
            code="plugin_contract_violation",
        )
    if name in BUILTIN_ANOMALIES or (name in ANOMALIES and not replace):
        raise PluginError(
            f"anomaly {name!r} is already registered",
            code="plugin_duplicate",
            context={"name": name},
        )
    ANOMALIES[name] = anomaly


def available_anomalies() -> list[str]:
    """Registered anomaly names, sorted."""
    return sorted(ANOMALIES)


def get_anomaly(name: str) -> BaseAnomaly:
    """Look up an injector.

    Raises:
        ConfigError: If the name is unknown.
    """
    try:
        return ANOMALIES[name]
    except KeyError:
        raise ConfigError(
            f"unknown anomaly {name!r}",
            code="anomaly_unknown",
            context={"name": name, "available": available_anomalies()},
        ) from None
