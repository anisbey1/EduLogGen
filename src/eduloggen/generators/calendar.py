"""When sessions start: a weighted course calendar (Level 2, design §6).

A :class:`SessionCalendar` draws session start times from a date range with
relative weights per hour of day (in the calendar's timezone), per weekday,
and surges in the days before deadlines. Generators use it instead of their
learned start-time model when one is given.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from typing import Any, Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from eduloggen.core import ConfigError

__all__ = ["Deadline", "SessionCalendar"]

_Date = date

_KEYS: Final = frozenset(
    {"start", "weeks", "days", "timezone", "hours", "weekdays", "deadlines"}
)


@dataclass(frozen=True, slots=True)
class Deadline:
    """Extra activity in the days before a date.

    Attributes:
        date: The deadline day.
        surge: Activity multiplier on the deadline and the preceding days.
        days_before: How many days before the deadline the surge starts.
    """

    date: _Date
    surge: float
    days_before: int = 3

    def factor(self, day: _Date) -> float:
        """Multiplier this deadline applies to ``day`` (1 outside its window)."""
        lead = (self.date - day).days
        return self.surge if 0 <= lead <= self.days_before else 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "date": self.date.isoformat(),
            "surge": self.surge,
            "days_before": self.days_before,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionCalendar:
    """Distribution of session start times.

    Attributes:
        start: First day of the period.
        days: Length of the period in days.
        timezone: IANA zone in which ``hours`` and ``weekdays`` apply.
        hours: 24 non-negative relative weights, hour 0 to 23.
        weekdays: 7 non-negative relative weights, Monday to Sunday.
        deadlines: Activity surges before deadlines.
    """

    start: date
    days: int
    timezone: str = "UTC"
    hours: tuple[float, ...] = (1.0,) * 24
    weekdays: tuple[float, ...] = (1.0,) * 7
    deadlines: tuple[Deadline, ...] = ()
    _zone: tzinfo = field(init=False, repr=False, compare=False, hash=False)
    _day_weights: tuple[float, ...] = field(
        init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        """Validate and precompute day weights.

        Raises:
            ConfigError: If a value is invalid or no time has positive weight.
        """
        if (
            isinstance(self.days, bool)
            or not isinstance(self.days, int)
            or self.days < 1
        ):
            raise _bad("days", "must be a positive integer")
        _weights("hours", self.hours, 24)
        _weights("weekdays", self.weekdays, 7)
        try:
            zone: tzinfo = ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError):
            raise _bad("timezone", "is not a known IANA timezone") from None
        day_weights = tuple(
            self.weekdays[day.weekday()]
            * _product(d.factor(day) for d in self.deadlines)
            for day in (self.start + timedelta(days=i) for i in range(self.days))
        )
        if not any(day_weights) or not any(self.hours):
            raise _bad("hours", "no day or hour of the calendar has positive weight")
        object.__setattr__(self, "hours", tuple(float(h) for h in self.hours))
        object.__setattr__(self, "weekdays", tuple(float(w) for w in self.weekdays))
        object.__setattr__(self, "deadlines", tuple(self.deadlines))
        object.__setattr__(self, "_zone", zone)
        object.__setattr__(self, "_day_weights", day_weights)

    @property
    def end(self) -> date:
        """Last day of the period."""
        return self.start + timedelta(days=self.days - 1)

    def day_weight(self, day: date) -> float:
        """Relative weight of ``day`` (0 outside the period)."""
        offset = (day - self.start).days
        return self._day_weights[offset] if 0 <= offset < self.days else 0.0

    def sample(self, rng: random.Random) -> datetime:
        """Draw one start time (UTC)."""
        day = self.start + timedelta(
            days=rng.choices(range(self.days), self._day_weights)[0]
        )
        hour = rng.choices(range(24), self.hours)[0]
        local = datetime.combine(day, time(hour), tzinfo=self._zone)
        return (local + timedelta(seconds=rng.uniform(0, 3600))).astimezone(UTC)

    def next_allowed(self, earliest: datetime) -> datetime:
        """First moment at or after ``earliest`` in a positive-weight hour and day.

        Used to move a session that would overlap the learner's previous one
        by as little as possible, so the calendar's distribution is kept and
        closed hours stay closed. Returns ``earliest`` unchanged if the period
        has no allowed time left.
        """
        local = earliest.astimezone(self._zone)
        day = local.date()
        while day <= self.end:
            if self.day_weight(day) > 0:
                for hour in range(24):
                    if self.hours[hour] <= 0:
                        continue
                    slot = datetime.combine(day, time(hour), tzinfo=self._zone)
                    if slot + timedelta(hours=1) > local:
                        return max(slot, local).astimezone(UTC)
            day += timedelta(days=1)
        return earliest

    def local_time(self, moment: datetime) -> datetime:
        """``moment`` in the calendar's timezone."""
        return moment.astimezone(self._zone)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data accepted by :meth:`from_dict`."""
        return {
            "start": self.start.isoformat(),
            "days": self.days,
            "timezone": self.timezone,
            "hours": list(self.hours),
            "weekdays": list(self.weekdays),
            "deadlines": [d.to_dict() for d in self.deadlines],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SessionCalendar:
        """Build from plain data, e.g. a ``calendar:`` YAML section.

        ``weeks`` may be given instead of ``days``.

        Raises:
            ConfigError: If keys are unknown or values are invalid.
        """
        if not isinstance(data, Mapping):
            raise _bad("calendar", "must be a mapping")
        unknown = sorted(set(data) - _KEYS)
        if unknown:
            raise ConfigError(
                f"unknown calendar keys: {', '.join(unknown)}",
                code="config_unknown_key",
                context={"keys": unknown, "allowed": sorted(_KEYS)},
            )
        if "days" in data and "weeks" in data:
            raise _bad("weeks", "use either days or weeks")
        if "weeks" in data:
            weeks = data["weeks"]
            if isinstance(weeks, bool) or not isinstance(weeks, int):
                raise _bad("weeks", "must be an integer")
            days = weeks * 7
        else:
            days = data.get("days", 0)
        deadlines = []
        for item in data.get("deadlines") or ():
            if not isinstance(item, Mapping) or set(item) - {
                "date",
                "surge",
                "days_before",
            }:
                raise _bad(
                    "deadlines", "entries need date, surge, and optional days_before"
                )
            surge = item.get("surge")
            before = item.get("days_before", 3)
            if (
                isinstance(surge, bool)
                or not isinstance(surge, int | float)
                or surge <= 0
            ):
                raise _bad("deadlines", "surge must be a positive number")
            if isinstance(before, bool) or not isinstance(before, int) or before < 0:
                raise _bad("deadlines", "days_before must be a non-negative integer")
            deadlines.append(
                Deadline(_date(item.get("date"), "deadlines"), float(surge), before)
            )
        return cls(
            start=_date(data.get("start"), "start"),
            days=days,
            timezone=str(data.get("timezone", "UTC")),
            hours=tuple(data.get("hours", (1.0,) * 24)),
            weekdays=tuple(data.get("weekdays", (1.0,) * 7)),
            deadlines=tuple(deadlines),
        )


def _product(values: Any) -> float:
    result = 1.0
    for value in values:
        result *= value
    return result


def _weights(key: str, values: Sequence[Any], length: int) -> None:
    if len(values) != length or not all(
        isinstance(v, int | float) and not isinstance(v, bool) and v >= 0
        for v in values
    ):
        raise _bad(key, f"must be {length} non-negative numbers")


def _date(value: Any, key: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise _bad(key, "must be a date like 2026-09-01")


def _bad(key: str, reason: str) -> ConfigError:
    return ConfigError(
        f"calendar.{key} {reason}",
        code="config_invalid_value",
        context={"key": f"calendar.{key}"},
    )
