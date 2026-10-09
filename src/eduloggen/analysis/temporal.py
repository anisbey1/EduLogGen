"""Activity over time: hours, weekdays, weeks, and deadlines.

All clock-based statistics use the given IANA timezone, so "evening" means the
learners' evening rather than UTC. Weeks are numbered from the week of the
dataset's first session (week 1), which makes courses with different start
dates comparable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, tzinfo
from itertools import pairwise
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from eduloggen.analysis.sequences import session_sequences
from eduloggen.analysis.stats import Summary, describe
from eduloggen.core import AnalysisError
from eduloggen.models import Dataset, Session

__all__ = [
    "DeadlineEffect",
    "TemporalProfile",
    "deadline_effects",
    "resolve_timezone",
    "temporal_profile",
    "week_index",
]

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def resolve_timezone(name: str) -> tzinfo:
    """Look up an IANA timezone.

    Raises:
        AnalysisError: If the name is unknown.
    """
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise AnalysisError(
            "unknown IANA timezone",
            code="analysis_invalid_parameter",
            context={"timezone": name},
        ) from None


def week_index(moment: datetime, first: date, zone: tzinfo) -> int:
    """1-based week of ``moment``, counted from the week containing ``first``."""
    monday = first - timedelta(days=first.weekday())
    return (moment.astimezone(zone).date() - monday).days // 7 + 1


@dataclass(frozen=True, slots=True, kw_only=True)
class TemporalProfile:
    """When learners are active.

    Attributes:
        timezone: Zone used for hours, weekdays, and days.
        events_by_hour, sessions_by_hour: 24 counts (sessions by start hour).
        events_by_weekday, sessions_by_weekday: 7 counts, Monday first.
        heatmap: 7 x 24 session starts (weekday rows, hour columns).
        weekly: Per week from the first session: events, sessions, active
            learners.
        daily_sessions: Session starts per local date.
        between_sessions_s: Gap from a session's end to the same learner's
            next session.
        first_day, last_day: Local dates of the first and last session.
    """

    timezone: str
    events_by_hour: tuple[int, ...]
    sessions_by_hour: tuple[int, ...]
    events_by_weekday: tuple[int, ...]
    sessions_by_weekday: tuple[int, ...]
    heatmap: tuple[tuple[int, ...], ...]
    weekly: tuple[dict[str, int], ...] = field(hash=False)
    daily_sessions: dict[str, int] = field(hash=False)
    between_sessions_s: Summary
    first_day: str | None
    last_day: str | None

    @property
    def peak_hour(self) -> int | None:
        """Hour with the most session starts (``None`` without sessions)."""
        return (
            max(range(24), key=lambda h: self.sessions_by_hour[h])
            if any(self.sessions_by_hour)
            else None
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "timezone": self.timezone,
            "events_by_hour": list(self.events_by_hour),
            "sessions_by_hour": list(self.sessions_by_hour),
            "events_by_weekday": list(self.events_by_weekday),
            "sessions_by_weekday": list(self.sessions_by_weekday),
            "heatmap": [list(row) for row in self.heatmap],
            "weekly": list(self.weekly),
            "daily_sessions": dict(self.daily_sessions),
            "between_sessions_s": self.between_sessions_s.to_dict(),
            "first_day": self.first_day,
            "last_day": self.last_day,
            "peak_hour": self.peak_hour,
        }

    def _four_hour_shares(self, total: int) -> list[float]:
        hours = self.sessions_by_hour
        return [sum(hours[start : start + 4]) / total for start in range(0, 24, 4)]

    def to_markdown(self) -> str:
        """Summary tables: weekdays, hours, weeks."""
        total = sum(self.sessions_by_weekday) or 1
        lines = [
            f"## Activity over time ({self.timezone})",
            "",
            f"Sessions from {self.first_day} to {self.last_day}; peak start hour "
            f"{self.peak_hour}; median gap between a learner's sessions "
            f"{_hours(self.between_sessions_s.median)}.",
            "",
            "| Weekday | " + " | ".join(WEEKDAYS) + " |",
            "| --- | " + " | ".join("---" for _ in WEEKDAYS) + " |",
            "| Sessions | "
            + " | ".join(f"{c / total:.0%}" for c in self.sessions_by_weekday)
            + " |",
            "",
            "| Hours | "
            + " | ".join(f"{h:02d}-{h + 3:02d}" for h in range(0, 24, 4))
            + " |",
            "| --- | " + " | ".join("---" for _ in range(6)) + " |",
            "| Sessions | "
            + " | ".join(f"{share:.0%}" for share in self._four_hour_shares(total))
            + " |",
            "",
            "| Week | Sessions | Events | Active learners |",
            "| ---- | -------- | ------ | --------------- |",
        ]
        lines += [
            f"| {w['week']} | {w['sessions']} | {w['events']} | {w['learners']} |"
            for w in self.weekly
        ]
        return "\n".join(lines) + "\n"


def temporal_profile(dataset: Dataset, *, timezone: str = "UTC") -> TemporalProfile:
    """Count activity by hour, weekday, week, and day.

    Raises:
        AnalysisError: If the dataset is not sessionized or the zone is unknown.
    """
    session_sequences(dataset)
    zone = resolve_timezone(timezone)
    sessions = sorted(
        dataset.sessions or (), key=lambda s: (s.start_time, s.session_id)
    )
    events_by_hour = [0] * 24
    events_by_weekday = [0] * 7
    for event in dataset.events:
        local = event.timestamp.astimezone(zone)
        events_by_hour[local.hour] += 1
        events_by_weekday[local.weekday()] += 1
    sessions_by_hour = [0] * 24
    sessions_by_weekday = [0] * 7
    heatmap = [[0] * 24 for _ in range(7)]
    daily: Counter[str] = Counter()
    for session in sessions:
        local = session.start_time.astimezone(zone)
        sessions_by_hour[local.hour] += 1
        sessions_by_weekday[local.weekday()] += 1
        heatmap[local.weekday()][local.hour] += 1
        daily[local.date().isoformat()] += 1
    weekly = _weekly(dataset, sessions, zone)
    return TemporalProfile(
        timezone=timezone,
        events_by_hour=tuple(events_by_hour),
        sessions_by_hour=tuple(sessions_by_hour),
        events_by_weekday=tuple(events_by_weekday),
        sessions_by_weekday=tuple(sessions_by_weekday),
        heatmap=tuple(tuple(row) for row in heatmap),
        weekly=weekly,
        daily_sessions=dict(sorted(daily.items())),
        between_sessions_s=describe(_between_sessions(sessions)),
        first_day=min(daily) if daily else None,
        last_day=max(daily) if daily else None,
    )


@dataclass(frozen=True, slots=True)
class DeadlineEffect:
    """Activity before a deadline compared with the rest of the period.

    Attributes:
        deadline: The deadline date.
        days_before: Window length before (and including) the deadline.
        window_per_day: Mean session starts per day in the window.
        other_per_day: Mean session starts per other day of the period.
        ratio: ``window_per_day / other_per_day`` (``None`` if undefined).
    """

    deadline: str
    days_before: int
    window_per_day: float
    other_per_day: float
    ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "deadline": self.deadline,
            "days_before": self.days_before,
            "window_per_day": self.window_per_day,
            "other_per_day": self.other_per_day,
            "ratio": self.ratio,
        }


def deadline_effects(
    dataset: Dataset,
    deadlines: Iterable[date | str],
    *,
    days_before: int = 3,
    timezone: str = "UTC",
) -> tuple[DeadlineEffect, ...]:
    """How much more active learners are in the days before each deadline.

    Raises:
        AnalysisError: If a date is malformed, ``days_before`` is negative, or
            the dataset is not sessionized.
    """
    if days_before < 0:
        raise AnalysisError(
            "days_before must be non-negative", code="analysis_invalid_parameter"
        )
    profile = temporal_profile(dataset, timezone=timezone)
    if profile.first_day is None or profile.last_day is None:
        return ()
    first = date.fromisoformat(profile.first_day)
    last = date.fromisoformat(profile.last_day)
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    counts = {date.fromisoformat(k): v for k, v in profile.daily_sessions.items()}
    effects = []
    for item in deadlines:
        try:
            deadline = item if isinstance(item, date) else date.fromisoformat(str(item))
        except ValueError:
            raise AnalysisError(
                "deadlines must be dates like 2026-10-16",
                code="analysis_invalid_parameter",
            ) from None
        window = {deadline - timedelta(days=i) for i in range(days_before + 1)} & set(
            days
        )
        others = [d for d in days if d not in window]
        window_rate = (
            sum(counts.get(d, 0) for d in window) / len(window) if window else 0.0
        )
        other_rate = (
            sum(counts.get(d, 0) for d in others) / len(others) if others else 0.0
        )
        effects.append(
            DeadlineEffect(
                deadline=deadline.isoformat(),
                days_before=days_before,
                window_per_day=window_rate,
                other_per_day=other_rate,
                ratio=window_rate / other_rate if other_rate and window else None,
            )
        )
    return tuple(effects)


def _weekly(
    dataset: Dataset, sessions: Sequence[Session], zone: tzinfo
) -> tuple[dict[str, int], ...]:
    if not sessions:
        return ()
    first = sessions[0].start_time.astimezone(zone).date()
    session_weeks: Counter[int] = Counter()
    learners: defaultdict[int, set[str]] = defaultdict(set)
    for session in sessions:
        week = week_index(session.start_time, first, zone)
        session_weeks[week] += 1
        learners[week].add(session.learner_id)
    event_weeks: Counter[int] = Counter(
        week_index(e.timestamp, first, zone) for e in dataset.events
    )
    last = max(session_weeks)
    return tuple(
        {
            "week": w,
            "sessions": session_weeks.get(w, 0),
            "events": event_weeks.get(w, 0),
            "learners": len(learners.get(w, ())),
        }
        for w in range(1, last + 1)
    )


def _between_sessions(sessions: Sequence[Session]) -> list[float]:
    by_learner: defaultdict[str, list[Session]] = defaultdict(list)
    for session in sessions:
        by_learner[session.learner_id].append(session)
    return [
        max((later.start_time - earlier.end_time).total_seconds(), 0.0)
        for owned in by_learner.values()
        for earlier, later in pairwise(owned)
    ]


def _hours(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"
