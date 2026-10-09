"""Manipulation check: did each control have its intended effect (design §10)?

The check compares a controlled sample with an **uncontrolled baseline** drawn
from the same model, seed, size, and calendar, so measured differences come
from the controls rather than from sampling noise in the original data. It
should be run on data *before* anomaly injection; anomalies are checked from
the injection report.

Deterministic rules (removed tokens, zero-weight hours, the period) must hold
exactly. Distributional checks (hour and weekday mix, deadline surges) use
sample-size-aware tolerances: with small samples they are less sensitive and
fail by chance about 1% of the time; ``details`` reports the tolerance.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import pairwise
from statistics import median
from typing import Any, Literal

from eduloggen.analysis import session_sequences
from eduloggen.generators import SessionCalendar
from eduloggen.models import Dataset, LogRecord
from eduloggen.scenarios.controls import Controls

__all__ = ["CheckItem", "ManipulationCheck", "manipulation_check"]

Status = Literal["ok", "failed", "not_measurable"]


@dataclass(frozen=True, slots=True)
class CheckItem:
    """One control and its measured effect.

    Attributes:
        control: Control family (e.g. ``event_weights``, ``calendar.hours``).
        target: What it applies to (a token, ``*``, a deadline, …).
        intended: The requested effect, in words.
        measured: What was observed, in words.
        status: ``ok``, ``failed``, or ``not_measurable``.
        details: JSON-compatible numbers behind the verdict.
    """

    control: str
    target: str
    intended: str
    measured: str
    status: Status
    details: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "control": self.control,
            "target": self.target,
            "intended": self.intended,
            "measured": self.measured,
            "status": self.status,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class ManipulationCheck:
    """All check items, with an overall verdict.

    Attributes:
        items: Checks in a stable order.
        tolerance: Relative tolerance used for ratio checks.
    """

    items: tuple[CheckItem, ...]
    tolerance: float

    @property
    def passed(self) -> bool:
        """``True`` if no measurable check failed."""
        return all(item.status != "failed" for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "passed": self.passed,
            "tolerance": self.tolerance,
            "items": [item.to_dict() for item in self.items],
        }

    def to_markdown(self) -> str:
        """Table suitable for a methods section."""
        mark = {"ok": "✓", "failed": "✗", "not_measurable": "-"}
        lines = [
            f"# Manipulation check: {'PASSED' if self.passed else 'FAILED'}",
            "",
            "Each control is compared with an uncontrolled baseline drawn from the "
            f"same model, seed, and size (relative tolerance {self.tolerance:.0%}).",
            "",
            "| Control | Target | Intended | Measured | OK |",
            "| ------- | ------ | -------- | -------- | -- |",
        ]
        lines += [
            f"| {i.control} | {i.target} | {i.intended} | {i.measured} "
            f"| {mark[i.status]} |"
            for i in self.items
        ]
        return "\n".join(lines) + "\n"


def manipulation_check(
    baseline: Dataset,
    controlled: Dataset,
    *,
    controls: Controls | None = None,
    calendar: SessionCalendar | None = None,
    anomaly_report: Mapping[str, Any] | None = None,
    tolerance: float = 0.2,
) -> ManipulationCheck:
    """Measure whether controls, calendar, and anomalies took effect.

    Args:
        baseline: Sample from the model *without* controls (same seed, size,
            and calendar).
        controlled: Sample from the controlled model, before anomalies.
        controls: The controls that were applied.
        calendar: The calendar used for both samples.
        anomaly_report: ``InjectionResult.report`` if anomalies were injected.
        tolerance: Relative tolerance for ratio-type checks.

    Returns:
        The check.
    """
    items: list[CheckItem] = []
    if controls is not None:
        items += _event_weights(baseline, controlled, controls)
        items += _dwell(baseline, controlled, controls, tolerance)
        items += _lengths(baseline, controlled, controls, tolerance)
        items += _per_learner(controlled, controls, tolerance)
    if calendar is not None:
        items += _calendar(controlled, calendar, tolerance)
    if anomaly_report is not None:
        for name, info in sorted(anomaly_report.get("anomalies", {}).items()):
            ok = info["injected"] == info["requested"]
            items.append(
                CheckItem(
                    "anomalies",
                    name,
                    f"{info['requested']} sessions ({info['rate']:.1%})",
                    f"{info['injected']} sessions",
                    "ok" if ok else "failed",
                    {"requested": info["requested"], "injected": info["injected"]},
                )
            )
    return ManipulationCheck(tuple(items), tolerance)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def _shares(dataset: Dataset) -> dict[str, float]:
    tokens = Counter(t for s in session_sequences(dataset) for t in s)
    total = sum(tokens.values()) or 1
    return {t: c / total for t, c in tokens.items()}


def _event_weights(
    baseline: Dataset, controlled: Dataset, controls: Controls
) -> list[CheckItem]:
    before, after = _shares(baseline), _shares(controlled)
    items = []
    for token, weight in sorted(controls.event_weights.items()):
        b, c = before.get(token, 0.0), after.get(token, 0.0)
        measured = f"share {b:.1%} → {c:.1%}"
        if weight == 0:
            status: Status = "ok" if c == 0 else "failed"
            intended = "removed"
        elif weight == 1:
            continue
        elif b == 0:
            status, intended = "not_measurable", f"weight x{weight:g}"
        else:
            intended = (
                f"weight x{weight:g} ({'more' if weight > 1 else 'less'} frequent)"
            )
            status = "ok" if (c > b if weight > 1 else c < b) else "failed"
        items.append(
            CheckItem(
                "event_weights",
                token,
                intended,
                measured,
                status,
                {"baseline_share": b, "controlled_share": c, "weight": weight},
            )
        )
    return items


def _gaps_after(dataset: Dataset, token: str) -> list[float]:
    members: defaultdict[str, list[LogRecord]] = defaultdict(list)
    for event in dataset.sorted_events():
        members[event.session_id or ""].append(event)
    tokens = dict(zip(sorted(members), session_sequences(dataset), strict=False))
    gaps = []
    for session_id, events in members.items():
        sequence = tokens.get(session_id, ())
        for (earlier, later), current in zip(pairwise(events), sequence, strict=False):
            if token == "*" or current == token:
                gaps.append((later.timestamp - earlier.timestamp).total_seconds())
    return gaps


def _dwell(
    baseline: Dataset, controlled: Dataset, controls: Controls, tolerance: float
) -> list[CheckItem]:
    items = []
    for token, factor in sorted(controls.dwell_scale.items()):
        before, after = _gaps_after(baseline, token), _gaps_after(controlled, token)
        intended = f"time spent x{factor:g}"
        if not before or not after or median(before) <= 0:
            items.append(
                CheckItem(
                    "dwell_scale",
                    token,
                    intended,
                    "no gaps to measure",
                    "not_measurable",
                )
            )
            continue
        ratio = median(after) / median(before)
        status: Status = "ok" if abs(ratio / factor - 1) <= tolerance else "failed"
        items.append(
            CheckItem(
                "dwell_scale",
                token,
                intended,
                f"median {median(before):.0f}s → {median(after):.0f}s (x{ratio:.2f})",
                status,
                {
                    "baseline_median_s": median(before),
                    "controlled_median_s": median(after),
                    "ratio": ratio,
                },
            )
        )
    return items


def _lengths(
    baseline: Dataset, controlled: Dataset, controls: Controls, tolerance: float
) -> list[CheckItem]:
    if not controls.session_length:
        return []
    mode, value = next(iter(controls.session_length.items()))
    before = [len(s) for s in session_sequences(baseline)]
    after = [len(s) for s in session_sequences(controlled)]
    if not before or not after:
        return [
            CheckItem(
                "session_length", mode, str(value), "no sessions", "not_measurable"
            )
        ]
    mean_b, mean_a = sum(before) / len(before), sum(after) / len(after)
    if mode == "fixed":
        status: Status = "ok" if set(after) == {int(value)} else "failed"
        return [
            CheckItem(
                "session_length",
                "fixed",
                f"{int(value)} events",
                f"lengths {sorted(set(after))[:5]}",
                status,
            )
        ]
    ratio = mean_a / mean_b
    status = "ok" if abs(ratio / value - 1) <= tolerance else "failed"
    return [
        CheckItem(
            "session_length",
            "scale",
            f"mean length x{value:g}",
            f"mean {mean_b:.1f} → {mean_a:.1f} (x{ratio:.2f})",
            status,
            {"baseline_mean": mean_b, "controlled_mean": mean_a, "ratio": ratio},
        )
    ]


def _per_learner(
    controlled: Dataset, controls: Controls, tolerance: float
) -> list[CheckItem]:
    if not controls.sessions_per_learner:
        return []
    mode, value = next(iter(controls.sessions_per_learner.items()))
    counts = Counter(s.learner_id for s in controlled.sessions or ())
    if not counts:
        return [
            CheckItem(
                "sessions_per_learner",
                mode,
                str(value),
                "no sessions",
                "not_measurable",
            )
        ]
    values = sorted(counts.values())
    mean = sum(values) / len(values)
    if mode == "fixed":
        # the last learner may be cut short by the requested session total
        status: Status = (
            "ok"
            if values[1:] == [int(value)] * (len(values) - 1)
            or values == [int(value)] * len(values)
            else "failed"
        )
        return [
            CheckItem(
                "sessions_per_learner",
                "fixed",
                f"{int(value)} per learner",
                f"mean {mean:.2f}",
                status,
            )
        ]
    status = "ok" if abs(mean / value - 1) <= tolerance else "failed"
    return [
        CheckItem(
            "sessions_per_learner",
            "mean",
            f"mean {value:g}",
            f"mean {mean:.2f}",
            status,
            {"measured_mean": mean},
        )
    ]


def _calendar(
    dataset: Dataset, calendar: SessionCalendar, tolerance: float
) -> list[CheckItem]:
    starts = [calendar.local_time(s.start_time) for s in dataset.sessions or ()]
    if not starts:
        return [CheckItem("calendar", "period", "", "no sessions", "not_measurable")]
    items = []
    outside = sum(
        1
        for s in starts
        if not calendar.start <= s.date() <= calendar.end + timedelta(days=1)
    )
    items.append(
        CheckItem(
            "calendar",
            "period",
            f"{calendar.start} to {calendar.end}",
            f"{outside} of {len(starts)} starts outside",
            "ok" if outside == 0 else "failed",
        )
    )
    closed = [h for h in range(24) if calendar.hours[h] == 0]
    if closed:
        in_closed = sum(1 for s in starts if s.hour in closed)
        items.append(
            CheckItem(
                "calendar.hours",
                "zero-weight hours",
                "no sessions start",
                f"{in_closed} of {len(starts)} start",
                "ok" if in_closed == 0 else "failed",
            )
        )
    items.append(
        _distribution_item("calendar.hours", [s.hour for s in starts], calendar.hours)
    )
    period_days = [calendar.start + timedelta(days=i) for i in range(calendar.days)]
    # expected weekday mix includes deadline surges, not just weekday weights
    weekday_target = [
        sum(calendar.day_weight(d) for d in period_days if d.weekday() == w)
        for w in range(7)
    ]
    items.append(
        _distribution_item(
            "calendar.weekdays", [s.weekday() for s in starts], weekday_target
        )
    )
    for deadline in calendar.deadlines:
        items.append(
            _deadline_item(
                starts,
                calendar,
                period_days,
                deadline.date,
                deadline.days_before,
                deadline.surge,
                tolerance,
            )
        )
    return items


def _distribution_item(
    control: str, observed: list[int], weights: Sequence[float]
) -> CheckItem:
    total_w = sum(weights)
    counts = Counter(observed)
    n = len(observed)
    tvd = 0.5 * sum(
        abs(counts.get(i, 0) / n - weights[i] / total_w) for i in range(len(weights))
    )
    # sampling noise of a total variation distance shrinks like sqrt(k / n)
    limit = max(0.1, 1.5 * math.sqrt(len(weights) / n) / 2)
    status: Status = "ok" if tvd <= limit else "failed"
    return CheckItem(
        control,
        "distribution",
        "matches the weights",
        f"TVD {tvd:.3f} (limit {limit:.3f})",
        status,
        {"tvd": tvd, "limit": limit, "n": n},
    )


def _deadline_item(
    starts: Sequence[Any],
    calendar: SessionCalendar,
    period_days: list[date],
    deadline: date,
    days_before: int,
    surge: float,
    tolerance: float,
) -> CheckItem:
    window = {deadline - timedelta(days=i) for i in range(days_before + 1)} & set(
        period_days
    )
    rest = [d for d in period_days if d not in window and calendar.day_weight(d) > 0]
    per_day = Counter(s.date() for s in starts)
    if not window or not rest:
        return CheckItem(
            "calendar.deadlines",
            str(deadline),
            f"x{surge:g}",
            "window outside the period",
            "not_measurable",
        )
    expected = (sum(calendar.day_weight(d) for d in window) / len(window)) / (
        sum(calendar.day_weight(d) for d in rest) / len(rest)
    )
    rest_rate = sum(per_day.get(d, 0) for d in rest) / len(rest)
    window_rate = sum(per_day.get(d, 0) for d in window) / len(window)
    if rest_rate == 0:
        return CheckItem(
            "calendar.deadlines",
            str(deadline),
            f"x{expected:.2f} sessions/day",
            "no sessions outside the window",
            "not_measurable",
        )
    measured = window_rate / rest_rate
    # ~3 standard errors of a ratio of counts (sessions cluster by learner,
    # so a tighter bound gives false alarms; sensitivity grows with sample size)
    window_n = sum(per_day.get(d, 0) for d in window)
    rest_n = sum(per_day.get(d, 0) for d in rest)
    if window_n < 20:
        return CheckItem(
            "calendar.deadlines",
            str(deadline),
            f"x{expected:.2f} sessions/day before deadline",
            f"only {window_n} sessions in the window (need 20)",
            "not_measurable",
        )
    limit = max(tolerance, 3.0 * math.sqrt(1 / max(window_n, 1) + 1 / max(rest_n, 1)))
    status: Status = "ok" if abs(measured / expected - 1) <= limit else "failed"
    return CheckItem(
        "calendar.deadlines",
        str(deadline),
        f"x{expected:.2f} sessions/day before deadline",
        f"x{measured:.2f}",
        status,
        {"expected_ratio": expected, "measured_ratio": measured, "tolerance": limit},
    )
