"""Stratified analysis: the same statistics, broken down by group (FR-A.7).

Sessions are grouped by a key:

- ``course`` — the session's course (``(none)`` if missing);
- ``week`` — week of the session start, counted from the dataset's first
  session (``week 01``, ``week 02``, …);
- ``weekday`` / ``hour`` — of the session start, in a chosen timezone;
- ``learner_group`` — a learner-to-group mapping you supply (e.g. cohorts);
- ``metadata:<key>`` — the most common value of an event metadata key in the
  session (e.g. ``metadata:device``).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from eduloggen.analysis.sequences import ngram_counts, session_sequences
from eduloggen.analysis.stats import Summary, describe
from eduloggen.analysis.temporal import WEEKDAYS, resolve_timezone, week_index
from eduloggen.analysis.timing import interevent_times, session_durations
from eduloggen.core import AnalysisError
from eduloggen.models import Dataset, LogRecord

__all__ = [
    "GROUP_KEYS",
    "GroupSummary",
    "StratifiedAnalysis",
    "analyze_by",
    "session_groups",
    "split_by",
]

GROUP_KEYS = ("course", "week", "weekday", "hour", "learner_group", "metadata:<key>")
NONE = "(none)"
UNASSIGNED = "(unassigned)"


def session_groups(
    dataset: Dataset,
    by: str,
    *,
    timezone: str = "UTC",
    groups: Mapping[str, str] | None = None,
) -> dict[str, list[str]]:
    """Group label to the ids of its sessions, labels sorted.

    Raises:
        AnalysisError: If the key is unknown, ``learner_group`` lacks
            ``groups``, or the dataset is not sessionized.
    """
    session_sequences(dataset)
    zone = resolve_timezone(timezone)
    sessions = sorted(
        dataset.sessions or (), key=lambda s: (s.start_time, s.session_id)
    )
    if by == "learner_group" and groups is None:
        raise AnalysisError(
            "grouping by learner_group needs a learner-to-group mapping",
            code="analysis_invalid_parameter",
        )
    members: defaultdict[str, list[LogRecord]] = defaultdict(list)
    if by.startswith("metadata:"):
        for event in dataset.events:
            if event.session_id is not None:
                members[event.session_id].append(event)
    first = sessions[0].start_time.astimezone(zone).date() if sessions else None
    result: defaultdict[str, list[str]] = defaultdict(list)
    for session in sessions:
        local = session.start_time.astimezone(zone)
        if by == "course":
            label = session.course_id or NONE
        elif by == "week":
            label = f"week {week_index(session.start_time, first, zone):02d}"  # type: ignore[arg-type]
        elif by == "weekday":
            label = f"{local.weekday() + 1}-{WEEKDAYS[local.weekday()]}"
        elif by == "hour":
            label = f"{local.hour:02d}"
        elif by == "learner_group":
            label = (groups or {}).get(session.learner_id, UNASSIGNED)
        elif by.startswith("metadata:") and len(by) > len("metadata:"):
            key = by.removeprefix("metadata:")
            values = Counter(
                str(e.metadata[key])
                for e in members[session.session_id]
                if key in e.metadata
            )
            label = values.most_common(1)[0][0] if values else NONE
        else:
            raise AnalysisError(
                f"unknown grouping key {by!r}",
                code="analysis_invalid_parameter",
                context={"by": by, "allowed": list(GROUP_KEYS)},
            )
        result[label].append(session.session_id)
    return dict(sorted(result.items()))


def split_by(
    dataset: Dataset,
    by: str,
    *,
    timezone: str = "UTC",
    groups: Mapping[str, str] | None = None,
) -> dict[str, Dataset]:
    """One sub-dataset (sessions and their events) per group label."""
    labels = session_groups(dataset, by, timezone=timezone, groups=groups)
    owner = {sid: label for label, sids in labels.items() for sid in sids}
    events: defaultdict[str, list[LogRecord]] = defaultdict(list)
    for event in dataset.events:
        if event.session_id in owner:
            events[owner[event.session_id]].append(event)
    sessions: defaultdict[str, list[Any]] = defaultdict(list)
    for session in dataset.sessions or ():
        sessions[owner[session.session_id]].append(session)
    return {
        label: Dataset(
            dataset_id=f"{dataset.dataset_id}[{by}={label}]",
            events=tuple(events[label]),
            sessions=tuple(sessions[label]),
            metadata=dataset.metadata,
        )
        for label in labels
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class GroupSummary:
    """Statistics of one group.

    Attributes:
        group: Group label (``all`` for the overall row).
        n_sessions, n_learners, n_events: Sizes.
        session_length: Events per session.
        session_duration_s: Session durations.
        interevent_s: Gaps between events within sessions.
        token_shares: Share of each session token.
        mix_tvd: Total variation distance between this group's token mix and
            the overall mix (0 = same).
        top_bigrams: Most frequent within-session token pairs and shares.
    """

    group: str
    n_sessions: int
    n_learners: int
    n_events: int
    session_length: Summary
    session_duration_s: Summary
    interevent_s: Summary
    token_shares: Mapping[str, float] = field(hash=False)
    mix_tvd: float
    top_bigrams: Mapping[str, float] = field(hash=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "group": self.group,
            "n_sessions": self.n_sessions,
            "n_learners": self.n_learners,
            "n_events": self.n_events,
            "session_length": self.session_length.to_dict(),
            "session_duration_s": self.session_duration_s.to_dict(),
            "interevent_s": self.interevent_s.to_dict(),
            "token_shares": dict(self.token_shares),
            "mix_tvd": self.mix_tvd,
            "top_bigrams": dict(self.top_bigrams),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class StratifiedAnalysis:
    """Per-group statistics plus the overall row.

    Attributes:
        by: Grouping key.
        timezone: Zone used for time-based keys.
        overall: Statistics of the whole dataset.
        groups: One summary per group, sorted by label.
        omitted: Groups below ``min_sessions`` (label to session count).
    """

    by: str
    timezone: str
    overall: GroupSummary
    groups: tuple[GroupSummary, ...]
    omitted: Mapping[str, int] = field(hash=False)

    def group(self, label: str) -> GroupSummary:
        """Look up a group.

        Raises:
            KeyError: If there is no such group.
        """
        for summary in self.groups:
            if summary.group == label:
                return summary
        raise KeyError(label)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "by": self.by,
            "timezone": self.timezone,
            "overall": self.overall.to_dict(),
            "groups": [g.to_dict() for g in self.groups],
            "omitted": dict(self.omitted),
        }

    def to_markdown(self) -> str:
        """Comparison table, one row per group."""
        header = [
            "Group",
            "Sessions",
            "Learners",
            "Events",
            "Mean length",
            "Median duration (min)",
            "Median gap (s)",
            "Mix vs overall (TVD)",
            "Top token",
        ]
        lines = [
            f"## Analysis by {self.by}",
            "",
            _row(header),
            _row(["---"] * len(header)),
        ]
        for g in (self.overall, *self.groups):
            top = max(
                g.token_shares.items(),
                key=lambda kv: (kv[1], kv[0]),
                default=("-", 0.0),
            )
            lines.append(
                _row(
                    [
                        g.group,
                        str(g.n_sessions),
                        str(g.n_learners),
                        str(g.n_events),
                        _fmt(g.session_length.mean),
                        _fmt(_minutes(g.session_duration_s.median)),
                        _fmt(g.interevent_s.median),
                        f"{g.mix_tvd:.3f}",
                        f"{top[0]} ({top[1]:.0%})",
                    ]
                )
            )
        if self.omitted:
            small = ", ".join(f"{k} ({v})" for k, v in self.omitted.items())
            lines += ["", f"Omitted (too few sessions): {small}"]
        return "\n".join(lines) + "\n"


def analyze_by(
    dataset: Dataset,
    by: str,
    *,
    timezone: str = "UTC",
    groups: Mapping[str, str] | None = None,
    min_sessions: int = 1,
    top: int = 5,
) -> StratifiedAnalysis:
    """Summarize each group and compare its token mix with the overall mix.

    Args:
        dataset: Sessionized dataset.
        by: Grouping key (see module docs).
        timezone: Zone for ``week``, ``weekday``, and ``hour``.
        groups: Learner id to group label, for ``learner_group``.
        min_sessions: Groups with fewer sessions are listed as omitted.
        top: Number of top bigrams per group.

    Raises:
        AnalysisError: If the key or parameters are invalid.
    """
    if min_sessions < 1:
        raise AnalysisError(
            "min_sessions must be at least 1", code="analysis_invalid_parameter"
        )
    parts = split_by(dataset, by, timezone=timezone, groups=groups)
    overall = _summary("all", dataset, None, top)
    kept, omitted = [], {}
    for label, part in parts.items():
        if part.n_sessions < min_sessions:
            omitted[label] = part.n_sessions
        else:
            kept.append(_summary(label, part, overall.token_shares, top))
    return StratifiedAnalysis(
        by=by, timezone=timezone, overall=overall, groups=tuple(kept), omitted=omitted
    )


def _summary(
    label: str, dataset: Dataset, reference: Mapping[str, float] | None, top: int
) -> GroupSummary:
    sequences = session_sequences(dataset)
    tokens = Counter(t for s in sequences for t in s)
    total = sum(tokens.values()) or 1
    shares = {
        t: c / total for t, c in sorted(tokens.items(), key=lambda kv: (-kv[1], kv[0]))
    }
    bigrams = ngram_counts(sequences, 2)
    pair_total = sum(bigrams.values()) or 1
    ranked = sorted(bigrams.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    mix = 0.0
    if reference is not None:
        keys = set(shares) | set(reference)
        mix = 0.5 * sum(abs(shares.get(k, 0.0) - reference.get(k, 0.0)) for k in keys)
    return GroupSummary(
        group=label,
        n_sessions=dataset.n_sessions,
        n_learners=len(dataset.learner_ids),
        n_events=sum(len(s) for s in sequences),
        session_length=describe(len(s) for s in sequences),
        session_duration_s=describe(session_durations(dataset)),
        interevent_s=describe(interevent_times(dataset)),
        token_shares=shares,
        mix_tvd=mix,
        top_bigrams={f"{a} -> {b}": c / pair_total for (a, b), c in ranked},
    )


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _minutes(seconds: float | None) -> float | None:
    return None if seconds is None else seconds / 60


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3g}"
