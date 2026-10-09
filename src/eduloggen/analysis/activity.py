"""Per-activity detail: one profile per session token.

For every token (event type or activity, depending on tokenization) report
how often it occurs, how long learners stay on it, what comes before and
after it, where sessions start and end, and outcomes such as success rates.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from eduloggen.analysis.stats import Summary, describe
from eduloggen.analysis.timing import _session_events
from eduloggen.models import Dataset

__all__ = ["ActivityProfile", "activity_profiles", "activity_table_markdown"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ActivityProfile:
    """Statistics of one token.

    Attributes:
        token: The session token.
        count: Occurrences.
        share: Share of all tokens.
        session_share: Share of sessions containing it at least once.
        learner_share: Share of learners who produced it at least once.
        start_share: Share of sessions that start with it.
        end_share: Share of sessions that end with it.
        repeat_share: Share of its occurrences followed by itself.
        dwell_s: Time until the next event in the same session.
        predecessors: Most common previous tokens with their shares.
        successors: Most common next tokens with their shares (``(end)`` for
            session ends).
        success_rate: Mean of ``success`` where recorded, else ``None``.
        mean_score: Mean ``score`` where recorded, else ``None``.
        companions: Most common values of the other event field.
    """

    token: str
    count: int
    share: float
    session_share: float
    learner_share: float
    start_share: float
    end_share: float
    repeat_share: float
    dwell_s: Summary
    predecessors: Mapping[str, float] = field(hash=False)
    successors: Mapping[str, float] = field(hash=False)
    success_rate: float | None
    mean_score: float | None
    companions: Mapping[str, float] = field(hash=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "token": self.token,
            "count": self.count,
            "share": self.share,
            "session_share": self.session_share,
            "learner_share": self.learner_share,
            "start_share": self.start_share,
            "end_share": self.end_share,
            "repeat_share": self.repeat_share,
            "dwell_s": self.dwell_s.to_dict(),
            "predecessors": dict(self.predecessors),
            "successors": dict(self.successors),
            "success_rate": self.success_rate,
            "mean_score": self.mean_score,
            "companions": dict(self.companions),
        }


def activity_profiles(dataset: Dataset, *, top: int = 5) -> tuple[ActivityProfile, ...]:
    """Profile every token of a sessionized dataset, most frequent first.

    Args:
        dataset: Sessionized dataset.
        top: How many predecessors, successors, and companions to keep.

    Raises:
        AnalysisError: If the dataset is not sessionized.
    """
    sessions = list(_session_events(dataset))
    if not sessions:
        return ()
    counts: Counter[str] = Counter()
    in_sessions: Counter[str] = Counter()
    learners: defaultdict[str, set[str]] = defaultdict(set)
    starts: Counter[str] = Counter()
    ends: Counter[str] = Counter()
    before: defaultdict[str, Counter[str]] = defaultdict(Counter)
    after: defaultdict[str, Counter[str]] = defaultdict(Counter)
    dwell: defaultdict[str, list[float]] = defaultdict(list)
    success: defaultdict[str, list[bool]] = defaultdict(list)
    scores: defaultdict[str, list[float]] = defaultdict(list)
    companions: defaultdict[str, Counter[str]] = defaultdict(Counter)
    all_learners: set[str] = set()

    for tokens, events in sessions:
        starts[tokens[0]] += 1
        ends[tokens[-1]] += 1
        in_sessions.update(set(tokens))
        for token, event in zip(tokens, events, strict=True):
            counts[token] += 1
            learners[token].add(event.learner_id)
            all_learners.add(event.learner_id)
            other = event.activity_id if token == event.event_type else event.event_type
            companions[token][other] += 1
            if event.success is not None:
                success[token].append(event.success)
            if event.score is not None:
                scores[token].append(event.score)
        for (a, b), (ea, eb) in zip(pairwise(tokens), pairwise(events), strict=True):
            after[a][b] += 1
            before[b][a] += 1
            dwell[a].append((eb.timestamp - ea.timestamp).total_seconds())
        after[tokens[-1]]["(end)"] += 1

    total = sum(counts.values())
    n_sessions = len(sessions)
    profiles = []
    for token, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        repeats = after[token].get(token, 0)
        profiles.append(
            ActivityProfile(
                token=token,
                count=count,
                share=count / total,
                session_share=in_sessions[token] / n_sessions,
                learner_share=len(learners[token]) / len(all_learners),
                start_share=starts[token] / n_sessions,
                end_share=ends[token] / n_sessions,
                repeat_share=repeats / count,
                dwell_s=describe(dwell[token]),
                predecessors=_top(before[token], top),
                successors=_top(after[token], top),
                success_rate=(
                    sum(success[token]) / len(success[token])
                    if success[token]
                    else None
                ),
                mean_score=(
                    sum(scores[token]) / len(scores[token]) if scores[token] else None
                ),
                companions=_top(companions[token], top),
            )
        )
    return tuple(profiles)


def activity_table_markdown(profiles: tuple[ActivityProfile, ...]) -> str:
    """One row per token: frequency, reach, dwell, entry/exit, outcomes, flow."""
    header = [
        "Token",
        "Share",
        "Sessions",
        "Starts",
        "Ends",
        "Repeats",
        "Median dwell (s)",
        "Success",
        "Most often next",
    ]
    lines = [_row(header), _row(["---"] * len(header))]
    for p in profiles:
        nxt = ", ".join(f"{t} {s:.0%}" for t, s in list(p.successors.items())[:3])
        lines.append(
            _row(
                [
                    p.token,
                    f"{p.share:.1%}",
                    f"{p.session_share:.0%}",
                    f"{p.start_share:.0%}",
                    f"{p.end_share:.0%}",
                    f"{p.repeat_share:.0%}",
                    "-" if p.dwell_s.median is None else f"{p.dwell_s.median:.0f}",
                    "-" if p.success_rate is None else f"{p.success_rate:.0%}",
                    nxt,
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _top(counter: Counter[str], n: int) -> dict[str, float]:
    total = sum(counter.values())
    if not total:
        return {}
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return {token: count / total for token, count in ranked}
