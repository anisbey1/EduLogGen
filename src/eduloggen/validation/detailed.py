"""Fine-grained comparison: *where* synthetic data differs from real data.

Headline metrics say *how much* two datasets differ. This module breaks the
difference down — per token, per transition, per session length, per hour
and weekday, and optionally per group — and ranks the largest gaps, so a
researcher can see which behaviours a generator gets wrong.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from eduloggen.analysis import (
    analyze_by,
    ngram_counts,
    session_sequences,
    sojourn_times,
    split_by,
    temporal_profile,
)
from eduloggen.models import Dataset
from eduloggen.validation.base import ValidationContext
from eduloggen.validation.registry import get_metric

__all__ = ["DetailedComparison", "compare_detailed"]

GROUP_METRICS = (
    "event_type_tvd",
    "bigram_tvd",
    "session_length_ks",
    "interevent_time_ks",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class DetailedComparison:
    """Ranked differences between a real and a synthetic dataset.

    Attributes:
        tokens: Per token: real and synthetic share, difference, and median
            dwell time in each; sorted by absolute share difference.
        transitions: Largest per-transition share differences.
        novel_transitions: Synthetic transitions never seen in real data.
        missing_transitions: Real transitions the synthetic data never produces.
        session_lengths: Share of sessions per length (``30+`` grouped).
        hours, weekdays: Share of session starts per hour / weekday.
        groups: Per group (when ``by`` is given): sizes and headline metrics.
        by: Grouping key, if any.
        timezone: Zone for hours, weekdays, and time-based groups.
    """

    tokens: tuple[dict[str, Any], ...] = field(hash=False)
    transitions: tuple[dict[str, Any], ...] = field(hash=False)
    novel_transitions: Mapping[str, Any] = field(hash=False)
    missing_transitions: Mapping[str, Any] = field(hash=False)
    session_lengths: tuple[dict[str, Any], ...] = field(hash=False)
    hours: tuple[dict[str, Any], ...] = field(hash=False)
    weekdays: tuple[dict[str, Any], ...] = field(hash=False)
    groups: tuple[dict[str, Any], ...] = field(hash=False)
    by: str | None
    timezone: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "by": self.by,
            "timezone": self.timezone,
            "tokens": list(self.tokens),
            "transitions": list(self.transitions),
            "novel_transitions": dict(self.novel_transitions),
            "missing_transitions": dict(self.missing_transitions),
            "session_lengths": list(self.session_lengths),
            "hours": list(self.hours),
            "weekdays": list(self.weekdays),
            "groups": list(self.groups),
        }

    def to_markdown(self, *, top: int = 10) -> str:
        """Readable report of the largest gaps."""
        pct = _pct
        lines = ["# Where synthetic differs from real", "", "## Activities", ""]
        lines += _table(
            [
                "Token",
                "Real",
                "Synthetic",
                "Difference",
                "Real dwell (s)",
                "Synthetic dwell (s)",
            ],
            [_token_cells(t) for t in self.tokens[:top]],
        )
        lines += ["", "## Transitions with the largest gaps", ""]
        lines += _table(
            ["Transition", "Real", "Synthetic", "Difference"],
            [
                [
                    t["transition"],
                    pct(t["real_share"]),
                    pct(t["synthetic_share"]),
                    f"{t['difference']:+.1%}",
                ]
                for t in self.transitions[:top]
            ],
        )
        novel, missing = self.novel_transitions, self.missing_transitions
        lines += [
            "",
            f"Synthetic transitions never seen in real data: {novel['count']} "
            f"({pct(novel['synthetic_share'])} of synthetic transitions)"
            + (f", e.g. {', '.join(novel['examples'])}" if novel["examples"] else ""),
            "",
            f"Real transitions never produced: {missing['count']} "
            f"({pct(missing['real_share'])} of real transitions)"
            + (
                f", e.g. {', '.join(missing['examples'])}"
                if missing["examples"]
                else ""
            ),
            "",
            "## Session lengths",
            "",
        ]
        lines += _table(
            ["Events", "Real", "Synthetic", "Difference"],
            [
                [
                    r["length"],
                    pct(r["real_share"]),
                    pct(r["synthetic_share"]),
                    f"{r['difference']:+.1%}",
                ]
                for r in self.session_lengths
            ],
        )
        lines += ["", f"## Session start hours ({self.timezone}), largest gaps", ""]
        hours = sorted(self.hours, key=lambda r: -abs(r["difference"]))[:6]
        lines += _table(
            ["Hour", "Real", "Synthetic", "Difference"],
            [
                [
                    f"{r['hour']:02d}",
                    pct(r["real_share"]),
                    pct(r["synthetic_share"]),
                    f"{r['difference']:+.1%}",
                ]
                for r in hours
            ],
        )
        lines += ["", "## Weekdays", ""]
        lines += _table(
            ["Weekday", "Real", "Synthetic", "Difference"],
            [
                [
                    r["weekday"],
                    pct(r["real_share"]),
                    pct(r["synthetic_share"]),
                    f"{r['difference']:+.1%}",
                ]
                for r in self.weekdays
            ],
        )
        if self.by is not None:
            lines += ["", f"## By {self.by}", ""]
            lines += _table(
                ["Group", "Real sessions", "Synthetic sessions", *GROUP_METRICS],
                [_group_cells(g) for g in self.groups],
            )
        return "\n".join(lines) + "\n"


def compare_detailed(
    real: Dataset,
    synthetic: Dataset,
    *,
    by: str | None = None,
    timezone: str = "UTC",
    groups: Mapping[str, str] | None = None,
    top: int = 10,
    seed: int = 0,
) -> DetailedComparison:
    """Break the real-versus-synthetic difference down by component.

    Args:
        real: Sessionized real dataset.
        synthetic: Sessionized synthetic dataset.
        by: Optional grouping key (see :mod:`eduloggen.analysis.strata`);
            headline metrics are then computed per group present in both.
        timezone: Zone for hours, weekdays, and time-based groups.
        groups: Learner-to-group mapping for ``learner_group`` (applied to
            both datasets; synthetic learners usually need their own mapping).
        top: How many transitions to keep (tokens are all kept).
        seed: Seed for metrics that subsample.

    Raises:
        AnalysisError: If a dataset is not sessionized or a key is invalid.
    """
    real_seqs, synth_seqs = session_sequences(real), session_sequences(synthetic)
    real_tokens = _shares(Counter(t for s in real_seqs for t in s))
    synth_tokens = _shares(Counter(t for s in synth_seqs for t in s))
    real_dwell, synth_dwell = sojourn_times(real), sojourn_times(synthetic)
    tokens = sorted(
        (
            {
                "token": t,
                "real_share": real_tokens.get(t, 0.0),
                "synthetic_share": synth_tokens.get(t, 0.0),
                "difference": synth_tokens.get(t, 0.0) - real_tokens.get(t, 0.0),
                "real_dwell_median_s": (
                    median(real_dwell[t]) if real_dwell.get(t) else None
                ),
                "synthetic_dwell_median_s": (
                    median(synth_dwell[t]) if synth_dwell.get(t) else None
                ),
            }
            for t in set(real_tokens) | set(synth_tokens)
        ),
        key=lambda row: (-abs(row["difference"]), row["token"]),
    )

    real_pairs = _shares(ngram_counts(real_seqs, 2))
    synth_pairs = _shares(ngram_counts(synth_seqs, 2))
    transition_rows: list[dict[str, Any]] = [
        {
            "transition": f"{a} -> {b}",
            "real_share": real_pairs.get((a, b), 0.0),
            "synthetic_share": synth_pairs.get((a, b), 0.0),
            "difference": synth_pairs.get((a, b), 0.0) - real_pairs.get((a, b), 0.0),
        }
        for a, b in set(real_pairs) | set(synth_pairs)
    ]
    transitions = sorted(
        transition_rows, key=lambda row: (-abs(row["difference"]), row["transition"])
    )[:top]
    novel = sorted(set(synth_pairs) - set(real_pairs), key=lambda p: -synth_pairs[p])
    missing = sorted(set(real_pairs) - set(synth_pairs), key=lambda p: -real_pairs[p])

    real_time = temporal_profile(real, timezone=timezone)
    synth_time = temporal_profile(synthetic, timezone=timezone)
    weekdays = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

    group_rows: list[dict[str, Any]] = []
    if by is not None:
        analyze_by(real, by, timezone=timezone, groups=groups)  # validates the key
        real_parts = split_by(real, by, timezone=timezone, groups=groups)
        synth_parts = split_by(synthetic, by, timezone=timezone, groups=groups)
        context = ValidationContext(seed=seed)
        for label in sorted(set(real_parts) | set(synth_parts)):
            r, s = real_parts.get(label), synth_parts.get(label)
            metrics: dict[str, float | None] = {}
            if r is not None and s is not None:
                for name in GROUP_METRICS:
                    metrics[name] = get_metric(name).compute(r, s, context).value
            group_rows.append(
                {
                    "group": label,
                    "real_sessions": r.n_sessions if r is not None else 0,
                    "synthetic_sessions": s.n_sessions if s is not None else 0,
                    "metrics": metrics,
                }
            )

    return DetailedComparison(
        tokens=tuple(tokens),
        transitions=tuple(transitions),
        novel_transitions={
            "count": len(novel),
            "synthetic_share": sum(synth_pairs[p] for p in novel),
            "examples": [f"{a} -> {b}" for a, b in novel[:5]],
        },
        missing_transitions={
            "count": len(missing),
            "real_share": sum(real_pairs[p] for p in missing),
            "examples": [f"{a} -> {b}" for a, b in missing[:5]],
        },
        session_lengths=tuple(_length_rows(real_seqs, synth_seqs)),
        hours=tuple(
            _compare_counts(
                real_time.sessions_by_hour,
                synth_time.sessions_by_hour,
                "hour",
                list(range(24)),
            )
        ),
        weekdays=tuple(
            _compare_counts(
                real_time.sessions_by_weekday,
                synth_time.sessions_by_weekday,
                "weekday",
                list(weekdays),
            )
        ),
        groups=tuple(group_rows),
        by=by,
        timezone=timezone,
    )


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _token_cells(row: Mapping[str, Any]) -> list[str]:
    def seconds(value: float | None) -> str:
        return "-" if value is None else f"{value:.0f}"

    return [
        row["token"],
        _pct(row["real_share"]),
        _pct(row["synthetic_share"]),
        f"{row['difference']:+.1%}",
        seconds(row["real_dwell_median_s"]),
        seconds(row["synthetic_dwell_median_s"]),
    ]


def _group_cells(row: Mapping[str, Any]) -> list[str]:
    metrics = row["metrics"]
    values = [
        "-" if metrics.get(m) is None else f"{metrics[m]:.3f}" for m in GROUP_METRICS
    ]
    return [
        row["group"],
        str(row["real_sessions"]),
        str(row["synthetic_sessions"]),
        *values,
    ]


def _shares(counts: Mapping[Any, int]) -> dict[Any, float]:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def _length_rows(
    real: tuple[tuple[str, ...], ...], synthetic: tuple[tuple[str, ...], ...]
) -> list[dict[str, Any]]:
    def bucket(n: int) -> str:
        return "30+" if n >= 30 else str(n)

    r = _shares(Counter(bucket(len(s)) for s in real))
    s = _shares(Counter(bucket(len(x)) for x in synthetic))
    keys = sorted(set(r) | set(s), key=lambda k: 30 if k == "30+" else int(k))
    return [
        {
            "length": k,
            "real_share": r.get(k, 0.0),
            "synthetic_share": s.get(k, 0.0),
            "difference": s.get(k, 0.0) - r.get(k, 0.0),
        }
        for k in keys
    ]


def _compare_counts(
    real: tuple[int, ...], synthetic: tuple[int, ...], key: str, labels: list[Any]
) -> list[dict[str, Any]]:
    rt, st = sum(real) or 1, sum(synthetic) or 1
    return [
        {
            key: label,
            "real_share": real[i] / rt,
            "synthetic_share": synthetic[i] / st,
            "difference": synthetic[i] / st - real[i] / rt,
        }
        for i, label in enumerate(labels)
    ]


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    def row(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    return [row(header), row(["---"] * len(header)), *(row(r) for r in rows)]
