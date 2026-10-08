"""Session timelines for a small, sampled set of sessions (SAD §29U.1).

Rows are labelled ``session 1..n`` rather than with session or learner ids,
and the number of sessions shown is capped, so the figure cannot be used to
look up individuals.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from eduloggen.analysis import session_sequences
from eduloggen.models import Dataset, LogRecord
from eduloggen.utils import make_rng
from eduloggen.visualization.base import PALETTE, new_figure

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["plot_timeline"]

MAX_SESSIONS = 30


def plot_timeline(dataset: Dataset, *, n_sessions: int = 10, seed: int = 0) -> Figure:
    """Events of randomly sampled sessions along minutes since session start.

    Args:
        dataset: Sessionized dataset.
        n_sessions: Sessions to show (capped at 30).
        seed: Seed for the session sample.
    """
    session_sequences(dataset)
    members: defaultdict[str, list[LogRecord]] = defaultdict(list)
    for event in dataset.sorted_events():
        if event.session_id is not None:
            members[event.session_id].append(event)
    ids = sorted(members)
    rng = make_rng(seed, "visualization", "timeline")
    chosen = sorted(rng.sample(ids, min(n_sessions, MAX_SESSIONS, len(ids))))
    types = sorted({e.event_type for sid in chosen for e in members[sid]})
    colors = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(types)}

    figure, ax = new_figure(height=max(2.5, 0.35 * len(chosen) + 1.2))
    for row, session_id in enumerate(chosen):
        events = members[session_id]
        start = events[0].timestamp
        minutes = [(e.timestamp - start).total_seconds() / 60 for e in events]
        ax.plot(minutes, [row] * len(minutes), color="#999999", linewidth=0.8, zorder=1)
        for minute, event in zip(minutes, events, strict=True):
            ax.scatter(minute, row, color=colors[event.event_type], s=22, zorder=2)
    for event_type, color in colors.items():
        ax.scatter([], [], color=color, label=event_type, s=22)
    ax.set_yticks(range(len(chosen)), [f"session {i + 1}" for i in range(len(chosen))])
    ax.invert_yaxis()
    ax.set_xlabel("minutes since session start")
    ax.set_title(f"Session timelines ({len(chosen)} sampled)")
    if colors:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return figure
