"""Per-learner aggregate profiles (SAD §11.9, §29C.6).

Profiles use pseudonymous learner ids only and contain no free text.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from eduloggen.analysis.sequences import session_sequences
from eduloggen.models import Dataset, Session

__all__ = ["learner_profiles"]


def learner_profiles(dataset: Dataset) -> list[dict[str, Any]]:
    """One row per learner, ordered by ``learner_id``.

    Columns: ``learner_id``, ``learner_n_sessions``, ``n_events``,
    ``mean_session_length``, ``mean_session_duration_s``, ``n_courses``.

    Raises:
        AnalysisError: If the dataset has not been sessionized.
    """
    session_sequences(dataset)
    sessions: defaultdict[str, list[Session]] = defaultdict(list)
    for session in dataset.sessions or ():
        sessions[session.learner_id].append(session)
    rows = []
    for participant in dataset.participants():
        owned = sessions[participant.learner_id]
        n_events = sum(session.n_events for session in owned)
        rows.append(
            {
                "learner_id": participant.learner_id,
                "learner_n_sessions": len(owned),
                "n_events": n_events,
                "mean_session_length": n_events / len(owned) if owned else 0.0,
                "mean_session_duration_s": (
                    sum(session.duration_s for session in owned) / len(owned)
                    if owned
                    else 0.0
                ),
                "n_courses": len(participant.course_ids),
            }
        )
    return rows
