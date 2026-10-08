"""Framework-wide constants.

Schema version strings and the default controlled vocabulary live here so that
every package agrees on them without importing higher-level modules.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "ARTIFACT_VERSION",
    "DEFAULT_EVENT_TYPES",
    "EVENT_REQUIRED_FIELDS",
    "OTHER_EVENT_TYPE",
    "REPORT_VERSION",
    "SCHEMA_VERSION",
    "SEED_SCHEME_VERSION",
    "SESSION_REQUIRED_FIELDS",
]

SCHEMA_VERSION: Final = "1.0"
"""Version of the canonical dataset schema (``schema_version`` field)."""

ARTIFACT_VERSION: Final = "1.0"
"""Version of the generator artifact layout (``artifact_version`` field)."""

REPORT_VERSION: Final = "1.0"
"""Version of the validation report layout (``report_version`` field)."""

SEED_SCHEME_VERSION: Final = 1
"""Version of the seed derivation scheme (SAD §29O)."""

OTHER_EVENT_TYPE: Final = "other"
"""Catch-all event type for unknown types when mapping policy requires it."""

DEFAULT_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "view",
        "attempt",
        "submit",
        "navigate",
        "forum_post",
        "video_play",
        OTHER_EVENT_TYPE,
    }
)
"""Default ``event_type`` vocabulary shipped with v1.0 (PRD §9.1, §9.6)."""

EVENT_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "event_id",
    "learner_id",
    "timestamp",
    "activity_id",
    "event_type",
)
"""Fields every canonical event record must provide (PRD §9.1)."""

SESSION_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "session_id",
    "learner_id",
    "start_time",
    "end_time",
    "n_events",
    "duration_s",
    "event_sequence",
)
"""Fields every canonical session record must provide (PRD §9.2)."""
