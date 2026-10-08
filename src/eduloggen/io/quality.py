"""Ingest data quality report (Gate A, SAD §29N).

The report summarizes what happened to every source row without containing
any source values: only counts, canonical field names, source column names,
row numbers, and reason codes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from eduloggen.io.mapping import RowIssue

__all__ = ["QualityReport", "QualityStatus"]

MAX_ISSUE_SAMPLES: Final = 100
"""Number of individual row issues kept as examples in the report."""

QualityStatus = Literal["ready", "failed"]


@dataclass(frozen=True, slots=True, kw_only=True)
class QualityReport:
    """Findings of one ingest run.

    Attributes:
        source: Source file name (not the full path).
        rows_read: Rows read from the source.
        rows_kept: Rows that became events.
        field_coverage: Canonical field name to share of kept events that
            have a value (``0.0``-``1.0``).
        naive_timestamps: Timestamps that had no zone and were given the
            mapping's timezone.
        out_of_order: Events earlier than the previous event of the same
            learner in source order (events are sorted after ingest).
        duplicate_events: Rows dropped because their ``event_id`` repeated.
        issue_counts: Reason code to number of row issues.
        issue_samples: Up to :data:`MAX_ISSUE_SAMPLES` example issues.
        unmapped_columns: Source columns neither mapped, kept, nor dropped.
        warnings: Recoverable problems, as short sentences.
        errors: Fatal problems; a corpus must not be published if non-empty.
    """

    source: str
    rows_read: int
    rows_kept: int
    field_coverage: Mapping[str, float] = field(default_factory=dict, hash=False)
    naive_timestamps: int = 0
    out_of_order: int = 0
    duplicate_events: int = 0
    issue_counts: Mapping[str, int] = field(default_factory=dict, hash=False)
    issue_samples: tuple[RowIssue, ...] = ()
    unmapped_columns: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def rows_dropped(self) -> int:
        """Rows that did not become events."""
        return self.rows_read - self.rows_kept

    @property
    def status(self) -> QualityStatus:
        """``"failed"`` if any fatal error was recorded, else ``"ready"``."""
        return "failed" if self.errors else "ready"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the ``quality_report.json`` layout."""
        return {
            "status": self.status,
            "source": self.source,
            "summary": {
                "rows_read": self.rows_read,
                "rows_kept": self.rows_kept,
                "rows_dropped": self.rows_dropped,
            },
            "field_coverage": dict(self.field_coverage),
            "timestamp_issues": {
                "naive_timezone_assumed": self.naive_timestamps,
                "out_of_order": self.out_of_order,
            },
            "duplicate_events": {"dropped": self.duplicate_events},
            "row_issues": {
                "counts": dict(self.issue_counts),
                "samples": [issue.to_dict() for issue in self.issue_samples],
            },
            "unmapped_columns": list(self.unmapped_columns),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


class IssueLog:
    """Collects row issues, keeping counts and a bounded set of samples."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.samples: list[RowIssue] = []

    def add(self, issue: RowIssue) -> None:
        self.counts[issue.code] += 1
        if len(self.samples) < MAX_ISSUE_SAMPLES:
            self.samples.append(issue)
