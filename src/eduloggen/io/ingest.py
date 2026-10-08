"""Turn a source log file into a validated :class:`Dataset` (Gate A).

``ingest`` streams rows from a reader, maps them with a :class:`FieldMapping`,
normalizes event types, drops duplicates, sorts events by learner and time
(FR-I.5), and records everything it did in a :class:`QualityReport`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from eduloggen.core import PathLike, SchemaError
from eduloggen.io.base import FormatOrAuto, resolve_source
from eduloggen.io.mapping import FieldMapping, RowIssue
from eduloggen.io.quality import IssueLog, QualityReport
from eduloggen.io.readers import get_reader
from eduloggen.models import Dataset, DatasetMetadata, EventVocabulary, LogRecord

__all__ = ["DuplicatePolicy", "IngestResult", "ingest"]

DuplicatePolicy = Literal["keep_first", "error"]
"""What to do when an ``event_id`` repeats."""

_FIXED_COLUMN_FORMATS = frozenset({"csv", "tsv", "parquet"})


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Output of :func:`ingest`.

    Attributes:
        dataset: Validated events sorted by learner and time.
        report: Data quality findings.
        mapping: Mapping that was applied.
    """

    dataset: Dataset
    report: QualityReport
    mapping: FieldMapping


def ingest(
    source: PathLike,
    mapping: FieldMapping,
    *,
    format: FormatOrAuto = "auto",
    strict: bool = False,
    on_duplicate: DuplicatePolicy = "keep_first",
    vocabulary: EventVocabulary | None = None,
    dataset_id: str | None = None,
    privacy_notes: str = "",
) -> IngestResult:
    """Read, map, and validate a source log file.

    Args:
        source: Source file path.
        mapping: Source-to-canonical field mapping.
        format: File format, or ``"auto"`` to detect it from the extension.
        strict: Raise on the first invalid row instead of dropping it.
        on_duplicate: Keep the first event of a repeated ``event_id``, or
            raise.
        vocabulary: Event type vocabulary; defaults to the built-in one with
            unknown types preserved.
        dataset_id: Dataset name; defaults to the source file stem.
        privacy_notes: Notes stored in the dataset metadata.

    Returns:
        The dataset, quality report, and mapping used.

    Raises:
        IngestionError: If the source is missing or cannot be parsed.
        SchemaError: If required columns are absent, or in strict mode if a
            row is invalid, or if a duplicate is found with
            ``on_duplicate="error"``.
    """
    path = resolve_source(source)
    reader = get_reader(format, path)
    vocab = vocabulary if vocabulary is not None else EventVocabulary()
    issues = IssueLog()
    events: list[LogRecord] = []
    seen_ids: set[str] = set()
    last_seen: dict[str, datetime] = {}
    columns: set[str] = set()
    rows_read = naive = out_of_order = duplicates = 0

    for row_number, row in enumerate(reader.read(path), start=1):
        rows_read = row_number
        if reader.format not in _FIXED_COLUMN_FORMATS:
            columns.update(row)
        elif row_number == 1:
            _require_columns(mapping, row)
            columns.update(row)

        mapped = mapping.apply(row, row_number)
        row_issues = list(mapped.issues)
        event: LogRecord | None = None
        if mapped.values is not None:
            try:
                mapped.values["event_type"] = vocab.normalize(
                    mapped.values["event_type"]
                )
                event = LogRecord(**mapped.values)
            except SchemaError as exc:
                field = str(exc.context.get("field", "event"))
                row_issues.append(RowIssue(row_number, field, exc.code))

        if event is None:
            if strict:
                first = row_issues[0]
                raise SchemaError(
                    f"row {row_number} is invalid: {first.field} ({first.code})",
                    code="ingest_row_invalid",
                    context=first.to_dict(),
                )
            for issue in row_issues:
                issues.add(issue)
            continue

        if event.event_id in seen_ids:
            if on_duplicate == "error":
                raise SchemaError(
                    f"row {row_number} repeats an earlier event_id",
                    code="ingest_duplicate_event",
                    context={"row": row_number},
                )
            duplicates += 1
            continue
        seen_ids.add(event.event_id)

        previous = last_seen.get(event.learner_id)
        if previous is not None and event.timestamp < previous:
            out_of_order += 1
        last_seen[event.learner_id] = max(previous or event.timestamp, event.timestamp)
        naive += mapped.assumed_timezone
        events.append(event)

    events.sort(key=lambda event: event.sort_key)
    report = _build_report(
        source=path.name,
        mapping=mapping,
        events=events,
        rows_read=rows_read,
        naive=naive,
        out_of_order=out_of_order,
        duplicates=duplicates,
        issues=issues,
        unmapped=sorted(columns - mapping.known_columns()),
    )
    dataset = Dataset(
        dataset_id=dataset_id or path.stem,
        events=tuple(events),
        metadata=DatasetMetadata(
            source_description=f"ingested from {path.name}",
            privacy_notes=privacy_notes,
        ),
    )
    return IngestResult(dataset=dataset, report=report, mapping=mapping)


def _require_columns(mapping: FieldMapping, row: dict[str, object]) -> None:
    missing = mapping.missing_columns(row)
    if missing:
        raise SchemaError(
            f"source is missing required columns: {', '.join(missing)}",
            code="ingest_missing_columns",
            context={"columns": missing},
        )


def _build_report(
    *,
    source: str,
    mapping: FieldMapping,
    events: list[LogRecord],
    rows_read: int,
    naive: int,
    out_of_order: int,
    duplicates: int,
    issues: IssueLog,
    unmapped: list[str],
) -> QualityReport:
    kept = len(events)
    coverage = {
        name: (
            sum(getattr(event, name) is not None for event in events) / kept
            if kept
            else 0.0
        )
        for name in sorted({*mapping.fields, "event_id"})
    }
    dropped = rows_read - kept
    warnings: list[str] = []
    errors: list[str] = []
    if dropped:
        warnings.append(f"{dropped} of {rows_read} rows were dropped")
    if duplicates:
        warnings.append(f"{duplicates} rows repeated an earlier event_id")
    if naive:
        warnings.append(f"{naive} naive timestamps were given the mapping timezone")
    if unmapped:
        warnings.append(f"{len(unmapped)} source columns were not mapped")
    if rows_read == 0:
        errors.append("source contains no rows")
    elif kept == 0:
        errors.append("no row produced a valid event")
    return QualityReport(
        source=source,
        rows_read=rows_read,
        rows_kept=kept,
        field_coverage=coverage,
        naive_timestamps=naive,
        out_of_order=out_of_order,
        duplicate_events=duplicates,
        issue_counts=dict(issues.counts),
        issue_samples=tuple(issues.samples),
        unmapped_columns=tuple(unmapped),
        warnings=tuple(warnings),
        errors=tuple(errors),
    )
