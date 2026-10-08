"""Corpus directories: the on-disk form of a dataset (SAD §29A.3).

Layout::

    corpus_dir/
      manifest.json         # schema version, counts, fingerprint, metadata
      events.<ext>          # canonical events
      sessions.<ext>        # canonical sessions, if sessionized
      mapping.used.json     # mapping applied at ingest (salts redacted)
      quality_report.json   # ingest Gate A findings

Writes go to a temporary sibling directory that is renamed into place, so a
failed write never leaves a half-written corpus. Existing paths are only
replaced with ``force=True`` and only if they already hold a corpus or are
empty. Reads verify the content fingerprint recorded in the manifest.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast, get_args

from eduloggen.core import ExportError, IngestionError, PathLike, SchemaError
from eduloggen.io import coerce
from eduloggen.io.base import Column, Format
from eduloggen.io.mapping import FieldMapping
from eduloggen.io.quality import QualityReport
from eduloggen.io.readers import get_reader
from eduloggen.io.writers import get_writer
from eduloggen.models import (
    Dataset,
    DatasetMetadata,
    GenerationMetadata,
    LogRecord,
    Session,
    SyntheticDataset,
    check_schema_version,
)
from eduloggen.utils.fs import atomic_directory, write_json

__all__ = ["EVENT_COLUMNS", "SESSION_COLUMNS", "read_corpus", "write_corpus"]

MANIFEST: Final = "manifest.json"
QUALITY_REPORT: Final = "quality_report.json"
MAPPING_USED: Final = "mapping.used.json"

EVENT_COLUMNS: Final[tuple[Column, ...]] = (
    ("event_id", "string"),
    ("learner_id", "string"),
    ("timestamp", "timestamp"),
    ("session_id", "string"),
    ("activity_id", "string"),
    ("event_type", "string"),
    ("course_id", "string"),
    ("score", "float"),
    ("success", "bool"),
    ("duration_ms", "int"),
    ("metadata", "json"),
)
"""Canonical event columns in file order."""

SESSION_COLUMNS: Final[tuple[Column, ...]] = (
    ("session_id", "string"),
    ("learner_id", "string"),
    ("start_time", "timestamp"),
    ("end_time", "timestamp"),
    ("n_events", "int"),
    ("duration_s", "float"),
    ("event_sequence", "json"),
    ("course_id", "string"),
)
"""Canonical session columns in file order."""

_SUFFIX: Final[dict[str, str]] = {
    "csv": ".csv",
    "tsv": ".tsv",
    "jsonl": ".jsonl",
    "parquet": ".parquet",
}


def write_corpus(
    dataset: Dataset,
    path: PathLike,
    *,
    format: Format = "csv",
    quality_report: QualityReport | None = None,
    mapping: FieldMapping | None = None,
    force: bool = False,
) -> Path:
    """Write a dataset as a corpus directory.

    Args:
        dataset: Dataset to write.
        path: Target directory.
        format: Table format for events and sessions.
        quality_report: Ingest report to store alongside the data.
        mapping: Ingest mapping to store for provenance.
        force: Replace an existing corpus (or empty directory) at ``path``.

    Returns:
        The absolute corpus path.

    Raises:
        ExportError: If the quality report has fatal errors, the target
            exists without ``force``, the target is not safe to replace, or
            writing fails.
    """
    if quality_report is not None and quality_report.status == "failed":
        raise ExportError(
            "refusing to write a corpus whose quality report has errors",
            code="export_quality_failed",
            context={"errors": list(quality_report.errors)},
        )
    if format not in _SUFFIX:
        raise ExportError(
            "unsupported format",
            code="export_unknown_format",
            context={"format": format, "supported": sorted(_SUFFIX)},
        )
    target = Path(path).expanduser().absolute()
    with atomic_directory(target, marker=MANIFEST, force=force) as staging:
        files = _write_contents(staging, dataset, format, quality_report, mapping)
        manifest = dataset.to_manifest() | {"format": format, "files": files}
        write_json(staging / MANIFEST, manifest)
    return target


def read_corpus(path: PathLike) -> Dataset:
    """Load a corpus directory written by :func:`write_corpus`.

    Args:
        path: Corpus directory.

    Returns:
        The dataset (a :class:`SyntheticDataset` if the manifest says so).

    Raises:
        IngestionError: If the directory, manifest, or files are missing or
            malformed, or the content fingerprint does not match.
        SchemaError: If records violate the canonical schema or the schema
            version is incompatible.
    """
    root = Path(path).expanduser().resolve()
    manifest = _read_manifest(root)
    check_schema_version(_get(manifest, "schema_version", str))
    fmt = _get(manifest, "format", str)
    if fmt not in get_args(Format):
        raise _corrupt(root, "manifest names an unsupported format")
    files = _get(manifest, "files", dict)

    events = tuple(
        LogRecord.from_dict(row)
        for row in _read_table(
            root, files.get("events"), cast(Format, fmt), EVENT_COLUMNS
        )
    )
    sessions: tuple[Session, ...] | None = None
    if files.get("sessions") is not None:
        sessions = tuple(
            Session.from_dict(row)
            for row in _read_table(
                root, files["sessions"], cast(Format, fmt), SESSION_COLUMNS
            )
        )

    common: dict[str, Any] = {
        "dataset_id": _get(manifest, "dataset_id", str),
        "events": events,
        "sessions": sessions,
        "metadata": DatasetMetadata.from_dict(_get(manifest, "metadata", dict)),
        "schema_version": manifest["schema_version"],
    }
    dataset: Dataset
    if manifest.get("synthetic"):
        generation = GenerationMetadata(**_get(manifest, "generation", dict))
        dataset = SyntheticDataset(generation=generation, **common)
    else:
        dataset = Dataset(**common)

    if dataset.fingerprint() != manifest.get("fingerprint"):
        raise IngestionError(
            "corpus content does not match its manifest fingerprint",
            code="io_corpus_integrity",
            context={"path": str(root)},
        )
    return dataset


# ---------------------------------------------------------------------------
# Writing helpers
# ---------------------------------------------------------------------------


def _write_contents(
    staging: Path,
    dataset: Dataset,
    fmt: Format,
    report: QualityReport | None,
    mapping: FieldMapping | None,
) -> dict[str, str | None]:
    writer = get_writer(fmt)
    events_name = "events" + _SUFFIX[fmt]
    writer.write(staging / events_name, map(_event_row, dataset.events), EVENT_COLUMNS)
    sessions_name: str | None = None
    if dataset.sessions is not None:
        sessions_name = "sessions" + _SUFFIX[fmt]
        writer.write(
            staging / sessions_name,
            map(_session_row, dataset.sessions),
            SESSION_COLUMNS,
        )
    files: dict[str, str | None] = {"events": events_name, "sessions": sessions_name}
    if report is not None:
        write_json(staging / QUALITY_REPORT, report.to_dict())
        files["quality_report"] = QUALITY_REPORT
    if mapping is not None:
        write_json(staging / MAPPING_USED, mapping.to_dict())
        files["mapping"] = MAPPING_USED
    return files


def _event_row(event: LogRecord) -> dict[str, Any]:
    return {
        name: dict(event.metadata) if name == "metadata" else getattr(event, name)
        for name, _ in EVENT_COLUMNS
    }


def _session_row(session: Session) -> dict[str, Any]:
    row = {name: getattr(session, name) for name, _ in SESSION_COLUMNS}
    row["event_sequence"] = list(session.event_sequence)
    return row


# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        raise IngestionError(
            "corpus manifest not found",
            code="io_not_found",
            context={"path": str(manifest_path)},
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _corrupt(root, "manifest is not readable JSON") from exc
    if not isinstance(manifest, dict):
        raise _corrupt(root, "manifest is not a JSON object")
    return manifest


def _get(manifest: Mapping[str, Any], key: str, kind: type[Any]) -> Any:
    value = manifest.get(key)
    if not isinstance(value, kind):
        raise IngestionError(
            f"manifest field {key!r} is missing or has the wrong type",
            code="io_corpus_corrupt",
            context={"field": key},
        )
    return value


def _read_table(
    root: Path, name: object, fmt: Format, columns: Sequence[Column]
) -> Iterator[dict[str, Any]]:
    if not isinstance(name, str) or Path(name).name != name:
        raise _corrupt(root, "manifest names an invalid data file")
    for row in get_reader(fmt).read(root / name):
        yield _decode_row(row, columns)


_DECODERS: Final[dict[str, Callable[[Any], Any]]] = {
    "int": coerce.to_int,
    "float": coerce.to_float,
    "bool": coerce.to_bool,
    "timestamp": lambda value: coerce.to_datetime(value)[0],
}


def _decode_row(row: Mapping[str, Any], columns: Sequence[Column]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for name, kind in columns:
        value = row.get(name)
        if value is None or value == "":
            decoded[name] = None
        elif kind == "json":
            try:
                decoded[name] = json.loads(value) if isinstance(value, str) else value
            except json.JSONDecodeError:
                raise SchemaError(
                    f"corpus column {name!r} is not valid JSON",
                    code="schema_invalid_value",
                    context={"field": name},
                ) from None
        elif kind == "string":
            decoded[name] = value
        else:
            try:
                decoded[name] = _DECODERS[kind](value)
            except coerce.CoercionError as exc:
                raise SchemaError(
                    f"corpus column {name!r} has an invalid value",
                    code="schema_invalid_value",
                    context={"field": name, "reason": exc.reason},
                ) from None
    return decoded


def _corrupt(root: Path, reason: str) -> IngestionError:
    return IngestionError(reason, code="io_corpus_corrupt", context={"path": str(root)})
