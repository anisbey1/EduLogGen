"""Built-in writers for CSV/TSV, JSON Lines, and Parquet.

Writers receive plain row mappings plus typed columns. Flat formats (CSV,
TSV, Parquet) store ``json`` columns as compact JSON strings; JSON Lines
stores them natively. Timestamps are written as ISO 8601 text, or as
microsecond UTC timestamps in Parquet.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from eduloggen.core import ExportError
from eduloggen.io.base import BaseWriter, Column, ColumnType, Format
from eduloggen.io.readers import require_pyarrow

__all__ = [
    "CsvWriter",
    "JsonlWriter",
    "ParquetWriter",
    "TsvWriter",
    "get_writer",
]


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _plain(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def _text_cell(value: Any, kind: ColumnType) -> str:
    if value is None:
        return ""
    if kind == "json":
        return _json_text(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class CsvWriter(BaseWriter):
    """Write comma-separated UTF-8 text with a header row."""

    format: ClassVar[Format] = "csv"
    suffix = ".csv"
    delimiter = ","

    def write(
        self,
        path: Path,
        rows: Iterable[Mapping[str, Any]],
        columns: Sequence[Column],
    ) -> None:
        """Write rows as delimited text."""
        names = [name for name, _ in columns]
        try:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle, delimiter=self.delimiter)
                writer.writerow(names)
                for row in rows:
                    writer.writerow(
                        [_text_cell(row.get(name), kind) for name, kind in columns]
                    )
        except (OSError, TypeError, ValueError) as exc:
            raise _export_error(path, exc) from exc


class TsvWriter(CsvWriter):
    """Write tab-separated UTF-8 text with a header row."""

    format = "tsv"
    suffix = ".tsv"
    delimiter = "\t"


class JsonlWriter(BaseWriter):
    """Write one JSON object per line."""

    format = "jsonl"
    suffix = ".jsonl"

    def write(
        self,
        path: Path,
        rows: Iterable[Mapping[str, Any]],
        columns: Sequence[Column],
    ) -> None:
        """Write rows as JSON Lines."""
        try:
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    record = {name: _plain(row.get(name)) for name, _ in columns}
                    handle.write(_json_text(record) + "\n")
        except (OSError, TypeError, ValueError) as exc:
            raise _export_error(path, exc) from exc


class ParquetWriter(BaseWriter):
    """Write a Parquet file with an explicit schema. Requires pyarrow."""

    format = "parquet"
    suffix = ".parquet"

    def write(
        self,
        path: Path,
        rows: Iterable[Mapping[str, Any]],
        columns: Sequence[Column],
    ) -> None:
        """Write rows as a single Parquet table."""
        pa = require_pyarrow()
        import pyarrow.parquet as pq

        types = {
            "string": pa.string(),
            "int": pa.int64(),
            "float": pa.float64(),
            "bool": pa.bool_(),
            "timestamp": pa.timestamp("us", tz="UTC"),
            "json": pa.string(),
        }
        schema = pa.schema([(name, types[kind]) for name, kind in columns])
        data: dict[str, list[Any]] = {name: [] for name, _ in columns}
        try:
            for row in rows:
                for name, kind in columns:
                    value = row.get(name)
                    if kind == "json" and value is not None:
                        value = _json_text(value)
                    data[name].append(value)
            table = pa.Table.from_pydict(data, schema=schema)
            pq.write_table(table, path)
        except (OSError, TypeError, ValueError, pa.ArrowException) as exc:
            raise _export_error(path, exc) from exc


def _export_error(path: Path, exc: Exception) -> ExportError:
    return ExportError(
        f"could not write file: {type(exc).__name__}",
        code="export_write_failed",
        context={"path": str(path)},
    )


_WRITERS: dict[Format, type[BaseWriter]] = {
    "csv": CsvWriter,
    "tsv": TsvWriter,
    "jsonl": JsonlWriter,
    "parquet": ParquetWriter,
}


def get_writer(fmt: Format) -> BaseWriter:
    """Return a writer for a format.

    Raises:
        ExportError: If the format is unknown.
    """
    try:
        return _WRITERS[fmt]()
    except KeyError:
        raise ExportError(
            "unsupported format",
            code="export_unknown_format",
            context={"format": fmt, "supported": sorted(_WRITERS)},
        ) from None
