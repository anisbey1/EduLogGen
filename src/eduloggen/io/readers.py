"""Built-in readers for CSV/TSV, JSON Lines, and Parquet (FR-I.1, FR-I.6)."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

from eduloggen.core import IngestionError, IoError
from eduloggen.io.base import BaseReader, Format, FormatOrAuto, detect_format

__all__ = [
    "CsvReader",
    "JsonlReader",
    "ParquetReader",
    "TsvReader",
    "get_reader",
    "require_pyarrow",
]


class CsvReader(BaseReader):
    """Read comma-separated files with a header row.

    Values are returned as strings; empty cells become ``None``. A UTF-8 byte
    order mark is ignored.
    """

    format: ClassVar[Format] = "csv"
    delimiter = ","

    def __init__(self, encoding: str = "utf-8-sig") -> None:
        """Initialize the reader.

        Args:
            encoding: Text encoding of the file.
        """
        self.encoding = encoding

    def _iter_rows(self, path: Path) -> Iterator[dict[str, Any]]:
        try:
            with path.open(newline="", encoding=self.encoding) as handle:
                reader = csv.DictReader(handle, delimiter=self.delimiter)
                if reader.fieldnames is None:
                    return
                if len(set(reader.fieldnames)) != len(reader.fieldnames):
                    raise IngestionError(
                        "header has duplicate column names",
                        code="io_duplicate_columns",
                        context={"path": str(path)},
                    )
                for line, row in enumerate(reader, start=2):
                    if None in row:
                        raise IngestionError(
                            "row has more cells than the header",
                            code="io_malformed_row",
                            context={"path": str(path), "line": line},
                        )
                    yield {key: (value or None) for key, value in row.items()}
        except UnicodeDecodeError as exc:
            raise IngestionError(
                f"file is not valid {self.encoding} text",
                code="io_bad_encoding",
                context={"path": str(path)},
            ) from exc
        except csv.Error as exc:
            raise IngestionError(
                f"malformed CSV: {exc}",
                code="io_malformed_row",
                context={"path": str(path)},
            ) from exc


class TsvReader(CsvReader):
    """Read tab-separated files with a header row."""

    format = "tsv"
    delimiter = "\t"


class JsonlReader(BaseReader):
    """Read JSON Lines: one JSON object per line; blank lines are skipped."""

    format = "jsonl"

    def _iter_rows(self, path: Path) -> Iterator[dict[str, Any]]:
        try:
            with path.open(encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        raise IngestionError(
                            "line is not valid JSON",
                            code="io_malformed_row",
                            context={"path": str(path), "line": line_no},
                        ) from None
                    if not isinstance(row, dict):
                        raise IngestionError(
                            "line is not a JSON object",
                            code="io_malformed_row",
                            context={"path": str(path), "line": line_no},
                        )
                    yield row
        except UnicodeDecodeError as exc:
            raise IngestionError(
                "file is not valid UTF-8 text",
                code="io_bad_encoding",
                context={"path": str(path)},
            ) from exc


class ParquetReader(BaseReader):
    """Read Parquet files batch by batch. Requires the ``parquet`` extra."""

    format = "parquet"

    def __init__(self, batch_size: int = 65_536) -> None:
        """Initialize the reader.

        Args:
            batch_size: Rows decoded per batch; bounds peak memory.
        """
        self.batch_size = batch_size

    def _iter_rows(self, path: Path) -> Iterator[dict[str, Any]]:
        require_pyarrow()
        import pyarrow.parquet as pq

        try:
            parquet_file = pq.ParquetFile(path)
        except Exception as exc:  # pyarrow raises several unrelated types
            raise IngestionError(
                "file is not a readable Parquet file",
                code="io_malformed_file",
                context={"path": str(path)},
            ) from exc
        for batch in parquet_file.iter_batches(batch_size=self.batch_size):
            yield from batch.to_pylist()


def require_pyarrow() -> Any:
    """Import pyarrow or explain how to install it.

    Raises:
        IoError: If pyarrow is not installed.
    """
    try:
        import pyarrow
    except ImportError:
        raise IoError(
            "Parquet support requires pyarrow; "
            "install it with: pip install 'eduloggen[parquet]'",
            code="io_missing_dependency",
            context={"package": "pyarrow", "extra": "parquet"},
        ) from None
    return pyarrow


_READERS: dict[Format, type[BaseReader]] = {
    "csv": CsvReader,
    "tsv": TsvReader,
    "jsonl": JsonlReader,
    "parquet": ParquetReader,
}


def get_reader(fmt: FormatOrAuto, path: Path | str | None = None) -> BaseReader:
    """Return a reader for a format, detecting it from ``path`` if ``"auto"``.

    Raises:
        IngestionError: If the format is unknown or cannot be detected.
    """
    if fmt == "auto":
        if path is None:
            raise IngestionError("format 'auto' needs a path", code="io_unknown_format")
        fmt = detect_format(path)
    try:
        return _READERS[fmt]()
    except KeyError:
        raise IngestionError(
            "unsupported format",
            code="io_unknown_format",
            context={"format": fmt, "supported": sorted(_READERS)},
        ) from None
