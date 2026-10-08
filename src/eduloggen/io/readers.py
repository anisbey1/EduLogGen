"""Built-in readers for CSV/TSV, JSON Lines, and Parquet (FR-I.1, FR-I.6)."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any, ClassVar

from eduloggen.core import IngestionError, IoError, PluginError
from eduloggen.io.base import (
    BaseReader,
    Format,
    detect_format,
    register_suffixes,
    unregister_suffixes,
)

__all__ = [
    "CsvReader",
    "JsonlReader",
    "ParquetReader",
    "ReaderFactory",
    "TsvReader",
    "available_readers",
    "get_reader",
    "register_reader",
    "require_pyarrow",
    "unregister_reader",
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


ReaderFactory = Callable[[], BaseReader]
"""Zero-argument callable returning a reader (a reader class works)."""

_READERS: dict[str, ReaderFactory] = {
    "csv": CsvReader,
    "tsv": TsvReader,
    "jsonl": JsonlReader,
    "parquet": ParquetReader,
}
BUILTIN_READERS: frozenset[str] = frozenset(_READERS)
"""Names of the readers shipped with EduLogGen."""


def register_reader(
    name: str,
    factory: ReaderFactory,
    *,
    suffixes: Iterable[str] = (),
    replace: bool = False,
) -> None:
    """Add a reader for a new source format (e.g. from a plugin).

    Args:
        name: Format name used with ``format=`` and ``--format``.
        factory: Zero-argument callable returning a :class:`BaseReader`.
        suffixes: File extensions (like ``".xes"``) detected as this format.
        replace: Allow overriding a non-built-in registration.

    Raises:
        PluginError: If the name is invalid, built in, or taken, the factory
            is not callable, or a suffix is already claimed.
    """
    if not isinstance(name, str) or not name.isidentifier():
        raise PluginError(
            "reader names must be identifiers",
            code="plugin_invalid_name",
            context={"name": repr(name)},
        )
    if not callable(factory):
        raise PluginError("factory must be callable", code="plugin_invalid_factory")
    if name in BUILTIN_READERS or (name in _READERS and not replace):
        raise PluginError(
            f"reader {name!r} is already registered",
            code="plugin_duplicate",
            context={"name": name},
        )
    register_suffixes(name, suffixes, replace=replace)
    _READERS[name] = factory


def unregister_reader(name: str) -> None:
    """Remove a non-built-in reader and its suffixes (no-op if absent)."""
    if name in BUILTIN_READERS:
        return
    _READERS.pop(name, None)
    unregister_suffixes(name)


def available_readers() -> list[str]:
    """Registered reader format names, sorted."""
    return sorted(_READERS)


def get_reader(fmt: str, path: Path | str | None = None) -> BaseReader:
    """Return a reader for a format, detecting it from ``path`` if ``"auto"``.

    Raises:
        IngestionError: If the format is unknown or cannot be detected.
        PluginError: If a plugin factory fails or returns a non-reader.
    """
    if fmt == "auto":
        if path is None:
            raise IngestionError("format 'auto' needs a path", code="io_unknown_format")
        fmt = detect_format(path)
    factory = _READERS.get(fmt)
    if factory is None:
        raise IngestionError(
            "unsupported format",
            code="io_unknown_format",
            context={"format": fmt, "supported": sorted(_READERS)},
        )
    try:
        reader = factory()
    except Exception as exc:
        raise PluginError(
            f"reader factory for {fmt!r} failed: {exc}",
            code="plugin_factory_failed",
            context={"name": fmt},
        ) from exc
    if not isinstance(reader, BaseReader):
        raise PluginError(
            f"reader factory for {fmt!r} did not return a BaseReader",
            code="plugin_contract_violation",
            context={"name": fmt},
        )
    return reader
