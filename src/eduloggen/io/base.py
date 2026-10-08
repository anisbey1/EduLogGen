"""Reader and writer contracts plus format detection (SAD §7.3, §24.1).

Readers stream raw rows (``dict`` of source column to value) one at a time so
large files never need to be fully loaded before validation. Mapping rows to
canonical records is the job of :class:`~eduloggen.io.mapping.FieldMapping`,
not of readers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

from eduloggen.core import IngestionError, PathLike

__all__ = [
    "BaseReader",
    "BaseWriter",
    "Column",
    "ColumnType",
    "Format",
    "FormatOrAuto",
    "detect_format",
    "resolve_source",
]

Format = Literal["csv", "tsv", "jsonl", "parquet"]
"""Supported tabular file formats."""

FormatOrAuto = Format | Literal["auto"]
"""A format, or ``"auto"`` to detect it from the file extension."""

ColumnType = Literal["string", "int", "float", "bool", "timestamp", "json"]
"""Logical column type; ``json`` holds nested values (lists, mappings)."""

Column = tuple[str, ColumnType]
"""A column name and its logical type."""

_SUFFIXES: dict[str, Format] = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".parquet": "parquet",
    ".pq": "parquet",
}


def detect_format(path: PathLike) -> Format:
    """Infer the file format from the extension.

    Raises:
        IngestionError: If the extension is not recognised.
    """
    suffix = Path(path).suffix.lower()
    try:
        return _SUFFIXES[suffix]
    except KeyError:
        raise IngestionError(
            "cannot detect file format from extension; pass format explicitly",
            code="io_unknown_format",
            context={"suffix": suffix, "supported": sorted(_SUFFIXES)},
        ) from None


def resolve_source(path: PathLike) -> Path:
    """Resolve a source path and ensure it is a readable regular file.

    Raises:
        IngestionError: If the path does not exist or is not a file.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise IngestionError(
            "source file does not exist",
            code="io_not_found",
            context={"path": str(resolved)},
        )
    if not resolved.is_file():
        raise IngestionError(
            "source path is not a regular file",
            code="io_not_a_file",
            context={"path": str(resolved)},
        )
    return resolved


class BaseReader(ABC):
    """Stream raw rows from one source file.

    Subclasses set :attr:`format` and implement :meth:`_iter_rows`.
    """

    format: ClassVar[Format]

    def read(self, path: PathLike) -> Iterator[dict[str, Any]]:
        """Yield the rows of ``path`` lazily.

        Args:
            path: Source file.

        Yields:
            One mapping of source column name to raw value per row.

        Raises:
            IngestionError: If the file is missing or cannot be parsed.
        """
        return self._iter_rows(resolve_source(path))

    @abstractmethod
    def _iter_rows(self, path: Path) -> Iterator[dict[str, Any]]:
        """Yield rows from an existing file."""


class BaseWriter(ABC):
    """Write rows of plain values to one file in a given format."""

    format: ClassVar[Format]
    suffix: ClassVar[str]

    @abstractmethod
    def write(
        self,
        path: Path,
        rows: Iterable[Mapping[str, Any]],
        columns: Sequence[Column],
    ) -> None:
        """Write ``rows`` to ``path`` with the given ordered, typed columns.

        Raises:
            ExportError: If the file cannot be written.
        """
