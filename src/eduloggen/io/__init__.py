"""Reading and writing educational interaction logs (SAD §7.2.4).

``io`` streams raw rows from CSV, TSV, JSON Lines, and Parquet files, maps
them onto the canonical schema with a :class:`FieldMapping`, and turns them
into validated :class:`~eduloggen.models.Dataset` snapshots with a data
quality report. It also reads and writes corpus directories.

Typical use::

    from eduloggen.io import FieldMapping, ingest, write_corpus

    mapping = FieldMapping.from_dict({"fields": {...}})
    result = ingest("events.csv", mapping)
    write_corpus(result.dataset, "corpus/", quality_report=result.report,
                 mapping=result.mapping)

Parquet support needs the ``parquet`` extra (``pip install
'eduloggen[parquet]'``).
"""

from __future__ import annotations

from eduloggen.io.base import (
    BaseReader,
    BaseWriter,
    Column,
    ColumnType,
    Format,
    FormatOrAuto,
    detect_format,
)
from eduloggen.io.corpus import (
    EVENT_COLUMNS,
    SESSION_COLUMNS,
    read_annotations,
    read_corpus,
    write_corpus,
)
from eduloggen.io.ingest import DuplicatePolicy, IngestResult, ingest
from eduloggen.io.mapping import (
    FieldMapping,
    FieldSpec,
    MappedRow,
    RowIssue,
    pseudonymize,
)
from eduloggen.io.quality import QualityReport, QualityStatus
from eduloggen.io.readers import (
    BUILTIN_READERS,
    CsvReader,
    JsonlReader,
    ParquetReader,
    ReaderFactory,
    TsvReader,
    available_readers,
    get_reader,
    register_reader,
    unregister_reader,
)
from eduloggen.io.streaming import chunked
from eduloggen.io.writers import (
    CsvWriter,
    JsonlWriter,
    ParquetWriter,
    TsvWriter,
    get_writer,
)

__all__ = [
    "BUILTIN_READERS",
    "EVENT_COLUMNS",
    "SESSION_COLUMNS",
    "BaseReader",
    "BaseWriter",
    "Column",
    "ColumnType",
    "CsvReader",
    "CsvWriter",
    "DuplicatePolicy",
    "FieldMapping",
    "FieldSpec",
    "Format",
    "FormatOrAuto",
    "IngestResult",
    "JsonlReader",
    "JsonlWriter",
    "MappedRow",
    "ParquetReader",
    "ParquetWriter",
    "QualityReport",
    "QualityStatus",
    "ReaderFactory",
    "RowIssue",
    "TsvReader",
    "TsvWriter",
    "available_readers",
    "chunked",
    "detect_format",
    "get_reader",
    "get_writer",
    "ingest",
    "pseudonymize",
    "read_annotations",
    "read_corpus",
    "register_reader",
    "unregister_reader",
    "write_corpus",
]
