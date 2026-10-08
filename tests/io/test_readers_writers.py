"""Tests for readers, writers, format detection, and chunking."""

from __future__ import annotations

import builtins
import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from eduloggen.core import ExportError, IngestionError, IoError
from eduloggen.io import (
    Column,
    CsvReader,
    JsonlReader,
    ParquetReader,
    TsvReader,
    chunked,
    detect_format,
    get_reader,
    get_writer,
)
from eduloggen.io.base import resolve_source

COLUMNS: tuple[Column, ...] = (
    ("id", "string"),
    ("when", "timestamp"),
    ("n", "int"),
    ("x", "float"),
    ("ok", "bool"),
    ("extra", "json"),
)
T0 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
ROWS: list[dict[str, Any]] = [
    {"id": "a", "when": T0, "n": 1, "x": 0.5, "ok": True, "extra": {"k": [1, 2]}},
    {"id": "b", "when": None, "n": None, "x": None, "ok": False, "extra": None},
]


@pytest.mark.parametrize(
    ("name", "fmt"),
    [
        ("a.csv", "csv"),
        ("a.TSV", "tsv"),
        ("a.jsonl", "jsonl"),
        ("a.ndjson", "jsonl"),
        ("a.parquet", "parquet"),
        ("a.pq", "parquet"),
    ],
)
def test_detect_format(name: str, fmt: str) -> None:
    assert detect_format(name) == fmt


def test_detect_format_unknown() -> None:
    with pytest.raises(IngestionError) as info:
        detect_format("a.xlsx")
    assert info.value.code == "io_unknown_format"


def test_get_reader() -> None:
    assert isinstance(get_reader("auto", "x.tsv"), TsvReader)
    assert isinstance(get_reader("jsonl"), JsonlReader)
    with pytest.raises(IngestionError):
        get_reader("auto")
    with pytest.raises(IngestionError):
        get_reader("xml")


def test_get_writer_unknown() -> None:
    with pytest.raises(ExportError):
        get_writer("xml")  # type: ignore[arg-type]


def test_resolve_source_errors(tmp_path: Path) -> None:
    with pytest.raises(IngestionError) as info:
        resolve_source(tmp_path / "missing.csv")
    assert info.value.code == "io_not_found"
    with pytest.raises(IngestionError) as info:
        resolve_source(tmp_path)
    assert info.value.code == "io_not_a_file"


# --------------------------------------------------------------------------
# CSV / TSV
# --------------------------------------------------------------------------


def test_csv_reader_handles_bom_and_empty_cells(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    path.write_bytes("﻿id,score\nx,\ny,3\n".encode())
    assert list(CsvReader().read(path)) == [
        {"id": "x", "score": None},
        {"id": "y", "score": "3"},
    ]


def test_csv_reader_short_rows_and_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    path.write_text("id,score\nx\n")
    assert list(CsvReader().read(path)) == [{"id": "x", "score": None}]
    path.write_text("")
    assert list(CsvReader().read(path)) == []


@pytest.mark.parametrize(
    ("content", "code"),
    [
        ("id,id\n1,2\n", "io_duplicate_columns"),
        ("id\n1,2\n", "io_malformed_row"),
    ],
)
def test_csv_reader_rejects_malformed(tmp_path: Path, content: str, code: str) -> None:
    path = tmp_path / "a.csv"
    path.write_text(content)
    with pytest.raises(IngestionError) as info:
        list(CsvReader().read(path))
    assert info.value.code == code


def test_csv_reader_wraps_parser_errors(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    path.write_text("id\n" + "x" * 50 + "\n")
    original = csv.field_size_limit(10)
    try:
        with pytest.raises(IngestionError) as info:
            list(CsvReader().read(path))
    finally:
        csv.field_size_limit(original)
    assert info.value.code == "io_malformed_row"


def test_csv_reader_rejects_bad_encoding(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    path.write_bytes(b"id\n\xff\xfe\n")
    with pytest.raises(IngestionError) as info:
        list(CsvReader().read(path))
    assert info.value.code == "io_bad_encoding"


def test_tsv_reader(tmp_path: Path) -> None:
    path = tmp_path / "a.tsv"
    path.write_text("id\tscore\nx\t1\n")
    assert list(TsvReader().read(path)) == [{"id": "x", "score": "1"}]


@pytest.mark.parametrize("fmt", ["csv", "tsv"])
def test_text_writer_round_trip(tmp_path: Path, fmt: Any) -> None:
    path = tmp_path / f"out.{fmt}"
    get_writer(fmt).write(path, ROWS, COLUMNS)
    rows = list(get_reader(fmt).read(path))
    assert rows[0] == {
        "id": "a",
        "when": "2026-03-01T09:00:00+00:00",
        "n": "1",
        "x": "0.5",
        "ok": "true",
        "extra": '{"k":[1,2]}',
    }
    assert rows[1] == {
        "id": "b",
        "when": None,
        "n": None,
        "x": None,
        "ok": "false",
        "extra": None,
    }


# --------------------------------------------------------------------------
# JSON Lines
# --------------------------------------------------------------------------


def test_jsonl_round_trip_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    get_writer("jsonl").write(path, ROWS, COLUMNS)
    path.write_text(path.read_text() + "\n   \n")
    rows = list(JsonlReader().read(path))
    assert rows[0]["when"] == "2026-03-01T09:00:00+00:00"
    assert rows[0]["extra"] == {"k": [1, 2]}
    assert rows[1]["n"] is None


@pytest.mark.parametrize("line", ["{bad json", "[1, 2]"])
def test_jsonl_rejects_malformed(tmp_path: Path, line: str) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"id": 1}\n' + line + "\n")
    with pytest.raises(IngestionError) as info:
        list(JsonlReader().read(path))
    assert info.value.context["line"] == 2


def test_jsonl_rejects_bad_encoding(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_bytes(b'{"id": "\xff"}\n')
    with pytest.raises(IngestionError) as info:
        list(JsonlReader().read(path))
    assert info.value.code == "io_bad_encoding"


def test_writer_wraps_os_errors(tmp_path: Path) -> None:
    missing_dir = tmp_path / "nope" / "a.csv"
    with pytest.raises(ExportError) as info:
        get_writer("csv").write(missing_dir, ROWS, COLUMNS)
    assert info.value.code == "export_write_failed"
    with pytest.raises(ExportError):
        get_writer("jsonl").write(tmp_path / "nope" / "a.jsonl", ROWS, COLUMNS)


def test_jsonl_writer_rejects_unserializable(tmp_path: Path) -> None:
    with pytest.raises(ExportError):
        get_writer("jsonl").write(
            tmp_path / "a.jsonl", [{"id": object()}], [("id", "string")]
        )


# --------------------------------------------------------------------------
# Parquet
# --------------------------------------------------------------------------


def test_parquet_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "a.parquet"
    get_writer("parquet").write(path, ROWS, COLUMNS)
    rows = list(ParquetReader(batch_size=1).read(path))
    assert rows[0]["when"] == T0
    assert rows[0]["extra"] == '{"k":[1,2]}'
    assert rows[0]["n"] == 1
    assert rows[1] == {
        "id": "b",
        "when": None,
        "n": None,
        "x": None,
        "ok": False,
        "extra": None,
    }


def test_parquet_reader_rejects_non_parquet(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "a.parquet"
    path.write_text("not parquet")
    with pytest.raises(IngestionError) as info:
        list(ParquetReader().read(path))
    assert info.value.code == "io_malformed_file"


def test_parquet_writer_rejects_bad_types(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    with pytest.raises(ExportError):
        get_writer("parquet").write(
            tmp_path / "a.parquet", [{"n": "not-int"}], [("n", "int")]
        )


def test_parquet_without_pyarrow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    path = tmp_path / "a.parquet"
    path.write_bytes(b"")
    with pytest.raises(IoError) as info:
        list(ParquetReader().read(path))
    assert info.value.code == "io_missing_dependency"
    assert "eduloggen[parquet]" in info.value.message
    with pytest.raises(IoError):
        get_writer("parquet").write(path, ROWS, COLUMNS)


# --------------------------------------------------------------------------
# Streaming
# --------------------------------------------------------------------------


def test_chunked() -> None:
    assert list(chunked(range(5), 2)) == [[0, 1], [2, 3], [4]]
    assert list(chunked([], 3)) == []
    with pytest.raises(ValueError):
        list(chunked([1], 0))
