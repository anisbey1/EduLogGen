"""Tests for corpus directory read/write."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from eduloggen.core import ExportError, IngestionError, SchemaError
from eduloggen.io import FieldMapping, QualityReport, read_corpus, write_corpus
from eduloggen.models import (
    Dataset,
    DatasetMetadata,
    GenerationMetadata,
    LogRecord,
    Session,
    SyntheticDataset,
)

T0 = datetime(2026, 3, 1, 9, 0, 0, 123456, tzinfo=UTC)
FORMATS = ["csv", "tsv", "jsonl", "parquet"]


def _event(i: int, learner: str, session: str | None, **extra: Any) -> LogRecord:
    return LogRecord(
        event_id=f"e{i}",
        learner_id=learner,
        timestamp=T0 + timedelta(seconds=37 * i),
        activity_id=f"a{i % 2}",
        event_type="view" if i % 2 else "attempt",
        session_id=session,
        **extra,
    )


@pytest.fixture
def dataset() -> Dataset:
    events = (
        _event(0, "L1", "s1", course_id="c1", score=0.1, success=True, duration_ms=5),
        _event(1, "L1", "s1", metadata={"device": "mobile", "tags": [1, "x"]}),
        _event(2, "L2", "s2", success=False),
        _event(3, "L2", None),
    )
    sessions = (
        Session.from_events("s1", events[:2]),
        Session.from_events("s2", events[2:3]),
    )
    return Dataset(
        dataset_id="demo",
        events=events,
        sessions=sessions,
        metadata=DatasetMetadata(source_description="unit test", seeds={"gen": 3}),
    )


def _ready_report() -> QualityReport:
    return QualityReport(source="x.csv", rows_read=4, rows_kept=4)


@pytest.mark.parametrize("fmt", FORMATS)
def test_round_trip(tmp_path: Path, dataset: Dataset, fmt: Any) -> None:
    if fmt == "parquet":
        pytest.importorskip("pyarrow")
    target = write_corpus(dataset, tmp_path / "corpus", format=fmt)
    restored = read_corpus(target)
    assert restored == dataset
    assert restored.fingerprint() == dataset.fingerprint()
    assert type(restored) is Dataset


def test_layout_and_manifest(tmp_path: Path, dataset: Dataset) -> None:
    mapping = FieldMapping.from_dict(
        {
            "fields": {
                "learner_id": {"source": "u", "hash": {"salt": "secret-salt"}},
                "timestamp": "t",
                "activity_id": "a",
                "event_type": "e",
            }
        }
    )
    target = write_corpus(
        dataset, tmp_path / "c", quality_report=_ready_report(), mapping=mapping
    )
    assert sorted(p.name for p in target.iterdir()) == [
        "events.csv",
        "manifest.json",
        "mapping.used.json",
        "quality_report.json",
        "sessions.csv",
    ]
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["format"] == "csv"
    assert manifest["files"]["sessions"] == "sessions.csv"
    assert manifest["counts"] == {"events": 4, "sessions": 2, "learners": 2}
    assert "secret-salt" not in (target / "mapping.used.json").read_text()
    assert json.loads((target / "quality_report.json").read_text())["status"] == "ready"
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_unsessionized_and_synthetic(tmp_path: Path, dataset: Dataset) -> None:
    raw = Dataset(dataset_id="raw", events=dataset.events)
    restored = read_corpus(write_corpus(raw, tmp_path / "raw", format="jsonl"))
    assert restored.sessions is None
    assert not (tmp_path / "raw" / "sessions.jsonl").exists()

    synthetic = SyntheticDataset(
        dataset_id="syn",
        events=dataset.events,
        sessions=dataset.sessions,
        generation=GenerationMetadata(
            generator_id="markov", model_fingerprint="sha256:x", seed=1, n_sessions=2
        ),
    )
    loaded = read_corpus(write_corpus(synthetic, tmp_path / "syn"))
    assert isinstance(loaded, SyntheticDataset)
    assert loaded == synthetic


def test_refuses_existing_without_force(tmp_path: Path, dataset: Dataset) -> None:
    target = write_corpus(dataset, tmp_path / "c")
    with pytest.raises(ExportError) as info:
        write_corpus(dataset, target)
    assert info.value.code == "export_exists"


def test_force_replaces_corpus_or_empty_dir(tmp_path: Path, dataset: Dataset) -> None:
    target = write_corpus(dataset, tmp_path / "c", format="jsonl")
    write_corpus(dataset, target, format="csv", force=True)
    assert (target / "events.csv").exists()
    assert not (target / "events.jsonl").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c"]

    empty = tmp_path / "empty"
    empty.mkdir()
    write_corpus(dataset, empty, force=True)
    assert (empty / "manifest.json").exists()


@pytest.mark.parametrize("kind", ["file", "foreign_dir"])
def test_force_refuses_non_corpus(tmp_path: Path, dataset: Dataset, kind: str) -> None:
    target = tmp_path / "target"
    if kind == "file":
        target.write_text("important")
    else:
        target.mkdir()
        (target / "thesis.tex").write_text("important")
    with pytest.raises(ExportError) as info:
        write_corpus(dataset, target, force=True)
    assert info.value.code == "export_unsafe_path"
    assert target.exists()


def test_refuses_symlink_target(tmp_path: Path, dataset: Dataset) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(ExportError) as info:
        write_corpus(dataset, link, force=True)
    assert info.value.code == "export_unsafe_path"


def test_refuses_failed_quality_report(tmp_path: Path, dataset: Dataset) -> None:
    report = QualityReport(
        source="x.csv",
        rows_read=1,
        rows_kept=0,
        errors=("no row produced a valid event",),
    )
    with pytest.raises(ExportError) as info:
        write_corpus(dataset, tmp_path / "c", quality_report=report)
    assert info.value.code == "export_quality_failed"
    assert not (tmp_path / "c").exists()


def test_unknown_format_rejected(tmp_path: Path, dataset: Dataset) -> None:
    with pytest.raises(ExportError):
        write_corpus(dataset, tmp_path / "c", format="xml")  # type: ignore[arg-type]


def test_failed_write_cleans_up(
    tmp_path: Path, dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = Dataset(
        dataset_id="bad", events=(_event(0, "L1", None, metadata={"o": object()}),)
    )
    with pytest.raises(ExportError):
        write_corpus(bad, tmp_path / "c", format="jsonl")
    assert list(tmp_path.iterdir()) == []


def test_swap_restores_old_corpus_on_failure(
    tmp_path: Path, dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = write_corpus(dataset, tmp_path / "c")
    original_rename = Path.rename
    calls = {"n": 0}

    def flaky_rename(self: Path, dest: Any) -> Path:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated")
        return original_rename(self, dest)

    monkeypatch.setattr(Path, "rename", flaky_rename)
    with pytest.raises(OSError):
        write_corpus(dataset, target, format="jsonl", force=True)
    monkeypatch.undo()
    assert read_corpus(target) == dataset
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c"]


# --------------------------------------------------------------------------
# Reading damaged corpora
# --------------------------------------------------------------------------


def _manifest(target: Path) -> dict[str, Any]:
    return json.loads((target / "manifest.json").read_text())  # type: ignore[no-any-return]


def _save(target: Path, manifest: dict[str, Any]) -> None:
    (target / "manifest.json").write_text(json.dumps(manifest))


def test_detects_tampering(tmp_path: Path, dataset: Dataset) -> None:
    target = write_corpus(dataset, tmp_path / "c")
    events = target / "events.csv"
    events.write_text(events.read_text().replace("attempt", "submit"))
    with pytest.raises(IngestionError) as info:
        read_corpus(target)
    assert info.value.code == "io_corpus_integrity"


def test_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(IngestionError) as info:
        read_corpus(tmp_path)
    assert info.value.code == "io_not_found"


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"schema_version": 1}'])
def test_corrupt_manifest(tmp_path: Path, content: str) -> None:
    (tmp_path / "manifest.json").write_text(content)
    with pytest.raises(IngestionError) as info:
        read_corpus(tmp_path)
    assert info.value.code == "io_corpus_corrupt"


@pytest.mark.parametrize(
    "change",
    [
        {"format": "xml"},
        {"files": {"events": "../outside.csv"}},
        {"files": {"events": 5}},
        {"metadata": None},
    ],
)
def test_bad_manifest_fields(
    tmp_path: Path, dataset: Dataset, change: dict[str, Any]
) -> None:
    target = write_corpus(dataset, tmp_path / "c")
    _save(target, _manifest(target) | change)
    with pytest.raises(IngestionError):
        read_corpus(target)


def test_incompatible_schema_version(tmp_path: Path, dataset: Dataset) -> None:
    target = write_corpus(dataset, tmp_path / "c")
    _save(target, _manifest(target) | {"schema_version": "2.0"})
    with pytest.raises(SchemaError) as info:
        read_corpus(target)
    assert info.value.code == "schema_unsupported_version"


@pytest.mark.parametrize(
    ("old", "new", "field"),
    [(",0.1,", ",zero,", "score"), ('{""device""', "{bad", "metadata")],
)
def test_invalid_cells(
    tmp_path: Path, dataset: Dataset, old: str, new: str, field: str
) -> None:
    target = write_corpus(dataset, tmp_path / "c")
    events = target / "events.csv"
    text = events.read_text()
    assert old in text
    events.write_text(text.replace(old, new, 1))
    with pytest.raises(SchemaError) as info:
        read_corpus(target)
    assert info.value.context["field"] == field
