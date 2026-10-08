"""End-to-end tests for ``ingest`` and the quality report."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from eduloggen.core import IngestionError, SchemaError
from eduloggen.io import FieldMapping, get_writer, ingest
from eduloggen.io.quality import MAX_ISSUE_SAMPLES
from eduloggen.models import EventVocabulary, UnknownEventPolicy

SECRET = "alice@example.org"

MAPPING = FieldMapping.from_dict(
    {
        "fields": {
            "event_id": "id",
            "learner_id": {"source": "user", "hash": {"salt": "pepper"}},
            "timestamp": "time",
            "activity_id": "item",
            "event_type": "action",
            "score": "grade",
        },
        "metadata": ["device"],
        "drop": ["email"],
    }
)

MESSY_CSV = f"""id,user,time,item,action,grade,device,email,notes
1,{SECRET},2026-03-01T09:10:00Z,quiz,submit,0.5,mobile,{SECRET},a
2,{SECRET},2026-03-01T09:00:00Z,quiz,view,,mobile,{SECRET},b
3,bob,2026-03-01T09:05:00Z,video,video_play,,desktop,x,c
2,{SECRET},2026-03-01T09:20:00Z,quiz,view,,mobile,{SECRET},dup
4,,2026-03-01T09:30:00Z,quiz,view,,,,missing user
5,bob,2026-03-01 09:40:00,quiz,view,,,,naive
6,bob,2026-03-01T09:50:00Z,quiz,hint,{SECRET},,,bad grade
"""


@pytest.fixture
def messy(tmp_path: Path) -> Path:
    path = tmp_path / "course_log.csv"
    path.write_text(MESSY_CSV)
    return path


def test_ingest_messy_csv(messy: Path) -> None:
    result = ingest(messy, MAPPING)
    dataset, report = result.dataset, result.report

    assert dataset.dataset_id == "course_log"
    assert sorted(e.event_id for e in dataset.events) == ["1", "2", "3"]
    assert dataset.events == dataset.sorted_events()
    assert all(SECRET not in e.learner_id for e in dataset.events)
    by_id = {e.event_id: e for e in dataset.events}
    assert by_id["2"].timestamp < by_id["1"].timestamp
    assert dict(by_id["3"].metadata) == {"device": "desktop"}
    assert by_id["1"].score == 0.5
    assert dataset.metadata.source_description == "ingested from course_log.csv"
    assert result.mapping is MAPPING

    assert report.status == "ready"
    assert report.rows_read == 7
    assert report.rows_kept == 3
    assert report.rows_dropped == 4
    assert report.duplicate_events == 1
    assert report.out_of_order == 1
    assert report.issue_counts == {
        "missing_required": 1,
        "naive_timestamp": 1,
        "not_a_number": 1,
    }
    assert report.unmapped_columns == ("notes",)
    assert report.field_coverage["score"] == pytest.approx(1 / 3)
    assert report.field_coverage["learner_id"] == 1.0
    assert len(report.warnings) == 3


def test_quality_report_never_contains_source_values(messy: Path) -> None:
    report = ingest(messy, MAPPING).report.to_dict()
    text = json.dumps(report)
    assert SECRET not in text
    assert "bob" not in text
    assert report["summary"] == {"rows_read": 7, "rows_kept": 3, "rows_dropped": 4}
    assert report["row_issues"]["samples"][0] == {
        "row": 5,
        "field": "learner_id",
        "code": "missing_required",
    }


def test_strict_mode_raises_on_first_bad_row(messy: Path) -> None:
    with pytest.raises(SchemaError) as info:
        ingest(messy, MAPPING, strict=True)
    assert info.value.code == "ingest_row_invalid"
    assert info.value.context == {
        "row": 5,
        "field": "learner_id",
        "code": "missing_required",
    }


def test_duplicate_policy_error(messy: Path) -> None:
    with pytest.raises(SchemaError) as info:
        ingest(messy, MAPPING, on_duplicate="error")
    assert info.value.code == "ingest_duplicate_event"
    assert info.value.context == {"row": 4}


def test_vocabulary_policies(messy: Path) -> None:
    mapped = ingest(
        messy,
        MAPPING,
        vocabulary=EventVocabulary(unknown_policy=UnknownEventPolicy.MAP_TO_OTHER),
    )
    assert {e.event_type for e in mapped.dataset.events} == {
        "submit",
        "view",
        "video_play",
    }

    rejected = ingest(
        messy,
        MAPPING,
        vocabulary=EventVocabulary(
            types=frozenset({"view"}), unknown_policy=UnknownEventPolicy.REJECT
        ),
    )
    assert {e.event_type for e in rejected.dataset.events} == {"view"}
    assert rejected.report.issue_counts["schema_invalid_value"] == 2


def test_missing_required_column_fails_fast(tmp_path: Path) -> None:
    path = tmp_path / "a.csv"
    path.write_text("id,user,item\n1,u,q\n")
    with pytest.raises(SchemaError) as info:
        ingest(path, MAPPING)
    assert info.value.code == "ingest_missing_columns"
    assert info.value.context["columns"] == ["action", "time"]


def test_empty_and_all_invalid_sources_fail(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("id,user,time,item,action\n")
    report = ingest(empty, MAPPING).report
    assert report.status == "failed"
    assert report.errors == ("source contains no rows",)

    bad = tmp_path / "bad.csv"
    bad.write_text("id,user,time,item,action\n1,,x,q,view\n")
    report = ingest(bad, MAPPING).report
    assert report.status == "failed"
    assert report.errors == ("no row produced a valid event",)
    assert report.field_coverage["event_id"] == 0.0


def test_issue_samples_are_bounded(tmp_path: Path) -> None:
    path = tmp_path / "many.csv"
    lines = ["id,user,time,item,action"]
    lines += [
        f"{i},,2026-03-01T09:00:00Z,q,view" for i in range(MAX_ISSUE_SAMPLES + 20)
    ]
    lines.append("x,u,2026-03-01T09:00:00Z,q,view")
    path.write_text("\n".join(lines) + "\n")
    report = ingest(path, MAPPING).report
    assert len(report.issue_samples) == MAX_ISSUE_SAMPLES
    assert report.issue_counts["missing_required"] == MAX_ISSUE_SAMPLES + 20


def _typed_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": i,
            "user": "u1",
            "time": datetime(2026, 3, 1, 9, i, tzinfo=UTC),
            "item": "q",
            "action": "view",
            "grade": 0.5,
            "extra_col": "x",
        }
        for i in range(3)
    ]


def test_ingest_jsonl_tracks_columns_across_rows(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    rows = _typed_rows()
    rows[2]["late_col"] = 1
    path.write_text(
        "\n".join(json.dumps(r | {"time": r["time"].isoformat()}) for r in rows) + "\n"
    )
    result = ingest(path, MAPPING)
    assert result.dataset.n_events == 3
    assert result.report.unmapped_columns == ("extra_col", "late_col")


def test_ingest_parquet(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "events.parquet"
    columns: list[Any] = [
        ("id", "int"),
        ("user", "string"),
        ("time", "timestamp"),
        ("item", "string"),
        ("action", "string"),
        ("grade", "float"),
    ]
    get_writer("parquet").write(path, _typed_rows(), columns)
    result = ingest(path, MAPPING, dataset_id="pq", privacy_notes="synthetic")
    assert result.dataset.dataset_id == "pq"
    assert result.dataset.metadata.privacy_notes == "synthetic"
    assert [e.event_id for e in result.dataset.events] == ["0", "1", "2"]
    assert result.dataset.events[0].score == 0.5


def test_ingest_missing_file(tmp_path: Path) -> None:
    with pytest.raises(IngestionError):
        ingest(tmp_path / "nope.csv", MAPPING)
