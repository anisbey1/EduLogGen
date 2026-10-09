"""Tests for annotations, their storage, and detection scoring."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

import eduloggen as elg
from eduloggen.analysis import sessionize
from eduloggen.cli import main
from eduloggen.core import IngestionError, SchemaError, ValidationError
from eduloggen.datasets import demo_dataset
from eduloggen.evaluation import average_precision, evaluate_detection, roc_auc
from eduloggen.io import read_annotations, read_corpus, write_corpus
from eduloggen.models import Annotation, Annotations, Dataset
from eduloggen.privacy import remap_ids, remap_ids_with_mapping


@pytest.fixture(scope="module")
def real() -> Dataset:
    return sessionize(demo_dataset(30, seed=2))


@pytest.fixture(scope="module")
def injected(real: Dataset) -> Any:
    return elg.inject_anomalies(
        real,
        [
            {"type": "inactivity", "rate": 0.1},
            {"type": "unexpected_transition", "rate": 0.1},
        ],
        seed=1,
    )


def _row(**overrides: Any) -> Annotation:
    values: dict[str, Any] = {
        "level": "session",
        "id": "S1",
        "annotation": "anomaly",
        "type": "inactivity",
        "category": "unusual_valid",
    }
    values.update(overrides)
    return Annotation(**values)


# --------------------------------------------------------------------------
# Annotation model
# --------------------------------------------------------------------------


def test_annotation_round_trip() -> None:
    row = _row(parameters={"gap_s": 60.0})
    assert Annotation.from_row(row.to_row()) == row
    profile = Annotation(
        level="learner", id="L1", annotation="profile", type="profile", value="steady"
    )
    assert Annotation.from_row(profile.to_row()).value == "steady"


@pytest.mark.parametrize(
    "overrides",
    [
        {"level": "course"},
        {"annotation": "guess"},
        {"id": ""},
        {"type": ""},
        {"category": None},
        {"category": "cheating"},
        {"annotation": "profile", "category": "unusual_valid"},
        {"value": ""},
        {"parameters": {"x": object()}},
        {"parameters": [1]},
    ],
)
def test_annotation_validation(overrides: dict[str, Any]) -> None:
    with pytest.raises(SchemaError):
        _row(**overrides)


def test_annotation_from_row_errors() -> None:
    with pytest.raises(SchemaError):
        Annotation.from_row({"level": "session"})
    with pytest.raises(SchemaError):
        Annotation.from_row(_row().to_row() | {"value": "{bad"})


def test_annotations_table() -> None:
    table = Annotations(
        (_row(), _row(id="S2", type="repetition"), _row(level="event", id="E1"))
    )
    assert len(table) == 3
    assert table.anomalous_ids("session") == {"S1", "S2"}
    assert table.anomalous_ids("session", type="repetition") == {"S2"}
    assert table.anomalous_ids("session", category="invalid_workflow") == frozenset()
    assert [r.id for r in table] == ["S1", "S2", "E1"]
    assert len(table + Annotations((_row(id="S3"),))) == 4
    assert table.fingerprint() == Annotations(tuple(reversed(table.rows))).fingerprint()
    assert table.summary()["anomalies"]["session"] == {"inactivity": 1, "repetition": 1}
    with pytest.raises(SchemaError):
        Annotations((_row(), _row()))
    with pytest.raises(SchemaError):
        Annotations(("x",))  # type: ignore[arg-type]


def test_annotations_remap_and_check(real: Dataset) -> None:
    table = Annotations((_row(id="old"),))
    assert table.remap({"session": {"old": "new"}}).anomalous_ids("session") == {"new"}
    with pytest.raises(SchemaError):
        table.remap({"session": {}})
    with pytest.raises(SchemaError):
        table.check_against(real)


def test_remap_ids_with_mapping(real: Dataset) -> None:
    remapped, mapping = remap_ids_with_mapping(real, seed=4)
    assert remapped.fingerprint() == remap_ids(real, seed=4).fingerprint()
    assert set(mapping.learners) == set(real.learner_ids)
    assert set(mapping.events.values()) == {e.event_id for e in remapped.events}
    old = real.events[0]
    new = next(e for e in remapped.events if e.event_id == mapping.events[old.event_id])
    assert (new.timestamp, new.activity_id) == (old.timestamp, old.activity_id)
    assert new.learner_id == mapping.learners[old.learner_id]


# --------------------------------------------------------------------------
# Corpus storage
# --------------------------------------------------------------------------


def test_corpus_round_trip(tmp_path: Path, injected: Any) -> None:
    target = write_corpus(
        injected.dataset, tmp_path / "c", annotations=injected.annotations
    )
    assert (target / "annotations.csv").exists()
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["files"]["annotations"] == "annotations.csv"
    assert manifest["annotations"]["fingerprint"] == injected.annotations.fingerprint()
    loaded = read_annotations(target)
    assert loaded is not None
    assert loaded.fingerprint() == injected.annotations.fingerprint()
    assert read_corpus(target).fingerprint() == injected.dataset.fingerprint()


def test_corpus_without_annotations(tmp_path: Path, real: Dataset) -> None:
    assert read_annotations(write_corpus(real, tmp_path / "c")) is None


def test_annotation_integrity(tmp_path: Path, injected: Any) -> None:
    target = write_corpus(
        injected.dataset, tmp_path / "c", annotations=injected.annotations
    )
    path = target / "annotations.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["type"] = "repetition" if rows[0]["type"] != "repetition" else "inactivity"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(IngestionError) as info:
        read_annotations(target)
    assert info.value.code == "io_corpus_integrity"
    path.unlink()
    with pytest.raises(IngestionError) as info:
        read_annotations(target)
    assert info.value.code == "io_not_found"


def test_annotations_must_match_dataset(tmp_path: Path, real: Dataset) -> None:
    with pytest.raises(SchemaError):
        write_corpus(real, tmp_path / "c", annotations=Annotations((_row(id="S404"),)))
    assert not (tmp_path / "c").exists()


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_perfect_and_empty_detectors(injected: Any) -> None:
    truth = injected.annotations.anomalous_ids("session")
    perfect = evaluate_detection(injected.dataset, injected.annotations, truth)
    assert (perfect.precision, perfect.recall, perfect.f1) == (1.0, 1.0, 1.0)
    assert perfect.false_positive_rate == 0.0
    assert perfect.recall_by_category == {"invalid_workflow": 1.0, "unusual_valid": 1.0}
    assert perfect.roc_auc is None

    nothing = evaluate_detection(injected.dataset, injected.annotations, [])
    assert nothing.precision is None
    assert nothing.recall == 0.0
    assert nothing.tn == nothing.n_items - nothing.n_anomalous


def test_scores_and_threshold(injected: Any) -> None:
    truth = injected.annotations.anomalous_ids("session")
    sessions = [s.session_id for s in injected.dataset.sessions or ()]
    inactivity = injected.annotations.anomalous_ids("session", type="inactivity")
    scores = {sid: (0.9 if sid in inactivity else 0.1) for sid in sessions}
    report = evaluate_detection(
        injected.dataset, injected.annotations, scores, threshold=0.5
    )
    assert report.recall_by_type == {"inactivity": 1.0, "unexpected_transition": 0.0}
    assert report.recall_by_category == {"invalid_workflow": 0.0, "unusual_valid": 1.0}
    assert report.precision == 1.0
    assert report.threshold == 0.5
    assert 0.5 < (report.roc_auc or 0) < 1
    assert report.n_anomalous == len(truth)
    ranked = evaluate_detection(
        injected.dataset, injected.annotations, {s: float(s in truth) for s in sessions}
    )
    assert ranked.roc_auc == 1.0
    assert ranked.average_precision == 1.0


def test_event_level(injected: Any) -> None:
    events = injected.annotations.anomalous_ids("event")
    report = evaluate_detection(
        injected.dataset, injected.annotations, events, level="event"
    )
    assert report.recall == 1.0
    assert report.n_items == injected.dataset.n_events


def test_scoring_errors(injected: Any) -> None:
    with pytest.raises(ValidationError) as info:
        evaluate_detection(injected.dataset, injected.annotations, ["nope"])
    assert info.value.code == "evaluation_unknown_id"
    sid = injected.dataset.sessions[0].session_id
    with pytest.raises(ValidationError):
        evaluate_detection(injected.dataset, injected.annotations, {sid: float("nan")})
    with pytest.raises(ValidationError):
        evaluate_detection(injected.dataset, injected.annotations, {sid: "high"})


def test_rank_metrics() -> None:
    assert roc_auc([(0.9, True), (0.1, False)]) == 1.0
    assert roc_auc([(0.5, True), (0.5, False)]) == 0.5
    assert roc_auc([(0.1, True), (0.9, False)]) == 0.0
    assert roc_auc([(0.1, True)]) is None
    assert average_precision([(0.9, True), (0.8, False), (0.7, True)]) == pytest.approx(
        (1 + 2 / 3) / 2
    )
    assert average_precision([(0.5, False)]) is None
    assert average_precision([(0.5, True), (0.5, False)]) == 0.5


def test_report_output(injected: Any) -> None:
    report = evaluate_detection(
        injected.dataset,
        injected.annotations,
        injected.annotations.anomalous_ids("session"),
    )
    data = json.loads(json.dumps(report.to_dict()))
    assert data["confusion"]["fp"] == 0
    text = report.to_markdown()
    assert "| Recall | 1.000 |" in text
    assert "| invalid_workflow | 1.000 |" in text


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_generate_with_anomalies_and_evaluate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "a.yaml").write_text(
        "anomalies:\n"
        "  - {type: inactivity, rate: 0.2}\n"
        "  - {type: repetition, rate: 0.1}\n"
    )
    commands = [
        "demo --output {d}/demo --learners 20",
        "fit --input {d}/demo --output {d}/m",
        "generate --model {d}/m --n-sessions 50 --seed 3 "
        "--anomalies {d}/a.yaml --output {d}/syn",
    ]
    for command in commands:
        assert main([*command.format(d=tmp_path).split(), "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "injected inactivity (unusual_valid): 10 of 10 sessions" in out
    annotations = read_annotations(tmp_path / "syn")
    assert annotations is not None

    truth = annotations.anomalous_ids("session")
    with (tmp_path / "flags.csv").open("w") as handle:
        handle.write("id,flag\n" + "".join(f"{sid},1\n" for sid in sorted(truth)))
    command = "evaluate --corpus {d}/syn --predictions {d}/flags.csv --output {d}/ev"
    code = main([*command.format(d=tmp_path).split(), "--quiet"])
    assert code == 0
    assert "| Recall | 1.000 |" in capsys.readouterr().out
    assert json.loads((tmp_path / "ev" / "evaluation.json").read_text())["f1"] == 1.0

    with (tmp_path / "ids.csv").open("w") as handle:
        handle.write("id\n" + "".join(f"{sid}\n" for sid in sorted(truth)))
    with (tmp_path / "scores.csv").open("w") as handle:
        handle.write("id,score\n" + "".join(f"{sid},0.9\n" for sid in sorted(truth)))
    for name in ("ids.csv", "scores.csv"):
        assert (
            main(
                [
                    "evaluate",
                    "--corpus",
                    str(tmp_path / "syn"),
                    "--predictions",
                    str(tmp_path / name),
                    "--quiet",
                ]
            )
            == 0
        )
    capsys.readouterr()


def test_cli_evaluate_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["demo", "--output", str(tmp_path / "demo"), "--learners", "5", "--quiet"])
    (tmp_path / "p.csv").write_text("id\nS1\n")
    assert (
        main(
            [
                "evaluate",
                "--corpus",
                str(tmp_path / "demo"),
                "--predictions",
                str(tmp_path / "p.csv"),
                "--quiet",
            ]
        )
        == 2
    )
    assert "cli_missing_annotations" in capsys.readouterr().err

    real = sessionize(demo_dataset(10))
    result = elg.inject_anomalies(real, [{"type": "inactivity", "rate": 0.5}], seed=0)
    write_corpus(result.dataset, tmp_path / "c", annotations=result.annotations)
    for content, code in (
        ("session\nS1\n", "'id' column"),
        ("id,score\nS1,high\n", "must be numbers"),
    ):
        (tmp_path / "bad.csv").write_text(content)
        assert (
            main(
                [
                    "evaluate",
                    "--corpus",
                    str(tmp_path / "c"),
                    "--predictions",
                    str(tmp_path / "bad.csv"),
                    "--quiet",
                ]
            )
            == 2
        )
        assert code in capsys.readouterr().err
    assert (
        main(
            [
                "evaluate",
                "--corpus",
                str(tmp_path / "c"),
                "--predictions",
                str(tmp_path / "missing.csv"),
                "--quiet",
            ]
        )
        == 2
    )
    capsys.readouterr()
