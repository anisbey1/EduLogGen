"""Tests for behavioural profiles: auto, provided, manual, mixtures, save/load."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

import eduloggen as elg
from eduloggen.analysis import session_sequences, sessionize
from eduloggen.benchmark import split_by_learner
from eduloggen.core import ConfigError, FitError, IngestionError
from eduloggen.datasets import demo_dataset
from eduloggen.evaluation import evaluate_clustering
from eduloggen.generators import SessionCalendar
from eduloggen.models import Dataset
from eduloggen.scenarios import (
    Controls,
    ProfileSet,
    define_profiles,
    fit_profiles,
    generate_profiles,
    load_profile_assignments,
    load_profiles,
)
from eduloggen.scenarios.clustering import (
    Standardizer,
    feature_means,
    kmeans,
    learner_features,
)
from eduloggen.utils import make_rng

DEFINITIONS: dict[str, Any] = {
    "steady": {
        "share": 2,
        "start": {"view": 1},
        "transitions": {
            "view": {"attempt": 0.8, "view": 0.2},
            "attempt": {"submit": 1},
            "submit": {"view": 1},
        },
        "session_length": {"mean": 5},
        "sessions_per_learner": {"mean": 4},
        "dwell_s": {"*": 30, "view": 120},
        "activities": {"view": ["m1-reading", "m2-reading"]},
        "course": "stats",
    },
    "crammer": {
        "start": {"attempt": 1},
        "transitions": {
            "attempt": {"attempt": 0.7, "submit": 0.3},
            "submit": {"attempt": 1},
        },
        "session_length": {"fixed": 10},
        "sessions_per_learner": {"fixed": 1},
        "dwell_s": {"*": 5},
    },
}


@pytest.fixture(scope="module")
def real() -> Dataset:
    return sessionize(demo_dataset(60, seed=1))


@pytest.fixture(scope="module")
def auto(real: Dataset) -> ProfileSet:
    return fit_profiles(real, n_profiles=3, min_learners=3, seed=0)


# --------------------------------------------------------------------------
# clustering primitives
# --------------------------------------------------------------------------


def test_learner_features(real: Dataset) -> None:
    names, vectors = learner_features(real)
    assert names[:4] == [
        "n_sessions",
        "mean_session_length",
        "log_mean_duration_s",
        "success_rate",
    ]
    assert all(len(v) == len(names) for v in vectors.values())
    shares = [i for i, n in enumerate(names) if n.startswith("share:")]
    assert all(sum(v[i] for i in shares) == pytest.approx(1) for v in vectors.values())
    _, fixed = learner_features(real, ["view"])
    assert len(next(iter(fixed.values()))) == 5


def test_standardizer_and_means() -> None:
    scaler = Standardizer.fit([[1.0, 5.0], [3.0, 5.0]])
    assert scaler.means == (2.0, 5.0)
    assert scaler.stds == (1.0, 1.0)  # constant feature keeps std 1
    assert scaler.transform([3.0, 6.0]) == [1.0, 1.0]
    assert scaler.to_dict() == {"means": [2.0, 5.0], "stds": [1.0, 1.0], "clip": 3.0}
    assert scaler.transform([20.0, -20.0]) == [3.0, -3.0]
    assert Standardizer.fit([[1.0], [3.0]], clip=None).transform([20.0]) == [18.0]
    assert feature_means({"a": [1.0], "b": [3.0]}, ["a", "b"]) == [2.0]


def test_kmeans_separates_blobs() -> None:
    rows = [[0.0, 0.1 * i] for i in range(5)] + [[10.0, 0.1 * i] for i in range(5)]
    result = kmeans(rows, 2, rng=make_rng(0, "t"))
    assert len(set(result.labels[:5])) == 1
    assert len(set(result.labels[5:])) == 1
    assert result.labels[0] != result.labels[5]
    assert result.inertia == pytest.approx(2 * 0.1, abs=0.01)
    same = kmeans(rows, 2, rng=make_rng(0, "t"))
    assert same == result
    duplicates = kmeans([[1.0, 1.0]] * 4, 2, rng=make_rng(0, "t"))
    assert duplicates.inertia == 0
    for k in (0, 11, True):
        with pytest.raises(ConfigError):
            kmeans(rows, k, rng=make_rng(0, "t"))


def test_kmeans_empty_cluster_is_moved() -> None:
    from eduloggen.scenarios.clustering import _update

    rows = [[0.0], [1.0], [9.0]]
    updated = _update(rows, [0, 0, 0], [[0.0], [50.0]])
    assert updated[0] == [10 / 3]
    assert updated[1] == [9.0]


# --------------------------------------------------------------------------
# auto / provided
# --------------------------------------------------------------------------


def test_auto_profiles(auto: ProfileSet, real: Dataset) -> None:
    assert auto.mode == "auto"
    assert auto.names == ["profile_1", "profile_2", "profile_3"]
    sizes = [p.n_learners for p in auto.profiles]
    assert sizes == sorted(sizes, reverse=True)
    assert sum(sizes) == len(real.learner_ids)
    assert sum(p.share for p in auto.profiles) == pytest.approx(1)
    assert auto.clustering is not None
    assert len(auto.clustering["centroids"]) == 3
    first = auto.profiles[0].description
    assert {"mean_sessions", "token_shares", "distinguishing"} <= set(first)
    assert auto.profile("profile_2").name == "profile_2"
    with pytest.raises(KeyError):
        auto.profile("nobody")
    again = fit_profiles(real, n_profiles=3, min_learners=3, seed=0)
    assert again.fingerprint() == auto.fingerprint()


def test_assign_uses_training_profiles(real: Dataset) -> None:
    train, test = split_by_learner(real, 0.3, seed=0)
    profiles = fit_profiles(train, n_profiles=2, min_learners=3, seed=0)
    before = profiles.fingerprint()
    holdout = profiles.assign(test)
    assert set(holdout) == {e.learner_id for e in test.events}
    assert set(holdout.values()) <= set(profiles.names)
    assert profiles.fingerprint() == before
    training = profiles.assign(train)
    groups = Counter(training.values())
    assert sorted(groups.values()) == sorted(p.n_learners for p in profiles.profiles)


def test_names_and_min_learners(real: Dataset) -> None:
    named = fit_profiles(
        real, n_profiles=2, min_learners=3, names={"profile_1": "steady"}
    )
    assert named.names[0] == "steady"
    with pytest.raises(ConfigError, match="unknown profiles"):
        fit_profiles(real, n_profiles=2, names={"profile_9": "x"})
    with pytest.raises(ConfigError, match="unique"):
        fit_profiles(real, n_profiles=2, names={"profile_1": "a", "profile_2": "a"})
    with pytest.raises(ConfigError, match="fewer than 30"):
        fit_profiles(real, n_profiles=3, min_learners=30)
    with pytest.raises(ConfigError):
        fit_profiles(real, min_learners=0)
    with pytest.raises(ConfigError):
        fit_profiles(real, mode="clusters")  # type: ignore[arg-type]


def test_provided_profiles(real: Dataset) -> None:
    learners = sorted({e.learner_id for e in real.events})
    mapping = {
        learner: ("early" if i < 30 else "late") for i, learner in enumerate(learners)
    }
    profiles = fit_profiles(
        real, mode="provided", assignments=mapping, generator="markov"
    )
    assert profiles.names == ["early", "late"]
    assert profiles.generator_id == "markov"
    assert profiles.clustering is None
    with pytest.raises(ConfigError, match="only auto"):
        profiles.assign(real)
    partial = fit_profiles(
        real, mode="provided", assignments=dict(list(mapping.items())[:10])
    )
    assert partial.profiles[0].n_learners == 10
    with pytest.raises(ConfigError):
        fit_profiles(real, mode="provided")
    with pytest.raises(ConfigError, match="no learner"):
        fit_profiles(real, mode="provided", assignments={"nobody": "x"})


def test_fit_errors() -> None:
    empty = sessionize(Dataset(dataset_id="e", events=()))
    with pytest.raises(FitError):
        fit_profiles(empty)


def test_profile_model_fit_error_names_profile(
    real: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    from eduloggen.generators import SemiMarkovGenerator

    def fail(*args: object, **kwargs: object) -> None:
        raise FitError("too little data", code="fit_insufficient_data")

    monkeypatch.setattr(SemiMarkovGenerator, "fit", fail)
    with pytest.raises(FitError, match="profile 'profile_1'"):
        fit_profiles(real, n_profiles=1)


# --------------------------------------------------------------------------
# manual
# --------------------------------------------------------------------------


def test_manual_profiles_follow_definitions() -> None:
    profiles = define_profiles(DEFINITIONS)
    assert profiles.mode == "manual"
    assert [p.share for p in profiles.profiles] == pytest.approx([2 / 3, 1 / 3])
    result = generate_profiles(profiles, 400, seed=2, id_strategy="preserve")
    by_profile: dict[str, list[tuple[str, ...]]] = {}
    learner_profile = {a.id: str(a.value) for a in result.annotations.rows}
    for session in result.dataset.sessions or ():
        by_profile.setdefault(learner_profile[session.learner_id], []).append(
            session.event_sequence
        )
    crammer = by_profile["crammer"]
    assert {len(s) for s in crammer} == {10}
    assert all(s[0] == "attempt" for s in crammer)
    assert "view" not in {t for s in crammer for t in s}
    steady = by_profile["steady"]
    assert all(s[0] == "view" for s in steady)
    mean = sum(len(s) for s in steady) / len(steady)
    assert 4 <= mean <= 6
    learners = Counter(learner_profile.values())
    assert learners["steady"] / sum(learners.values()) == pytest.approx(2 / 3, abs=0.06)
    activities = {
        e.activity_id for e in result.dataset.events if e.event_type == "view"
    }
    assert activities == {"m1-reading", "m2-reading"}
    assert {e.course_id for e in result.dataset.events} == {"stats", None}


def test_manual_dwell_times() -> None:
    result = generate_profiles(
        define_profiles({"crammer": DEFINITIONS["crammer"]}), 50, seed=0
    )
    events = sorted(result.dataset.events, key=lambda e: (e.session_id, e.timestamp))
    gaps = [
        (b.timestamp - a.timestamp).total_seconds()
        for a, b in pairwise(events)
        if a.session_id == b.session_id
    ]
    gaps.sort()
    assert 3 <= gaps[len(gaps) // 2] <= 8


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"start": None}, "behavioural parameters"),
        ({"colour": "red"}, "unknown keys"),
        ({"share": -1}, "share"),
        ({"start": {"view": -1}}, "non-negative"),
        ({"start": {"<start>": 1}}, "reserved"),
        ({"start": []}, "start must map"),
        ({"transitions": {}}, "transitions must map"),
        ({"session_length": {"mean": 0}}, "at least 1"),
        ({"session_length": {"median": 3}}, "session_length must be"),
        ({"session_length": {"fixed": 1.5}}, "positive integer"),
        ({"sessions_per_learner": {"counts": {"0": 1}}}, "positive sizes"),
        ({"sessions_per_learner": {"counts": {}}}, "map sizes"),
        ({"dwell_s": {"view": 0}}, "dwell_s"),
        ({"dwell_s": {"quiz": 3}}, "unknown tokens"),
        ({"dwell_spread": 0}, "dwell_spread"),
        ({"activities": ["a"]}, "activities must map"),
        ({"activities": {"view": "m1"}}, "non-empty list"),
    ],
)
def test_manual_validation(change: dict[str, Any], match: str) -> None:
    spec = {**DEFINITIONS["steady"], **change}
    spec = {k: v for k, v in spec.items() if v is not None}
    with pytest.raises(ConfigError, match=match):
        define_profiles({"steady": spec})


def test_manual_validation_of_the_set() -> None:
    with pytest.raises(ConfigError):
        define_profiles({})
    with pytest.raises(ConfigError):
        define_profiles({"": DEFINITIONS["steady"]})
    with pytest.raises(ConfigError, match="must be a mapping"):
        define_profiles({"a": [1]})  # type: ignore[dict-item]
    with pytest.raises(ConfigError, match="all be zero"):
        define_profiles({"a": {**DEFINITIONS["steady"], "share": 0}})


def test_manual_counts_forms() -> None:
    spec = {
        **DEFINITIONS["steady"],
        "session_length": {"counts": {"2": 1, "3": 1}},
        "sessions_per_learner": {"counts": {"1": 1, "2": 3}},
    }
    result = generate_profiles(define_profiles({"a": spec}), 100, seed=0)
    assert {len(s) for s in session_sequences(result.dataset)} == {2, 3}
    integer = {**DEFINITIONS["steady"], "sessions_per_learner": {"mean": 2}}
    model = define_profiles({"a": integer}).profiles[0].model
    assert model.parameters["population"]["sessions_per_learner"] == {"2": 1.0}


# --------------------------------------------------------------------------
# mixtures
# --------------------------------------------------------------------------


def test_mixture_from_auto(auto: ProfileSet) -> None:
    result = generate_profiles(auto, 300, seed=1)
    assert result.dataset.n_sessions == 300
    assert sum(a["sessions"] for a in result.allocation.values()) == 300
    ids = {e.learner_id for e in result.dataset.events}
    rows = result.annotations.rows
    assert {r.id for r in rows} == ids
    assert all(r.level == "learner" and r.type == "profile" for r in rows)
    assert not any(i.startswith("P1-") for i in ids)  # remapped
    assert result.dataset.generation is not None
    assert result.dataset.generation.model_fingerprint == auto.fingerprint()
    again = generate_profiles(auto, 300, seed=1)
    assert again.dataset.events == result.dataset.events


def test_custom_mixture_and_controls(auto: ProfileSet) -> None:
    result = generate_profiles(
        auto,
        200,
        seed=3,
        mixture={"profile_2": 1},
        controls={"profile_2": Controls(session_length={"fixed": 4})},
    )
    assert result.allocation["profile_1"]["sessions"] == 0
    assert {str(a.value) for a in result.annotations.rows} == {"profile_2"}
    assert {len(s) for s in session_sequences(result.dataset)} == {4}


def test_mixture_errors(auto: ProfileSet) -> None:
    with pytest.raises(ConfigError, match="unknown profiles"):
        generate_profiles(auto, 10, seed=0, mixture={"x": 1})
    with pytest.raises(ConfigError, match="unknown profiles"):
        generate_profiles(auto, 10, seed=0, controls={"x": Controls()})
    with pytest.raises(ConfigError):
        generate_profiles(auto, 10, seed=0, mixture={"profile_1": 0})
    with pytest.raises(ConfigError):
        generate_profiles(auto, 0, seed=0)
    with pytest.raises(ConfigError):
        generate_profiles(auto, 10, seed=0, id_strategy="hash")  # type: ignore[arg-type]
    manual = define_profiles({"m": DEFINITIONS["steady"]})
    mixed = ProfileSet("manual", "x", (*manual.profiles, *auto.profiles))
    tokenized = mixed.profiles[1].model
    other = type(tokenized)(
        **{
            **{f: getattr(tokenized, f) for f in tokenized.__dataclass_fields__},
            "parameters": {
                **tokenized.parameters,
                "population": {
                    **tokenized.parameters["population"],
                    "tokenization": "event_type+activity",
                },
            },
        }
    )
    bad = ProfileSet(
        "manual",
        "x",
        (
            manual.profiles[0],
            type(auto.profiles[0])(name="o", model=other, share=0.5, n_learners=0),
        ),
    )
    with pytest.raises(ConfigError, match="tokenizations"):
        generate_profiles(bad, 10, seed=0)


def test_preserve_warns_once(
    auto: ProfileSet, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="eduloggen"):
        result = generate_profiles(auto, 50, seed=0, id_strategy="preserve")
    messages = [r.getMessage() for r in caplog.records if "preserve" in r.getMessage()]
    assert len(messages) == 1
    assert all(e.learner_id.startswith("P") for e in result.dataset.events)


def test_mixture_with_calendar() -> None:
    calendar = SessionCalendar(
        start=date(2026, 3, 2),
        days=7,
        timezone="UTC",
        hours=(0.0,) * 9 + (1.0,) + (0.0,) * 14,
    )
    result = generate_profiles(
        define_profiles(DEFINITIONS), 40, seed=0, calendar=calendar
    )
    starts = {s.start_time.hour for s in result.dataset.sessions or ()}
    assert starts == {9}


def test_profile_recovery_is_measurable(auto: ProfileSet) -> None:
    result = generate_profiles(define_profiles(DEFINITIONS), 400, seed=5)
    recovered = fit_profiles(result.dataset, n_profiles=2, seed=0)
    report = evaluate_clustering(result.annotations, recovered.assign(result.dataset))
    assert report.ari > 0.9
    assert report.purity > 0.95


# --------------------------------------------------------------------------
# persistence and reporting
# --------------------------------------------------------------------------


def test_save_and_load(auto: ProfileSet, tmp_path: Path) -> None:
    target = auto.save(tmp_path / "profiles")
    assert {p.name for p in target.iterdir()} == {
        "profiles.json",
        "PROFILES.md",
        "models",
    }
    loaded = load_profiles(target)
    assert loaded.fingerprint() == auto.fingerprint()
    assert loaded.names == auto.names
    a = generate_profiles(auto, 60, seed=4)
    b = generate_profiles(loaded, 60, seed=4)
    assert a.dataset.events == b.dataset.events
    manual = define_profiles(DEFINITIONS)
    assert (
        load_profiles(manual.save(tmp_path / "manual")).fingerprint()
        == manual.fingerprint()
    )


def test_load_errors(auto: ProfileSet, tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="not found"):
        load_profiles(tmp_path / "none")
    target = auto.save(tmp_path / "p")
    data = json.loads((target / "profiles.json").read_text())
    data["profiles"][0]["share"] = 0.99
    (target / "profiles.json").write_text(json.dumps(data))
    with pytest.raises(IngestionError, match="fingerprint"):
        load_profiles(target)
    (target / "profiles.json").write_text("{")
    with pytest.raises(IngestionError, match="readable"):
        load_profiles(target)


def test_markdown_and_dict(auto: ProfileSet) -> None:
    text = auto.to_markdown()
    assert text.startswith("# Profiles (auto, semi_markov)")
    assert "not validated psychological" in text
    assert "| profile_1 |" in text
    manual = define_profiles(DEFINITIONS).to_markdown()
    assert "| steady | 67% | 0 | - | - | - |" in manual
    data = json.loads(json.dumps(auto.to_dict()))
    assert data["note"].startswith("Profiles are descriptive")


def test_top_level_api() -> None:
    profiles = elg.define_profiles(DEFINITIONS)
    result = elg.generate_profiles(profiles, 30, seed=0)
    assert (
        elg.evaluate_clustering(
            result.annotations, {r.id: str(r.value) for r in result.annotations.rows}
        ).ari
        == 1.0
    )
    assert callable(elg.fit_profiles)


def test_load_profile_assignments(tmp_path: Path) -> None:
    good = tmp_path / "a.csv"
    good.write_text("learner_id,profile,note\nl1,early,x\nl2,late,\nl1,early,\n,x,\n")
    assert load_profile_assignments(good) == {"l1": "early", "l2": "late"}
    (tmp_path / "b.csv").write_text("learner_id,group\nl1,x\n")
    with pytest.raises(IngestionError, match="columns"):
        load_profile_assignments(tmp_path / "b.csv")
    (tmp_path / "c.csv").write_text("learner_id,profile\nl1,x\nl1,y\n")
    with pytest.raises(IngestionError, match="two profiles"):
        load_profile_assignments(tmp_path / "c.csv")
    with pytest.raises(IngestionError, match="not found"):
        load_profile_assignments(tmp_path / "none.csv")


def test_rare_tokens_are_pooled() -> None:
    from eduloggen.scenarios.clustering import OTHER

    from ..generators.conftest import build

    data = build([("a", "b")] * 60 + [("a", "rare")])
    names, vectors = learner_features(data)
    assert names[4:] == ["share:a", "share:b", f"share:{OTHER}"]
    assert all(sum(v[4:]) == pytest.approx(1) for v in vectors.values())
    names, _ = learner_features(data, min_share=0.0)
    assert "share:rare" in names
    names, vectors = learner_features(data, ["a", OTHER])
    assert names[4:] == ["share:a", f"share:{OTHER}"]
    assert all(v[5] == pytest.approx(0.5) for v in vectors.values())


def test_profiles_saved_without_clip_assign_unclipped(
    auto: ProfileSet, real: Dataset
) -> None:
    assert auto.clustering is not None
    assert auto.clustering["clip"] == 3.0
    legacy_clustering = {k: v for k, v in auto.clustering.items() if k != "clip"}
    legacy = ProfileSet("auto", auto.generator_id, auto.profiles, legacy_clustering)
    assert set(legacy.assign(real).values()) <= set(auto.names)


def test_profiles_back_off_on_short_sessions() -> None:
    from ..generators.conftest import build

    # learner 0 has three-event sessions, learner 1 only one-event sessions
    data = build([("a", "b", "a"), ("a",)] * 10, learners=2)
    first, second = sorted(data.learner_ids)
    mapping = {first: "many", second: "one"}
    profiles = fit_profiles(data, mode="provided", assignments=mapping, min_learners=1)
    assert profiles.names == ["many", "one"]
    sessions = generate_profiles(profiles, 20, seed=0).dataset.sessions or ()
    assert {len(s.event_sequence) for s in sessions} == {1, 3}
    with pytest.raises(FitError, match="profile 'one'"):
        fit_profiles(
            data,
            mode="provided",
            assignments=mapping,
            min_learners=1,
            hyperparameters={"on_insufficient_data": "fail"},
        )
