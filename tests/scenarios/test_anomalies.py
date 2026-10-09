"""Tests for anomaly injection (Level 2, M1)."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import datetime
from itertools import pairwise
from typing import Any, ClassVar

import pytest

from eduloggen.analysis import ngram_counts, session_sequences, sessionize
from eduloggen.core import AnalysisError, ConfigError, PluginError
from eduloggen.datasets import demo_dataset
from eduloggen.generators import get_generator
from eduloggen.models import Dataset, LogRecord, SyntheticDataset
from eduloggen.scenarios import (
    ANOMALIES,
    AnomalySpec,
    BaseAnomaly,
    Injection,
    InjectionContext,
    available_anomalies,
    get_anomaly,
    inject_anomalies,
    load_anomaly_specs,
    register_anomaly,
)

ALL_P0 = [
    "event_frequency",
    "abnormal_timing",
    "unexpected_transition",
    "repetition",
    "inactivity",
]


@pytest.fixture(scope="module")
def real() -> Dataset:
    return sessionize(demo_dataset(40, seed=5))


@pytest.fixture(scope="module")
def synthetic(real: Dataset) -> SyntheticDataset:
    generator = get_generator("semi_markov")
    return generator.generate(generator.fit(real), 200, seed=1)


def _members(dataset: Dataset) -> dict[str, list[LogRecord]]:
    members: defaultdict[str, list[LogRecord]] = defaultdict(list)
    for event in dataset.sorted_events():
        members[event.session_id or ""].append(event)
    return dict(members)


def _sessions_of(result: Any, kind: str) -> list[list[LogRecord]]:
    members = _members(result.dataset)
    return [
        members[sid] for sid in result.annotations.anomalous_ids("session", type=kind)
    ]


# --------------------------------------------------------------------------
# General guarantees
# --------------------------------------------------------------------------


def test_registry() -> None:
    assert available_anomalies() == sorted(ALL_P0)
    assert get_anomaly("inactivity").category == "unusual_valid"
    assert get_anomaly("unexpected_transition").category == "invalid_workflow"
    with pytest.raises(ConfigError) as info:
        get_anomaly("cheating")
    assert info.value.code == "anomaly_unknown"


def test_all_anomalies_together(synthetic: SyntheticDataset) -> None:
    specs = [{"type": name, "rate": 0.05} for name in ALL_P0]
    result = inject_anomalies(synthetic, specs, seed=3)
    assert isinstance(result.dataset, SyntheticDataset)
    assert result.dataset.n_sessions == synthetic.n_sessions
    report = result.report["anomalies"]
    for name in ALL_P0:
        assert report[name]["injected"] == report[name]["requested"] == 10
        assert len(result.annotations.anomalous_ids("session", type=name)) == 10
    sessions = result.annotations.anomalous_ids("session")
    assert len(sessions) == 50
    assert result.annotations.summary()["session_categories"] == {
        "unusual_valid": 40,
        "invalid_workflow": 10,
    }
    result.annotations.check_against(result.dataset)


def test_one_anomaly_per_session(synthetic: SyntheticDataset) -> None:
    specs = [{"type": name, "rate": 0.2} for name in ALL_P0]
    result = inject_anomalies(synthetic, specs, seed=1)
    counts = Counter(a.id for a in result.annotations.anomalies("session"))
    assert max(counts.values()) == 1


def test_ids_do_not_reveal_injection(synthetic: SyntheticDataset) -> None:
    result = inject_anomalies(
        synthetic, [{"type": "event_frequency", "rate": 0.1}], seed=2
    )
    ids = [e.event_id for e in result.dataset.events]
    assert all(i.startswith("E") and i[1:].isdigit() for i in ids)
    assert ids == sorted(ids)
    assert result.dataset.events == result.dataset.sorted_events()


def test_sessions_of_a_learner_never_overlap(synthetic: SyntheticDataset) -> None:
    specs = [
        {"type": "inactivity", "rate": 0.3, "gap_s": 50_000},
        {"type": "event_frequency", "rate": 0.3, "count": 50, "window_s": 20_000},
    ]
    dataset = inject_anomalies(synthetic, specs, seed=4).dataset
    by_learner: defaultdict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
    for session in dataset.sessions or ():
        by_learner[session.learner_id].append((session.start_time, session.end_time))
    for spans in by_learner.values():
        spans.sort()
        for (_, end), (start, _) in pairwise(spans):
            assert end < start


def test_reproducible_and_seed_sensitive(synthetic: SyntheticDataset) -> None:
    specs = [{"type": "repetition", "rate": 0.1}, {"type": "inactivity", "rate": 0.1}]
    a = inject_anomalies(synthetic, specs, seed=9)
    b = inject_anomalies(synthetic, specs, seed=9)
    c = inject_anomalies(synthetic, specs, seed=10)
    assert a.dataset.fingerprint() == b.dataset.fingerprint()
    assert a.annotations.fingerprint() == b.annotations.fingerprint()
    assert a.annotations.fingerprint() != c.annotations.fingerprint()


def test_input_is_not_modified(synthetic: SyntheticDataset) -> None:
    before = synthetic.fingerprint()
    inject_anomalies(synthetic, [{"type": "abnormal_timing", "rate": 0.5}], seed=0)
    assert synthetic.fingerprint() == before


def test_works_on_real_unsessionized_input_errors(real: Dataset) -> None:
    result = inject_anomalies(real, [{"type": "inactivity", "rate": 0.1}], seed=0)
    assert type(result.dataset) is Dataset
    with pytest.raises(AnalysisError):
        inject_anomalies(
            Dataset(dataset_id="d", events=real.events),
            [{"type": "inactivity", "rate": 0.1}],
            seed=0,
        )


# --------------------------------------------------------------------------
# Each anomaly does what it says
# --------------------------------------------------------------------------


def test_event_frequency(synthetic: SyntheticDataset) -> None:
    result = inject_anomalies(
        synthetic,
        [
            {
                "type": "event_frequency",
                "rate": 0.05,
                "token": "attempt",
                "count": 25,
                "window_s": 180,
            }
        ],
        seed=1,
    )
    for events in _sessions_of(result, "event_frequency"):
        tokens = [e.event_type for e in events]
        assert tokens.count("attempt") >= 25
    row = result.annotations.anomalies("session", type="event_frequency")[0]
    assert dict(row.parameters) == {
        "token": "attempt",
        "count": 25,
        "window_s": 180.0,
        "injected_events": 25,
    }
    assert (
        len(result.annotations.anomalous_ids("event", type="event_frequency"))
        == 25 * 10
    )


def test_abnormal_timing(synthetic: SyntheticDataset) -> None:
    result = inject_anomalies(
        synthetic, [{"type": "abnormal_timing", "rate": 0.1, "factor": 0.01}], seed=1
    )
    for row in result.annotations.anomalies("session", type="abnormal_timing"):
        before = row.parameters["duration_before_s"]
        after = row.parameters["duration_after_s"]
        assert after <= max(before * 0.01, 0.001 * 30) + 1e-3
    for events in _sessions_of(result, "abnormal_timing"):
        assert all(a.timestamp < b.timestamp for a, b in pairwise(events))


def test_unexpected_transition_uses_unobserved_pairs(
    real: Dataset, synthetic: SyntheticDataset
) -> None:
    observed = set(ngram_counts(session_sequences(real), 2))
    result = inject_anomalies(
        synthetic,
        [{"type": "unexpected_transition", "rate": 0.1}],
        seed=1,
        reference=real,
    )
    for row in result.annotations.anomalies("session", type="unexpected_transition"):
        assert (row.parameters["from"], row.parameters["to"]) not in observed
        assert row.category == "invalid_workflow"
    for events in _sessions_of(result, "unexpected_transition"):
        pairs = set(pairwise(e.event_type for e in events))
        assert pairs - observed


def test_unexpected_transition_with_configured_pairs(
    synthetic: SyntheticDataset,
) -> None:
    result = inject_anomalies(
        synthetic,
        [
            {
                "type": "unexpected_transition",
                "rate": 0.05,
                "transitions": [["submit", "navigate"]],
            }
        ],
        seed=1,
    )
    for events in _sessions_of(result, "unexpected_transition"):
        assert ("submit", "navigate") in set(pairwise(e.event_type for e in events))


def test_unexpected_transition_skips_when_everything_was_observed() -> None:
    from ..generators.conftest import build

    data = build([("a", "b", "a", "a", "b", "b")] * 5)
    result = inject_anomalies(
        data, [{"type": "unexpected_transition", "rate": 0.4}], seed=0
    )
    info = result.report["anomalies"]["unexpected_transition"]
    assert info["injected"] == 0
    assert info["not_applicable"] == 5


@pytest.mark.parametrize("pattern", ["loop", "ping_pong"])
def test_repetition(synthetic: SyntheticDataset, pattern: str) -> None:
    result = inject_anomalies(
        synthetic,
        [{"type": "repetition", "rate": 0.05, "pattern": pattern, "length": 10}],
        seed=1,
    )
    for row in result.annotations.anomalies("session", type="repetition"):
        assert len(row.parameters["tokens"]) == (1 if pattern == "loop" else 2)
        assert row.parameters["length"] == 10


def test_inactivity(synthetic: SyntheticDataset) -> None:
    result = inject_anomalies(
        synthetic, [{"type": "inactivity", "rate": 0.1, "gap_s": 3600}], seed=1
    )
    for events in _sessions_of(result, "inactivity"):
        gaps = [
            (b.timestamp - a.timestamp).total_seconds() for a, b in pairwise(events)
        ]
        assert max(gaps) >= 3600


def test_not_applicable_sessions_are_skipped_and_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from ..generators.conftest import build

    data = build([("a",)] * 8 + [("a", "b", "c")] * 2)
    with caplog.at_level(logging.WARNING, logger="eduloggen.scenarios"):
        result = inject_anomalies(
            data, [{"type": "abnormal_timing", "rate": 0.5, "min_events": 3}], seed=0
        )
    info = result.report["anomalies"]["abnormal_timing"]
    assert info["injected"] == 2
    assert info["not_applicable"] == 8
    assert "injected 2 of 5" in caplog.text


# --------------------------------------------------------------------------
# Configuration errors
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("specs", "message"),
    [
        ([], "no anomalies"),
        ([{"type": "inactivity"}], "'type' and 'rate'"),
        ([{"type": "inactivity", "rate": 0}], "rate"),
        ([{"type": "inactivity", "rate": 1.5}], "rate"),
        ([{"type": "inactivity", "rate": True}], "rate"),
        (
            [{"type": "inactivity", "rate": 0.6}, {"type": "repetition", "rate": 0.6}],
            "more than 1",
        ),
        ([{"type": "inactivity", "rate": 0.1, "gap": 3}], "unknown parameters"),
        ([{"type": "inactivity", "rate": 0.1, "gap_s": -1}], "gap_s"),
        ([{"type": "abnormal_timing", "rate": 0.1, "factor": 2}], "factor"),
        ([{"type": "abnormal_timing", "rate": 0.1, "min_events": 1}], "min_events"),
        ([{"type": "event_frequency", "rate": 0.1, "count": 1}], "count"),
        (
            [{"type": "event_frequency", "rate": 0.1, "token": "teleport"}],
            "not in the vocabulary",
        ),
        ([{"type": "repetition", "rate": 0.1, "pattern": "zigzag"}], "pattern"),
        (
            [{"type": "unexpected_transition", "rate": 0.1, "transitions": [["a"]]}],
            "pairs",
        ),
        (
            [{"type": "unexpected_transition", "rate": 0.1, "transitions": "a->b"}],
            "pairs",
        ),
    ],
)
def test_invalid_specs(
    synthetic: SyntheticDataset, specs: list[dict[str, Any]], message: str
) -> None:
    with pytest.raises(ConfigError) as info:
        inject_anomalies(synthetic, specs, seed=0)
    assert message in info.value.message


def test_load_anomaly_specs(tmp_path: Any) -> None:
    path = tmp_path / "a.yaml"
    path.write_text("anomalies:\n  - {type: inactivity, rate: 0.1, gap_s: 60}\n")
    assert load_anomaly_specs(path) == [AnomalySpec("inactivity", 0.1, {"gap_s": 60})]
    for content in (
        "anomalies: []\n",
        "other: 1\n",
        "anomalies: [{type: x, rate: 1}]\nextra: 1\n",
    ):
        path.write_text(content)
        with pytest.raises(ConfigError):
            load_anomaly_specs(path)


# --------------------------------------------------------------------------
# Custom injectors
# --------------------------------------------------------------------------


class Truncate(BaseAnomaly):
    name: ClassVar[str] = "truncate"
    category: ClassVar[Any] = "unusual_valid"
    defaults: ClassVar[dict[str, Any]] = {}

    def inject(
        self, events: list[LogRecord], params: Any, context: InjectionContext, rng: Any
    ) -> Injection | None:
        if len(events) < 2:
            return None
        return Injection(events[:1], [events[0].event_id], {"kept": 1})


@pytest.fixture
def truncate() -> Iterator[None]:
    register_anomaly(Truncate())
    yield
    ANOMALIES.pop("truncate", None)


def test_custom_anomaly(synthetic: SyntheticDataset, truncate: None) -> None:
    result = inject_anomalies(synthetic, [{"type": "truncate", "rate": 0.1}], seed=0)
    for events in _sessions_of(result, "truncate"):
        assert len(events) == 1
    with pytest.raises(PluginError) as info:
        register_anomaly(Truncate())
    assert info.value.code == "plugin_duplicate"


def test_register_anomaly_contract() -> None:
    with pytest.raises(PluginError):
        register_anomaly("inactivity")  # type: ignore[arg-type]
    with pytest.raises(PluginError):
        register_anomaly(get_anomaly("inactivity"))

    class BadCategory(Truncate):
        name: ClassVar[str] = "bad_category"
        category: ClassVar[Any] = "cheating"

    with pytest.raises(PluginError):
        register_anomaly(BadCategory())

    class BadName(Truncate):
        name: ClassVar[str] = "bad name"

    with pytest.raises(PluginError):
        register_anomaly(BadName())


def test_example_anomalies_file_is_valid(synthetic: SyntheticDataset) -> None:
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "examples/configs/anomalies.yaml"
    specs = load_anomaly_specs(path)
    assert sorted(spec.type for spec in specs) == sorted(ALL_P0)
    result = inject_anomalies(synthetic, specs, seed=0)
    assert len(result.annotations.anomalous_ids("session")) == 30
