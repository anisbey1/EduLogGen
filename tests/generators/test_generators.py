"""Behavioural and contract tests for the built-in generators."""

from __future__ import annotations

import logging
import re
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

import pytest

from eduloggen.analysis import (
    interevent_times,
    ngram_counts,
    session_sequences,
    sessionize,
)
from eduloggen.core import ConfigError, FitError, GenerationError
from eduloggen.generators import (
    BaseGenerator,
    IndependentGenerator,
    MarkovGenerator,
    SemiMarkovGenerator,
    get_generator,
)
from eduloggen.models import Dataset, GeneratorModel, Session, SyntheticDataset

from .conftest import T0, build

ALL = ["markov", "semi_markov", "independent"]


# --------------------------------------------------------------------------
# Contract: every generator
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_contract(name: str, varied: Dataset) -> None:
    generator = get_generator(name)
    model = generator.fit(varied)
    synthetic = generator.generate(model, n_sessions=40, seed=3)

    assert isinstance(synthetic, SyntheticDataset)
    assert synthetic.n_sessions == 40
    assert synthetic.generation.generator_id == name
    assert synthetic.generation.seed == 3
    assert synthetic.generation.model_fingerprint == model.fingerprint()
    assert dict(synthetic.metadata.seeds) == {"generate": 3}
    assert set(model.vocabulary) >= {e.event_type for e in synthetic.events}

    for event in synthetic.events:
        assert re.fullmatch(r"E\d+", event.event_id)
        assert re.fullmatch(r"L\d+", event.learner_id)
        assert re.fullmatch(r"S\d+", event.session_id or "")

    by_session: defaultdict[str, list[datetime]] = defaultdict(list)
    for event in synthetic.sorted_events():
        by_session[event.session_id or ""].append(event.timestamp)
    for times in by_session.values():
        assert all(a < b for a, b in pairwise(times))

    training_pairs = {(e.event_type, e.activity_id) for e in varied.events}
    assert {(e.event_type, e.activity_id) for e in synthetic.events} <= training_pairs

    resessionized = sessionize(synthetic, strategy="explicit")
    assert session_sequences(resessionized) == session_sequences(synthetic)


@pytest.mark.parametrize("name", ALL)
def test_reproducible(name: str, varied: Dataset) -> None:
    generator = get_generator(name)
    model = generator.fit(varied)
    first = generator.generate(model, 25, seed=11)
    assert first.fingerprint() == generator.generate(model, 25, seed=11).fingerprint()
    assert first.fingerprint() != generator.generate(model, 25, seed=12).fingerprint()
    assert generator.fit(varied).fingerprint() == model.fingerprint()


def test_sample_alias(abc: Dataset) -> None:
    generator = MarkovGenerator()
    model = generator.fit(abc)
    assert generator.sample(model, 2, 0).fingerprint() == (
        generator.generate(model, 2, 0).fingerprint()
    )


def test_preserve_keeps_generation_order_ids(abc: Dataset, caplog: Any) -> None:
    generator = MarkovGenerator()
    model = generator.fit(abc)
    with caplog.at_level(logging.WARNING):
        synthetic = generator.generate(model, 3, seed=0, id_strategy="preserve")
    assert synthetic.events[0].event_id == "E1"
    assert synthetic.generation.id_strategy == "preserve"


def test_start_time_override(abc: Dataset) -> None:
    generator = MarkovGenerator()
    model = generator.fit(abc)
    start = datetime(2030, 1, 1, tzinfo=UTC)
    synthetic = generator.generate(model, 5, seed=0, start_time=start)
    assert min(e.timestamp for e in synthetic.events) >= start


def test_custom_dataset_id(abc: Dataset) -> None:
    generator = MarkovGenerator()
    synthetic = generator.generate(generator.fit(abc), 1, 0, dataset_id="mine")
    assert synthetic.dataset_id == "mine"


# --------------------------------------------------------------------------
# Markov behaviour
# --------------------------------------------------------------------------


def test_markov_reproduces_deterministic_paths(abc: Dataset) -> None:
    generator = MarkovGenerator()
    synthetic = generator.generate(generator.fit(abc), 20, seed=1)
    assert set(session_sequences(synthetic)) == {("a", "b", "c")}


def test_markov_backs_off_after_dead_end(abc: Dataset) -> None:
    generator = MarkovGenerator()
    model = generator.fit(abc, {"length_model": "fixed", "fixed_length": 6})
    for sequence in session_sequences(generator.generate(model, 20, seed=1)):
        assert len(sequence) == 6
        assert sequence[:3] == ("a", "b", "c")


def test_markov_start_distribution(varied: Dataset) -> None:
    generator = MarkovGenerator()
    synthetic = generator.generate(generator.fit(varied), 300, seed=2)
    real_starts = {s[0] for s in session_sequences(varied)}
    assert {s[0] for s in session_sequences(synthetic)} <= real_starts


def test_markov_alpha_zero_only_observed_bigrams(varied: Dataset) -> None:
    generator = MarkovGenerator()
    model = generator.fit(varied)
    dead_ends = {
        token
        for token in model.vocabulary
        if not any(a == token for a, _ in ngram_counts(session_sequences(varied), 2))
    }
    assert not dead_ends
    synthetic = generator.generate(model, 200, seed=4)
    real = set(ngram_counts(session_sequences(varied), 2))
    assert set(ngram_counts(session_sequences(synthetic), 2)) <= real


def test_markov_smoothing_creates_unseen_transitions(abc: Dataset) -> None:
    generator = MarkovGenerator()
    model = generator.fit(abc, {"smoothing_alpha": 1.0})
    sequences = session_sequences(generator.generate(model, 50, seed=0))
    assert any(s != ("a", "b", "c") for s in sequences)


def test_markov_second_order(varied: Dataset) -> None:
    generator = MarkovGenerator()
    model = generator.fit(varied, {"order": 2})
    assert model.parameters["order"] == 2
    assert set(model.parameters["transitions"]) == {"1", "2"}
    assert generator.describe(model)["order"] == 2


def test_markov_timing_is_constant(varied: Dataset) -> None:
    generator = MarkovGenerator()
    synthetic = generator.generate(generator.fit(varied), 30, seed=0)
    assert len(set(interevent_times(synthetic))) == 1


def test_insufficient_data_fails_or_backs_off(abc: Dataset, caplog: Any) -> None:
    generator = MarkovGenerator()
    with pytest.raises(FitError) as info:
        generator.fit(abc, {"order": 3})
    assert info.value.code == "fit_insufficient_data"
    assert "Lower 'order'" in info.value.message
    with caplog.at_level(logging.WARNING, logger="eduloggen.generators"):
        model = generator.fit(abc, {"order": 5, "on_insufficient_data": "backoff"})
    assert model.parameters["order"] == 2
    assert "reducing Markov order" in caplog.text


def test_single_token_vocabulary() -> None:
    data = build([("a",), ("a", "a")])
    generator = MarkovGenerator()
    synthetic = generator.generate(generator.fit(data), 5, seed=0)
    assert {e.event_type for e in synthetic.events} == {"a"}


def test_length_models(varied: Dataset) -> None:
    generator = MarkovGenerator()
    fixed = generator.fit(varied, {"length_model": "fixed", "fixed_length": 4})
    assert {s.n_events for s in generator.generate(fixed, 10, 0).sessions or ()} == {4}
    poisson = generator.fit(varied, {"length_model": "poisson"})
    lengths = [s.n_events for s in generator.generate(poisson, 300, 0).sessions or ()]
    real = [s.n_events for s in varied.sessions or ()]
    assert statistics.mean(lengths) == pytest.approx(statistics.mean(real), rel=0.15)


def test_activity_tokenization() -> None:
    data = sessionize(
        build([("a", "b")] * 5, activities={"a": "intro", "b": "quiz"}),
        tokenization="activity_id",
    )
    generator = MarkovGenerator()
    model = generator.fit(data)
    assert model.parameters["population"]["tokenization"] == "activity_id"
    synthetic = generator.generate(model, 5, seed=0)
    assert {(e.activity_id, e.event_type) for e in synthetic.events} == {
        ("intro", "a"),
        ("quiz", "b"),
    }


def test_population_statistics(varied: Dataset) -> None:
    generator = MarkovGenerator()
    synthetic = generator.generate(generator.fit(varied), 60, seed=0)
    per_learner = Counter(s.learner_id for s in synthetic.sessions or ())
    assert max(per_learner.values()) <= 3
    assert {e.course_id for e in synthetic.events} <= {"c1", "c2"}


# --------------------------------------------------------------------------
# Semi-Markov behaviour
# --------------------------------------------------------------------------


def _gaps_after(dataset: Dataset) -> dict[str, list[float]]:
    result: defaultdict[str, list[float]] = defaultdict(list)
    events = dataset.sorted_events()
    for earlier, later in pairwise(events):
        if earlier.session_id == later.session_id:
            result[earlier.event_type].append(
                (later.timestamp - earlier.timestamp).total_seconds()
            )
    return result


@pytest.mark.parametrize("family", ["empirical", "lognormal", "gamma", "exponential"])
def test_semi_markov_token_timing(varied: Dataset, family: str) -> None:
    generator = SemiMarkovGenerator()
    model = generator.fit(varied, {"timing_family": family})
    gaps = _gaps_after(generator.generate(model, 200, seed=0))
    assert statistics.median(gaps["video_play"]) > 5 * statistics.median(gaps["view"])
    assert statistics.median(gaps["navigate"]) < statistics.median(gaps["view"])


def test_semi_markov_falls_back_to_pooled(varied: Dataset) -> None:
    generator = SemiMarkovGenerator()
    model = generator.fit(varied, {"min_samples": 10_000})
    assert model.parameters["timing"]["per_token"] == {}
    assert generator.describe(model)["tokens_with_own_timing"] == 0
    synthetic = generator.generate(model, 10, seed=0)
    assert synthetic.n_sessions == 10


def test_semi_markov_zero_gaps_stay_increasing() -> None:
    data = build([("a", "b", "a")] * 6, gap_after={"a": 0.0, "b": 0.0})
    generator = SemiMarkovGenerator()
    synthetic = generator.generate(generator.fit(data), 5, seed=0)
    assert min(interevent_times(synthetic)) >= 0.001


# --------------------------------------------------------------------------
# Independent baseline
# --------------------------------------------------------------------------


def test_independent_ignores_order(abc: Dataset) -> None:
    generator = IndependentGenerator()
    sequences = session_sequences(generator.generate(generator.fit(abc), 100, 0))
    assert any(s != ("a", "b", "c") for s in sequences)
    assert {t for s in sequences for t in s} == {"a", "b", "c"}


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "params", "key"),
    [
        ("markov", {"ordre": 1}, None),
        ("markov", {"order": 0}, "generator.order"),
        ("markov", {"order": True}, "generator.order"),
        ("markov", {"smoothing_alpha": -1}, "generator.smoothing_alpha"),
        ("markov", {"smoothing_alpha": "0.1"}, "generator.smoothing_alpha"),
        ("markov", {"on_insufficient_data": "skip"}, "generator.on_insufficient_data"),
        ("markov", {"length_model": "zipf"}, "generator.length_model"),
        ("markov", {"length_model": "fixed"}, "generator.fixed_length"),
        ("markov", {"fixed_length": 0}, "generator.fixed_length"),
        ("semi_markov", {"timing_family": "weibull"}, "generator.timing_family"),
        ("semi_markov", {"min_samples": 0}, "generator.min_samples"),
        ("independent", {"order": 1}, None),
    ],
)
def test_invalid_hyperparameters(
    name: str, params: dict[str, Any], key: str | None, abc: Dataset
) -> None:
    with pytest.raises(ConfigError) as info:
        get_generator(name).fit(abc, params)
    if key is None:
        assert info.value.code == "config_unknown_key"
    else:
        assert info.value.context["key"] == key


def test_fit_errors(abc: Dataset) -> None:
    generator = MarkovGenerator()
    with pytest.raises(FitError) as info:
        generator.fit(Dataset(dataset_id="d", events=abc.events))
    assert info.value.code == "fit_not_sessionized"
    with pytest.raises(FitError) as info:
        generator.fit(Dataset(dataset_id="d", events=(), sessions=()))
    assert info.value.code == "fit_empty_corpus"


def test_fit_rejects_unknown_tokenization(abc: Dataset) -> None:
    sessions = tuple(
        Session(
            session_id=s.session_id,
            learner_id=s.learner_id,
            start_time=s.start_time,
            end_time=s.end_time,
            event_sequence=tuple("x" for _ in s.event_sequence),
        )
        for s in abc.sessions or ()
    )
    tampered = Dataset(dataset_id="d", events=abc.events, sessions=sessions)
    with pytest.raises(FitError) as info:
        MarkovGenerator().fit(tampered)
    assert info.value.code == "fit_unknown_tokenization"


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"n_sessions": 0, "seed": 0}, "generation_invalid_argument"),
        ({"n_sessions": True, "seed": 0}, "generation_invalid_argument"),
        ({"n_sessions": 1, "seed": -1}, "generation_invalid_argument"),
        ({"n_sessions": 1, "seed": 1.5}, "generation_invalid_argument"),
        (
            {"n_sessions": 1, "seed": 0, "start_time": datetime(2030, 1, 1)},
            "generation_invalid_argument",
        ),
    ],
)
def test_generate_argument_errors(
    abc: Dataset, kwargs: dict[str, Any], code: str
) -> None:
    generator = MarkovGenerator()
    model = generator.fit(abc)
    with pytest.raises(GenerationError) as info:
        generator.generate(model, **kwargs)
    assert info.value.code == code


def test_model_mismatch(abc: Dataset) -> None:
    model = MarkovGenerator().fit(abc)
    other = SemiMarkovGenerator()
    with pytest.raises(GenerationError) as info:
        other.generate(model, 1, 0)
    assert info.value.code == "generation_model_mismatch"
    with pytest.raises(GenerationError) as info:
        other.describe(model)
    assert info.value.code == "generation_model_mismatch"


def _with_parameters(model: GeneratorModel, **changes: Any) -> GeneratorModel:
    data = model.to_dict()
    data["parameters"] = {**data["parameters"], **changes}
    return GeneratorModel.from_dict(data)


@pytest.mark.parametrize("name", ALL)
def test_malformed_parameters(name: str, abc: Dataset) -> None:
    generator: BaseGenerator = get_generator(name)
    model = generator.fit(abc)
    broken = _with_parameters(model, population={"tokenization": "event_type"})
    with pytest.raises(GenerationError) as info:
        generator.generate(broken, 1, 0)
    assert info.value.code == "generation_invalid_model"


def test_missing_companion(abc: Dataset) -> None:
    generator = MarkovGenerator()
    model = generator.fit(abc)
    population = dict(model.parameters["population"])
    population["companions"] = {"a": {"act-a": 1}}
    with pytest.raises(GenerationError) as info:
        generator.generate(_with_parameters(model, population=population), 1, 0)
    assert info.value.code == "generation_invalid_model"


def test_malformed_transition_parameters(abc: Dataset) -> None:
    generator = MarkovGenerator()
    model = generator.fit(abc)
    with pytest.raises(GenerationError):
        generator.generate(_with_parameters(model, transitions={}), 1, 0)
    with pytest.raises(GenerationError):
        IndependentGenerator().generate(
            _with_parameters(IndependentGenerator().fit(abc), unigram=None), 1, 0
        )


def test_start_time_respects_origin(abc: Dataset) -> None:
    generator = MarkovGenerator()
    synthetic = generator.generate(generator.fit(abc), 5, seed=0)
    assert min(e.timestamp for e in synthetic.events) >= T0
