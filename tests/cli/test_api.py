"""Tests for the top-level Python API façade."""

from __future__ import annotations

from pathlib import Path

import pytest

import eduloggen as elg
from eduloggen.config import AppConfig
from eduloggen.core import ConfigError, GenerationError
from eduloggen.io import FieldMapping
from eduloggen.models import Dataset

from .conftest import EMAIL


@pytest.fixture
def real(export: Path) -> Dataset:
    return elg.sessionize(
        elg.ingest(export / "lms.csv", export / "mapping.yaml").dataset
    )


def test_workflow_from_top_level(export: Path, tmp_path: Path) -> None:
    result = elg.ingest(export / "lms.csv", export / "mapping.yaml")
    assert result.report.status == "ready"
    assert all(EMAIL.format(0) not in e.learner_id for e in result.dataset.events)

    real = elg.sessionize(result.dataset)
    assert elg.analyze(real).feature("n_learners").value == 12

    model = elg.fit_generator("markov", real)
    synthetic = elg.generate(model, n_sessions=20, seed=1)
    assert synthetic.n_sessions == 20
    report = elg.validate(real, synthetic, thresholds={"event_type_tvd": 1.0})
    assert report.passed

    path = elg.save_dataset(synthetic, tmp_path / "synthetic")
    assert elg.load_dataset(path).fingerprint() == synthetic.fingerprint()


def test_config_drives_defaults(export: Path, tmp_path: Path) -> None:
    config_file = tmp_path / "experiment.yaml"
    config_file.write_text("""
io: {output_format: jsonl, event_types: [hint], unknown_event_policy: map_to_other}
sessionization: {strategy: idle_timeout, idle_timeout_s: 60, tokenization: activity_id}
analysis: {ngram_order: 3}
generator: {name: semi_markov, order: 2, on_insufficient_data: backoff}
generation: {seed: 5, n_sessions: 7, id_strategy: preserve}
validation:
  metrics: [event_type_tvd, bigram_tvd]
  thresholds: {event_type_tvd: 0.9, nn_distance_p05: 0.1}
privacy: {strip_metadata_keys: [ip]}
""")
    config = elg.load_config(config_file)
    ingested = elg.ingest(export / "lms.csv", export / "mapping.yaml", config=config)
    assert all("ip" not in e.metadata for e in ingested.dataset.events)
    assert all("device" in e.metadata for e in ingested.dataset.events)

    real = elg.sessionize(ingested.dataset, config=config)
    assert real.sessions is not None
    assert real.sessions[0].event_sequence[0].startswith("res-")
    assert elg.analyze(real, config=config).ngram_order == 3

    model = elg.fit_generator(None, real, config=config)
    assert model.generator_id == "semi_markov"
    assert model.hyperparameters["order"] == 2

    synthetic = elg.generate(model, config=config)
    assert synthetic.n_sessions == 7
    assert synthetic.generation.seed == 5
    assert synthetic.generation.id_strategy == "preserve"

    report = elg.validate(real, synthetic, config=config)
    assert [m.name for m in report.metrics] == ["event_type_tvd", "bigram_tvd"]
    assert report.metric("event_type_tvd").threshold == 0.9
    assert report.protocol["seed"] == 5

    path = elg.save_dataset(synthetic, tmp_path / "out", config=config)
    assert (path / "events.jsonl").exists()


def test_config_hyperparameters_only_for_configured_generator(
    real: Dataset,
) -> None:
    config = AppConfig.from_dict(
        {"generator": {"name": "semi_markov", "min_samples": 2}}
    )
    model = elg.fit_generator("markov", real, config=config)
    assert "min_samples" not in model.hyperparameters
    explicit = elg.fit_generator(
        "markov", real, hyperparameters={"smoothing_alpha": 0.5}
    )
    assert explicit.hyperparameters["smoothing_alpha"] == 0.5


def test_generate_requires_seed(real: Dataset) -> None:
    model = elg.fit_generator("markov", real)
    with pytest.raises(GenerationError) as info:
        elg.generate(model)
    assert info.value.code == "generation_missing_seed"


def test_validate_threshold_rules(real: Dataset) -> None:
    config = AppConfig.from_dict({"validation": {"thresholds": {"bigram_tvd": 0.5}}})
    report = elg.validate(real, real, config=config, metrics=["event_type_tvd"])
    assert report.metric("event_type_tvd").status == "info"
    with pytest.raises(ConfigError):
        elg.validate(
            real, real, metrics=["event_type_tvd"], thresholds={"bigram_tvd": 1}
        )


def test_ingest_accepts_mapping_object(export: Path) -> None:
    mapping = FieldMapping.from_file(export / "mapping.yaml")
    result = elg.ingest(export / "lms.csv", mapping, strict=True, dataset_id="x")
    assert result.dataset.dataset_id == "x"
    assert result.mapping is mapping
