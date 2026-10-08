"""Tests for the registry, GeneratorModel, describe, and artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from eduloggen.core import ConfigError, IngestionError, PluginError, SchemaError
from eduloggen.generators import (
    IndependentGenerator,
    MarkovGenerator,
    SemiMarkovGenerator,
    available_generators,
    get_generator,
    load_model,
    register_generator,
    unregister_generator,
)
from eduloggen.models import Dataset, GeneratorModel

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@pytest.fixture
def cleanup() -> Iterator[None]:
    yield
    unregister_generator("custom")
    register_generator("markov", MarkovGenerator, replace=True)


def test_builtins_registered() -> None:
    assert {"markov", "semi_markov", "independent"} <= set(available_generators())
    assert isinstance(get_generator("semi_markov"), SemiMarkovGenerator)


def test_unknown_generator() -> None:
    with pytest.raises(ConfigError) as info:
        get_generator("gan")
    assert info.value.code == "generator_unknown"
    assert "markov" in info.value.context["available"]


def test_register_custom(cleanup: None, abc: Dataset) -> None:
    class Custom(IndependentGenerator):
        name = "custom"

    register_generator("custom", Custom)
    assert "custom" in available_generators()
    model = get_generator("custom").fit(abc)
    assert model.generator_id == "custom"
    with pytest.raises(PluginError) as info:
        register_generator("custom", Custom)
    assert info.value.code == "plugin_duplicate"
    register_generator("markov", Custom, replace=True)
    assert isinstance(get_generator("markov"), Custom)
    unregister_generator("custom")
    unregister_generator("custom")
    assert "custom" not in available_generators()


@pytest.mark.parametrize("name", ["", "my-gen", "1abc", None])
def test_register_rejects_bad_names(name: Any) -> None:
    with pytest.raises(PluginError) as info:
        register_generator(name, MarkovGenerator)
    assert info.value.code == "plugin_invalid_name"


def test_register_rejects_non_callable() -> None:
    with pytest.raises(PluginError):
        register_generator("custom", "markov")  # type: ignore[arg-type]


def test_bad_factories(cleanup: None) -> None:
    def broken() -> Any:
        raise RuntimeError("boom")

    register_generator("custom", broken)
    with pytest.raises(PluginError) as info:
        get_generator("custom")
    assert info.value.code == "plugin_factory_failed"

    register_generator("custom", lambda: object(), replace=True)  # type: ignore[arg-type,return-value]
    with pytest.raises(PluginError) as info:
        get_generator("custom")
    assert info.value.code == "plugin_contract_violation"


# --------------------------------------------------------------------------
# GeneratorModel
# --------------------------------------------------------------------------


@pytest.fixture
def model(varied: Dataset) -> GeneratorModel:
    return SemiMarkovGenerator().fit(varied, {"order": 2})


def test_model_round_trip_and_fingerprint(model: GeneratorModel) -> None:
    data = json.loads(json.dumps(model.to_dict()))
    restored = GeneratorModel.from_dict(data)
    assert restored == model
    later = GeneratorModel.from_dict(data | {"fitted_at": "2031-01-01T00:00:00+00:00"})
    assert later.fingerprint() == model.fingerprint()
    retuned = GeneratorModel.from_dict(
        data | {"hyperparameters": {**data["hyperparameters"], "order": 1}}
    )
    assert retuned.fingerprint() != model.fingerprint()


def test_model_payloads_are_frozen_copies(model: GeneratorModel) -> None:
    params = {"x": [1, 2]}
    copy = GeneratorModel(
        generator_id="g",
        generator_version="1",
        hyperparameters={},
        vocabulary=("a",),
        parameters=params,
        training_fingerprint="sha256:x",
    )
    params["x"].append(3)
    assert copy.parameters == {"x": [1, 2]}


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"generator_id": ""}, "generator_id"),
        ({"vocabulary": ["a", ""]}, "vocabulary"),
        ({"vocabulary": "ab"}, "vocabulary"),
        ({"parameters": []}, "parameters"),
        ({"parameters": {"x": float("nan")}}, "parameters"),
        ({"hyperparameters": {"x": object()}}, "hyperparameters"),
        ({"fitted_at": "2026-01-01T00:00:00"}, "fitted_at"),
    ],
)
def test_model_validation(
    model: GeneratorModel, change: dict[str, Any], field: str
) -> None:
    with pytest.raises(SchemaError) as info:
        GeneratorModel.from_dict(model.to_dict() | change)
    assert info.value.context["field"] == field


def test_model_from_dict_requires_all_keys(model: GeneratorModel) -> None:
    data = model.to_dict()
    del data["training_fingerprint"]
    with pytest.raises(SchemaError) as info:
        GeneratorModel.from_dict(data)
    assert info.value.code == "schema_missing_field"


def test_artifact_version_check(model: GeneratorModel) -> None:
    model.check_artifact_version()
    future = GeneratorModel.from_dict(model.to_dict() | {"artifact_version": "2.0"})
    with pytest.raises(SchemaError) as info:
        future.check_artifact_version()
    assert info.value.code == "schema_unsupported_version"


# --------------------------------------------------------------------------
# describe / artifacts
# --------------------------------------------------------------------------


def test_describe(model: GeneratorModel) -> None:
    info = SemiMarkovGenerator().describe(model)
    assert info["generator"] == "semi_markov"
    assert info["order"] == 2
    assert info["vocabulary_size"] == len(model.vocabulary)
    assert info["training_sessions"] == 60
    assert info["tokenization"] == "event_type"
    assert info["model_fingerprint"] == model.fingerprint()
    assert info["tokens_with_own_timing"] > 0
    baseline = IndependentGenerator()
    assert baseline.describe(baseline.fit(_tiny()))["generator"] == "independent"


def _tiny() -> Dataset:
    from .conftest import build

    return build([("a", "b")])


def test_save_and_load(tmp_path: Path, model: GeneratorModel) -> None:
    generator = SemiMarkovGenerator()
    target = generator.save(model, tmp_path / "model")
    assert sorted(p.name for p in target.iterdir()) == [
        "DESCRIPTION.md",
        "hyperparameters.json",
        "manifest.json",
        "parameters.json",
        "vocabulary.json",
    ]
    assert generator.load(target) == model
    assert load_model(target) == model
    description = (target / "DESCRIPTION.md").read_text()
    assert description.startswith("# semi_markov model")
    assert "| timing_family | empirical |" in description
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["model_fingerprint"] == model.fingerprint()
    synthetic = generator.generate(generator.load(target), 5, seed=1)
    assert synthetic.fingerprint() == generator.generate(model, 5, seed=1).fingerprint()


def test_save_respects_force(tmp_path: Path, model: GeneratorModel) -> None:
    from eduloggen.core import ExportError

    generator = SemiMarkovGenerator()
    generator.save(model, tmp_path / "m")
    with pytest.raises(ExportError):
        generator.save(model, tmp_path / "m")
    generator.save(model, tmp_path / "m", force=True)


def test_load_with_wrong_generator(tmp_path: Path, model: GeneratorModel) -> None:
    from eduloggen.core import GenerationError

    target = SemiMarkovGenerator().save(model, tmp_path / "m")
    with pytest.raises(GenerationError):
        MarkovGenerator().load(target)


def test_load_detects_tampering(tmp_path: Path, model: GeneratorModel) -> None:
    target = SemiMarkovGenerator().save(model, tmp_path / "m")
    params = json.loads((target / "parameters.json").read_text())
    params["order"] = 1
    (target / "parameters.json").write_text(json.dumps(params))
    with pytest.raises(IngestionError) as info:
        load_model(target)
    assert info.value.code == "io_model_integrity"


@pytest.mark.parametrize(
    ("file", "content", "error"),
    [
        ("parameters.json", None, IngestionError),
        ("parameters.json", "{broken", IngestionError),
        ("vocabulary.json", "[1, 2]", SchemaError),
        ("manifest.json", '{"generator_id": "x"}', SchemaError),
    ],
)
def test_load_rejects_damaged_artifacts(
    tmp_path: Path,
    model: GeneratorModel,
    file: str,
    content: str | None,
    error: type[Exception],
) -> None:
    target = SemiMarkovGenerator().save(model, tmp_path / "m")
    if content is None:
        (target / file).unlink()
    else:
        (target / file).write_text(content)
    with pytest.raises(error):
        load_model(target)


def test_load_rejects_future_artifact_version(
    tmp_path: Path, model: GeneratorModel
) -> None:
    target = SemiMarkovGenerator().save(model, tmp_path / "m")
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["artifact_version"] = "2.0"
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SchemaError):
        load_model(target)


def test_vocabulary_without_tokens(tmp_path: Path, model: GeneratorModel) -> None:
    target = SemiMarkovGenerator().save(model, tmp_path / "m")
    (target / "vocabulary.json").write_text("{}")
    with pytest.raises(IngestionError):
        load_model(target)


def test_fitted_at_is_recent(model: GeneratorModel) -> None:
    assert model.fitted_at.year >= 2026
    assert isinstance(model.fitted_at, datetime)
