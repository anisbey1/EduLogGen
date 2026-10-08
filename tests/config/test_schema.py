"""Tests for AppConfig validation and serialization."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import pytest

from eduloggen.config import AppConfig, config_fingerprint
from eduloggen.core import ConfigError


def test_defaults() -> None:
    config = AppConfig()
    assert config.project.name == "eduloggen"
    assert config.logging.level == "INFO"
    assert config.io.format == "auto"
    assert config.sessionization.strategy == "composite"
    assert config.generator.name == "markov"
    assert dict(config.generator.params) == {}
    assert config.generation.seed is None
    assert config.generation.id_strategy == "remap"
    assert config.visualization.formats == ("png",)
    assert config.benchmark.generators == ("markov", "semi_markov")
    assert AppConfig.from_dict({}) == config


def test_from_dict_converts_types() -> None:
    config = AppConfig.from_dict(
        {
            "sessionization": {"idle_timeout_s": 600},
            "io": {"event_types": ["hint", "chat"]},
            "validation": {"thresholds": {"tvd": 1}},
            "generation": {"seed": 0, "n_sessions": 5},
            "privacy": None,
        }
    )
    assert config.sessionization.idle_timeout_s == 600.0
    assert isinstance(config.sessionization.idle_timeout_s, float)
    assert config.io.event_types == ("hint", "chat")
    assert config.validation.thresholds["tvd"] == 1.0
    assert config.generation.seed == 0


def test_generator_extra_keys_become_params() -> None:
    config = AppConfig.from_dict(
        {"generator": {"name": "semi_markov", "order": 2, "smoothing": "laplace"}}
    )
    assert config.generator.name == "semi_markov"
    assert dict(config.generator.params) == {"order": 2, "smoothing": "laplace"}
    assert config.to_dict()["generator"] == {
        "name": "semi_markov",
        "order": 2,
        "smoothing": "laplace",
    }


def test_round_trip() -> None:
    data: dict[str, Any] = {
        "project": {"name": "x", "output_dir": "out"},
        "io": {"input": "a.csv", "event_types": ["hint"]},
        "generator": {"name": "markov", "order": 1},
        "validation": {"metrics": ["tvd"], "thresholds": {"tvd": 0.2}},
    }
    config = AppConfig.from_dict(data)
    assert AppConfig.from_dict(config.to_dict()) == config


def test_is_immutable() -> None:
    config = AppConfig.from_dict({"validation": {"thresholds": {"a": 1.0}}})
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.generation.seed = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        config.validation.thresholds["a"] = 2.0  # type: ignore[index]
    with pytest.raises(TypeError):
        config.generator.params["x"] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    ("data", "key"),
    [
        ({"logging": {"level": "LOUD"}}, "logging.level"),
        ({"logging": {"json": "yes"}}, "logging.json"),
        ({"generation": {"n_sessions": 0}}, "generation.n_sessions"),
        ({"generation": {"n_sessions": True}}, "generation.n_sessions"),
        ({"generation": {"n_sessions": 2.5}}, "generation.n_sessions"),
        ({"generation": {"seed": -1}}, "generation.seed"),
        ({"sessionization": {"idle_timeout_s": 0}}, "sessionization.idle_timeout_s"),
        (
            {"sessionization": {"idle_timeout_s": "30m"}},
            "sessionization.idle_timeout_s",
        ),
        (
            {"sessionization": {"tokenization": "course_id"}},
            "sessionization.tokenization",
        ),
        ({"analysis": {"ngram_order": 0}}, "analysis.ngram_order"),
        ({"io": {"input": ""}}, "io.input"),
        ({"io": {"event_types": "hint"}}, "io.event_types"),
        ({"io": {"event_types": ["ok", 3]}}, "io.event_types[1]"),
        ({"validation": {"thresholds": ["a"]}}, "validation.thresholds"),
        ({"validation": {"thresholds": {"a": "low"}}}, "validation.thresholds.a"),
        ({"validation": {"thresholds": {1: 0.1}}}, "validation.thresholds"),
        ({"visualization": {"formats": []}}, "visualization.formats"),
        ({"visualization": {"formats": ["gif"]}}, "visualization.formats[0]"),
        ({"visualization": {"dpi": 0}}, "visualization.dpi"),
        ({"benchmark": {"generators": []}}, "benchmark.generators"),
        ({"benchmark": {"repeats": 0}}, "benchmark.repeats"),
        ({"project": "demo"}, "project"),
    ],
)
def test_invalid_values(data: dict[str, Any], key: str) -> None:
    with pytest.raises(ConfigError) as info:
        AppConfig.from_dict(data)
    assert info.value.code == "config_invalid_value"
    assert info.value.context["key"] == key


@pytest.mark.parametrize(
    ("data", "key"),
    [
        ({"proj": {}}, "proj"),
        ({"io": {"inptu": "a.csv"}}, "io.inptu"),
        ({"generator": {"params": {"order": 1}}}, "generator.params"),
    ],
)
def test_unknown_keys_rejected(data: dict[str, Any], key: str) -> None:
    with pytest.raises(ConfigError) as info:
        AppConfig.from_dict(data)
    assert info.value.code == "config_unknown_key"
    assert info.value.context["key"] == key


def test_unknown_keys_allowed_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="eduloggen.config"):
        config = AppConfig.from_dict(
            {"future": {}, "io": {"new_option": 1}}, allow_unknown=True
        )
    assert config == AppConfig()
    assert "future" in caplog.text
    assert "io.new_option" in caplog.text


def test_top_level_must_be_mapping() -> None:
    with pytest.raises(ConfigError):
        AppConfig.from_dict(["io"])  # type: ignore[arg-type]


def test_resolve_path(tmp_path: Path) -> None:
    config = AppConfig(base_dir=tmp_path)
    assert config.resolve_path("data/a.csv") == tmp_path / "data" / "a.csv"
    assert config.resolve_path("/abs/a.csv") == Path("/abs/a.csv")
    assert AppConfig().resolve_path("a.csv") == Path.cwd() / "a.csv"


def test_base_dir_not_compared_or_serialized(tmp_path: Path) -> None:
    assert AppConfig(base_dir=tmp_path) == AppConfig()
    assert "base_dir" not in AppConfig(base_dir=tmp_path).to_dict()


def test_fingerprint() -> None:
    base = AppConfig.from_dict({"generation": {"seed": 1}})
    renamed = AppConfig.from_dict(
        {
            "generation": {"seed": 1},
            "project": {"name": "other"},
            "logging": {"level": "DEBUG"},
        }
    )
    reseeded = AppConfig.from_dict({"generation": {"seed": 2}})
    retuned = AppConfig.from_dict(
        {"generation": {"seed": 1}, "generator": {"order": 2}}
    )

    assert config_fingerprint(base).startswith("sha256:")
    assert config_fingerprint(base) == config_fingerprint(renamed)
    assert config_fingerprint(base) != config_fingerprint(reseeded)
    assert config_fingerprint(base) != config_fingerprint(retuned)
