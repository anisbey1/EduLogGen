"""Tests for file loading, environment variables, and precedence."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from eduloggen.config import (
    apply_overrides,
    deep_merge,
    env_overrides,
    load_config,
    load_file,
    resolve_config,
)
from eduloggen.core import ConfigError
from eduloggen.io import FieldMapping

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "configs"

YAML = """
generation:
  seed: 1
  n_sessions: 10
logging:
  level: WARNING
"""
TOML = """
[generation]
seed = 1
n_sessions = 10

[logging]
level = "WARNING"
"""
JSON = '{"generation": {"seed": 1, "n_sessions": 10}, "logging": {"level": "WARNING"}}'


@pytest.mark.parametrize(
    ("name", "content"),
    [("c.yaml", YAML), ("c.yml", YAML), ("c.toml", TOML), ("c.json", JSON)],
)
def test_formats_load_identically(tmp_path: Path, name: str, content: str) -> None:
    path = tmp_path / name
    path.write_text(content)
    config = load_config(path)
    assert config.generation.seed == 1
    assert config.generation.n_sessions == 10
    assert config.logging.level == "WARNING"
    assert config.base_dir == tmp_path.resolve()


def test_empty_yaml_is_defaults(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("# nothing yet\n")
    assert load_file(path) == {}


@pytest.mark.parametrize(
    ("name", "content", "code"),
    [
        ("c.ini", "", "config_unsupported_format"),
        ("c.yaml", "a: [1,\n", "config_parse_error"),
        ("c.toml", "a = \n", "config_parse_error"),
        ("c.json", "{", "config_parse_error"),
        ("c.yaml", "- a\n- b\n", "config_invalid"),
        (
            "c.yaml",
            "!!python/object/apply:os.system ['echo hi']\n",
            "config_parse_error",
        ),
    ],
)
def test_bad_files(tmp_path: Path, name: str, content: str, code: str) -> None:
    path = tmp_path / name
    path.write_text(content)
    with pytest.raises(ConfigError) as info:
        load_file(path)
    assert info.value.code == code


def test_missing_and_unreadable_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as info:
        load_file(tmp_path / "nope.yaml")
    assert info.value.code == "config_not_found"

    binary = tmp_path / "bin.yaml"
    binary.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ConfigError) as info:
        load_file(binary)
    assert info.value.code == "config_unreadable"


def test_example_config_and_mapping_load() -> None:
    config = load_config(EXAMPLES / "minimal.yaml")
    assert config.project.name == "demo-markov"
    assert config.sessionization.tokenization == "activity_id"
    assert config.generator.params["smoothing_alpha"] == 0.1
    assert config.validation.thresholds["event_type_tvd"] == 0.25
    assert (
        config.resolve_path(config.io.mapping or "") == EXAMPLES / "demo_mapping.yaml"
    )

    mapping = FieldMapping.from_file(config.resolve_path(config.io.mapping or ""))
    assert mapping.timezone == "Europe/Paris"
    assert mapping.fields["learner_id"].hash_salt == "demo-salt-change-me"


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def test_env_overrides(caplog: pytest.LogCaptureFixture) -> None:
    environ = {
        "EDULOGGEN_SEED": " 7 ",
        "EDULOGGEN_LOG_LEVEL": "debug",
        "EDULOGGEN_OUTPUT_DIR": "out",
        "EDULOGGEN_TYPO": "x",
        "EDULOGGEN_CONFIG": "c.yaml",
        "HOME": "/home/x",
    }
    with caplog.at_level(logging.DEBUG, logger="eduloggen.config"):
        overrides = env_overrides(environ)
    assert overrides == {
        "generation.seed": 7,
        "logging.level": "DEBUG",
        "project.output_dir": "out",
    }
    assert "EDULOGGEN_TYPO" in caplog.text


def test_env_ignores_blank_and_rejects_bad_seed() -> None:
    assert env_overrides({"EDULOGGEN_SEED": "  "}) == {}
    with pytest.raises(ConfigError) as info:
        env_overrides({"EDULOGGEN_SEED": "forty-two"})
    assert info.value.context["variable"] == "EDULOGGEN_SEED"


# --------------------------------------------------------------------------
# Precedence
# --------------------------------------------------------------------------


def test_deep_merge_and_overrides() -> None:
    base = {"a": {"x": 1, "y": [1]}, "b": 2}
    merged = deep_merge(base, {"a": {"y": [2]}, "c": 3})
    assert merged == {"a": {"x": 1, "y": [2]}, "b": 2, "c": 3}
    assert base == {"a": {"x": 1, "y": [1]}, "b": 2}
    assert apply_overrides({"a": {"x": 1}}, {"a.z": 2}) == {"a": {"x": 1, "z": 2}}


@pytest.mark.parametrize("key", ["seed", ".seed", "generation.", "a.b.c"])
def test_bad_override_keys(key: str) -> None:
    with pytest.raises(ConfigError) as info:
        apply_overrides({}, {key: 1})
    assert info.value.code == "config_invalid_override"


def test_precedence_cli_over_env_over_file(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(YAML)

    file_only = resolve_config(path, environ={})
    assert file_only.generation.seed == 1

    with_env = resolve_config(path, environ={"EDULOGGEN_SEED": "2"})
    assert with_env.generation.seed == 2
    assert with_env.generation.n_sessions == 10

    with_cli = resolve_config(
        path,
        environ={"EDULOGGEN_SEED": "2"},
        overrides={"generation.seed": 3, "io.input": None},
    )
    assert with_cli.generation.seed == 3
    assert with_cli.io.input is None
    assert with_cli.logging.level == "WARNING"


def test_config_path_from_env(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(YAML)
    config = resolve_config(environ={"EDULOGGEN_CONFIG": str(path)})
    assert config.generation.seed == 1
    assert config.base_dir == tmp_path.resolve()


def test_defaults_without_any_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDULOGGEN_CONFIG", raising=False)
    monkeypatch.delenv("EDULOGGEN_SEED", raising=False)
    config = resolve_config()
    assert config.base_dir is None
    assert config.generation.seed is None


def test_overrides_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        resolve_config(environ={}, overrides={"generation.seed": "x"})
    with pytest.raises(ConfigError):
        resolve_config(environ={}, overrides={"generation.sede": 1})
    assert (
        resolve_config(
            environ={}, overrides={"generation.sede": 1}, allow_unknown=True
        ).generation.seed
        is None
    )
