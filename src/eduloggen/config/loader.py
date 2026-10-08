"""Load YAML, TOML, or JSON files into plain data (SAD §16.2).

YAML is parsed with ``yaml.safe_load``, so files cannot construct arbitrary
Python objects.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, Final

import yaml

from eduloggen.config.schema import AppConfig
from eduloggen.core import ConfigError, PathLike

__all__ = ["SUPPORTED_SUFFIXES", "load_config", "load_file"]

SUPPORTED_SUFFIXES: Final = (".yaml", ".yml", ".toml", ".json")


def load_file(path: PathLike) -> dict[str, Any]:
    """Parse a YAML, TOML, or JSON file whose top level is a mapping.

    Args:
        path: File to read; the format is chosen by extension.

    Returns:
        The parsed mapping (empty for an empty YAML file).

    Raises:
        ConfigError: If the file is missing, has an unsupported extension,
            cannot be parsed, or its top level is not a mapping.
    """
    file = Path(path).expanduser()
    suffix = file.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ConfigError(
            "unsupported config file type",
            code="config_unsupported_format",
            context={"path": str(file), "supported": list(SUPPORTED_SUFFIXES)},
        )
    try:
        text = file.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(
            "config file does not exist",
            code="config_not_found",
            context={"path": str(file)},
        ) from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(
            f"cannot read config file: {type(exc).__name__}",
            code="config_unreadable",
            context={"path": str(file)},
        ) from exc

    try:
        if suffix == ".toml":
            data: Any = tomllib.loads(text)
        elif suffix == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
    except (yaml.YAMLError, tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"cannot parse config file: {exc}",
            code="config_parse_error",
            context={"path": str(file)},
        ) from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            "config file must contain a mapping at the top level",
            code="config_invalid",
            context={"path": str(file)},
        )
    return data


def load_config(path: PathLike, *, allow_unknown: bool = False) -> AppConfig:
    """Load and validate a single configuration file.

    Relative paths inside the file resolve against the file's directory.
    Environment variables and CLI overrides are not applied; use
    :func:`~eduloggen.config.resolve_config` for the full precedence chain.

    Raises:
        ConfigError: If the file cannot be loaded or is invalid.
    """
    file = Path(path).expanduser().resolve()
    return AppConfig.from_dict(
        load_file(file), allow_unknown=allow_unknown, base_dir=file.parent
    )
