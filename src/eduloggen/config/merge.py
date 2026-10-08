"""Precedence merge: CLI > environment > file > defaults (ADR-007)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eduloggen.config.env import CONFIG_PATH_VAR, env_overrides
from eduloggen.config.loader import load_file
from eduloggen.config.schema import AppConfig
from eduloggen.core import ConfigError, PathLike

__all__ = ["apply_overrides", "deep_merge", "resolve_config"]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base``; nested mappings merge key by key.

    Non-mapping values (including lists) in ``override`` replace those in
    ``base``. Neither input is modified.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def apply_overrides(
    data: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply dotted-key overrides such as ``{"generation.seed": 42}``.

    Raises:
        ConfigError: If a key is not of the form ``section.key``.
    """
    nested: dict[str, Any] = {}
    for dotted, value in overrides.items():
        section, _, key = dotted.partition(".")
        if not section or not key or "." in key:
            raise ConfigError(
                "override keys must look like 'section.key'",
                code="config_invalid_override",
                context={"key": dotted},
            )
        nested.setdefault(section, {})[key] = value
    return deep_merge(data, nested)


def resolve_config(
    path: PathLike | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
    allow_unknown: bool = False,
) -> AppConfig:
    """Build the effective configuration from every source.

    Precedence, highest first: ``overrides`` (CLI flags), allowlisted
    ``EDULOGGEN_*`` environment variables, the config file, built-in
    defaults. Without ``path``, ``EDULOGGEN_CONFIG`` names the file; without
    either, only defaults, environment, and overrides apply.

    Args:
        path: Config file (YAML, TOML, or JSON).
        overrides: Dotted keys to values, e.g. ``{"generation.seed": 7}``.
            ``None`` values are skipped, so unset CLI flags can be passed.
        environ: Environment mapping; defaults to ``os.environ``.
        allow_unknown: Ignore unknown keys instead of failing.

    Returns:
        The validated configuration.

    Raises:
        ConfigError: If any source is invalid.
    """
    env = os.environ if environ is None else environ
    if path is None and env.get(CONFIG_PATH_VAR):
        path = env[CONFIG_PATH_VAR]

    data: dict[str, Any] = {}
    base_dir: Path | None = None
    if path is not None:
        file = Path(path).expanduser().resolve()
        data = load_file(file)
        base_dir = file.parent

    data = apply_overrides(data, env_overrides(env))
    if overrides:
        data = apply_overrides(
            data, {key: value for key, value in overrides.items() if value is not None}
        )
    return AppConfig.from_dict(data, allow_unknown=allow_unknown, base_dir=base_dir)
