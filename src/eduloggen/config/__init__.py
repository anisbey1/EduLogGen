"""Typed configuration with documented precedence (PRD §16, SAD §16).

Typical use::

    from eduloggen.config import resolve_config

    config = resolve_config("experiment.yaml", overrides={"generation.seed": 7})
    config.generation.seed          # 7
    config.resolve_path(config.io.input)

Precedence, highest first: explicit overrides (CLI flags), allowlisted
``EDULOGGEN_*`` environment variables, the config file, built-in defaults.
Unknown keys are rejected unless ``allow_unknown=True``.
"""

from __future__ import annotations

from eduloggen.config.env import CONFIG_PATH_VAR, ENV_ALLOWLIST, env_overrides
from eduloggen.config.fingerprint import config_fingerprint
from eduloggen.config.loader import SUPPORTED_SUFFIXES, load_config, load_file
from eduloggen.config.merge import apply_overrides, deep_merge, resolve_config
from eduloggen.config.schema import (
    AnalysisConfig,
    AppConfig,
    BenchmarkConfig,
    GenerationConfig,
    GeneratorConfig,
    IoConfig,
    LoggingConfig,
    PrivacyConfig,
    ProjectConfig,
    SessionizationConfig,
    ValidationConfig,
    VisualizationConfig,
)

__all__ = [
    "CONFIG_PATH_VAR",
    "ENV_ALLOWLIST",
    "SUPPORTED_SUFFIXES",
    "AnalysisConfig",
    "AppConfig",
    "BenchmarkConfig",
    "GenerationConfig",
    "GeneratorConfig",
    "IoConfig",
    "LoggingConfig",
    "PrivacyConfig",
    "ProjectConfig",
    "SessionizationConfig",
    "ValidationConfig",
    "VisualizationConfig",
    "apply_overrides",
    "config_fingerprint",
    "deep_merge",
    "env_overrides",
    "load_config",
    "load_file",
    "resolve_config",
]
