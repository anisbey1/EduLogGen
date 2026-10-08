"""Allowlisted ``EDULOGGEN_*`` environment variables (SAD §29M.2).

Only the variables below are read. Others with the prefix are ignored and
reported at DEBUG level, so a typo never silently changes a run.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Final

from eduloggen.core import ConfigError

__all__ = ["CONFIG_PATH_VAR", "ENV_ALLOWLIST", "PREFIX", "env_overrides"]

logger = logging.getLogger(__name__)

PREFIX: Final = "EDULOGGEN_"

CONFIG_PATH_VAR: Final = "EDULOGGEN_CONFIG"
"""Config file used when no path is given explicitly."""

ENV_ALLOWLIST: Final[Mapping[str, str]] = {
    "EDULOGGEN_LOG_LEVEL": "logging.level",
    "EDULOGGEN_OUTPUT_DIR": "project.output_dir",
    "EDULOGGEN_SEED": "generation.seed",
}
"""Environment variable to dotted config key."""


def env_overrides(environ: Mapping[str, str]) -> dict[str, Any]:
    """Extract config overrides from environment variables.

    Args:
        environ: Environment mapping, usually ``os.environ``.

    Returns:
        Dotted config key to typed value, for allowlisted variables that are
        set to a non-empty value.

    Raises:
        ConfigError: If a variable holds a value of the wrong type.
    """
    overrides: dict[str, Any] = {}
    for name, value in environ.items():
        if not name.startswith(PREFIX) or name == CONFIG_PATH_VAR:
            continue
        key = ENV_ALLOWLIST.get(name)
        if key is None:
            logger.debug("ignoring unknown environment variable %s", name)
            continue
        if not value.strip():
            continue
        overrides[key] = _parse(name, key, value.strip())
    return overrides


def _parse(name: str, key: str, value: str) -> Any:
    if key == "generation.seed":
        try:
            return int(value)
        except ValueError:
            raise ConfigError(
                f"{name} must be an integer",
                code="config_invalid_value",
                context={"key": key, "variable": name},
            ) from None
    if key == "logging.level":
        return value.upper()
    return value
