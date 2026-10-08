"""Stable configuration fingerprints (SAD §16.4).

The fingerprint covers every section that can change results. ``project`` and
``logging`` only affect where outputs go and how runs are reported, so they
are excluded: renaming a run or raising the log level keeps the fingerprint.
Paths are hashed as written, so configs using relative paths fingerprint the
same on every machine.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final

from eduloggen.config.schema import AppConfig

__all__ = ["EXCLUDED_SECTIONS", "config_fingerprint"]

EXCLUDED_SECTIONS: Final = frozenset({"project", "logging"})


def config_fingerprint(config: AppConfig) -> str:
    """Hash the result-affecting part of a configuration.

    Returns:
        ``"sha256:"`` followed by the hex digest of canonical JSON.
    """
    payload = {
        name: section
        for name, section in config.to_dict().items()
        if name not in EXCLUDED_SECTIONS
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
