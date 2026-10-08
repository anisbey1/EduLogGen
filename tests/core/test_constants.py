"""Tests for framework-wide constants and the ``core`` import boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import eduloggen.core
from eduloggen.core import constants

CORE_DIR = Path(eduloggen.core.__file__).parent


def test_default_vocabulary_matches_prd() -> None:
    assert sorted(constants.DEFAULT_EVENT_TYPES) == [
        "attempt",
        "forum_post",
        "navigate",
        "other",
        "submit",
        "video_play",
        "view",
    ]
    assert constants.OTHER_EVENT_TYPE in constants.DEFAULT_EVENT_TYPES


def test_required_fields() -> None:
    assert "timestamp" in constants.EVENT_REQUIRED_FIELDS
    assert "session_id" not in constants.EVENT_REQUIRED_FIELDS
    assert "event_sequence" in constants.SESSION_REQUIRED_FIELDS


def test_version_strings_are_major_minor() -> None:
    for version in (
        constants.SCHEMA_VERSION,
        constants.ARTIFACT_VERSION,
        constants.REPORT_VERSION,
    ):
        major, minor = version.split(".")
        assert major.isdigit()
        assert minor.isdigit()


def test_package_exports_resolve() -> None:
    for name in eduloggen.core.__all__:
        assert hasattr(eduloggen.core, name)


def test_core_imports_no_other_subpackages() -> None:
    """``core`` may import only the stdlib, itself, and the version module."""
    allowed = ("eduloggen.core", "eduloggen.__version__")
    for path in CORE_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            for name in names:
                if name.startswith("eduloggen"):
                    assert name.startswith(allowed), f"{path.name} imports {name}"
