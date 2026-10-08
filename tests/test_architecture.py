"""Enforce the package dependency rules of SAD §8.2-8.3.

Each subpackage may import only the EduLogGen packages listed for it below.
Adding a dependency means updating this table (and the architecture doc).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import eduloggen

SRC = Path(eduloggen.__file__).parent

ALL = frozenset(
    {
        "core",
        "config",
        "models",
        "io",
        "ingestion",
        "analysis",
        "generators",
        "validation",
        "benchmark",
        "visualization",
        "plugins",
        "privacy",
        "utils",
    }
)

ALLOWED: dict[str, frozenset[str]] = {
    "core": frozenset(),
    "models": frozenset({"core"}),
    "utils": frozenset({"core"}),
    "config": frozenset({"core", "models"}),
    "privacy": frozenset({"core", "models", "utils"}),
    "io": frozenset({"core", "config", "models", "utils", "privacy"}),
    "ingestion": frozenset({"io"}),
    "analysis": frozenset({"core", "models", "utils"}),
    "generators": frozenset(
        {"core", "config", "models", "analysis", "utils", "privacy"}
    ),
    "validation": frozenset(
        {"core", "config", "models", "analysis", "utils", "privacy"}
    ),
    "benchmark": frozenset(
        {"core", "config", "models", "analysis", "generators", "validation", "utils"}
    ),
    "visualization": frozenset(
        {"core", "models", "analysis", "validation", "benchmark"}
    ),
    "plugins": frozenset(
        {"core", "config", "io", "analysis", "generators", "validation", "benchmark"}
    ),
    "cli": ALL,
}


def _package_of(path: Path) -> str | None:
    parts = path.relative_to(SRC).parts
    if len(parts) > 1:
        return parts[0]
    stem = path.stem
    return None if stem in {"__init__", "__version__"} else stem


def _eduloggen_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        for name in names:
            parts = name.split(".")
            if parts[0] == "eduloggen" and len(parts) > 1:
                targets.add(parts[1])
    return targets


SOURCES = sorted(SRC.rglob("*.py"))


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(SRC)))
def test_imports_follow_dependency_rules(path: Path) -> None:
    package = _package_of(path)
    if package is None:
        return
    assert package in ALLOWED, f"add {package!r} to the dependency table"
    imported = _eduloggen_imports(path) - {package, "__version__"}
    forbidden = imported - ALLOWED[package]
    assert not forbidden, f"{package} must not import {sorted(forbidden)}"


def test_generators_and_validation_stay_separate() -> None:
    """ADR-005: neither may depend on the other."""
    assert "validation" not in ALLOWED["generators"]
    assert "generators" not in ALLOWED["validation"]
    assert all("cli" not in deps for pkg, deps in ALLOWED.items() if pkg != "cli")
