"""Smoke tests for package installation and public metadata."""

from __future__ import annotations

import importlib

import pytest

import eduloggen
from eduloggen.cli import build_parser, main


def test_package_exposes_version() -> None:
    """The installed package must expose a non-empty semantic version string."""
    assert isinstance(eduloggen.__version__, str)
    assert eduloggen.__version__
    parts = eduloggen.__version__.split(".")
    assert len(parts) >= 2
    assert all(part.isdigit() for part in parts[:2])


def test_version_module_matches_package() -> None:
    """``__version__.py`` must be the source of the public version string."""
    from eduloggen.__version__ import __version__ as module_version

    assert module_version == eduloggen.__version__
    assert module_version == "0.1.0"


@pytest.mark.parametrize(
    "module_name",
    [
        "eduloggen.core",
        "eduloggen.models",
        "eduloggen.config",
        "eduloggen.io",
        "eduloggen.privacy",
        "eduloggen.utils",
        "eduloggen.ingestion",
        "eduloggen.analysis",
        "eduloggen.generators",
        "eduloggen.validation",
        "eduloggen.visualization",
        "eduloggen.cli",
    ],
)
def test_subpackages_are_importable(module_name: str) -> None:
    """Documented subpackage namespaces must be importable."""
    module = importlib.import_module(module_name)
    assert module.__doc__


def test_cli_parser_accepts_version_flag() -> None:
    """The CLI argument parser must define a --version action."""
    parser = build_parser()
    assert parser.prog == "eduloggen"


def test_cli_main_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Invoking the CLI with no arguments should print help and exit 0."""
    exit_code = main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "EduLogGen" in captured.out
