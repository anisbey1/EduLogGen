"""Command-line interface (PRD §14, SAD §19).

The ``eduloggen`` console script calls :func:`main`. Commands are thin
adapters over :mod:`eduloggen.api`; see ``eduloggen --help``.
"""

from __future__ import annotations

from eduloggen.cli.main import build_parser, main

__all__ = ["build_parser", "main"]
