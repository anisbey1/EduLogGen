"""Command-line interface for EduLogGen.

The CLI entry point is registered as the ``eduloggen`` console script in
``pyproject.toml``. Subcommands for analysis, generation, validation, and
benchmarking will be added as features land. The current entry point reports
package metadata so installation and packaging can be verified early.
"""

from __future__ import annotations

import argparse
import sys

from eduloggen import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for the EduLogGen CLI.

    Returns:
        Configured argument parser for the ``eduloggen`` command.
    """
    parser = argparse.ArgumentParser(
        prog="eduloggen",
        description=(
            "EduLogGen: analyze, generate, validate, and benchmark "
            "synthetic educational interaction logs."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the EduLogGen command-line interface.

    Args:
        argv: Optional argument vector. When ``None``, ``sys.argv[1:]`` is used.

    Returns:
        Process exit code (``0`` on success).
    """
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
