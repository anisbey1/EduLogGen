"""Logging setup for CLI processes (PRD §17).

Library code only logs through ``eduloggen.*`` loggers; the CLI attaches a
single stderr handler, plain or JSON, tagged with the run id.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Final

__all__ = ["configure_logging", "level_for"]

_HANDLER_NAME: Final = "eduloggen-cli"


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", None),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def level_for(verbose: int, quiet: bool, configured: str) -> int:
    """Pick the log level: ``--quiet`` > ``-v`` count > configured level."""
    if quiet:
        return logging.ERROR
    if verbose >= 2:
        return logging.DEBUG
    if verbose == 1:
        return logging.INFO
    return logging.getLevelNamesMapping()[configured]


def configure_logging(level: int, *, json_logs: bool, run_id: str) -> None:
    """Attach (or replace) the CLI handler on the ``eduloggen`` logger."""
    logger = logging.getLogger("eduloggen")
    for handler in list(logger.handlers):
        if handler.get_name() == _HANDLER_NAME:
            logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.addFilter(_RunIdFilter(run_id))
    handler.setFormatter(
        _JsonFormatter()
        if json_logs
        else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
