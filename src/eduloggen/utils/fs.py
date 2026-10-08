"""Safe, atomic writes of output directories (SAD §25.2).

:func:`atomic_directory` stages files in a hidden sibling directory and
renames it into place only when everything was written, so a failure never
leaves a half-written output. Existing paths are replaced only with
``force=True``, only if they are empty or already hold an output of the same
kind (identified by a marker file), and never through a symbolic link.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from eduloggen.core import ExportError, PathLike

__all__ = ["atomic_directory", "write_json"]


@contextmanager
def atomic_directory(
    path: PathLike, *, marker: str, force: bool = False
) -> Iterator[Path]:
    """Stage a directory and move it to ``path`` if the block succeeds.

    Args:
        path: Final directory location.
        marker: File name that identifies a replaceable existing output
            (e.g. ``"manifest.json"``).
        force: Allow replacing an existing output or empty directory.

    Yields:
        The staging directory to write into.

    Raises:
        ExportError: If the target exists without ``force``, is a symbolic
            link, or is not a replaceable directory.
    """
    target = Path(path).expanduser().absolute()
    _check_target(target, marker, force)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        yield staging
        _swap_into_place(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write pretty, key-sorted JSON.

    Raises:
        ExportError: If the payload is not JSON-serializable or the write fails.
    """
    try:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ExportError(
            f"could not write {path.name}",
            code="export_write_failed",
            context={"path": str(path)},
        ) from exc


def _check_target(target: Path, marker: str, force: bool) -> None:
    if target.is_symlink():
        raise ExportError(
            "refusing to write through a symbolic link",
            code="export_unsafe_path",
            context={"path": str(target)},
        )
    if not target.exists():
        return
    if not force:
        raise ExportError(
            "output path already exists; pass force=True to replace it",
            code="export_exists",
            context={"path": str(target)},
        )
    if not target.is_dir() or not (
        (target / marker).is_file() or not any(target.iterdir())
    ):
        raise ExportError(
            "refusing to replace a path that is not an output or empty directory",
            code="export_unsafe_path",
            context={"path": str(target)},
        )


def _swap_into_place(staging: Path, target: Path) -> None:
    if not target.exists():
        staging.rename(target)
        return
    backup = Path(tempfile.mkdtemp(prefix=f".{target.name}.old.", dir=target.parent))
    backup.rmdir()
    target.rename(backup)
    try:
        staging.rename(target)
    except OSError:
        backup.rename(target)
        raise
    shutil.rmtree(backup, ignore_errors=True)
