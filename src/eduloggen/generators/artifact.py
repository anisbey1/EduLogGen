"""Model artifact directories (SAD §29A.4, ADR-009).

Layout::

    model_dir/
      manifest.json         # ids, versions, fingerprints, fitted_at
      hyperparameters.json
      vocabulary.json
      parameters.json       # family-specific fitted parameters
      DESCRIPTION.md        # human-readable summary

Everything is JSON or Markdown; nothing is pickled, so loading an artifact
never executes code. Loading verifies the model fingerprint recorded in the
manifest.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from eduloggen.core import IngestionError, PathLike, SchemaError
from eduloggen.models import GeneratorModel
from eduloggen.utils.fs import atomic_directory, write_json

__all__ = ["describe_markdown", "load_model", "save_model"]

MANIFEST: Final = "manifest.json"
_PARTS: Final = ("hyperparameters", "vocabulary", "parameters")


def save_model(
    model: GeneratorModel,
    path: PathLike,
    *,
    description: Mapping[str, Any] | None = None,
    force: bool = False,
) -> Path:
    """Write a model artifact directory.

    Args:
        model: Fitted model.
        path: Target directory.
        description: Output of ``describe()`` for ``DESCRIPTION.md``.
        force: Replace an existing artifact or empty directory.

    Returns:
        The absolute artifact path.

    Raises:
        ExportError: If the target is unsafe or exists without ``force``.
    """
    target = Path(path).expanduser().absolute()
    data = model.to_dict()
    with atomic_directory(target, marker=MANIFEST, force=force) as staging:
        write_json(staging / "hyperparameters.json", data["hyperparameters"])
        write_json(staging / "vocabulary.json", {"tokens": data["vocabulary"]})
        write_json(staging / "parameters.json", data["parameters"])
        manifest = {
            key: data[key]
            for key in (
                "generator_id",
                "generator_version",
                "eduloggen_version",
                "artifact_version",
                "training_fingerprint",
                "fitted_at",
            )
        }
        manifest["model_fingerprint"] = model.fingerprint()
        write_json(staging / MANIFEST, manifest)
        if description is not None:
            (staging / "DESCRIPTION.md").write_text(
                describe_markdown(description), encoding="utf-8"
            )
    return target


def load_model(path: PathLike) -> GeneratorModel:
    """Read a model artifact directory.

    Raises:
        IngestionError: If files are missing or unreadable, or the content
            does not match the manifest fingerprint.
        SchemaError: If the content is invalid or the artifact version is
            incompatible.
    """
    root = Path(path).expanduser().resolve()
    manifest = _read(root, MANIFEST)
    expected = manifest.pop("model_fingerprint", None)
    parts = {name: _read(root, f"{name}.json") for name in _PARTS}
    tokens = parts["vocabulary"].get("tokens")
    model = GeneratorModel.from_dict(
        {
            **manifest,
            "hyperparameters": parts["hyperparameters"],
            "vocabulary": tokens if tokens is not None else (),
            "parameters": parts["parameters"],
        }
    )
    model.check_artifact_version()
    if model.fingerprint() != expected:
        raise IngestionError(
            "model content does not match its manifest fingerprint",
            code="io_model_integrity",
            context={"path": str(root)},
        )
    return model


def describe_markdown(description: Mapping[str, Any]) -> str:
    """Render a ``describe()`` mapping as a Markdown summary."""
    lines = [f"# {description.get('generator', 'Generator')} model", ""]
    lines += ["| Property | Value |", "| -------- | ----- |"]
    for key, value in description.items():
        if key == "hyperparameters":
            continue
        lines.append(f"| {key} | {value} |")
    hyperparameters = description.get("hyperparameters") or {}
    if hyperparameters:
        lines += ["", "## Hyperparameters", "", "| Name | Value |", "| ---- | ----- |"]
        lines += [f"| {k} | {v} |" for k, v in sorted(hyperparameters.items())]
    return "\n".join(lines) + "\n"


def _read(root: Path, name: str) -> dict[str, Any]:
    file = root / name
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise IngestionError(
            f"model artifact is missing {name}",
            code="io_not_found",
            context={"path": str(file)},
        ) from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IngestionError(
            f"model artifact file {name} is not readable JSON",
            code="io_model_corrupt",
            context={"path": str(file)},
        ) from exc
    if not isinstance(data, dict):
        raise SchemaError(
            f"{name} must contain a JSON object",
            code="schema_invalid_value",
            context={"path": str(file)},
        )
    return data
