"""Run context shared by every stage of a pipeline run.

A :class:`RunContext` identifies a single run (CLI invocation or API workflow)
and records the facts needed for reproducibility and audit trails: run id,
global seed, start time, framework version, and platform (SAD §29A.1).
"""

from __future__ import annotations

import platform
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

from eduloggen.__version__ import __version__
from eduloggen.core.exceptions import ConfigError

__all__ = ["RunContext", "current_platform"]


def current_platform() -> Mapping[str, str]:
    """Describe the interpreter and host platform.

    Returns:
        Read-only mapping with ``python``, ``implementation``, ``os``, and
        ``machine`` keys.
    """
    return MappingProxyType(
        {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "os": platform.system(),
            "machine": platform.machine(),
        }
    )


@dataclass(frozen=True, slots=True)
class RunContext:
    """Immutable identity and provenance of a single run.

    Use :meth:`create` to build a context with a fresh run id and the current
    time; the constructor is available for restoring a recorded context.

    Attributes:
        run_id: Unique identifier correlating logs and artifacts of the run.
        seed: Global default seed, or ``None`` for non-deterministic runs.
        started_at: Timezone-aware UTC start time.
        eduloggen_version: Framework version that executed the run.
        platform: Interpreter and host details.
    """

    run_id: str
    seed: int | None
    started_at: datetime
    eduloggen_version: str = __version__
    platform: Mapping[str, str] = field(default_factory=current_platform)

    def __post_init__(self) -> None:
        """Validate fields and freeze the platform mapping.

        Raises:
            ConfigError: If the run id is empty, the seed is not a
                non-negative integer, or ``started_at`` is not UTC-aware.
        """
        if not self.run_id:
            raise ConfigError("run_id must be a non-empty string")
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ConfigError(
                "seed must be a non-negative integer or None",
                context={"seed": repr(self.seed)},
            )
        offset = self.started_at.utcoffset()
        if offset is None or offset.total_seconds() != 0:
            raise ConfigError(
                "started_at must be a timezone-aware UTC datetime",
                context={"started_at": self.started_at.isoformat()},
            )
        object.__setattr__(self, "platform", MappingProxyType(dict(self.platform)))

    @classmethod
    def create(cls, seed: int | None = None) -> RunContext:
        """Create a context for a new run.

        Args:
            seed: Global default seed, or ``None`` for non-deterministic runs.

        Returns:
            Context with a random UUID4 run id and the current UTC time.
        """
        return cls(
            run_id=uuid.uuid4().hex,
            seed=seed,
            started_at=datetime.now(UTC),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the context for run manifests and logs.

        Returns:
            JSON-serializable mapping of the context fields.
        """
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "started_at": self.started_at.isoformat(),
            "eduloggen_version": self.eduloggen_version,
            "platform": dict(self.platform),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunContext:
        """Restore a context serialized with :meth:`to_dict`.

        Args:
            data: Mapping produced by :meth:`to_dict`.

        Returns:
            The restored context.

        Raises:
            ConfigError: If required keys are missing or values are invalid.
        """
        try:
            return cls(
                run_id=data["run_id"],
                seed=data["seed"],
                started_at=datetime.fromisoformat(data["started_at"]),
                eduloggen_version=data["eduloggen_version"],
                platform=data["platform"],
            )
        except KeyError as exc:
            raise ConfigError(
                "run context is missing a required key",
                context={"key": exc.args[0]},
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"invalid run context: {exc}") from exc
