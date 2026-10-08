"""Validation report (PRD §9.5, §12.3-12.4, SAD §9.10)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal

from eduloggen.core import REPORT_VERSION
from eduloggen.validation.base import MetricResult

__all__ = ["ReportStatus", "ValidationReport"]

ReportStatus = Literal["pass", "fail"]

_DISCLAIMER = (
    "Privacy indicators estimate memorisation risk; they do not prove that "
    "synthetic data is anonymous."
)


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationReport:
    """Outcome of comparing one real and one synthetic dataset.

    Attributes:
        run_id: Run identifier for correlating logs and artifacts.
        created_at: UTC creation time.
        real: Summary of the real dataset (id, fingerprint, counts).
        synthetic: Summary of the synthetic dataset, including generation
            provenance when available.
        protocol: Comparison protocol (real split, sample size, seed, id
            remapping), as required by PRD §12.4.
        metrics: Metric results in evaluation order.
        duration_s: Wall-clock time spent computing metrics.
        eduloggen_version: Framework version.
        report_version: Report layout version.
    """

    run_id: str
    created_at: datetime
    real: Mapping[str, Any] = field(hash=False)
    synthetic: Mapping[str, Any] = field(hash=False)
    protocol: Mapping[str, Any] = field(hash=False)
    metrics: tuple[MetricResult, ...]
    duration_s: float
    eduloggen_version: str
    report_version: str = REPORT_VERSION

    @property
    def status(self) -> ReportStatus:
        """``fail`` if any metric failed its threshold, else ``pass``."""
        return "fail" if any(m.status == "fail" for m in self.metrics) else "pass"

    @property
    def passed(self) -> bool:
        """Whether no metric failed."""
        return self.status == "pass"

    def metric(self, name: str) -> MetricResult:
        """Look up a metric result by name.

        Raises:
            KeyError: If the metric was not computed.
        """
        for result in self.metrics:
            if result.name == name:
                return result
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "report_version": self.report_version,
            "status": self.status,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "eduloggen_version": self.eduloggen_version,
            "duration_s": self.duration_s,
            "real": dict(self.real),
            "synthetic": dict(self.synthetic),
            "protocol": dict(self.protocol),
            "metrics": [result.to_dict() for result in self.metrics],
            "notes": [_DISCLAIMER],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ValidationReport:
        """Restore a report serialized with :meth:`to_dict`.

        Additive fields from newer minor versions are ignored.
        """
        return cls(
            run_id=data["run_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            real=MappingProxyType(dict(data["real"])),
            synthetic=MappingProxyType(dict(data["synthetic"])),
            protocol=MappingProxyType(dict(data["protocol"])),
            metrics=tuple(MetricResult.from_dict(m) for m in data["metrics"]),
            duration_s=float(data["duration_s"]),
            eduloggen_version=data["eduloggen_version"],
            report_version=data.get("report_version", REPORT_VERSION),
        )

    def to_markdown(self) -> str:
        """Human-readable summary (FR-V.5)."""
        icon = {"pass": "PASS", "fail": "FAIL", "warn": "WARN", "skip": "SKIP"}
        lines = [
            f"# Validation report: **{self.status.upper()}**",
            "",
            f"- Real: `{self.real.get('dataset_id')}` "
            f"({self.real.get('n_sessions')} sessions)",
            f"- Synthetic: `{self.synthetic.get('dataset_id')}` "
            f"({self.synthetic.get('n_sessions')} sessions)",
            f"- Real split: {self.protocol.get('real_split')}; seed "
            f"{self.protocol.get('seed')}; synthetic ids remapped: "
            f"{self.protocol.get('ids_remapped')}",
            f"- Run `{self.run_id}`, {self.duration_s:.2f}s",
            "",
            "| Metric | Category | Value | Threshold | Better | Status |",
            "| ------ | -------- | ----- | --------- | ------ | ------ |",
        ]
        for m in self.metrics:
            better = "lower" if m.direction == "lower_better" else "higher"
            lines.append(
                f"| {m.name} | {m.category} | {_fmt(m.value)} | "
                f"{_fmt(m.threshold)} | {better} | {icon.get(m.status, '-')} |"
            )
        skipped = [m for m in self.metrics if m.status == "skip"]
        if skipped:
            lines += ["", "Skipped:"]
            lines += [f"- {m.name}: {m.details.get('reason', '')}" for m in skipped]
        lines += ["", f"_{_DISCLAIMER}_"]
        return "\n".join(lines) + "\n"


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4g}"
