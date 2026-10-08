"""Compatibility alias for ingest helpers now living in :mod:`eduloggen.io`.

Ingestion is a workflow over the ``io`` package (ADR-017). New code should
import from :mod:`eduloggen.io`; this module re-exports the ingest entry
points for the 0.1.0 namespace.
"""

from __future__ import annotations

from eduloggen.io import FieldMapping, FieldSpec, IngestResult, QualityReport, ingest

__all__ = ["FieldMapping", "FieldSpec", "IngestResult", "QualityReport", "ingest"]
