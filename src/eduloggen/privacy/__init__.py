"""Privacy controls applied across the pipeline (PRD §19, SAD §29J).

Default generation remaps learner, session, and event ids (PR-3), and
configured metadata keys can be stripped before export. These controls reduce
risk; they do not guarantee anonymity. Residual risk is measured by the
privacy indicators in :mod:`eduloggen.validation`.
"""

from __future__ import annotations

from eduloggen.privacy.policies import (
    IdStrategy,
    apply_id_strategy,
    remap_ids,
    strip_metadata,
)

__all__ = ["IdStrategy", "apply_id_strategy", "remap_ids", "strip_metadata"]
