"""EduLogGen: synthetic educational interaction log framework.

EduLogGen is an open-source Python framework for analyzing, generating,
validating, and benchmarking synthetic educational interaction logs.

The package is organized into focused subpackages:

- ``core``: exceptions, constants, run context, and shared types
- ``models``: canonical events, sessions, datasets, and vocabulary
- ``config``: typed settings, file loading, and precedence rules
- ``io``: readers, writers, field mapping, ingest, and corpus directories
- ``ingestion``: compatibility alias for the ingest API in ``io``
- ``analysis``: session and statistical analysis of learner behavior
- ``generators``: statistical and learning-based synthetic generators
- ``validation``: quality, similarity, and privacy validation
- ``visualization``: plotting and exploratory visualization utilities
- ``cli``: command-line interface entry points

Public API surface will expand as generators and validators are implemented.
Until then, consumers should rely on ``__version__`` and the documented
subpackage layout.

See the project vision document (``docs/01_VISION.md``) for goals and scope.
"""

from __future__ import annotations

from eduloggen.__version__ import __version__

__all__ = ["__version__"]
