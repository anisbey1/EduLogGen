"""Allow ``python -m eduloggen``."""

from __future__ import annotations

import sys

from eduloggen.cli import main

if __name__ == "__main__":
    sys.exit(main())
