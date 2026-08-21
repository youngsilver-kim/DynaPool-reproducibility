"""Backward-compatible entry point.

Use ``python analyze.py --experiment-dir ...``. This module intentionally does
not duplicate plotting logic, so paper figures and CSV tables share one source.
"""

from analyze import main


if __name__ == "__main__":
    raise SystemExit(main())
