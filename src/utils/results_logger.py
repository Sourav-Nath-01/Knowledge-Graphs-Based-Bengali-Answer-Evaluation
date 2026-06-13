"""
results_logger.py
=================
DEPRECATED — superseded by results_saver.py.

This module is kept for backwards compatibility only.
Use ``src.utils.results_saver.ResultsSaver`` for all new code.
"""

import warnings
warnings.warn(
    "results_logger.ResultsLogger is deprecated. "
    "Use src.utils.results_saver.ResultsSaver instead.",
    DeprecationWarning,
    stacklevel=2,
)

from src.utils.results_saver import ResultsSaver as ResultsLogger  # noqa: F401

__all__ = ['ResultsLogger']
