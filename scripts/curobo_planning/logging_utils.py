"""Unified logging setup for the cuRobo planning stack.

Every module in :mod:`curobo_planning` obtains its logger through
:func:`get_logger`. A single ``curobo_planning`` root logger is configured
on first use with a consistent format (timestamp, level, module name) that
writes to stdout. Child loggers propagate to it, so each record is emitted
exactly once regardless of how many modules are imported.
"""

import logging
import sys

_LOGGER_NAME = "curobo_planning"
_FORMAT = "%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the unified ``curobo_planning`` root.

    The root logger is configured (stdout handler, ``INFO`` level, unified
    format) on the first call; later calls only fetch child loggers.
    """
    root = logging.getLogger(_LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        # Do not let records bubble up to the Python root logger. A dependency
        # (e.g. a robot SDK) may install its own handler on the root logger,
        # which would otherwise print each record a second time.
        root.propagate = False
    if name == _LOGGER_NAME or name.startswith(_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")