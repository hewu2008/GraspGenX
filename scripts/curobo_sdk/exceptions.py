"""Exceptions raised by the curobo_sdk low-level driver."""

from __future__ import annotations

from typing import Sequence


class CuroboSdkError(RuntimeError):
    """Base class for all curobo_sdk errors."""


class RobotConnectionError(CuroboSdkError):
    """Robot is not connected or connect/deinit failed."""


class ControlModeError(CuroboSdkError):
    """Robot is not in the expected (LOW_LEVEL) control mode."""


class InitStateError(CuroboSdkError):
    """Robot init state is not Init_Complete (e.g. Error_State)."""


class LimitViolationError(CuroboSdkError):
    """A command or feedback position exceeds the vendor soft limits."""


class FeedbackError(CuroboSdkError):
    """Feedback read failed, contained NaN/Inf, or a non-zero error flag."""


class CommandError(CuroboSdkError):
    """One or more joint commands failed to send.

    ``failures`` carries per-joint human-readable reasons, aggregated into a
    single exception so the caller can inspect all failed axes at once.
    """

    def __init__(self, message: str, failures: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.failures = list(failures)


class CalibrationError(CuroboSdkError):
    """Joint calibration is malformed or invalid."""


class ConfigurationError(CuroboSdkError):
    """Static configuration (motor mapping, ID whitelist) is invalid."""