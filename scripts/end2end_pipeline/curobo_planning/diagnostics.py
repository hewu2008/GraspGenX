"""Per-candidate / per-stage failure diagnostics and the fail-closed error.

Keeps the moderate success/failure statistics (stage counts, seed success)
that the planner records without the verbose per-field summaries.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .trajectory import to_numpy


def _tensor_any(value) -> bool:
    return value is not None and bool(value.any().item())


def _mask_summary(value) -> dict[str, object] | None:
    if value is None:
        return None
    flat = to_numpy(value).astype(bool, copy=False).reshape(-1)
    return {
        "total": int(flat.size),
        "true_count": int(np.count_nonzero(flat)),
    }


def _solver_result_summary(result) -> dict[str, object]:
    if result is None:
        return {"returned": False}
    summary: dict[str, object] = {
        "returned": True,
        "success": _mask_summary(getattr(result, "success", None)),
        "feasible": _mask_summary(getattr(result, "feasible", None)),
    }
    for name in ("solve_time", "total_time"):
        value = getattr(result, name, None)
        if isinstance(value, (int, float)):
            summary[name] = float(value)
    return summary


def _aggregate_ik_attempts(
    stage_diagnostics: Mapping[str, object] | None,
) -> dict[str, object]:
    attempts = (
        list(stage_diagnostics.get("ik_attempts", ()))
        if isinstance(stage_diagnostics, Mapping)
        else []
    )
    success_counts = []
    feasible_counts = []
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            continue
        success = attempt.get("success")
        feasible = attempt.get("feasible")
        if isinstance(success, Mapping):
            success_counts.append(int(success.get("true_count", 0)))
        if isinstance(feasible, Mapping):
            feasible_counts.append(int(feasible.get("true_count", 0)))
    return {
        "attempt_count": len(attempts),
        "any_success": any(count > 0 for count in success_counts),
        "best_success_seed_count": max(success_counts, default=0),
        "best_feasible_seed_count": max(feasible_counts, default=0),
    }


def _failure_stage(result) -> str:
    if result is None:
        return "planner"
    status = str(getattr(result, "status", "") or "").lower()
    if "goalset" in status or "goal set" in status:
        return "grasp_goalset"
    if "approach" in status and "failed" in status:
        return "approach"
    if "grasp pose" in status and "failed" in status:
        return "grasp"
    if "lift" in status and "failed" in status:
        return "lift"
    if not _tensor_any(getattr(result, "approach_success", None)):
        return "approach"
    if not _tensor_any(getattr(result, "grasp_success", None)):
        return "grasp"
    return "planner"


def _failure_stage_counts(attempts: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for attempt in attempts:
        if attempt.get("success") is True:
            continue
        stage = str(attempt.get("failure_stage", "unknown"))
        counts[stage] = counts.get(stage, 0) + 1
    return counts


class CuroboPlanningError(RuntimeError):
    """Fail-closed planning error with JSON-serializable candidate diagnostics."""

    def __init__(self, message: str, diagnostics: Mapping[str, object]):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)