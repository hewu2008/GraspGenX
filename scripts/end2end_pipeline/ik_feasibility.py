"""IK feasibility check using Pinocchio.

Loads the Zerith H1 URDF (assets/zerith/urdf/ZR_H1PRO-1.2.00.H.V4.3_URDF_2025.12.02.urdf),
extracts a *reduced single-arm model* rooted at ``body_yaw_link`` and ending at the
arm's ``*_end_effector_link`` (the same frame ``setArm_high``/``getHandRelative``
control), then solves inverse kinematics for a requested EEF pose. If a solution
exists within the arm's joint limits (+ a configurable margin) the pose is deemed
reachable.

Frame convention
----------------
The SDK end-effector pose ``getHandRelative`` / the composed absolute EEF target in
``move_arm_to_grasp`` is expressed relative to the arm's zero (motor-zero) frame.
Here that zero frame is taken to be ``body_yaw_link`` at the URDF zero configuration,
so a target EEF pose can be checked directly against the reduced model. Borderline
poses may be affected by SDK-vs-URDF joint-zero offsets; the check is therefore used
as a *filter*, with a safe reachability tolerance and a generous joint-limit margin.
"""

from __future__ import annotations

import io
import os
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np

try:  # pinocchio is only required for the IK path.
    import pinocchio as pin
except Exception as _e:  # pragma: no cover - degraded mode (no pinocchio)
    pin = None

from .logging_utils import get_logger

logger = get_logger(__name__)

_URDF_REL = os.path.join(
    "assets", "zerith", "urdf",
    "ZR_H1PRO-1.2.00.H.V4.3_URDF_2025.12.02.urdf",
)
# project root = two levels up from this file (scripts/end2end_pipeline/...).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_URDF_PATH = os.path.join(_PROJECT_ROOT, _URDF_REL)

# Reduced model: root link = body_yaw_link (shared by both arms), then the chain
# down to each arm's end-effector. Joint + link names per side.
_ARM_JOINT_TAIL = "_end_effector_joint"
_LINK_ROOT = "body_yaw_link"

# arm joints in order (root->leaf), same for left/right with the arm prefix.
_ARM_JOINTS = [
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_roll_joint",
    "wrist_yaw_joint",
    "wrist_pitch_joint",
]

DEFAULT_SEEDS = 6
DEFAULT_MAX_ITERS = 150
DEFAULT_RESIDUAL_TOL = 0.02     # 20 mm / 20 mrad IK convergence tolerance
                                 # (was 5e-3; widened to match the ~2 cm real arm
                                 #  execution accuracy so borderline-but-reachable
                                 #  poses near wrist joint limits are not mis-rejected)
# Joint-limit margin. KEEP AT 0.0 (DISABLED): the URDF joint limits / zero do
# not match the real robot, so URDF IK solutions sit right against the limits
# even for the working ready pose (left ARM_1=87.8 vs upper 90, right ARM_1=90.0
# exactly at upper). Enabling a non-zero margin therefore mis-rejects poses the
# real robot can reach. Near-limit hard-stops must instead be handled by FK
# calibration (verify_fk_against_sdk) or a runtime safety stop on the offending
# joint, not by an IK margin gate.
DEFAULT_JOINT_MARGIN_DEG = 0.0

_lock = threading.Lock()
_model_cache = {}   # side -> (model, data, joint_names, frame_id)
_range_cache = {}   # side -> (q_lower[7], q_upper[7])


# ---------------------------------------------------------------------------
# Reduced-URDF construction
# ---------------------------------------------------------------------------
def _read_full_urdf():
    """Return the parsed full URDF ``<robot>`` element, or None."""
    if not os.path.exists(_URDF_PATH):
        logger.error(f"[IK] URDF not found: {_URDF_PATH}")
        return None
    try:
        tree = ET.parse(_URDF_PATH)
    except Exception as e:  # pragma: no cover
        logger.error(f"[IK] Failed to parse URDF: {e}")
        return None
    return tree.getroot()


def _build_reduced_urdf(side):
    """Build a reduced single-arm URDF string for ``side`` (left/right)."""
    robot = _read_full_urdf()
    if robot is None:
        return None
    prefix = side.lower()
    if prefix not in ("left", "right"):
        logger.error(f"[IK] Unknown side '{side}'.")
        return None

    links = {link.get("name"): link for link in robot.findall("link")}
    joints = {jt.get("name"): jt for jt in robot.findall("joint")}

    arm_joints = [prefix + "_" + jn for jn in _ARM_JOINTS] + [prefix + _ARM_JOINT_TAIL]
    arm_links = [_LINK_ROOT] + [jt.find("child").get("link") for jt in
                                (joints[n] for n in arm_joints if n in joints)]

    reduced = ET.Element("robot", {"name": f"{prefix}_arm"})
    for ln in arm_links:
        if ln in links:
            reduced.append(links[ln])
    for jn in arm_joints:
        if jn in joints:
            reduced.append(joints[jn])
    return ET.tostring(reduced, encoding="unicode")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _load_model(side):
    """Load (or cache) the pinocchio model/data for ``side``."""
    with _lock:
        if side.lower() in _model_cache:
            return _model_cache[side.lower()]

        if pin is None:
            logger.error("[IK] pinocchio unavailable; IK feasibility disabled.")
            return None

        urdf_str = _build_reduced_urdf(side)
        if not urdf_str:
            return None
        try:
            model = pin.buildModelFromXML(urdf_str)
        except Exception as e:  # pragma: no cover
            logger.error(f"[IK] pinocchio build failed for {side}: {e}")
            return None

        if model.nq < 7:
            logger.error(f"[IK] Reduced {side} model has too few dof (nq={model.nq}).")
            return None

        data = model.createData()
        frame_id = model.getFrameId(f"{side.lower()}_end_effector_link")

        # The reduced model is a fixed-base serial chain: no freeflyer, nq = 7.
        # Joint id -> q index mapping (used to index q, Jacobian columns, limits).
        arm_joint_ids = [model.getJointId(f"{side.lower()}_{n}") for n in _ARM_JOINTS]
        joint_names = [f"{side.lower()}_{n}" for n in _ARM_JOINTS]
        q_indices = np.array([model.joints[j].idx_q for j in arm_joint_ids])

        lower = np.array([model.lowerPositionLimit[qi] for qi in q_indices])
        upper = np.array([model.upperPositionLimit[qi] for qi in q_indices])

        model_data = (model, data, joint_names, frame_id, arm_joint_ids, q_indices)
        _model_cache[side.lower()] = model_data
        _range_cache[side.lower()] = (lower, upper)
        logger.info(
            f"[IK] Loaded reduced {side} arm model: nq={model.nq}, "
            f"frame={model.frames[frame_id].name}"
        )
        return model_data


def _full_config(arm_q, q_indices, model):
    """Assemble the full q vector from the arm joint values (fixed-base model)."""
    q = np.zeros(model.nq)
    q[q_indices] = arm_q
    return q


# ---------------------------------------------------------------------------
# SDK <-> URDF base-frame calibration
#
# The IK model is rooted at URDF ``body_yaw_link``.  The SDK reports EEF poses
# relative to the arm's *motor-zero* frame, which need not coincide with that
# body frame.  ``SDKzero_T_body`` maps a pose in the URDF body frame into the
# SDK motor-zero frame:
#     S_T_B   so that   S_T_E = S_T_B @ B_T_E
# measured from a single (joint q, SDK EEF) pair:
#     S_T_B = S_T_E @ inv(B_T_E),  B_T_E = URDF FK(q)
# ---------------------------------------------------------------------------
_ECI = None          # cached EtherCAT_Motor_Index (lazy, so offline tests work)
_ARM_ACTION = None   # cached ArmAction (lazy)
_sdkzero_to_body = {}  # side -> S_T_B (4x4), only set after offline/store or verify()


def _lazy_sdk_imports():
    """Import the SDK enums lazily (avoids breaking offline model-only tests)."""
    global _ECI, _ARM_ACTION
    if _ECI is None:
        from lib_h1_sdk_python import ArmAction, EtherCAT_Motor_Index
        _ECI, _ARM_ACTION = EtherCAT_Motor_Index, ArmAction
    return _ECI, _ARM_ACTION


def _arm_side(side):
    return "LEFT" if str(side).lower().startswith("l") else "RIGHT"


def _read_arm_joint_q_sdk(robot, side):
    """Read the arm's 7 joint motor positions [rad] directly from the SDK."""
    eci, _ = _lazy_sdk_imports()
    prefix = _arm_side(side)
    q = []
    for i in range(1, 8):
        mid = getattr(eci, f"MOTOR_{prefix}_ARM_{i}", None)
        if mid is None:
            return None
        ok, info = robot.getMotorState(mid)
        if not ok or info is None:
            return None
        q.append(float(info.Position_Actual))
    return np.asarray(q, dtype=np.float64)


def _read_sdk_eef(robot, side):
    """Return the SDK EEF pose S_T_E (4x4, in the arm motor-zero frame)."""
    _, arm_action = _lazy_sdk_imports()
    arm = arm_action.LEFT_ARM if str(side).lower().startswith("l") else arm_action.RIGHT_ARM
    ok, arm_state = robot.getHandRelative(arm)
    if not ok or arm_state is None:
        return None
    pos = getattr(arm_state, "position", None)
    quat = getattr(arm_state, "rotation", None)
    if pos is None or quat is None:
        return None
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def _fk_eef_pose(side, q_arm):
    """URDF forward kinematics: given arm joint angles, return B_T_E (4x4).

    ``B_T_E`` is the EEF pose in the URDF ``body_yaw_link`` frame (the reduced
    model's base frame), computed for the *same* physical configuration that the
    SDK reports as ``S_T_E`` via ``getHandRelative``.
    """
    model_data = _load_model(side)
    if model_data is None or pin is None:
        return None
    model, data, _jn, frame_id, _aj, q_indices = model_data
    q = _full_config(np.asarray(q_arm, dtype=np.float64), q_indices, model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return np.asarray(data.oMf[frame_id].homogeneous, dtype=np.float64)


def compute_sdkzero_to_body(robot, side):
    """Return ``(S_T_B, S_T_E, B_T_E, q_sdk)`` from one live SDK read.

    ``S_T_B`` maps the URDF body frame into the SDK motor-zero frame:
        S_T_B = S_T_E @ inv(B_T_E)
    Returns ``(None, ...)`` for a side if any read/FK fails.
    """
    q_sdk = _read_arm_joint_q_sdk(robot, side)
    s_t_e = _read_sdk_eef(robot, side)
    if q_sdk is None or s_t_e is None:
        logger.error(f"[IK] Could not read SDK joint/EEF state for {side}.")
        return None, None, None, None
    b_t_e = _fk_eef_pose(side, q_sdk)
    if b_t_e is None:
        logger.error(f"[IK] URDF FK failed for {side}.")
        return None, s_t_e, None, q_sdk
    s_t_b = s_t_e @ np.linalg.inv(b_t_e)
    return s_t_b, s_t_e, b_t_e, q_sdk


def set_sdkzero_to_body_offset(side, T):
    """Explicitly set the SDK<->body base transform ``S_T_B`` for a side."""
    side = str(side).lower()
    _sdkzero_to_body[side] = np.asarray(T, dtype=np.float64).reshape(4, 4)


def get_sdkzero_to_body_offset(side):
    """Return the cached ``S_T_B`` for a side, or None if not set."""
    side = str(side).lower()
    return _sdkzero_to_body.get(side)


def verify_fk_against_sdk(robot, side=None, samples=3, settle_s=0.3, store=True):
    """Diagnose the SDK-vs-URDF base-frame offset across several poses.

    Reads ``(joint q, SDK EEF)`` ``samples`` times per side (pausing ``settle_s``
    between reads so the arm is static), solves ``S_T_B`` each time, and reports
    its mean + spread.  A *stable* ``S_T_B`` across poses means the two frames
    differ only by a constant base transform (safe to inject into IK); a large
    spread indicates per-joint zero / axis mismatch that a single base transform
    cannot fix.  If ``store``, the mean ``S_T_B`` is cached for IK use.

    Args:
        robot: connected H1Robot instance.
        side: 'left', 'right', or None (both).
        samples: number of reads per side.
        settle_s: pause between reads (s).
        store: whether to cache the mean offset for use by check_grasp_reachable.

    Returns:
        dict: side -> {mean_S_T_B, pos_spread_m, ang_spread_deg, n, table}
    """
    sides = ["left", "right"] if side is None else [str(side).lower()]
    results = {}
    for s in sides:
        offsets = []
        for _ in range(max(1, samples)):
            s_t_b = compute_sdkzero_to_body(robot, s)[0]
            if s_t_b is None:
                continue
            offsets.append(s_t_b)
            time.sleep(settle_s)
        if not offsets:
            logger.warning(f"[IK] verify_fk_against_sdk: no valid samples for {s}.")
            results[s] = {"n": 0, "mean_S_T_B": None}
            continue

        mean_t = np.mean(np.stack(offsets), axis=0)
        # translation / rotation spread of the sample offsets around the mean.
        from scipy.spatial.transform import Rotation as R
        pos_spread = np.max([np.linalg.norm(o[:3, 3] - mean_t[:3, 3]) for o in offsets])
        ang_spread = np.max([
            np.degrees((R.from_matrix(mean_t[:3, :3])
                        * R.from_matrix(o[:3, :3]).inv()).magnitude()) for o in offsets
        ])
        if store:
            set_sdkzero_to_body_offset(s, mean_t)
        results[s] = {
            "n": len(offsets),
            "mean_S_T_B": mean_t,
            "mean_pos": mean_t[:3, 3].tolist(),
            "mean_rotation_deg": R.from_matrix(mean_t[:3, :3]).as_euler(
                "xyz", degrees=True).tolist(),
            "pos_spread_m": float(pos_spread),
            "ang_spread_deg": float(ang_spread),
        }
        logger.info(
            f"[IK] {s}: S_T_B(pos)={results[s]['mean_pos']}, "
            f"euler_deg={results[s]['mean_rotation_deg']}, "
            f"spread: pos={pos_spread:.4f} m, ang={ang_spread:.2f} deg (n={len(offsets)})"
        )
    return results


# ---------------------------------------------------------------------------
# IK
# ---------------------------------------------------------------------------
def _solve_ik(model, data, joint_names, frame_id, q_indices, target_oM, seeds):
    """Levenberg-style damped least-squares IK over the arm joints (base fixed).

    Args:
        model, data: the pinocchio model/data.
        joint_names: arm joint names (for limits lookup).
        frame_id: EEF frame id in ``model``.
        q_indices: q-vector indices of the arm joints.
        target_oM: 4x4 target EEF pose in the model base frame.
        seeds: number of IK seeds.

    Returns (q_arm, residual_norm) of the best run, or (None, inf) if none
    converged to the joint-truncated residual tolerance.
    """
    best_q = None
    best_cost = float("inf")

    rng = np.random.default_rng(0)
    n_joints = len(q_indices)
    lo = np.array([model.lowerPositionLimit[qi] for qi in q_indices])
    hi = np.array([model.upperPositionLimit[qi] for qi in q_indices])

    candidates = [np.zeros(n_joints)]  # zero seed
    candidates.append((lo + hi) / 2.0)  # mid-range seed
    for _ in range(max(0, seeds - 2)):
        candidates.append(rng.uniform(lo, hi))

    for q0 in candidates:
        q_arm = np.clip(np.asarray(q0, dtype=np.float64), lo, hi)
        converged = False
        cost = float("inf")
        for _ in range(DEFAULT_MAX_ITERS):
            q = _full_config(q_arm, q_indices, model)
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            oMf = data.oMf[frame_id]

            err = -pin.log(oMf.inverse().homogeneous @ target_oM)
            cost = float(np.linalg.norm(err))
            if cost <= DEFAULT_RESIDUAL_TOL:
                converged = True
                break

            # Local-frame Jacobian, consistent with the local-frame log error above.
            pin.computeJointJacobians(model, data, q)
            Jf = pin.getFrameJacobian(model, data, frame_id, pin.LOCAL)
            Jarm = Jf[:, q_indices]

            lam = 0.05
            dq = np.linalg.solve(Jarm.T @ Jarm + lam * np.eye(n_joints), -Jarm.T @ err)
            q_arm = np.clip(q_arm + dq, lo, hi)
        if converged:
            q = _full_config(q_arm, q_indices, model)
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            final_err = np.linalg.norm(-pin.log(data.oMf[frame_id].inverse().homogeneous @ target_oM))
        else:
            final_err = cost
        if final_err < best_cost:
            best_cost = final_err
            best_q = q_arm.copy()

    return best_q, best_cost


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def check_eef_reachable(side, eef_pos, eef_quat,
                        seeds=DEFAULT_SEEDS,
                        joint_margin_deg=DEFAULT_JOINT_MARGIN_DEG):
    """Return (reachable, detail) for an EEF target pose in the arm zero frame.

    Args:
        side: 'left' or 'right'.
        eef_pos: 3-float position (m) of the end effector in the arm zero frame.
        eef_quat: 4-float quaternion [x, y, z, w] of the end effector.
        seeds: number of random IK seeds to try.
        joint_margin_deg: reject solutions that sit within this margin of a limit.

    Returns:
        (reachable: bool, detail: dict) where detail holds the residual, the best
        solution, its margin, or the reason a side could not be checked.
    """
    side = side.lower()
    model_data = _load_model(side)
    if model_data is None:
        return None, {"error": "model unavailable"}

    model, data, joint_names, frame_id, _arm_joint_ids, q_indices = model_data
    lower, upper = _range_cache[side]

    from scipy.spatial.transform import Rotation as R
    target = np.eye(4)
    target[:3, :3] = R.from_quat(eef_quat).as_matrix()
    target[:3, 3] = np.asarray(eef_pos, dtype=np.float64)

    q_arm, residual = _solve_ik(model, data, joint_names, frame_id, q_indices, target, seeds)

    if q_arm is None:
        return False, {"residual": residual, "reason": "no IK solution found"}

    margin = np.minimum(np.abs(q_arm - lower), np.abs(q_arm - upper))
    min_margin_rad = float(margin.min())

    if residual > DEFAULT_RESIDUAL_TOL:
        return False, {
            "residual": residual, "q_arm": q_arm.tolist(),
            "reason": f"IK residual {residual:.4f} > tol",
        }
    if min_margin_rad < np.radians(joint_margin_deg):
        return False, {
            "residual": residual, "q_arm": q_arm.tolist(),
            "min_joint_margin_deg": float(np.degrees(min_margin_rad)),
            "reason": "joint within margin of a limit",
        }
    return True, {
        "residual": residual, "q_arm": q_arm.tolist(),
        "min_joint_margin_deg": float(np.degrees(min_margin_rad)),
    }


def check_grasp_reachable(robot, side, arm, target_pos, target_quat,
                          seeds=DEFAULT_SEEDS,
                          joint_margin_deg=DEFAULT_JOINT_MARGIN_DEG):
    """Check whether an arm-relative grasp target is reachable.

    ``target_pos/target_quat`` are the SDK EEF target produced by
    ``resolve_grasp_target_hand`` (relative to the current eef). We compose it
    with the *current* arm pose (read via ``get_arm_relative_pose``) to obtain
    the absolute EEF pose in the arm zero frame, then run the IK feasibility
    check. ``arm`` is the SDK ArmAction of the executing side.

    Returns:
        (reachable, detail) as in :func:`check_eef_reachable`.
    """
    from .robot_motion import compose_relative_pose, get_arm_relative_pose
    cur_pos, cur_quat = get_arm_relative_pose(robot, arm=arm)
    if cur_pos is None:
        logger.error("[IK] Could not read current arm pose; skipping feasibility.")
        return None, {"error": "no current arm pose"}
    abs_pos, abs_quat = compose_relative_pose(cur_pos, cur_quat, target_pos, target_quat)

    # The composed absolute target is in the SDK motor-zero frame, but the IK
    # model is rooted at the URDF body_yaw_link frame. If a calibration offset
    # S_T_B has been set (verify_fk_against_sdk / set_sdkzero_to_body_offset),
    # map the target into the model base frame: target_B = inv(S_T_B) @ target_S.
    sdkzero_to_body = get_sdkzero_to_body_offset(side)
    if sdkzero_to_body is not None:
        from scipy.spatial.transform import Rotation as R
        t_abs = np.eye(4)
        t_abs[:3, :3] = R.from_quat(abs_quat).as_matrix()
        t_abs[:3, 3] = np.asarray(abs_pos, dtype=np.float64)
        t_body = np.linalg.inv(sdkzero_to_body) @ t_abs
        abs_pos = t_body[:3, 3].tolist()
        abs_quat = R.from_matrix(t_body[:3, :3]).as_quat()

    reachable, detail = check_eef_reachable(side, abs_pos, abs_quat,
                                            seeds=seeds,
                                            joint_margin_deg=joint_margin_deg)
    logger.info(
        f"[IK] {side} eef target reachable={reachable} "
        f"(pos={np.round(abs_pos, 3).tolist()}, detail={detail})"
    )
    return reachable, detail