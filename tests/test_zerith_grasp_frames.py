"""Frame-contract tests for the legacy Zerith Cartesian grasp path."""

from __future__ import annotations

import importlib
import pdb
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _import_pipeline_modules(monkeypatch):
    """Import the pipeline without requiring the real robot SDK binary."""

    sdk = types.ModuleType("lib_h1_sdk_python")

    class ArmAction:
        LEFT_ARM = "left"
        RIGHT_ARM = "right"

    class EtherCATMotorIndex:
        MOTOR_LEFT_ARM_8 = "left_gripper"

    class Pose:
        pass

    sdk.ArmAction = ArmAction
    sdk.EtherCAT_Motor_Index = EtherCATMotorIndex
    sdk.ArmPose = Pose
    sdk.ArmEndPose = Pose
    sdk.Motor_Control = Pose
    monkeypatch.setitem(sys.modules, "lib_h1_sdk_python", sdk)

    # Match scripts/end2end_grasp_pipeline.py: when launched directly it puts
    # <repo>/scripts on sys.path and imports the package as end2end_pipeline.
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))

    for name in (
        "end2end_pipeline.config",
        "end2end_pipeline.robot_motion",
        "end2end_pipeline.grasp_executor",
    ):
        sys.modules.pop(name, None)

    config = importlib.import_module("end2end_pipeline.config")
    motion = importlib.import_module("end2end_pipeline.robot_motion")
    executor = importlib.import_module("end2end_pipeline.grasp_executor")
    return config, motion, executor


def _transform(position, quaternion):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    transform[:3, 3] = position
    return transform


def test_grasp_base_is_converted_to_sdk_end_effector(monkeypatch):
    config, _motion, executor = _import_pipeline_modules(monkeypatch)

    cam_pos = np.array([0.01, -0.02, 0.03])
    cam_quat = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_quat()
    arm_pos = np.array([-0.1, 0.04, 0.3])
    arm_quat = Rotation.from_euler("xyz", [-0.2, 0.1, 0.4]).as_quat()
    camera_T_grasp = _transform(
        [0.12, -0.06, 0.55],
        Rotation.from_euler("xyz", [0.3, -0.1, -0.2]).as_quat(),
    )

    target_pos, target_quat = executor.calculate_target_relative_pose(
        cam_pos, cam_quat, arm_pos, arm_quat, camera_T_grasp
    )

    head_zero_T_camera = _transform(cam_pos, cam_quat)
    chassis_T_head_zero = _transform(
        [0.2194, 0.0325, 0.6075],
        Rotation.from_euler("xyz", [-1.7802, 0.0, -1.5708]).as_quat(),
    )
    arm_zero_T_chassis = _transform(
        [-0.5743, -0.1800, -0.1208], [0.0, 0.0, 0.0, 1.0]
    )
    arm_zero_T_current_eef = _transform(arm_pos, arm_quat)
    grasp_T_wrist = np.eye(4)
    grasp_T_wrist[:3, :3] = np.array(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
    )
    wrist_T_sdk_eef = np.eye(4)
    wrist_T_sdk_eef[:3, 3] = config.WRIST_TO_SDK_EEF_OFFSET_M

    expected = (
        np.linalg.inv(arm_zero_T_current_eef)
        @ arm_zero_T_chassis
        @ chassis_T_head_zero
        @ head_zero_T_camera
        @ camera_T_grasp
        @ grasp_T_wrist
        @ wrist_T_sdk_eef
    )
    actual = _transform(target_pos, target_quat)
    np.testing.assert_allclose(actual, expected, atol=1e-9)

    without_eef_offset = expected @ np.linalg.inv(wrist_T_sdk_eef)
    offset = expected[:3, 3] - without_eef_offset[:3, 3]
    np.testing.assert_allclose(np.linalg.norm(offset), 0.1435, atol=1e-9)
    np.testing.assert_allclose(
        offset,
        expected[:3, :3] @ np.array([0.1435, 0.0, 0.0]),
        atol=1e-9,
    )


def test_sdk_eef_offset_matches_zerith_urdf(monkeypatch):
    config, _motion, _executor = _import_pipeline_modules(monkeypatch)
    urdf_path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "zerith"
        / "urdf"
        / "left_gripper.urdf"
    )
    root = ET.parse(urdf_path).getroot()
    joint = root.find("./joint[@name='left_end_effector_joint']")
    assert joint is not None
    origin = joint.find("origin")
    assert origin is not None
    urdf_offset = np.fromstring(origin.attrib["xyz"], sep=" ")
    np.testing.assert_allclose(
        config.WRIST_TO_SDK_EEF_OFFSET_M, urdf_offset, atol=1e-12
    )


def test_relative_target_composition_rotates_translation(monkeypatch):
    _config, motion, _executor = _import_pipeline_modules(monkeypatch)

    start_xyz = [0.1, -0.2, 0.3]
    start_quat = Rotation.from_euler("z", 90.0, degrees=True).as_quat()
    relative_xyz = [0.2, 0.0, -0.05]
    relative_quat = Rotation.from_euler("y", 30.0, degrees=True).as_quat()

    absolute_xyz, absolute_quat = motion.compose_relative_pose(
        start_xyz, start_quat, relative_xyz, relative_quat
    )

    np.testing.assert_allclose(absolute_xyz, [0.1, 0.0, 0.25], atol=1e-9)
    expected_rotation = Rotation.from_quat(start_quat) * Rotation.from_quat(
        relative_quat
    )
    np.testing.assert_allclose(
        Rotation.from_quat(absolute_quat).as_matrix(),
        expected_rotation.as_matrix(),
        atol=1e-9,
    )


def test_move_arm_to_grasp_dispatches_composed_sdk_target(monkeypatch):
    _config, motion, _executor = _import_pipeline_modules(monkeypatch)

    start_xyz = np.array([0.1, -0.2, 0.3])
    start_quat = Rotation.from_euler("z", 90.0, degrees=True).as_quat()
    relative_xyz = np.array([0.2, 0.0, -0.05])
    relative_quat = Rotation.from_euler("y", 30.0, degrees=True).as_quat()
    expected_xyz, expected_quat = motion.compose_relative_pose(
        start_xyz, start_quat, relative_xyz, relative_quat
    )

    state = types.SimpleNamespace(position=start_xyz, rotation=start_quat)
    robot = types.SimpleNamespace(getHandRelative=lambda _arm: (True, state))
    dispatched = []

    def record_dispatch(*args):
        dispatched.append(args)

    monkeypatch.setattr(motion, "_move_arm_to_pose", record_dispatch)
    monkeypatch.setattr(motion.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(pdb, "set_trace", lambda: None)

    motion.move_arm_to_grasp(robot, relative_xyz, relative_quat)

    assert len(dispatched) == 3
    approach = Rotation.from_quat(expected_quat).as_matrix()[:, 0]
    np.testing.assert_allclose(dispatched[0][4], expected_xyz - 0.10 * approach)
    np.testing.assert_allclose(dispatched[1][5], expected_quat)
    np.testing.assert_allclose(dispatched[2][4], expected_xyz)
    np.testing.assert_allclose(dispatched[2][5], expected_quat)
