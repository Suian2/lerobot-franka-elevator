from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from hardware_test.franka.record_lerobot_dataset import (
    end_effector_pose,
    matrix4_to_xyz_rpy,
    measured_ee_action,
)


def _matrix_from_xyz_rpy(x: float, y: float, z: float, roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, x],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, y],
        [-sp, cp * sr, cp * cr, z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_matrix4_to_xyz_rpy_extracts_base_frame_translation_and_yaw():
    matrix = [
        [0.0, -1.0, 0.0, 0.4],
        [1.0, 0.0, 0.0, -0.2],
        [0.0, 0.0, 1.0, 0.7],
        [0.0, 0.0, 0.0, 1.0],
    ]

    pose = matrix4_to_xyz_rpy(matrix)

    assert pose == pytest.approx([0.4, -0.2, 0.7, 0.0, 0.0, math.pi / 2])


def test_matrix4_to_xyz_rpy_uses_vita_singularity_convention():
    roll = 0.7
    matrix = [
        [0.0, math.sin(roll), math.cos(roll), 0.0],
        [0.0, math.cos(roll), -math.sin(roll), 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    pose = matrix4_to_xyz_rpy(matrix)

    assert pose[3:] == pytest.approx([roll, math.pi / 2, 0.0])


def test_end_effector_pose_reads_nested_transform_from_state():
    state = {
        "ee": [
            [1.0, 0.0, 0.0, 0.8],
            [0.0, 1.0, 0.0, -0.3],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    assert end_effector_pose(state) == pytest.approx([0.8, -0.3, 0.5, 0.0, 0.0, 0.0])


def test_measured_ee_action_returns_forward_pose_delta_and_binary_gripper():
    previous = SimpleNamespace(
        state={"ee": _matrix_from_xyz_rpy(0.1, -0.2, 0.3, 0.0, 0.0, 0.0)},
        state_timestamp_s=4.0,
    )
    current = SimpleNamespace(
        state={"ee": _matrix_from_xyz_rpy(0.15, -0.1, 0.25, 0.2, -0.3, 0.4)},
        state_timestamp_s=4.2,
    )

    action = measured_ee_action(previous, current, units="delta", gripper_cmd=0.8)

    assert action == pytest.approx(
        {
            "delta_ee_pose.x": 0.05,
            "delta_ee_pose.y": 0.1,
            "delta_ee_pose.z": -0.05,
            "delta_ee_pose.rx": 0.2,
            "delta_ee_pose.ry": -0.3,
            "delta_ee_pose.rz": 0.4,
            "gripper_cmd_bin": 1.0,
        }
    )


def test_measured_ee_action_divides_pose_delta_by_measured_dt_for_velocity_units():
    previous = SimpleNamespace(
        state={"ee": _matrix_from_xyz_rpy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
        state_timestamp_s=10.0,
    )
    current = SimpleNamespace(
        state={"ee": _matrix_from_xyz_rpy(0.1, -0.2, 0.05, 0.2, -0.1, 0.3)},
        state_timestamp_s=10.25,
    )

    action = measured_ee_action(previous, current, units="velocity", gripper_cmd=0.2)

    assert action == pytest.approx(
        {
            "delta_ee_pose.x": 0.4,
            "delta_ee_pose.y": -0.8,
            "delta_ee_pose.z": 0.2,
            "delta_ee_pose.rx": 0.8,
            "delta_ee_pose.ry": -0.4,
            "delta_ee_pose.rz": 1.2,
            "gripper_cmd_bin": 0.0,
        }
    )


@pytest.mark.parametrize(
    ("previous_yaw", "current_yaw", "expected_delta"),
    [
        (math.pi - 0.1, -math.pi + 0.2, 0.3),
        (-math.pi + 0.1, math.pi - 0.2, -0.3),
    ],
)
def test_measured_ee_action_wraps_yaw_delta_across_plus_minus_pi(
    previous_yaw: float,
    current_yaw: float,
    expected_delta: float,
):
    previous = SimpleNamespace(
        state={"ee": _matrix_from_xyz_rpy(0.0, 0.0, 0.0, 0.0, 0.0, previous_yaw)},
        state_timestamp_s=1.0,
    )
    current = SimpleNamespace(
        state={"ee": _matrix_from_xyz_rpy(0.0, 0.0, 0.0, 0.0, 0.0, current_yaw)},
        state_timestamp_s=1.1,
    )

    action = measured_ee_action(previous, current, units="delta", gripper_cmd=1.0)

    assert action["delta_ee_pose.rz"] == pytest.approx(expected_delta)


def test_measured_ee_action_rejects_duplicate_state_timestamp():
    sample = SimpleNamespace(
        state={"ee": _matrix_from_xyz_rpy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
        state_timestamp_s=2.0,
    )

    with pytest.raises(ValueError, match="positive"):
        measured_ee_action(sample, sample, units="delta", gripper_cmd=0.0)


def test_measured_ee_action_rejects_unsupported_units():
    previous = SimpleNamespace(
        state={"ee": _matrix_from_xyz_rpy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
        state_timestamp_s=2.0,
    )
    current = SimpleNamespace(state=previous.state, state_timestamp_s=2.1)

    with pytest.raises(ValueError, match="Unsupported measured EE action units"):
        measured_ee_action(previous, current, units="acceleration", gripper_cmd=0.0)


@pytest.mark.parametrize(
    "matrix",
    [
        [[1.0, 0.0, 0.0, 0.0]] * 3,
        [
            [1.0, 0.0, 0.0, math.nan],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    ],
)
def test_matrix4_to_xyz_rpy_rejects_invalid_matrix(matrix):
    with pytest.raises(ValueError, match="finite 4x4"):
        matrix4_to_xyz_rpy(matrix)


def test_measured_ee_action_does_not_clip_large_measured_value():
    previous = SimpleNamespace(
        state={"ee": _matrix_from_xyz_rpy(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
        state_timestamp_s=6.0,
    )
    current = SimpleNamespace(
        state={"ee": _matrix_from_xyz_rpy(2.5, 0.0, 0.0, 0.0, 0.0, 0.0)},
        state_timestamp_s=6.1,
    )

    action = measured_ee_action(previous, current, units="delta", gripper_cmd=0.0)

    assert action["delta_ee_pose.x"] == pytest.approx(2.5)
