from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import hardware_test.franka.franka_robot as franka_robot
from hardware_test.franka.franka_robot import POSE_KEYS, FrankaRobot, FrankaRobotConfig


class SnapshotCache:
    def __init__(self, snapshot: SimpleNamespace, calls: list[str]):
        self.snapshot = snapshot
        self.calls = calls
        self.latest_calls = 0

    def latest(self, *, max_age_s: float) -> SimpleNamespace:
        self.calls.append("state")
        self.latest_calls += 1
        assert max_age_s > 0.0
        return self.snapshot


class OrderingCamera:
    def __init__(self, calls: list[str]):
        self.calls = calls

    def read(self) -> np.ndarray:
        self.calls.append("camera")
        return np.zeros((2, 3, 3), dtype=np.uint8)


class FakeVelocitySender:
    def __init__(self, calls: list[str]):
        self.calls = calls
        self.submissions: list[dict[str, Any]] = []

    def submit(self, command: dict[str, Any]) -> None:
        self.calls.append("submit_zero")
        self.submissions.append(dict(command))

    def send_latest_once(self) -> bool:
        self.calls.append("send_latest_once")
        return True


class FakeMotionClient:
    def __init__(
        self,
        *,
        join_results: list[bool] | None = None,
        velocity_statuses: list[dict[str, Any]] | None = None,
    ):
        self.calls: list[str] = []
        self.join_results = list(join_results or [])
        self.velocity_statuses = list(velocity_statuses or [])
        self.last_velocity_status = self.velocity_statuses[-1] if self.velocity_statuses else None
        self.sender = FakeVelocitySender(self.calls)
        self.joint_calls: list[dict[str, Any]] = []

    def stop_cartesian_velocity_control(self) -> dict[str, Any]:
        self.calls.append("queued_cartesian_zero")
        return {"is_ok": 1}

    def stop_cartesian_velocity_control_direct(self) -> dict[str, Any]:
        self.calls.append("stop_cartesian_direct")
        return {"is_ok": 1}

    def stop_joint_position_control(self) -> dict[str, Any]:
        self.calls.append("stop_joint")
        return {"is_ok": 1}

    def join_motion(self, timeout_s: float = 0.0) -> bool:
        self.calls.append("join")
        assert timeout_s == 0.0
        return self.join_results.pop(0) if self.join_results else False

    def joint_position_control(
        self,
        joints: list[float],
        *,
        mode: str,
        is_async: bool,
        timeout: float,
    ) -> dict[str, Any]:
        self.calls.append("joint_position")
        self.joint_calls.append(
            {
                "joints": list(joints),
                "mode": mode,
                "is_async": is_async,
                "timeout": timeout,
            }
        )
        return {"is_ok": 1}

    def recover(self) -> dict[str, Any]:
        self.calls.append("recover")
        return {"is_ok": 1}

    def velocity_loop_status(self) -> dict[str, Any]:
        self.calls.append("velocity_status")
        if self.velocity_statuses:
            self.last_velocity_status = self.velocity_statuses.pop(0)
        if self.last_velocity_status is None:
            raise AssertionError("No fake velocity status was configured")
        return dict(self.last_velocity_status)

    def _get_zmq_sender(self) -> FakeVelocitySender:
        return self.sender


def _connected_robot(
    client: Any,
    *,
    velocity_transport: str = "http",
    cameras: dict[str, Any] | None = None,
) -> FrankaRobot:
    config = FrankaRobotConfig(
        validate_connection=False,
        state_cache_enabled=False,
        use_gripper=False,
        velocity_transport=velocity_transport,
        camera_read_mode="read",
        camera_shapes={"camera": (2, 3, 3)} if cameras else {},
    )
    robot = FrankaRobot(config, client=client, cameras=cameras)
    robot._is_connected = True
    return robot


def test_ui_motion_config_defaults_are_explicit() -> None:
    config = FrankaRobotConfig(validate_connection=False, state_cache_enabled=False)

    assert config.home_joint_tolerance_rad == 0.02
    assert config.home_stable_samples == 5
    assert config.control_transition_timeout_s == 2.0
    assert config.control_status_poll_s == 0.02


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("home_joint_tolerance_rad", 0.0, "home_joint_tolerance_rad must be positive"),
        ("home_stable_samples", 0, "home_stable_samples must be positive"),
        ("control_transition_timeout_s", 0.0, "control_transition_timeout_s must be positive"),
        ("control_status_poll_s", 0.0, "control_status_poll_s must be positive"),
    ],
)
def test_ui_motion_config_rejects_nonpositive_values(field_name: str, value: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FrankaRobotConfig(
            validate_connection=False,
            state_cache_enabled=False,
            **{field_name: value},
        )


def test_observation_sample_uses_one_snapshot_before_reading_camera() -> None:
    calls: list[str] = []
    state = {
        "joint": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "ee": [[1.0, 0.0, 0.0, 0.4], [0.0, 1.0, 0.0, 0.5], [0.0, 0.0, 1.0, 0.6]],
    }
    snapshot = SimpleNamespace(state=state, gripper_state={}, timestamp_s=123.5)
    cache = SnapshotCache(snapshot, calls)
    robot = _connected_robot(object(), cameras={"camera": OrderingCamera(calls)})
    robot._state_cache = cache

    sample = robot.get_observation_sample()

    assert cache.latest_calls == 1
    assert calls == ["state", "camera"]
    assert sample.state is state
    assert sample.state_timestamp_s == 123.5
    assert sample.observation["joint_1.pos"] == 0.1
    assert sample.observation["joint_7.pos"] == 0.7
    np.testing.assert_array_equal(sample.observation["camera"], np.zeros((2, 3, 3), dtype=np.uint8))


def test_observation_sample_is_frozen() -> None:
    state = {"joint": [0.0] * 7}
    snapshot = SimpleNamespace(state=state, gripper_state={}, timestamp_s=1.0)
    robot = _connected_robot(object())
    robot._state_cache = SnapshotCache(snapshot, [])

    sample = robot.get_observation_sample()

    with pytest.raises(FrozenInstanceError):
        sample.state_timestamp_s = 2.0


def test_get_observation_returns_atomic_sample_observation() -> None:
    robot = _connected_robot(object())
    observation = {"joint_1.pos": 0.25}
    sample_type = franka_robot.FrankaObservationSample
    robot.get_observation_sample = lambda: sample_type(observation, {"joint": [0.25] * 7}, 1.0)

    assert robot.get_observation() is observation


def test_send_zero_cartesian_velocity_exposes_safe_zero_operation() -> None:
    client = FakeMotionClient()
    robot = _connected_robot(client)

    robot.send_zero_cartesian_velocity()

    assert client.calls == ["queued_cartesian_zero"]


def test_start_home_async_uses_configured_target_without_waiting() -> None:
    client = FakeMotionClient()
    robot = _connected_robot(client)

    robot.start_home_async()

    assert client.calls == ["joint_position"]
    assert client.joint_calls == [
        {
            "joints": list(robot.config.home_joints or ()),
            "mode": "absolute",
            "is_async": True,
            "timeout": robot.config.timeout_s,
        }
    ]


def test_start_home_async_rejects_missing_home_target() -> None:
    client = FakeMotionClient()
    robot = _connected_robot(client)
    robot.config.home_joints = None

    with pytest.raises(RuntimeError, match="home_joints"):
        robot.start_home_async()

    assert client.calls == []


def test_stop_home_motion_stops_joint_control_and_polls_until_joined() -> None:
    client = FakeMotionClient(join_results=[False, True])
    robot = _connected_robot(client)
    robot.config.control_status_poll_s = 0.0001

    robot.stop_home_motion()

    assert client.calls == ["stop_joint", "join", "join"]


def test_stop_home_motion_times_out_if_join_never_completes() -> None:
    client = FakeMotionClient()
    robot = _connected_robot(client)
    robot.config.control_transition_timeout_s = 0.003
    robot.config.control_status_poll_s = 0.0005

    with pytest.raises(TimeoutError, match="motion completion"):
        robot.stop_home_motion()

    assert client.calls[0] == "stop_joint"
    assert "join" in client.calls


def test_recover_delegates_to_control_client() -> None:
    client = FakeMotionClient()
    robot = _connected_robot(client)

    assert robot.recover() is None
    assert client.calls == ["recover"]


@pytest.mark.parametrize(("join_result", "expected"), [(False, False), (True, True)])
def test_motion_complete_is_exact_nonblocking_join_result(join_result: bool, expected: bool) -> None:
    client = FakeMotionClient(join_results=[join_result])
    robot = _connected_robot(client)

    assert robot.motion_complete() is expected
    assert client.calls == ["join"]


def test_home_tolerance_uses_all_seven_joints_and_includes_boundary() -> None:
    robot = _connected_robot(FakeMotionClient())
    target = list(robot.config.home_joints or ())
    at_boundary = list(target)
    at_boundary[0] += robot.config.home_joint_tolerance_rad
    last_joint_outside = list(target)
    last_joint_outside[-1] += robot.config.home_joint_tolerance_rad * 2.0

    assert robot.is_home_state({"joint": target}) is True
    assert robot.is_home_state({"joint": at_boundary}) is True
    assert robot.is_home_state({"joint": last_joint_outside}) is False


def test_home_state_is_false_when_home_target_is_not_configured() -> None:
    robot = _connected_robot(FakeMotionClient())
    robot.config.home_joints = None

    assert robot.is_home_state({"joint": [0.0] * 7}) is False


def test_http_quiesce_uses_direct_stop_then_waits_for_motion_completion() -> None:
    client = FakeMotionClient(join_results=[False, True])
    robot = _connected_robot(client, velocity_transport="http")
    robot.config.control_status_poll_s = 0.0001

    robot.quiesce_cartesian_velocity_control()

    assert client.calls == ["stop_cartesian_direct", "join", "join"]


def test_http_quiesce_times_out_instead_of_treating_incomplete_join_as_done() -> None:
    client = FakeMotionClient()
    robot = _connected_robot(client, velocity_transport="http")
    robot.config.control_transition_timeout_s = 0.003
    robot.config.control_status_poll_s = 0.0005

    with pytest.raises(TimeoutError, match="motion completion"):
        robot.quiesce_cartesian_velocity_control()

    assert client.calls[0] == "stop_cartesian_direct"
    assert "join" in client.calls


def test_zmq_quiesce_requires_new_dispatched_inactive_status_before_direct_stop() -> None:
    client = FakeMotionClient(
        join_results=[False, True],
        velocity_statuses=[
            {"is_ok": 1, "latest_seq": 7, "dispatched_seq": 7, "motion_active": False},
            {"is_ok": 1, "latest_seq": 7, "dispatched_seq": 7, "motion_active": False},
            {"is_ok": 1, "latest_seq": 8, "dispatched_seq": 7, "motion_active": False},
            {"is_ok": 1, "latest_seq": 8, "dispatched_seq": 8, "motion_active": True},
            {"is_ok": 1, "latest_seq": 8, "dispatched_seq": 8, "motion_active": False},
        ],
    )
    robot = _connected_robot(client, velocity_transport="zmq")
    robot.config.control_status_poll_s = 0.0001

    robot.quiesce_cartesian_velocity_control()

    assert client.calls == [
        "velocity_status",
        "submit_zero",
        "send_latest_once",
        "velocity_status",
        "velocity_status",
        "velocity_status",
        "velocity_status",
        "stop_cartesian_direct",
        "join",
        "join",
    ]
    zero = client.sender.submissions[0]
    assert all(zero[key] == 0.0 for key in POSE_KEYS)
    assert zero["duration"] == robot.config.command_duration_ms
    assert zero["is_async"] == 1


def test_zmq_quiesce_times_out_on_unchanged_already_idle_status() -> None:
    unchanged_idle = {"is_ok": 1, "latest_seq": 7, "dispatched_seq": 7, "motion_active": False}
    client = FakeMotionClient(velocity_statuses=[unchanged_idle])
    robot = _connected_robot(client, velocity_transport="zmq")
    robot.config.control_transition_timeout_s = 0.003
    robot.config.control_status_poll_s = 0.0005

    with pytest.raises(TimeoutError, match="velocity zero acknowledgement"):
        robot.quiesce_cartesian_velocity_control()

    assert "submit_zero" in client.calls
    assert "send_latest_once" in client.calls
    assert "stop_cartesian_direct" not in client.calls


def test_zmq_quiesce_rejects_status_without_required_sequence() -> None:
    client = FakeMotionClient(velocity_statuses=[{"is_ok": 1, "dispatched_seq": 0, "motion_active": False}])
    robot = _connected_robot(client, velocity_transport="zmq")

    with pytest.raises(franka_robot.FrankaControlError, match="latest_seq"):
        robot.quiesce_cartesian_velocity_control()

    assert client.calls == ["velocity_status"]
