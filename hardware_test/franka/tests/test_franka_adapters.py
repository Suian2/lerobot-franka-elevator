from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

import hardware_test.franka.franka_spacemouse_teleop as franka_spacemouse_teleop
import hardware_test.franka.run_record as run_record
import hardware_test.franka.run_record_ui as run_record_ui
import lerobot.datasets as lerobot_datasets
from hardware_test.franka.defaults import DEFAULT_CONTROL_HOST

from hardware_test.franka.franka_robot import FrankaControlError, FrankaRobot, FrankaRobotConfig
from hardware_test.franka.franka_spacemouse_teleop import (
    SPNAV_EVENT_MOTION,
    FrankaSpaceMouseTeleop,
    FrankaSpaceMouseTeleopConfig,
    SpacenavReader,
)
from hardware_test.franka.record_lerobot_dataset import (
    build_lerobot_features,
    create_lerobot_dataset,
    make_lerobot_frame,
    record_lerobot_episode,
)
from hardware_test.franka.run_record import (
    build_arg_parser,
    build_camera_configs,
    build_robot_config,
    build_teleop_config,
    main as run_record_main,
)
from hardware_test.franka.state_cache import StaleFrankaStateError


class FakeCamera:
    def __init__(self, frame: np.ndarray, *, stale: bool = False):
        self.frame = frame
        self.stale = stale
        self.connected = False
        self.disconnect_calls = 0
        self.latest_calls = []

    def connect(self):
        self.connected = True

    def read(self):
        return self.frame

    def read_latest(self, max_age_ms: int = 500):
        self.latest_calls.append(max_age_ms)
        if self.stale:
            raise TimeoutError("latest frame is stale")
        return self.frame

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False


class FakeLatestCamera:
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.connected = False
        self.latest_calls = []

    def connect(self):
        self.connected = True

    def latest(self, *, max_age_ms: int = 500, timeout_ms: int = 50):
        self.latest_calls.append((max_age_ms, timeout_ms))
        return SimpleNamespace(image=self.frame, image_age_ms=12.5, dropped_frame_count=0)


class FakeFrankaClient:
    def __init__(self):
        self.cartesian_calls = []
        self.joint_calls = []
        self.gripper_calls = []
        self.get_curr_calls = 0
        self.gripper_state_calls = 0
        self.close_calls = 0
        self.state = {
            "joint": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "gripper_width": 0.0425,
        }

    def get_curr(self):
        self.get_curr_calls += 1
        return self.state

    def gripper_get_state(self):
        self.gripper_state_calls += 1
        return {"width": self.state["gripper_width"]}

    def cartesian_velocity_control(self, command):
        self.cartesian_calls.append(dict(command))
        return {"is_ok": 1}

    def stop_cartesian_velocity_control(self):
        self.cartesian_calls.append({"x": 0.0, "y": 0.0, "z": 0.0, "R": 0.0, "P": 0.0, "Y": 0.0})
        return {"is_ok": 1}

    def joint_position_control(self, joints, mode="absolute", timeout=None):
        self.joint_calls.append((list(joints), mode, timeout))
        return {"is_ok": 1}

    def gripper_open(self):
        self.gripper_calls.append("open")

    def gripper_close(self):
        self.gripper_calls.append("close")

    def close(self):
        self.close_calls += 1


class FakeSpaceMouseReader:
    def __init__(self, samples):
        self.samples = list(samples)
        self.closed = False

    def open(self):
        return "fake-spacemouse"

    def get_action(self):
        if self.samples:
            action, buttons = self.samples.pop(0)
        else:
            action, buttons = np.zeros(6), [0, 0]
        return np.asarray(action, dtype=np.float64), list(buttons)

    def close(self):
        self.closed = True


class FakeSpnavLibrary:
    def __init__(self, events):
        self.events = list(events)
        self.opened = False
        self.closed = False

    def open(self):
        self.opened = True

    def poll_event(self):
        if not self.events:
            return 0, SimpleNamespace()
        return 1, self.events.pop(0)

    def close(self):
        self.closed = True


class FlakyStateClient(FakeFrankaClient):
    def __init__(self, *, failures_before_success: int = 0, gripper_fails: bool = False):
        super().__init__()
        self.failures_before_success = failures_before_success
        self.gripper_fails = gripper_fails
        self.timeouts_seen = []

    def get_curr(self, timeout=None):
        self.timeouts_seen.append(timeout)
        self.get_curr_calls += 1
        if self.get_curr_calls <= self.failures_before_success:
            raise TimeoutError("state timeout")
        return self.state

    def gripper_get_state(self, timeout=None):
        self.gripper_state_calls += 1
        if self.gripper_fails:
            raise TimeoutError("gripper timeout")
        return {"width": self.state["gripper_width"]}


class FakeRemoteRobot:
    name = "fake_remote_franka"
    observation_features = {"joint_1.pos": float}
    action_features = {"delta_ee_pose.x": float}

    def __init__(self, failures_before_observation: int):
        self.failures_before_observation = failures_before_observation
        self.get_observation_calls = 0
        self.sent_actions = []
        self.diagnostics = {"state_age_ms": 11.0}

    def get_observation(self):
        self.get_observation_calls += 1
        if self.get_observation_calls <= self.failures_before_observation:
            raise StaleFrankaStateError("state is stale")
        return {"joint_1.pos": 0.1}

    def send_action(self, action):
        self.sent_actions.append(dict(action))
        return dict(action)


class FakeRemoteTeleop:
    action_features = {"delta_ee_pose.x": float}

    def get_action(self):
        return {"delta_ee_pose.x": 0.0}


class FakeRemoteDataset:
    def __init__(self, features):
        self.features = features
        self.frames = []

    def add_frame(self, frame):
        self.frames.append(frame)


def test_spacemouse_teleop_applies_axis_mapping_scaling_and_signs():
    teleop = FrankaSpaceMouseTeleop(
        FrankaSpaceMouseTeleopConfig(
            pose_scaler=(0.01, 0.1),
            channel_signs=(-1, 1, 1, 1, 1, 1),
            linear_axis_map=("z", "y", "x"),
            angular_axis_map=("z", "x", "y"),
            angular_output_signs=(-1, 1, -1),
        ),
        reader=FakeSpaceMouseReader([([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0, 0])]),
    )

    teleop.connect()
    action = teleop.get_action()

    np.testing.assert_allclose(
        [action[key] for key in DELTA_EE_ACTION_KEYS],
        [0.003, 0.002, -0.001, -0.06, 0.04, -0.05],
    )
    assert action["gripper_cmd_bin"] == 1.0
    assert action["reset_requested"] is False


@pytest.mark.parametrize(("raw_y", "expected_robot_z"), [(-500, -1.0), (500, 1.0)])
def test_spacemouse_teleop_matches_vita_spnav_transform_for_robot_z(monkeypatch, raw_y, expected_robot_z):
    event = SimpleNamespace(
        type=SPNAV_EVENT_MOTION,
        motion=SimpleNamespace(x=0, y=raw_y, z=0, rx=0, ry=0, rz=0),
    )
    library = FakeSpnavLibrary([event])
    monkeypatch.setattr(franka_spacemouse_teleop, "_SpnavLibrary", lambda _path=None: library)
    teleop = FrankaSpaceMouseTeleop(
        FrankaSpaceMouseTeleopConfig(pose_scaler=(1.0, 1.0), use_gripper=False),
        reader=SpacenavReader(axis_scale=500.0, deadband=0.0),
    )

    teleop.connect()
    action = teleop.get_action()
    teleop.disconnect()

    assert action["delta_ee_pose.x"] == 0.0
    assert action["delta_ee_pose.y"] == 0.0
    assert action["delta_ee_pose.z"] == expected_robot_z
    assert library.opened is True
    assert library.closed is True


def test_spacemouse_teleop_toggles_gripper_and_latches_reset_on_rising_edges():
    reader = FakeSpaceMouseReader(
        [
            ([0, 0, 0, 0, 0, 0], [0, 0]),
            ([0, 0, 0, 0, 0, 0], [1, 0]),
            ([0, 0, 0, 0, 0, 0], [1, 0]),
            ([0, 0, 0, 0, 0, 0], [0, 0]),
            ([0, 0, 0, 0, 0, 0], [0, 1]),
        ]
    )
    teleop = FrankaSpaceMouseTeleop(FrankaSpaceMouseTeleopConfig(), reader=reader)

    teleop.connect()

    assert teleop.get_action()["gripper_cmd_bin"] == 1.0
    assert teleop.get_action()["gripper_cmd_bin"] == 0.0
    assert teleop.get_action()["gripper_cmd_bin"] == 0.0
    assert teleop.get_action()["reset_requested"] is False
    assert teleop.get_action()["reset_requested"] is True


def test_franka_robot_features_and_observation_include_joints_gripper_and_l515_rgb():
    frame = np.zeros((240, 424, 3), dtype=np.uint8)
    robot = FrankaRobot(
        FrankaRobotConfig(
            id="franka-test",
            camera_shapes={"l515": (240, 424, 3)},
            validate_connection=True,
            state_cache_enabled=False,
        ),
        client=FakeFrankaClient(),
        cameras={"l515": FakeCamera(frame)},
    )

    assert robot.observation_features["joint_1.pos"] is float
    assert robot.observation_features["joint_7.pos"] is float
    assert robot.observation_features["gripper_width_norm"] is float
    assert robot.observation_features["l515"] == (240, 424, 3)
    assert robot.action_features["delta_ee_pose.x"] is float
    assert robot.action_features["gripper_cmd_bin"] is float

    robot.connect()
    obs = robot.get_observation()

    assert obs["joint_1.pos"] == 0.1
    assert obs["joint_7.pos"] == 0.7
    assert obs["gripper_width_norm"] == 0.5
    np.testing.assert_array_equal(obs["l515"], frame)
    assert robot.cameras["l515"].latest_calls == [250]


def test_franka_robot_rejects_stale_l515_frame_instead_of_recording_old_image():
    robot = FrankaRobot(
        FrankaRobotConfig(
            camera_shapes={"l515": (240, 424, 3)},
            camera_read_mode="latest",
            max_camera_age_s=0.1,
            validate_connection=False,
            state_cache_enabled=False,
        ),
        client=FakeFrankaClient(),
        cameras={"l515": FakeCamera(np.zeros((240, 424, 3), dtype=np.uint8), stale=True)},
    )
    robot.connect()

    try:
        robot.get_observation()
    except TimeoutError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("stale camera frame should stop observation collection")


def test_franka_robot_passes_async_timeout_to_latest_camera_client():
    frame = np.zeros((240, 424, 3), dtype=np.uint8)
    camera = FakeLatestCamera(frame)
    robot = FrankaRobot(
        FrankaRobotConfig(
            camera_shapes={"l515": (240, 424, 3)},
            camera_read_mode="latest",
            max_camera_age_s=0.25,
            camera_async_timeout_s=0.2,
            validate_connection=False,
            state_cache_enabled=False,
        ),
        client=FakeFrankaClient(),
        cameras={"l515": camera},
    )
    robot.connect()

    obs = robot.get_observation()

    np.testing.assert_array_equal(obs["l515"], frame)
    assert camera.latest_calls == [(250, 200)]
    assert robot.diagnostics["l515_image_age_ms"] == 12.5


def test_franka_robot_disconnects_cameras_when_connect_fails_after_camera_connect():
    frame = np.zeros((240, 424, 3), dtype=np.uint8)
    camera = FakeCamera(frame)
    robot = FrankaRobot(
        FrankaRobotConfig(
            camera_shapes={"l515": (240, 424, 3)},
            state_cache_enabled=True,
            state_timeout_s=0.01,
            validate_connection=False,
        ),
        client=FlakyStateClient(failures_before_success=1),
        cameras={"l515": camera},
    )

    with pytest.raises(TimeoutError):
        robot.connect()

    assert camera.disconnect_calls == 1
    assert camera.connected is False
    assert robot.is_connected is False


def test_franka_robot_uses_background_state_cache_after_connect():
    client = FakeFrankaClient()
    robot = FrankaRobot(
        FrankaRobotConfig(
            camera_shapes={},
            validate_connection=False,
            state_cache_enabled=True,
            state_poll_hz=1.0,
            max_state_age_s=5.0,
        ),
        client=client,
    )
    robot.connect()
    calls_after_connect = client.get_curr_calls
    assert calls_after_connect == 1

    obs = robot.get_observation()

    assert obs["joint_1.pos"] == 0.1
    assert client.get_curr_calls == calls_after_connect
    robot.disconnect()


def test_franka_robot_sends_delta_pose_as_latest_cartesian_velocity_and_gripper_command():
    client = FakeFrankaClient()
    robot = FrankaRobot(
        FrankaRobotConfig(
            action_mode="delta_ee_pose",
            cartesian_action_units="delta",
            control_hz=15.0,
            validate_connection=False,
            state_cache_enabled=False,
            command_duration_ms=300,
        ),
        client=client,
    )
    robot.connect()

    sent = robot.send_action(
        {
            "delta_ee_pose.x": 1.0 / 300.0,
            "delta_ee_pose.y": 0.0,
            "delta_ee_pose.z": -1.0 / 600.0,
            "delta_ee_pose.rx": 0.0,
            "delta_ee_pose.ry": 1.0 / 37.5,
            "delta_ee_pose.rz": 0.0,
            "gripper_cmd_bin": 0.0,
        }
    )

    assert sent["delta_ee_pose.x"] == 1.0 / 300.0
    assert client.cartesian_calls[-1] == {
        "x": 0.05,
        "y": 0.0,
        "z": -0.025,
        "R": 0.0,
        "P": 0.4,
        "Y": 0.0,
        "duration": 300,
        "is_async": 1,
    }
    assert client.gripper_calls == ["close"]

    robot.disconnect()
    assert client.cartesian_calls[-1] == {"x": 0.0, "y": 0.0, "z": 0.0, "R": 0.0, "P": 0.0, "Y": 0.0}
    assert client.close_calls == 1


def test_franka_robot_returns_clipped_sent_action_for_dataset_recording():
    client = FakeFrankaClient()
    robot = FrankaRobot(
        FrankaRobotConfig(
            action_mode="delta_ee_pose",
            cartesian_action_units="delta",
            control_hz=10.0,
            max_linear_velocity=0.05,
            max_angular_velocity=0.4,
            validate_connection=False,
            state_cache_enabled=False,
        ),
        client=client,
    )
    robot.connect()

    sent = robot.send_action(
        {
            "delta_ee_pose.x": 1.0,
            "delta_ee_pose.y": 0.0,
            "delta_ee_pose.z": 0.0,
            "delta_ee_pose.rx": 0.0,
            "delta_ee_pose.ry": 1.0,
            "delta_ee_pose.rz": 0.0,
            "gripper_cmd_bin": 0.0,
        }
    )

    assert sent["delta_ee_pose.x"] == 0.005
    assert sent["delta_ee_pose.ry"] == 0.04
    assert sent["gripper_cmd_bin"] == 0.0
    assert client.cartesian_calls[-1]["x"] == 0.05
    assert client.cartesian_calls[-1]["P"] == 0.4


def test_franka_robot_joint_mode_sends_seven_joint_targets():
    client = FakeFrankaClient()
    robot = FrankaRobot(
        FrankaRobotConfig(action_mode="joint", validate_connection=False, state_cache_enabled=False),
        client=client,
    )
    robot.connect()

    action = {f"joint_{idx}.pos": idx * 0.1 for idx in range(1, 8)}
    action["gripper_cmd_bin"] = 1.0
    sent = robot.send_action(action)

    assert sent == action
    assert client.joint_calls == [
        ([0.1, 0.2, 0.30000000000000004, 0.4, 0.5, 0.6000000000000001, 0.7000000000000001], "absolute", None)
    ]
    assert client.gripper_calls == []


def test_lerobot_recording_helpers_build_features_and_frame_for_direct_dataset_write():
    robot = FrankaRobot(
        FrankaRobotConfig(
            camera_shapes={"l515": (240, 424, 3)},
            validate_connection=False,
            state_cache_enabled=False,
        ),
        client=FakeFrankaClient(),
        cameras={"l515": FakeCamera(np.ones((240, 424, 3), dtype=np.uint8))},
    )
    teleop = FrankaSpaceMouseTeleop(FrankaSpaceMouseTeleopConfig(), reader=FakeSpaceMouseReader([]))

    features = build_lerobot_features(robot, teleop, use_videos=True)

    assert features["observation.state"]["shape"] == (8,)
    assert features["observation.images.l515"]["dtype"] == "video"
    assert features["action"]["shape"] == (7,)
    assert features["action"]["names"] == [
        "delta_ee_pose.x",
        "delta_ee_pose.y",
        "delta_ee_pose.z",
        "delta_ee_pose.rx",
        "delta_ee_pose.ry",
        "delta_ee_pose.rz",
        "gripper_cmd_bin",
    ]

    obs = {
        **{f"joint_{idx}.pos": float(idx) for idx in range(1, 8)},
        "gripper_width_norm": 0.5,
        "l515": np.ones((240, 424, 3), dtype=np.uint8),
    }
    action = dict.fromkeys(DELTA_EE_ACTION_KEYS, 0.0)
    action["gripper_cmd_bin"] = 1.0
    frame = make_lerobot_frame(features, obs, action, task="pick")

    assert frame["task"] == "pick"
    assert frame["observation.state"].shape == (8,)
    assert frame["action"].shape == (7,)
    np.testing.assert_array_equal(frame["observation.images.l515"], obs["l515"])



def test_create_lerobot_dataset_discards_empty_initialized_root(tmp_path, monkeypatch):
    root = tmp_path / "franka_l515_smoke"
    info_path = root / "meta" / "info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text(
        json.dumps(
            {
                "total_episodes": 0,
                "total_frames": 0,
                "total_tasks": 0,
            }
        )
    )

    class FakeLeRobotDataset:
        @classmethod
        def create(cls, **kwargs):
            assert not Path(kwargs["root"]).exists()
            return kwargs

    monkeypatch.setattr(lerobot_datasets, "LeRobotDataset", FakeLeRobotDataset)

    robot = FrankaRobot(
        FrankaRobotConfig(validate_connection=False, state_cache_enabled=False),
        client=FakeFrankaClient(),
    )
    teleop = FrankaSpaceMouseTeleop(FrankaSpaceMouseTeleopConfig(), reader=FakeSpaceMouseReader([]))

    dataset = create_lerobot_dataset(
        repo_id="local/franka_l515_smoke",
        root=str(root),
        fps=15,
        robot=robot,
        teleop=teleop,
        task="Franka SpaceMouse teleoperation",
    )

    assert dataset["root"] == str(root)


def test_record_lerobot_episode_retries_transient_stale_state_without_sending_action():
    robot = FakeRemoteRobot(failures_before_observation=2)
    teleop = FakeRemoteTeleop()
    features = build_lerobot_features(robot, teleop, use_videos=False)
    dataset = FakeRemoteDataset(features)

    frames = record_lerobot_episode(
        robot=robot,
        teleop=teleop,
        dataset=dataset,
        fps=30,
        duration_s=1 / 30,
        task="pick",
    )

    assert frames == 1
    assert robot.get_observation_calls == 3
    assert robot.sent_actions == [{"delta_ee_pose.x": 0.0}]
    assert len(dataset.frames) == 1


def test_record_lerobot_episode_waits_through_longer_state_cache_stale_burst():
    robot = FakeRemoteRobot(failures_before_observation=20)
    teleop = FakeRemoteTeleop()
    features = build_lerobot_features(robot, teleop, use_videos=False)
    dataset = FakeRemoteDataset(features)

    frames = record_lerobot_episode(
        robot=robot,
        teleop=teleop,
        dataset=dataset,
        fps=30,
        duration_s=1 / 30,
        task="pick",
        state_retry_sleep_s=0.0,
    )

    assert frames == 1
    assert robot.get_observation_calls == 21
    assert len(robot.sent_actions) == 1


def test_recording_keeps_episode_after_robot_action_fault(capsys):
    stop_event = Event()

    class FaultingRobot(FakeRemoteRobot):
        def __init__(self):
            super().__init__(failures_before_observation=0)
            self.send_attempts = 0

        def get_observation(self):
            observation = super().get_observation()
            if self.get_observation_calls == 3:
                stop_event.set()
            return observation

        def send_action(self, action):
            self.send_attempts += 1
            if self.send_attempts == 2:
                raise FrankaControlError("Franka fault after button press")
            return super().send_action(action)

    class MovingTeleop(FakeRemoteTeleop):
        def get_action(self):
            return {"delta_ee_pose.x": 0.1}

    robot = FaultingRobot()
    teleop = MovingTeleop()
    features = build_lerobot_features(
        robot,
        teleop,
        use_videos=False,
    )
    dataset = FakeRemoteDataset(features)

    frames = record_lerobot_episode(
        robot=robot,
        teleop=teleop,
        dataset=dataset,
        fps=1000,
        duration_s=1.0,
        task="pick",
        stop_event=stop_event,
        tolerate_robot_faults=True,
    )

    assert frames == 2
    assert robot.send_attempts == 2
    np.testing.assert_array_equal(dataset.frames[0]["action"], np.array([0.1], dtype=np.float32))
    np.testing.assert_array_equal(dataset.frames[1]["action"], np.zeros(1, dtype=np.float32))
    assert "robot fault" in capsys.readouterr().out.lower()


def test_recording_waits_until_episode_end_after_persistent_stale_state(capsys):
    class FreshThenStaleRobot(FakeRemoteRobot):
        def get_observation(self):
            self.get_observation_calls += 1
            if self.get_observation_calls == 1:
                return {"joint_1.pos": 0.1}
            raise StaleFrankaStateError("state stopped after Franka fault")

    robot = FreshThenStaleRobot(failures_before_observation=0)
    teleop = FakeRemoteTeleop()
    features = build_lerobot_features(
        robot,
        teleop,
        use_videos=False,
    )
    dataset = FakeRemoteDataset(features)

    started_t = time.perf_counter()
    frames = record_lerobot_episode(
        robot=robot,
        teleop=teleop,
        dataset=dataset,
        fps=1000,
        duration_s=0.03,
        task="pick",
        max_consecutive_state_misses=1,
        state_retry_sleep_s=0.001,
        tolerate_robot_faults=True,
    )
    elapsed_s = time.perf_counter() - started_t

    assert frames == 1
    assert len(dataset.frames) == 1
    assert robot.sent_actions == [{"delta_ee_pose.x": 0.0}]
    assert elapsed_s >= 0.02
    assert "robot fault" in capsys.readouterr().out.lower()


def test_robot_fault_tolerance_does_not_hide_dataset_write_errors():
    stop_event = Event()

    class WriteFailingDataset(FakeRemoteDataset):
        def add_frame(self, frame):
            raise OSError("disk full")

    robot = FakeRemoteRobot(failures_before_observation=0)
    teleop = FakeRemoteTeleop()
    features = build_lerobot_features(
        robot,
        teleop,
        use_videos=False,
    )
    dataset = WriteFailingDataset(features)

    with pytest.raises(OSError, match="disk full"):
        record_lerobot_episode(
            robot=robot,
            teleop=teleop,
            dataset=dataset,
            fps=30,
            duration_s=1 / 30,
            task="pick",
            stop_event=stop_event,
            tolerate_robot_faults=True,
        )


def test_robot_fault_tolerance_does_not_hide_invalid_action_errors():
    class InvalidActionRobot(FakeRemoteRobot):
        def send_action(self, action):
            raise ValueError("invalid action shape")

    robot = InvalidActionRobot(failures_before_observation=0)
    teleop = FakeRemoteTeleop()
    dataset = FakeRemoteDataset(
        build_lerobot_features(
            robot,
            teleop,
            use_videos=False,
        )
    )

    with pytest.raises(ValueError, match="invalid action shape"):
        record_lerobot_episode(
            robot=robot,
            teleop=teleop,
            dataset=dataset,
            fps=30,
            duration_s=1 / 30,
            task="pick",
            tolerate_robot_faults=True,
        )


def test_robot_fault_tolerance_does_not_hide_unclassified_runtime_errors():
    class BrokenRobot(FakeRemoteRobot):
        def send_action(self, action):
            raise RuntimeError("unexpected local control bug")

    robot = BrokenRobot(failures_before_observation=0)
    teleop = FakeRemoteTeleop()
    dataset = FakeRemoteDataset(
        build_lerobot_features(
            robot,
            teleop,
            use_videos=False,
        )
    )

    with pytest.raises(RuntimeError, match="unexpected local control bug"):
        record_lerobot_episode(
            robot=robot,
            teleop=teleop,
            dataset=dataset,
            fps=30,
            duration_s=1 / 30,
            task="pick",
            tolerate_robot_faults=True,
        )


def test_robot_fault_tolerance_does_not_hide_invalid_control_url():
    from requests.exceptions import InvalidURL

    class MisconfiguredRobot(FakeRemoteRobot):
        def send_action(self, action):
            raise InvalidURL("bad control URL")

    robot = MisconfiguredRobot(failures_before_observation=0)
    teleop = FakeRemoteTeleop()
    dataset = FakeRemoteDataset(
        build_lerobot_features(
            robot,
            teleop,
            use_videos=False,
        )
    )

    with pytest.raises(InvalidURL, match="bad control URL"):
        record_lerobot_episode(
            robot=robot,
            teleop=teleop,
            dataset=dataset,
            fps=30,
            duration_s=1 / 30,
            task="pick",
            tolerate_robot_faults=True,
        )


def test_legacy_recording_still_fails_fast_on_explicit_franka_fault():
    class FaultingRobot(FakeRemoteRobot):
        def send_action(self, action):
            raise FrankaControlError("Franka fault")

    robot = FaultingRobot(failures_before_observation=0)
    teleop = FakeRemoteTeleop()
    dataset = FakeRemoteDataset(build_lerobot_features(robot, teleop, use_videos=False))

    with pytest.raises(FrankaControlError, match="Franka fault"):
        record_lerobot_episode(
            robot=robot,
            teleop=teleop,
            dataset=dataset,
            fps=30,
            duration_s=1 / 30,
            task="pick",
        )


def test_run_record_dry_run_builds_config_without_touching_hardware():
    exit_code = run_record_main(
        [
            "--dry-run-config",
            "--camera-backend",
            "none",
            "--repo-id",
            "local/test",
            "--root",
            "/tmp/franka-test",
        ]
    )

    assert exit_code == 0


def test_legacy_run_record_and_ui_parsers_do_not_advertise_target_floor():
    for parser in (build_arg_parser(), run_record_ui.build_ui_arg_parser()):
        assert "--target-floor" not in parser.format_help()
        assert not hasattr(parser.parse_args([]), "target_floor")
        with pytest.raises(SystemExit):
            parser.parse_args(["--target-floor", "4"])


def test_run_record_defaults_to_direct_realsense_camera_backend():
    args = build_arg_parser().parse_args([])

    camera_configs, camera_shapes = build_camera_configs(args)
    robot_config = build_robot_config(args)

    assert args.camera_backend == "realsense"
    assert args.camera_serial_or_name == "Intel RealSense L515"
    assert args.fps == 30
    assert camera_shapes == {"l515": (540, 960, 3)}
    assert sorted(camera_configs) == ["l515"]
    assert camera_configs["l515"].serial_number_or_name == "Intel RealSense L515"
    assert sorted(robot_config.cameras) == ["l515"]
    assert robot_config.camera_shapes == {"l515": (540, 960, 3)}


def test_run_record_explicit_opencv_camera_backend_keeps_legacy_l515_auto_selection(monkeypatch):
    args = build_arg_parser().parse_args(["--camera-backend", "opencv"])
    monkeypatch.setattr(run_record, "_find_l515_opencv_color_device", lambda: "/dev/video6")

    camera_configs, camera_shapes = build_camera_configs(args)
    robot_config = build_robot_config(args)

    assert sorted(camera_configs) == ["l515"]
    assert camera_configs["l515"].index_or_path == "/dev/video6"
    assert camera_configs["l515"].fourcc == "YUYV"
    assert camera_shapes == {"l515": (540, 960, 3)}
    assert sorted(robot_config.cameras) == ["l515"]
    assert robot_config.camera_shapes == {"l515": (540, 960, 3)}


def test_run_record_treats_legacy_video0_as_l515_color_auto_profile(monkeypatch):
    args = build_arg_parser().parse_args(
        ["--camera-backend", "opencv", "--opencv-index-or-path", "/dev/video0"]
    )
    monkeypatch.setattr(run_record, "_find_l515_opencv_color_device", lambda: "/dev/video6")

    camera_configs, camera_shapes = build_camera_configs(args)

    assert camera_configs["l515"].index_or_path == "/dev/video6"
    assert camera_configs["l515"].width == 960
    assert camera_configs["l515"].height == 540
    assert camera_shapes == {"l515": (540, 960, 3)}


def test_run_record_preserves_explicit_video0_grey_profile(monkeypatch):
    args = build_arg_parser().parse_args(
        [
            "--camera-backend",
            "opencv",
            "--opencv-index-or-path",
            "/dev/video0",
            "--camera-width",
            "480",
            "--camera-height",
            "640",
        ]
    )
    monkeypatch.setattr(run_record, "_find_l515_opencv_color_device", lambda: "/dev/video6")

    camera_configs, camera_shapes = build_camera_configs(args)

    assert camera_configs["l515"].index_or_path == "/dev/video0"
    assert camera_configs["l515"].width == 480
    assert camera_configs["l515"].height == 640
    assert camera_shapes == {"l515": (640, 480, 3)}


def test_run_record_default_spacemouse_scale_tracks_recording_fps():
    args = build_arg_parser().parse_args(["--fps", "30"])

    teleop_config = build_teleop_config(args)

    assert teleop_config.pose_scaler == (0.05 / 30.0, 0.40 / 30.0)


def test_run_record_can_disable_camera_for_config_dry_run():
    args = build_arg_parser().parse_args(["--camera-backend", "none"])

    camera_configs, camera_shapes = build_camera_configs(args)
    robot_config = build_robot_config(args)

    assert camera_configs == {}
    assert camera_shapes == {}
    assert robot_config.cameras == {}
    assert robot_config.camera_shapes == {}


def test_run_record_defaults_to_gripper_tolerant_state_timeout():
    args = build_arg_parser().parse_args([])

    robot_config = build_robot_config(args)

    assert args.state_timeout_s == 0.2
    assert robot_config.state_timeout_s == 0.2
    assert args.max_state_wait_s == 1.0
    assert args.state_max_consecutive_misses == 60


def test_run_record_builds_realsense_camera_and_robot_config_from_args():
    args = build_arg_parser().parse_args(
        [
            "--camera-backend",
            "realsense",
            "--camera-serial-or-name",
            "123456",
            "--camera-width",
            "424",
            "--camera-height",
            "240",
            "--camera-fps",
            "30",
            "--fps",
            "15",
            "--control-host",
            "test-controller",
            "--max-camera-age-s",
            "0.12",
            "--state-timeout-s",
            "0.03",
        ]
    )

    camera_configs, camera_shapes = build_camera_configs(args)
    robot_config = build_robot_config(args)
    teleop_config = build_teleop_config(args)

    assert sorted(camera_configs) == ["l515"]
    assert camera_shapes == {"l515": (240, 424, 3)}
    assert robot_config.camera_shapes == camera_shapes
    assert robot_config.control_hz == 15.0
    assert robot_config.control_host == "test-controller"
    assert robot_config.max_camera_age_s == 0.12
    assert robot_config.state_timeout_s == 0.03
    assert teleop_config.pose_scaler == (0.05 / 15.0, 0.40 / 15.0)




DELTA_EE_ACTION_KEYS = [
    "delta_ee_pose.x",
    "delta_ee_pose.y",
    "delta_ee_pose.z",
    "delta_ee_pose.rx",
    "delta_ee_pose.ry",
    "delta_ee_pose.rz",
]
