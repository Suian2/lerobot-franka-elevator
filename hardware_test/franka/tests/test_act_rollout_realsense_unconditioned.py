from __future__ import annotations

import builtins
import importlib
import importlib.util
import sys
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hardware_test.franka.defaults import DEFAULT_CONTROL_HOST
from hardware_test.franka.franka_robot import DELTA_EE_KEYS, JOINT_KEYS
from lerobot.utils.constants import OBS_ENV_STATE

MODULE_NAME = "hardware_test.franka.run_act_rollout_realsense_unconditioned"


def _load_rollout_module():
    spec = importlib.util.find_spec(MODULE_NAME)
    assert spec is not None, f"missing {MODULE_NAME}"
    return importlib.import_module(MODULE_NAME)


def _make_observation():
    return {
        **{key: float(index) / 10 for index, key in enumerate(JOINT_KEYS, start=1)},
        "gripper_width_norm": 0.008811,
        "l515": np.full((540, 960, 3), 127, dtype=np.uint8),
    }


def _make_policy_config(*, environment_shape=None):
    input_features = {
        "observation.state": SimpleNamespace(shape=(8,)),
        "observation.images.l515": SimpleNamespace(shape=(3, 540, 960)),
    }
    if environment_shape is not None:
        input_features[OBS_ENV_STATE] = SimpleNamespace(shape=environment_shape)
    return SimpleNamespace(
        type="act",
        input_features=input_features,
        output_features={"action": SimpleNamespace(shape=(7,))},
    )


def test_parser_does_not_accept_or_require_target_floor():
    rollout = _load_rollout_module()

    args = rollout.build_arg_parser().parse_args(["--policy-path", "/tmp/policy"])

    assert args.camera_serial_or_name == "Intel RealSense L515"
    assert not hasattr(args, "target_floor")
    assert not hasattr(args, "image_zmq")
    with pytest.raises(SystemExit):
        rollout.build_arg_parser().parse_args(
            ["--policy-path", "/tmp/policy", "--target-floor", "4"]
        )


def test_build_policy_observation_has_only_unconditioned_inputs():
    rollout = _load_rollout_module()

    result = rollout.build_policy_observation(_make_observation())

    assert set(result) == {"observation.state", "observation.images.l515"}
    assert result["observation.state"].shape == (8,)
    assert result["observation.state"].dtype == torch.float32
    assert result["observation.images.l515"].shape == (3, 540, 960)
    assert result["observation.images.l515"].dtype == torch.float32


def test_validate_policy_features_accepts_only_unconditioned_schema():
    rollout = _load_rollout_module()

    rollout.validate_policy_features(_make_policy_config())

    with pytest.raises(ValueError, match="conditioned checkpoint"):
        rollout.validate_policy_features(_make_policy_config(environment_shape=(5,)))


def test_policy_action_to_robot_action_scales_pose_and_suppresses_gripper():
    rollout = _load_rollout_module()

    result = rollout.policy_action_to_robot_action(
        torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 1.0]),
        action_scale=0.25,
    )

    assert result == dict(
        zip(DELTA_EE_KEYS, [0.25, -0.5, 0.75, -1.0, 1.25, -1.5], strict=True)
    )
    assert "gripper.pos" not in result


def test_select_robot_action_does_not_inject_floor_condition():
    rollout = _load_rollout_module()
    received = []

    class BatchProcessor:
        def __call__(self, observation):
            assert set(observation) == {"observation.state", "observation.images.l515"}
            return {key: value.unsqueeze(0) for key, value in observation.items()}

    class CapturingPolicy:
        def select_action(self, batch):
            received.append(batch)
            return torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0]])

    bundle = rollout.PolicyBundle(CapturingPolicy(), BatchProcessor(), lambda value: value)

    result = rollout.select_robot_action(bundle, _make_observation(), action_scale=0.1)

    assert len(received) == 1
    assert OBS_ENV_STATE not in received[0]
    assert list(result) == list(DELTA_EE_KEYS)
    assert result[DELTA_EE_KEYS[0]] == pytest.approx(0.1)


def test_build_robot_uses_direct_realsense(monkeypatch):
    rollout = _load_rollout_module()

    class CapturedCameraConfig:
        type = "intelrealsense"

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class CapturedRobot:
        def __init__(self, config):
            self.config = config

    monkeypatch.setitem(
        sys.modules,
        "lerobot.cameras.realsense",
        SimpleNamespace(RealSenseCameraConfig=CapturedCameraConfig),
    )
    monkeypatch.setattr(rollout, "FrankaRobot", CapturedRobot)

    robot = rollout.build_robot(
        control_host=DEFAULT_CONTROL_HOST,
        camera_serial_or_name="Intel RealSense L515",
        safety=rollout.RolloutSafetyConfig(),
    )

    assert robot.config.action_mode == "delta_ee_pose"
    assert robot.config.cartesian_action_units == "delta"
    assert robot.config.velocity_transport == "zmq"
    assert robot.config.camera_shapes == {"l515": (540, 960, 3)}
    assert list(robot.config.cameras) == ["l515"]
    camera = robot.config.cameras["l515"]
    assert camera.type == "intelrealsense"
    assert camera.serial_number_or_name == "Intel RealSense L515"
    assert camera.width == 960
    assert camera.height == 540
    assert camera.fps == 30
    assert camera.warmup_s == 1
    assert camera.use_rgb is True
    assert camera.use_depth is False


def test_source_has_no_ros2_or_image_bridge_dependency():
    rollout = _load_rollout_module()

    source = Path(rollout.__file__).read_text()

    assert "RealSenseCameraConfig" in source
    for forbidden in ("ros2_image_bridge", "ZmqRgbImageClient", "image_zmq", "rclpy", "sensor_msgs"):
        assert forbidden not in source


def test_dry_run_selects_action_without_sending_motion():
    rollout = _load_rollout_module()
    events = []

    class Resettable:
        def reset(self):
            events.append("reset")

    class Processor(Resettable):
        def __call__(self, observation):
            return {key: value.unsqueeze(0) for key, value in observation.items()}

    class Postprocessor(Resettable):
        def __call__(self, action):
            return action

    class Policy(Resettable):
        def select_action(self, batch):
            return torch.zeros((1, 7))

    class Robot:
        def get_observation(self):
            events.append("observation")
            return _make_observation()

        def send_action(self, action):
            raise AssertionError(f"dry run sent action: {action}")

        def send_zero_cartesian_velocity(self):
            raise AssertionError("dry run sent zero velocity")

    bundle = rollout.PolicyBundle(Policy(), Processor(), Postprocessor())

    rollout.run_control_loop(Robot(), bundle, rollout.RolloutSafetyConfig(), Event())

    assert events == ["reset", "reset", "reset", "observation"]


def test_execute_brackets_control_with_zero_velocity_when_already_stopped():
    rollout = _load_rollout_module()
    events = []

    class Policy:
        def select_action(self, batch):
            return torch.zeros((1, 7))

    class Robot:
        def get_observation(self):
            events.append("observation")
            return _make_observation()

        def send_action(self, action):
            events.append(("action", action))

        def send_zero_cartesian_velocity(self):
            events.append("zero")

    stop_event = Event()
    stop_event.set()
    bundle = rollout.PolicyBundle(Policy(), lambda value: value, lambda value: value)
    safety = rollout.RolloutSafetyConfig(execute=True)

    rollout.run_control_loop(Robot(), bundle, safety, stop_event, clock=lambda: 1.0)

    assert events == ["zero", "observation", "zero"]


def test_main_runs_without_floor_condition(monkeypatch):
    rollout = _load_rollout_module()
    events = []

    class FakeRobot:
        def connect(self):
            events.append(("connect",))

        def send_zero_cartesian_velocity(self):
            events.append(("zero",))

        def disconnect(self):
            events.append(("disconnect",))

    monkeypatch.setattr(builtins, "print", lambda message: events.append(("print", message)))
    monkeypatch.setattr(
        rollout,
        "load_policy_bundle",
        lambda policy_path, **kwargs: events.append(("load", policy_path)) or object(),
    )
    monkeypatch.setattr(
        rollout,
        "build_robot",
        lambda **kwargs: events.append(("robot", kwargs)) or FakeRobot(),
    )
    monkeypatch.setattr(rollout, "install_signal_handlers", lambda stop_event: None)
    monkeypatch.setattr(
        rollout,
        "run_control_loop",
        lambda robot, bundle, safety, stop_event: events.append(("run",)),
    )

    exit_code = rollout.main(
        [
            "--policy-path",
            "/tmp/../tmp/policy",
            "--camera-serial-or-name",
            "Intel RealSense L515",
        ]
    )

    assert exit_code == 0
    assert events[:2] == [
        ("print", f"checkpoint 路径: {Path('/tmp/policy').resolve()}"),
        ("print", "相机输入: direct RealSense SDK/OpenCV (Intel RealSense L515)"),
    ]
    assert ("run",) in events
    assert events[-2:] == [("zero",), ("disconnect",)]
