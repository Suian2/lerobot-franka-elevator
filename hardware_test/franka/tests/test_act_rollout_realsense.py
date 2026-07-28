from __future__ import annotations

import builtins
import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hardware_test.franka.defaults import DEFAULT_CONTROL_HOST
from hardware_test.franka.floor_condition import encode_target_floor
from hardware_test.franka.franka_robot import JOINT_KEYS
from lerobot.utils.constants import OBS_ENV_STATE

MODULE_NAME = "hardware_test.franka.run_act_rollout_realsense"


def _load_rollout_module():
    spec = importlib.util.find_spec(MODULE_NAME)
    assert spec is not None, f"missing {MODULE_NAME}"
    return importlib.import_module(MODULE_NAME)


def test_parser_uses_direct_realsense_device_selection():
    rollout = _load_rollout_module()

    cli = ["--policy-path", "/tmp/policy", "--target-floor", "4"]
    args = rollout.build_arg_parser().parse_args(cli)

    assert args.camera_serial_or_name == "Intel RealSense L515"
    assert args.target_floor == 4
    assert not hasattr(args, "image_zmq")


@pytest.mark.parametrize("target_floor", [1, 4, 5])
def test_parser_accepts_only_trained_target_floors(target_floor):
    rollout = _load_rollout_module()

    args = rollout.build_arg_parser().parse_args(
        ["--policy-path", "/tmp/policy", "--target-floor", str(target_floor)]
    )

    assert args.target_floor == target_floor


@pytest.mark.parametrize("target_floor", [2, 3])
def test_parser_rejects_untrained_target_floors(target_floor):
    rollout = _load_rollout_module()

    with pytest.raises(SystemExit):
        rollout.build_arg_parser().parse_args(
            ["--policy-path", "/tmp/policy", "--target-floor", str(target_floor)]
        )


def test_parser_requires_target_floor():
    rollout = _load_rollout_module()

    with pytest.raises(SystemExit):
        rollout.build_arg_parser().parse_args(["--policy-path", "/tmp/policy"])


def _make_observation():
    return {
        **{key: float(index) / 10 for index, key in enumerate(JOINT_KEYS, start=1)},
        "gripper_width_norm": 0.008811,
        "l515": np.full((540, 960, 3), 127, dtype=np.uint8),
    }


def test_build_policy_observation_adds_canonical_floor_condition():
    rollout = _load_rollout_module()

    result = rollout.build_policy_observation(
        _make_observation(),
        floor_condition=encode_target_floor(4),
    )

    assert result["observation.state"].shape == (8,)
    assert result[OBS_ENV_STATE].shape == (5,)
    assert result[OBS_ENV_STATE].dtype == torch.float32
    assert result[OBS_ENV_STATE].tolist() == [0, 0, 0, 1, 0]


def _make_policy_config(*, environment_shape=(5,)):
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


def test_validate_policy_features_requires_environment_state():
    rollout = _load_rollout_module()

    rollout.validate_policy_features(_make_policy_config())

    with pytest.raises(ValueError, match=OBS_ENV_STATE):
        rollout.validate_policy_features(_make_policy_config(environment_shape=None))
    with pytest.raises(ValueError, match=OBS_ENV_STATE):
        rollout.validate_policy_features(_make_policy_config(environment_shape=(4,)))


@pytest.mark.parametrize(
    "processed_floor,match",
    [
        (torch.zeros(5, dtype=torch.float32), "shape"),
        (torch.zeros(1, 5, dtype=torch.float64), "float32"),
        (torch.full((1, 5), torch.nan, dtype=torch.float32), "finite"),
    ],
)
def test_select_robot_action_rejects_invalid_processed_floor(processed_floor, match):
    rollout = _load_rollout_module()

    class BadFloorProcessor:
        def __call__(self, observation):
            batch = {key: value.unsqueeze(0) for key, value in observation.items()}
            batch[OBS_ENV_STATE] = processed_floor
            return batch

    class UnreachablePolicy:
        def select_action(self, batch):
            raise AssertionError("policy must not receive an invalid floor condition")

    bundle = rollout.PolicyBundle(UnreachablePolicy(), BadFloorProcessor(), lambda value: value)

    with pytest.raises(ValueError, match=match):
        rollout.select_robot_action(
            bundle,
            _make_observation(),
            floor_condition=encode_target_floor(5),
            action_scale=0.25,
        )


def test_build_robot_uses_run_record_realsense_camera_configuration(monkeypatch):
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


def test_realsense_rollout_source_has_no_ros2_image_bridge_dependency():
    rollout = _load_rollout_module()

    source = Path(rollout.__file__).read_text()

    assert "RealSenseCameraConfig" in source
    for forbidden in ("ros2_image_bridge", "ZmqRgbImageClient", "image_zmq", "rclpy", "sensor_msgs"):
        assert forbidden not in source


def test_main_prints_floor_checkpoint_and_direct_camera_before_connecting(monkeypatch):
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
        lambda robot, bundle, safety, stop_event, *, floor_condition: events.append(
            ("run", floor_condition.tolist())
        ),
    )

    exit_code = rollout.main(
        [
            "--policy-path",
            "/tmp/../tmp/policy",
            "--target-floor",
            "5",
            "--camera-serial-or-name",
            "Intel RealSense L515",
        ]
    )

    assert exit_code == 0
    assert events[:4] == [
        ("print", "目标楼层: 5"),
        ("print", "one-hot 向量: [0.0, 0.0, 0.0, 0.0, 1.0]"),
        ("print", f"checkpoint 路径: {Path('/tmp/policy').resolve()}"),
        ("print", "相机输入: direct RealSense SDK/OpenCV (Intel RealSense L515)"),
    ]
    assert ("run", [0.0, 0.0, 0.0, 0.0, 1.0]) in events
