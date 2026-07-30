from __future__ import annotations

import argparse
import contextlib
import logging
import signal
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (SRC_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hardware_test.franka.defaults import get_control_host  # noqa: E402
from hardware_test.franka.franka_robot import (  # noqa: E402
    DELTA_EE_KEYS,
    JOINT_KEYS,
    FrankaRobot,
    FrankaRobotConfig,
)
from lerobot.utils.constants import OBS_ENV_STATE  # noqa: E402

logger = logging.getLogger(__name__)

CAMERA_KEY = "l515"
EXPECTED_IMAGE_SHAPE = (540, 960, 3)
EXPECTED_STATE_KEYS = (*JOINT_KEYS, "gripper_width_norm")
EXPECTED_ACTION_DIM = 7
APPROVED_FPS = 30
MAX_DURATION_S = 200.0
MAX_ACTION_SCALE = 2.5
MAX_LINEAR_VELOCITY = 0.10
MAX_ANGULAR_VELOCITY = 0.20


@dataclass(frozen=True)
class RolloutSafetyConfig:
    execute: bool = False
    fps: int = APPROVED_FPS
    duration_s: float = MAX_DURATION_S
    action_scale: float = MAX_ACTION_SCALE
    max_linear_velocity: float = MAX_LINEAR_VELOCITY
    max_angular_velocity: float = MAX_ANGULAR_VELOCITY

    def __post_init__(self) -> None:
        checks = (
            (self.fps == APPROVED_FPS, "fps must be exactly 30"),
            (
                0 < self.duration_s <= MAX_DURATION_S,
                f"duration_s must be in (0, {MAX_DURATION_S:g}]",
            ),
            (
                0 < self.action_scale <= MAX_ACTION_SCALE,
                f"action_scale must be in (0, {MAX_ACTION_SCALE:g}]",
            ),
            (
                0 < self.max_linear_velocity <= MAX_LINEAR_VELOCITY,
                f"max_linear_velocity must be in (0, {MAX_LINEAR_VELOCITY:g}]",
            ),
            (
                0 < self.max_angular_velocity <= MAX_ANGULAR_VELOCITY,
                f"max_angular_velocity must be in (0, {MAX_ANGULAR_VELOCITY:g}]",
            ),
        )
        for valid, message in checks:
            if not valid:
                raise ValueError(message)


def build_policy_observation(observation: dict[str, Any]) -> dict[str, torch.Tensor]:
    missing = [key for key in (*EXPECTED_STATE_KEYS, CAMERA_KEY) if key not in observation]
    if missing:
        raise ValueError(f"observation is missing required keys: {missing}")

    state = np.asarray([observation[key] for key in EXPECTED_STATE_KEYS], dtype=np.float32)
    if state.shape != (8,):
        raise ValueError(f"observation.state has shape {state.shape}, expected (8,)")
    if not np.isfinite(state).all():
        raise ValueError("observation.state must contain only finite values")

    image = np.asarray(observation[CAMERA_KEY])
    if image.shape != EXPECTED_IMAGE_SHAPE:
        raise ValueError(f"l515 image has shape {image.shape}, expected {EXPECTED_IMAGE_SHAPE}")
    if image.dtype != np.uint8:
        raise ValueError(f"l515 image must be uint8, got {image.dtype}")

    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)
    return {
        "observation.state": torch.from_numpy(state),
        "observation.images.l515": image_tensor.contiguous(),
    }


def policy_action_to_robot_action(action: torch.Tensor, *, action_scale: float) -> dict[str, float]:
    values = torch.as_tensor(action).detach().cpu().float()
    if values.shape != (EXPECTED_ACTION_DIM,):
        raise ValueError(f"policy action has shape {tuple(values.shape)}, expected (7,)")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("policy action must contain only finite values")

    scaled_pose = values[:6] * float(action_scale)
    return {key: float(value) for key, value in zip(DELTA_EE_KEYS, scaled_pose, strict=True)}


@dataclass(frozen=True)
class PolicyBundle:
    policy: Any
    preprocessor: Any
    postprocessor: Any


def validate_policy_features(config: Any) -> None:
    if config.type != "act":
        raise ValueError(f"expected an ACT checkpoint, got {config.type!r}")
    if OBS_ENV_STATE in config.input_features:
        raise ValueError(
            f"conditioned checkpoint includes {OBS_ENV_STATE}; "
            "this single-button rollout requires an unconditioned checkpoint"
        )

    expected_inputs = {
        "observation.state": (8,),
        "observation.images.l515": (3, 540, 960),
    }
    for key, shape in expected_inputs.items():
        feature = config.input_features.get(key)
        if feature is None or tuple(feature.shape) != shape:
            actual = None if feature is None else tuple(feature.shape)
            raise ValueError(f"checkpoint feature {key} has shape {actual}, expected {shape}")

    action = config.output_features.get("action")
    if action is None or tuple(action.shape) != (EXPECTED_ACTION_DIM,):
        actual = None if action is None else tuple(action.shape)
        raise ValueError(f"checkpoint action has shape {actual}, expected ({EXPECTED_ACTION_DIM},)")


def load_policy_bundle(policy_path: str | Path, *, device: str = "cuda") -> PolicyBundle:
    from lerobot import policies as lerobot_policies
    from lerobot.configs import PreTrainedConfig

    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the approved physical smoke test requires an available CUDA device")

    path = Path(policy_path).expanduser().resolve()
    config = PreTrainedConfig.from_pretrained(path, local_files_only=True)
    validate_policy_features(config)
    config.device = device
    policy = (
        lerobot_policies.get_policy_class(config.type)
        .from_pretrained(path, config=config, local_files_only=True)
        .to(device)
        .eval()
    )
    preprocessor, postprocessor = lerobot_policies.make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return PolicyBundle(policy, preprocessor, postprocessor)


def reset_inference_state(bundle: PolicyBundle) -> None:
    for component in (bundle.policy, bundle.preprocessor, bundle.postprocessor):
        reset = getattr(component, "reset", None)
        if callable(reset):
            reset()


def select_robot_action(
    bundle: PolicyBundle,
    raw_observation: dict[str, Any],
    *,
    action_scale: float,
) -> dict[str, float]:
    observation = build_policy_observation(raw_observation)
    processed = bundle.preprocessor(observation)
    with torch.inference_mode():
        selected = bundle.policy.select_action(processed)
        selected = bundle.postprocessor(selected)

    action = torch.as_tensor(selected).detach().cpu()
    if action.shape != (1, EXPECTED_ACTION_DIM):
        raise ValueError(
            f"postprocessed selected action has shape {tuple(action.shape)}, expected (1, {EXPECTED_ACTION_DIM})"
        )
    return policy_action_to_robot_action(action[0], action_scale=action_scale)


def build_robot(
    *,
    control_host: str,
    camera_serial_or_name: str,
    safety: RolloutSafetyConfig,
) -> FrankaRobot:
    from lerobot.cameras.realsense import RealSenseCameraConfig

    camera_config = RealSenseCameraConfig(
        serial_number_or_name=camera_serial_or_name,
        fps=APPROVED_FPS,
        width=EXPECTED_IMAGE_SHAPE[1],
        height=EXPECTED_IMAGE_SHAPE[0],
        warmup_s=1,
        use_rgb=True,
        use_depth=False,
    )
    config = FrankaRobotConfig(
        action_mode="delta_ee_pose",
        cartesian_action_units="delta",
        control_hz=float(safety.fps),
        max_linear_velocity=safety.max_linear_velocity,
        max_angular_velocity=safety.max_angular_velocity,
        command_duration_ms=100,
        control_host=control_host,
        velocity_transport="zmq",
        state_cache_enabled=True,
        state_poll_hz=30.0,
        state_timeout_s=0.2,
        max_state_age_s=0.25,
        use_gripper=True,
        cameras={CAMERA_KEY: camera_config},
        camera_shapes={CAMERA_KEY: EXPECTED_IMAGE_SHAPE},
        camera_read_mode="latest",
        max_camera_age_s=0.25,
    )
    return FrankaRobot(config)


def run_control_loop(
    robot: Any,
    bundle: PolicyBundle,
    safety: RolloutSafetyConfig,
    stop_event: Event,
    *,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    reset_inference_state(bundle)
    if not safety.execute:
        dry_run_action = select_robot_action(
            bundle,
            robot.get_observation(),
            action_scale=safety.action_scale,
        )
        logger.info("dry-run action (gripper suppressed): %s", dry_run_action)
        return

    control_interval = 1.0 / safety.fps
    robot.send_zero_cartesian_velocity()
    sent_frames = 0
    started: float | None = None
    try:
        action = select_robot_action(
            bundle,
            robot.get_observation(),
            action_scale=safety.action_scale,
        )
        logger.info("first action (gripper suppressed): %s", action)
        started = clock()
        while not stop_event.is_set() and clock() - started < safety.duration_s:
            loop_started = clock()
            robot.send_action(action)
            sent_frames += 1
            sleep(max(0.0, control_interval - (clock() - loop_started)))
            if stop_event.is_set() or clock() - started >= safety.duration_s:
                break
            action = select_robot_action(
                bundle,
                robot.get_observation(),
                action_scale=safety.action_scale,
            )
    finally:
        robot.send_zero_cartesian_velocity()
        elapsed_s = 0.0 if started is None else max(0.0, clock() - started)
        achieved_hz = sent_frames / elapsed_s if elapsed_s > 0.0 else 0.0
        logger.info(
            "rollout complete sent_frames=%d elapsed_s=%.3f achieved_hz=%.1f stop_requested=%s",
            sent_frames,
            elapsed_s,
            achieved_hz,
            stop_event.is_set(),
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an unconditioned bounded Franka ACT physical smoke test.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--control-host", default=get_control_host())
    parser.add_argument("--camera-serial-or-name", default="Intel RealSense L515")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--fps", type=int, default=APPROVED_FPS)
    parser.add_argument("--duration-s", type=float, default=MAX_DURATION_S)
    parser.add_argument("--action-scale", type=float, default=MAX_ACTION_SCALE)
    parser.add_argument("--max-linear-velocity", type=float, default=MAX_LINEAR_VELOCITY)
    parser.add_argument("--max-angular-velocity", type=float, default=MAX_ANGULAR_VELOCITY)
    parser.add_argument("--execute", action="store_true")
    return parser


def install_signal_handlers(stop_event: Event) -> None:
    def request_stop(signum, frame):
        del signum, frame
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    policy_path = Path(args.policy_path).expanduser().resolve()
    safety = RolloutSafetyConfig(
        execute=args.execute,
        fps=args.fps,
        duration_s=args.duration_s,
        action_scale=args.action_scale,
        max_linear_velocity=args.max_linear_velocity,
        max_angular_velocity=args.max_angular_velocity,
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    print(f"checkpoint 路径: {policy_path}")
    print(f"相机输入: direct RealSense SDK/OpenCV ({args.camera_serial_or_name})")
    logger.info(
        "mode=%s duration_s=%.2f action_scale=%.3f max_linear_velocity=%.3f "
        "max_angular_velocity=%.3f gripper=SUPPRESSED",
        "EXECUTE" if safety.execute else "DRY_RUN",
        safety.duration_s,
        safety.action_scale,
        safety.max_linear_velocity,
        safety.max_angular_velocity,
    )

    bundle = load_policy_bundle(policy_path, device=args.device)
    robot = build_robot(
        control_host=args.control_host,
        camera_serial_or_name=args.camera_serial_or_name,
        safety=safety,
    )
    stop_event = Event()
    install_signal_handlers(stop_event)
    connected = False
    try:
        robot.connect()
        connected = True
        run_control_loop(robot, bundle, safety, stop_event)
    finally:
        if connected:
            with contextlib.suppress(Exception):
                robot.send_zero_cartesian_velocity()
            robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
