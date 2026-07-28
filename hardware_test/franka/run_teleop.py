from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from threading import Event
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware_test.franka.defaults import get_control_host  # noqa: E402
from hardware_test.franka.franka_robot import DELTA_EE_KEYS, FrankaRobot, FrankaRobotConfig
from hardware_test.franka.franka_spacemouse_teleop import (
    FrankaSpaceMouseTeleop,
    FrankaSpaceMouseTeleopConfig,
)
from lerobot.types import RobotAction
from lerobot.utils.robot_utils import precise_sleep


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teleoperate Franka through the ZMQ/HTTP control chain.")
    parser.add_argument("--control-host", default=get_control_host())
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--velocity-transport", choices=("zmq", "http"), default="zmq")
    parser.add_argument("--zmq-url", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-linear-velocity", type=float, default=0.05)
    parser.add_argument("--max-angular-velocity", type=float, default=0.40)
    parser.add_argument("--command-duration-ms", type=int, default=300)
    parser.add_argument("--backend", choices=("spacemouse", "keyboard"), default="spacemouse")
    parser.add_argument("--spnav-lib-path", default=None)
    parser.add_argument("--spnav-axis-scale", type=float, default=500.0)
    parser.add_argument("--deadband", type=float, default=0.05)
    parser.add_argument("--motion-timeout", type=float, default=0.2)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--keyboard-deadman-key", default="space")
    parser.add_argument("--no-keyboard-deadman", dest="keyboard_deadman", action="store_false")
    parser.set_defaults(keyboard_deadman=True)
    return parser


def build_robot(args: argparse.Namespace) -> FrankaRobot:
    return FrankaRobot(
        FrankaRobotConfig(
            control_hz=float(args.fps),
            max_linear_velocity=args.max_linear_velocity,
            max_angular_velocity=args.max_angular_velocity,
            command_duration_ms=args.command_duration_ms,
            base_url=args.base_url,
            control_host=args.control_host,
            velocity_transport=args.velocity_transport,
            zmq_url=args.zmq_url,
            validate_connection=True,
            state_cache_enabled=False,
            cameras={},
            camera_shapes={},
        )
    )


def build_teleop(args: argparse.Namespace):
    linear_scale = args.max_linear_velocity / float(args.fps)
    angular_scale = args.max_angular_velocity / float(args.fps)
    if args.backend == "spacemouse":
        return FrankaSpaceMouseTeleop(
            FrankaSpaceMouseTeleopConfig(
                spnav_lib_path=args.spnav_lib_path,
                spnav_axis_scale=args.spnav_axis_scale,
                deadband=args.deadband,
                motion_timeout=args.motion_timeout,
                pose_scaler=(linear_scale, angular_scale),
                mirror=args.mirror,
            )
        )
    return KeyboardDeltaTeleop(
        linear_step=linear_scale,
        angular_step=angular_scale,
        require_deadman=args.keyboard_deadman,
        deadman_key=args.keyboard_deadman_key,
    )


class KeyboardDeltaTeleop:
    action_features = {key: float for key in DELTA_EE_KEYS} | {"gripper_cmd_bin": float}

    def __init__(
        self,
        *,
        linear_step: float,
        angular_step: float,
        require_deadman: bool = True,
        deadman_key: str = "space",
    ):
        self.linear_step = float(linear_step)
        self.angular_step = float(angular_step)
        self.require_deadman = bool(require_deadman)
        self.deadman_key = deadman_key
        self._pressed: set[str] = set()
        self._listener = None
        self._gripper_cmd = 1.0
        self._last_toggle = False

    @property
    def is_connected(self) -> bool:
        return self._listener is not None

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002
        try:
            from pynput import keyboard
        except Exception as exc:
            raise ImportError("keyboard teleop requires pynput in the LeRobot environment") from exc

        self._keyboard = keyboard
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def get_action(self) -> RobotAction:
        enabled = True
        if self.require_deadman:
            enabled = self.deadman_key in self._pressed

        action: RobotAction = {key: 0.0 for key in DELTA_EE_KEYS}
        if enabled:
            action["delta_ee_pose.x"] = self.linear_step * _axis(self._pressed, positive="w", negative="s")
            action["delta_ee_pose.y"] = self.linear_step * _axis(self._pressed, positive="a", negative="d")
            action["delta_ee_pose.z"] = self.linear_step * _axis(self._pressed, positive="q", negative="e")
            action["delta_ee_pose.rx"] = self.angular_step * _axis(self._pressed, positive="u", negative="o")
            action["delta_ee_pose.ry"] = self.angular_step * _axis(self._pressed, positive="i", negative="k")
            action["delta_ee_pose.rz"] = self.angular_step * _axis(self._pressed, positive="j", negative="l")

        toggle_pressed = "c" in self._pressed
        if toggle_pressed and not self._last_toggle:
            self._gripper_cmd = 1.0 - self._gripper_cmd
        self._last_toggle = toggle_pressed

        action["gripper_cmd_bin"] = self._gripper_cmd
        action["reset_requested"] = "h" in self._pressed
        return action

    def _on_press(self, key: Any) -> None:
        self._pressed.add(_key_name(key))

    def _on_release(self, key: Any) -> None:
        self._pressed.discard(_key_name(key))


def _axis(pressed: set[str], *, positive: str, negative: str) -> float:
    return float(positive in pressed) - float(negative in pressed)


def _key_name(key: Any) -> str:
    char = getattr(key, "char", None)
    if char:
        return str(char).lower()
    name = getattr(key, "name", None)
    if name:
        return str(name).lower()
    return str(key).lower().removeprefix("key.")


def install_signal_handlers(stop_event: Event) -> None:
    def _request_stop(signum, frame):  # noqa: ARG001
        stop_event.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.fps <= 0:
        raise ValueError("fps must be positive")

    stop_event = Event()
    install_signal_handlers(stop_event)
    robot = build_robot(args)
    teleop = build_teleop(args)
    interval_s = 1.0 / float(args.fps)

    try:
        robot.connect(calibrate=False)
        teleop.connect()
        print(
            f"teleop ready backend={args.backend} control_host={args.control_host} "
            f"transport={args.velocity_transport} fps={args.fps}",
            flush=True,
        )
        while not stop_event.is_set():
            loop_t = time.perf_counter()
            action = teleop.get_action()
            robot.send_action(action)
            precise_sleep(max(0.0, interval_s - (time.perf_counter() - loop_t)))
    finally:
        with _suppress_errors():
            robot.send_action({key: 0.0 for key in DELTA_EE_KEYS})
        with _suppress_errors():
            teleop.disconnect()
        with _suppress_errors():
            robot.disconnect()
    return 0


class _suppress_errors:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return True


if __name__ == "__main__":
    raise SystemExit(main())
