from __future__ import annotations

import ctypes
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

SPNAV_EVENT_ANY = 0
SPNAV_EVENT_MOTION = 1
SPNAV_EVENT_BUTTON = 2

RAW_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
DELTA_EE_KEYS = tuple(f"delta_ee_pose.{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz"))


@TeleoperatorConfig.register_subclass("hardware_test_franka_spacemouse")
@dataclass(kw_only=True)
class FrankaSpaceMouseTeleopConfig(TeleoperatorConfig):
    """SpaceMouse teleop adapter for Franka hardware tests."""

    id: str | None = "franka_spacemouse"
    input_backend: str = "spnav"
    spnav_lib_path: str | None = None
    spnav_axis_scale: float = 500.0
    deadband: float = 0.05
    motion_timeout: float = 0.2

    # Defaults match the VITA record_cfg SpaceMouse calibration for the local wireless receiver.
    linear_axis_map: tuple[str, str, str] = ("z", "y", "x")
    channel_signs: tuple[int, int, int, int, int, int] = (-1, -1, 1, 1, 1, 1)
    angular_axis_map: tuple[str, str, str] = ("z", "x", "y")
    angular_output_signs: tuple[int, int, int] = (-1, 1, -1)
    pose_scaler: tuple[float, float] = (0.05 / 15.0, 0.40 / 15.0)
    mirror: bool = False

    use_gripper: bool = True
    initial_gripper_cmd: float = 1.0
    toggle_gripper: bool = True
    gripper_button_index: int = 0
    home_button_index: int = 1

    def __post_init__(self):
        _validate_axis_map(self.linear_axis_map)
        _validate_axis_map(self.angular_axis_map)
        if len(self.channel_signs) != 6 or any(sign not in (-1, 1) for sign in self.channel_signs):
            raise ValueError("channel_signs must contain six values in {-1, 1}")
        if len(self.angular_output_signs) != 3 or any(
            sign not in (-1, 1) for sign in self.angular_output_signs
        ):
            raise ValueError("angular_output_signs must contain three values in {-1, 1}")
        if len(self.pose_scaler) != 2:
            raise ValueError("pose_scaler must be (linear_scale, angular_scale)")


class FrankaSpaceMouseTeleop(Teleoperator):
    """LeRobot Teleoperator that emits Franka delta end-effector actions."""

    config_class = FrankaSpaceMouseTeleopConfig
    name = "hardware_test_franka_spacemouse"

    def __init__(self, config: FrankaSpaceMouseTeleopConfig, *, reader: Any | None = None):
        super().__init__(config)
        self.config = config
        self._reader = reader or self._make_reader(config)
        self._is_connected = False
        self._gripper_cmd = float(config.initial_gripper_cmd)
        self._last_buttons: list[int] = []

    @property
    def action_features(self) -> dict[str, type]:
        features = {key: float for key in DELTA_EE_KEYS}
        if self.config.use_gripper:
            features["gripper_cmd_bin"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def gripper_command(self) -> float:
        return float(self._gripper_cmd)

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        open_reader = getattr(self._reader, "open", None)
        if callable(open_reader):
            open_reader()
        self._is_connected = True
        if calibrate:
            self.calibrate()

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_not_connected
    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def clear_input(self) -> None:
        clear_reader = getattr(self._reader, "clear", None)
        if callable(clear_reader):
            clear_reader()
        self._last_buttons.clear()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        motion, buttons = self._reader.get_action()
        motion = np.asarray(motion, dtype=np.float64)
        if motion.shape[0] < 6:
            padded = np.zeros(6, dtype=np.float64)
            padded[: motion.shape[0]] = motion
            motion = padded
        mapped = _map_motion(motion[:6], self.config)
        reset_requested = self._button_rising(buttons, self.config.home_button_index)

        if self.config.use_gripper and self._button_rising(buttons, self.config.gripper_button_index):
            self._gripper_cmd = 1.0 - self._gripper_cmd if self.config.toggle_gripper else 0.0
        self._last_buttons = [int(bool(value)) for value in buttons]

        action: RobotAction = {key: float(value) for key, value in zip(DELTA_EE_KEYS, mapped, strict=True)}
        if self.config.use_gripper:
            action["gripper_cmd_bin"] = float(self._gripper_cmd)
        action["reset_requested"] = bool(reset_requested)
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        close_reader = getattr(self._reader, "close", None)
        if callable(close_reader):
            close_reader()
        self._is_connected = False

    def _button_rising(self, buttons: list[int], index: int) -> bool:
        current = bool(buttons[index]) if 0 <= index < len(buttons) else False
        previous = bool(self._last_buttons[index]) if 0 <= index < len(self._last_buttons) else False
        return current and not previous

    @staticmethod
    def _make_reader(config: FrankaSpaceMouseTeleopConfig):
        if config.input_backend == "spnav":
            return SpacenavReader(
                spnav_lib_path=config.spnav_lib_path,
                axis_scale=config.spnav_axis_scale,
                deadband=config.deadband,
                motion_timeout=config.motion_timeout,
            )
        raise ValueError(f"Unsupported SpaceMouse input_backend: {config.input_backend}")


class _SpnavMotionEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("z", ctypes.c_int),
        ("rx", ctypes.c_int),
        ("ry", ctypes.c_int),
        ("rz", ctypes.c_int),
        ("period", ctypes.c_uint),
        ("data", ctypes.POINTER(ctypes.c_int)),
    ]


class _SpnavButtonEvent(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("press", ctypes.c_int), ("bnum", ctypes.c_int)]


class _SpnavEvent(ctypes.Union):
    _fields_ = [("type", ctypes.c_int), ("motion", _SpnavMotionEvent), ("button", _SpnavButtonEvent)]


class _SpnavLibrary:
    def __init__(self, explicit_path: str | None = None):
        self.path = self._resolve_path(explicit_path)
        self._lib = ctypes.CDLL(self.path)
        self._bind()

    def open(self) -> None:
        if self._lib.spnav_open() == -1:
            raise RuntimeError("failed to connect to spacenavd")
        self._lib.spnav_client_name(b"lerobot_franka_spacemouse")
        self.remove_events()

    def remove_events(self) -> None:
        self._lib.spnav_remove_events(SPNAV_EVENT_ANY)

    def close(self) -> None:
        self._lib.spnav_close()

    def poll_event(self) -> tuple[int, _SpnavEvent]:
        event = _SpnavEvent()
        event_type = self._lib.spnav_poll_event(ctypes.byref(event))
        return event_type, event

    def _bind(self) -> None:
        self._lib.spnav_open.restype = ctypes.c_int
        self._lib.spnav_close.restype = ctypes.c_int
        self._lib.spnav_client_name.argtypes = [ctypes.c_char_p]
        self._lib.spnav_client_name.restype = ctypes.c_int
        self._lib.spnav_remove_events.argtypes = [ctypes.c_int]
        self._lib.spnav_remove_events.restype = ctypes.c_int
        self._lib.spnav_poll_event.argtypes = [ctypes.POINTER(_SpnavEvent)]
        self._lib.spnav_poll_event.restype = ctypes.c_int

    @staticmethod
    def _resolve_path(explicit_path: str | None) -> str:
        candidates: list[Path] = []
        if explicit_path:
            candidates.append(Path(explicit_path))
        candidates.extend(
            [
                Path("/usr/local/lib/libspnav.so"),
                Path("/usr/lib/x86_64-linux-gnu/libspnav.so"),
                Path("/usr/lib/libspnav.so"),
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError("libspnav not found; pass spnav_lib_path explicitly")


class SpacenavReader:
    """Non-blocking SpaceMouse reader backed by spacenavd/libspnav."""

    def __init__(
        self,
        *,
        spnav_lib_path: str | None = None,
        axis_scale: float = 500.0,
        deadband: float = 0.05,
        motion_timeout: float = 0.2,
    ):
        self.axis_scale = float(axis_scale)
        self.deadband = float(deadband)
        self.motion_timeout = max(0.0, float(motion_timeout))
        self._library = _SpnavLibrary(spnav_lib_path)
        self._raw_motion = np.zeros(6, dtype=np.float64)
        self._action = np.zeros(6, dtype=np.float64)
        self._buttons = [0, 0, 0, 0]
        self._last_motion_time = time.monotonic()

    def open(self) -> None:
        self._library.open()

    def get_action(self) -> tuple[np.ndarray, list[int]]:
        self._drain_events()
        if self.motion_timeout > 0.0 and time.monotonic() - self._last_motion_time > self.motion_timeout:
            self._raw_motion = np.zeros(6, dtype=np.float64)
            self._action = np.zeros(6, dtype=np.float64)
        return self._action.copy(), list(self._buttons)

    def clear(self) -> None:
        self._library.remove_events()
        self._raw_motion.fill(0.0)
        self._action.fill(0.0)
        for index in range(len(self._buttons)):
            self._buttons[index] = 0
        self._last_motion_time = time.monotonic()

    def close(self) -> None:
        self._library.close()

    def _drain_events(self) -> None:
        while True:
            event_type, event = self._library.poll_event()
            if event_type == 0:
                return
            if event.type == SPNAV_EVENT_MOTION:
                self._raw_motion = np.array(
                    [
                        _normalize_axis(event.motion.x, self.axis_scale, self.deadband),
                        _normalize_axis(event.motion.y, self.axis_scale, self.deadband),
                        _normalize_axis(event.motion.z, self.axis_scale, self.deadband),
                        _normalize_axis(event.motion.rx, self.axis_scale, self.deadband),
                        _normalize_axis(event.motion.ry, self.axis_scale, self.deadband),
                        _normalize_axis(event.motion.rz, self.axis_scale, self.deadband),
                    ],
                    dtype=np.float64,
                )
                x, y, z, roll, pitch, yaw = self._raw_motion
                self._action = np.array([-y, x, z, -roll, -pitch, -yaw], dtype=np.float64)
                self._last_motion_time = time.monotonic()
            elif event.type == SPNAV_EVENT_BUTTON:
                button_index = int(event.button.bnum)
                if 0 <= button_index < len(self._buttons):
                    self._buttons[button_index] = int(bool(event.button.press))


def _map_motion(raw_motion: np.ndarray, config: FrankaSpaceMouseTeleopConfig) -> np.ndarray:
    signed = np.asarray(raw_motion, dtype=np.float64).copy()
    signed *= np.asarray(config.channel_signs, dtype=np.float64)

    linear_source = signed[:3].copy()
    angular_source = signed[3:6].copy()
    mapped_linear = np.array(
        [linear_source[RAW_AXIS_INDEX[axis]] for axis in config.linear_axis_map], dtype=np.float64
    )
    mapped_angular = np.array(
        [angular_source[RAW_AXIS_INDEX[axis]] for axis in config.angular_axis_map], dtype=np.float64
    )
    mapped_angular *= np.asarray(config.angular_output_signs, dtype=np.float64)
    mapped = np.concatenate([mapped_linear, mapped_angular])
    if config.mirror:
        mapped *= np.asarray([-1, -1, 1, -1, -1, -1], dtype=np.float64)

    mapped[:3] *= float(config.pose_scaler[0])
    mapped[3:] *= float(config.pose_scaler[1])
    return mapped


def _normalize_axis(value: float, axis_scale: float, deadband: float) -> float:
    axis_scale = max(float(axis_scale), 1.0)
    deadband = max(0.0, min(float(deadband), 0.95))
    normalized = max(-1.0, min(1.0, float(value) / axis_scale))
    if abs(normalized) <= deadband:
        return 0.0
    scaled = (abs(normalized) - deadband) / (1.0 - deadband)
    return math.copysign(scaled, normalized)


def _validate_axis_map(axes: tuple[str, str, str]) -> None:
    if len(axes) != 3 or set(axes) != {"x", "y", "z"}:
        raise ValueError(f"axis map must contain x/y/z exactly once, got {axes}")
