from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any
from urllib.parse import urlparse

import numpy as np

from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .defaults import get_control_host
from .state_cache import FrankaStateCache

JOINT_KEYS = tuple(f"joint_{idx}.pos" for idx in range(1, 8))
DELTA_EE_KEYS = tuple(f"delta_ee_pose.{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz"))
POSE_KEYS = ("x", "y", "z", "R", "P", "Y")
VITA_HOME_JOINTS = (
    0.04169132933020592,
    -0.04916822165250778,
    0.02022768370807171,
    -2.2447776794433594,
    0.020603381097316742,
    2.2057414054870605,
    0.047726742923259735,
)


@RobotConfig.register_subclass("hardware_test_franka")
@dataclass(kw_only=True)
class FrankaRobotConfig(RobotConfig):
    """Franka adapter for hardware tests, not yet registered in LeRobot factories."""

    id: str | None = "franka"
    action_mode: str = "delta_ee_pose"  # "delta_ee_pose" or "joint"
    cartesian_action_units: str = "delta"  # "delta" per LeRobot frame, or "velocity" in SI units
    control_hz: float = 15.0
    max_linear_velocity: float = 0.05
    max_angular_velocity: float = 0.40
    command_duration_ms: int = 300
    validate_connection: bool = True

    base_url: str | None = None
    control_host: str = field(default_factory=get_control_host)
    velocity_transport: str = "zmq"  # "zmq" or "http"
    zmq_url: str | None = None
    timeout_s: float = 2.0
    state_cache_enabled: bool = True
    state_poll_hz: float = 30.0
    state_timeout_s: float = 0.2
    max_state_age_s: float = 0.5

    use_gripper: bool = True
    gripper_max_open: float = 0.085
    initial_gripper_cmd: float = 1.0

    home_joints: tuple[float, ...] | None = VITA_HOME_JOINTS
    home_timeout_s: float = 20.0
    home_joint_tolerance_rad: float = 0.02
    home_stable_samples: int = 5
    control_transition_timeout_s: float = 2.0
    control_status_poll_s: float = 0.02

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    camera_shapes: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    camera_read_mode: str = "latest"  # "latest", "async", or "read"
    max_camera_age_s: float = 0.25
    camera_async_timeout_s: float = 0.2

    def __post_init__(self):
        super().__post_init__()
        if self.action_mode not in {"delta_ee_pose", "joint"}:
            raise ValueError(f"Unsupported action_mode: {self.action_mode}")
        if self.cartesian_action_units not in {"delta", "velocity"}:
            raise ValueError(f"Unsupported cartesian_action_units: {self.cartesian_action_units}")
        if self.control_hz <= 0.0:
            raise ValueError("control_hz must be positive")
        if self.use_gripper and self.gripper_max_open <= 0.0:
            raise ValueError("gripper_max_open must be positive")
        if self.state_poll_hz <= 0.0:
            raise ValueError("state_poll_hz must be positive")
        if self.state_timeout_s <= 0.0:
            raise ValueError("state_timeout_s must be positive")
        if self.max_state_age_s <= 0.0:
            raise ValueError("max_state_age_s must be positive")
        if self.home_joint_tolerance_rad <= 0.0:
            raise ValueError("home_joint_tolerance_rad must be positive")
        if self.home_stable_samples <= 0:
            raise ValueError("home_stable_samples must be positive")
        if self.control_transition_timeout_s <= 0.0:
            raise ValueError("control_transition_timeout_s must be positive")
        if self.control_status_poll_s <= 0.0:
            raise ValueError("control_status_poll_s must be positive")
        if self.camera_read_mode not in {"latest", "async", "read"}:
            raise ValueError(f"Unsupported camera_read_mode: {self.camera_read_mode}")
        if self.max_camera_age_s <= 0.0:
            raise ValueError("max_camera_age_s must be positive")
        if self.camera_async_timeout_s <= 0.0:
            raise ValueError("camera_async_timeout_s must be positive")


@dataclass(frozen=True)
class FrankaObservationSample:
    observation: RobotObservation
    state: dict[str, Any]
    state_timestamp_s: float


class FrankaRobot(Robot):
    """LeRobot Robot wrapper for a Franka FR3 controlled through VITA-style services."""

    config_class = FrankaRobotConfig
    name = "hardware_test_franka"

    def __init__(
        self,
        config: FrankaRobotConfig,
        *,
        client: Any | None = None,
        cameras: dict[str, Any] | None = None,
    ):
        super().__init__(config)
        self.config = config
        self._client = client or FrankaControlClient(
            base_url=config.base_url,
            control_host=config.control_host,
            velocity_transport=config.velocity_transport,
            zmq_url=config.zmq_url,
            timeout_s=config.timeout_s,
            command_duration_ms=config.command_duration_ms,
        )
        self.cameras = cameras if cameras is not None else make_cameras_from_configs(config.cameras)
        self._state_cache: FrankaStateCache | None = None
        self._is_connected = False
        self._last_gripper_cmd = float(config.initial_gripper_cmd)
        self._diagnostics: dict[str, float | int] = {}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        features: dict[str, type | tuple[int, int, int]] = {key: float for key in JOINT_KEYS}
        if self.config.use_gripper:
            features["gripper_width_norm"] = float
        features.update(self.config.camera_shapes)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.config.action_mode == "joint":
            features = {key: float for key in JOINT_KEYS}
        else:
            features = {key: float for key in DELTA_EE_KEYS}
        if self.config.use_gripper:
            features["gripper_cmd_bin"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def diagnostics(self) -> dict[str, float | int]:
        return dict(self._diagnostics)

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        connected_cameras = []
        try:
            for camera in self.cameras.values():
                camera.connect()
                connected_cameras.append(camera)
            if self.config.state_cache_enabled:
                self._state_cache = FrankaStateCache(
                    self._client,
                    poll_hz=self.config.state_poll_hz,
                    timeout_s=self.config.state_timeout_s,
                    max_age_s=self.config.max_state_age_s,
                )
                self._state_cache.start()
            elif self.config.validate_connection:
                self._client.get_curr()
            if self.config.velocity_transport == "zmq":
                init_zmq = getattr(self._client, "_get_zmq_sender", None)
                if callable(init_zmq):
                    init_zmq()
            self._is_connected = True
            if calibrate:
                self.calibrate()
        except BaseException:
            if self._state_cache is not None:
                self._state_cache.stop()
                self._state_cache = None
            for camera in connected_cameras:
                close = getattr(camera, "disconnect", None)
                if callable(close):
                    with contextlib.suppress(Exception):
                        close()
            close_client = getattr(self._client, "close", None)
            if callable(close_client):
                with contextlib.suppress(Exception):
                    close_client()
            self._is_connected = False
            raise

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_not_connected
    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        return self.get_observation_sample().observation

    @check_if_not_connected
    def get_observation_sample(self) -> FrankaObservationSample:
        if self._state_cache is not None:
            snapshot = self._state_cache.latest(max_age_s=self.config.max_state_age_s)
            state = snapshot.state
            gripper_state = snapshot.gripper_state
            state_timestamp_s = snapshot.timestamp_s
            self._diagnostics["state_age_ms"] = (time.monotonic() - snapshot.timestamp_s) * 1000.0
        else:
            state = self._client.get_curr()
            gripper_state = self._safe_gripper_state()
            state_timestamp_s = time.monotonic()
            self._diagnostics["state_age_ms"] = 0.0
        joints = _extract_joints(state)
        observation: RobotObservation = {
            key: float(value) for key, value in zip(JOINT_KEYS, joints, strict=True)
        }

        if self.config.use_gripper:
            width = _extract_gripper_width(state, gripper_state)
            observation["gripper_width_norm"] = float(np.clip(width / self.config.gripper_max_open, 0.0, 1.0))

        for key, camera in self.cameras.items():
            frame, camera_diagnostics = _read_fresh_camera_frame(
                camera,
                read_mode=self.config.camera_read_mode,
                max_age_s=self.config.max_camera_age_s,
                async_timeout_s=self.config.camera_async_timeout_s,
            )
            observation[key] = frame
            for diagnostic_key, value in camera_diagnostics.items():
                self._diagnostics[f"{key}_{diagnostic_key}"] = value
        return FrankaObservationSample(
            observation=observation,
            state=state,
            state_timestamp_s=state_timestamp_s,
        )

    def send_zero_cartesian_velocity(self) -> None:
        self._send_zero_cartesian_velocity()

    def start_home_async(self) -> None:
        if self.config.home_joints is None:
            raise RuntimeError("Franka home_joints is not configured")
        self._client.joint_position_control(
            list(self.config.home_joints),
            mode="absolute",
            is_async=True,
            timeout=self.config.timeout_s,
        )

    def stop_home_motion(self) -> None:
        self._client.stop_joint_position_control()
        self._wait_for_motion_completion()

    def recover(self) -> None:
        self._client.recover()

    def motion_complete(self) -> bool:
        return bool(self._client.join_motion(timeout_s=0.0))

    def is_home_state(self, state: dict[str, Any]) -> bool:
        if self.config.home_joints is None:
            return False
        joints = np.asarray(_extract_joints(state), dtype=np.float64)
        target = np.asarray(self.config.home_joints, dtype=np.float64)
        tolerance = float(self.config.home_joint_tolerance_rad)
        boundary_epsilon = np.finfo(np.float64).eps * max(1.0, tolerance)
        return bool(np.max(np.abs(joints - target)) <= tolerance + boundary_epsilon)

    def quiesce_cartesian_velocity_control(self) -> None:
        deadline_s = time.monotonic() + self.config.control_transition_timeout_s
        if self.config.velocity_transport == "zmq":
            baseline_status = self._client.velocity_loop_status()
            baseline_latest_seq = _required_velocity_status_int(baseline_status, "latest_seq")
            sender = self._client._get_zmq_sender()
            zero_command = _normalize_velocity_command(
                dict.fromkeys(POSE_KEYS, 0.0),
                self.config.command_duration_ms,
            )
            sender.submit(zero_command)
            sender.send_latest_once()
            self._wait_for_zmq_zero_acknowledgement(
                baseline_latest_seq=baseline_latest_seq,
                deadline_s=deadline_s,
            )

        self._client.stop_cartesian_velocity_control_direct()
        self._wait_for_motion_completion(deadline_s=deadline_s)

    def _wait_for_zmq_zero_acknowledgement(self, *, baseline_latest_seq: int, deadline_s: float) -> None:
        while True:
            status = self._client.velocity_loop_status()
            latest_seq = _required_velocity_status_int(status, "latest_seq")
            dispatched_seq = _required_velocity_status_int(status, "dispatched_seq")
            motion_active = _required_velocity_motion_active(status)
            if latest_seq > baseline_latest_seq and dispatched_seq >= latest_seq and not motion_active:
                return
            self._sleep_for_control_status(
                deadline_s=deadline_s,
                timeout_message="Timed out waiting for Franka velocity zero acknowledgement",
            )

    def _wait_for_motion_completion(self, *, deadline_s: float | None = None) -> None:
        if deadline_s is None:
            deadline_s = time.monotonic() + self.config.control_transition_timeout_s
        while not self.motion_complete():
            self._sleep_for_control_status(
                deadline_s=deadline_s,
                timeout_message="Timed out waiting for Franka motion completion",
            )

    def _sleep_for_control_status(self, *, deadline_s: float, timeout_message: str) -> None:
        remaining_s = deadline_s - time.monotonic()
        if remaining_s <= 0.0:
            raise TimeoutError(timeout_message)
        time.sleep(min(self.config.control_status_poll_s, remaining_s))

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if bool(action.get("reset_requested", False)):
            before_send_t = time.monotonic()
            self._send_zero_cartesian_velocity()
            self._diagnostics["action_send_latency_ms"] = (time.monotonic() - before_send_t) * 1000.0
            self._send_home()
            if self.config.use_gripper:
                self._set_gripper(1.0)
            return _zero_action_like(self.action_features)

        before_send_t = time.monotonic()
        if self.config.action_mode == "joint":
            sent_action = self._send_joint_action(action)
        else:
            sent_action = self._send_delta_ee_action(action)
        self._diagnostics["action_send_latency_ms"] = (time.monotonic() - before_send_t) * 1000.0

        if self.config.use_gripper and "gripper_cmd_bin" in action:
            self._set_gripper(float(action["gripper_cmd_bin"]))
            sent_action["gripper_cmd_bin"] = float(1.0 if float(action["gripper_cmd_bin"]) >= 0.5 else 0.0)
        return sent_action

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            self._send_zero_cartesian_velocity()
        finally:
            if self._state_cache is not None:
                self._state_cache.stop()
                self._state_cache = None
            for camera in self.cameras.values():
                camera.disconnect()
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
            self._is_connected = False

    def _send_delta_ee_action(self, action: RobotAction) -> RobotAction:
        values = np.array([float(action.get(key, 0.0) or 0.0) for key in DELTA_EE_KEYS], dtype=np.float64)
        if self.config.cartesian_action_units == "delta":
            values *= self.config.control_hz

        values[:3] = np.clip(values[:3], -self.config.max_linear_velocity, self.config.max_linear_velocity)
        values[3:] = np.clip(values[3:], -self.config.max_angular_velocity, self.config.max_angular_velocity)
        recorded_values = (
            values / self.config.control_hz if self.config.cartesian_action_units == "delta" else values
        )
        command = {
            "x": float(values[0]),
            "y": float(values[1]),
            "z": float(values[2]),
            "R": float(values[3]),
            "P": float(values[4]),
            "Y": float(values[5]),
            "duration": int(self.config.command_duration_ms),
            "is_async": 1,
        }
        _call_first(self._client, ("cartesian_velocity_control", "_cartesian_velocity_control"), command)
        sent_action: RobotAction = {
            key: float(value) for key, value in zip(DELTA_EE_KEYS, recorded_values, strict=True)
        }
        if self.config.use_gripper:
            sent_action["gripper_cmd_bin"] = float(action.get("gripper_cmd_bin", self._last_gripper_cmd))
        return sent_action

    def _send_joint_action(self, action: RobotAction) -> RobotAction:
        joints = [float(action[key]) for key in JOINT_KEYS]
        _call_first(
            self._client,
            ("joint_position_control",),
            joints,
            mode="absolute",
            timeout=None,
        )
        sent_action: RobotAction = {key: float(value) for key, value in zip(JOINT_KEYS, joints, strict=True)}
        if self.config.use_gripper:
            sent_action["gripper_cmd_bin"] = float(action.get("gripper_cmd_bin", self._last_gripper_cmd))
        return sent_action

    def _send_home(self) -> None:
        if self.config.home_joints is None:
            return
        _call_first(
            self._client,
            ("joint_position_control",),
            list(self.config.home_joints),
            mode="absolute",
            timeout=self.config.home_timeout_s,
        )

    def _send_zero_cartesian_velocity(self) -> None:
        if hasattr(self._client, "stop_cartesian_velocity_control"):
            self._client.stop_cartesian_velocity_control()
            return
        if hasattr(self._client, "_stop_cartesian_velocity_control"):
            self._client._stop_cartesian_velocity_control()
            return
        command = {key: 0.0 for key in POSE_KEYS}
        _call_first(self._client, ("cartesian_velocity_control", "_cartesian_velocity_control"), command)

    def _set_gripper(self, target: float) -> None:
        target = 1.0 if target >= 0.5 else 0.0
        if target == self._last_gripper_cmd:
            return
        if target >= 0.5:
            _call_first(self._client, ("gripper_open",))
        else:
            _call_first(self._client, ("gripper_close",))
        self._last_gripper_cmd = target

    def _safe_gripper_state(self) -> dict[str, Any]:
        getter = getattr(self._client, "gripper_get_state", None)
        if not callable(getter):
            return {}
        try:
            value = getter()
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}


class FrankaControlError(RuntimeError):
    pass


def _validated_reply(payload: Any, *, allow_incomplete: bool = False) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise FrankaControlError(f"Expected a JSON object reply, got {type(payload).__name__}")
    if payload.get("is_ok"):
        return payload
    if allow_incomplete and "error" not in payload and "error_type" not in payload:
        return payload

    error_type = payload.get("error_type") or "FrankaControlError"
    error = payload.get("error") or "Control request failed"
    raise FrankaControlError(f"{error_type}: {error}")


class FrankaControlClient:
    """Small VITA-compatible HTTP/ZMQ client used by the hardware-test adapter."""

    def __init__(
        self,
        *,
        base_url: str | None,
        control_host: str,
        velocity_transport: str,
        zmq_url: str | None,
        timeout_s: float,
        command_duration_ms: int,
    ):
        self.base_url = (base_url or f"http://{control_host}:29000/ctl").rstrip("/")
        self.velocity_transport = velocity_transport
        self.timeout_s = float(timeout_s)
        self.command_duration_ms = int(command_duration_ms)
        self._session = None
        self._zmq_sender = None
        self.zmq_url = zmq_url or _control_http_url_to_zmq_url(self.base_url)

    def get_curr(self, timeout: float | None = None) -> dict[str, Any]:
        return self._get("get_curr", timeout=timeout)

    def gripper_get_state(self, timeout: float | None = None) -> dict[str, Any]:
        return self._get("gripper_state", timeout=timeout)

    def recover(self) -> dict[str, Any]:
        return self._get("recover")

    def velocity_loop_status(self) -> dict[str, Any]:
        return self._get("velocity_ws_status")

    def cartesian_velocity_control(self, data: dict[str, Any]) -> dict[str, Any]:
        command = _normalize_velocity_command(data, self.command_duration_ms)
        if self.velocity_transport == "zmq":
            self._get_zmq_sender().submit(command)
            return {"is_ok": 1, "queued": 1}
        return self._post("cartesian_velocity_control", command)

    def stop_cartesian_velocity_control(self) -> dict[str, Any]:
        return self.cartesian_velocity_control({key: 0.0 for key in POSE_KEYS})

    def stop_cartesian_velocity_control_direct(self) -> dict[str, Any]:
        return self._get("stop_cartesian_velocity_control")

    def joint_position_control(
        self,
        joints: list[float],
        *,
        mode: str = "absolute",
        is_async: bool = False,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "joint_position_control",
            {
                "joints_lst": [float(value) for value in joints],
                "mode": mode,
                "is_async": int(bool(is_async)),
            },
            timeout=timeout,
        )

    def stop_joint_position_control(self) -> dict[str, Any]:
        return self._get("stop_joint_position_control")

    def join_motion(self, timeout_s: float = 0.0) -> bool:
        reply = self._post(
            "join",
            {"timeout": float(timeout_s)},
            allow_incomplete=True,
        )
        return bool(reply.get("is_ok"))

    def gripper_open(self) -> dict[str, Any]:
        return self._post("gripper_control", {"mode": "release", "is_async": 1})

    def gripper_close(self) -> dict[str, Any]:
        return self._post(
            "gripper_control",
            {"mode": "move", "width": 0.0, "speed": 255, "force": 150, "is_async": 1},
        )

    def close(self) -> None:
        if self._zmq_sender is not None:
            self._zmq_sender.close()
            self._zmq_sender = None

    def _get(self, path: str, timeout: float | None = None) -> dict[str, Any]:
        response = self._requests_session().get(
            f"{self.base_url}/{path.lstrip('/')}",
            timeout=self.timeout_s if timeout is None else timeout,
        )
        response.raise_for_status()
        return _validated_reply(response.json())

    def _post(
        self,
        path: str,
        data: dict[str, Any],
        timeout: float | None = None,
        *,
        allow_incomplete: bool = False,
    ) -> dict[str, Any]:
        response = self._requests_session().post(
            f"{self.base_url}/{path.lstrip('/')}",
            json=data,
            timeout=self.timeout_s if timeout is None else timeout,
        )
        response.raise_for_status()
        return _validated_reply(response.json(), allow_incomplete=allow_incomplete)

    def _requests_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.trust_env = False
        return self._session

    def _get_zmq_sender(self):
        if self._zmq_sender is None:
            self._zmq_sender = LatestZmqVelocitySender(self.zmq_url)
            self._zmq_sender.ensure_connected()
        return self._zmq_sender


class LatestZmqVelocitySender:
    """Latest-only ZMQ PUSH sender matching VITA's no-backlog velocity semantics."""

    def __init__(self, endpoint: str, send_period_s: float | None = None):
        self.endpoint = endpoint
        self.send_period_s = float(
            send_period_s if send_period_s is not None else os.getenv("FRANKA_ZMQ_SEND_PERIOD_S", "0.02")
        )
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._latest: dict[str, Any] | None = None
        self._latest_seq = 0
        self._sent_seq = 0
        self._running = True
        self._socket = None
        self._thread = threading.Thread(target=self._run, name="franka-zmq-velocity-sender", daemon=True)
        self._thread.start()

    def submit(self, data: dict[str, Any]) -> None:
        with self._wake:
            self._latest_seq += 1
            self._latest = dict(data, seq=self._latest_seq)
            self._wake.notify()

    def send_latest_once(self) -> bool:
        with self._lock:
            if self._latest is None or self._latest_seq == self._sent_seq:
                return False
            data = dict(self._latest)
            self._sent_seq = self._latest_seq
        self._send(data)
        return True

    def ensure_connected(self) -> None:
        self._ensure_socket()

    def close(self) -> None:
        self.submit({key: 0.0 for key in POSE_KEYS})
        self.send_latest_once()
        with self._wake:
            self._running = False
            self._wake.notify_all()
        self._thread.join(timeout=1.0)
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    def _run(self) -> None:
        while True:
            with self._wake:
                if not self._running:
                    return
                self._wake.wait(timeout=self.send_period_s)
                if not self._running:
                    return
            try:
                self.send_latest_once()
            except BaseException:
                if self._socket is not None:
                    self._socket.close(linger=0)
                    self._socket = None
                time.sleep(min(self.send_period_s, 0.05))

    def _send(self, data: dict[str, Any]) -> None:
        self._ensure_socket()
        payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
        try:
            import zmq

            self._socket.send(payload, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def _ensure_socket(self) -> None:
        if self._socket is not None:
            return
        try:
            import zmq
        except Exception as exc:
            raise ImportError(
                "Franka ZMQ velocity transport requires pyzmq in the LeRobot environment."
            ) from exc

        context = zmq.Context.instance()
        self._socket = context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.connect(self.endpoint)


def _extract_joints(state: dict[str, Any]) -> list[float]:
    for key in ("joint", "joints", "joint_positions"):
        value = state.get(key)
        if value is not None:
            joints = list(value)
            break
    else:
        raise KeyError("Franka state did not include joint/joints/joint_positions")
    if len(joints) != 7:
        raise ValueError(f"Expected 7 Franka joints, got {len(joints)}")
    return [float(value) for value in joints]


def _extract_gripper_width(state: dict[str, Any], gripper_state: dict[str, Any]) -> float:
    for source in (gripper_state, state):
        for key in ("width", "gripper_width"):
            if key in source and source[key] is not None:
                return float(source[key])
    if "gripper_width_norm" in state:
        return float(state["gripper_width_norm"])
    return 0.0


def _normalize_velocity_command(data: dict[str, Any], default_duration_ms: int) -> dict[str, Any]:
    command = {key: float(data.get(key, 0.0) or 0.0) for key in POSE_KEYS}
    command["duration"] = int(data.get("duration", default_duration_ms) or default_duration_ms)
    command["is_async"] = int(data.get("is_async", 1))
    return command


def _required_velocity_status_int(status: dict[str, Any], field_name: str) -> int:
    if field_name not in status:
        raise FrankaControlError(f"Franka velocity status is missing {field_name}")
    value = status[field_name]
    if isinstance(value, bool):
        raise FrankaControlError(f"Franka velocity status {field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise FrankaControlError(f"Franka velocity status {field_name} must be an integer") from exc


def _required_velocity_motion_active(status: dict[str, Any]) -> bool:
    if "motion_active" not in status:
        raise FrankaControlError("Franka velocity status is missing motion_active")
    value = status["motion_active"]
    if not isinstance(value, (bool, int)) or value not in (False, True, 0, 1):
        raise FrankaControlError("Franka velocity status motion_active must be boolean")
    return bool(value)


def _read_fresh_camera_frame(
    camera: Any,
    *,
    read_mode: str,
    max_age_s: float,
    async_timeout_s: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    diagnostics: dict[str, float | int] = {}
    if read_mode == "latest" and hasattr(camera, "latest"):
        sample = camera.latest(
            max_age_ms=max(1, int(max_age_s * 1000)),
            timeout_ms=max(1, int(async_timeout_s * 1000)),
        )
        frame = sample.image
        diagnostics["image_age_ms"] = float(sample.image_age_ms)
        diagnostics["dropped_frame_count"] = int(sample.dropped_frame_count)
    elif read_mode == "latest" and hasattr(camera, "read_latest"):
        before_read_t = time.monotonic()
        frame = camera.read_latest(max_age_ms=max(1, int(max_age_s * 1000)))
        diagnostics["image_age_ms"] = (time.monotonic() - before_read_t) * 1000.0
    elif read_mode == "async" and hasattr(camera, "async_read"):
        before_read_t = time.monotonic()
        frame = camera.async_read(timeout_ms=max(1, int(async_timeout_s * 1000)))
        diagnostics["image_age_ms"] = (time.monotonic() - before_read_t) * 1000.0
    else:
        before_read_t = time.monotonic()
        frame = camera.read()
        diagnostics["image_age_ms"] = (time.monotonic() - before_read_t) * 1000.0

    if isinstance(frame, tuple) and len(frame) >= 1:
        frame = frame[0]
    return np.asarray(frame).copy(), diagnostics


def _control_http_url_to_zmq_url(base_url: str) -> str:
    parsed = urlparse(base_url.rstrip("/"))
    host = parsed.hostname or "localhost"
    port = int(os.getenv("FRANKA_ZMQ_PORT", "29010"))
    return f"tcp://{host}:{port}"


def _call_first(target: Any, method_names: tuple[str, ...], *args, **kwargs):
    for name in method_names:
        method = getattr(target, name, None)
        if callable(method):
            return method(*args, **kwargs)
    raise AttributeError(f"{target!r} does not implement any of {method_names}")


def _zero_action_like(features: dict[str, type]) -> RobotAction:
    return {key: 0.0 for key in features}
