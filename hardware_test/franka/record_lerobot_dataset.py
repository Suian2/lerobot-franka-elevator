from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from threading import Event
from typing import Any, Protocol

import numpy as np

from hardware_test.franka.franka_robot import FrankaControlError
from hardware_test.franka.state_cache import StaleFrankaStateError
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep


class _FrankaObservationSampleLike(Protocol):
    state: dict[str, Any]
    state_timestamp_s: float


_MEASURED_EE_ACTION_KEYS = tuple(f"delta_ee_pose.{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz"))


def matrix4_to_xyz_rpy(matrix: Any) -> list[float]:
    """Convert a base-frame homogeneous transform to ``[x, y, z, roll, pitch, yaw]``."""

    try:
        matrix = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("End-effector transform must be a finite 4x4 matrix") from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("End-effector transform must be a finite 4x4 matrix")

    x = float(matrix[0][3])
    y = float(matrix[1][3])
    z = float(matrix[2][3])
    r00 = float(matrix[0][0])
    r10 = float(matrix[1][0])
    r11 = float(matrix[1][1])
    r12 = float(matrix[1][2])
    r20 = float(matrix[2][0])
    r21 = float(matrix[2][1])
    r22 = float(matrix[2][2])
    sy = math.sqrt(r00 * r00 + r10 * r10)
    if sy < 1e-6:
        roll = math.atan2(-r12, r11)
        pitch = math.atan2(-r20, sy)
        yaw = 0.0
    else:
        roll = math.atan2(r21, r22)
        pitch = math.atan2(-r20, sy)
        yaw = math.atan2(r10, r00)
    return [x, y, z, roll, pitch, yaw]


def end_effector_pose(state: dict[str, Any]) -> list[float]:
    """Extract the Cartesian end-effector pose from a Franka state payload."""

    return matrix4_to_xyz_rpy(state["ee"])


def measured_ee_action(
    previous: _FrankaObservationSampleLike,
    current: _FrankaObservationSampleLike,
    *,
    units: str,
    gripper_cmd: float,
) -> RobotAction:
    """Build an action from consecutive measured Franka end-effector poses."""

    dt_s = float(current.state_timestamp_s) - float(previous.state_timestamp_s)
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("Measured Franka state timestamp delta must be positive")
    if units not in {"delta", "velocity"}:
        raise ValueError(f"Unsupported measured EE action units: {units}")

    previous_pose = end_effector_pose(previous.state)
    current_pose = end_effector_pose(current.state)
    values = [
        current_value - previous_value
        for previous_value, current_value in zip(previous_pose, current_pose, strict=True)
    ]
    values[3:] = [(value + math.pi) % (2.0 * math.pi) - math.pi for value in values[3:]]
    if units == "velocity":
        values = [value / dt_s for value in values]
    action: RobotAction = {
        key: float(value) for key, value in zip(_MEASURED_EE_ACTION_KEYS, values, strict=True)
    }
    action["gripper_cmd_bin"] = float(float(gripper_cmd) >= 0.5)
    return action


def build_lerobot_features(
    robot: Any,
    teleop: Any | None = None,
    *,
    use_videos: bool = True,
) -> dict[str, dict]:
    """Build LeRobotDataset features from the local Franka robot and teleop adapters."""

    action_features = teleop.action_features if teleop is not None else robot.action_features
    features = combine_feature_dicts(
        hw_to_dataset_features(action_features, ACTION, use_video=use_videos),
        hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=use_videos),
    )
    return features


def make_lerobot_frame(
    features: dict[str, dict],
    observation: RobotObservation,
    action: RobotAction,
    *,
    task: str,
) -> dict[str, Any]:
    """Pack one synchronized observation/action pair for LeRobotDataset.add_frame()."""

    observation_frame = build_dataset_frame(features, observation, prefix=OBS_STR)
    action_frame = build_dataset_frame(features, action, prefix=ACTION)
    frame = {**observation_frame, **action_frame, "task": task}
    return frame



def create_lerobot_dataset(
    *,
    repo_id: str,
    root: str | None,
    fps: int,
    robot: Any,
    teleop: Any | None,
    task: str,
    use_videos: bool = True,
    image_writer_processes: int = 0,
    image_writer_threads_per_camera: int = 4,
    streaming_encoding: bool = False,
    encoder_queue_maxsize: int = 30,
    encoder_threads: int | None = None,
):
    """Create a LeRobotDataset for Franka recording without importing it at module load time."""

    _discard_empty_initialized_lerobot_root(root)
    _reject_existing_lerobot_root(root)

    from lerobot.datasets import LeRobotDataset

    features = build_lerobot_features(
        robot,
        teleop,
        use_videos=use_videos,
    )
    num_cameras = sum(1 for value in robot.observation_features.values() if isinstance(value, tuple))
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type=robot.name,
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads_per_camera * num_cameras,
        streaming_encoding=streaming_encoding,
        encoder_queue_maxsize=encoder_queue_maxsize,
        encoder_threads=encoder_threads,
    )
    return dataset


def _discard_empty_initialized_lerobot_root(root: str | None) -> None:
    """Remove a create-only dataset root that contains metadata but no recorded data."""

    if root is None:
        return

    root_path = Path(root)
    if not root_path.exists():
        return

    info_path = root_path / "meta" / "info.json"
    if not info_path.is_file():
        return

    try:
        info = json.loads(info_path.read_text())
    except (OSError, json.JSONDecodeError):
        return

    if any(info.get(key) != 0 for key in ("total_episodes", "total_frames", "total_tasks")):
        return

    files = {path for path in root_path.rglob("*") if path.is_file()}
    if files != {info_path}:
        return

    shutil.rmtree(root_path)


def _reject_existing_lerobot_root(root: str | None) -> None:
    if root is None:
        return
    root_path = Path(root)
    if root_path.exists():
        raise FileExistsError(
            f"LeRobot dataset root already exists: {root_path}. "
            "Choose a new --root or remove the existing directory after backing it up."
        )


def record_lerobot_episode(
    *,
    robot: Any,
    teleop: Any,
    dataset: Any,
    fps: int,
    duration_s: float,
    task: str,
    stop_event: Event | None = None,
    max_consecutive_state_misses: int = 60,
    max_state_wait_s: float = 1.0,
    state_retry_sleep_s: float = 0.01,
    tolerate_robot_faults: bool = False,
) -> int:
    """Record one episode into an already-created LeRobotDataset.

    The observation and action are copied into one frame before enqueueing, so later
    async image/video writing does not change frame/action alignment.
    """

    # ========== 新增：调试计时开关 ==========
    DEBUG_TIMING = True   # 改为 False 可关闭每帧计时打印

    features = dataset.features
    control_interval_s = 1.0 / float(fps)
    frames = 0
    consecutive_state_misses = 0
    stale_started_t: float | None = None
    robot_faulted = False
    last_sent_action: RobotAction | None = None
    start_t = time.perf_counter()
    last_diagnostics_t = start_t
    while time.perf_counter() - start_t < duration_s:
        loop_t = time.perf_counter()          # 循环起始时间
        if stop_event is not None and stop_event.is_set():
            break

        # ---------- 1. get_observation ----------
        t_get_obs = None
        try:
            observation = robot.get_observation()
            consecutive_state_misses = 0
            stale_started_t = None
            t_get_obs = time.perf_counter()
        except StaleFrankaStateError as exc:
            consecutive_state_misses += 1
            now = time.perf_counter()
            if stale_started_t is None:
                stale_started_t = now
            state_wait_exhausted = (
                consecutive_state_misses > max_consecutive_state_misses
                or now - stale_started_t > max_state_wait_s
            )
            if state_wait_exhausted:
                if not tolerate_robot_faults:
                    raise
                if not robot_faulted:
                    _report_tolerated_robot_fault(robot, exc)
                    robot_faulted = True
            time.sleep(max(0.0, float(state_retry_sleep_s)))
            continue

        # ---------- 2. teleop.get_action ----------
        t_get_action = None
        action = teleop.get_action()
        t_get_action = time.perf_counter()

        # ---------- 3. robot.send_action ----------
        t_send = None
        if robot_faulted:
            sent_action = _stationary_action_after_fault(
                robot.action_features,
                last_sent_action=last_sent_action,
                attempted_action=action,
            )
            t_send = time.perf_counter()   # 模拟快速返回
        else:
            try:
                sent_action = robot.send_action(action)
                t_send = time.perf_counter()
            except Exception as exc:
                if not tolerate_robot_faults or not _is_recoverable_robot_fault(exc):
                    raise
                _report_tolerated_robot_fault(robot, exc)
                robot_faulted = True
                dt_s = time.perf_counter() - loop_t
                precise_sleep(max(0.0, control_interval_s - dt_s))
                continue
            else:
                last_sent_action = dict(sent_action)

        # ---------- 4. dataset.add_frame ----------
        t_add = None
        dataset.add_frame(
            make_lerobot_frame(
                features,
                observation,
                sent_action,
                task=task,
            )
        )
        t_add = time.perf_counter()
        frames += 1

        # ---------- 打印每帧各阶段耗时 ----------
        if DEBUG_TIMING:
            dur_get_obs = (t_get_obs - loop_t) * 1000 if t_get_obs is not None else -1.0
            dur_get_action = (t_get_action - t_get_obs) * 1000 if t_get_obs is not None and t_get_action is not None else -1.0
            dur_send = (t_send - t_get_action) * 1000 if t_get_action is not None and t_send is not None else -1.0
            dur_add = (t_add - t_send) * 1000 if t_send is not None and t_add is not None else -1.0
            dur_total = (t_add - loop_t) * 1000 if t_add is not None else -1.0
            print(
                f"timing frame {frames}: "
                f"get_obs={dur_get_obs:.1f}ms, "
                f"get_action={dur_get_action:.1f}ms, "
                f"send_action={dur_send:.1f}ms, "
                f"add_frame={dur_add:.1f}ms, "
                f"total={dur_total:.1f}ms"
            )

        # ---------- 原有的每秒诊断信息（保留） ----------
        now = time.perf_counter()
        if now - last_diagnostics_t >= 1.0:
            diagnostics = getattr(robot, "diagnostics", {})
            loop_late_ms = max(0.0, (t_add - loop_t) - control_interval_s) * 1000.0 if t_add is not None else 0.0
            print(
                "record diagnostics "
                f"frames={frames} "
                f"image_age_ms={diagnostics.get('l515_image_age_ms', -1):.1f} "
                f"state_age_ms={diagnostics.get('state_age_ms', -1):.1f} "
                f"action_send_latency_ms={diagnostics.get('action_send_latency_ms', -1):.1f} "
                f"loop_late_ms={loop_late_ms:.1f} "
                f"dropped_frame_count={diagnostics.get('l515_dropped_frame_count', 0)}",
                flush=True,
            )
            last_diagnostics_t = now

        # ---------- 睡眠补偿 ----------
        dt_s = time.perf_counter() - loop_t
        precise_sleep(max(0.0, control_interval_s - dt_s))

    return frames

def _is_recoverable_robot_fault(exc: Exception) -> bool:
    if isinstance(exc, FrankaControlError):
        return True
    try:
        from requests.exceptions import InvalidURL, RequestException
    except ImportError:
        return False
    return isinstance(exc, RequestException) and not isinstance(exc, InvalidURL)


def _report_tolerated_robot_fault(robot: Any, exc: Exception) -> None:
    print(
        "record warning: robot fault tolerated; motion commands are disabled for "
        f"the rest of this episode ({type(exc).__name__}: {exc})",
        flush=True,
    )
    stop = getattr(robot, "send_zero_cartesian_velocity", None)
    if callable(stop):
        try:
            stop()
        except Exception as stop_exc:
            print(
                "record warning: failed to send zero Cartesian velocity after robot fault "
                f"({type(stop_exc).__name__}: {stop_exc})",
                flush=True,
            )


def _stationary_action_after_fault(
    action_features: dict[str, Any],
    *,
    last_sent_action: RobotAction | None,
    attempted_action: RobotAction,
) -> RobotAction:
    action: RobotAction = {}
    for key in action_features:
        if key.startswith("delta_ee_pose."):
            action[key] = 0.0
        elif key == "gripper_cmd_bin":
            source = last_sent_action if last_sent_action is not None else attempted_action
            action[key] = float(float(source.get(key, 0.0)) >= 0.5)
        elif last_sent_action is not None and key in last_sent_action:
            action[key] = float(last_sent_action[key])
        else:
            action[key] = float(attempted_action.get(key, 0.0))
    return action


def record_hardware_smoke_episode(
    *,
    robot: Any,
    teleop: Any,
    fps: int,
    duration_s: float,
    stop_event: Event | None = None,
) -> int:
    """Run the hardware loop without importing/writing LeRobotDataset."""

    control_interval_s = 1.0 / float(fps)
    frames = 0
    start_t = time.perf_counter()
    while time.perf_counter() - start_t < duration_s:
        loop_t = time.perf_counter()
        if stop_event is not None and stop_event.is_set():
            break

        robot.get_observation()
        action = teleop.get_action()
        robot.send_action(action)
        frames += 1

        dt_s = time.perf_counter() - loop_t
        precise_sleep(max(0.0, control_interval_s - dt_s))
    return frames
