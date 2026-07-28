from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from hardware_test.franka.record_lerobot_dataset import measured_ee_action
from hardware_test.franka.state_cache import StaleFrankaStateError


class RecorderState(StrEnum):
    CONNECTING = "connecting"
    IDLE = "idle"
    PREPARING = "preparing"
    RECORDING = "recording"
    HOMING_IDLE = "homing_idle"
    HOMING_RECORDING = "homing_recording"
    RECOVERING = "recovering"
    SAVING = "saving"
    FAULTED = "faulted"
    FATAL_ERROR = "fatal_error"
    PAUSING_CLOSE = "pausing_close"
    PAUSED_CLOSE = "paused_close"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    root: str
    task: str


@dataclass(frozen=True)
class RecorderOptions:
    fps: int
    duration_s: float = 0.0
    num_episodes: int = 0
    cartesian_action_units: str = "delta"
    max_consecutive_state_misses: int = 60
    max_state_wait_s: float = 1.0
    state_retry_sleep_s: float = 0.01

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.duration_s < 0.0:
            raise ValueError("duration_s must be non-negative")
        if self.num_episodes < 0:
            raise ValueError("num_episodes must be non-negative")
        if self.cartesian_action_units not in {"delta", "velocity"}:
            raise ValueError(f"Unsupported Cartesian action units: {self.cartesian_action_units}")


@dataclass(frozen=True)
class RecorderSnapshot:
    state: RecorderState
    message: str
    frame_count: int
    saved_episodes: int
    dataset_locked: bool
    pending_valid: bool
    wall_elapsed_s: float
    can_save_on_close: bool


class RecorderCommandKind(StrEnum):
    START = "start"
    END = "end"
    HOME = "home"
    CLEAR_FAULT = "clear_fault"
    PREPARE_CLOSE = "prepare_close"
    CANCEL_CLOSE = "cancel_close"
    CLOSE = "close"


@dataclass(frozen=True)
class RecorderCommand:
    kind: RecorderCommandKind
    spec: DatasetSpec | None = None
    save_pending: bool = False

    @classmethod
    def start(cls, spec: DatasetSpec) -> RecorderCommand:
        return cls(RecorderCommandKind.START, spec=spec)

    @classmethod
    def end(cls) -> RecorderCommand:
        return cls(RecorderCommandKind.END)

    @classmethod
    def home(cls) -> RecorderCommand:
        return cls(RecorderCommandKind.HOME)

    @classmethod
    def clear_fault(cls) -> RecorderCommand:
        return cls(RecorderCommandKind.CLEAR_FAULT)

    @classmethod
    def prepare_close(cls) -> RecorderCommand:
        return cls(RecorderCommandKind.PREPARE_CLOSE)

    @classmethod
    def cancel_close(cls) -> RecorderCommand:
        return cls(RecorderCommandKind.CANCEL_CLOSE)

    @classmethod
    def close(cls, *, save_pending: bool) -> RecorderCommand:
        return cls(RecorderCommandKind.CLOSE, save_pending=save_pending)


@dataclass(frozen=True)
class RecorderEvent:
    snapshot: RecorderSnapshot
    level: str = "info"
    message: str = ""


DatasetFactory = Callable[[DatasetSpec], Any]
FrameBuilder = Callable[..., dict[str, Any]]


class FrankaRecorderSession:
    """Synchronous Franka recorder state owned by one worker thread."""

    def __init__(
        self,
        *,
        robot: Any,
        teleop: Any,
        dataset_factory: DatasetFactory,
        options: RecorderOptions,
        frame_builder: FrameBuilder,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.robot = robot
        self.teleop = teleop
        self.dataset_factory = dataset_factory
        self.options = options
        self.frame_builder = frame_builder
        self._clock = clock
        self._state = RecorderState.CONNECTING
        self._message = "等待连接"
        self._dataset: Any | None = None
        self._dataset_spec: DatasetSpec | None = None
        self._frame_count = 0
        self._dataset_dirty = False
        self._saved_episodes = 0
        self._recording_started_s: float | None = None
        self._pending_valid = True
        self._held_home_sample: Any | None = None
        self._home_started_s: float | None = None
        self._home_stable_samples = 0
        self._home_join_timestamp_s: float | None = None
        self._resume_state: RecorderState | None = None
        self._input_armed = True
        self._paused_from: RecorderState | None = None

    @property
    def state(self) -> RecorderState:
        return self._state

    @property
    def can_save_on_close(self) -> bool:
        return (
            self._state is RecorderState.PAUSED_CLOSE
            and self._paused_from is RecorderState.RECORDING
            and self._frame_count > 0
            and self._pending_valid
        )

    @property
    def snapshot(self) -> RecorderSnapshot:
        wall_elapsed_s = 0.0
        if self._recording_started_s is not None:
            wall_elapsed_s = max(0.0, self._clock() - self._recording_started_s)
        return RecorderSnapshot(
            state=self._state,
            message=self._message,
            frame_count=self._frame_count,
            saved_episodes=self._saved_episodes,
            dataset_locked=self._dataset is not None,
            pending_valid=self._dataset_dirty and self._pending_valid,
            wall_elapsed_s=wall_elapsed_s,
            can_save_on_close=self.can_save_on_close,
        )

    def connect(self) -> None:
        if self._state is not RecorderState.CONNECTING:
            return
        self.robot.connect()
        try:
            self.teleop.connect()
        except BaseException:
            self.robot.disconnect()
            raise
        self._set_state(RecorderState.IDLE, "已连接，等待录制")

    def start_recording(self, spec: DatasetSpec) -> None:
        if self._state is not RecorderState.IDLE:
            raise RuntimeError(f"cannot start recording while {self._state.value}")
        if self.options.num_episodes and self._saved_episodes >= self.options.num_episodes:
            raise RuntimeError("episode limit has been reached")
        if self._dataset_spec is not None and spec != self._dataset_spec:
            raise ValueError("dataset fields are locked after the first recording starts")

        if self._dataset is None:
            self._set_state(RecorderState.PREPARING, "正在创建数据集")
            try:
                self._dataset = self.dataset_factory(spec)
            except BaseException:
                self._set_state(RecorderState.IDLE, "数据集创建失败")
                raise
            self._dataset_spec = spec

        self._frame_count = 0
        self._dataset_dirty = False
        self._pending_valid = True
        self._recording_started_s = self._clock()
        self._set_state(RecorderState.RECORDING, "正在录制")

    def end_recording(self) -> None:
        if self._state is not RecorderState.RECORDING:
            raise RuntimeError(f"cannot end recording while {self._state.value}")
        self.robot.send_zero_cartesian_velocity()
        self._set_state(RecorderState.SAVING, "正在保存")
        try:
            if self._frame_count > 0:
                self._dataset.save_episode()
                self._saved_episodes += 1
                message = f"已保存 episode {self._saved_episodes}"
            else:
                self._dataset.clear_episode_buffer(delete_images=True)
                message = "空 episode 已丢弃"
        except BaseException as exc:
            self._mark_fatal_data_error(exc)
            raise
        self._frame_count = 0
        self._dataset_dirty = False
        self._pending_valid = True
        self._recording_started_s = None
        self._set_state(RecorderState.IDLE, message)

    def tick(self) -> None:
        if self._state in {RecorderState.HOMING_IDLE, RecorderState.HOMING_RECORDING}:
            self._tick_home()
            return
        if self._state not in {RecorderState.IDLE, RecorderState.RECORDING}:
            return
        sample = self._capture_sample()
        action = self.teleop.get_action()
        if bool(action.get("reset_requested", False)):
            self.home()
            return
        if not self._input_armed:
            if not _is_neutral_action(action):
                self.robot.send_zero_cartesian_velocity()
                return
            self._input_armed = True
        sent_action = self.robot.send_action(action)
        if self._state is RecorderState.IDLE:
            return

        self._add_frame(sample, sent_action)
        if self.options.duration_s > 0.0 and self._frame_count / self.options.fps >= self.options.duration_s:
            self.end_recording()

    def home(self) -> None:
        if self._state not in {RecorderState.IDLE, RecorderState.RECORDING}:
            raise RuntimeError(f"cannot start home while {self._state.value}")
        recording = self._state is RecorderState.RECORDING
        self.teleop.clear_input()
        self._input_armed = False
        self.robot.quiesce_cartesian_velocity_control()
        self._held_home_sample = self._capture_sample()
        self.robot.start_home_async()
        self._home_started_s = self._clock()
        self._home_stable_samples = 0
        self._home_join_timestamp_s = None
        state = RecorderState.HOMING_RECORDING if recording else RecorderState.HOMING_IDLE
        self._set_state(state, "归位中，完成后自动保存" if recording else "正在归位")

    def clear_fault(self) -> None:
        if self._state in {
            RecorderState.CONNECTING,
            RecorderState.PREPARING,
            RecorderState.RECOVERING,
            RecorderState.SAVING,
            RecorderState.FATAL_ERROR,
            RecorderState.CLOSED,
        }:
            raise RuntimeError(f"cannot clear fault while {self._state.value}")

        resume_state = self._resume_state or self._state
        was_homing = resume_state in {RecorderState.HOMING_IDLE, RecorderState.HOMING_RECORDING}
        self.teleop.clear_input()
        self._input_armed = False
        self._set_state(RecorderState.RECOVERING, "正在清除 Fault")
        if was_homing:
            with contextlib.suppress(Exception):
                self.robot.stop_home_motion()
        with contextlib.suppress(Exception):
            self.robot.quiesce_cartesian_velocity_control()
        try:
            self.robot.recover()
            if was_homing:
                self.robot.stop_home_motion()
            self.robot.quiesce_cartesian_velocity_control()
            if was_homing:
                self._held_home_sample = self._capture_sample()
                self.robot.start_home_async()
                self._home_started_s = self._clock()
                self._home_stable_samples = 0
                self._home_join_timestamp_s = None
            self._pending_valid = True
            self._resume_state = None
            self._set_state(resume_state, "Fault 已清除")
        except BaseException:
            if self._dataset is not None and self._dataset_dirty:
                try:
                    self._dataset.clear_episode_buffer(delete_images=True)
                except Exception:
                    pass
                else:
                    self._dataset_dirty = False
            self._frame_count = 0
            self._pending_valid = False
            self._recording_started_s = None
            self._set_state(RecorderState.FATAL_ERROR, "Fault 清除失败，当前 episode 已丢弃")
            raise

    def prepare_close(self) -> None:
        if self._state in {RecorderState.CLOSING, RecorderState.CLOSED}:
            return
        if self._state in {RecorderState.CONNECTING, RecorderState.PREPARING, RecorderState.SAVING}:
            raise RuntimeError(f"cannot prepare close while {self._state.value}")
        self._paused_from = self._state
        self._set_state(RecorderState.PAUSING_CLOSE, "正在停止机械臂")
        try:
            self.teleop.clear_input()
            self._input_armed = False
            if self._paused_from in {RecorderState.HOMING_IDLE, RecorderState.HOMING_RECORDING}:
                self.robot.stop_home_motion()
            self.robot.quiesce_cartesian_velocity_control()
        except BaseException as exc:
            resume_state = self._paused_from
            self._paused_from = None
            if resume_state is not None:
                self._state = resume_state
            self.mark_runtime_fault(exc)
            raise
        self._set_state(RecorderState.PAUSED_CLOSE, "机械臂已停止")

    def close(self, *, save_pending: bool) -> None:
        if self._state is RecorderState.CLOSED:
            return
        if self._state is not RecorderState.PAUSED_CLOSE:
            self.prepare_close()
        can_save = self.can_save_on_close
        if save_pending and not can_save:
            raise RuntimeError("the pending episode is not safe to save")
        self._set_state(RecorderState.CLOSING, "正在关闭")
        failure: BaseException | None = None
        try:
            if self._dataset is not None and self._dataset_dirty:
                if save_pending:
                    self._dataset.save_episode()
                    self._saved_episodes += 1
                else:
                    self._dataset.clear_episode_buffer(delete_images=True)
                self._frame_count = 0
                self._dataset_dirty = False
        except BaseException as exc:
            failure = exc
        if self._dataset is not None:
            try:
                self._dataset.finalize()
            except BaseException as exc:
                failure = failure or exc
        try:
            self.teleop.disconnect()
        except BaseException as exc:
            failure = failure or exc
        try:
            self.robot.disconnect()
        except BaseException as exc:
            failure = failure or exc
        self._recording_started_s = None
        self._set_state(RecorderState.CLOSED, "已关闭" if failure is None else "已关闭，但清理过程中发生错误")
        if failure is not None:
            raise failure

    def cancel_close(self) -> None:
        if self._state is not RecorderState.PAUSED_CLOSE or self._paused_from is None:
            raise RuntimeError("close is not paused")
        resume_state = self._paused_from
        self._paused_from = None
        if resume_state in {RecorderState.HOMING_IDLE, RecorderState.HOMING_RECORDING}:
            self.robot.quiesce_cartesian_velocity_control()
            self._held_home_sample = self._capture_sample()
            self.robot.start_home_async()
            self._home_started_s = self._clock()
            self._home_stable_samples = 0
            self._home_join_timestamp_s = None
        self._set_state(resume_state, "已取消关闭")

    def mark_runtime_fault(self, exc: BaseException) -> None:
        if self._state in {RecorderState.CLOSED, RecorderState.FATAL_ERROR}:
            return
        was_homing = self._state in {RecorderState.HOMING_IDLE, RecorderState.HOMING_RECORDING}
        self._resume_state = self._state
        if was_homing:
            with contextlib.suppress(Exception):
                self.robot.stop_home_motion()
        with contextlib.suppress(Exception):
            self.robot.send_zero_cartesian_velocity()
        self._pending_valid = False
        self._set_state(RecorderState.FAULTED, f"{type(exc).__name__}: {exc}")

    def _tick_home(self) -> None:
        assert self._held_home_sample is not None
        assert self._home_started_s is not None
        if self._clock() - self._home_started_s > float(self.robot.config.home_timeout_s):
            self.robot.stop_home_motion()
            self._pending_valid = False
            self._resume_state = self._state
            self._set_state(RecorderState.FAULTED, "归位超时，请清除 Fault 后重试")
            return

        current = self._capture_sample()
        if current.state_timestamp_s <= self._held_home_sample.state_timestamp_s:
            return

        if self._state is RecorderState.HOMING_RECORDING:
            action = measured_ee_action(
                self._held_home_sample,
                current,
                units=self.options.cartesian_action_units,
                gripper_cmd=self.teleop.gripper_command,
            )
            self._add_frame(self._held_home_sample, action)
        self._held_home_sample = current

        if self.robot.is_home_state(current.state):
            self._home_stable_samples += 1
        else:
            self._home_stable_samples = 0
            self._home_join_timestamp_s = None

        required_stable = int(self.robot.config.home_stable_samples)
        if self._home_join_timestamp_s is None and self._home_stable_samples >= required_stable:
            if self.robot.motion_complete():
                self._home_join_timestamp_s = current.state_timestamp_s
            return

        if self._home_join_timestamp_s is None or current.state_timestamp_s <= self._home_join_timestamp_s:
            return
        if not self.robot.is_home_state(current.state):
            self._pending_valid = False
            self._resume_state = self._state
            self._set_state(RecorderState.FAULTED, "归位结束后关节不在容差内")
            return
        self._finish_home()

    def _finish_home(self) -> None:
        recorded = self._state is RecorderState.HOMING_RECORDING
        if recorded:
            assert self._held_home_sample is not None
            zero_action = dict.fromkeys(self.robot.action_features, 0.0)
            if "gripper_cmd_bin" in zero_action:
                zero_action["gripper_cmd_bin"] = float(self.teleop.gripper_command >= 0.5)
            self._add_frame(self._held_home_sample, zero_action)
            self._save_episode("归位完成，episode 已保存")
        else:
            self._recording_started_s = None
            self._set_state(RecorderState.IDLE, "归位完成")
        self._held_home_sample = None
        self._home_started_s = None
        self._home_join_timestamp_s = None
        self._home_stable_samples = 0

    def _add_frame(self, sample: Any, action: dict[str, Any]) -> None:
        assert self._dataset is not None
        assert self._dataset_spec is not None
        self._dataset_dirty = True
        try:
            frame = self.frame_builder(
                self._dataset.features,
                sample.observation,
                action,
                task=self._dataset_spec.task,
            )
            self._dataset.add_frame(frame)
        except BaseException as exc:
            self._mark_fatal_data_error(exc)
            raise
        self._frame_count += 1

    def _save_episode(self, message: str) -> None:
        self._set_state(RecorderState.SAVING, "正在保存")
        try:
            self._dataset.save_episode()
        except BaseException as exc:
            self._mark_fatal_data_error(exc)
            raise
        self._saved_episodes += 1
        self._frame_count = 0
        self._dataset_dirty = False
        self._pending_valid = True
        self._recording_started_s = None
        self._set_state(RecorderState.IDLE, message)

    def _mark_fatal_data_error(self, exc: BaseException) -> None:
        if self._state in {RecorderState.HOMING_IDLE, RecorderState.HOMING_RECORDING}:
            with contextlib.suppress(Exception):
                self.robot.stop_home_motion()
        with contextlib.suppress(Exception):
            self.robot.send_zero_cartesian_velocity()
        self._pending_valid = False
        self._recording_started_s = None
        self._set_state(RecorderState.FATAL_ERROR, f"数据集写入失败: {type(exc).__name__}: {exc}")

    def _capture_sample(self) -> Any:
        misses = 0
        stale_started_s: float | None = None
        while True:
            try:
                return self.robot.get_observation_sample()
            except StaleFrankaStateError:
                misses += 1
                now_s = self._clock()
                if stale_started_s is None:
                    stale_started_s = now_s
                if misses > self.options.max_consecutive_state_misses:
                    raise
                if now_s - stale_started_s > self.options.max_state_wait_s:
                    raise
                time.sleep(max(0.0, self.options.state_retry_sleep_s))

    def _set_state(self, state: RecorderState, message: str) -> None:
        self._state = state
        self._message = message


def _is_neutral_action(action: dict[str, Any]) -> bool:
    if bool(action.get("reset_requested", False)):
        return False
    return all(
        abs(float(value or 0.0)) <= 1e-9 for key, value in action.items() if key.startswith("delta_ee_pose.")
    )


class RecorderWorker:
    """Own a recorder session on one background thread."""

    def __init__(self, session: FrankaRecorderSession, *, fps: int) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.session = session
        self.fps = int(fps)
        self._commands: queue.Queue[RecorderCommand] = queue.Queue()
        self._events: queue.Queue[RecorderEvent] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._thread_ident: int | None = None
        self._last_snapshot: RecorderSnapshot | None = None

    @property
    def thread_ident(self) -> int | None:
        return self._thread_ident

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("recorder worker has already started")
        self._thread = threading.Thread(target=self._run, name="franka-recorder", daemon=False)
        self._thread.start()

    def submit(self, command: RecorderCommand) -> None:
        self._commands.put(command)

    def get_event_nowait(self) -> RecorderEvent:
        return self._events.get_nowait()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def _run(self) -> None:
        self._thread_ident = threading.get_ident()
        try:
            self.session.connect()
            self._emit()
        except BaseException as exc:
            self._emit(level="error", message=f"连接失败: {type(exc).__name__}: {exc}")
            return

        interval_s = 1.0 / self.fps
        while self.session.state is not RecorderState.CLOSED:
            loop_started_s = time.perf_counter()
            self._drain_commands()
            if self.session.state is RecorderState.CLOSED:
                break
            if self.session.state in {
                RecorderState.IDLE,
                RecorderState.RECORDING,
                RecorderState.HOMING_IDLE,
                RecorderState.HOMING_RECORDING,
            }:
                try:
                    self.session.tick()
                except BaseException as exc:
                    self.session.mark_runtime_fault(exc)
                    self._emit(level="error", message=f"运行错误: {type(exc).__name__}: {exc}")
            self._emit()
            remaining_s = interval_s - (time.perf_counter() - loop_started_s)
            if remaining_s > 0.0:
                time.sleep(remaining_s)
        self._emit()

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                self._execute(command)
            except BaseException as exc:
                self._emit(level="error", message=f"命令失败: {type(exc).__name__}: {exc}")
            else:
                self._emit()

    def _execute(self, command: RecorderCommand) -> None:
        if command.kind is RecorderCommandKind.START:
            if command.spec is None:
                raise ValueError("start command requires a dataset spec")
            self.session.start_recording(command.spec)
        elif command.kind is RecorderCommandKind.END:
            self.session.end_recording()
        elif command.kind is RecorderCommandKind.HOME:
            self.session.home()
        elif command.kind is RecorderCommandKind.CLEAR_FAULT:
            self.session.clear_fault()
        elif command.kind is RecorderCommandKind.PREPARE_CLOSE:
            self.session.prepare_close()
        elif command.kind is RecorderCommandKind.CANCEL_CLOSE:
            self.session.cancel_close()
        elif command.kind is RecorderCommandKind.CLOSE:
            self.session.close(save_pending=command.save_pending)
        else:
            raise ValueError(f"Unsupported recorder command: {command.kind}")

    def _emit(self, *, level: str = "info", message: str = "") -> None:
        snapshot = self.session.snapshot
        if level == "info" and not message and snapshot == self._last_snapshot:
            return
        self._last_snapshot = snapshot
        self._events.put(RecorderEvent(snapshot=snapshot, level=level, message=message or snapshot.message))
