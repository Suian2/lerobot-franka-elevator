# Franka Recorder UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a VITA-style shell-launched Tkinter UI that reuses the existing Franka `run_record.py` hardware stack, records multiple LeRobot episodes, records asynchronous Home motion, and handles Fault recovery safely.

**Architecture:** A shell script manages process lifecycle only. Tkinter callbacks submit local commands to one `RecorderWorker`; that worker exclusively owns `FrankaRobot`, `FrankaSpaceMouseTeleop`, cameras, the server connection, and the dataset. Pure session/action helpers are separated from the thread and view so all control and dataset semantics can be tested without hardware or a display.

**Tech Stack:** Python 3.12, Tkinter, threading/queue, NumPy, LeRobotDataset v3, requests, ZMQ, pytest, Bash/tmux.

---

## Workspace constraint

The target `hardware_test/franka` files already contain user-owned staged and unstaged work. Do not create implementation commits that would accidentally absorb those pre-existing changes. Keep each task's diff narrow, run the stated verification after every task, and leave code uncommitted for the user's existing change set. The two already committed design-only changes are unaffected.

## File map

- Modify `hardware_test/franka/franka_robot.py`: validated server protocol, atomic observation samples, transport quiescence, async Home, joint stop/join, recovery, and home tolerance.
- Modify `hardware_test/franka/franka_spacemouse_teleop.py`: drain/clear input and expose current gripper target.
- Modify `hardware_test/franka/record_lerobot_dataset.py`: pure VITA-compatible measured end-effector transition helpers.
- Create `hardware_test/franka/franka_recording_controller.py`: synchronous recorder session/state machine plus the sole-owner worker thread.
- Create `hardware_test/franka/run_record_ui.py`: UI CLI, object construction through `run_record.py`, and Tkinter view only.
- Create `hardware_test/franka/scripts/start_franka_record_ui.sh`: VITA-style `start-ui`, `stop-ui`, and `status` launcher.
- Create `hardware_test/franka/test_franka_record_ui.py`: focused unit and integration tests for all new behavior.
- Modify `hardware_test/franka/README.md`: operator command, ownership boundary, workflow, and safety notes.

### Task 1: Validate and extend the Franka control protocol

**Files:**
- Modify: `hardware_test/franka/franka_robot.py:354-447`
- Create/Test: `hardware_test/franka/test_franka_record_ui.py`

- [ ] **Step 1: Write failing client protocol tests**

Add response/session fakes and tests covering final paths, payloads, business errors, and nonblocking join:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from hardware_test.franka.franka_robot import FrankaControlClient, FrankaControlError


@dataclass
class FakeResponse:
    payload: Any
    status_error: Exception | None = None

    def raise_for_status(self) -> None:
        if self.status_error is not None:
            raise self.status_error

    def json(self) -> Any:
        return self.payload


class FakeRequestsSession:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("GET", url, kwargs))
        return self.responses.pop(0)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("POST", url, kwargs))
        return self.responses.pop(0)


def make_http_client(session: FakeRequestsSession) -> FrankaControlClient:
    client = FrankaControlClient(
        base_url="http://test-controller:29000/ctl",
        control_host="test-controller",
        velocity_transport="http",
        zmq_url=None,
        timeout_s=2.0,
        command_duration_ms=300,
    )
    client._session = session
    return client


def test_control_client_async_home_recover_stop_and_join_routes() -> None:
    session = FakeRequestsSession([FakeResponse({"is_ok": 1}) for _ in range(4)])
    client = make_http_client(session)

    client.joint_position_control([0.0] * 7, mode="absolute", is_async=True)
    client.recover()
    client.stop_joint_position_control()
    assert client.join_motion(timeout_s=0.0) is True

    assert session.calls[0][1].endswith("/ctl/joint_position_control")
    assert session.calls[0][2]["json"]["is_async"] == 1
    assert session.calls[1][1].endswith("/ctl/recover")
    assert session.calls[2][1].endswith("/ctl/stop_joint_position_control")
    assert session.calls[3][1].endswith("/ctl/join")
    assert session.calls[3][2]["json"] == {"timeout": 0.0}


def test_control_client_rejects_http_200_business_error() -> None:
    client = make_http_client(
        FakeRequestsSession([FakeResponse({"is_ok": 0, "error_type": "RuntimeError", "error": "fault"})])
    )

    with pytest.raises(FrankaControlError, match="RuntimeError: fault"):
        client.recover()


def test_control_client_join_false_means_motion_is_still_running() -> None:
    client = make_http_client(FakeRequestsSession([FakeResponse({"is_ok": 0})]))

    assert client.join_motion(timeout_s=0.0) is False
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run pytest hardware_test/franka/test_franka_record_ui.py -q
```

Expected: collection or assertion failures because `FrankaControlError`, `recover`, `stop_joint_position_control`, `join_motion`, and `is_async` are absent.

- [ ] **Step 3: Implement the minimal validated client API**

Add these interfaces to `franka_robot.py` and route `_get`/`_post` through validation:

```python
class FrankaControlError(RuntimeError):
    pass


def _validated_reply(payload: Any, *, allow_incomplete: bool = False) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise FrankaControlError(f"Franka server returned {type(payload).__name__}, expected JSON object")
    if bool(payload.get("is_ok", 0)):
        return payload
    if allow_incomplete and "error" not in payload and "error_type" not in payload:
        return payload
    error_type = str(payload.get("error_type", "FrankaControlError"))
    error = str(payload.get("error", "server returned is_ok=0"))
    raise FrankaControlError(f"{error_type}: {error}")
```

Use exact public signatures:

```python
def recover(self) -> dict[str, Any]:
    return self._get("recover")

def velocity_loop_status(self) -> dict[str, Any]:
    return self._get("velocity_ws_status")

def stop_cartesian_velocity_control_direct(self) -> dict[str, Any]:
    return self._get("stop_cartesian_velocity_control")

def stop_joint_position_control(self) -> dict[str, Any]:
    return self._get("stop_joint_position_control")

def join_motion(self, timeout_s: float = 0.0) -> bool:
    reply = self._post("join", {"timeout": float(timeout_s)}, allow_incomplete=True)
    return bool(reply.get("is_ok", 0))
```

Extend `joint_position_control(self, joints: list[float], *, mode: str = "absolute", is_async: bool = False, timeout: float | None = None)` and serialize `int(bool(is_async))`. Add `allow_incomplete` to `_post`; `_get` and normal `_post` must call `_validated_reply` after `raise_for_status()`.

- [ ] **Step 4: Run focused and existing adapter tests**

Run:

```bash
uv run pytest hardware_test/franka/test_franka_record_ui.py hardware_test/franka/test_franka_adapters.py -q
```

Expected: all client tests pass and existing adapter tests remain green.

### Task 2: Add atomic observation and safe motion-mode operations

**Files:**
- Modify: `hardware_test/franka/franka_robot.py:29-351`
- Test: `hardware_test/franka/test_franka_record_ui.py`

- [ ] **Step 1: Write failing atomic-sample and motion tests**

Create a fake client with an `ee` matrix, join sequence, and call log. Assert:

```python
def test_robot_observation_sample_uses_one_state_snapshot() -> None:
    client = FakeRobotClient()
    robot = connected_robot(client)

    sample = robot.get_observation_sample()

    assert sample.observation["joint_1.pos"] == 0.1
    assert sample.state["ee"][0][3] == 0.4
    assert sample.state_timestamp_s > 0.0
    assert client.get_curr_calls == 1


def test_robot_async_home_quiesces_velocity_before_joint_motion() -> None:
    client = FakeRobotClient(join_results=[False, True])
    robot = connected_robot(client, velocity_transport="http")

    robot.quiesce_cartesian_velocity_control()
    robot.start_home_async()

    assert client.calls[:3] == ["stop_cartesian_direct", "join", "join"]
    assert client.joint_calls[-1]["is_async"] is True


def test_robot_home_tolerance_uses_all_seven_joints() -> None:
    robot = connected_robot(FakeRobotClient())
    target = list(robot.config.home_joints or ())

    assert robot.is_home_state({"joint": target}) is True
    target[-1] += robot.config.home_joint_tolerance_rad * 2.0
    assert robot.is_home_state({"joint": target}) is False
```

- [ ] **Step 2: Verify the tests fail**

Run `uv run pytest hardware_test/franka/test_franka_record_ui.py -q`.

Expected: failures for missing `FrankaObservationSample`, `get_observation_sample`, quiescence, async Home, and tolerance fields.

- [ ] **Step 3: Implement atomic sampling and motion operations**

Add configuration fields and validate them as positive:

```python
home_joint_tolerance_rad: float = 0.02
home_stable_samples: int = 5
control_transition_timeout_s: float = 2.0
control_status_poll_s: float = 0.02
```

Add the immutable sample:

```python
@dataclass(frozen=True)
class FrankaObservationSample:
    observation: RobotObservation
    state: dict[str, Any]
    state_timestamp_s: float
```

Factor current `get_observation()` so it returns `self.get_observation_sample().observation`. `get_observation_sample()` must obtain one cache snapshot, build joint/gripper fields from that same state, then append camera images without rereading robot state.

Add public methods with these responsibilities:

```python
def send_zero_cartesian_velocity(self) -> None:
    self._send_zero_cartesian_velocity()

def start_home_async(self) -> None:
    if self.config.home_joints is None:
        raise RuntimeError("Franka home_joints is not configured")
    self._client.joint_position_control(
        list(self.config.home_joints), mode="absolute", is_async=True, timeout=self.config.timeout_s
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
    return bool(np.max(np.abs(joints - target)) <= self.config.home_joint_tolerance_rad)
```

Implement HTTP quiescence as direct stop plus `_wait_for_motion_completion()`. Implement ZMQ quiescence by reading `/velocity_ws_status`, synchronously sending the latest zero with `send_latest_once()`, waiting for a newer `latest_seq` to be dispatched with `motion_active == False`, then direct stop plus join polling. Use the configured transition deadline and poll interval.

- [ ] **Step 4: Verify robot and regression tests**

Run:

```bash
uv run pytest hardware_test/franka/test_franka_record_ui.py hardware_test/franka/test_franka_adapters.py -q
```

Expected: PASS.

### Task 3: Make SpaceMouse input safely clearable

**Files:**
- Modify: `hardware_test/franka/franka_spacemouse_teleop.py:65-157,182-286`
- Test: `hardware_test/franka/test_franka_record_ui.py`

- [ ] **Step 1: Write failing input-reset tests**

```python
def test_spacemouse_clear_input_drains_motion_without_changing_gripper_target() -> None:
    reader = ClearableReader()
    teleop = FrankaSpaceMouseTeleop(FrankaSpaceMouseTeleopConfig(), reader=reader)
    teleop.connect()
    teleop.get_action()

    teleop.clear_input()

    action = teleop.get_action()
    assert reader.clear_calls == 1
    assert all(action[key] == 0.0 for key in DELTA_EE_KEYS)
    assert action["reset_requested"] is False
    assert action["gripper_cmd_bin"] == teleop.gripper_command
```

Add a lower-level test proving `SpacenavReader.clear()` calls the library event removal operation and zeros `_raw_motion`, `_action`, and `_buttons`.

- [ ] **Step 2: Run and verify RED**

Run `uv run pytest hardware_test/franka/test_franka_record_ui.py -q`.

Expected: missing `clear_input`, `gripper_command`, and reader clear methods.

- [ ] **Step 3: Implement clearing**

Add:

```python
@property
def gripper_command(self) -> float:
    return float(self._gripper_cmd)

def clear_input(self) -> None:
    clear = getattr(self._reader, "clear", None)
    if callable(clear):
        clear()
    self._last_buttons = []
```

Add `_SpnavLibrary.remove_events()` around `spnav_remove_events(SPNAV_EVENT_ANY)`. `SpacenavReader.clear()` calls it and resets all cached motion/buttons plus `_last_motion_time`. Preserve `_gripper_cmd`.

- [ ] **Step 4: Run SpaceMouse and axis regressions**

Run:

```bash
uv run pytest hardware_test/franka/test_franka_record_ui.py hardware_test/franka/test_franka_adapters.py -q -k 'spacemouse or vita or clear_input'
```

Expected: PASS, including both positive and negative robot-Z mapping cases.

### Task 4: Add forward-aligned measured Home actions

**Files:**
- Modify: `hardware_test/franka/record_lerobot_dataset.py`
- Test: `hardware_test/franka/test_franka_record_ui.py`

- [ ] **Step 1: Write failing pure-action tests**

Use two `FrankaObservationSample` values with base-frame transforms and assert delta/velocity units and angle wrapping:

```python
def test_measured_home_action_is_forward_delta_from_held_sample() -> None:
    previous = sample_at(x=0.40, yaw=3.13, timestamp=10.0)
    current = sample_at(x=0.43, yaw=-3.13, timestamp=10.1)

    action = measured_ee_action(previous, current, units="delta", gripper_cmd=1.0)

    assert action["delta_ee_pose.x"] == pytest.approx(0.03)
    assert action["delta_ee_pose.rz"] == pytest.approx(0.023185307, abs=1e-6)
    assert action["gripper_cmd_bin"] == 1.0


def test_measured_home_velocity_uses_robot_state_timestamps() -> None:
    previous = sample_at(x=0.40, yaw=0.0, timestamp=2.0)
    current = sample_at(x=0.41, yaw=0.0, timestamp=2.2)

    action = measured_ee_action(previous, current, units="velocity", gripper_cmd=0.0)

    assert action["delta_ee_pose.x"] == pytest.approx(0.05)


def test_measured_home_action_rejects_duplicate_or_invalid_state() -> None:
    with pytest.raises(ValueError, match="newer state timestamp"):
        measured_ee_action(sample_at(timestamp=1.0), sample_at(timestamp=1.0), units="delta", gripper_cmd=1.0)
```

- [ ] **Step 2: Verify RED**

Run `uv run pytest hardware_test/franka/test_franka_record_ui.py -q -k measured_home`.

Expected: missing helper failures.

- [ ] **Step 3: Implement the pure helpers**

Add `matrix4_to_xyz_rpy()`, `_wrap_angle_delta()`, and:

```python
def measured_ee_action(
    previous: FrankaObservationSample,
    current: FrankaObservationSample,
    *,
    units: str,
    gripper_cmd: float,
) -> RobotAction:
    dt_s = current.state_timestamp_s - previous.state_timestamp_s
    if dt_s <= 0.0:
        raise ValueError("measured action requires a newer state timestamp")
    previous_pose = end_effector_pose(previous.state)
    current_pose = end_effector_pose(current.state)
    delta = current_pose - previous_pose
    delta[3:] = np.asarray([_wrap_angle_delta(value) for value in delta[3:]])
    if units == "velocity":
        delta /= dt_s
    elif units != "delta":
        raise ValueError(f"Unsupported measured action units: {units}")
    action = {key: float(value) for key, value in zip(DELTA_EE_KEYS, delta, strict=True)}
    action["gripper_cmd_bin"] = float(1.0 if gripper_cmd >= 0.5 else 0.0)
    return action
```

Reject non-finite and non-4x4 `ee` matrices. Do not clip measured values.

- [ ] **Step 4: Verify pure helpers and frame packing**

Run:

```bash
uv run pytest hardware_test/franka/test_franka_record_ui.py hardware_test/franka/test_franka_adapters.py -q -k 'measured_home or lerobot_recording_helpers'
```

Expected: PASS.

### Task 5: Build the synchronous recording session

**Files:**
- Create: `hardware_test/franka/franka_recording_controller.py`
- Test: `hardware_test/franka/test_franka_record_ui.py`

- [ ] **Step 1: Write failing Start/End/multi-episode tests**

Define fake robot, teleop, and dataset objects. Cover connection ownership, one-time dataset creation, normal frame flow, zero-frame discard, duration, and multiple saves:

```python
def test_session_connects_once_and_saves_multiple_episodes() -> None:
    rig = SessionRig()
    session = rig.make_session(duration_s=0.0, num_episodes=0)
    session.connect()

    session.start_recording(DatasetSpec("local/test", "/tmp/test", "pick"))
    session.tick()
    session.end_recording()
    session.start_recording(DatasetSpec("local/test", "/tmp/test", "pick"))
    session.tick()
    session.end_recording()

    assert rig.robot.connect_calls == 1
    assert rig.teleop.connect_calls == 1
    assert rig.dataset_factory_calls == 1
    assert rig.dataset.save_calls == 2
    assert session.snapshot.saved_episodes == 2


def test_session_intercepts_spacemouse_reset_instead_of_sending_blocking_reset() -> None:
    rig = SessionRig(actions=[zero_action(reset_requested=True)])
    session = rig.make_session()
    session.connect()

    session.tick()

    assert rig.robot.sent_actions == []
    assert session.state is RecorderState.HOMING_IDLE
```

- [ ] **Step 2: Verify RED**

Run `uv run pytest hardware_test/franka/test_franka_record_ui.py -q -k session`.

Expected: module/class import failures.

- [ ] **Step 3: Implement focused session types and normal tick flow**

Create exact public types:

```python
class RecorderState(str, Enum):
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
    duration_s: float
    num_episodes: int
    cartesian_action_units: str
    max_consecutive_state_misses: int
    max_state_wait_s: float
    state_retry_sleep_s: float


@dataclass(frozen=True)
class RecorderSnapshot:
    state: RecorderState
    message: str
    frame_count: int
    saved_episodes: int
    dataset_locked: bool
    pending_valid: bool
    wall_elapsed_s: float
```

`FrankaRecorderSession.connect()` connects robot then teleop and enters Idle. `start_recording()` creates the dataset only once and locks `DatasetSpec`; later starts must match it. `tick()` in Idle/Recording reads one atomic sample and one action, intercepts `reset_requested`, sends ordinary actions, and adds frames only in Recording. `end_recording()` sends zero, saves non-empty buffers, clears zero-frame buffers, and returns Idle. Duration uses `frame_count / fps`.

- [ ] **Step 4: Run core session tests**

Run `uv run pytest hardware_test/franka/test_franka_record_ui.py -q -k session`.

Expected: core Start/End tests PASS.

### Task 6: Complete Home, recovery, and close semantics

**Files:**
- Modify: `hardware_test/franka/franka_recording_controller.py`
- Test: `hardware_test/franka/test_franka_record_ui.py`

- [ ] **Step 1: Write failing Home trajectory tests**

Use ordered samples with distinct state timestamps and a robot `motion_complete()` sequence. Prove the controller holds the earlier observation, records every transition during Home, waits for five stable samples plus join, captures a fresh post-join sample, appends a final zero action, and auto-saves:

```python
def test_recorded_home_keeps_motion_frames_and_auto_saves_after_fresh_final_sample() -> None:
    rig = HomingRig()
    session = rig.recording_session()

    session.home()
    for _ in range(rig.required_ticks):
        session.tick()

    assert rig.robot.start_home_calls == 1
    assert len(rig.dataset.frames) >= 6
    assert any(frame["action"][0] != 0.0 for frame in rig.dataset.frames)
    assert rig.dataset.frames[-1]["action"].tolist()[:6] == [0.0] * 6
    assert rig.dataset.save_calls == 1
    assert session.state is RecorderState.IDLE
```

Also add tests for duration not truncating Home, duplicate state timestamps not advancing stability, and Home timeout stopping joint motion without saving.

- [ ] **Step 2: Write failing recovery and close tests**

Cover successful recovery of ordinary recording, recorded-Home stop/recover/rebaseline/restart, failed recovery discard, neutral input gating, and pause-before-close:

```python
def test_failed_recovery_discards_pending_episode() -> None:
    rig = SessionRig(recover_error=RuntimeError("recover failed"))
    session = rig.recording_session_with_one_frame()

    session.clear_fault()

    assert rig.dataset.clear_calls == 1
    assert session.state is RecorderState.FATAL_ERROR
    assert session.snapshot.pending_valid is False


def test_prepare_close_stops_motion_before_exposing_close_choices() -> None:
    rig = HomingRig()
    session = rig.recording_session()
    session.home()

    session.prepare_close()

    assert rig.robot.stop_home_calls == 1
    assert rig.robot.quiesce_calls >= 2
    assert session.state is RecorderState.PAUSED_CLOSE
    assert session.can_save_on_close is False
```

- [ ] **Step 3: Implement Home as a forward-aligned held-sample state**

Add private fields `_held_home_sample`, `_home_started_s`, `_stable_home_samples`, `_join_seen`, `_join_sample_timestamp`, `_resume_state`, `_input_armed`, and `_pending_valid`.

`home()` must clear input, disarm teleop, quiesce Cartesian motion, capture a fresh baseline, start async Home, and choose `HOMING_IDLE` or `HOMING_RECORDING`.

`tick()` in a Homing state must:

1. reject duplicate timestamps;
2. add the held observation with `measured_ee_action(held, current, units=self.options.cartesian_action_units, gripper_cmd=self.teleop.gripper_command)` when recording;
3. replace held with current;
4. update stability only for distinct samples;
5. poll `motion_complete()` after five stable samples while continuing capture;
6. after join, require one newer sample, record the last transition and final zero action, then auto-save;
7. on deadline, stop Home and enter Faulted without saving.

- [ ] **Step 4: Implement recovery and close decisions**

`clear_fault()` clears/disarms SpaceMouse, pauses frame writes, stops Home when applicable, calls `robot.recover()`, requiesces control, and either resumes Recording or captures a new baseline and restarts Home. Any recovery failure calls `dataset.clear_episode_buffer(delete_images=True)` and enters Fatal Error.

`prepare_close()` stops all motion before entering Paused Close. `cancel_close()` rearms only after a neutral action and restarts Home from a new baseline when needed. `close(save_pending: bool)` saves only a normal valid recording, otherwise clears pending frames, finalizes the dataset, disconnects teleop then robot, and enters Closed.

- [ ] **Step 5: Verify full state-machine behavior**

Run:

```bash
uv run pytest hardware_test/franka/test_franka_record_ui.py -q -k 'home or recovery or close or session'
```

Expected: PASS.

### Task 7: Add the sole-owner worker and Tkinter UI

**Files:**
- Modify: `hardware_test/franka/franka_recording_controller.py`
- Create: `hardware_test/franka/run_record_ui.py`
- Test: `hardware_test/franka/test_franka_record_ui.py`

- [ ] **Step 1: Write failing worker/parser/view-policy tests**

Test worker-thread ownership by recording thread IDs in all fake hardware calls. Test UI defaults and pure button policy without creating a Tk root:

```python
def test_worker_is_only_thread_that_touches_hardware() -> None:
    rig = SessionRig()
    worker = RecorderWorker(rig.make_session())
    worker.start()
    worker.submit(RecorderCommand.connect())
    worker.submit(RecorderCommand.start(DatasetSpec("local/test", "/tmp/test", "pick")))
    worker.submit(RecorderCommand.stop())
    worker.submit(RecorderCommand.close(save_pending=False))
    worker.join(timeout=2.0)

    assert len(rig.hardware_thread_ids) == 1


def test_ui_parser_uses_manual_recording_defaults_and_reuses_hardware_flags() -> None:
    args = build_ui_arg_parser().parse_args(
        ["--camera-backend", "none", "--control-host", "test-controller"]
    )

    assert args.duration_s == 0.0
    assert args.num_episodes == 0
    assert args.control_host == "test-controller"


def test_button_policy_for_recording_and_homing() -> None:
    assert button_policy(RecorderState.RECORDING) == ButtonPolicy(False, True, True, True)
    assert button_policy(RecorderState.HOMING_RECORDING) == ButtonPolicy(False, False, False, True)
```

- [ ] **Step 2: Verify RED**

Run `uv run pytest hardware_test/franka/test_franka_record_ui.py -q -k 'worker or ui_parser or button_policy'`.

Expected: missing worker, UI module, and policy types.

- [ ] **Step 3: Implement `RecorderWorker`**

Use one non-daemon thread, a command queue, and a snapshot/event queue. The thread connects, drains commands, calls `session.tick()` at `1 / fps`, and converts uncaught exceptions into a Fatal Error snapshot after best-effort motion stop. Public methods are `start()`, `submit(command)`, `get_event_nowait()`, `join(timeout)`, and `is_alive()`.

Commands are immutable and include Connect, Start, End, Home, Clear Fault, Prepare Close, Save Close, Discard Close, and Cancel Close. No command handler runs on the Tk thread.

- [ ] **Step 4: Implement the UI CLI and pure view policy**

`build_ui_arg_parser()` starts from `run_record.build_arg_parser()`, overrides `duration_s=0.0` and `num_episodes=0`, and rejects `action_mode != "delta_ee_pose"` or Hub upload flags before hardware construction. `--dry-run-config --camera-backend none` prints configuration and returns without importing a display.

Add exact pure policy types:

```python
@dataclass(frozen=True)
class ButtonPolicy:
    start: bool
    end: bool
    home: bool
    clear_fault: bool
```

The Tk view contains editable repo/root/task entries, Chinese primary buttons, state/episode/frame/elapsed labels, and a timestamped scrolling log. It polls worker events with `root.after(50, self._poll_worker)`. Dataset entries disable only after `dataset_locked=True`. `WM_DELETE_WINDOW`, SIGINT, and SIGTERM all request Prepare Close and wait for the worker acknowledgement before showing the save/discard/cancel dialog.

- [ ] **Step 5: Verify headless UI behavior and dry run**

Run:

```bash
uv run pytest hardware_test/franka/test_franka_record_ui.py -q -k 'worker or ui_parser or button_policy'
uv run python hardware_test/franka/run_record_ui.py --dry-run-config --camera-backend none
```

Expected: tests PASS; dry run exits 0 and states that no hardware or Tk window was opened.

### Task 8: Add the VITA-style shell launcher and operator documentation

**Files:**
- Create: `hardware_test/franka/scripts/start_franka_record_ui.sh`
- Modify: `hardware_test/franka/README.md`
- Test: `hardware_test/franka/test_franka_record_ui.py`

- [ ] **Step 1: Write failing launcher contract test**

```python
def test_launcher_has_vita_style_lifecycle_commands() -> None:
    script = Path("hardware_test/franka/scripts/start_franka_record_ui.sh")
    result = subprocess.run(["bash", str(script), "help"], capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert "start-ui" in result.stdout
    assert "stop-ui" in result.stdout
    assert "status" in result.stdout
```

- [ ] **Step 2: Verify RED**

Run `uv run pytest hardware_test/franka/test_franka_record_ui.py -q -k launcher`.

Expected: script not found.

- [ ] **Step 3: Implement the launcher**

Use `set -euo pipefail`, resolve `REPO_ROOT`, default `LEROBOT_PYTHON` to `/home/yanrihong/miniconda3/envs/lerobot/bin/python`, default the tmux session to `lerobot_franka_record_ui`, and construct the tmux command with `printf %q` for every forwarded argument. `start-ui` rejects an existing session. `stop-ui` sends Ctrl-C and waits for graceful exit; it reports a still-running confirmation dialog rather than force-killing the robot process. `status` prints tmux session state. `help` prints usage and exits zero.

Make the script executable and verify syntax:

```bash
chmod +x hardware_test/franka/scripts/start_franka_record_ui.sh
bash -n hardware_test/franka/scripts/start_franka_record_ui.sh
```

- [ ] **Step 4: Document the exact operator flow**

Add a README section with the ownership boundary and command:

```bash
./hardware_test/franka/scripts/start_franka_record_ui.sh start-ui \
  --repo-id local/franka_l515_smoke \
  --root outputs/hardware_test/franka_l515_smoke \
  --task "Franka SpaceMouse teleoperation" \
  --duration-s 30 \
  --num-episodes 1 \
  --camera-backend realsense \
  --camera-serial-or-name "Intel RealSense L515" \
  --streaming-encoding \
  --encoder-threads 2
```

State explicitly that the shell launches only, Tk submits local commands only, and RecorderWorker owns the server connection. Describe Start, End, recorded Home auto-save, Clear Fault continuation/discard, and existing-root rejection.

- [ ] **Step 5: Verify launcher and documentation tests**

Run:

```bash
bash -n hardware_test/franka/scripts/start_franka_record_ui.sh
uv run pytest hardware_test/franka/test_franka_record_ui.py -q -k launcher
```

Expected: PASS.

### Task 9: Full verification and hardware handoff

**Files:**
- Verify all changed files; no new implementation file is added in this task.

- [ ] **Step 1: Run focused tests**

```bash
uv run pytest hardware_test/franka/test_franka_record_ui.py hardware_test/franka/test_franka_adapters.py -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run static checks on changed Python and shell files**

```bash
uv run ruff check \
  hardware_test/franka/franka_robot.py \
  hardware_test/franka/franka_spacemouse_teleop.py \
  hardware_test/franka/record_lerobot_dataset.py \
  hardware_test/franka/franka_recording_controller.py \
  hardware_test/franka/run_record_ui.py \
  hardware_test/franka/test_franka_record_ui.py
uv run python -m py_compile \
  hardware_test/franka/franka_robot.py \
  hardware_test/franka/franka_spacemouse_teleop.py \
  hardware_test/franka/record_lerobot_dataset.py \
  hardware_test/franka/franka_recording_controller.py \
  hardware_test/franka/run_record_ui.py
bash -n hardware_test/franka/scripts/start_franka_record_ui.sh
git diff --check
```

Expected: zero errors. If Ruff reports the repository's pre-existing C420 findings in the hardware test files, rerun with the existing local exception only after confirming no new finding was introduced.

- [ ] **Step 3: Run no-hardware CLI checks**

```bash
uv run python hardware_test/franka/run_record_ui.py --dry-run-config --camera-backend none
./hardware_test/franka/scripts/start_franka_record_ui.sh help
./hardware_test/franka/scripts/start_franka_record_ui.sh status
```

Expected: dry run and help exit 0; status accurately reports whether the tmux UI session exists.

- [ ] **Step 4: Inspect the final diff for ownership and safety invariants**

Confirm with `rg` that `run_record_ui.py` contains no `requests`, HTTP URL, or ZMQ socket construction, and that every Tk callback calls `worker.submit(command)` rather than robot, teleop, camera, or dataset methods. Confirm existing `run_record.py` terminal behavior and its blocking reset path remain unchanged.

- [ ] **Step 5: Provide the manual hardware checklist**

Do not claim physical verification without the operator. Hand off these steps:

1. Start the UI with the documented shell command.
2. Confirm idle SpaceMouse motion in both positive and negative Z.
3. Record and End one episode; verify episode/frame counts.
4. Record another episode, press Home, and verify frames continue increasing until automatic save.
5. Inspect the dataset's second episode for nonzero measured Home actions and a final home frame.
6. Trigger or simulate a recoverable Fault; verify successful recovery resumes the same episode.
7. Verify failed recovery discards the pending episode.
