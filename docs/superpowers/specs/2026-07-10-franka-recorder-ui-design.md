# Franka LeRobot Recorder UI Design

Status: approved in conversation on 2026-07-10

## Problem

The current `hardware_test/franka/run_record.py` command records fixed-duration episodes in a blocking terminal loop. It cannot be operated safely as an interactive recording tool, and its blocking home operation stops the capture loop for the entire return-to-home motion. Wrapping that command in a GUI process launcher would therefore fail the requirement to include the home trajectory in the recorded episode.

The requested UI must keep SpaceMouse teleoperation available, let an operator start and end individual LeRobot episodes, return the robot to its configured home pose, and clear Franka faults using the same control semantics as the working VITA UI.

## Goals

- Provide Start Recording, End Recording, Home, and Clear Fault controls.
- Show connection, dataset, episode, frame, operation, and error status.
- Allow `repo_id`, `root`, and `task` to be edited until the first episode starts.
- Store every Start-to-End interval as a separate episode in one dataset for the lifetime of the UI process.
- Keep SpaceMouse control active while connected and idle, not only while recording.
- When Home is pressed during recording, capture the complete home motion and automatically save the episode after the robot reaches home.
- When Clear Fault succeeds during recording, resume the same pending episode. If recovery fails, discard that pending episode.
- Add no third-party UI dependency.
- Keep all robot commands on one worker thread so UI events cannot issue concurrent control requests.

## Non-goals

- Appending to a dataset root created by an earlier process.
- Editing camera, transport, or SpaceMouse tuning options after the UI has started.
- Replacing the existing terminal recorder.
- Adding dataset browsing, episode deletion, annotation, playback, or Hub management controls.
- Supporting joint-action recording in this first UI version.
- Uploading to Hugging Face Hub from the UI process.
- Reimplementing the VITA PySide6 application or importing VITA at runtime.

## Chosen approach

Use a standard-library Tkinter view backed by a single-owner recording controller running on a background thread, launched through a VITA-style shell entry point.

This is preferred over launching `run_record.py` as a subprocess because subprocess wrapping cannot capture frames while the current blocking `_send_home()` call is running. It is preferred over reusing the VITA PySide6 UI because the active LeRobot environment has Tkinter but not PySide6, and cross-repository runtime coupling would make the recorder harder to deploy and test.

The implementation will reuse the existing `run_record.py` argument/configuration builders, `FrankaRobot`, `FrankaSpaceMouseTeleop`, and LeRobot dataset helpers rather than create a second hardware stack.

The ownership boundary matches VITA: the shell script activates the environment and launches the application; the Python recording backend owns the arm, camera, SpaceMouse, and dataset connections. Tk callbacks never connect to or issue requests against the Franka server.

## User interface

The selected layout is recording-first:

1. A header displays the connection indicator and current controller state.
2. Editable `repo_id`, `root`, and `task` fields appear above the controls.
3. Start Recording and End Recording are the large primary controls.
4. Home and Clear Fault are secondary controls.
5. Episode number, current frame count, recorded elapsed time (`frames / fps`), wall elapsed time, and the configured duration limit are visible during capture.
6. A timestamped scrolling log reports connection, recording, home, recovery, save, discard, and error events.

The UI text is Chinese, while log messages include enough English identifiers to match exceptions and control endpoints.

The three dataset fields are locked only after dataset creation succeeds. If the selected root already exists or creation otherwise fails, they remain editable.

### Button availability

| State | Start | End | Home | Clear Fault |
| --- | --- | --- | --- | --- |
| Connecting | disabled | disabled | disabled | disabled |
| Preparing dataset | disabled | disabled | disabled | disabled |
| Idle | enabled | disabled | enabled | enabled |
| Idle after episode limit | disabled | disabled | enabled | enabled |
| Recording | disabled | enabled | enabled | enabled |
| Homing without recording | disabled | disabled | disabled | enabled |
| Homing while recording | disabled | disabled | disabled | enabled |
| Recovering or saving | disabled | disabled | disabled | disabled |
| Recoverable fault | disabled | disabled | disabled | enabled |
| Fatal error | disabled | disabled | disabled | disabled |

While recorded homing is active, the status explicitly says that the episode will be saved automatically after home is reached.

## Command-line behavior

The entry point is `hardware_test/franka/run_record_ui.py`. It accepts the local dataset, hardware, camera, transport, and encoding arguments supported by `run_record.py` so an existing recording command can be changed to the UI entry point without rewriting its hardware configuration.

The normal operator entry point is:

```bash
./hardware_test/franka/scripts/start_franka_record_ui.sh start-ui [recorder arguments]
```

The launcher also provides `stop-ui` and `status`. It activates the configured LeRobot environment, establishes repository paths, preserves desktop variables such as `DISPLAY`, and forwards recorder arguments unchanged. It may perform read-only availability checks, but it does not SSH into the control host, start FCI, or own a persistent robot connection. The long-lived Python `RecorderWorker` is the only component that calls `robot.connect()` and `teleop.connect()`.

UI-specific defaults are:

- `duration_s=0`: no automatic duration limit; End Recording or recorded Home ends the episode.
- `duration_s>0`: automatically save when the recorded-frame limit is reached; End Recording can still save earlier. Once recorded Home starts, Home completion overrides the duration limit so the trajectory is never truncated.
- `num_episodes=0`: no session episode limit.
- `num_episodes>0`: disable Start after that many episodes have been saved.

The UI supports only `action_mode=delta_ee_pose`, which is the action schema emitted by `FrankaSpaceMouseTeleop`. Supplying `--action-mode joint` is rejected before hardware connection instead of failing later with mismatched action keys. Joint-action recording is outside this UI's first scope.

`--dry-run-config` continues to print configuration and exits without opening hardware or a window. Hub upload flags are intentionally not part of the UI: the operator can validate the local dataset and upload it with existing LeRobot tooling after the recorder closes.

## Architecture

### Tkinter view

`run_record_ui.py` owns Tk widgets and the Tk event loop. Button callbacks enqueue typed commands and return immediately. A short `after()` callback drains controller events and updates labels, logs, and button states. The Tk thread never calls robot, camera, SpaceMouse, or dataset methods.

### Shell launcher

`hardware_test/franka/scripts/start_franka_record_ui.sh` follows the VITA `start-ui` pattern. It manages one named UI process/session and supports start, graceful stop, and status inspection. Starting a second instance is rejected. `stop-ui` requests normal Python shutdown first so zero motion, pending-episode handling, dataset finalization, and device disconnect are not bypassed.

The shell script is orchestration only. It cannot replace the long-lived Python backend with one `run_record.py` process per episode: that would reconnect hardware repeatedly, conflict with the existing-root dataset policy, and retain the blocking Home behavior that loses trajectory frames.

### Recording controller

A focused controller module owns:

- the robot, teleoperator, and dataset objects;
- one non-daemon worker thread;
- a UI-to-worker command queue;
- a worker-to-UI event queue;
- the recording state machine;
- episode frame count and timing;
- home completion tracking;
- pending-episode save/discard behavior.

The controller accepts injected robot, teleoperator, dataset factory, clock, and sleeper collaborators so state transitions can be tested without a display or hardware.

It reuses `run_record.py` configuration builders and the existing adapters. UI code does not import `requests` or create ZMQ sockets; all control-host communication remains behind `FrankaRobot` and `FrankaControlClient` on the worker thread.

### Franka adapter extensions

`FrankaControlClient` gains:

- an `is_async` option for joint position commands, mapped to the server's existing `is_async` payload;
- a `recover()` request using the relative `recover` endpoint, producing the final URL `/ctl/recover`;
- direct Cartesian-stop, velocity-loop status, `stop_joint_position_control()`, and `join()` requests using the server's existing endpoints.

Every UI safety-critical response is validated at two levels: a successful HTTP status and a JSON object with `is_ok == 1`. An HTTP 200 response containing `is_ok: 0` raises a control error with the server's `error_type` and `error` fields. The `join` endpoint is the one intentional exception: `join(timeout=0)` returns `False` to mean "motion is still running" and is never interpreted as completion; only `True` permits a control-mode transition or completed Home.

`FrankaRobot` gains small public operations for:

- sending a safe zero Cartesian velocity;
- quiescing the active Cartesian velocity transport and waiting for its stop motion to complete before switching control modes;
- starting home asynchronously;
- stopping asynchronous joint-position motion and waiting for the stop motion to complete;
- clearing a fault;
- returning an atomic observation sample containing the camera observation, the exact raw Franka state snapshot used for joint fields, and that snapshot's monotonic timestamp;
- testing whether the seven joints are within the configured home tolerance.

`FrankaSpaceMouseTeleop` gains a public input-reset operation that drains queued events and clears cached motion and button state after Home, Clear Fault, or close preparation. The controller then requires one neutral SpaceMouse sample before nonzero commands are accepted again. This is the LeRobot equivalent of VITA clearing its shared increments after recovery and prevents a held or stale event from being replayed.

Home completion uses the existing `home_timeout_s` value (20 seconds by default), a maximum absolute joint error of 0.02 radians, and five consecutive in-tolerance samples. Control-mode stop operations use a 2.0-second transition timeout and 20-millisecond status or nonblocking-join polling. These values are explicit robot configuration fields so hardware tuning does not require controller changes.

For ZMQ velocity transport, quiescing first reads the shared server loop's `latest_seq`, `dispatched_seq`, and `motion_active` from the relative `velocity_ws_status` endpoint (final URL `/ctl/velocity_ws_status`). It synchronously sends the latest-only zero command, then requires a newer loop sequence to become dispatched with `motion_active=false`. `/ctl/velocity_zmq_status` reports receiver health only and is not used as motion acknowledgement. After loop acknowledgement, the client issues the direct Cartesian-stop endpoint and polls `join(timeout=0)` until true or the 2.0-second transition timeout expires. HTTP transport uses the direct zero/stop request followed by the same join polling. Failure to confirm quiescence prevents Home from starting; an already-idle loop is not accepted as acknowledgement of a zero that the server never received.

The existing blocking reset behavior remains available to terminal callers. The UI controller intercepts every SpaceMouse `reset_requested` action and routes it through the same asynchronous Home state machine as the UI button; it never passes that flag to the blocking `FrankaRobot.send_action()` reset path.

### Dataset helpers

Existing dataset creation, frame packing, saving, clearing, and finalization remain the source of truth. Small pure helpers may be added for measured end-effector action construction, but the UI will not create an alternative dataset format.

## State machine

The worker starts in `CONNECTING`, connects the robot and SpaceMouse, then enters `IDLE`. It samples and sends SpaceMouse actions at the configured FPS in both `IDLE` and `RECORDING`; only `RECORDING` writes frames. `PREPARING` covers first-time dataset creation, and `SAVING` covers the synchronous episode save operation. `PAUSING_CLOSE` and `PAUSED_CLOSE` ensure all motion is stopped before the Tk thread asks how to handle a pending episode.

The meaningful transitions are:

```text
CONNECTING -> IDLE
IDLE -> PREPARING -> RECORDING -> SAVING -> IDLE
PREPARING -> IDLE (dataset creation rejected)
IDLE -> RECORDING (dataset already exists in this process)
IDLE -> HOMING_IDLE -> IDLE
RECORDING -> HOMING_RECORDING -> SAVING -> IDLE
RECORDING -> RECOVERING_RECORDING -> RECORDING
HOMING_RECORDING -> RECOVERING_RECORDING -> HOMING_RECORDING
IDLE/HOMING_IDLE -> RECOVERING_IDLE -> IDLE/HOMING_IDLE
RECORDING/HOMING_RECORDING -> FAULTED_RECORDING -> RECOVERING_RECORDING
IDLE/HOMING_IDLE -> FAULTED_IDLE -> RECOVERING_IDLE
unrecoverable failure -> FATAL_ERROR
active state -> PAUSING_CLOSE -> PAUSED_CLOSE
PAUSED_CLOSE -> previous safe state (Cancel)
PAUSED_CLOSE -> CLOSING -> CLOSED (Save or Discard)
```

An unexpected connection or control failure produces an error event and sends a best-effort zero-velocity command. Startup connection failure, dataset save failure, and unrecoverable internal errors enter `FATAL_ERROR`. A control or state failure that may be a Franka fault enters `FAULTED_IDLE` or `FAULTED_RECORDING`, where Clear Fault remains enabled. A fault during a pending episode marks the buffer as requiring successful recovery; it cannot be saved while in that condition. A failed Clear Fault request discards that buffer and enters `FATAL_ERROR`; it never silently saves a questionable episode.

## Recording data flow

### Normal teleoperation frame

At each FPS tick:

1. Read one atomic observation/raw-state sample from `FrankaRobot`.
2. Read the current SpaceMouse action.
3. Send the action through `FrankaRobot.send_action()`.
4. Pack the observation and returned, clamped action with the task string.
5. Add the frame to the current LeRobot episode buffer.

This preserves the existing terminal recorder's action semantics.

Normal frames store the returned command action. The accompanying raw state is used only for status and for the transition into measured recorded Home.

### Manual End Recording

The controller sends zero Cartesian velocity, stops adding frames, saves a non-empty episode, increments the saved episode counter, and returns to idle teleoperation. An episode stopped before its first frame is discarded instead of saved.

### Home without recording

The controller pauses SpaceMouse output, drains and clears SpaceMouse input, quiesces Cartesian velocity control, takes a fresh baseline state, and starts the configured joint-space home motion asynchronously. It polls atomic state samples at the capture rate. After all seven joints are within 0.02 radians of their targets for five consecutive distinct state snapshots, the worker polls `join(timeout=0)` without stopping state sampling. Once join returns true, one newer atomic state must still be observed and remain in tolerance before idle teleoperation resumes. No dataset is created or modified.

### Home while recording

The controller pauses and clears SpaceMouse input, quiesces Cartesian velocity control, captures a fresh baseline observation/state sample, and starts the same asynchronous joint-space home motion. The baseline sample is held until the next distinct robot-state sample is available. The recording loop continues to read observations and camera frames throughout the movement.

For the six arm action fields during this phase, the controller follows VITA's measured-motion policy:

1. Use the nested row-major 4x4 `O_T_EE` matrix returned in raw state `ee`, expressed in the robot base frame. Translation is in meters. Rotation is converted to roll/pitch/yaw in radians with the same XYZ convention and singularity handling as VITA's `matrix4_to_xyz_rpy` helper.
2. Subtract consecutive poses and wrap rotational differences into `[-pi, pi]`.
3. Pair the earlier held observation `obs[i-1]` with the measured transition ending at state `i`. This one-frame buffer makes the action forward-looking relative to its observation rather than attaching past motion to `obs[i]`.
4. For `cartesian_action_units=delta`, store the measured pose difference. For `velocity`, divide by the positive difference between the two monotonic robot-state timestamps.
5. Do not clip measured values: they describe the joint controller's actual motion and may exceed the teleoperation command limits. The UI logs the first limit exceedance in an episode for operator visibility.
6. Preserve the current binary gripper target.

Duplicate timestamps do not advance the one-frame buffer or the stable-sample count. A non-finite or incorrectly shaped `ee` matrix is retried under the existing state-wait budget; it is not emitted as a zero action because that would erase real home motion. Exceeding the state-wait budget enters the recoverable fault path.

This conversion deliberately avoids copying VITA's velocity values into LeRobot's default delta-action dataset. Normal teleoperation frames retain commanded-action labels; only externally controlled asynchronous homing frames use measured-action labels because no per-frame Cartesian command exists for that joint-space motion. The mixed source is intentional and documented: the action unit and forward transition meaning stay consistent, while the source changes from command to measurement during Home.

Each distinct state transition emits the previously held observation with its measured action, then replaces the held sample. After five consecutive distinct in-tolerance samples, the worker enters a settling phase: it polls `join(timeout=0)` at most once per 20 milliseconds while continuing to capture and emit every distinct state transition. Once join returns true, it waits for one newer atomic final sample. If that sample differs from the held sample, the held observation is emitted with that last measured transition; the new final observation is then emitted with a zero action. The final joints must still be in tolerance before the episode is saved automatically and the controller returns to idle.

The full `home_timeout_s` deadline covers motion, settling, join polling, and the fresh final sample. Timeout does not save automatically: the controller requests `stop_joint_position_control`, polls join until stopped or the transition timeout expires, marks the pending buffer as requiring recovery, enters `FAULTED_RECORDING`, and reports the final joint error and elapsed time. Successful Clear Fault establishes a new baseline, reissues Home, and continues the same buffer.

## Clear Fault behavior

Clear Fault always pauses SpaceMouse output, drains and clears its cached input, and attempts to quiesce Cartesian velocity control. If an asynchronous Home may be active, it also requests joint-position stop before recovery. A stop request that fails because the robot is faulted is logged but does not suppress recovery; after recovery, Cartesian quiescence and joint stop plus `join` must succeed before Home can be reissued.

- From idle: call the relative `recover` endpoint (final URL `/ctl/recover`) and return to idle after a neutral SpaceMouse sample on success.
- From normal recording: pause frame writes during the recovery request, then resume the same episode on success.
- From recorded homing: pause frame writes, stop the old joint motion, recover, confirm joint motion stopped, take a new baseline, reissue the asynchronous home request, and continue recorded homing on success.
- From unrecorded homing: recover, reissue home, and continue without recording.
- On any recovery failure, or failure to confirm stop after recovery: log the exception; if an episode is pending, clear its buffer and temporary images; enter `FATAL_ERROR`.

Pausing frame writes during recovery avoids inventing action labels while robot state may be unavailable. Resumed frames remain in the same LeRobot episode with consecutive frame indices. Recorded-Home recovery discards its held transition baseline and captures a new one, so unrecorded recovery motion cannot become one large measured action.

The duration limit and displayed recorded elapsed time use `saved_or_pending_frame_count / fps`, so recovery gaps are excluded. Wall elapsed time is displayed separately. Recorded Home ignores the limit until it completes or times out, ensuring that a requested home trajectory is never cut short by the timer.

## Shutdown and safety

Closing first asks the worker to enter `PAUSING_CLOSE`. The worker pauses and clears SpaceMouse input, quiesces Cartesian velocity control, stops and joins any asynchronous joint-position motion, then reports `PAUSED_CLOSE`. Only after that acknowledgement may the Tk thread show a pending-episode decision dialog.

If a normal, valid recording has a non-empty episode pending, the UI presents Save, Discard, and Cancel choices. An incomplete recorded Home, a buffer requiring recovery, or an invalid buffer offers only Discard and Cancel. Cancel restores ordinary Idle/Recording states only after the neutral-input latch clears; it restarts Home from a new baseline if the previous state was Homing, and returns without motion if the previous state was Faulted. Save or Discard proceeds to shutdown.

Shutdown disconnects the SpaceMouse and robot, finalizes the dataset if one was created, and joins the worker thread before destroying the window. Joint stop/join and zero velocity are repeated as best-effort cleanup. Individual cleanup failures are logged without skipping the remaining cleanup steps.

The worker is the only component allowed to issue motion commands. Repeated button presses are coalesced or rejected according to state, so duplicate Home or Save requests cannot overlap.

## Error reporting

- Initial connection failure is shown in the status and log; no controls that move the robot are enabled.
- Existing dataset roots are rejected using the current recorder policy and leave the dataset fields editable.
- Camera/state staleness uses the existing bounded retry policy. Exhaustion pauses motion and reports the underlying exception.
- Save or encoding failure enters `FATAL_ERROR` and does not increment the saved episode count.
- Home timeout reports target error and elapsed time and never claims that the episode was saved.
- All error events include a concise operator message plus the exception type/detail in the log.

## Verification strategy

Automated tests use fakes and do not require Tk, a display server, a SpaceMouse, a camera, or a Franka robot.

1. HTTP client tests verify `is_async=1`, relative `recover`, `/ctl/velocity_ws_status`, direct stops, join payloads/final URLs, rejection of HTTP-200 `is_ok: 0`, and the distinction between join-false and join-complete.
2. Robot adapter tests verify safe zero, transport quiescence ordering, asynchronous home, stop/join, atomic observation samples, distinct-snapshot home tolerance, and recovery delegation.
3. SpaceMouse tests verify queued-input drain, cached-state clearing, neutral re-arm, and interception of `reset_requested` by the controller.
4. Pure action tests cover forward alignment, delta units, velocity units, wrapped angles, duplicate timestamps, invalid matrices, and un-clipped measured values.
5. Controller tests verify idle teleoperation, Start/End save, zero-frame discard, duration auto-save, multiple episodes, and dataset field locking events.
6. Recorded-home tests prove that frames continue to be added while home and join settling are active, the first and final motion intervals are retained, measured actions are nonzero, a fresh post-join final observation is present, the duration limit cannot truncate Home, and save occurs only after five distinct stable samples plus join completion.
7. Recovery tests prove successful continuation of the same episode, stop-before-restart, baseline reset, frame-write pause, neutral-input gating, and failed-recovery discard.
8. Shutdown tests verify motion is stopped before the decision dialog, episode-validity choices, Cancel restoration, finalization, disconnect order, and worker termination.
9. Existing Franka hardware-test unit tests, Ruff on changed files, and Python compilation must pass.

Hardware validation remains a manual final step because automated tests cannot prove physical motion or camera timing. The manual checklist is: connect, idle jog in both Z directions, record and stop one episode, record and Home another episode, inspect both episode counts/frame counts, induce or simulate a recoverable fault, clear it, and record a final episode.

## Planned file scope

- Add `hardware_test/franka/run_record_ui.py` for the CLI and Tkinter view.
- Add `hardware_test/franka/scripts/start_franka_record_ui.sh` with `start-ui`, `stop-ui`, and `status` commands.
- Add a focused controller module under `hardware_test/franka/` for the state machine and recording loop.
- Extend `hardware_test/franka/franka_robot.py` with atomic samples, async home, joint stop/join, and recovery operations.
- Extend `hardware_test/franka/franka_spacemouse_teleop.py` with safe input clearing used by Home, recovery, and shutdown.
- Extend `hardware_test/franka/record_lerobot_dataset.py` only with reusable pure recording helpers needed by the controller.
- Add focused UI/controller tests under `hardware_test/franka/`.
- Update `hardware_test/franka/README.md` with the launch command, controls, dataset lifecycle, and safety behavior.
