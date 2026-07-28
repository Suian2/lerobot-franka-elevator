# Franka ACT Safe Rollout Runner Design

Date: 2026-07-10

## Context

The trained ACT checkpoint is located at:

`outputs/train/act_press_button_29ep_20260710/checkpoints/last/pretrained_model`

It expects an eight-element state vector (seven joint positions plus
`gripper_width_norm`), one `l515` RGB image with shape `3x540x960`, and produces
seven actions (six `delta_ee_pose` values plus `gripper_cmd_bin`).

The generic `lerobot-rollout` path cannot be used for this adapter because its
hardware feature reconciliation currently keeps only scalar keys ending in
`.pos`. That drops the gripper state and every delta end-effector action. The
generic ZMQ camera also uses a JSON/JPEG protocol that is incompatible with the
raw RGB protocol used by `hardware_test/cameras/ros2_image_bridge.py`.

The dataset also contains a gripper-label inconsistency. The physical
`gripper_width_norm` is approximately `0.008811` for every frame (closed), while
24,252 of 24,281 action labels are `gripper_cmd_bin=1`, which the Franka adapter
defines as open. The first rollout must therefore never forward policy gripper
output to the robot.

## Goal

Provide a narrow, synchronous ACT runner for a five-second, reduced-speed
physical smoke test using the same Franka state and ROS2-to-ZMQ L515 image path
as recording.

The runner must:

- execute only the six end-effector delta actions;
- leave the gripper completely untouched;
- infer from the latest observation at 30 Hz and execute only the first action
  of each predicted ACT chunk;
- scale the six pose deltas by `0.25`;
- cap linear velocity at `0.01 m/s` and angular velocity at `0.08 rad/s`;
- run for five seconds by default;
- send zero Cartesian velocity on startup, normal exit, Ctrl-C, timeout, and
  every exception;
- never home or otherwise reposition the robot automatically.

## Non-goals

- No change to generic `lerobot-rollout`.
- No recording, Hub upload, DAgger, RTC, or asynchronous policy server.
- No gripper-policy evaluation or control.
- No automatic homing or recovery operation.
- No claim of task success from this smoke test.

## Approaches Considered

### Dedicated synchronous runner (selected)

Add `hardware_test/franka/run_act_rollout.py`. It directly reuses
`FrankaRobot`, `FrankaRobotConfig`, and `ZmqRgbImageClient`, loads the checkpoint
and saved processors, and performs a small explicit control loop. This keeps the
custom protocol and safety policy visible and testable without changing LeRobot
core behavior.

### Extend generic `lerobot-rollout` (rejected)

This would require changing core feature filtering, custom robot registration,
and camera construction. It is broader than the hardware-test objective and
could alter behavior for unrelated robots.

### Use async policy server/client (rejected)

The client cannot consume the recorder's raw RGB ZMQ messages, has no narrow
first-test action-scaling boundary, and adds queueing behavior that makes the
first physical test harder to reason about.

## Architecture and Data Flow

1. Parse and validate CLI settings without touching hardware.
2. Import/register ACT, load the checkpoint, load its saved preprocessor and
   postprocessor, move the policy to CUDA, and set evaluation mode.
3. Construct `ZmqRgbImageClient` for `tcp://127.0.0.1:5557` and inject it into a
   `FrankaRobot` configured for 30 Hz delta-pose control.
4. Connect the camera/state path and capture one observation.
5. Build `observation.state` in the exact order
   `joint_1.pos` through `joint_7.pos`, then `gripper_width_norm`.
6. Convert the L515 RGB image from HWC `uint8` to CHW `float32` in `[0, 1]`.
7. Apply the checkpoint preprocessor, call `predict_action_chunk`, take action
   `[0, 0]`, and apply the checkpoint postprocessor.
8. Discard dimension seven (`gripper_cmd_bin`) unconditionally.
9. Multiply the six pose deltas by `0.25`, map them to the Franka delta-pose
   keys, and call `FrankaRobot.send_action` without a gripper key.
10. Maintain the 30 Hz loop until five seconds elapse or a stop signal arrives.
11. In `finally`, send zero Cartesian velocity and disconnect. Do not home.

The policy is queried every control tick. The checkpoint's configured
`n_action_steps=100` is deliberately bypassed for this smoke test, preventing a
multi-second open-loop action sequence.

## CLI Modes

The default mode is a live read-only dry run. It connects to the state and image
sources, validates inputs, performs inference, and prints predicted actions, but
never calls `send_action`.

Physical motion requires the explicit `--execute` flag. Both modes use the same
checkpoint, observation construction, preprocessing, and inference path.

Relevant options:

- `--policy-path`
- `--control-host` (shared Franka default)
- `--image-zmq` (default `tcp://127.0.0.1:5557`)
- `--fps` (fixed to `30` for this checkpoint)
- `--duration-s` (default and maximum: `5`)
- `--action-scale` (default and maximum for the smoke test: `0.25`)
- `--max-linear-velocity` (default and maximum: `0.01`)
- `--max-angular-velocity` (default and maximum: `0.08`)
- `--execute`

Values above the approved action scale or velocity limits are rejected rather
than silently accepted.

## Safety and Error Handling

- Validate checkpoint features, observation keys, state shape, image shape,
  data types, and finite values before enabling motion.
- Run an inference warm-up without sending a nonzero command.
- Send zero velocity immediately before the timed execution window.
- Treat stale state, stale image, model error, non-finite output, shape mismatch,
  and missed operator stop as fatal to the run.
- Clip through the Franka adapter after action scaling as a second boundary.
- Ignore policy gripper output for every mode and every code path.
- Handle SIGINT and SIGTERM with the same zero-velocity teardown.
- Leave the arm at its final pose; the operator owns initial placement and any
  later home/recovery command.

Before `--execute`, the operator must place the robot at the same initial pose
used for data collection, clear the workspace, hold the emergency stop, and
confirm the L515 view matches the demonstrations.

## Verification

Unit tests use fake policy, processors, camera, and robot objects to prove:

- exact state and image tensor construction;
- first-action-only ACT behavior;
- pose scaling and key order;
- gripper output is never sent;
- approved limits cannot be exceeded;
- dry-run never calls `send_action`;
- startup and all exit/error paths send zero velocity;
- shape, stale-data, and non-finite checks fail closed.

An offline integration smoke test loads the real checkpoint and one recorded
dataset observation, then verifies a finite `1x100x7` prediction and successful
postprocessing. Hardware execution is never part of the automated test suite.

## Acceptance Criteria

- Tests pass without robot or camera hardware.
- Dry-run successfully loads the final checkpoint and processes a live L515 and
  Franka observation without sending motion.
- `--execute` is the only path that can send nonzero pose commands.
- The first physical command runs for at most five seconds with the approved
  scale and velocity limits, never changes the gripper, and always finishes by
  sending zero Cartesian velocity.
