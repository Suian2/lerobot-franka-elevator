# Franka ACT Multi-Floor Direct RealSense Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a tested physical rollout entry point for floors 1, 4, and 5 that reads L515 RGB directly through `pyrealsense2`/OpenCV and supplies the canonical five-dimensional floor condition to the trained ACT checkpoint.

**Architecture:** Keep the direct camera entry point isolated in `run_act_rollout_realsense.py`. Copy the already-tested conditioned policy boundary from `run_act_rollout.py`, while retaining a `RealSenseCameraConfig`-owned camera and a source-level prohibition on ROS/ZMQ image bridge dependencies.

**Tech Stack:** Python 3.12, PyTorch, LeRobot ACT processors, LeRobot RealSenseCamera, pyrealsense2, OpenCV/NumPy, pytest, ruff.

---

### Task 1: Lock the conditioned direct-camera contract

**Files:**
- Modify: `hardware_test/franka/test_act_rollout_realsense.py`
- Test: `hardware_test/franka/test_act_rollout_realsense.py`

- [ ] **Step 1: Add parser and observation tests**

Require `--target-floor`, accept only 1, 4, and 5, and verify that
`build_policy_observation(..., floor_condition=encode_target_floor(4))`
returns an unchanged eight-dimensional robot state plus a float32 `(5,)`
`observation.environment_state` tensor.

- [ ] **Step 2: Add checkpoint and processor-boundary tests**

Assert that `validate_policy_features` requires
`observation.environment_state: (5,)`, and that `select_robot_action` rejects a
processed condition whose shape is not `(1, 5)` or whose dtype is not
`torch.float32`.

- [ ] **Step 3: Strengthen the direct-camera dependency test**

Assert that the direct source contains `RealSenseCameraConfig` and contains none
of `ros2_image_bridge`, `ZmqRgbImageClient`, `image_zmq`, `rclpy`, or
`sensor_msgs`.

- [ ] **Step 4: Run the tests and verify RED**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  hardware_test/franka/test_act_rollout_realsense.py
```

Expected: failures for the missing required `--target-floor` and missing floor
condition in policy observations/checkpoint validation.

### Task 2: Implement canonical floor conditioning in the direct entry point

**Files:**
- Modify: `hardware_test/franka/run_act_rollout_realsense.py`
- Test: `hardware_test/franka/test_act_rollout_realsense.py`

- [ ] **Step 1: Import the shared floor contract**

Import `FLOOR_CONDITION_KEY`, `NUM_ELEVATOR_FLOORS`,
`TRAINED_ROLLOUT_FLOORS`, and `encode_target_floor` from
`hardware_test.franka.floor_condition`.

- [ ] **Step 2: Add condition validation and preprocessing**

Add `_floor_condition_tensor`, pass `floor_condition` through
`build_policy_observation` and `select_robot_action`, and verify the processed
condition is finite `torch.float32` with shape `(1, 5)` before calling ACT.

- [ ] **Step 3: Enforce the conditioned checkpoint contract**

Add `observation.environment_state: (5,)` to the required checkpoint inputs so
an unconditioned checkpoint cannot run silently.

- [ ] **Step 4: Add floor selection and reset-per-run behavior**

Require `--target-floor` with choices `(1, 4, 5)`, encode it before robot
connection, print the floor/one-hot/checkpoint/direct camera identifier, and
pass the condition into `run_control_loop`. Preserve
`reset_inference_state(bundle)` at the start of every run.

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  hardware_test/franka/test_act_rollout_realsense.py \
  hardware_test/franka/test_act_rollout.py
```

Expected: all direct and conditioned rollout tests pass.

### Task 3: Verify the real checkpoint and direct L515 path

**Files:**
- Verify: `outputs/train/act_press_button_floors_1_4_5/checkpoints/last/pretrained_model`
- Verify: `hardware_test/franka/run_act_rollout_realsense.py`

- [ ] **Step 1: Run static checks**

```bash
uv run ruff check \
  hardware_test/franka/run_act_rollout_realsense.py \
  hardware_test/franka/test_act_rollout_realsense.py
uv run ruff format --check \
  hardware_test/franka/run_act_rollout_realsense.py \
  hardware_test/franka/test_act_rollout_realsense.py
```

Expected: both commands exit zero.

- [ ] **Step 2: Verify hardware prerequisites without motion**

Confirm `checkpoints/last` resolves to step 100000, CUDA is available, and
`rs-enumerate-devices -s` reports the unique device name
`Intel RealSense L515`. Use that name for the rollout because this version of
the RealSense wrapper treats the alphanumeric serial `f1381152` as a name.

- [ ] **Step 3: Run one dry-run per floor**

Invoke the direct entry point without `--execute` for floors 1, 4, and 5. Each
run must load the real checkpoint, open the direct L515 stream, print the
canonical one-hot, process a `(1, 5)` condition, produce a finite action, send
no policy action, then disconnect.

### Task 4: Hand off physical commands

**Files:**
- No code changes.

- [ ] **Step 1: Provide three explicit physical commands**

Provide separate commands for floors 1, 4, and 5 using the same step-100000
checkpoint, unique device name `Intel RealSense L515`, and explicit bounded
motion parameters. Include `--execute` only in the physical commands and state
that each process resets the cached ACT action chunk.
