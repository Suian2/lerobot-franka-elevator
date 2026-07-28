# Franka ACT Unconditioned RealSense Rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone Franka ACT rollout for checkpoints whose inputs are only eight-dimensional robot state and direct L515 RGB images, with no target-floor feature.

**Architecture:** Preserve the existing conditioned direct-RealSense runner unchanged. Create an unconditioned sibling by retaining its safety, policy, camera, Franka transport, and teardown behavior while deleting only the floor-specific CLI, observation, validation, and logging paths.

**Tech Stack:** Python 3.12, PyTorch, NumPy, LeRobot ACT processors, LeRobot `RealSenseCameraConfig`, pytest, Ruff.

---

## File Structure

- Create `hardware_test/franka/run_act_rollout_realsense_unconditioned.py`: standalone direct-L515 unconditioned ACT deployment command.
- Create `hardware_test/franka/test_act_rollout_realsense_unconditioned.py`: focused contract tests for the new runner.
- Keep `hardware_test/franka/run_act_rollout_realsense.py` and its tests unchanged as the multi-floor implementation.

### Task 1: Lock the unconditioned CLI and observation schema

**Files:**
- Create: `hardware_test/franka/test_act_rollout_realsense_unconditioned.py`
- Create: `hardware_test/franka/run_act_rollout_realsense_unconditioned.py`

- [x] **Step 1: Write failing tests for the new module contract**

Create tests that import `hardware_test.franka.run_act_rollout_realsense_unconditioned` and assert:

```python
def test_parser_does_not_accept_or_require_target_floor():
    rollout = _load_rollout_module()
    args = rollout.build_arg_parser().parse_args(["--policy-path", "/tmp/policy"])
    assert not hasattr(args, "target_floor")
    with pytest.raises(SystemExit):
        rollout.build_arg_parser().parse_args(
            ["--policy-path", "/tmp/policy", "--target-floor", "4"]
        )


def test_build_policy_observation_has_only_unconditioned_inputs():
    rollout = _load_rollout_module()
    result = rollout.build_policy_observation(_make_observation())
    assert set(result) == {"observation.state", "observation.images.l515"}
    assert result["observation.state"].shape == (8,)
    assert result["observation.images.l515"].shape == (3, 540, 960)
```

Add schema fixtures and assertions proving an ACT config with state `(8,)`, L515 `(3, 540, 960)`, and action `(7,)` passes, while a config containing `OBS_ENV_STATE` fails with an error containing `conditioned checkpoint`.

- [x] **Step 2: Run the tests and verify RED**

Run:

```bash
uv run pytest -q hardware_test/franka/test_act_rollout_realsense_unconditioned.py
```

Expected: collection fails because `hardware_test.franka.run_act_rollout_realsense_unconditioned` does not exist.

- [x] **Step 3: Add the minimal standalone runner**

Create the new module from the existing conditioned runner, retaining the existing public helpers and safety constants. Make the unconditioned observation builder return exactly:

```python
return {
    "observation.state": torch.from_numpy(state),
    "observation.images.l515": image_tensor.contiguous(),
}
```

Make checkpoint validation reject any floor-conditioned input:

```python
if FLOOR_CONDITION_KEY in config.input_features:
    raise ValueError(
        f"conditioned checkpoint includes {FLOOR_CONDITION_KEY}; "
        "use run_act_rollout_realsense.py with --target-floor"
    )
```

Define `FLOOR_CONDITION_KEY` locally from `lerobot.utils.constants.OBS_ENV_STATE` so the runner does not import the Franka floor encoding module. Preserve validation of the two expected inputs and seven-dimensional action.

Remove `floor_condition` from `select_robot_action`, `run_control_loop`, and their call sites. The parser must contain all existing options except `--target-floor`. `main()` must print only checkpoint and direct-camera information before loading hardware.

- [x] **Step 4: Run the tests and verify GREEN**

Run:

```bash
uv run pytest -q hardware_test/franka/test_act_rollout_realsense_unconditioned.py
```

Expected: all new tests pass.

### Task 2: Lock direct RealSense and safe execution behavior

**Files:**
- Modify: `hardware_test/franka/test_act_rollout_realsense_unconditioned.py`
- Modify: `hardware_test/franka/run_act_rollout_realsense_unconditioned.py`

- [x] **Step 1: Add failing integration-boundary tests**

Add tests equivalent to these contracts:

```python
def test_build_robot_uses_direct_realsense(monkeypatch):
    # Replace lerobot.cameras.realsense.RealSenseCameraConfig with a capturing fake.
    # Assert the l515 camera is 960x540 RGB at 30 FPS with depth disabled.
    # Assert Franka transport remains ZMQ velocity transport.


def test_source_has_no_ros2_or_image_bridge_dependency():
    source = Path(_load_rollout_module().__file__).read_text()
    assert "RealSenseCameraConfig" in source
    for forbidden in (
        "ros2_image_bridge",
        "ZmqRgbImageClient",
        "image_zmq",
        "rclpy",
        "sensor_msgs",
    ):
        assert forbidden not in source


def test_main_runs_without_floor_condition(monkeypatch):
    # Capture print/load/connect/run/disconnect events.
    # Assert checkpoint and direct-camera messages precede connection.
    # Assert run_control_loop is called without a floor_condition keyword.
```

Also retain focused tests for seven-dimensional action conversion, inference-state reset, dry-run behavior, zero-velocity bracketing, and teardown by adapting the existing conditioned test patterns without a floor argument.

- [x] **Step 2: Run the new tests and verify RED where behavior is missing**

Run:

```bash
uv run pytest -q hardware_test/franka/test_act_rollout_realsense_unconditioned.py
```

Expected: any omitted direct-camera, lifecycle, or no-floor behavior fails with a targeted assertion.

- [x] **Step 3: Complete only the behavior required by the failing tests**

Retain the current `RealSenseCameraConfig` construction:

```python
camera_config = RealSenseCameraConfig(
    serial_number_or_name=camera_serial_or_name,
    fps=30,
    width=960,
    height=540,
    warmup_s=1,
    use_rgb=True,
    use_depth=False,
)
```

Retain dry-run inference, `--execute` gating, action scaling, velocity bounds, gripper suppression, signal handling, double zero-velocity safety, and robot disconnect behavior. Do not add compatibility branches or shared abstractions.

- [x] **Step 4: Run new and existing rollout tests**

Run:

```bash
uv run pytest -q \
  hardware_test/franka/test_act_rollout_realsense_unconditioned.py \
  hardware_test/franka/test_act_rollout_realsense.py
```

Expected: both unconditioned and conditioned suites pass.

### Task 3: Static verification and handoff

**Files:**
- Verify: `hardware_test/franka/run_act_rollout_realsense_unconditioned.py`
- Verify: `hardware_test/franka/test_act_rollout_realsense_unconditioned.py`

- [x] **Step 1: Run Ruff**

```bash
uv run ruff check \
  hardware_test/franka/run_act_rollout_realsense_unconditioned.py \
  hardware_test/franka/test_act_rollout_realsense_unconditioned.py
```

Expected: exit code 0 with no diagnostics.

- [x] **Step 2: Compile the runner**

```bash
uv run python -m py_compile \
  hardware_test/franka/run_act_rollout_realsense_unconditioned.py
```

Expected: exit code 0.

- [x] **Step 3: Verify CLI help without hardware access**

```bash
PYTHONPATH=src uv run python \
  hardware_test/franka/run_act_rollout_realsense_unconditioned.py --help
```

Expected: help lists direct RealSense and safety options, does not list `--target-floor` or `--image-zmq`, and exits 0.

- [x] **Step 4: Review the final diff**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only the new rollout/test/plan plus pre-existing unrelated workspace changes are present.

- [x] **Step 5: Report the safe invocation**

Provide the user the new command path and preserve their existing arguments verbatim except for switching the script filename and omitting `--target-floor`.
