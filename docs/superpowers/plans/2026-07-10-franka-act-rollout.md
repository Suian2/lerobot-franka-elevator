# Franka ACT Safe Rollout Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a testable, synchronous Franka ACT runner that performs a five-second reduced-speed physical smoke test while permanently suppressing the mislabeled gripper action.

**Architecture:** Add one hardware-test runner and one focused test module. The runner constructs the exact eight-state/one-image policy observation, calls `predict_action_chunk` every control tick, postprocesses only the first action, drops the gripper dimension, scales the six pose deltas, and executes them through the existing `FrankaRobot` with hard velocity and duration limits. The generic LeRobot rollout stack remains unchanged.

**Tech Stack:** Python 3.12, PyTorch, LeRobot ACT policy/processors, NumPy, pytest, existing Franka and ROS2-to-ZMQ adapters.

**Workspace constraint:** The required Franka adapter and camera bridge are uncommitted user work in the current worktree. Do not create a clean worktree that omits them, and never commit unrelated staged files. Every commit must use `git commit --only` with the exact new runner/test/plan paths.

---

## File Structure

- Create `hardware_test/franka/run_act_rollout.py`: CLI, safety validation, observation conversion, checkpoint loading, first-action inference, robot construction, timed control loop, and fail-closed teardown.
- Create `hardware_test/franka/test_act_rollout.py`: unit tests with fake policies/processors/robots plus configuration tests; no real hardware access.
- Keep `src/lerobot/rollout/*`, `hardware_test/franka/franka_robot.py`, and the dirty recorder files unchanged.

### Task 1: Safety Configuration and Pure Data Conversion

**Files:**
- Create: `hardware_test/franka/run_act_rollout.py`
- Create: `hardware_test/franka/test_act_rollout.py`

- [ ] **Step 1: Write failing tests for hard limits, observation construction, and action mapping**

```python
import numpy as np
import pytest
import torch

from hardware_test.franka.franka_robot import DELTA_EE_KEYS, JOINT_KEYS
from hardware_test.franka.run_act_rollout import (
    RolloutSafetyConfig,
    build_policy_observation,
    policy_action_to_robot_action,
)


def make_observation():
    return {
        **{key: float(index) / 10 for index, key in enumerate(JOINT_KEYS, start=1)},
        "gripper_width_norm": 0.008811,
        "l515": np.full((540, 960, 3), 127, dtype=np.uint8),
    }


def test_safety_config_rejects_values_above_approved_limits():
    with pytest.raises(ValueError, match="duration_s"):
        RolloutSafetyConfig(duration_s=5.01)
    with pytest.raises(ValueError, match="action_scale"):
        RolloutSafetyConfig(action_scale=0.251)
    with pytest.raises(ValueError, match="max_linear_velocity"):
        RolloutSafetyConfig(max_linear_velocity=0.011)
    with pytest.raises(ValueError, match="max_angular_velocity"):
        RolloutSafetyConfig(max_angular_velocity=0.081)
    with pytest.raises(ValueError, match="fps"):
        RolloutSafetyConfig(fps=29)


def test_build_policy_observation_uses_training_order_and_image_layout():
    result = build_policy_observation(make_observation())
    assert result["observation.state"].shape == (8,)
    assert result["observation.state"].tolist() == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.008811]
    )
    assert result["observation.images.l515"].shape == (3, 540, 960)
    assert result["observation.images.l515"].dtype == torch.float32
    assert result["observation.images.l515"].min() >= 0
    assert result["observation.images.l515"].max() <= 1


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda obs: obs.pop("joint_7.pos"), "joint_7.pos"),
        (lambda obs: obs.update({"l515": np.zeros((480, 640, 3), dtype=np.uint8)}), "shape"),
        (lambda obs: obs.update({"joint_1.pos": float("nan")}), "finite"),
    ],
)
def test_build_policy_observation_fails_closed(mutate, match):
    observation = make_observation()
    mutate(observation)
    with pytest.raises(ValueError, match=match):
        build_policy_observation(observation)


def test_policy_action_mapping_scales_pose_and_drops_gripper():
    policy_action = torch.tensor([0.004, -0.004, 0.002, 0.04, -0.04, 0.02, 1.0])
    result = policy_action_to_robot_action(policy_action, action_scale=0.25)
    assert list(result) == list(DELTA_EE_KEYS)
    assert list(result.values()) == pytest.approx([0.001, -0.001, 0.0005, 0.01, -0.01, 0.005])
    assert "gripper_cmd_bin" not in result


def test_policy_action_mapping_rejects_wrong_shape_and_non_finite_values():
    with pytest.raises(ValueError, match="shape"):
        policy_action_to_robot_action(torch.zeros(6), action_scale=0.25)
    bad = torch.zeros(7)
    bad[2] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        policy_action_to_robot_action(bad, action_scale=0.25)
```

- [ ] **Step 2: Run the focused tests and verify the module is missing**

Run:

```bash
source /home/yanrihong/rs_modes/env_lerobot_sdk.sh
pytest -q hardware_test/franka/test_act_rollout.py
```

Expected: collection fails with `ModuleNotFoundError: hardware_test.franka.run_act_rollout`.

- [ ] **Step 3: Implement the minimal safety and conversion layer**

Add the path bootstrap already used by `hardware_test/franka/run_record.py`, then implement:

```python
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (SRC_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hardware_test.franka.franka_robot import DELTA_EE_KEYS, JOINT_KEYS

CAMERA_KEY = "l515"
EXPECTED_IMAGE_SHAPE = (540, 960, 3)
EXPECTED_STATE_KEYS = (*JOINT_KEYS, "gripper_width_norm")
EXPECTED_ACTION_DIM = 7
APPROVED_FPS = 30
MAX_DURATION_S = 5.0
MAX_ACTION_SCALE = 0.25
MAX_LINEAR_VELOCITY = 0.01
MAX_ANGULAR_VELOCITY = 0.08


@dataclass(frozen=True)
class RolloutSafetyConfig:
    execute: bool = False
    fps: int = APPROVED_FPS
    duration_s: float = MAX_DURATION_S
    action_scale: float = MAX_ACTION_SCALE
    max_linear_velocity: float = MAX_LINEAR_VELOCITY
    max_angular_velocity: float = MAX_ANGULAR_VELOCITY

    def __post_init__(self) -> None:
        checks = (
            (self.fps == APPROVED_FPS, "fps must be exactly 30"),
            (0 < self.duration_s <= MAX_DURATION_S, "duration_s must be in (0, 5]"),
            (0 < self.action_scale <= MAX_ACTION_SCALE, "action_scale must be in (0, 0.25]"),
            (
                0 < self.max_linear_velocity <= MAX_LINEAR_VELOCITY,
                "max_linear_velocity must be in (0, 0.01]",
            ),
            (
                0 < self.max_angular_velocity <= MAX_ANGULAR_VELOCITY,
                "max_angular_velocity must be in (0, 0.08]",
            ),
        )
        for valid, message in checks:
            if not valid:
                raise ValueError(message)


def build_policy_observation(observation: dict[str, Any]) -> dict[str, torch.Tensor]:
    missing = [key for key in (*EXPECTED_STATE_KEYS, CAMERA_KEY) if key not in observation]
    if missing:
        raise ValueError(f"observation is missing required keys: {missing}")
    state = np.asarray([observation[key] for key in EXPECTED_STATE_KEYS], dtype=np.float32)
    if state.shape != (8,):
        raise ValueError(f"observation.state has shape {state.shape}, expected (8,)")
    if not np.isfinite(state).all():
        raise ValueError("observation.state must contain only finite values")
    image = np.asarray(observation[CAMERA_KEY])
    if image.shape != EXPECTED_IMAGE_SHAPE:
        raise ValueError(f"l515 image has shape {image.shape}, expected {EXPECTED_IMAGE_SHAPE}")
    if image.dtype != np.uint8:
        raise ValueError(f"l515 image must be uint8, got {image.dtype}")
    image_tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div_(255.0)
    return {
        "observation.state": torch.from_numpy(state),
        "observation.images.l515": image_tensor.contiguous(),
    }


def policy_action_to_robot_action(action: torch.Tensor, *, action_scale: float) -> dict[str, float]:
    values = torch.as_tensor(action).detach().cpu().float()
    if values.shape != (EXPECTED_ACTION_DIM,):
        raise ValueError(f"policy action has shape {tuple(values.shape)}, expected (7,)")
    if not torch.isfinite(values).all():
        raise ValueError("policy action must contain only finite values")
    scaled_pose = values[:6] * float(action_scale)
    return {key: float(value) for key, value in zip(DELTA_EE_KEYS, scaled_pose, strict=True)}
```

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `pytest -q hardware_test/franka/test_act_rollout.py`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit only the two new files**

```bash
git add hardware_test/franka/run_act_rollout.py hardware_test/franka/test_act_rollout.py
git commit --only hardware_test/franka/run_act_rollout.py hardware_test/franka/test_act_rollout.py \
  -m "Fail closed before translating ACT output into Franka motion" \
  -m $'Constraint: First physical test is limited to 5 seconds, 25% pose deltas, 0.01 m/s, and 0.08 rad/s\nDirective: Never add gripper_cmd_bin to the robot action until the dataset label mismatch is resolved\nTested: Focused pure conversion and validation tests\nConfidence: high\nScope-risk: narrow'
```

### Task 2: Checkpoint Loading and First-Action Inference

**Files:**
- Modify: `hardware_test/franka/run_act_rollout.py`
- Modify: `hardware_test/franka/test_act_rollout.py`

- [ ] **Step 1: Add failing tests for feature validation and first-action-only inference**

```python
from types import SimpleNamespace

from hardware_test.franka.run_act_rollout import (
    PolicyBundle,
    predict_first_robot_action,
    validate_policy_features,
)


class IdentityProcessor:
    def __call__(self, value):
        return value


class FakePolicy:
    def __init__(self, output):
        self.output = output
        self.calls = 0

    def predict_action_chunk(self, batch):
        self.calls += 1
        assert batch["observation.state"].shape == (8,)
        return self.output


def test_validate_policy_features_accepts_current_checkpoint_contract():
    config = SimpleNamespace(
        type="act",
        input_features={
            "observation.state": SimpleNamespace(shape=(8,)),
            "observation.images.l515": SimpleNamespace(shape=(3, 540, 960)),
        },
        output_features={"action": SimpleNamespace(shape=(7,))},
    )
    validate_policy_features(config)


def test_predict_first_robot_action_uses_first_action_and_never_returns_gripper():
    chunk = torch.zeros(1, 100, 7)
    chunk[0, 0] = torch.tensor([0.004, -0.004, 0.002, 0.04, -0.04, 0.02, 0.0])
    chunk[0, 1] = 999
    bundle = PolicyBundle(FakePolicy(chunk), IdentityProcessor(), IdentityProcessor())
    result = predict_first_robot_action(bundle, make_observation(), action_scale=0.25)
    assert bundle.policy.calls == 1
    assert list(result) == list(DELTA_EE_KEYS)
    assert list(result.values()) == pytest.approx([0.001, -0.001, 0.0005, 0.01, -0.01, 0.005])


def test_predict_first_robot_action_rejects_wrong_chunk_shape():
    bundle = PolicyBundle(FakePolicy(torch.zeros(1, 7)), IdentityProcessor(), IdentityProcessor())
    with pytest.raises(ValueError, match="chunk"):
        predict_first_robot_action(bundle, make_observation(), action_scale=0.25)
```

- [ ] **Step 2: Run the new tests and verify they fail for missing inference APIs**

Run: `pytest -q hardware_test/franka/test_act_rollout.py -k 'policy_features or predict_first'`

Expected: import errors for `PolicyBundle`, `validate_policy_features`, and `predict_first_robot_action`.

- [ ] **Step 3: Implement checkpoint loading and synchronous first-action inference**

```python
@dataclass(frozen=True)
class PolicyBundle:
    policy: Any
    preprocessor: Any
    postprocessor: Any


def validate_policy_features(config: Any) -> None:
    if config.type != "act":
        raise ValueError(f"expected an ACT checkpoint, got {config.type!r}")
    expected_inputs = {
        "observation.state": (8,),
        "observation.images.l515": (3, 540, 960),
    }
    for key, shape in expected_inputs.items():
        feature = config.input_features.get(key)
        if feature is None or tuple(feature.shape) != shape:
            actual = None if feature is None else tuple(feature.shape)
            raise ValueError(f"checkpoint feature {key} has shape {actual}, expected {shape}")
    action = config.output_features.get("action")
    if action is None or tuple(action.shape) != (7,):
        actual = None if action is None else tuple(action.shape)
        raise ValueError(f"checkpoint action has shape {actual}, expected (7,)")


def load_policy_bundle(policy_path: str | Path, *, device: str = "cuda") -> PolicyBundle:
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import ACTConfig as _ACTConfig
    from lerobot.policies import get_policy_class, make_pre_post_processors

    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the approved physical smoke test requires an available CUDA device")
    path = Path(policy_path).expanduser().resolve()
    config = PreTrainedConfig.from_pretrained(path, local_files_only=True)
    validate_policy_features(config)
    config.device = device
    policy = get_policy_class(config.type).from_pretrained(
        path, config=config, local_files_only=True
    ).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return PolicyBundle(policy, preprocessor, postprocessor)


def predict_first_robot_action(
    bundle: PolicyBundle,
    raw_observation: dict[str, Any],
    *,
    action_scale: float,
) -> dict[str, float]:
    observation = build_policy_observation(raw_observation)
    processed = bundle.preprocessor(observation)
    with torch.inference_mode():
        predicted = bundle.policy.predict_action_chunk(processed)
        predicted = bundle.postprocessor(predicted)
    chunk = torch.as_tensor(predicted).detach().cpu()
    if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[1] < 1 or chunk.shape[2] != 7:
        raise ValueError(f"postprocessed action chunk has shape {tuple(chunk.shape)}, expected (1, N, 7)")
    return policy_action_to_robot_action(chunk[0, 0], action_scale=action_scale)
```

- [ ] **Step 4: Run the entire focused test module**

Run: `pytest -q hardware_test/franka/test_act_rollout.py`

Expected: all Task 1 and Task 2 tests pass.

- [ ] **Step 5: Commit the inference boundary**

```bash
git add hardware_test/franka/run_act_rollout.py hardware_test/franka/test_act_rollout.py
git commit --only hardware_test/franka/run_act_rollout.py hardware_test/franka/test_act_rollout.py \
  -m "Recompute one ACT action from each fresh Franka observation" \
  -m $'Constraint: The checkpoint action queue must not execute a 100-step open-loop chunk during the first physical test\nRejected: ACT select_action queue | would reuse stale observations for up to 100 ticks\nDirective: Keep predict_action_chunk followed by [0, 0] selection for the approved smoke test\nTested: Fake-policy feature, shape, and first-action tests\nConfidence: high\nScope-risk: narrow'
```

### Task 3: Fail-Closed Timed Control Loop and Robot Construction

**Files:**
- Modify: `hardware_test/franka/run_act_rollout.py`
- Modify: `hardware_test/franka/test_act_rollout.py`

- [ ] **Step 1: Add failing tests for dry-run, timed execution, gripper suppression, and exception teardown**

```python
from threading import Event

from hardware_test.franka.run_act_rollout import build_robot, run_control_loop


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds)


class FakeRobot:
    def __init__(self, observation):
        self.observation = observation
        self.actions = []
        self.zero_calls = 0

    def get_observation(self):
        return self.observation

    def send_action(self, action):
        self.actions.append(dict(action))

    def send_zero_cartesian_velocity(self):
        self.zero_calls += 1


def make_bundle():
    chunk = torch.zeros(1, 100, 7)
    chunk[0, 0, :6] = 0.004
    chunk[0, 0, 6] = 1.0
    return PolicyBundle(FakePolicy(chunk), IdentityProcessor(), IdentityProcessor())


def test_dry_run_infers_once_without_sending_policy_action():
    robot = FakeRobot(make_observation())
    run_control_loop(robot, make_bundle(), RolloutSafetyConfig(execute=False), Event())
    assert robot.actions == []
    assert robot.zero_calls == 0


def test_execute_runs_for_bounded_duration_and_zeroes_before_and_after():
    robot = FakeRobot(make_observation())
    clock = FakeClock()
    config = RolloutSafetyConfig(execute=True, duration_s=0.1)
    run_control_loop(robot, make_bundle(), config, Event(), clock=clock, sleep=clock.sleep)
    assert len(robot.actions) == 3
    assert robot.zero_calls == 2
    assert all("gripper_cmd_bin" not in action for action in robot.actions)


def test_execute_zeroes_when_inference_raises():
    class RaisingPolicy:
        def predict_action_chunk(self, batch):
            raise RuntimeError("inference failed")

    robot = FakeRobot(make_observation())
    bundle = PolicyBundle(RaisingPolicy(), IdentityProcessor(), IdentityProcessor())
    with pytest.raises(RuntimeError, match="inference failed"):
        run_control_loop(robot, bundle, RolloutSafetyConfig(execute=True), Event())
    assert robot.zero_calls == 2
    assert robot.actions == []


def test_build_robot_locks_approved_delta_control_and_injects_raw_zmq_camera():
    robot = build_robot(
        control_host=DEFAULT_CONTROL_HOST,
        image_zmq="tcp://127.0.0.1:5557",
        safety=RolloutSafetyConfig(),
    )
    assert robot.config.action_mode == "delta_ee_pose"
    assert robot.config.cartesian_action_units == "delta"
    assert robot.config.control_hz == 30
    assert robot.config.max_linear_velocity == 0.01
    assert robot.config.max_angular_velocity == 0.08
    assert robot.config.camera_shapes == {"l515": (540, 960, 3)}
    assert robot.cameras["l515"].endpoint == "tcp://127.0.0.1:5557"
```

- [ ] **Step 2: Run the control tests and verify the APIs are missing**

Run: `pytest -q hardware_test/franka/test_act_rollout.py -k 'dry_run or execute or build_robot'`

Expected: import errors for `build_robot` and `run_control_loop`.

- [ ] **Step 3: Implement robot construction and the timed loop**

```python
import logging
import time
from collections.abc import Callable
from threading import Event

from hardware_test.cameras.ros2_image_bridge import (
    DEFAULT_IMAGE_ZMQ_ENDPOINT,
    ZmqRgbImageClient,
)
from hardware_test.franka.franka_robot import FrankaRobot, FrankaRobotConfig

logger = logging.getLogger(__name__)


def build_robot(
    *,
    control_host: str,
    image_zmq: str,
    safety: RolloutSafetyConfig,
) -> FrankaRobot:
    camera = ZmqRgbImageClient(image_zmq, max_age_ms=250, startup_timeout_ms=2000)
    config = FrankaRobotConfig(
        action_mode="delta_ee_pose",
        cartesian_action_units="delta",
        control_hz=float(safety.fps),
        max_linear_velocity=safety.max_linear_velocity,
        max_angular_velocity=safety.max_angular_velocity,
        command_duration_ms=100,
        control_host=control_host,
        velocity_transport="zmq",
        state_cache_enabled=True,
        state_poll_hz=30.0,
        state_timeout_s=0.2,
        max_state_age_s=0.25,
        use_gripper=True,
        cameras={},
        camera_shapes={CAMERA_KEY: EXPECTED_IMAGE_SHAPE},
        camera_read_mode="latest",
        max_camera_age_s=0.25,
    )
    return FrankaRobot(config, cameras={CAMERA_KEY: camera})


def run_control_loop(
    robot: Any,
    bundle: PolicyBundle,
    safety: RolloutSafetyConfig,
    stop_event: Event,
    *,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    if not safety.execute:
        warmup_action = predict_first_robot_action(
            bundle,
            robot.get_observation(),
            action_scale=safety.action_scale,
        )
        logger.info("dry-run action (gripper suppressed): %s", warmup_action)
        return

    control_interval = 1.0 / safety.fps
    robot.send_zero_cartesian_velocity()
    try:
        warmup_action = predict_first_robot_action(
            bundle,
            robot.get_observation(),
            action_scale=safety.action_scale,
        )
        logger.info("warm-up action (gripper suppressed): %s", warmup_action)
        started = clock()
        while not stop_event.is_set() and clock() - started < safety.duration_s:
            loop_started = clock()
            action = predict_first_robot_action(
                bundle,
                robot.get_observation(),
                action_scale=safety.action_scale,
            )
            robot.send_action(action)
            sleep(max(0.0, control_interval - (clock() - loop_started)))
    finally:
        robot.send_zero_cartesian_velocity()
```

- [ ] **Step 4: Run all focused tests**

Run: `pytest -q hardware_test/franka/test_act_rollout.py`

Expected: all tests pass and no hardware modules attempt a connection.

- [ ] **Step 5: Commit the safety loop**

```bash
git add hardware_test/franka/run_act_rollout.py hardware_test/franka/test_act_rollout.py
git commit --only hardware_test/franka/run_act_rollout.py hardware_test/franka/test_act_rollout.py \
  -m "Stop Franka motion on every ACT smoke-test exit path" \
  -m $'Constraint: Ctrl-C, stale data, inference failure, and normal completion must all issue zero Cartesian velocity\nDirective: Do not add automatic home or gripper actuation to this loop\nTested: Deterministic dry-run, bounded-loop, exception, and robot-config unit tests\nConfidence: high\nScope-risk: narrow'
```

### Task 4: CLI, Signals, and Teardown Ownership

**Files:**
- Modify: `hardware_test/franka/run_act_rollout.py`
- Modify: `hardware_test/franka/test_act_rollout.py`

- [ ] **Step 1: Add failing parser and main-lifecycle tests**

```python
import hardware_test.franka.run_act_rollout as run_act_rollout


def test_parser_defaults_to_non_executing_approved_profile():
    args = run_act_rollout.build_arg_parser().parse_args(["--policy-path", "/tmp/policy"])
    assert args.execute is False
    assert args.fps == 30
    assert args.duration_s == 5.0
    assert args.action_scale == 0.25
    assert args.max_linear_velocity == 0.01
    assert args.max_angular_velocity == 0.08


def test_main_connects_runs_and_disconnects_with_final_zero(monkeypatch):
    robot = FakeRobot(make_observation())
    robot.connected = False
    robot.disconnected = False
    robot.connect = lambda: setattr(robot, "connected", True)
    robot.disconnect = lambda: setattr(robot, "disconnected", True)
    monkeypatch.setattr(run_act_rollout, "load_policy_bundle", lambda *a, **k: make_bundle())
    monkeypatch.setattr(run_act_rollout, "build_robot", lambda **k: robot)
    monkeypatch.setattr(run_act_rollout, "install_signal_handlers", lambda stop_event: None)
    exit_code = run_act_rollout.main(["--policy-path", "/tmp/policy"])
    assert exit_code == 0
    assert robot.connected is True
    assert robot.disconnected is True
    assert robot.actions == []
```

- [ ] **Step 2: Run the CLI tests and verify the entry points are missing**

Run: `pytest -q hardware_test/franka/test_act_rollout.py -k 'parser or main'`

Expected: import or attribute errors for `build_arg_parser` and `main`.

- [ ] **Step 3: Implement the CLI and signal-safe lifecycle**

```python
import argparse
import contextlib
import signal


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded Franka ACT physical smoke test.")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--control-host", default=get_control_host())
    parser.add_argument("--image-zmq", default=DEFAULT_IMAGE_ZMQ_ENDPOINT)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--fps", type=int, default=APPROVED_FPS)
    parser.add_argument("--duration-s", type=float, default=MAX_DURATION_S)
    parser.add_argument("--action-scale", type=float, default=MAX_ACTION_SCALE)
    parser.add_argument("--max-linear-velocity", type=float, default=MAX_LINEAR_VELOCITY)
    parser.add_argument("--max-angular-velocity", type=float, default=MAX_ANGULAR_VELOCITY)
    parser.add_argument("--execute", action="store_true")
    return parser


def install_signal_handlers(stop_event: Event) -> None:
    def request_stop(signum, frame):
        del signum, frame
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    safety = RolloutSafetyConfig(
        execute=args.execute,
        fps=args.fps,
        duration_s=args.duration_s,
        action_scale=args.action_scale,
        max_linear_velocity=args.max_linear_velocity,
        max_angular_velocity=args.max_angular_velocity,
    )
    bundle = load_policy_bundle(args.policy_path, device=args.device)
    robot = build_robot(control_host=args.control_host, image_zmq=args.image_zmq, safety=safety)
    stop_event = Event()
    install_signal_handlers(stop_event)
    connected = False
    try:
        robot.connect()
        connected = True
        run_control_loop(robot, bundle, safety, stop_event)
    finally:
        if connected:
            with contextlib.suppress(Exception):
                robot.send_zero_cartesian_velocity()
            robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the focused test module and CLI help**

Run:

```bash
source /home/yanrihong/rs_modes/env_lerobot_sdk.sh
pytest -q hardware_test/franka/test_act_rollout.py
PYTHONPATH=src python hardware_test/franka/run_act_rollout.py --help
```

Expected: tests pass; help lists `--execute` and all approved safety options without loading CUDA or connecting hardware.

- [ ] **Step 5: Commit the CLI lifecycle**

```bash
git add hardware_test/franka/run_act_rollout.py hardware_test/franka/test_act_rollout.py
git commit --only hardware_test/franka/run_act_rollout.py hardware_test/franka/test_act_rollout.py \
  -m "Require explicit execution before ACT can move the Franka" \
  -m $'Constraint: Default invocation may infer but must not send a policy action\nDirective: Keep --execute opt-in and retain final zero before disconnect\nTested: Parser defaults, lifecycle fakes, focused unit suite, and CLI help\nConfidence: high\nScope-risk: narrow'
```

### Task 5: Real-Checkpoint Offline Verification and Final Audit

**Files:**
- Verify: `hardware_test/franka/run_act_rollout.py`
- Verify: `hardware_test/franka/test_act_rollout.py`
- Verify: `docs/superpowers/specs/2026-07-10-franka-act-rollout-design.md`

- [ ] **Step 1: Format and statically check only the new Python files**

Run:

```bash
source /home/yanrihong/rs_modes/env_lerobot_sdk.sh
ruff format hardware_test/franka/run_act_rollout.py hardware_test/franka/test_act_rollout.py
ruff check hardware_test/franka/run_act_rollout.py hardware_test/franka/test_act_rollout.py
```

Expected: both commands exit zero. If formatting changes files, commit only those two paths with a Lore-format `Tested:` trailer.

- [ ] **Step 2: Run focused and existing Franka regression tests**

Run:

```bash
source /home/yanrihong/rs_modes/env_lerobot_sdk.sh
pytest -q hardware_test/franka/test_act_rollout.py
pytest -q hardware_test/franka/test_franka_adapters.py hardware_test/franka/test_franka_robot_ui_support.py
```

Expected: all selected tests pass; no real camera, robot, ROS2 node, or control endpoint is accessed.

- [ ] **Step 3: Run a real-checkpoint offline inference through the new helpers**

Run this with CUDA access but no hardware access:

```bash
source /home/yanrihong/rs_modes/env_lerobot_sdk.sh
export HF_DATASETS_CACHE=/tmp/lerobot_hf_datasets
python - <<'PY'
from lerobot.configs import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import ACTConfig

from hardware_test.franka.run_act_rollout import load_policy_bundle, predict_first_robot_action

checkpoint = "outputs/train/act_press_button_29ep_20260710/checkpoints/last/pretrained_model"
root = "outputs/hardware_test/press_button_train"
repo_id = "local/press_button_train"
config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
metadata = LeRobotDatasetMetadata(repo_id, root=root)
dataset = LeRobotDataset(
    repo_id,
    root=root,
    delta_timestamps=resolve_delta_timestamps(config, metadata),
    return_uint8=True,
    video_backend="torchcodec",
)
sample = dataset[len(dataset) // 2]
raw_observation = {
    **{
        name: float(sample["observation.state"][index])
        for index, name in enumerate(
            [
                "joint_1.pos", "joint_2.pos", "joint_3.pos", "joint_4.pos",
                "joint_5.pos", "joint_6.pos", "joint_7.pos", "gripper_width_norm",
            ]
        )
    },
    "l515": sample["observation.images.l515"].permute(1, 2, 0).cpu().numpy(),
}
if raw_observation["l515"].dtype != "uint8":
    raw_observation["l515"] = (raw_observation["l515"] * 255).round().clip(0, 255).astype("uint8")
bundle = load_policy_bundle(checkpoint, device="cuda")
action = predict_first_robot_action(bundle, raw_observation, action_scale=0.25)
assert list(action) == [
    "delta_ee_pose.x", "delta_ee_pose.y", "delta_ee_pose.z",
    "delta_ee_pose.rx", "delta_ee_pose.ry", "delta_ee_pose.rz",
]
assert all(value == value for value in action.values())
print(action)
PY
```

Expected: one finite six-key action dictionary; no `gripper_cmd_bin`; no robot or live camera imports are exercised beyond class definitions.

- [ ] **Step 4: Audit the final diff and process state**

Run:

```bash
git diff HEAD -- hardware_test/franka/run_act_rollout.py hardware_test/franka/test_act_rollout.py
git status --short
pgrep -af "run_act_rollout.py|lerobot-rollout|policy_server|robot_client"
```

Expected: only intentional runner/test changes are attributable to this work; pre-existing dirty files remain untouched; no rollout process is running.

- [ ] **Step 5: Hand off the two-stage operator commands without executing them**

Provide:

1. ROS2 shell command for `hardware_test/cameras/ros2_image_bridge.py` on `/l515/color/image_raw` and `tcp://127.0.0.1:5557`.
2. LeRobot shell dry-run command without `--execute`.
3. LeRobot shell five-second physical command with `--execute` only after dry-run passes.

State explicitly that the agent did not execute the live dry-run or physical command, that the gripper is suppressed, and that the operator must own workspace clearance, initial pose, and emergency stop.
