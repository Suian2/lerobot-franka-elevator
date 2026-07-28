# Franka ACT Rollout Action Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute ACT checkpoint actions in learned temporal order while retaining bounded Franka motion and visible operator telemetry.

**Architecture:** Replace direct `predict_action_chunk()[0, 0]` use with one `select_action()` call per control tick so `ACTPolicy` owns and consumes its action queue. Reset policy and processor inference state at rollout start, retain the first selected action for the first send, and preserve zero-velocity bracketing. Configure logging with `force=True` and emit one end-of-rollout summary.

**Tech Stack:** Python 3.12, PyTorch, LeRobot ACT policy API, pytest, Ruff.

---

### Task 1: Lock action queue and updated safety behavior

**Files:**
- Modify: `hardware_test/franka/test_act_rollout.py`
- Test: `hardware_test/franka/test_act_rollout.py`

- [ ] **Step 1: Update the safety-bound expectations**

Change the limit regression test to reject values above the revised specification:

```python
with pytest.raises(ValueError, match="duration_s"):
    RolloutSafetyConfig(duration_s=20.01)
with pytest.raises(ValueError, match="max_linear_velocity"):
    RolloutSafetyConfig(max_linear_velocity=0.051)
with pytest.raises(ValueError, match="max_angular_velocity"):
    RolloutSafetyConfig(max_angular_velocity=0.201)
```

Update parser and robot assertions to expect `20.0`, `0.05`, and `0.20`.

- [ ] **Step 2: Replace the chunk-only fake with a queue-aware fake**

Use a fake policy whose `select_action()` returns successive elements and whose
`reset()` rewinds the queue:

```python
class FakePolicy:
    def __init__(self, output):
        self.output = output
        self.index = 0
        self.calls = 0
        self.reset_calls = 0

    def reset(self):
        self.index = 0
        self.reset_calls += 1

    def select_action(self, batch):
        self.calls += 1
        action = self.output[:, self.index]
        self.index += 1
        return action
```

Give the fake processors a `reset()` counter so the test proves all inference
state is reset once per rollout.

- [ ] **Step 3: Add a failing sequential-action regression test**

Construct a chunk whose first three pose actions are `0.004`, `0.008`, and
`0.012`, execute for 0.1 seconds with the fake clock, and assert the robot sees
the scaled values in that order:

```python
assert [action["delta_ee_pose.x"] for action in robot.actions] == pytest.approx(
    [0.001, 0.002, 0.003]
)
assert bundle.policy.calls == 3
assert bundle.policy.reset_calls == 1
```

Update the stop-during-selection test to require exactly the already-selected
first action to be sent, while the second action is suppressed after the stop
event is raised.

- [ ] **Step 4: Add a failing visible-logging regression test**

Monkeypatch `logging.basicConfig`, run `main()` with fake hardware, and assert:

```python
assert logging_kwargs["level"] == logging.INFO
assert logging_kwargs["force"] is True
```

Use `caplog` around the bounded fake-clock loop and assert the summary contains
`sent_frames=3` and `achieved_hz=30.0`.

- [ ] **Step 5: Run the focused tests and verify RED**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yanrihong/miniconda3/envs/lerobot/bin/python \
  -m pytest hardware_test/franka/test_act_rollout.py -q
```

Expected: failures because production still exposes `predict_first_robot_action`,
repeats chunk element zero, retains the old limits, and omits `force=True`.

### Task 2: Implement standard ACT queue execution

**Files:**
- Modify: `hardware_test/franka/run_act_rollout.py`
- Test: `hardware_test/franka/test_act_rollout.py`

- [ ] **Step 1: Apply the revised numeric limits**

Set:

```python
MAX_DURATION_S = 20.0
MAX_LINEAR_VELOCITY = 0.05
MAX_ANGULAR_VELOCITY = 0.20
```

Keep `APPROVED_FPS = 30`, `MAX_ACTION_SCALE = 0.25`, execute opt-in, and gripper
suppression unchanged.

- [ ] **Step 2: Add inference reset and single-action selection helpers**

Replace `predict_first_robot_action()` with:

```python
def reset_inference_state(bundle: PolicyBundle) -> None:
    for component in (bundle.policy, bundle.preprocessor, bundle.postprocessor):
        reset = getattr(component, "reset", None)
        if callable(reset):
            reset()


def select_robot_action(bundle, raw_observation, *, action_scale):
    observation = build_policy_observation(raw_observation)
    processed = bundle.preprocessor(observation)
    with torch.inference_mode():
        selected = bundle.policy.select_action(processed)
        selected = bundle.postprocessor(selected)
    action = torch.as_tensor(selected).detach().cpu()
    if action.shape != (1, EXPECTED_ACTION_DIM):
        raise ValueError(
            f"postprocessed selected action has shape {tuple(action.shape)}, "
            f"expected (1, {EXPECTED_ACTION_DIM})"
        )
    return policy_action_to_robot_action(action[0], action_scale=action_scale)
```

- [ ] **Step 3: Consume the queue in the bounded control loop**

Reset inference state before dry-run or execute. For execute mode, select the
first action before starting the duration clock, log it, then send it on the
first iteration. Select the next queued action only after the preceding action
has been sent. Preserve stop checks and zero velocity in `finally`.

Track `sent_frames` and `started`; in `finally` log:

```python
logger.info(
    "rollout complete sent_frames=%d elapsed_s=%.3f achieved_hz=%.1f stop_requested=%s",
    sent_frames,
    elapsed_s,
    sent_frames / elapsed_s if elapsed_s > 0.0 else 0.0,
    stop_event.is_set(),
)
```

- [ ] **Step 4: Force operator-visible INFO logging**

Configure:

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
```

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

### Task 3: Verify the physical-rollout boundary without moving hardware

**Files:**
- Modify: `docs/superpowers/specs/2026-07-11-franka-act-rollout-action-queue-design.md`
- Verify: `hardware_test/franka/run_act_rollout.py`
- Verify: `hardware_test/franka/test_act_rollout.py`

- [ ] **Step 1: Run static checks**

```bash
.venv/bin/ruff check hardware_test/franka/run_act_rollout.py hardware_test/franka/test_act_rollout.py
.venv/bin/ruff format --check hardware_test/franka/run_act_rollout.py hardware_test/franka/test_act_rollout.py
/home/yanrihong/miniconda3/envs/lerobot/bin/python -m py_compile hardware_test/franka/run_act_rollout.py
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 2: Run rollout-related tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/yanrihong/miniconda3/envs/lerobot/bin/python \
  -m pytest hardware_test/franka/test_act_rollout.py \
  hardware_test/franka/test_franka_robot_ui_support.py -q
```

Expected: all tests pass with no hardware access.

- [ ] **Step 3: Run a non-executing real-checkpoint dry-run**

With the existing bridge and control server available, invoke the script without
`--execute`, using the real checkpoint and current conservative command values
(`duration-s=5`, linear `0.01`, angular `0.08`). Expected: visible `DRY_RUN` and
selected-action INFO lines, no policy action sent, and a clean exit.

- [ ] **Step 4: Commit only the scoped files**

Commit `run_act_rollout.py`, `test_act_rollout.py`, and the reviewed design update
without including unrelated dirty-worktree changes. Use a Lore message recording
the discarded-first-action root cause, the `select_action()` decision, tests, and
the absence of a physical execute run.
