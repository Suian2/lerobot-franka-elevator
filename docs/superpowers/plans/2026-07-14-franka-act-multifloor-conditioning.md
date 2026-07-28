# Franka ACT Multi-Floor Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a canonical five-dimensional floor condition to Franka recording, floor-1 migration, strict dataset merge validation, ACT training smoke testing, and physical rollout without changing ACT core.

**Architecture:** A small shared floor module owns encoding and schema. Franka recording and rollout inject the feature at their existing adapter boundaries. Migration and aggregation use public LeRobot dataset tools, while a standalone validator proves schema/data integrity and executes one real ACT training forward pass.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Hugging Face Datasets, LeRobotDataset v3, pytest, ruff

---

## File map

- Create `hardware_test/franka/floor_condition.py`: canonical encoder, schema, and constants.
- Modify `hardware_test/franka/record_lerobot_dataset.py`: optional environment feature and per-frame insertion.
- Modify `hardware_test/franka/run_record.py`: optional target-floor propagation and per-episode encoding.
- Modify `hardware_test/franka/run_record_lerobot.py`: conditioned entry point requiring a target floor.
- Create `hardware_test/franka/migrate_add_target_floor.py`: official feature-add migration plus exact preservation checks.
- Create `hardware_test/franka/validate_multifloor_dataset.py`: source validation, official merge, aggregate validation, and ACT smoke test.
- Modify `hardware_test/franka/run_act_rollout.py`: conditioned checkpoint contract and processor input.
- Create `hardware_test/franka/test_floor_condition.py`: shared encoder tests.
- Modify `hardware_test/franka/test_franka_adapters.py`: recording schema/frame tests.
- Modify `hardware_test/franka/test_act_rollout.py`: conditioned rollout and reset tests.
- Create `hardware_test/franka/test_multifloor_dataset.py`: migration, validation, merge, and smoke tests.

### Task 1: Canonical floor contract

**Files:**
- Create: `hardware_test/franka/floor_condition.py`
- Create: `hardware_test/franka/test_floor_condition.py`

- [ ] **Step 1: Write failing encoder tests**

```python
import numpy as np
import pytest

from hardware_test.franka.floor_condition import encode_target_floor


@pytest.mark.parametrize("floor", range(1, 6))
def test_encode_target_floor_returns_canonical_one_hot(floor):
    result = encode_target_floor(floor)
    expected = np.zeros(5, dtype=np.float32)
    expected[floor - 1] = 1.0
    assert result.dtype == np.float32
    assert result.shape == (5,)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("floor", [0, 6, True, 1.0, "1"])
def test_encode_target_floor_rejects_non_integer_or_out_of_range_values(floor):
    with pytest.raises((TypeError, ValueError)):
        encode_target_floor(floor)
```

- [ ] **Step 2: Run RED**

Run: `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q hardware_test/franka/test_floor_condition.py`

Expected: collection fails because `floor_condition.py` does not exist.

- [ ] **Step 3: Implement the shared contract**

```python
from __future__ import annotations

from typing import Final

import numpy as np

from lerobot.utils.constants import OBS_ENV_STATE

FLOOR_CONDITION_KEY: Final = OBS_ENV_STATE
NUM_ELEVATOR_FLOORS: Final = 5
TRAINED_ROLLOUT_FLOORS: Final = (1, 4, 5)
FLOOR_CONDITION_FEATURE: Final = {
    "dtype": "float32",
    "shape": (NUM_ELEVATOR_FLOORS,),
    "names": None,
}


def encode_target_floor(floor: int) -> np.ndarray:
    if isinstance(floor, bool) or not isinstance(floor, int):
        raise TypeError("target floor must be an integer")
    if not 1 <= floor <= NUM_ELEVATOR_FLOORS:
        raise ValueError("target floor must be between 1 and 5")
    condition = np.zeros(NUM_ELEVATOR_FLOORS, dtype=np.float32)
    condition[floor - 1] = 1.0
    return condition
```

- [ ] **Step 4: Run GREEN and lint**

Run: `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q hardware_test/franka/test_floor_condition.py`

Run: `uv run ruff check hardware_test/franka/floor_condition.py hardware_test/franka/test_floor_condition.py`

Expected: all tests and lint pass.

- [ ] **Step 5: Commit with Lore trailers**

```text
Make every floor consumer share one mapping

Constraint: Elevator identity is a five-dimensional float32 one-hot vector
Confidence: high
Scope-risk: narrow
Tested: Floor encoder unit tests for floors 1-5 and invalid inputs
```

### Task 2: Conditioned Franka recording

**Files:**
- Modify: `hardware_test/franka/record_lerobot_dataset.py`
- Modify: `hardware_test/franka/run_record.py`
- Modify: `hardware_test/franka/run_record_lerobot.py`
- Modify: `hardware_test/franka/test_franka_adapters.py`

- [ ] **Step 1: Write failing schema and frame propagation tests**

Add tests that call `build_lerobot_features(..., include_environment_state=True)` and assert the canonical feature exists while `observation.state` remains `(8,)`. Add a frame test that passes `environment_state=encode_target_floor(4)` to `make_lerobot_frame` and asserts the new key is float32 `(5,)`. Add a recorder-loop test proving every captured frame receives floor 5 without changing state or action.

```python
def test_conditioned_features_keep_floor_separate_from_robot_state(robot, teleop):
    features = build_lerobot_features(
        robot, teleop, use_videos=True, include_environment_state=True
    )
    assert features[OBS_ENV_STATE] == FLOOR_CONDITION_FEATURE
    assert features["observation.state"]["shape"] == (8,)


def test_conditioned_frame_contains_copied_floor_vector(robot, teleop):
    features = build_lerobot_features(
        robot, teleop, use_videos=True, include_environment_state=True
    )
    floor = encode_target_floor(4)
    frame = make_lerobot_frame(
        features, robot.get_observation(), teleop.get_action(), task="press", environment_state=floor
    )
    np.testing.assert_array_equal(frame[OBS_ENV_STATE], floor)
    assert frame[OBS_ENV_STATE] is not floor
```

- [ ] **Step 2: Run RED**

Run: `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q hardware_test/franka/test_franka_adapters.py -k 'conditioned or environment_state or target_floor'`

Expected: failures report unsupported arguments or missing environment-state fields.

- [ ] **Step 3: Implement optional adapter support**

Extend `build_lerobot_features` with `include_environment_state=False`, copying `FLOOR_CONDITION_FEATURE` into the returned feature dict when enabled. Extend `make_lerobot_frame` and `record_lerobot_episode` with `environment_state: np.ndarray | None`; filter the environment key out of `build_dataset_frame`, validate it through the shared contract shape/dtype, and insert `environment_state.copy()` into every frame.

Extend `create_lerobot_dataset` with `include_environment_state=False` and use it during feature creation. Existing callers retain legacy behavior.

- [ ] **Step 4: Implement CLI propagation**

Add `--target-floor` to `run_record.build_arg_parser` with `choices=range(1, 6)` and conditional `required`. Let `run_record.main(..., require_target_floor=False)` preserve legacy callers. The `run_record_lerobot.main` wrapper calls it with `require_target_floor=True`.

Create the dataset with `include_environment_state=args.target_floor is not None`. Inside the episode loop, call `encode_target_floor(args.target_floor)` exactly once for that episode and pass it into `record_lerobot_episode`. Include floor and vector in `describe_config`.

Hardware follow-up: retain the collection pattern that produced the original
`press_button1...` datasets: invoke the conditioned CLI with
`--num-episodes 1`, use one unique root per 30-second demonstration, recover and
home the arm between invocations, then aggregate with the official LeRobot API.
For conditioned Cartesian recording only, tolerate explicit
`FrankaControlError`, requests transport errors, and exhausted stale-state waits
through the episode deadline. Skip the ambiguous failed-action frame, disable
further motion after the first such fault, label later valid stationary frames
with zero Cartesian deltas, and keep joint-mode, invalid-action, and
dataset-write errors fail-fast.

- [ ] **Step 5: Run GREEN, compatibility tests, and dry config**

Run: `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q hardware_test/franka/test_franka_adapters.py hardware_test/franka/test_franka_control_host_defaults.py`

Run: `uv run python hardware_test/franka/run_record_lerobot.py --target-floor 4 --repo-id local/dry --root /tmp/dry --dry-run-config`

Expected: tests pass and dry-run prints floor 4 without hardware access.

- [ ] **Step 6: Commit with Lore trailers**

```text
Keep target identity out of Franka proprioception

Constraint: Legacy recording entry points must remain usable without conditioning
Confidence: high
Scope-risk: moderate
Tested: Adapter tests, parser compatibility, and hardware-free dry config
```

### Task 3: Conditioned ACT rollout

**Files:**
- Modify: `hardware_test/franka/run_act_rollout.py`
- Modify: `hardware_test/franka/test_act_rollout.py`

- [ ] **Step 1: Write failing rollout tests**

Update fixtures so conditioned checkpoints include `observation.environment_state: (5,)`. Add tests that:

```python
def test_build_policy_observation_keeps_floor_separate():
    floor = encode_target_floor(5)
    result = build_policy_observation(make_observation(), floor_condition=floor)
    assert result[OBS_ENV_STATE].shape == (5,)
    assert result[OBS_ENV_STATE].dtype == torch.float32
    assert result["observation.state"].shape == (8,)


def test_parser_only_accepts_trained_floors():
    parser = run_act_rollout.build_arg_parser()
    for floor in (1, 4, 5):
        assert parser.parse_args(["--policy-path", "/tmp/p", "--target-floor", str(floor)]).target_floor == floor
    for floor in (2, 3):
        with pytest.raises(SystemExit):
            parser.parse_args(["--policy-path", "/tmp/p", "--target-floor", str(floor)])
```

Make the fake preprocessor add a batch dimension and assert policy input is `(1, 5)` float32. Assert `run_control_loop` resets the policy before its first conditioned inference.

- [ ] **Step 2: Run RED**

Run: `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q hardware_test/franka/test_act_rollout.py`

Expected: new floor tests fail. Four pre-existing assertions also fail because the test profile still says 20 seconds, 0.25 action scale, and 0.05 linear velocity while current code is 200, 0.8, and 0.10.

- [ ] **Step 3: Implement floor injection and checkpoint validation**

Require `floor_condition` in `build_policy_observation`, `select_robot_action`, and `run_control_loop`. Insert `torch.from_numpy(floor_condition.copy())` before the preprocessor. After preprocessing, fail unless the environment tensor is finite `torch.float32` with shape `(1, 5)`.

Require `observation.environment_state: (5,)` in `validate_policy_features`. Add required `--target-floor` choices `(1, 4, 5)`. Encode and print the floor, vector, and resolved checkpoint before `load_policy_bundle` and before robot construction.

- [ ] **Step 4: Preserve the current safety profile while fixing stale diagnostics**

Keep the user's uncommitted `MAX_ACTION_SCALE = 0.8`. Align stale tests and `RolloutSafetyConfig` error strings with current constants: 200 seconds, 0.8 action scale, 0.10 linear velocity, and 0.20 angular velocity. Do not reduce those values or change motion behavior.

- [ ] **Step 5: Run GREEN and lint**

Run: `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q hardware_test/franka/test_act_rollout.py hardware_test/franka/test_franka_control_host_defaults.py`

Run: `uv run ruff check hardware_test/franka/run_act_rollout.py hardware_test/franka/test_act_rollout.py`

Expected: all rollout tests pass with no hardware access.

- [ ] **Step 6: Commit with Lore trailers**

```text
Reject rollouts that cannot honor the selected floor

Constraint: Only floors represented in training data may execute on hardware
Rejected: Accept an old unconditioned checkpoint | it could silently ignore target-floor
Confidence: high
Scope-risk: moderate
Tested: Rollout tensor contract, parser rejection, reset, and teardown tests
```

### Task 4: Official floor-1 migration

**Files:**
- Create: `hardware_test/franka/migrate_add_target_floor.py`
- Create: `hardware_test/franka/test_multifloor_dataset.py`

- [ ] **Step 1: Write a failing real-dataset migration test**

Create a tiny source with `LeRobotDataset.create`, two episodes, numeric state/action, timestamps, and a small image feature. Call `migrate_dataset(..., target_floor=1)` and assert:

- source and destination roots differ;
- source schema lacks the environment key after migration;
- destination has the canonical key and stats;
- episode/frame counts and lengths match;
- state/action/timestamp arrays are exactly equal;
- every destination condition equals floor 1.

- [ ] **Step 2: Run RED**

Run: `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q hardware_test/franka/test_multifloor_dataset.py -k migration`

Expected: import fails because the migration module does not exist.

- [ ] **Step 3: Implement migration and exact verification**

Implement a CLI with required `--source-repo-id`, `--source-root`, `--output-repo-id`, `--output-root`, and optional `--target-floor` defaulting to 1. Reject equal roots, existing outputs, missing sources, and sources already containing the field.

Load the source with `LeRobotDataset`, call:

```python
destination = add_features(
    source,
    features={
        OBS_ENV_STATE: (
            lambda frame, episode_index, frame_index: encode_target_floor(target_floor),
            dict(FLOOR_CONDITION_FEATURE),
        )
    },
    output_dir=output_root,
    repo_id=output_repo_id,
)
recompute_stats(destination, skip_image_video=True)
```

Then compare preserved arrays, episode lengths, and every relative video path with `filecmp.cmp(..., shallow=False)`. Return a report and exit nonzero on any mismatch.

- [ ] **Step 4: Run GREEN and lint**

Run: `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q hardware_test/franka/test_multifloor_dataset.py -k migration`

Run: `uv run ruff check hardware_test/franka/migrate_add_target_floor.py hardware_test/franka/test_multifloor_dataset.py`

Expected: migration tests pass.

- [ ] **Step 5: Commit with Lore trailers**

```text
Reuse floor-one demonstrations without touching their source

Constraint: Dataset media and timestamps must remain byte/exact-value stable
Rejected: Decode and re-encode videos | unnecessary fidelity and runtime risk
Confidence: high
Scope-risk: moderate
Tested: Real tiny LeRobotDataset migration and exact preservation assertions
```

### Task 5: Validation, official merge, and ACT smoke test

**Files:**
- Create: `hardware_test/franka/validate_multifloor_dataset.py`
- Modify: `hardware_test/franka/test_multifloor_dataset.py`

- [ ] **Step 1: Write failing validator tests**

Create three tiny conditioned datasets for floors 1, 4, and 5. Test valid reports and separate failures for schema mismatch, float64 declaration, wrong shape, invalid one-hot sum, changing floor within one episode, NaN state/action, empty/missing episode metadata, and missing image data.

Test `merge_validated_datasets` creates a new official aggregate with contiguous episode indices and floor counts matching all three inputs.

- [ ] **Step 2: Write a failing real ACT smoke test**

Use the tiny merged dataset and assert `smoke_test_act_training` returns a finite loss with:

```python
assert report.environment_shape == (1, 5)
assert report.state_shape[-1] == 8
assert report.action_shape[-1] == 7
assert math.isfinite(report.loss)
```

- [ ] **Step 3: Run RED**

Run: `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q hardware_test/franka/test_multifloor_dataset.py -k 'validation or merge or smoke'`

Expected: validator imports/functions are missing.

- [ ] **Step 4: Implement strict validation and merge**

Add repeatable CLI `--dataset FLOOR REPO_ID ROOT`, `--merge`, `--output-repo-id`, `--output-root`, `--smoke-test`, and `--skip-image-decode`. Validate every numeric column through the Hugging Face dataset's NumPy format, every episode length/index, every one-hot vector, and all image samples by default.

Before the final merge, require the exact floor set `{1, 4, 5}` and strict schema equality. Permit multiple source roots per floor, and expose `--expected-floors 4` / `5` so the historical one-episode roots can first be aggregated with the same strict transactional path. Call `merge_datasets` only after validation and reject an existing output. Reload and validate every merged output, permitting mixed expected floors while counting each one.

- [ ] **Step 5: Implement the real ACT smoke path**

Construct a small CPU `ACTConfig` with real dataset-inferred input/output features, no pretrained backbone download, chunk size 2, 64 model dimensions, one encoder/decoder/VAE layer, and real dataset stats. Resolve ACT delta timestamps, load one DataLoader batch, apply the normal ACT preprocessor, and call `policy.forward`. Assert the environment tensor `(B, 5)`, unchanged state/action final dimensions, and finite loss.

- [ ] **Step 6: Run GREEN and lint**

Run: `env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q hardware_test/franka/test_multifloor_dataset.py`

Run: `uv run ruff check hardware_test/franka/validate_multifloor_dataset.py hardware_test/franka/test_multifloor_dataset.py`

Expected: validation, merge, and ACT smoke tests pass.

- [ ] **Step 7: Commit with Lore trailers**

```text
Make bad floor data fail before training or robot use

Constraint: Merge must use the official LeRobot aggregation path
Confidence: high
Scope-risk: moderate
Directive: Never bypass full validation for physical rollout candidates
Tested: Negative validator cases, official merge, DataLoader, and ACT finite-loss forward
```

### Task 6: Migrate the available floor-1 dataset

**Files:**
- Source (read-only): `outputs/hardware_test/press_button_train_29ep_backup_20260713`
- Create data output: `outputs/hardware_test/press_floor_1_conditioned`

- [ ] **Step 1: Record source evidence before migration**

Run: `sha256sum outputs/hardware_test/press_button_train_29ep_backup_20260713/meta/info.json outputs/hardware_test/press_button_train_29ep_backup_20260713/meta/stats.json`

Run: `du -sh outputs/hardware_test/press_button_train_29ep_backup_20260713`

Expected: source reports 29 episodes and 24,281 frames; store hashes in the run log.

- [ ] **Step 2: Run the migration**

```bash
uv run python hardware_test/franka/migrate_add_target_floor.py \
  --source-repo-id local/press_button_train \
  --source-root outputs/hardware_test/press_button_train_29ep_backup_20260713 \
  --output-repo-id local/press_floor_1_conditioned \
  --output-root outputs/hardware_test/press_floor_1_conditioned \
  --target-floor 1
```

Expected: 29 episodes and 24,281 frames migrated, preservation checks pass, and all videos compare byte-for-byte.

- [ ] **Step 3: Prove the source did not change**

Re-run the Step 1 `sha256sum` command and compare hashes exactly. Run the validator on the migrated dataset with floor 1.

Expected: hashes unchanged; floor-1 dataset validation passes.

### Task 7: Final verification and handoff

**Files:**
- Verify every file listed in the file map.

- [ ] **Step 1: Run focused tests**

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  hardware_test/franka/test_floor_condition.py \
  hardware_test/franka/test_multifloor_dataset.py \
  hardware_test/franka/test_act_rollout.py \
  hardware_test/franka/test_franka_adapters.py \
  hardware_test/franka/test_franka_control_host_defaults.py
```

- [ ] **Step 2: Run static checks**

```bash
uv run ruff check \
  hardware_test/franka/floor_condition.py \
  hardware_test/franka/migrate_add_target_floor.py \
  hardware_test/franka/validate_multifloor_dataset.py \
  hardware_test/franka/record_lerobot_dataset.py \
  hardware_test/franka/run_record.py \
  hardware_test/franka/run_record_lerobot.py \
  hardware_test/franka/run_act_rollout.py \
  hardware_test/franka/test_floor_condition.py \
  hardware_test/franka/test_multifloor_dataset.py \
  hardware_test/franka/test_act_rollout.py \
  hardware_test/franka/test_franka_adapters.py
```

Run the same file list through `uv run ruff format --check` and `uv run python -m compileall -q hardware_test/franka`.

- [ ] **Step 3: Inspect the final diff and run independent review**

Run: `git diff --check`

Run: `git status --short`

Dispatch a spec-compliance reviewer and then a code-quality reviewer. Fix every critical or important finding and repeat the affected verification.

- [ ] **Step 4: Report commands and unavailable-data boundary**

The final response must list changed files and exact migration, floor-4 recording, floor-5 recording, validation/merge/smoke, training, and floor-1/4/5 rollout commands. State explicitly that the local source contains 29 rather than 30 floor-1 episodes and that an actual ~89-episode merge cannot be produced until the user records floor-4 and floor-5 hardware data.
