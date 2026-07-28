# Franka ACT Multi-Floor Conditioning Design

## Goal

Train one LeRobot ACT checkpoint on elevator-button demonstrations for floors 1,
4, and 5, while selecting the desired floor at inference with an independent
five-dimensional `observation.environment_state` one-hot input.

## Existing capability

The local ACT implementation already supports this input and must not be changed:

- `ACTConfig` recognizes `observation.environment_state` as an optional ACT input.
- `dataset_to_policy_features` classifies that exact key as `FeatureType.ENV`.
- `ACT` projects the environment vector through
  `encoder_env_state_input_proj` and inserts it as its own transformer token.
- `AddBatchDimensionObservationStep` converts a single `(5,)` vector into the
  `(1, 5)` inference batch expected by ACT.

No file under `src/lerobot/policies/act/` is in implementation scope.

## Shared floor contract

`hardware_test/franka/floor_condition.py` owns the only floor encoder. It accepts
integer floors 1 through 5 and returns a fresh, finite `numpy.float32` vector of
shape `(5,)`, with index `floor - 1` set to one. It also exports the canonical
dataset feature schema:

```python
{
    "dtype": "float32",
    "shape": (5,),
    "names": None,
}
```

Recording, migration, validation, smoke testing, and rollout import this shared
contract. The condition is never appended to `observation.state`.

## Floor-1 migration

`migrate_add_target_floor.py` loads the source through `LeRobotDataset`, refuses
an existing destination, and uses the official `add_features` API to create a
new dataset containing a floor-1 condition on every frame. It then calls the
official `recompute_stats` API so normalization statistics include the new
feature. This path preserves source files and copies videos without decoding or
re-encoding them.

Post-migration verification compares:

- total episode and frame counts;
- episode lengths;
- `observation.state`, `action`, timestamps, and index columns exactly;
- all source video files byte-for-byte against the destination copies;
- every new floor vector against the shared encoder.

The local source currently available is
`outputs/hardware_test/press_button_train_29ep_backup_20260713` with 29 episodes
and 24,281 frames. The requested unsuffixed backup path does not exist, so the
suffixed backup is the migration source and remains untouched.

## Conditioned recording

`run_record_lerobot.py` remains the conditioned user entry point and requires
`--target-floor 1..5`. The shared `run_record.py` implementation accepts the
same option without making it mandatory for legacy entry points.

When a target is present:

1. Dataset creation adds the canonical environment-state feature once.
2. At the beginning of every episode, the loop calls the shared encoder once.
3. Every frame receives a copy of that episode vector under the canonical key.
4. Hardware collection follows the proven command-line workflow: one 30-second
   episode per invocation and per dataset root, followed by the official LeRobot
   aggregation API.

The state, action, camera, timestamp, finalize, and disconnect paths stay
unchanged. Conditioned Cartesian recording treats an explicit
`FrankaControlError`, requests transport error, or exhausted stale-state wait
as a recoverable physical Fault. It skips the ambiguous failed-action frame,
stops sending motion for the rest of that episode, and saves every subsequent
valid stationary frame through the fixed deadline. Joint-mode recording,
invalid actions, and dataset/video/Parquet write failures remain fail-fast so
corrupted demonstrations are never published.

## Validation, merge, and smoke test

`validate_multifloor_dataset.py` accepts repeatable `(floor, repo_id, root)`
dataset specifications. Validation is fail-closed and checks:

- identical feature dictionaries across sources;
- declared and loaded environment-state dtype/shape;
- finite, exact one-hot vectors and a constant condition within each episode;
- the requested floor for each source dataset;
- finite state/action arrays and consistent state/action shapes across sources;
- contiguous, non-empty episodes with metadata lengths matching frame data;
- image/video presence, with optional full frame decoding enabled by default.

With `--merge`, exactly floors 1, 4, and 5 are required. Only after all source
checks pass does the script call the official `merge_datasets` wrapper around
`aggregate_datasets`. It refuses an existing output root, reloads the merged
dataset, verifies all floor counts, and reports totals.

The merge accepts more than one source root per floor. `--expected-floors 4`
or `--expected-floors 5` provides the same strict, transactional official merge
for the one-episode command-line shards before the final default 1/4/5 merge.

With `--smoke-test`, the merged dataset is loaded through a real `DataLoader`
using ACT delta timestamps. A lightweight ACT configuration is inferred from
the real schema, the normal training preprocessor is applied, and one real ACT
forward pass must produce a finite loss. The check asserts environment shape
`(B, 5)` and verifies state/action dimensions before and after preprocessing.

## Conditioned rollout

`run_act_rollout.py` requires `--target-floor` with choices limited to 1, 4, and
5. Before any robot connection, it prints the target floor, exact one-hot vector,
and resolved checkpoint path. Floors 2 and 3 are rejected by argument parsing.

The encoded `(5,)` tensor is inserted into the observation dictionary before the
saved policy preprocessor. The processed tensor is checked to be finite
`torch.float32` with shape `(1, 5)`. Checkpoint validation requires the matching
environment feature, so an old unconditioned checkpoint cannot silently ignore
the floor.

Every `run_control_loop` invocation represents one episode and begins with
`reset_inference_state`, which calls `policy.reset()` and resets both processors.
This clears ACT's cached action chunk before a new target episode.

## Failure handling

- Invalid floors, schemas, shapes, dtypes, non-finite values, missing images,
  empty episodes, source mismatch, and existing output roots raise before merge
  or robot connection.
- Migration never removes or modifies the source root.
- Merge never starts when any source validation fails.
- Physical rollout retains the existing dry-run default and zero-velocity
  teardown.

## Test strategy

Tests follow red-green-refactor and cover:

- exact floor encoding and invalid floors;
- conditioned recorder schema and per-frame propagation;
- migration preservation on a real tiny LeRobot dataset;
- schema/one-hot/episode validation and official merge;
- a real small ACT DataLoader/forward finite-loss smoke test;
- rollout parser restrictions, checkpoint contract, processor input shape, and
  reset behavior;
- preservation of the user's existing rollout action scale of `0.8` while
  aligning stale safety tests/messages with the current committed profile.
