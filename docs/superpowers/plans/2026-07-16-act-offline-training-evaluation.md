# ACT Offline Training Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing single-checkpoint ACT evaluator with a small offline command that compares compatible checkpoints, recomputes CVAE loss on fixed expert episodes, labels unavailable historical metrics, and writes static visual output.

**Architecture:** Keep `eval_act_offline.py` unchanged as the deployment/expert metric engine. Add one sibling module that owns CVAE replay, checkpoint metadata extraction, cross-checkpoint orchestration, CSV/JSON/PNG/HTML reporting, and its CLI. Test pure loss/provenance/report helpers with synthetic data and test orchestration through injected fake loaders before running one real-checkpoint smoke test.

**Tech Stack:** Python 3.12, PyTorch, NumPy, Pillow, safetensors, pytest, existing LeRobot processors and ACT policy loader.

---

## File Structure

- Preserve and commit: `hardware_test/evaluation/__init__.py`
- Preserve and commit: `hardware_test/evaluation/eval_act_offline.py`
- Preserve and commit: `hardware_test/evaluation/test_eval_act_offline.py`
- Create: `hardware_test/evaluation/eval_act_checkpoints.py`
  - loss replay dataclasses and aggregation;
  - checkpoint/config/normalization extraction;
  - static plots and HTML report;
  - multi-checkpoint orchestration and CLI.
- Create: `hardware_test/evaluation/test_eval_act_checkpoints.py`
  - pure unit tests, fake-runtime integration tests, and report artifact tests.

The existing evaluator stays separate because its 17 tests already lock the
expert-action and trajectory metric semantics. The new module imports its
public runtime functions and the two internal helpers needed for identical
episode selection and atomic text output; it does not copy their logic.

## Test Environment Note

This shell exposes ROS pytest plugins from `/opt/ros`, but their optional `lark`
dependency is absent from the active Python environment. The failure occurs
during pytest plugin discovery, before repository tests are collected. Use this
prefix for every pytest command in this plan:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache uv run pytest
```

Do not install a dependency or change project configuration for this unrelated
environment issue.

### Task 1: Record the Existing Evaluator as the Regression Baseline

**Files:**
- Preserve: `hardware_test/evaluation/__init__.py`
- Preserve: `hardware_test/evaluation/eval_act_offline.py`
- Preserve: `hardware_test/evaluation/test_eval_act_offline.py`

- [ ] **Step 1: Verify the three files are still untracked and inspect their exact names**

Run:

```bash
git status --short -- hardware_test/evaluation
```

Expected: the directory is untracked and contains only the three baseline
files before new work starts.

- [ ] **Step 2: Run the complete existing evaluator regression test**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache uv run pytest hardware_test/evaluation/test_eval_act_offline.py -q
```

Expected: `17 passed`.

- [ ] **Step 3: Commit the behavior-locked baseline without editing it**

Run:

```bash
git add hardware_test/evaluation/__init__.py hardware_test/evaluation/eval_act_offline.py hardware_test/evaluation/test_eval_act_offline.py
git commit
```

Use a Lore commit whose intent is to preserve the already-tested expert
comparison before adding CVAE reporting. Include `Tested: 17 offline evaluator
tests` and state that no behavior was changed.

### Task 2: Add Exact CVAE Loss Aggregation

**Files:**
- Create: `hardware_test/evaluation/eval_act_checkpoints.py`
- Create: `hardware_test/evaluation/test_eval_act_checkpoints.py`

- [ ] **Step 1: Write failing pure loss tests**

Create the test file with these imports and tests:

```python
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hardware_test.evaluation.eval_act_checkpoints import (
    ReplayBatch,
    aggregate_replay_batches,
)


def test_aggregate_replay_batches_uses_act_denominators_and_formula() -> None:
    result = aggregate_replay_batches(
        seed=7,
        kl_weight=10.0,
        batches=[
            ReplayBatch(
                batch_size=2,
                valid_action_elements=14,
                l1_loss=2.0,
                kld_loss=0.5,
                total_loss=7.0,
            ),
            ReplayBatch(
                batch_size=1,
                valid_action_elements=7,
                l1_loss=5.0,
                kld_loss=2.0,
                total_loss=25.0,
            ),
        ],
    )

    assert result.seed == 7
    assert result.num_batches == 2
    assert result.num_samples == 3
    assert result.valid_action_elements == 21
    assert result.l1_loss == pytest.approx(3.0)
    assert result.kld_loss == pytest.approx(1.0)
    assert result.weighted_kld_loss == pytest.approx(10.0)
    assert result.total_loss == pytest.approx(13.0)


def test_aggregate_replay_batches_rejects_loss_identity_violation() -> None:
    with pytest.raises(ValueError, match="ACT loss identity"):
        aggregate_replay_batches(
            seed=0,
            kl_weight=10.0,
            batches=[ReplayBatch(1, 7, 1.0, 0.5, 9.0)],
        )
```

- [ ] **Step 2: Run the new tests and verify import failure**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache uv run pytest hardware_test/evaluation/test_eval_act_checkpoints.py -q
```

Expected: collection fails because `eval_act_checkpoints` does not exist.

- [ ] **Step 3: Implement the loss dataclasses and aggregation**

Create `eval_act_checkpoints.py` with these definitions:

```python
#!/usr/bin/env python
"""Compare ACT checkpoints and replay their CVAE objective on fixed episodes."""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import re
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from hardware_test.evaluation.eval_act_offline import (
    ACTION_DIM,
    EXPECTED_ACTION_NAMES,
    EXPECTED_STATE_NAMES,
    EvaluationRunResult,
    EvaluationRuntime,
    _atomic_write_text,
    _csv_text,
    _episode_frame_indices,
    _reset_if_supported,
    load_evaluation_runtime,
    run_evaluation,
)


@dataclass(frozen=True)
class ReplayBatch:
    batch_size: int
    valid_action_elements: int
    l1_loss: float
    kld_loss: float
    total_loss: float


@dataclass(frozen=True)
class ReplayLoss:
    seed: int
    num_batches: int
    num_samples: int
    valid_action_elements: int
    l1_loss: float
    kld_loss: float
    weighted_kld_loss: float
    total_loss: float


def aggregate_replay_batches(
    *, seed: int, kl_weight: float, batches: Sequence[ReplayBatch]
) -> ReplayLoss:
    if not batches:
        raise ValueError("CVAE replay produced no batches")
    if not math.isfinite(kl_weight) or kl_weight < 0:
        raise ValueError(f"kl_weight must be finite and non-negative, got {kl_weight}")

    for index, batch in enumerate(batches):
        values = (batch.l1_loss, batch.kld_loss, batch.total_loss)
        if batch.batch_size <= 0 or batch.valid_action_elements <= 0:
            raise ValueError(f"replay batch {index} has invalid counts")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"replay batch {index} contains NaN or Inf")
        expected_total = batch.l1_loss + kl_weight * batch.kld_loss
        if not math.isclose(batch.total_loss, expected_total, rel_tol=1e-5, abs_tol=1e-7):
            raise ValueError(
                "ACT loss identity failed for replay batch "
                f"{index}: total={batch.total_loss}, expected={expected_total}"
            )

    valid_elements = sum(batch.valid_action_elements for batch in batches)
    samples = sum(batch.batch_size for batch in batches)
    l1_loss = sum(batch.l1_loss * batch.valid_action_elements for batch in batches) / valid_elements
    kld_loss = sum(batch.kld_loss * batch.batch_size for batch in batches) / samples
    weighted_kld_loss = kl_weight * kld_loss
    return ReplayLoss(
        seed=int(seed),
        num_batches=len(batches),
        num_samples=samples,
        valid_action_elements=valid_elements,
        l1_loss=l1_loss,
        kld_loss=kld_loss,
        weighted_kld_loss=weighted_kld_loss,
        total_loss=l1_loss + weighted_kld_loss,
    )
```

- [ ] **Step 4: Run the pure tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache uv run pytest hardware_test/evaluation/test_eval_act_checkpoints.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit the loss core**

Run:

```bash
git add hardware_test/evaluation/eval_act_checkpoints.py hardware_test/evaluation/test_eval_act_checkpoints.py
git commit
```

Use a Lore message stating that dataset-level L1 is weighted by valid action
elements, KLD by samples, and total is recomputed from those components.

### Task 3: Replay the ACT Action-Conditioned VAE Path

**Files:**
- Modify: `hardware_test/evaluation/eval_act_checkpoints.py`
- Modify: `hardware_test/evaluation/test_eval_act_checkpoints.py`

- [ ] **Step 1: Add a failing replay integration test**

Append these fakes and test:

```python
class _ReplayDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.episode_ids = [2, 2, 5]
        self.hf_dataset = SimpleNamespace(
            data=SimpleNamespace(
                column=lambda name: SimpleNamespace(
                    to_numpy=lambda: np.asarray(self.episode_ids, dtype=np.int64)
                )
            )
        )

    def __len__(self) -> int:
        return len(self.episode_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "observation.state": torch.zeros(8),
            "observation.images.l515": torch.zeros(3, 2, 2),
            "action": torch.full((2, 7), float(index + 1)),
            "action_is_pad": torch.tensor([False, index == 1]),
        }


class _ReplayPreprocessor:
    def reset(self) -> None:
        return None

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return batch


class _ReplayPolicy:
    def __init__(self) -> None:
        self.training = False
        self.forward_seeds: list[int] = []

    def train(self) -> _ReplayPolicy:
        self.training = True
        return self

    def reset(self) -> None:
        return None

    def __call__(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        assert self.training
        self.forward_seeds.append(torch.initial_seed())
        valid = ~batch["action_is_pad"].unsqueeze(-1)
        l1 = (batch["action"].abs() * valid).sum() / (valid.sum() * 7)
        kld = torch.tensor(0.25)
        total = l1 + 10.0 * kld
        return total, {"l1_loss": float(l1), "kld_loss": float(kld)}


def test_replay_cvae_objective_uses_fixed_episode_subset_and_training_path() -> None:
    from hardware_test.evaluation.eval_act_checkpoints import replay_cvae_objective

    policy = _ReplayPolicy()
    result = replay_cvae_objective(
        policy=policy,
        preprocessor=_ReplayPreprocessor(),
        dataset=_ReplayDataset(),
        episode_ids=[5],
        batch_size=2,
        num_workers=0,
        seed=11,
        kl_weight=10.0,
        device="cpu",
    )

    assert result.seed == 11
    assert result.num_samples == 1
    assert result.valid_action_elements == 14
    assert result.l1_loss == pytest.approx(3.0)
    assert result.kld_loss == pytest.approx(0.25)
    assert result.total_loss == pytest.approx(5.5)
    assert policy.forward_seeds == [11]
```

- [ ] **Step 2: Run the targeted test and verify the missing function failure**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache uv run pytest hardware_test/evaluation/test_eval_act_checkpoints.py::test_replay_cvae_objective_uses_fixed_episode_subset_and_training_path -q
```

Expected: FAIL because `replay_cvae_objective` is not defined.

- [ ] **Step 3: Implement deterministic CVAE replay**

Add this function:

```python
def replay_cvae_objective(
    *,
    policy: object,
    preprocessor: object,
    dataset: object,
    episode_ids: Sequence[int],
    batch_size: int,
    num_workers: int,
    seed: int,
    kl_weight: float,
    device: str,
) -> ReplayLoss:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    train_method = getattr(policy, "train", None)
    if not callable(train_method) or not callable(policy) or not callable(preprocessor):
        raise TypeError("CVAE replay requires callable policy/preprocessor and policy.train()")

    indices_by_episode = _episode_frame_indices(dataset, episode_ids)
    flat_indices = np.concatenate([indices_by_episode[int(item)] for item in episode_ids])
    loader = DataLoader(
        Subset(dataset, flat_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    torch_device = torch.device(device)
    fork_devices: list[int] = []
    if torch_device.type == "cuda":
        fork_devices = [torch_device.index if torch_device.index is not None else torch.cuda.current_device()]

    batches: list[ReplayBatch] = []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        if torch_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        policy.train()
        _reset_if_supported(policy)
        _reset_if_supported(preprocessor)
        with torch.no_grad():
            for raw_batch in loader:
                processed_batch = preprocessor(raw_batch)
                if not isinstance(processed_batch, dict):
                    raise TypeError("preprocessor must return a batch dictionary")
                if "action" not in processed_batch or "action_is_pad" not in processed_batch:
                    raise ValueError("CVAE replay batch is missing action or action_is_pad")
                action = processed_batch["action"]
                padding = processed_batch["action_is_pad"]
                if not isinstance(action, torch.Tensor) or not isinstance(padding, torch.Tensor):
                    raise TypeError("action and action_is_pad must be torch tensors")
                if padding.dtype != torch.bool:
                    raise ValueError("action_is_pad must be boolean")
                loss, output = policy(processed_batch)
                if not isinstance(loss, torch.Tensor) or not isinstance(output, dict):
                    raise TypeError("ACT policy must return a loss tensor and metric dictionary")
                if "l1_loss" not in output or "kld_loss" not in output:
                    raise ValueError("ACT CVAE replay requires l1_loss and kld_loss")
                valid_elements = int((~padding).sum().item()) * int(action.shape[-1])
                batches.append(
                    ReplayBatch(
                        batch_size=int(action.shape[0]),
                        valid_action_elements=valid_elements,
                        l1_loss=float(output["l1_loss"]),
                        kld_loss=float(output["kld_loss"]),
                        total_loss=float(loss.item()),
                    )
                )

    return aggregate_replay_batches(seed=seed, kl_weight=kl_weight, batches=batches)
```

- [ ] **Step 4: Run all new and baseline tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache uv run pytest hardware_test/evaluation/test_eval_act_offline.py hardware_test/evaluation/test_eval_act_checkpoints.py -q
```

Expected: `20 passed`.

- [ ] **Step 5: Commit replay behavior**

Run:

```bash
git add hardware_test/evaluation/eval_act_checkpoints.py hardware_test/evaluation/test_eval_act_checkpoints.py
git commit
```

The Lore message must state that replay runs the training-only VAE path without
gradients and that its values are checkpoint recomputations, not historical
training loss.

### Task 4: Extract Checkpoint Configuration and Normalization State

**Files:**
- Modify: `hardware_test/evaluation/eval_act_checkpoints.py`
- Modify: `hardware_test/evaluation/test_eval_act_checkpoints.py`

- [ ] **Step 1: Write failing metadata extraction tests**

Add a helper that creates a minimal checkpoint directory with `config.json`,
`policy_preprocessor.json`, one normalizer safetensors file, training step, and
optimizer param groups. Then assert:

```python
def test_read_checkpoint_metadata_marks_historical_losses_unavailable(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    from hardware_test.evaluation.eval_act_checkpoints import read_checkpoint_metadata

    checkpoint = tmp_path / "run_a" / "checkpoints" / "020000" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "input_features": {
                    "observation.state": {"shape": [8]},
                    "observation.images.l515": {"shape": [3, 540, 960]},
                },
                "output_features": {"action": {"shape": [7]}},
                "use_vae": True,
                "latent_dim": 32,
                "kl_weight": 10.0,
                "optimizer_lr": 1e-5,
                "optimizer_lr_backbone": 1e-5,
                "normalization_mapping": {
                    "VISUAL": "MEAN_STD",
                    "STATE": "MEAN_STD",
                    "ACTION": "MEAN_STD",
                },
            }
        )
    )
    (checkpoint / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "normalizer_processor",
                        "config": {
                            "norm_map": {
                                "VISUAL": "MEAN_STD",
                                "STATE": "MEAN_STD",
                                "ACTION": "MEAN_STD",
                            }
                        },
                        "state_file": "normalizer.safetensors",
                    }
                ]
            }
        )
    )
    (checkpoint / "train_config.json").write_text(json.dumps({"scheduler": None}))
    save_file(
        {
            "action.mean": torch.zeros(7),
            "action.std": torch.ones(7),
            "observation.state.mean": torch.zeros(8),
            "observation.state.std": torch.ones(8),
            "observation.images.l515.mean": torch.zeros(3, 1, 1),
            "observation.images.l515.std": torch.ones(3, 1, 1),
        },
        checkpoint / "normalizer.safetensors",
    )
    training_state = checkpoint.parent / "training_state"
    training_state.mkdir()
    (training_state / "training_step.json").write_text(json.dumps({"step": 20000}))
    (training_state / "optimizer_param_groups.json").write_text(
        json.dumps([{"lr": 1e-5}, {"lr": 1e-5}])
    )

    metadata = read_checkpoint_metadata(checkpoint)

    assert metadata.checkpoint_id == "run_a-020000"
    assert metadata.step == 20000
    assert metadata.use_vae is True
    assert metadata.kl_weight == 10.0
    assert metadata.optimizer_group_lrs == (1e-5, 1e-5)
    assert metadata.metric_status["full_lr_history"] == "inferred"
    assert metadata.metric_status["historical_l1_loss"] == "unavailable"
    assert metadata.metric_status["cvae_l1_loss"] == "recomputed"
    assert metadata.normalization_stats["action"]["mean"] == [0.0] * 7
```

Also import `json` at the top of the test file.

- [ ] **Step 2: Run the metadata test and verify failure**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache uv run pytest hardware_test/evaluation/test_eval_act_checkpoints.py::test_read_checkpoint_metadata_marks_historical_losses_unavailable -q
```

Expected: FAIL because `read_checkpoint_metadata` is missing.

- [ ] **Step 3: Implement metadata and normalization extraction**

Add `CheckpointMetadata`, JSON loading, safe checkpoint ID creation, normalizer
state loading, and explicit status fields. The implementation must:

```python
@dataclass(frozen=True)
class CheckpointMetadata:
    checkpoint_id: str
    checkpoint: str
    step: int
    use_vae: bool
    latent_dim: int
    kl_weight: float
    optimizer_lr: float
    optimizer_lr_backbone: float
    optimizer_group_lrs: tuple[float, ...]
    normalization_mapping: dict[str, str]
    normalization_stats: dict[str, dict[str, object]]
    metric_status: dict[str, str]
    lr_history_note: str


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact {path}: {error}") from error


def _checkpoint_id(checkpoint: Path) -> str:
    step_name = checkpoint.parent.name
    run_name = checkpoint.parents[2].name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{run_name}-{step_name}")


def read_checkpoint_metadata(checkpoint: Path) -> CheckpointMetadata:
    checkpoint = Path(checkpoint).expanduser().resolve()
    config = _read_json(checkpoint / "config.json")
    processor = _read_json(checkpoint / "policy_preprocessor.json")
    train_config = _read_json(checkpoint / "train_config.json")
    if not isinstance(config, dict) or not isinstance(processor, dict) or not isinstance(train_config, dict):
        raise ValueError("checkpoint configuration artifacts must contain JSON objects")
    if config.get("type") != "act" or not bool(config.get("use_vae")):
        raise ValueError("version one requires an ACT checkpoint with use_vae=true")
    input_keys = set(config.get("input_features", {}))
    if input_keys != {"observation.state", "observation.images.l515"}:
        raise ValueError("multi-floor or incompatible ACT checkpoint is out of scope")

    normalizer_steps = [
        step for step in processor.get("steps", []) if step.get("registry_name") == "normalizer_processor"
    ]
    if len(normalizer_steps) != 1:
        raise ValueError("checkpoint must contain exactly one normalizer processor")
    normalizer = normalizer_steps[0]
    state_file = checkpoint / str(normalizer.get("state_file", ""))
    tensors = load_file(state_file, device="cpu")
    features = ("observation.images.l515", "observation.state", "action")
    stats: dict[str, dict[str, object]] = {}
    for feature in features:
        feature_stats = {
            key.removeprefix(f"{feature}."): value.detach().cpu().tolist()
            for key, value in tensors.items()
            if key.startswith(f"{feature}.")
        }
        if "mean" not in feature_stats or "std" not in feature_stats:
            raise ValueError(f"normalization state is missing mean/std for {feature}")
        stats[feature] = feature_stats

    mapping = {str(key): str(value) for key, value in config.get("normalization_mapping", {}).items()}
    processor_mapping = {
        str(key): str(value) for key, value in normalizer.get("config", {}).get("norm_map", {}).items()
    }
    if mapping != processor_mapping:
        raise ValueError("config and processor normalization mappings differ")

    training_state = checkpoint.parent / "training_state"
    step_payload = _read_json(training_state / "training_step.json")
    param_groups = _read_json(training_state / "optimizer_param_groups.json")
    if not isinstance(step_payload, dict) or not isinstance(param_groups, list):
        raise ValueError("training state artifacts have invalid JSON shapes")
    optimizer_group_lrs = tuple(float(group["lr"]) for group in param_groups)
    configured_lrs = (float(config["optimizer_lr"]), float(config["optimizer_lr_backbone"]))
    scheduler_absent = train_config.get("scheduler") is None
    constant_lr_is_supported = scheduler_absent and len(optimizer_group_lrs) == 2 and all(
        math.isclose(saved, configured, rel_tol=0.0, abs_tol=0.0)
        for saved, configured in zip(optimizer_group_lrs, configured_lrs, strict=True)
    )
    lr_history_status = "inferred" if constant_lr_is_supported else "unavailable"
    lr_history_note = (
        "configuration-implied constant learning rates; no per-step observations were saved"
        if constant_lr_is_supported
        else "a full learning-rate history cannot be inferred from these artifacts"
    )

    return CheckpointMetadata(
        checkpoint_id=_checkpoint_id(checkpoint),
        checkpoint=str(checkpoint),
        step=int(step_payload["step"]),
        use_vae=bool(config["use_vae"]),
        latent_dim=int(config["latent_dim"]),
        kl_weight=float(config["kl_weight"]),
        optimizer_lr=float(config["optimizer_lr"]),
        optimizer_lr_backbone=float(config["optimizer_lr_backbone"]),
        optimizer_group_lrs=optimizer_group_lrs,
        normalization_mapping=mapping,
        normalization_stats=stats,
        metric_status={
            "configuration": "exact",
            "normalization": "exact",
            "historical_total_loss": "unavailable",
            "historical_l1_loss": "unavailable",
            "historical_kld_loss": "unavailable",
            "historical_grad_norm": "unavailable",
            "full_lr_history": lr_history_status,
            "deployment_l1": "recomputed",
            "cvae_l1_loss": "recomputed",
            "cvae_kld_loss": "recomputed",
        },
        lr_history_note=lr_history_note,
    )
```

- [ ] **Step 4: Run the new suite**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache uv run pytest hardware_test/evaluation/test_eval_act_offline.py hardware_test/evaluation/test_eval_act_checkpoints.py -q
```

Expected: `21 passed`.

- [ ] **Step 5: Commit artifact extraction**

Run:

```bash
git add hardware_test/evaluation/eval_act_checkpoints.py hardware_test/evaluation/test_eval_act_checkpoints.py
git commit
```

The commit directive must preserve the `unavailable` classification unless an
actual original-run metric artifact is later supplied.

### Task 5: Generate CSV, PNG, and Self-Contained HTML Output

**Files:**
- Modify: `hardware_test/evaluation/eval_act_checkpoints.py`
- Modify: `hardware_test/evaluation/test_eval_act_checkpoints.py`

- [ ] **Step 1: Write a failing report artifact test**

Create two synthetic checkpoint results with different metrics and call a
`save_comparison_outputs` function. Assert the complete output contract:

```python
def test_save_comparison_outputs_labels_recomputed_and_unavailable(tmp_path: Path) -> None:
    from hardware_test.evaluation.eval_act_checkpoints import (
        CheckpointComparison,
        save_comparison_outputs,
    )

    metadata = SimpleNamespace(
        checkpoint_id="run_a-020000",
        checkpoint="/run_a/checkpoints/020000/pretrained_model",
        step=20000,
        use_vae=True,
        latent_dim=32,
        kl_weight=10.0,
        optimizer_lr=1e-5,
        optimizer_lr_backbone=1e-5,
        optimizer_group_lrs=(1e-5, 1e-5),
        normalization_mapping={"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
        normalization_stats={"action": {"mean": [0.0] * 7, "std": [1.0] * 7}},
        metric_status={"historical_l1_loss": "unavailable", "cvae_l1_loss": "recomputed"},
        lr_history_note="configuration-implied constant learning rates; no per-step observations were saved",
    )
    summary = SimpleNamespace(
        metrics={
            "eval_normalized_l1": SimpleNamespace(mean=0.2, std=0.01),
            "translation_error_mm": SimpleNamespace(mean=1.5, std=0.2),
            "rotation_error_deg": SimpleNamespace(mean=2.5, std=0.3),
            "ade_mm": SimpleNamespace(mean=3.5, std=0.4),
            "fde_mm": SimpleNamespace(mean=4.5, std=0.5),
            "xyz_rmse_mm": SimpleNamespace(mean=2.0, std=0.2),
            "gripper_accuracy": SimpleNamespace(mean=0.8, std=0.1),
        }
    )
    comparison = CheckpointComparison(
        metadata=metadata,
        deployment_summary=summary,
        replay_losses=(ReplayLoss(0, 1, 2, 14, 0.3, 0.04, 0.4, 0.7),),
        output_dir=tmp_path / "checkpoints" / "run_a-020000",
    )

    paths = save_comparison_outputs(
        output_dir=tmp_path,
        dataset_root=Path("/dataset"),
        episode_ids=[1, 4],
        seeds=[0],
        comparisons=[comparison],
        failures=[],
    )

    assert set(paths) == {
        "manifest",
        "checkpoint_metrics",
        "loss_replay",
        "normalization",
        "loss_plot",
        "comparison_plot",
        "report",
    }
    assert (tmp_path / "loss_components.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    report = (tmp_path / "report.html").read_text()
    assert "Historical training loss: unavailable" in report
    assert "Recomputed CVAE objective" in report
    assert "data:image/png;base64," in report
```

- [ ] **Step 2: Run the targeted test and verify missing report API**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache uv run pytest hardware_test/evaluation/test_eval_act_checkpoints.py::test_save_comparison_outputs_labels_recomputed_and_unavailable -q
```

Expected: FAIL because `CheckpointComparison` and
`save_comparison_outputs` are missing.

- [ ] **Step 3: Implement report dataclass and row aggregation**

Add:

```python
@dataclass(frozen=True)
class CheckpointComparison:
    metadata: object
    deployment_summary: object
    replay_losses: tuple[ReplayLoss, ...]
    output_dir: Path


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("report values must be finite and non-empty")
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def _comparison_rows(comparisons: Sequence[CheckpointComparison]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    checkpoint_rows: list[dict[str, object]] = []
    loss_rows: list[dict[str, object]] = []
    for item in comparisons:
        metric_row: dict[str, object] = {
            "checkpoint_id": item.metadata.checkpoint_id,
            "step": item.metadata.step,
            "use_vae": item.metadata.use_vae,
            "kl_weight": item.metadata.kl_weight,
            "optimizer_lr": item.metadata.optimizer_lr,
            "optimizer_lr_backbone": item.metadata.optimizer_lr_backbone,
            "historical_loss_status": "unavailable",
        }
        for name, value in item.deployment_summary.metrics.items():
            metric_row[f"{name}_mean"] = value.mean
            metric_row[f"{name}_std"] = value.std
        checkpoint_rows.append(metric_row)

        l1_mean, l1_std = _mean_std([row.l1_loss for row in item.replay_losses])
        kld_mean, kld_std = _mean_std([row.kld_loss for row in item.replay_losses])
        weighted_mean, weighted_std = _mean_std([row.weighted_kld_loss for row in item.replay_losses])
        total_mean, total_std = _mean_std([row.total_loss for row in item.replay_losses])
        loss_rows.append(
            {
                "checkpoint_id": item.metadata.checkpoint_id,
                "step": item.metadata.step,
                "status": "recomputed",
                "seeds": ",".join(str(row.seed) for row in item.replay_losses),
                "l1_loss_mean": l1_mean,
                "l1_loss_std": l1_std,
                "kld_loss_mean": kld_mean,
                "kld_loss_std": kld_std,
                "weighted_kld_loss_mean": weighted_mean,
                "weighted_kld_loss_std": weighted_std,
                "total_loss_mean": total_mean,
                "total_loss_std": total_std,
            }
        )
    return checkpoint_rows, loss_rows
```

- [ ] **Step 4: Implement atomic Pillow charts and HTML**

Add small reusable bar-panel helpers using `PIL.Image` and `PIL.ImageDraw`.
`loss_components.png` must stack L1 and weighted KLD and print total above each
bar. `checkpoint_comparison.png` must draw five panels for normalized L1,
translation error, rotation error, ADE, and FDE. Write images through a temporary
file and `Path.replace`, matching the existing evaluator.

Build `report.html` from escaped values and base64-embedded PNG bytes. The HTML
must include these literal sections so availability cannot be ambiguous:

```html
<h2>Metric availability</h2>
<p class="unavailable">Historical training loss: unavailable</p>
<p class="recomputed">Recomputed CVAE objective: fixed expert episodes and declared seeds</p>
<h2>Checkpoint configuration</h2>
<h2>Recomputed loss</h2>
<h2>Expert comparison</h2>
<h2>Normalization</h2>
<h2>Limitations</h2>
```

Implement `save_comparison_outputs` to atomically write:

```python
paths = {
    "manifest": output_dir / "evaluation_manifest.json",
    "checkpoint_metrics": output_dir / "checkpoint_metrics.csv",
    "loss_replay": output_dir / "loss_replay.csv",
    "normalization": output_dir / "normalization_summary.json",
    "loss_plot": output_dir / "loss_components.png",
    "comparison_plot": output_dir / "checkpoint_comparison.png",
    "report": output_dir / "report.html",
}
```

The manifest records dataset path, episodes, seeds, checkpoint paths/steps,
`expert-data replay / original train membership unknown`, and failures. The
normalization JSON serializes each checkpoint's mapping and saved statistics.

- [ ] **Step 5: Run report and regression tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache uv run pytest hardware_test/evaluation/test_eval_act_offline.py hardware_test/evaluation/test_eval_act_checkpoints.py -q
```

Expected: `22 passed`.

- [ ] **Step 6: Commit static reporting**

Run:

```bash
git add hardware_test/evaluation/eval_act_checkpoints.py hardware_test/evaluation/test_eval_act_checkpoints.py
git commit
```

The Lore directive must prohibit renaming commanded delta trajectories to
measured end-effector trajectories or offline metrics to task success.

### Task 6: Add Multi-Checkpoint CLI Orchestration

**Files:**
- Modify: `hardware_test/evaluation/eval_act_checkpoints.py`
- Modify: `hardware_test/evaluation/test_eval_act_checkpoints.py`

- [ ] **Step 1: Write failing CLI and injected-orchestration tests**

Test that parsing accepts multiple checkpoints, requires unique nonnegative
episodes and seeds, and rejects an existing output directory. Add an injected
orchestration test whose fake runtime loader and replay policy loader record
every call; assert all checkpoints receive the exact same episode tuple and
each seed receives a fresh policy object.

The parser call must be:

```python
args = parse_args(
    [
        "--checkpoints",
        "/run/checkpoints/020000/pretrained_model",
        "/run/checkpoints/040000/pretrained_model",
        "--dataset-root",
        "/dataset",
        "--episodes",
        "1",
        "4",
        "--output-dir",
        str(tmp_path / "report"),
        "--device",
        "cpu",
        "--cvae-seeds",
        "0",
        "7",
    ]
)
assert args.episodes == [1, 4]
assert args.cvae_seeds == [0, 7]
assert len(args.checkpoints) == 2
```

- [ ] **Step 2: Run the targeted parser/orchestration tests and verify failure**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache uv run pytest hardware_test/evaluation/test_eval_act_checkpoints.py -k 'parse_args or orchestration' -q
```

Expected: FAIL because the CLI functions are absent.

- [ ] **Step 3: Implement parser and validation**

Add `build_arg_parser()` and `parse_args()` with:

```python
parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
parser.add_argument("--dataset-root", type=Path, required=True)
parser.add_argument("--episodes", type=int, nargs="+", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--device", default="cuda")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--cvae-seeds", type=int, nargs="+", default=[0, 1, 2])
parser.add_argument("--video-backend", choices=("torchcodec", "pyav"), default=None)
parser.add_argument("--dataset-repo-id", default=None)
```

Validate unique checkpoint paths, episode IDs, and seeds; nonnegative episodes
and seeds; positive batch size; nonnegative workers; nonempty device; and a
nonexistent output directory.

- [ ] **Step 4: Implement fresh policy loading and orchestration**

Add:

```python
def load_replay_policy(checkpoint: Path, config: object, device: str) -> object:
    from lerobot import policies as lerobot_policies

    return (
        lerobot_policies.get_policy_class(config.type)
        .from_pretrained(
            checkpoint,
            config=config,
            local_files_only=True,
            strict=True,
        )
        .to(device)
    )
```

Implement `run_comparison` with injectable `runtime_loader`,
`policy_loader`, and `single_evaluator`. For every checkpoint:

1. build a `SimpleNamespace` compatible with `load_evaluation_runtime`;
2. write the existing outputs under
   `checkpoints/<run-and-step-id>/`;
3. load and replay one fresh policy per seed;
4. append `CheckpointComparison` on success;
5. append `{checkpoint, error_type, message}` on failure and continue;
6. call `save_comparison_outputs` after the loop;
7. return a result containing comparisons, failures, and output paths.

Use these result and orchestration definitions:

```python
@dataclass(frozen=True)
class ComparisonRunResult:
    comparisons: tuple[CheckpointComparison, ...]
    failures: tuple[dict[str, str], ...]
    output_paths: dict[str, Path]


def run_comparison(
    args: argparse.Namespace,
    *,
    runtime_loader: Callable[[argparse.Namespace], EvaluationRuntime] = load_evaluation_runtime,
    policy_loader: Callable[[Path, object, str], object] = load_replay_policy,
    single_evaluator: Callable[..., EvaluationRunResult] = run_evaluation,
) -> ComparisonRunResult:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    comparisons: list[CheckpointComparison] = []
    failures: list[dict[str, str]] = []

    for checkpoint_arg in args.checkpoints:
        checkpoint = Path(checkpoint_arg).expanduser().resolve()
        checkpoint_output = output_dir / "checkpoints" / _checkpoint_id(checkpoint)
        checkpoint_args = SimpleNamespace(
            checkpoint=checkpoint,
            dataset_root=Path(args.dataset_root),
            episodes=list(args.episodes),
            output_dir=checkpoint_output,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            video_backend=args.video_backend,
            dataset_repo_id=args.dataset_repo_id,
        )
        try:
            runtime = runtime_loader(checkpoint_args)
            metadata = read_checkpoint_metadata(runtime.checkpoint)
            deployment = single_evaluator(checkpoint_args, runtime=runtime)
            replay_losses: list[ReplayLoss] = []
            for seed in args.cvae_seeds:
                replay_policy = policy_loader(runtime.checkpoint, runtime.config, runtime.device)
                replay_losses.append(
                    replay_cvae_objective(
                        policy=replay_policy,
                        preprocessor=runtime.preprocessor,
                        dataset=runtime.dataset,
                        episode_ids=args.episodes,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        seed=seed,
                        kl_weight=metadata.kl_weight,
                        device=runtime.device,
                    )
                )
                del replay_policy
            comparisons.append(
                CheckpointComparison(
                    metadata=metadata,
                    deployment_summary=deployment.summary,
                    replay_losses=tuple(replay_losses),
                    output_dir=checkpoint_output,
                )
            )
        except Exception as error:
            failures.append(
                {
                    "checkpoint": str(checkpoint),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )

    output_paths = save_comparison_outputs(
        output_dir=output_dir,
        dataset_root=Path(args.dataset_root),
        episode_ids=args.episodes,
        seeds=args.cvae_seeds,
        comparisons=comparisons,
        failures=failures,
    )
    return ComparisonRunResult(tuple(comparisons), tuple(failures), output_paths)
```

`save_comparison_outputs` must write a failure-only HTML report and placeholder
PNGs when `comparisons` is empty, so a completely failed invocation still
records why it failed.

The default `main()` returns `1` when any requested checkpoint failed and `0`
otherwise:

```python
def main(argv: Sequence[str] | None = None) -> int:
    result = run_comparison(parse_args(argv))
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run all evaluation tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache uv run pytest hardware_test/evaluation -q
```

Expected: all existing and new evaluation tests pass.

- [ ] **Step 6: Run focused Ruff checks without changing unrelated files**

Run:

```bash
UV_CACHE_DIR=/tmp/lerobot-uv-cache uv run ruff check hardware_test/evaluation
UV_CACHE_DIR=/tmp/lerobot-uv-cache uv run ruff format --check hardware_test/evaluation
```

Expected: both commands exit zero. If formatting is required, run
`uv run ruff format` only on the two new files and rerun tests.

- [ ] **Step 7: Commit CLI orchestration**

Run:

```bash
git add hardware_test/evaluation/eval_act_checkpoints.py hardware_test/evaluation/test_eval_act_checkpoints.py
git commit
```

The Lore message must record that multi-floor checkpoints fail closed and that
per-checkpoint failures do not erase successful artifacts.

### Task 7: Real Checkpoint Smoke Test and Final Verification

**Files:**
- Verify only: `hardware_test/evaluation/`
- Produce temporary artifacts under: `/tmp/act_eval_smoke_20260716/`

- [ ] **Step 1: Run the full evaluation regression suite fresh**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 UV_CACHE_DIR=/tmp/lerobot-uv-cache uv run pytest hardware_test/evaluation -q
```

Expected: zero failures.

- [ ] **Step 2: Run the real single-checkpoint, single-seed smoke test**

Choose a new nonexisting output path if the listed path already exists. Run:

```bash
UV_CACHE_DIR=/tmp/lerobot-uv-cache uv run python -m hardware_test.evaluation.eval_act_checkpoints \
  --checkpoints outputs/train/act_press_button_29ep_20260710/checkpoints/100000/pretrained_model \
  --dataset-root outputs/hardware_test/press_button_train \
  --episodes 0 \
  --output-dir /tmp/act_eval_smoke_20260716/report \
  --device cuda \
  --batch-size 8 \
  --num-workers 0 \
  --cvae-seeds 0
```

Expected: exit zero and all output-contract files exist. This run is a smoke
test, not the final 29-versus-50 comparison.

- [ ] **Step 3: Verify loss identity and report labels from generated artifacts**

Run:

```bash
python3 -c 'import csv, pathlib; p=pathlib.Path("/tmp/act_eval_smoke_20260716/report"); rows=list(csv.DictReader((p/"loss_replay.csv").open())); assert rows; r=rows[0]; assert abs(float(r["total_loss_mean"])-(float(r["l1_loss_mean"])+float(r["weighted_kld_loss_mean"]))) < 1e-6; h=(p/"report.html").read_text(); assert "Historical training loss: unavailable" in h; assert "Recomputed CVAE objective" in h; print("smoke artifacts verified")'
```

Expected: `smoke artifacts verified`.

- [ ] **Step 4: Run final diff and status review**

Run:

```bash
git diff --check
git status --short
git log --oneline -7
```

Expected: no whitespace errors, no uncommitted evaluation changes, and all
unrelated pre-existing worktree changes remain untouched.

- [ ] **Step 5: Report the delivered command and remaining evidence boundary**

The handoff must state:

- exact files added/committed;
- test counts and real smoke command result;
- the report path from the smoke run;
- historical total/L1/KLD/gradient remain unavailable;
- CVAE loss is a fixed-data recomputation, not the original training curve;
- the full two-model comparison can now be run with both checkpoint lists and
  the same selected episode IDs.
