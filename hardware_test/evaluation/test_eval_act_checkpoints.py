from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hardware_test.evaluation.eval_act_checkpoints import ReplayBatch, aggregate_replay_batches


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
                "output_features": {
                    "action": {"shape": [7]},
                },
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
    (training_state / "optimizer_param_groups.json").write_text(json.dumps([{"lr": 1e-5}, {"lr": 1e-5}]))

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


def test_read_checkpoint_metadata_rejects_multifloor_input_schema(tmp_path: Path) -> None:
    from hardware_test.evaluation.eval_act_checkpoints import read_checkpoint_metadata

    checkpoint = tmp_path / "run" / "checkpoints" / "last" / "pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "use_vae": True,
                "input_features": {
                    "observation.state": {"shape": [8]},
                    "observation.images.l515": {"shape": [3, 540, 960]},
                    "observation.environment_state": {"shape": [5]},
                },
            }
        )
    )
    (checkpoint / "policy_preprocessor.json").write_text("{}")
    (checkpoint / "train_config.json").write_text("{}")

    with pytest.raises(ValueError, match="multi-floor or incompatible"):
        read_checkpoint_metadata(checkpoint)


def test_save_comparison_outputs_labels_recomputed_and_unavailable(tmp_path: Path) -> None:
    from hardware_test.evaluation.eval_act_checkpoints import (
        CheckpointComparison,
        ReplayLoss,
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
        normalization_mapping={
            "VISUAL": "MEAN_STD",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        },
        normalization_stats={"action": {"mean": [0.0] * 7, "std": [1.0] * 7}},
        metric_status={
            "historical_l1_loss": "unavailable",
            "cvae_l1_loss": "recomputed",
        },
        lr_history_note=(
            "configuration-implied constant learning rates; no per-step observations were saved"
        ),
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
            "gripper_gt_open_rate": SimpleNamespace(mean=0.95, std=0.02),
        }
    )
    comparison = CheckpointComparison(
        metadata=metadata,
        deployment_summary=summary,
        replay_losses=(
            ReplayLoss(0, 1, 2, 14, 0.3, 0.04, 0.4, 0.7),
            ReplayLoss(7, 1, 2, 14, 0.5, 0.02, 0.2, 0.7),
        ),
        output_dir=tmp_path / "checkpoints" / "run_a-020000",
    )

    paths = save_comparison_outputs(
        output_dir=tmp_path,
        dataset_root=Path("/dataset"),
        episode_ids=[1, 4],
        seeds=[0, 7],
        comparisons=[comparison],
        failures=[],
        evaluation_parameters={
            "device": "cpu",
            "batch_size": 2,
            "num_workers": 0,
            "video_backend": "pyav",
        },
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
    assert (tmp_path / "checkpoint_comparison.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    report = (tmp_path / "report.html").read_text()
    assert "Historical training loss: unavailable" in report
    assert "Recomputed CVAE objective" in report
    assert "data:image/png;base64," in report
    assert "expert-data replay / original train membership unknown" in report
    assert "0.400000 ± 0.141421" in report
    assert "Gripper GT open rate" in report
    manifest = json.loads((tmp_path / "evaluation_manifest.json").read_text())
    assert manifest["evaluation_parameters"] == {
        "device": "cpu",
        "batch_size": 2,
        "num_workers": 0,
        "video_backend": "pyav",
    }


def test_parse_args_accepts_multiple_checkpoints_and_fixed_seeds(tmp_path: Path) -> None:
    from hardware_test.evaluation.eval_act_checkpoints import parse_args

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


def test_parse_args_rejects_existing_output_and_duplicate_ids(tmp_path: Path) -> None:
    from hardware_test.evaluation.eval_act_checkpoints import parse_args

    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    common = [
        "--checkpoints",
        "/run/checkpoints/020000/pretrained_model",
        "--dataset-root",
        "/dataset",
        "--output-dir",
        str(tmp_path / "new-report"),
        "--episodes",
    ]
    with pytest.raises(ValueError, match="episode IDs must be unique"):
        parse_args([*common, "1", "1"])
    with pytest.raises(ValueError, match="CVAE seeds must be unique"):
        parse_args([*common, "1", "--cvae-seeds", "3", "3"])
    with pytest.raises(ValueError, match="output directory already exists"):
        parse_args(
            [
                "--checkpoints",
                "/run/checkpoints/020000/pretrained_model",
                "--dataset-root",
                "/dataset",
                "--output-dir",
                str(output_dir),
                "--episodes",
                "1",
            ]
        )


def test_run_comparison_reuses_episode_manifest_and_reloads_each_seed(tmp_path: Path) -> None:
    from hardware_test.evaluation.eval_act_checkpoints import run_comparison

    checkpoints = [
        tmp_path / "run_a" / "checkpoints" / "020000" / "pretrained_model",
        tmp_path / "run_a" / "checkpoints" / "040000" / "pretrained_model",
    ]
    colliding_checkpoint = tmp_path / "other_root" / "run_a" / "checkpoints" / "020000" / "pretrained_model"
    args = SimpleNamespace(
        checkpoints=[*checkpoints, colliding_checkpoint],
        dataset_root=tmp_path / "dataset",
        episodes=[5],
        output_dir=tmp_path / "report",
        device="cpu",
        batch_size=2,
        num_workers=0,
        cvae_seeds=[0, 7],
        video_backend=None,
        dataset_repo_id="local/expert-demo",
    )
    runtime_episode_calls: list[tuple[int, ...]] = []
    loaded_policies: list[_ReplayPolicy] = []

    def runtime_loader(checkpoint_args: SimpleNamespace) -> SimpleNamespace:
        runtime_episode_calls.append(tuple(checkpoint_args.episodes))
        return SimpleNamespace(
            checkpoint=Path(checkpoint_args.checkpoint),
            dataset_root=Path(checkpoint_args.dataset_root),
            device="cpu",
            config=SimpleNamespace(type="act", chunk_size=2),
            preprocessor=_ReplayPreprocessor(),
            dataset=_ReplayDataset(),
        )

    def metadata_loader(checkpoint: Path) -> SimpleNamespace:
        step = int(checkpoint.parent.name)
        return SimpleNamespace(
            checkpoint_id=f"run_a-{step:06d}",
            checkpoint=str(checkpoint),
            step=step,
            use_vae=True,
            latent_dim=32,
            kl_weight=10.0,
            optimizer_lr=1e-5,
            optimizer_lr_backbone=1e-5,
            optimizer_group_lrs=(1e-5, 1e-5),
            normalization_mapping={"ACTION": "MEAN_STD"},
            normalization_stats={"action": {"mean": [0.0] * 7, "std": [1.0] * 7}},
            metric_status={
                "historical_l1_loss": "unavailable",
                "cvae_l1_loss": "recomputed",
            },
            lr_history_note="configuration-implied constant learning rates",
        )

    def policy_loader(checkpoint: Path, config: object, device: str) -> _ReplayPolicy:
        assert checkpoint in checkpoints
        assert config.type == "act"
        assert device == "cpu"
        policy = _ReplayPolicy()
        loaded_policies.append(policy)
        return policy

    summary = SimpleNamespace(
        metrics={
            "eval_normalized_l1": SimpleNamespace(mean=0.2, std=0.01),
            "translation_error_mm": SimpleNamespace(mean=1.5, std=0.2),
            "rotation_error_deg": SimpleNamespace(mean=2.5, std=0.3),
            "ade_mm": SimpleNamespace(mean=3.5, std=0.4),
            "fde_mm": SimpleNamespace(mean=4.5, std=0.5),
            "xyz_rmse_mm": SimpleNamespace(mean=2.0, std=0.2),
            "gripper_accuracy": SimpleNamespace(mean=0.8, std=0.1),
            "gripper_gt_open_rate": SimpleNamespace(mean=0.95, std=0.02),
        }
    )

    def single_evaluator(checkpoint_args: SimpleNamespace, *, runtime: object) -> SimpleNamespace:
        assert runtime.checkpoint == checkpoint_args.checkpoint
        return SimpleNamespace(summary=summary)

    result = run_comparison(
        args,
        runtime_loader=runtime_loader,
        metadata_loader=metadata_loader,
        policy_loader=policy_loader,
        single_evaluator=single_evaluator,
    )

    assert len(result.comparisons) == 2
    assert len(result.failures) == 1
    assert result.failures[0]["checkpoint"] == str(colliding_checkpoint.resolve())
    assert "derived checkpoint ID collision" in result.failures[0]["message"]
    assert runtime_episode_calls == [(5,), (5,)]
    assert len(loaded_policies) == 4
    assert len({id(policy) for policy in loaded_policies}) == 4
    assert result.output_paths["report"].is_file()
    manifest = json.loads(result.output_paths["manifest"].read_text())
    assert manifest["evaluation_parameters"]["dataset_repo_id"] == "local/expert-demo"
    assert manifest["evaluation_parameters"]["requested_dataset_repo_id"] == "local/expert-demo"


def test_short_checkpoint_label_preserves_training_run_discriminator() -> None:
    from hardware_test.evaluation.eval_act_checkpoints import _short_checkpoint_label

    assert _short_checkpoint_label("act_press_button_29ep_20260710-100000") == "29ep_20260710-100000"
    assert _short_checkpoint_label("run_a-020000") == "run_a-020000"


def test_checkpoint_id_rejects_malformed_path_with_actionable_error(tmp_path: Path) -> None:
    from hardware_test.evaluation.eval_act_checkpoints import _checkpoint_id

    with pytest.raises(ValueError, match="must end with <run>/checkpoints/<step>/pretrained_model"):
        _checkpoint_id(tmp_path / "pretrained_model")
