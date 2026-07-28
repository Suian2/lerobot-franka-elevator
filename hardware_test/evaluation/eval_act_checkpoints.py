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
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Subset

from hardware_test.evaluation.eval_act_offline import (
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


@dataclass(frozen=True)
class CheckpointComparison:
    metadata: object
    deployment_summary: object
    replay_losses: tuple[ReplayLoss, ...]
    output_dir: Path


@dataclass(frozen=True)
class ComparisonRunResult:
    comparisons: tuple[CheckpointComparison, ...]
    failures: tuple[dict[str, str], ...]
    output_paths: dict[str, Path]


def aggregate_replay_batches(*, seed: int, kl_weight: float, batches: Sequence[ReplayBatch]) -> ReplayLoss:
    """Aggregate ACT loss components using their native dataset denominators."""

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
    """Recompute the action-conditioned ACT objective for one fixed RNG seed."""

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


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON artifact {path}: {error}") from error


def _checkpoint_id(checkpoint: Path) -> str:
    checkpoint = Path(checkpoint)
    if (
        checkpoint.name != "pretrained_model"
        or len(checkpoint.parents) < 3
        or checkpoint.parents[1].name != "checkpoints"
        or not checkpoint.parent.name
        or not checkpoint.parents[2].name
    ):
        raise ValueError("checkpoint path must end with <run>/checkpoints/<step>/pretrained_model")
    step_name = checkpoint.parent.name
    run_name = checkpoint.parents[2].name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{run_name}-{step_name}")


def read_checkpoint_metadata(checkpoint: Path) -> CheckpointMetadata:
    """Read exactly stored ACT configuration, optimizer, and processor state."""

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
    constant_lr_is_supported = (
        scheduler_absent
        and len(optimizer_group_lrs) == 2
        and all(
            math.isclose(saved, configured, rel_tol=0.0, abs_tol=0.0)
            for saved, configured in zip(optimizer_group_lrs, configured_lrs, strict=True)
        )
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


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("report values must be finite and non-empty")
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return float(array.mean()), std


def _comparison_rows(
    comparisons: Sequence[CheckpointComparison],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
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
        if not math.isclose(total_mean, l1_mean + weighted_mean, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f"ACT loss identity failed for checkpoint {item.metadata.checkpoint_id}")
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


def _atomic_save_png(path: Path, image: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png", dir=path.parent, prefix=f".{path.stem}.", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
        image.save(temporary_path, format="PNG")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _draw_empty_plot(path: Path, title: str) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1200, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 30), title, fill="black")
    draw.text((40, 100), "No compatible checkpoint completed successfully.", fill="#8a1c1c")
    _atomic_save_png(path, image)


def _short_checkpoint_label(label: str, max_length: int = 24) -> str:
    common_prefix = "act_press_button_"
    if label.startswith(common_prefix):
        label = label.removeprefix(common_prefix)
    if len(label) <= max_length:
        return label
    suffix_length = max_length - 13
    return f"{label[:10]}...{label[-suffix_length:]}"


def _save_loss_plot(path: Path, loss_rows: Sequence[dict[str, object]]) -> None:
    from PIL import Image, ImageDraw

    if not loss_rows:
        _draw_empty_plot(path, "Recomputed ACT CVAE objective")
        return
    image = Image.new("RGB", (1400, 760), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 110, 100, 1350, 620
    draw.text((40, 30), "Recomputed ACT CVAE objective (fixed expert episodes)", fill="black")
    draw.rectangle((left, top, right, bottom), outline="black", width=2)
    totals = [float(row["total_loss_mean"]) for row in loss_rows]
    maximum = max(max(totals) * 1.15, 1e-9)
    for tick in range(6):
        value = maximum * tick / 5
        y = bottom - round((bottom - top) * tick / 5)
        draw.line((left, y, right, y), fill="#dddddd", width=1)
        draw.text((30, y - 7), f"{value:.3f}", fill="#444444")
    slot = (right - left) / len(loss_rows)
    bar_width = max(24, min(110, int(slot * 0.55)))
    for index, row in enumerate(loss_rows):
        center = left + (index + 0.5) * slot
        x0 = round(center - bar_width / 2)
        x1 = round(center + bar_width / 2)
        l1 = float(row["l1_loss_mean"])
        total = float(row["total_loss_mean"])
        y_l1 = bottom - round((bottom - top) * l1 / maximum)
        y_total = bottom - round((bottom - top) * total / maximum)
        draw.rectangle((x0, y_l1, x1, bottom), fill="#2f77b4", outline="black")
        draw.rectangle((x0, y_total, x1, y_l1), fill="#f28e2b", outline="black")
        draw.text((x0, max(top, y_total - 18)), f"{total:.3f}", fill="black")
        label = _short_checkpoint_label(str(row["checkpoint_id"]))
        draw.text((round(center - len(label) * 3), bottom + 14), label, fill="black")
    draw.rectangle((1030, 40, 1050, 60), fill="#2f77b4", outline="black")
    draw.text((1060, 42), "normalized L1", fill="black")
    draw.rectangle((1190, 40, 1210, 60), fill="#f28e2b", outline="black")
    draw.text((1220, 42), "weighted KLD", fill="black")
    draw.text((570, 700), "checkpoint", fill="black")
    _atomic_save_png(path, image)


def _draw_metric_panel(
    draw: object,
    bounds: tuple[int, int, int, int],
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
) -> None:
    left, top, right, bottom = bounds
    draw.text((left, top - 24), title, fill="black")
    draw.rectangle(bounds, outline="black", width=2)
    maximum = max(max(values) * 1.15, 1e-9)
    slot = (right - left) / len(values)
    width = max(16, min(70, int(slot * 0.5)))
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        center = left + (index + 0.5) * slot
        x0, x1 = round(center - width / 2), round(center + width / 2)
        y = bottom - round((bottom - top) * value / maximum)
        draw.rectangle((x0, y, x1, bottom), fill="#59a14f", outline="black")
        draw.text((x0, max(top, y - 17)), f"{value:.3g}", fill="black")
        short_label = _short_checkpoint_label(label)
        draw.text((round(center - len(short_label) * 3), bottom + 8), short_label, fill="black")


def _save_comparison_plot(path: Path, rows: Sequence[dict[str, object]]) -> None:
    from PIL import Image, ImageDraw

    if not rows:
        _draw_empty_plot(path, "Expert-action comparison")
        return
    image = Image.new("RGB", (1500, 1240), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 24), "Teacher-forced expert-action comparison by checkpoint", fill="black")
    labels = [str(row["checkpoint_id"]) for row in rows]
    panels = (
        ("eval_normalized_l1_mean", "Normalized L1", (100, 100, 700, 390)),
        ("translation_error_mm_mean", "Translation error (mm)", (820, 100, 1420, 390)),
        ("rotation_error_deg_mean", "Rotation error (deg)", (100, 500, 700, 790)),
        ("ade_mm_mean", "Commanded-delta ADE (mm)", (820, 500, 1420, 790)),
        ("fde_mm_mean", "Commanded-delta FDE (mm)", (460, 900, 1060, 1190)),
    )
    for field, title, bounds in panels:
        _draw_metric_panel(draw, bounds, labels, [float(row[field]) for row in rows], title)
    _atomic_save_png(path, image)


def _metadata_payload(metadata: object) -> dict[str, object]:
    return {
        "checkpoint_id": metadata.checkpoint_id,
        "checkpoint": metadata.checkpoint,
        "step": metadata.step,
        "use_vae": metadata.use_vae,
        "latent_dim": metadata.latent_dim,
        "kl_weight": metadata.kl_weight,
        "optimizer_lr": metadata.optimizer_lr,
        "optimizer_lr_backbone": metadata.optimizer_lr_backbone,
        "optimizer_group_lrs": list(metadata.optimizer_group_lrs),
        "normalization_mapping": metadata.normalization_mapping,
        "metric_status": metadata.metric_status,
        "lr_history_note": metadata.lr_history_note,
    }


def _png_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _html_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def save_comparison_outputs(
    *,
    output_dir: Path,
    dataset_root: Path,
    episode_ids: Sequence[int],
    seeds: Sequence[int],
    comparisons: Sequence[CheckpointComparison],
    failures: Sequence[dict[str, str]],
    evaluation_parameters: dict[str, object],
) -> dict[str, Path]:
    """Write versioned cross-checkpoint artifacts and a self-contained report."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": output_dir / "evaluation_manifest.json",
        "checkpoint_metrics": output_dir / "checkpoint_metrics.csv",
        "loss_replay": output_dir / "loss_replay.csv",
        "normalization": output_dir / "normalization_summary.json",
        "loss_plot": output_dir / "loss_components.png",
        "comparison_plot": output_dir / "checkpoint_comparison.png",
        "report": output_dir / "report.html",
    }
    checkpoint_rows, loss_rows = _comparison_rows(comparisons)
    checkpoint_fields = list(checkpoint_rows[0]) if checkpoint_rows else ["checkpoint_id", "step"]
    loss_fields = list(loss_rows[0]) if loss_rows else ["checkpoint_id", "step", "status"]
    _atomic_write_text(paths["checkpoint_metrics"], _csv_text(checkpoint_fields, checkpoint_rows))
    _atomic_write_text(paths["loss_replay"], _csv_text(loss_fields, loss_rows))

    manifest = {
        "schema_version": 1,
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "episodes": [int(item) for item in episode_ids],
        "cvae_seeds": [int(item) for item in seeds],
        "evaluation_parameters": evaluation_parameters,
        "dataset_role": "expert-data replay / original train membership unknown",
        "metric_provenance": {
            "configuration_and_normalization": "exact",
            "historical_training_loss": "unavailable",
            "deployment_and_cvae_metrics": "recomputed",
        },
        "checkpoints": [_metadata_payload(item.metadata) for item in comparisons],
        "failures": list(failures),
    }
    normalization = {
        item.metadata.checkpoint_id: {
            "mapping": item.metadata.normalization_mapping,
            "stats": item.metadata.normalization_stats,
        }
        for item in comparisons
    }
    _atomic_write_text(paths["manifest"], json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    _atomic_write_text(paths["normalization"], json.dumps(normalization, indent=2, allow_nan=False) + "\n")
    _save_loss_plot(paths["loss_plot"], loss_rows)
    _save_comparison_plot(paths["comparison_plot"], checkpoint_rows)

    config_table = _html_table(
        ("Checkpoint", "Step", "use_vae", "kl_weight", "optimizer LR", "backbone LR"),
        [
            (
                item.metadata.checkpoint_id,
                item.metadata.step,
                item.metadata.use_vae,
                item.metadata.kl_weight,
                item.metadata.optimizer_lr,
                item.metadata.optimizer_lr_backbone,
            )
            for item in comparisons
        ],
    )
    loss_table = _html_table(
        (
            "Checkpoint",
            "Status",
            "L1",
            "KLD",
            "Weighted KLD",
            "Total",
            "Seeds",
        ),
        [
            (
                row["checkpoint_id"],
                row["status"],
                f"{float(row['l1_loss_mean']):.6f} ± {float(row['l1_loss_std']):.6f}",
                f"{float(row['kld_loss_mean']):.6f} ± {float(row['kld_loss_std']):.6f}",
                f"{float(row['weighted_kld_loss_mean']):.6f} ± {float(row['weighted_kld_loss_std']):.6f}",
                f"{float(row['total_loss_mean']):.6f} ± {float(row['total_loss_std']):.6f}",
                row["seeds"],
            )
            for row in loss_rows
        ],
    )
    expert_metrics = (
        ("Normalized L1", "eval_normalized_l1"),
        ("Translation error (mm)", "translation_error_mm"),
        ("Rotation error (deg)", "rotation_error_deg"),
        ("ADE (mm)", "ade_mm"),
        ("FDE (mm)", "fde_mm"),
        ("XYZ RMSE (mm)", "xyz_rmse_mm"),
        ("Gripper accuracy", "gripper_accuracy"),
        ("Gripper GT open rate", "gripper_gt_open_rate"),
    )
    expert_table = _html_table(
        ("Checkpoint", *(label for label, _ in expert_metrics)),
        [
            (
                row["checkpoint_id"],
                *(
                    f"{float(row[f'{key}_mean']):.6f} ± {float(row[f'{key}_std']):.6f}"
                    for _, key in expert_metrics
                ),
            )
            for row in checkpoint_rows
        ],
    )
    failure_table = _html_table(
        ("Checkpoint", "Error", "Message"),
        [(row["checkpoint"], row["error_type"], row["message"]) for row in failures],
    )
    normalization_text = json.dumps(normalization, indent=2, allow_nan=False)
    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>ACT offline evaluation</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 20px;color:#222}}
table{{border-collapse:collapse;width:100%;margin:12px 0 24px}}th,td{{border:1px solid #ccc;padding:7px;text-align:left}}
th{{background:#f2f4f7}}img{{max-width:100%;height:auto;border:1px solid #ddd}}pre{{overflow:auto;background:#f6f8fa;padding:12px}}
.unavailable{{color:#9c1c1c;font-weight:700}}.recomputed{{color:#145c2e;font-weight:700}}.warning{{background:#fff4ce;padding:10px}}
</style></head><body>
<h1>ACT offline training-result evaluation</h1>
<p class="warning">Dataset role: expert-data replay / original train membership unknown.</p>
<h2>Metric availability</h2>
<p class="unavailable">Historical training loss: unavailable</p>
<p class="recomputed">Recomputed CVAE objective: fixed expert episodes and declared seeds</p>
<h2>Checkpoint configuration</h2>{config_table}
<h2>Recomputed loss</h2>{loss_table}<img alt="Recomputed loss components" src="{_png_data_uri(paths["loss_plot"])}">
<h2>Expert comparison</h2><p>Teacher-forced action agreement; commanded deltas are not measured absolute end-effector trajectories.</p>
{expert_table}
<img alt="Expert comparison" src="{_png_data_uri(paths["comparison_plot"])}">
<h2>Normalization</h2><pre>{html.escape(normalization_text)}</pre>
<h2>Failures</h2>{failure_table}
<h2>Limitations</h2><ul><li>Offline agreement is not closed-loop task success.</li><li>Gripper accuracy must be read together with ground-truth class prevalence.</li><li>Recomputed loss is not an original training curve.</li></ul>
</body></html>"""
    _atomic_write_text(paths["report"], report)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare ACT checkpoints on fixed expert episodes and recompute CVAE loss."
    )
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
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    if any(item < 0 for item in args.episodes):
        raise ValueError("episode IDs must be non-negative")
    if len(set(args.episodes)) != len(args.episodes):
        raise ValueError("episode IDs must be unique")
    if any(item < 0 for item in args.cvae_seeds):
        raise ValueError("CVAE seeds must be non-negative")
    if len(set(args.cvae_seeds)) != len(args.cvae_seeds):
        raise ValueError("CVAE seeds must be unique")
    checkpoint_keys = [str(Path(item).expanduser().resolve()) for item in args.checkpoints]
    if len(set(checkpoint_keys)) != len(checkpoint_keys):
        raise ValueError("checkpoint paths must be unique")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.num_workers < 0:
        raise ValueError("num workers must be non-negative")
    if not str(args.device).strip():
        raise ValueError("device must not be empty")
    if Path(args.output_dir).expanduser().exists():
        raise ValueError(f"output directory already exists: {args.output_dir}")
    return args


def load_replay_policy(checkpoint: Path, config: object, device: str) -> object:
    """Load a fresh checkpoint copy so one replay seed cannot affect another."""

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


def run_comparison(
    args: argparse.Namespace,
    *,
    runtime_loader: Callable[[argparse.Namespace], EvaluationRuntime] = load_evaluation_runtime,
    metadata_loader: Callable[[Path], CheckpointMetadata] = read_checkpoint_metadata,
    policy_loader: Callable[[Path, object, str], object] = load_replay_policy,
    single_evaluator: Callable[..., EvaluationRunResult] = run_evaluation,
) -> ComparisonRunResult:
    """Evaluate independent checkpoints and retain failures in the final report."""

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    comparisons: list[CheckpointComparison] = []
    failures: list[dict[str, str]] = []
    claimed_checkpoint_ids: dict[str, Path] = {}

    for checkpoint_arg in args.checkpoints:
        checkpoint = Path(checkpoint_arg).expanduser()
        try:
            checkpoint = checkpoint.resolve()
            checkpoint_id = _checkpoint_id(checkpoint)
            previous_checkpoint = claimed_checkpoint_ids.get(checkpoint_id)
            if previous_checkpoint is not None:
                raise ValueError(
                    "derived checkpoint ID collision "
                    f"for {checkpoint_id!r}: {previous_checkpoint} and {checkpoint}"
                )
            claimed_checkpoint_ids[checkpoint_id] = checkpoint
            checkpoint_output = output_dir / "checkpoints" / checkpoint_id
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
            runtime = runtime_loader(checkpoint_args)
            metadata = metadata_loader(runtime.checkpoint)
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
        except Exception as error:  # noqa: BLE001 - one bad checkpoint must not hide other results.
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
        evaluation_parameters={
            "dataset_repo_id": args.dataset_repo_id
            or f"local/{Path(args.dataset_root).expanduser().resolve().name}",
            "requested_dataset_repo_id": args.dataset_repo_id,
            "device": str(args.device),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "video_backend": args.video_backend,
        },
    )
    return ComparisonRunResult(tuple(comparisons), tuple(failures), output_paths)


def main(argv: Sequence[str] | None = None) -> int:
    result = run_comparison(parse_args(argv))
    return 1 if result.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
