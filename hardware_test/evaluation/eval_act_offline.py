#!/usr/bin/env python
"""Offline ACT action-chunk evaluation on fixed LeRobot dataset episodes.

The pure metric helpers in this module are intentionally independent of LeRobot's
runtime imports so they can be tested without loading a policy or decoding video.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ACTION_DIM = 7
XYZ_SLICE = slice(0, 3)
ROTATION_SLICE = slice(3, 6)
GRIPPER_INDEX = 6
EXPECTED_ACTION_NAMES = (
    "delta_ee_pose.x",
    "delta_ee_pose.y",
    "delta_ee_pose.z",
    "delta_ee_pose.rx",
    "delta_ee_pose.ry",
    "delta_ee_pose.rz",
    "gripper_cmd_bin",
)
EXPECTED_STATE_NAMES = (
    "joint_1.pos",
    "joint_2.pos",
    "joint_3.pos",
    "joint_4.pos",
    "joint_5.pos",
    "joint_6.pos",
    "joint_7.pos",
    "gripper_width_norm",
)
ACTION_UNITS = ("m", "m", "m", "rad", "rad", "rad", "binary")
EXPECTED_INPUT_SHAPES = {
    "observation.state": (8,),
    "observation.images.l515": (3, 540, 960),
}
EXPECTED_DATASET_IMAGE_SHAPE = (540, 960, 3)


@dataclass(frozen=True)
class ScalarSummary:
    """Episode-balanced mean and sample standard deviation."""

    mean: float
    std: float


@dataclass(frozen=True)
class EpisodeMetrics:
    """All metrics for one episode after pairing every valid chunk horizon."""

    episode_id: int
    num_frames: int
    valid_pairs: int
    eval_normalized_l1: float
    translation_error_mm: float
    rotation_error_deg: float
    ade_mm: float
    fde_mm: float
    xyz_rmse_mm: float
    action_mae: np.ndarray
    gripper_mae: float
    gripper_accuracy: float
    gripper_gt_open_rate: float
    horizon_translation_error_mm: np.ndarray
    horizon_trajectory_error_mm: np.ndarray
    horizon_valid_pairs: np.ndarray


@dataclass(frozen=True)
class EvaluationSummary:
    """Unweighted summary across episode-level metrics."""

    metrics: dict[str, ScalarSummary]
    action_mae_mean: np.ndarray
    action_mae_std: np.ndarray
    horizon_translation_mean_mm: np.ndarray
    horizon_translation_std_mm: np.ndarray
    horizon_trajectory_mean_mm: np.ndarray
    horizon_trajectory_std_mm: np.ndarray
    horizon_episode_count: np.ndarray
    horizon_valid_pairs: np.ndarray


@dataclass(frozen=True)
class EvaluationRuntime:
    """Loaded local components needed for one checkpoint evaluation."""

    checkpoint: Path
    dataset_root: Path
    device: str
    config: object
    policy: object
    preprocessor: object
    postprocessor: object
    dataset: object
    fps: float
    action_names: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationRunResult:
    """Programmatic result returned by the CLI orchestration."""

    episode_metrics: list[EpisodeMetrics]
    summary: EvaluationSummary
    output_paths: dict[str, Path]


def _require_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _validate_chunks(
    pred_physical: np.ndarray,
    gt_physical: np.ndarray,
    pred_normalized: np.ndarray,
    gt_normalized: np.ndarray,
    action_is_pad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = {
        "pred_physical": np.asarray(pred_physical, dtype=np.float64),
        "gt_physical": np.asarray(gt_physical, dtype=np.float64),
        "pred_normalized": np.asarray(pred_normalized, dtype=np.float64),
        "gt_normalized": np.asarray(gt_normalized, dtype=np.float64),
    }
    expected_shape = arrays["pred_physical"].shape
    if len(expected_shape) != 3 or expected_shape[-1] != ACTION_DIM:
        raise ValueError(
            f"action chunks must have shape (frames, horizon, 7); got pred_physical={expected_shape}"
        )
    if expected_shape[0] == 0 or expected_shape[1] == 0:
        raise ValueError("action chunks must contain at least one frame and one horizon")

    for name, values in arrays.items():
        if values.shape != expected_shape:
            raise ValueError(f"{name} has shape {values.shape}, expected {expected_shape}")
        _require_finite(name, values)

    pad = np.asarray(action_is_pad)
    if pad.dtype != np.bool_:
        raise ValueError(f"action_is_pad must be boolean, got dtype={pad.dtype}")
    if pad.shape != expected_shape[:2]:
        raise ValueError(f"action_is_pad has shape {pad.shape}, expected {expected_shape[:2]}")

    valid = ~pad
    if not valid[:, 0].all():
        raise ValueError("each evaluated frame must have a valid horizon 0 action")
    if np.any((~valid[:, :-1]) & valid[:, 1:]):
        raise ValueError("each action chunk must contain one contiguous valid prefix followed by padding")

    return (
        arrays["pred_physical"],
        arrays["gt_physical"],
        arrays["pred_normalized"],
        arrays["gt_normalized"],
        pad,
    )


def rpy_to_rotation_matrices(rpy: np.ndarray) -> np.ndarray:
    """Convert roll/pitch/yaw vectors to matrices using Rz(yaw) @ Ry(pitch) @ Rx(roll)."""

    values = np.asarray(rpy)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError(f"rpy must have shape (..., 3), got {values.shape}")
    _require_finite("rpy", values)

    roll = values[..., 0]
    pitch = values[..., 1]
    yaw = values[..., 2]
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)

    result = np.empty((*values.shape[:-1], 3, 3), dtype=np.result_type(values, np.float64))
    result[..., 0, 0] = cy * cp
    result[..., 0, 1] = cy * sp * sr - sy * cr
    result[..., 0, 2] = cy * sp * cr + sy * sr
    result[..., 1, 0] = sy * cp
    result[..., 1, 1] = sy * sp * sr + cy * cr
    result[..., 1, 2] = sy * sp * cr - cy * sr
    result[..., 2, 0] = -sp
    result[..., 2, 1] = cp * sr
    result[..., 2, 2] = cp * cr
    return result


def rotation_geodesic_degrees(pred_rpy: np.ndarray, gt_rpy: np.ndarray) -> np.ndarray:
    """Return the SO(3) geodesic angle between paired delta-RPY rotations."""

    pred = np.asarray(pred_rpy)
    gt = np.asarray(gt_rpy)
    if pred.shape != gt.shape:
        raise ValueError(f"rotation shapes differ: pred={pred.shape}, gt={gt.shape}")
    pred_matrix = rpy_to_rotation_matrices(pred)
    gt_matrix = rpy_to_rotation_matrices(gt)
    relative = pred_matrix @ np.swapaxes(gt_matrix, -1, -2)
    trace = np.trace(relative, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def compute_episode_metrics(
    episode_id: int,
    pred_physical: np.ndarray,
    gt_physical: np.ndarray,
    pred_normalized: np.ndarray,
    gt_normalized: np.ndarray,
    action_is_pad: np.ndarray,
) -> EpisodeMetrics:
    """Compute paired metrics for one episode, excluding every padded action."""

    pred, gt, pred_norm, gt_norm, pad = _validate_chunks(
        pred_physical,
        gt_physical,
        pred_normalized,
        gt_normalized,
        action_is_pad,
    )
    valid = ~pad
    valid_pairs = int(valid.sum())
    valid_gt_gripper = gt[..., GRIPPER_INDEX][valid]
    if not np.isin(valid_gt_gripper, (0.0, 1.0)).all():
        invalid_values = np.unique(valid_gt_gripper[~np.isin(valid_gt_gripper, (0.0, 1.0))])
        raise ValueError(
            f"physical expert gripper actions must be binary 0/1; found {invalid_values[:5].tolist()}"
        )

    normalized_abs_error = np.abs(pred_norm - gt_norm)
    physical_abs_error = np.abs(pred - gt)
    xyz_delta = pred[..., XYZ_SLICE] - gt[..., XYZ_SLICE]
    translation_error_m = np.linalg.norm(xyz_delta, axis=-1)
    rotation_error_deg = rotation_geodesic_degrees(pred[..., ROTATION_SLICE], gt[..., ROTATION_SLICE])

    pred_relative_position = np.cumsum(pred[..., XYZ_SLICE], axis=1)
    gt_relative_position = np.cumsum(gt[..., XYZ_SLICE], axis=1)
    trajectory_error_m = np.linalg.norm(pred_relative_position - gt_relative_position, axis=-1)

    valid_lengths = valid.sum(axis=1)
    chunk_ade_m = (trajectory_error_m * valid).sum(axis=1) / valid_lengths
    chunk_fde_m = trajectory_error_m[np.arange(pred.shape[0]), valid_lengths - 1]

    horizon_count = valid.sum(axis=0).astype(np.int64)
    horizon_translation_mm = np.full(pred.shape[1], np.nan, dtype=np.float64)
    horizon_trajectory_mm = np.full(pred.shape[1], np.nan, dtype=np.float64)
    supported = horizon_count > 0
    horizon_translation_mm[supported] = (
        (translation_error_m * valid).sum(axis=0)[supported] / horizon_count[supported] * 1000.0
    )
    horizon_trajectory_mm[supported] = (
        (trajectory_error_m * valid).sum(axis=0)[supported] / horizon_count[supported] * 1000.0
    )

    pred_gripper_open = pred[..., GRIPPER_INDEX] >= 0.5
    gt_gripper_open = gt[..., GRIPPER_INDEX] >= 0.5
    action_mae = physical_abs_error[valid].mean(axis=0)

    return EpisodeMetrics(
        episode_id=int(episode_id),
        num_frames=int(pred.shape[0]),
        valid_pairs=valid_pairs,
        eval_normalized_l1=float(normalized_abs_error[valid].mean()),
        translation_error_mm=float(translation_error_m[valid].mean() * 1000.0),
        rotation_error_deg=float(rotation_error_deg[valid].mean()),
        ade_mm=float(chunk_ade_m.mean() * 1000.0),
        fde_mm=float(chunk_fde_m.mean() * 1000.0),
        xyz_rmse_mm=float(np.sqrt(np.mean(np.square(translation_error_m[valid]))) * 1000.0),
        action_mae=action_mae,
        gripper_mae=float(action_mae[GRIPPER_INDEX]),
        gripper_accuracy=float((pred_gripper_open[valid] == gt_gripper_open[valid]).mean()),
        gripper_gt_open_rate=float(gt_gripper_open[valid].mean()),
        horizon_translation_error_mm=horizon_translation_mm,
        horizon_trajectory_error_mm=horizon_trajectory_mm,
        horizon_valid_pairs=horizon_count,
    )


def _mean_std(values: np.ndarray) -> ScalarSummary:
    _require_finite("episode metric", values)
    return ScalarSummary(
        mean=float(values.mean()),
        std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    )


def _horizon_mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon = values.shape[1]
    means = np.full(horizon, np.nan, dtype=np.float64)
    stds = np.full(horizon, np.nan, dtype=np.float64)
    episode_counts = np.isfinite(values).sum(axis=0).astype(np.int64)
    for index, count in enumerate(episode_counts):
        if count == 0:
            continue
        supported = values[:, index][np.isfinite(values[:, index])]
        means[index] = supported.mean()
        stds[index] = supported.std(ddof=1) if count > 1 else 0.0
    return means, stds, episode_counts


def summarize_episode_metrics(episodes: list[EpisodeMetrics]) -> EvaluationSummary:
    """Aggregate episode means without allowing longer episodes to dominate."""

    if not episodes:
        raise ValueError("at least one episode is required for summary")
    horizon = len(episodes[0].horizon_translation_error_mm)
    if any(len(item.horizon_translation_error_mm) != horizon for item in episodes):
        raise ValueError("all episodes must use the same action horizon")

    scalar_names = (
        "eval_normalized_l1",
        "translation_error_mm",
        "rotation_error_deg",
        "ade_mm",
        "fde_mm",
        "xyz_rmse_mm",
        "gripper_mae",
        "gripper_accuracy",
        "gripper_gt_open_rate",
    )
    scalar_metrics = {
        name: _mean_std(np.asarray([getattr(item, name) for item in episodes], dtype=np.float64))
        for name in scalar_names
    }

    action_mae = np.stack([item.action_mae for item in episodes])
    action_mae_mean = action_mae.mean(axis=0)
    action_mae_std = action_mae.std(axis=0, ddof=1) if len(episodes) > 1 else np.zeros(ACTION_DIM)

    horizon_translation = np.stack([item.horizon_translation_error_mm for item in episodes])
    horizon_trajectory = np.stack([item.horizon_trajectory_error_mm for item in episodes])
    translation_mean, translation_std, episode_count = _horizon_mean_std(horizon_translation)
    trajectory_mean, trajectory_std, trajectory_episode_count = _horizon_mean_std(horizon_trajectory)
    if not np.array_equal(episode_count, trajectory_episode_count):
        raise ValueError("translation and trajectory horizon support differ")

    return EvaluationSummary(
        metrics=scalar_metrics,
        action_mae_mean=action_mae_mean,
        action_mae_std=action_mae_std,
        horizon_translation_mean_mm=translation_mean,
        horizon_translation_std_mm=translation_std,
        horizon_trajectory_mean_mm=trajectory_mean,
        horizon_trajectory_std_mm=trajectory_std,
        horizon_episode_count=episode_count,
        horizon_valid_pairs=np.stack([item.horizon_valid_pairs for item in episodes]).sum(axis=0),
    )


def validate_act_dataset_contract(config: object, dataset_features: dict[str, dict]) -> None:
    """Fail closed unless checkpoint and dataset use the verified Franka ACT schema."""

    policy_type = getattr(config, "type", None)
    if policy_type != "act":
        raise ValueError(f"expected an ACT checkpoint, got {policy_type!r}")

    input_features = getattr(config, "input_features", {})
    if set(input_features) != set(EXPECTED_INPUT_SHAPES):
        raise ValueError(
            "checkpoint input feature keys differ from the deployed Franka ACT contract: "
            f"got {sorted(input_features)}, expected {sorted(EXPECTED_INPUT_SHAPES)}"
        )
    for key, expected_shape in EXPECTED_INPUT_SHAPES.items():
        feature = input_features[key]
        actual_shape = tuple(getattr(feature, "shape", ()))
        if actual_shape != expected_shape:
            raise ValueError(f"checkpoint feature {key} has shape {actual_shape}, expected {expected_shape}")

    output_features = getattr(config, "output_features", {})
    action_feature = output_features.get("action")
    action_shape = tuple(getattr(action_feature, "shape", ())) if action_feature is not None else None
    if set(output_features) != {"action"} or action_shape != (ACTION_DIM,):
        raise ValueError(f"checkpoint action has shape {action_shape}, expected ({ACTION_DIM},)")

    chunk_size = getattr(config, "chunk_size", None)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"checkpoint chunk_size must be a positive integer, got {chunk_size!r}")

    required_dataset_shapes = {
        "action": (ACTION_DIM,),
        "observation.state": (8,),
        "observation.images.l515": EXPECTED_DATASET_IMAGE_SHAPE,
    }
    for key, expected_shape in required_dataset_shapes.items():
        feature = dataset_features.get(key)
        if feature is None:
            raise ValueError(f"dataset is missing required feature {key!r}")
        actual_shape = tuple(feature.get("shape", ()))
        if actual_shape != expected_shape:
            raise ValueError(f"dataset feature {key} has shape {actual_shape}, expected {expected_shape}")

    action_names = tuple(dataset_features["action"].get("names", ()))
    if action_names != EXPECTED_ACTION_NAMES:
        raise ValueError(
            "dataset action names do not match the verified delta-EEF contract: "
            f"got {action_names}, expected {EXPECTED_ACTION_NAMES}"
        )
    state_names = tuple(dataset_features["observation.state"].get("names", ()))
    if state_names != EXPECTED_STATE_NAMES:
        raise ValueError(
            "dataset observation.state names do not match the trained checkpoint contract: "
            f"got {state_names}, expected {EXPECTED_STATE_NAMES}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one ACT checkpoint on fixed LeRobot episodes using complete action chunks. "
            "Run the command once per training-data-size checkpoint with the same episode list."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint pretrained_model/")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Local LeRobot dataset root")
    parser.add_argument("--episodes", type=int, nargs="+", required=True, help="Fixed test episode IDs")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSV/JSON/PNG outputs")
    parser.add_argument("--device", default="cuda", help="PyTorch device, for example cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="Frames evaluated per inference batch")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument(
        "--video-backend",
        choices=("torchcodec", "pyav"),
        default=None,
        help="Optional LeRobot video decoder override",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help="Local dataset identifier; defaults to local/<dataset directory name>",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_arg_parser().parse_args(argv)
    if any(episode < 0 for episode in args.episodes):
        raise ValueError("episode IDs must be non-negative")
    if len(set(args.episodes)) != len(args.episodes):
        raise ValueError("episode IDs must be unique")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.num_workers < 0:
        raise ValueError("num workers must be non-negative")
    if not str(args.device).strip():
        raise ValueError("device must not be empty")
    return args


def _metric_unit(name: str) -> str:
    if name in {"translation_error_mm", "ade_mm", "fde_mm", "xyz_rmse_mm"}:
        return "mm"
    if name == "rotation_error_deg":
        return "deg"
    if name == "eval_normalized_l1":
        return "normalized_action"
    if name == "gripper_accuracy" or name == "gripper_gt_open_rate":
        return "fraction"
    return "native_action_unit"


def _action_column_name(index: int, action_name: str) -> str:
    safe_name = action_name.replace(".", "_").replace("/", "_")
    return f"mae_dim_{index}_{safe_name}"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _csv_text(fieldnames: list[str], rows: list[dict[str, object]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _plot_horizon_panel(
    draw: object,
    bounds: tuple[int, int, int, int],
    mean: np.ndarray,
    std: np.ndarray,
    title: str,
) -> None:
    left, top, right, bottom = bounds
    finite = np.isfinite(mean) & np.isfinite(std)
    if not finite.any():
        raise ValueError(f"cannot plot {title}: no supported horizons")
    indices = np.arange(len(mean))[finite]
    upper = np.maximum(mean[finite] + std[finite], 0.0)
    maximum = max(float(upper.max()) * 1.08, 1e-6)

    draw.rectangle(bounds, outline="black", width=2)
    draw.text((left, top - 24), title, fill="black")
    for tick in range(6):
        y_value = maximum * tick / 5
        y = bottom - round((bottom - top) * tick / 5)
        draw.line((left, y, right, y), fill="#dddddd", width=1)
        draw.text((left - 62, y - 7), f"{y_value:.2f}", fill="black")

    horizon_denominator = max(len(mean) - 1, 1)

    def x_coordinate(index: int) -> int:
        return left + round((right - left) * index / horizon_denominator)

    def y_coordinate(value: float) -> int:
        return bottom - round((bottom - top) * max(value, 0.0) / maximum)

    for tick_index in np.linspace(0, len(mean) - 1, num=min(6, len(mean)), dtype=int):
        x = x_coordinate(int(tick_index))
        draw.line((x, bottom, x, bottom + 5), fill="black", width=1)
        draw.text((x - 8, bottom + 8), str(int(tick_index)), fill="black")

    upper_points = [
        (x_coordinate(int(index)), y_coordinate(float(value)))
        for index, value in zip(indices, upper, strict=True)
    ]
    lower = np.maximum(mean[finite] - std[finite], 0.0)
    lower_points = [
        (x_coordinate(int(index)), y_coordinate(float(value)))
        for index, value in zip(indices[::-1], lower[::-1], strict=True)
    ]
    if len(upper_points) >= 2:
        draw.polygon([*upper_points, *lower_points], fill="#cfe8ff")
    line_points = [
        (x_coordinate(int(index)), y_coordinate(float(value)))
        for index, value in zip(indices, mean[finite], strict=True)
    ]
    if len(line_points) == 1:
        x, y = line_points[0]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="#1368aa")
    else:
        draw.line(line_points, fill="#1368aa", width=3)
    draw.text((left + 8, top + 8), "error (mm)", fill="#555555")
    draw.text((right - 150, top + 8), "mean +/- episode std", fill="#1368aa")
    draw.text(((left + right) // 2 - 40, bottom + 28), "horizon", fill="black")


def _save_horizon_plot(path: Path, summary: EvaluationSummary) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:  # pragma: no cover - Pillow is a LeRobot dependency.
        raise RuntimeError("Pillow is required to write horizon_error.png") from error

    image = Image.new("RGB", (1200, 900), "white")
    draw = ImageDraw.Draw(image)
    _plot_horizon_panel(
        draw,
        (100, 70, 1150, 380),
        summary.horizon_translation_mean_mm,
        summary.horizon_translation_std_mm,
        "Delta XYZ translation error by horizon",
    )
    _plot_horizon_panel(
        draw,
        (100, 510, 1150, 820),
        summary.horizon_trajectory_mean_mm,
        summary.horizon_trajectory_std_mm,
        "Cumulative relative-position trajectory error by horizon",
    )

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


def save_evaluation_outputs(
    *,
    output_dir: Path,
    episode_metrics: list[EpisodeMetrics],
    summary: EvaluationSummary,
    checkpoint: str,
    dataset_root: str,
    requested_episode_ids: Sequence[int],
    fps: float,
    chunk_size: int,
    device: str,
    action_names: Sequence[str],
) -> dict[str, Path]:
    """Write all required outputs after a successful complete evaluation."""

    output_dir = Path(output_dir)
    if len(action_names) != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM} action names, got {len(action_names)}")
    if len(episode_metrics) != len(requested_episode_ids):
        raise ValueError("episode result count differs from requested episode count")
    output_dir.mkdir(parents=True, exist_ok=True)

    scalar_fields = [
        "episode_id",
        "num_frames",
        "valid_pairs",
        "eval_normalized_l1",
        "translation_error_mm",
        "rotation_error_deg",
        "ade_mm",
        "fde_mm",
        "xyz_rmse_mm",
        "gripper_mae",
        "gripper_accuracy",
        "gripper_gt_open_rate",
    ]
    action_fields = [_action_column_name(index, name) for index, name in enumerate(action_names)]
    episode_rows: list[dict[str, object]] = []
    for metrics in episode_metrics:
        row = {field: getattr(metrics, field) for field in scalar_fields}
        row.update({field: float(metrics.action_mae[index]) for index, field in enumerate(action_fields)})
        episode_rows.append(row)

    per_episode_path = output_dir / "per_episode_metrics.csv"
    _atomic_write_text(per_episode_path, _csv_text([*scalar_fields, *action_fields], episode_rows))

    summary_payload = {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "episodes": [int(item) for item in requested_episode_ids],
        "num_episodes": len(episode_metrics),
        "fps": float(fps),
        "chunk_size": int(chunk_size),
        "device": device,
        "aggregation": "episode_mean_then_unweighted_mean_std",
        "std_definition": "sample_standard_deviation_ddof_1; zero_for_one_episode",
        "action_names": list(action_names),
        "action_units": list(ACTION_UNITS),
        "action_semantics": {
            "xyz": "base-frame delta translation per control frame",
            "rotation": "delta roll/pitch/yaw; Rz(yaw) @ Ry(pitch) @ Rx(roll)",
            "gripper": "binary command; 1=open, 0=closed",
            "ade_fde": "relative commanded EEF trajectory from separate XYZ cumulative sums",
        },
        "metric_definitions": {
            "eval_normalized_l1": "mean absolute error over every valid horizon and all 7 normalized action dimensions",
            "translation_error_mm": "mean(||pred_delta_xyz - gt_delta_xyz||_2) * 1000",
            "rotation_error_deg": "SO(3) geodesic angle between paired delta-RPY rotation matrices",
            "xyz_rmse_mm": "sqrt(mean(||pred_delta_xyz - gt_delta_xyz||_2^2)) * 1000",
            "ade_mm": "episode mean of per-chunk mean cumulative-delta-XYZ trajectory error",
            "fde_mm": "episode mean of each chunk's last-valid cumulative-delta-XYZ trajectory error",
        },
        "metrics": {
            name: {"mean": value.mean, "std": value.std, "unit": _metric_unit(name)}
            for name, value in summary.metrics.items()
        },
        "per_action_dimension_mae": {
            name: {
                "mean": float(summary.action_mae_mean[index]),
                "std": float(summary.action_mae_std[index]),
                "unit": ACTION_UNITS[index],
            }
            for index, name in enumerate(action_names)
        },
        "notes": [
            "This is teacher-forced offline action prediction on recorded expert observations.",
            "ADE/FDE are relative commanded trajectories because absolute measured EEF poses are not observations.",
            "Gripper accuracy should be interpreted with the reported ground-truth open rate.",
        ],
    }
    summary_path = output_dir / "summary_metrics.json"
    _atomic_write_text(
        summary_path,
        json.dumps(summary_payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )

    horizon_fields = [
        "horizon",
        "time_s",
        "translation_error_mean_mm",
        "translation_error_std_mm",
        "trajectory_error_mean_mm",
        "trajectory_error_std_mm",
        "episode_count",
        "valid_pairs",
    ]
    horizon_rows: list[dict[str, object]] = []
    for horizon in range(chunk_size):
        values = (
            summary.horizon_translation_mean_mm[horizon],
            summary.horizon_translation_std_mm[horizon],
            summary.horizon_trajectory_mean_mm[horizon],
            summary.horizon_trajectory_std_mm[horizon],
        )
        horizon_rows.append(
            {
                "horizon": horizon,
                "time_s": horizon / fps,
                "translation_error_mean_mm": "" if not np.isfinite(values[0]) else float(values[0]),
                "translation_error_std_mm": "" if not np.isfinite(values[1]) else float(values[1]),
                "trajectory_error_mean_mm": "" if not np.isfinite(values[2]) else float(values[2]),
                "trajectory_error_std_mm": "" if not np.isfinite(values[3]) else float(values[3]),
                "episode_count": int(summary.horizon_episode_count[horizon]),
                "valid_pairs": int(summary.horizon_valid_pairs[horizon]),
            }
        )
    horizon_path = output_dir / "horizon_translation_error.csv"
    _atomic_write_text(horizon_path, _csv_text(horizon_fields, horizon_rows))

    plot_path = output_dir / "horizon_error.png"
    _save_horizon_plot(plot_path, summary)
    return {
        "per_episode": per_episode_path,
        "summary": summary_path,
        "horizon": horizon_path,
        "plot": plot_path,
    }


def _to_numpy(name: str, value: object) -> np.ndarray:
    result = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    _require_finite(name, result)
    return result


def _reset_if_supported(component: object) -> None:
    reset = getattr(component, "reset", None)
    if callable(reset):
        reset()


def _episode_frame_indices(dataset: object, episode_ids: Sequence[int]) -> dict[int, np.ndarray]:
    try:
        episode_column = dataset.hf_dataset.data.column("episode_index").to_numpy()
    except AttributeError as error:
        raise ValueError("dataset does not expose its episode_index column") from error
    values = np.asarray(episode_column, dtype=np.int64)
    if values.shape != (len(dataset),):
        raise ValueError(f"dataset episode_index column has shape {values.shape}, expected ({len(dataset)},)")

    result: dict[int, np.ndarray] = {}
    for episode_id in episode_ids:
        indices = np.flatnonzero(values == episode_id)
        if len(indices) == 0:
            raise ValueError(f"requested episode {episode_id} has no frames in the loaded dataset")
        result[int(episode_id)] = indices
    return result


def evaluate_policy_on_dataset(
    *,
    policy: object,
    preprocessor: object,
    postprocessor: object,
    dataset: object,
    episode_ids: Sequence[int],
    input_feature_keys: Sequence[str],
    chunk_size: int,
    batch_size: int,
    num_workers: int,
) -> list[EpisodeMetrics]:
    """Predict complete chunks and compute one independently aggregated result per episode."""

    if not episode_ids:
        raise ValueError("at least one episode ID is required")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be non-negative")

    eval_method = getattr(policy, "eval", None)
    reset_method = getattr(policy, "reset", None)
    predict_method = getattr(policy, "predict_action_chunk", None)
    if not callable(eval_method) or not callable(reset_method) or not callable(predict_method):
        raise TypeError("policy must provide eval(), reset(), and predict_action_chunk()")
    if not callable(preprocessor) or not callable(postprocessor):
        raise TypeError("preprocessor and postprocessor must be callable")

    policy.eval()
    indices_by_episode = _episode_frame_indices(dataset, episode_ids)
    results: list[EpisodeMetrics] = []

    with torch.inference_mode():
        for episode_id in episode_ids:
            _reset_if_supported(policy)
            _reset_if_supported(preprocessor)
            _reset_if_supported(postprocessor)

            loader = DataLoader(
                Subset(dataset, indices_by_episode[int(episode_id)].tolist()),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
            )
            pred_physical_batches: list[np.ndarray] = []
            gt_physical_batches: list[np.ndarray] = []
            pred_normalized_batches: list[np.ndarray] = []
            gt_normalized_batches: list[np.ndarray] = []
            padding_batches: list[np.ndarray] = []

            for batch_index, raw_batch in enumerate(loader):
                if "action" not in raw_batch or "action_is_pad" not in raw_batch:
                    raise ValueError("dataset batch must contain action and action_is_pad")
                gt_physical = _to_numpy(
                    f"episode {episode_id} batch {batch_index} physical expert action",
                    raw_batch["action"],
                ).copy()
                action_is_pad = _to_numpy(
                    f"episode {episode_id} batch {batch_index} padding mask",
                    raw_batch["action_is_pad"],
                )
                if action_is_pad.dtype != np.bool_:
                    raise ValueError(
                        f"action_is_pad must be boolean, got dtype={action_is_pad.dtype} "
                        f"in episode {episode_id} batch {batch_index}"
                    )

                processed_batch = preprocessor(raw_batch)
                if not isinstance(processed_batch, dict):
                    raise TypeError("preprocessor must return a batch dictionary")
                if "action" not in processed_batch:
                    raise ValueError("preprocessor output is missing normalized action")
                missing_inputs = [key for key in input_feature_keys if key not in processed_batch]
                if missing_inputs:
                    raise ValueError(f"preprocessor output is missing policy inputs: {missing_inputs}")

                gt_normalized = _to_numpy(
                    f"episode {episode_id} batch {batch_index} normalized expert action",
                    processed_batch["action"],
                ).copy()
                policy_batch = {key: processed_batch[key] for key in input_feature_keys}
                pred_normalized_tensor = policy.predict_action_chunk(policy_batch)
                if not isinstance(pred_normalized_tensor, torch.Tensor):
                    raise TypeError("predict_action_chunk() must return a torch.Tensor")
                pred_normalized = _to_numpy(
                    f"episode {episode_id} batch {batch_index} normalized prediction",
                    pred_normalized_tensor,
                ).copy()
                if pred_normalized.shape != gt_normalized.shape:
                    raise ValueError(
                        "predicted and expert normalized chunk shapes differ: "
                        f"pred={pred_normalized.shape}, gt={gt_normalized.shape}"
                    )
                expected_shape = (pred_normalized.shape[0], chunk_size, ACTION_DIM)
                if pred_normalized.shape != expected_shape:
                    raise ValueError(
                        f"predict_action_chunk() returned {pred_normalized.shape}, expected {expected_shape}"
                    )

                pred_physical_tensor = postprocessor(pred_normalized_tensor)
                pred_physical = _to_numpy(
                    f"episode {episode_id} batch {batch_index} physical prediction",
                    pred_physical_tensor,
                ).copy()
                if pred_physical.shape != gt_physical.shape:
                    raise ValueError(
                        "postprocessed prediction and physical expert chunk shapes differ: "
                        f"pred={pred_physical.shape}, gt={gt_physical.shape}"
                    )
                if action_is_pad.shape != pred_physical.shape[:2]:
                    raise ValueError(
                        f"padding mask has shape {action_is_pad.shape}, expected {pred_physical.shape[:2]}"
                    )

                pred_physical_batches.append(pred_physical)
                gt_physical_batches.append(gt_physical)
                pred_normalized_batches.append(pred_normalized)
                gt_normalized_batches.append(gt_normalized)
                padding_batches.append(action_is_pad)

            if not pred_physical_batches:
                raise ValueError(f"episode {episode_id} produced no evaluation batches")
            results.append(
                compute_episode_metrics(
                    episode_id=int(episode_id),
                    pred_physical=np.concatenate(pred_physical_batches, axis=0),
                    gt_physical=np.concatenate(gt_physical_batches, axis=0),
                    pred_normalized=np.concatenate(pred_normalized_batches, axis=0),
                    gt_normalized=np.concatenate(gt_normalized_batches, axis=0),
                    action_is_pad=np.concatenate(padding_batches, axis=0),
                )
            )

    return results


def validate_runtime_paths(
    checkpoint: Path,
    dataset_root: Path,
    output_dir: Path,
    device: str,
) -> tuple[Path, Path]:
    """Validate local-only inputs before loading a large policy or decoding video."""

    checkpoint = Path(checkpoint).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint}")
    required_checkpoint_files = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    missing_checkpoint_files = [
        name for name in required_checkpoint_files if not (checkpoint / name).is_file()
    ]
    if missing_checkpoint_files:
        raise ValueError(
            "checkpoint must point to a complete pretrained_model directory; "
            f"missing {missing_checkpoint_files} in {checkpoint}"
        )
    if not (dataset_root / "meta" / "info.json").is_file():
        raise ValueError(f"dataset root is missing meta/info.json: {dataset_root}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path exists and is not a directory: {output_dir}")

    try:
        torch_device = torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"invalid PyTorch device {device!r}") from error
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is not available: {device}")
    return checkpoint, dataset_root


def format_terminal_summary(
    checkpoint: str | Path,
    episode_count: int,
    summary: EvaluationSummary,
) -> str:
    """Format the concise human-facing result while preserving JSON as the full record."""

    normalized = summary.metrics["eval_normalized_l1"]
    translation = summary.metrics["translation_error_mm"]
    rotation = summary.metrics["rotation_error_deg"]
    ade = summary.metrics["ade_mm"]
    fde = summary.metrics["fde_mm"]
    return "\n".join(
        (
            f"Checkpoint: {checkpoint}",
            f"Episodes: {episode_count}",
            f"Normalized L1: {normalized.mean:.6f} ± {normalized.std:.6f}",
            f"Translation error: {translation.mean:.2f} ± {translation.std:.2f} mm",
            f"Rotation error: {rotation.mean:.2f} ± {rotation.std:.2f} deg",
            f"ADE: {ade.mean:.2f} ± {ade.std:.2f} mm",
            f"FDE: {fde.mean:.2f} ± {fde.std:.2f} mm",
        )
    )


def load_evaluation_runtime(args: argparse.Namespace) -> EvaluationRuntime:
    """Load one local ACT checkpoint, its saved processors, and selected dataset episodes."""

    checkpoint, dataset_root = validate_runtime_paths(
        args.checkpoint,
        args.dataset_root,
        args.output_dir,
        args.device,
    )
    repo_id = args.dataset_repo_id or f"local/{dataset_root.name}"

    from lerobot import policies as lerobot_policies
    from lerobot.configs import PreTrainedConfig
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root)
    validate_act_dataset_contract(config, metadata.features)

    invalid_episode_ids = [
        episode_id for episode_id in args.episodes if episode_id < 0 or episode_id >= metadata.total_episodes
    ]
    if invalid_episode_ids:
        raise ValueError(
            f"episode IDs {invalid_episode_ids} are outside dataset range [0, {metadata.total_episodes - 1}]"
        )

    config.device = args.device
    policy = (
        lerobot_policies.get_policy_class(config.type)
        .from_pretrained(
            checkpoint,
            config=config,
            local_files_only=True,
            strict=True,
        )
        .to(args.device)
        .eval()
    )
    preprocessor, postprocessor = lerobot_policies.make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    chunk_size = int(config.chunk_size)
    delta_timestamps = {"action": [horizon / metadata.fps for horizon in range(chunk_size)]}
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        episodes=list(args.episodes),
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
        return_uint8=False,
    )
    validate_act_dataset_contract(config, dataset.features)
    return EvaluationRuntime(
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        device=args.device,
        config=config,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        fps=float(dataset.fps),
        action_names=EXPECTED_ACTION_NAMES,
    )


def run_evaluation(
    args: argparse.Namespace,
    *,
    runtime: EvaluationRuntime | None = None,
) -> EvaluationRunResult:
    """Run inference, aggregate by episode, persist outputs, and print a concise summary."""

    runtime = load_evaluation_runtime(args) if runtime is None else runtime
    chunk_size = int(runtime.config.chunk_size)
    input_feature_keys = tuple(runtime.config.input_features)
    episode_metrics = evaluate_policy_on_dataset(
        policy=runtime.policy,
        preprocessor=runtime.preprocessor,
        postprocessor=runtime.postprocessor,
        dataset=runtime.dataset,
        episode_ids=args.episodes,
        input_feature_keys=input_feature_keys,
        chunk_size=chunk_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    summary = summarize_episode_metrics(episode_metrics)
    output_paths = save_evaluation_outputs(
        output_dir=args.output_dir,
        episode_metrics=episode_metrics,
        summary=summary,
        checkpoint=str(runtime.checkpoint),
        dataset_root=str(runtime.dataset_root),
        requested_episode_ids=args.episodes,
        fps=runtime.fps,
        chunk_size=chunk_size,
        device=runtime.device,
        action_names=runtime.action_names,
    )
    print(format_terminal_summary(runtime.checkpoint, len(episode_metrics), summary))
    return EvaluationRunResult(
        episode_metrics=episode_metrics,
        summary=summary,
        output_paths=output_paths,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_evaluation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
