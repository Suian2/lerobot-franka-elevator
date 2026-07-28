from __future__ import annotations

import argparse
import filecmp
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hardware_test.franka.floor_condition import (
    FLOOR_CONDITION_FEATURE,
    FLOOR_CONDITION_KEY,
    NUM_ELEVATOR_FLOORS,
    encode_target_floor,
)
from lerobot.datasets import LeRobotDataset, add_features, recompute_stats
from lerobot.utils.constants import ACTION, OBS_STATE

PRESERVED_FRAME_KEYS = (
    OBS_STATE,
    ACTION,
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


@dataclass(frozen=True)
class MigrationReport:
    source_repo_id: str
    output_repo_id: str
    source_root: Path
    output_root: Path
    target_floor: int
    total_episodes: int
    total_frames: int
    total_videos: int


def _resolve_roots(source_root: str | Path, output_root: str | Path) -> tuple[Path, Path]:
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output == source or source in output.parents:
        raise ValueError("output root must be outside the source dataset")
    if not (source / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"source dataset does not exist at {source}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output root already exists: {output}")
    return source, output


def _create_owned_staging_root(output_root: Path) -> tuple[Path, Path]:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    owned_temp = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.migration-",
            dir=output_root.parent,
        )
    )
    return owned_temp, owned_temp / "dataset"


def _episode_lengths(dataset: LeRobotDataset) -> tuple[int, ...]:
    return tuple(
        int(episode["dataset_to_index"]) - int(episode["dataset_from_index"])
        for episode in dataset.meta.episodes
    )


def _numeric_batch(dataset: LeRobotDataset, keys: tuple[str, ...] | list[str]) -> dict[str, np.ndarray]:
    missing = set(keys) - set(dataset.hf_dataset.column_names)
    if missing:
        raise RuntimeError(f"dataset is missing required frame columns: {sorted(missing)}")
    return dataset.hf_dataset.select_columns(list(keys)).with_format("numpy")[:]


def _relative_video_files(root: Path) -> set[Path]:
    video_root = root / "videos"
    if not video_root.exists():
        return set()
    return {path.relative_to(root) for path in video_root.rglob("*") if path.is_file()}


def _verify_video_files(source_root: Path, output_root: Path) -> int:
    source_paths = _relative_video_files(source_root)
    output_paths = _relative_video_files(output_root)
    if source_paths != output_paths:
        missing = sorted(str(path) for path in source_paths - output_paths)
        extra = sorted(str(path) for path in output_paths - source_paths)
        raise RuntimeError(
            f"source and output relative video paths differ (missing={missing}, extra={extra})"
        )

    filecmp.clear_cache()
    for relative_path in sorted(source_paths):
        if not filecmp.cmp(
            source_root / relative_path,
            output_root / relative_path,
            shallow=False,
        ):
            raise RuntimeError(f"video differs byte-for-byte after migration: {relative_path}")
    return len(source_paths)


def _verify_feature_schema(source: LeRobotDataset, destination: LeRobotDataset) -> None:
    if FLOOR_CONDITION_KEY in source.meta.features:
        raise RuntimeError("source unexpectedly contains the floor condition")

    destination_features = dict(destination.meta.features)
    condition_feature = destination_features.pop(FLOOR_CONDITION_KEY, None)
    if condition_feature is None:
        raise RuntimeError(f"destination is missing {FLOOR_CONDITION_KEY}")
    if condition_feature.get("dtype") != "float32":
        raise RuntimeError(f"{FLOOR_CONDITION_KEY} must have dtype float32")
    if tuple(condition_feature.get("shape", ())) != tuple(FLOOR_CONDITION_FEATURE["shape"]):
        raise RuntimeError(f"{FLOOR_CONDITION_KEY} must have shape (5,)")
    if condition_feature.get("names") is not None:
        raise RuntimeError(f"{FLOOR_CONDITION_KEY} names must be null")
    if destination_features != source.meta.features:
        raise RuntimeError("migration changed an existing feature schema")


def verify_migration(
    source: LeRobotDataset,
    destination: LeRobotDataset,
    *,
    target_floor: int,
) -> int:
    expected_condition = encode_target_floor(target_floor)
    _verify_feature_schema(source, destination)

    if source.meta.total_episodes != destination.meta.total_episodes:
        raise RuntimeError("migration changed the episode count")
    if source.meta.total_frames != destination.meta.total_frames:
        raise RuntimeError("migration changed the frame count")
    if len(source) != len(destination):
        raise RuntimeError("migration changed the readable frame count")
    if _episode_lengths(source) != _episode_lengths(destination):
        raise RuntimeError("migration changed one or more episode lengths")

    source_values = _numeric_batch(source, PRESERVED_FRAME_KEYS)
    destination_values = _numeric_batch(destination, PRESERVED_FRAME_KEYS)
    for key in PRESERVED_FRAME_KEYS:
        source_value = np.asarray(source_values[key])
        destination_value = np.asarray(destination_values[key])
        if source_value.dtype != destination_value.dtype:
            raise RuntimeError(f"migration changed {key} dtype")
        if not np.array_equal(source_value, destination_value):
            raise RuntimeError(f"migration changed {key} values")

    conditions = np.asarray(_numeric_batch(destination, [FLOOR_CONDITION_KEY])[FLOOR_CONDITION_KEY])
    expected_shape = (destination.meta.total_frames, *expected_condition.shape)
    if conditions.dtype != np.float32:
        raise RuntimeError(f"{FLOOR_CONDITION_KEY} values must be float32")
    if conditions.shape != expected_shape:
        raise RuntimeError(
            f"{FLOOR_CONDITION_KEY} values must have shape {expected_shape}, got {conditions.shape}"
        )
    expected_conditions = np.repeat(
        expected_condition[None, :],
        repeats=destination.meta.total_frames,
        axis=0,
    )
    if not np.array_equal(conditions, expected_conditions):
        raise RuntimeError("destination floor conditions do not match the requested floor")
    if destination.meta.stats is None or FLOOR_CONDITION_KEY not in destination.meta.stats:
        raise RuntimeError(f"destination stats are missing {FLOOR_CONDITION_KEY}")

    return _verify_video_files(source.root, destination.root)


def migrate_dataset(
    *,
    source_repo_id: str,
    source_root: str | Path,
    output_repo_id: str,
    output_root: str | Path,
    target_floor: int = 1,
) -> MigrationReport:
    expected_condition = encode_target_floor(target_floor)
    resolved_source, resolved_output = _resolve_roots(source_root, output_root)
    source = LeRobotDataset(repo_id=source_repo_id, root=resolved_source)
    if FLOOR_CONDITION_KEY in source.meta.features:
        raise ValueError(f"source dataset already contains {FLOOR_CONDITION_KEY}")

    missing_keys = set(PRESERVED_FRAME_KEYS) - set(source.meta.features)
    if missing_keys:
        raise ValueError(f"source dataset is missing required features: {sorted(missing_keys)}")

    def floor_for_frame(frame: dict[str, Any], episode_index: int, frame_index: int) -> np.ndarray:
        del frame, episode_index, frame_index
        return expected_condition.copy()

    owned_temp, staging_root = _create_owned_staging_root(resolved_output)
    try:
        destination = add_features(
            source,
            features={
                FLOOR_CONDITION_KEY: (
                    floor_for_frame,
                    dict(FLOOR_CONDITION_FEATURE),
                )
            },
            output_dir=staging_root,
            repo_id=output_repo_id,
        )
        recompute_stats(destination, skip_image_video=True)
        total_videos = verify_migration(source, destination, target_floor=target_floor)
        report = MigrationReport(
            source_repo_id=source_repo_id,
            output_repo_id=output_repo_id,
            source_root=resolved_source,
            output_root=resolved_output,
            target_floor=target_floor,
            total_episodes=destination.meta.total_episodes,
            total_frames=destination.meta.total_frames,
            total_videos=total_videos,
        )

        if resolved_output.exists() or resolved_output.is_symlink():
            raise FileExistsError(f"output root appeared during migration: {resolved_output}")
        staging_root.rename(resolved_output)
        destination.root = resolved_output
        destination.meta.root = resolved_output
        return report
    finally:
        shutil.rmtree(owned_temp, ignore_errors=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a new LeRobot dataset with a canonical target-floor condition."
    )
    parser.add_argument("--source-repo-id", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-repo-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--target-floor",
        type=int,
        choices=range(1, NUM_ELEVATOR_FLOORS + 1),
        default=1,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = migrate_dataset(
        source_repo_id=args.source_repo_id,
        source_root=args.source_root,
        output_repo_id=args.output_repo_id,
        output_root=args.output_root,
        target_floor=args.target_floor,
    )
    print(
        "Migration verified: "
        f"floor={report.target_floor}, episodes={report.total_episodes}, "
        f"frames={report.total_frames}, videos={report.total_videos}, "
        f"output={report.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
