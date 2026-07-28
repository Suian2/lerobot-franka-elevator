#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Aggregate one numbered hardware recording group into a training dataset.

Example:
    uv run --extra dataset python hardware_test/aggregate_lerobot_datasets.py \
        --source-name press_01 \
        --output-name press_train \
        --count 30

``--source-name`` is the first recording directory in a group. The script
derives the shared prefix and requires the requested number of consecutively
numbered directories starting at ``01``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_DATASETS_ROOT = Path("outputs/hardware_test")
FIRST_RECORDING_INDEX = 1
DEFAULT_DATASET_COUNT = 30


@dataclass(frozen=True)
class SourceGroup:
    prefix: str
    width: int = 2


@dataclass(frozen=True)
class AggregationReport:
    output_repo_id: str
    output_root: Path
    total_episodes: int
    total_frames: int
    source_count: int


def _load_dataset(repo_id: str, root: Path) -> Any:
    from lerobot.datasets import LeRobotDataset

    return LeRobotDataset(repo_id=repo_id, root=root)


def _run_official_merge(datasets: list[Any], output_repo_id: str, output_dir: Path) -> Any:
    from lerobot.datasets import merge_datasets

    return merge_datasets(
        datasets,
        output_repo_id=output_repo_id,
        output_dir=output_dir,
        concatenate_videos=False,
        concatenate_data=False,
    )


def _read_episode_records(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    records = []
    for path in paths:
        records.extend(pq.read_table(path).to_pylist())
    return records


def _require_local_reference(root: Path, relative_path: str, *, kind: str) -> None:
    path = (root / relative_path).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"referenced {kind} escapes dataset root {root}: {relative_path}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or empty referenced {kind}: {path}")


def _validate_local_dataset_root(root: Path) -> None:
    root = root.resolve()
    required_files = (
        root / "meta" / "info.json",
        root / "meta" / "stats.json",
        root / "meta" / "tasks.parquet",
    )
    missing = [str(path.relative_to(root)) for path in required_files if not path.is_file()]
    episode_files = tuple(sorted((root / "meta" / "episodes").glob("**/*.parquet")))
    if not episode_files:
        missing.append("meta/episodes/**/*.parquet")
    if not any((root / "data").glob("**/*.parquet")):
        missing.append("data/**/*.parquet")
    if missing:
        raise ValueError(f"incomplete local LeRobot dataset at {root}: missing {', '.join(missing)}")

    try:
        info = json.loads((root / "meta" / "info.json").read_text())
        episode_records = _read_episode_records(episode_files)
        total_episodes = int(info["total_episodes"])
        data_path_template = str(info["data_path"])
        video_path_template = str(info["video_path"])
        video_keys = tuple(
            key for key, feature in info["features"].items() if feature.get("dtype") == "video"
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid local LeRobot metadata at {root}: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"failed to read local LeRobot metadata at {root}: {exc}") from exc

    if len(episode_records) != total_episodes:
        raise ValueError(
            f"dataset {root} metadata reports {total_episodes} episodes, "
            f"but episode metadata has {len(episode_records)} records"
        )

    try:
        for episode in episode_records:
            data_relative_path = data_path_template.format(
                chunk_index=int(episode["data/chunk_index"]),
                file_index=int(episode["data/file_index"]),
            )
            _require_local_reference(root, data_relative_path, kind="data")
            for video_key in video_keys:
                video_relative_path = video_path_template.format(
                    video_key=video_key,
                    chunk_index=int(episode[f"videos/{video_key}/chunk_index"]),
                    file_index=int(episode[f"videos/{video_key}/file_index"]),
                )
                _require_local_reference(root, video_relative_path, kind="video")
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(("missing or empty", "referenced")):
            raise
        raise ValueError(f"invalid episode file mapping at {root}: {exc}") from exc


def _validate_dataset_integrity(dataset: Any, *, expected_episodes: int) -> tuple[int, int]:
    total_episodes = int(dataset.meta.total_episodes)
    total_frames = int(dataset.meta.total_frames)
    if total_episodes != expected_episodes:
        raise ValueError(
            f"dataset {dataset.root} expected {expected_episodes} episode(s), "
            f"but metadata reports {total_episodes}"
        )

    actual_frames = len(dataset.hf_dataset)
    if actual_frames != total_frames:
        raise ValueError(
            f"dataset {dataset.root} metadata reports {total_frames} frames, "
            f"but actual parquet data has {actual_frames}"
        )

    index_columns = ("episode_index", "frame_index", "index")
    missing_columns = [column for column in index_columns if column not in dataset.hf_dataset.column_names]
    if missing_columns:
        raise ValueError(f"dataset {dataset.root} is missing index columns: {', '.join(missing_columns)}")
    indices = dataset.hf_dataset.select_columns(index_columns).with_format(None)[:]
    episode_indices = [int(value) for value in indices["episode_index"]]
    frame_indices = [int(value) for value in indices["frame_index"]]
    global_indices = [int(value) for value in indices["index"]]
    if global_indices != list(range(total_frames)):
        raise ValueError(f"dataset {dataset.root} has a non-contiguous global index")
    if episode_indices != sorted(episode_indices) or set(episode_indices) != set(range(total_episodes)):
        raise ValueError(f"dataset {dataset.root} has invalid or non-contiguous episode indices")

    episode_lengths = [0] * total_episodes
    for episode_index, frame_index in zip(episode_indices, frame_indices, strict=True):
        if frame_index != episode_lengths[episode_index]:
            raise ValueError(
                f"dataset {dataset.root} episode {episode_index} has a non-contiguous frame index"
            )
        episode_lengths[episode_index] += 1

    for episode_index in range(total_episodes):
        data_path = dataset.root / dataset.meta.get_data_file_path(episode_index)
        if not data_path.is_file() or data_path.stat().st_size <= 0:
            raise ValueError(f"missing or empty parquet data: {data_path}")
        for video_key in dataset.meta.video_keys:
            video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, video_key)
            if not video_path.is_file() or video_path.stat().st_size <= 0:
                raise ValueError(f"missing or empty video for {video_key}: {video_path}")

        metadata_length = int(dataset.meta.episodes[episode_index]["length"])
        if metadata_length != episode_lengths[episode_index]:
            raise ValueError(
                f"dataset {dataset.root} episode {episode_index} metadata length is {metadata_length}, "
                f"but parquet data has {episode_lengths[episode_index]} frames"
            )

    return total_episodes, total_frames


def parse_source_name(source_name: str) -> SourceGroup:
    """Extract a group's common prefix from its ``01`` directory name."""
    if Path(source_name).name != source_name:
        raise ValueError("source name must be a single directory name")
    if len(source_name) <= 2 or not source_name.endswith("01"):
        raise ValueError("source name must end with '01', for example 'press_button_01'")
    return SourceGroup(prefix=source_name[:-2])


def build_source_roots(
    datasets_root: str | Path,
    source_name: str,
    *,
    count: int = DEFAULT_DATASET_COUNT,
) -> tuple[Path, ...]:
    """Return the requested number of existing source roots in numeric order."""
    if count <= 0:
        raise ValueError("count must be positive")
    root = Path(datasets_root).expanduser().resolve()
    group = parse_source_name(source_name)
    candidates = tuple(
        root / f"{group.prefix}{index:0{group.width}d}" for index in range(FIRST_RECORDING_INDEX, count + 1)
    )
    missing = [path.name for path in candidates if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing source dataset directories: {', '.join(missing)}")
    return tuple(path.resolve() for path in candidates)


def resolve_output_root(datasets_root: str | Path, output_name: str) -> Path:
    """Resolve a new output directory while keeping it under ``datasets_root``."""
    if not output_name or output_name in {".", ".."} or Path(output_name).name != output_name:
        raise ValueError("output name must be a single directory name")

    output_root = (Path(datasets_root).expanduser().resolve() / output_name).resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"output dataset already exists: {output_root}")
    return output_root


def merge_recording_group(
    datasets_root: str | Path,
    source_name: str,
    output_name: str,
    *,
    count: int = DEFAULT_DATASET_COUNT,
) -> AggregationReport:
    """Merge one complete numbered group and atomically publish the result."""
    source_roots = build_source_roots(datasets_root, source_name, count=count)
    output_root = resolve_output_root(datasets_root, output_name)
    output_repo_id = f"local/{output_name}"
    for source_root in source_roots:
        _validate_local_dataset_root(source_root)
    datasets = [
        _load_dataset(repo_id=f"local/{source_root.name}", root=source_root) for source_root in source_roots
    ]
    source_totals = [_validate_dataset_integrity(dataset, expected_episodes=1) for dataset in datasets]
    expected_episodes = sum(total[0] for total in source_totals)
    expected_frames = sum(total[1] for total in source_totals)

    owned_temp = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.merge-",
            dir=output_root.parent,
        )
    )
    staging_root = owned_temp / "dataset"
    try:
        merged = _run_official_merge(
            datasets,
            output_repo_id,
            staging_root,
        )
        actual_totals = _validate_dataset_integrity(merged, expected_episodes=expected_episodes)
        expected_totals = (expected_episodes, expected_frames)
        if actual_totals != expected_totals:
            raise RuntimeError(
                "official merge changed dataset totals: "
                f"expected episodes={expected_episodes}, frames={expected_frames}; "
                f"got episodes={actual_totals[0]}, frames={actual_totals[1]}"
            )
        if not staging_root.is_dir():
            raise RuntimeError(f"official merge did not create its output directory: {staging_root}")
        if output_root.exists() or output_root.is_symlink():
            raise FileExistsError(f"output dataset appeared during merge: {output_root}")

        staging_root.rename(output_root)
        return AggregationReport(
            output_repo_id=output_repo_id,
            output_root=output_root,
            total_episodes=actual_totals[0],
            total_frames=actual_totals[1],
            source_count=len(source_roots),
        )
    finally:
        try:
            shutil.rmtree(owned_temp)
        except OSError as exc:
            raise RuntimeError(f"failed to remove staging directory {owned_temp}: {exc}") from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate consecutive LeRobot hardware recordings into a training dataset."
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=DEFAULT_DATASETS_ROOT,
        help=f"Directory containing source datasets (default: {DEFAULT_DATASETS_ROOT}).",
    )
    parser.add_argument(
        "--source-name",
        required=True,
        help="Name of recording 01; the shared prefix is derived automatically (for example press_01).",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="New dataset directory name under --datasets-root (for example press_train).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_DATASET_COUNT,
        help=f"Number of consecutive datasets to merge, starting at 01 (default: {DEFAULT_DATASET_COUNT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only verify that all requested source directories exist and the output name is available.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if args.dry_run:
            source_roots = build_source_roots(args.datasets_root, args.source_name, count=args.count)
            output_root = resolve_output_root(args.datasets_root, args.output_name)
            print(
                f"Found {len(source_roots)} source datasets ({source_roots[0].name} through "
                f"{source_roots[-1].name}); output will be {output_root}"
            )
            return 0

        report = merge_recording_group(
            args.datasets_root,
            args.source_name,
            args.output_name,
            count=args.count,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"Merged {report.source_count} source datasets into {report.output_root}: "
        f"episodes={report.total_episodes}, frames={report.total_frames}, "
        f"repo_id={report.output_repo_id}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
