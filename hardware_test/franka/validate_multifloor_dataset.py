from __future__ import annotations

import argparse
import math
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.compute as pc
import torch
from torch.utils.data import DataLoader

from hardware_test.franka.floor_condition import (
    FLOOR_CONDITION_FEATURE,
    FLOOR_CONDITION_KEY,
    NUM_ELEVATOR_FLOORS,
    TRAINED_ROLLOUT_FLOORS,
    encode_target_floor,
)
from lerobot.configs import FeatureType
from lerobot.datasets import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    merge_datasets,
    resolve_delta_timestamps,
)
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.feature_utils import dataset_to_policy_features

DEFAULT_OUTPUT_REPO_ID = "local/press_button_floors_1_4_5"
DEFAULT_OUTPUT_ROOT = Path("outputs/hardware_test/press_button_floors_1_4_5")
NUMERIC_DTYPE_PREFIXES = ("float", "int", "uint")


@dataclass(frozen=True)
class DatasetSpec:
    floor: int
    repo_id: str
    root: Path

    def __post_init__(self) -> None:
        encode_target_floor(self.floor)
        if not self.repo_id:
            raise ValueError("dataset repo_id must not be empty")
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())


@dataclass(frozen=True)
class DatasetValidationReport:
    repo_id: str
    root: Path
    total_episodes: int
    total_frames: int
    floor_episode_counts: dict[int, int]
    state_dim: int
    action_dim: int
    camera_keys: tuple[str, ...]
    feature_schema: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class MultiFloorValidationReport:
    sources: tuple[DatasetValidationReport, ...]
    total_episodes: int
    total_frames: int
    floor_episode_counts: dict[int, int]
    state_dim: int
    action_dim: int
    camera_keys: tuple[str, ...]


@dataclass(frozen=True)
class MergeReport:
    output_repo_id: str
    output_root: Path
    total_episodes: int
    total_frames: int
    floor_episode_counts: dict[int, int]
    state_dim: int
    action_dim: int
    camera_keys: tuple[str, ...]


@dataclass(frozen=True)
class ACTSmokeTestReport:
    repo_id: str
    root: Path
    environment_shape: tuple[int, ...]
    state_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    loss: float


def _normalise_schema(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _normalise_schema(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_normalise_schema(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _numeric_batch(dataset: LeRobotDataset, keys: Sequence[str]) -> dict[str, np.ndarray]:
    missing = set(keys) - set(dataset.hf_dataset.column_names)
    if missing:
        raise ValueError(f"dataset is missing required frame columns: {sorted(missing)}")
    try:
        batch = dataset.hf_dataset.select_columns(list(keys)).with_format("numpy")[:]
    except Exception as exc:
        raise ValueError(f"failed to read numeric frame columns: {sorted(keys)}") from exc
    return {key: np.asarray(value) for key, value in batch.items()}


def _validate_environment_schema(features: Mapping[str, Mapping[str, Any]]) -> None:
    feature = features.get(FLOOR_CONDITION_KEY)
    if feature is None:
        raise ValueError(f"dataset is missing {FLOOR_CONDITION_KEY}")
    if feature.get("dtype") != FLOOR_CONDITION_FEATURE["dtype"]:
        raise ValueError(f"{FLOOR_CONDITION_KEY} must declare dtype float32")
    if tuple(feature.get("shape", ())) != tuple(FLOOR_CONDITION_FEATURE["shape"]):
        raise ValueError(f"{FLOOR_CONDITION_KEY} must declare shape (5,)")
    if feature.get("names") is not None:
        raise ValueError(f"{FLOOR_CONDITION_KEY} names must be null")


def _validate_environment_storage(dataset: LeRobotDataset) -> None:
    try:
        storage_feature = dataset.hf_dataset.features[FLOOR_CONDITION_KEY]
    except KeyError as exc:
        raise ValueError(f"dataset is missing frame column {FLOOR_CONDITION_KEY}") from exc
    scalar_feature = getattr(storage_feature, "feature", None)
    if getattr(scalar_feature, "dtype", None) != "float32":
        raise ValueError(f"{FLOOR_CONDITION_KEY} stored values must be float32")
    if getattr(storage_feature, "length", None) != NUM_ELEVATOR_FLOORS:
        raise ValueError(f"{FLOOR_CONDITION_KEY} stored values must have shape ({NUM_ELEVATOR_FLOORS},)")


def _validate_vector_schema(
    features: Mapping[str, Mapping[str, Any]],
    key: str,
) -> int:
    feature = features.get(key)
    if feature is None:
        raise ValueError(f"dataset is missing {key}")
    shape = tuple(feature.get("shape", ()))
    if len(shape) != 1 or not shape or not isinstance(shape[0], int) or shape[0] <= 0:
        raise ValueError(f"{key} must declare a non-empty one-dimensional shape")
    if not str(feature.get("dtype", "")).startswith(("float", "int", "uint")):
        raise ValueError(f"{key} must declare a numeric dtype")
    return shape[0]


def _episode_rows(dataset: LeRobotDataset) -> list[dict[str, Any]]:
    episodes = dataset.meta.episodes
    if episodes is None:
        raise ValueError("episode metadata is missing")
    if dataset.meta.total_episodes <= 0 or len(episodes) == 0:
        raise ValueError("episode metadata is empty")
    if len(episodes) != dataset.meta.total_episodes:
        raise ValueError(
            "episode metadata count does not match total_episodes "
            f"({len(episodes)} != {dataset.meta.total_episodes})"
        )
    return [episodes[index] for index in range(len(episodes))]


def _validate_episode_structure(
    dataset: LeRobotDataset,
    frame_values: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows = _episode_rows(dataset)
    total_frames = int(dataset.meta.total_frames)
    if total_frames <= 0 or len(dataset) != total_frames:
        raise ValueError(f"frame count must be positive and readable ({len(dataset)} != {total_frames})")

    expected_from = 0
    episode_indices = frame_values["episode_index"].reshape(-1)
    frame_indices = frame_values["frame_index"].reshape(-1)
    indices = frame_values["index"].reshape(-1)
    if not np.array_equal(indices, np.arange(total_frames, dtype=indices.dtype)):
        raise ValueError("frame index column does not cover the dataset contiguously")

    for expected_episode_index, row in enumerate(rows):
        episode_index = int(row.get("episode_index", -1))
        if episode_index != expected_episode_index:
            raise ValueError(
                "episode metadata indices must be contiguous from zero "
                f"(expected {expected_episode_index}, got {episode_index})"
            )
        start = int(row.get("dataset_from_index", -1))
        stop = int(row.get("dataset_to_index", -1))
        length = int(row.get("length", stop - start))
        if start != expected_from:
            raise ValueError(
                f"episode metadata ranges must be contiguous (expected start {expected_from}, got {start})"
            )
        if stop <= start or length <= 0 or stop - start != length:
            raise ValueError(f"empty episode or invalid range for episode {episode_index}")
        if stop > total_frames:
            raise ValueError(f"episode {episode_index} range exceeds total_frames")
        expected_episode_values = np.full(length, episode_index, dtype=episode_indices.dtype)
        if not np.array_equal(episode_indices[start:stop], expected_episode_values):
            raise ValueError(f"frame episode_index values do not match episode {episode_index}")
        if not np.array_equal(
            frame_indices[start:stop],
            np.arange(length, dtype=frame_indices.dtype),
        ):
            raise ValueError(f"frame_index values are not contiguous in episode {episode_index}")
        expected_from = stop

    if expected_from != total_frames:
        raise ValueError(
            f"episode metadata ranges do not cover all frames ({expected_from} != {total_frames})"
        )
    return rows


def _validate_conditions(
    conditions: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    expected_floors: frozenset[int],
) -> dict[int, int]:
    expected_shape = (sum(int(row["length"]) for row in rows), NUM_ELEVATOR_FLOORS)
    if conditions.dtype != np.float32:
        raise ValueError(f"{FLOOR_CONDITION_KEY} values must be float32")
    if conditions.shape != expected_shape:
        raise ValueError(
            f"{FLOOR_CONDITION_KEY} values must have shape {expected_shape}, got {conditions.shape}"
        )
    if not np.isfinite(conditions).all():
        raise ValueError(f"{FLOOR_CONDITION_KEY} contains NaN or Inf")
    if (
        not np.logical_or(conditions == 0.0, conditions == 1.0).all()
        or not np.equal(conditions.sum(axis=1), 1.0).all()
    ):
        raise ValueError(f"{FLOOR_CONDITION_KEY} must contain exact one-hot vectors")

    counts: Counter[int] = Counter()
    for row in rows:
        episode_index = int(row["episode_index"])
        start = int(row["dataset_from_index"])
        stop = int(row["dataset_to_index"])
        episode_conditions = conditions[start:stop]
        first = episode_conditions[0]
        if not np.equal(episode_conditions, first).all():
            raise ValueError(f"floor condition changes within episode {episode_index}")
        floor = int(np.argmax(first)) + 1
        counts[floor] += 1

    observed_floors = frozenset(counts)
    if observed_floors != expected_floors:
        raise ValueError(
            f"expected floors {sorted(expected_floors)}, observed floors {sorted(observed_floors)}"
        )
    return dict(sorted(counts.items()))


def _validate_numeric_values(
    values: Mapping[str, np.ndarray],
    *,
    state_dim: int,
    action_dim: int,
    total_frames: int,
) -> None:
    for key, expected_dim in ((OBS_STATE, state_dim), (ACTION, action_dim)):
        value = values[key]
        expected_shape = (total_frames, expected_dim)
        if value.shape != expected_shape:
            raise ValueError(f"{key} values must have shape {expected_shape}, got {value.shape}")
        if not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"{key} values must be numeric")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains NaN or Inf")


def _is_numeric_dtype(dtype: Any) -> bool:
    return isinstance(dtype, str) and dtype.startswith(NUMERIC_DTYPE_PREFIXES)


def _storage_dtype_and_shape(storage_feature: Any) -> tuple[str | None, tuple[int, ...]]:
    dtype = getattr(storage_feature, "dtype", None)
    shape = getattr(storage_feature, "shape", None)
    if dtype is not None and shape is not None:
        return str(dtype), tuple(shape)
    if dtype is not None:
        return str(dtype), ()

    child = getattr(storage_feature, "feature", None)
    if child is None:
        return None, ()
    child_dtype, child_shape = _storage_dtype_and_shape(child)
    length = getattr(storage_feature, "length", None)
    if not isinstance(length, int) or length <= 0:
        return child_dtype, ()
    return child_dtype, (length, *child_shape)


def _validate_all_numeric_features(dataset: LeRobotDataset, *, total_frames: int) -> None:
    numeric_features = {
        key: feature
        for key, feature in dataset.meta.features.items()
        if _is_numeric_dtype(feature.get("dtype"))
    }
    columns = set(dataset.hf_dataset.column_names)
    for key in numeric_features:
        if key not in columns:
            raise ValueError(f"numeric feature {key} is missing frame column")

    try:
        loaded = dataset.hf_dataset.select_columns(list(numeric_features)).with_format("numpy")[:]
    except Exception as exc:
        raise ValueError("failed to read one or more declared numeric features") from exc

    for key, feature in numeric_features.items():
        declared_dtype = str(feature["dtype"])
        declared_shape = tuple(feature.get("shape", ()))
        if not declared_shape or any(not isinstance(size, int) or size <= 0 for size in declared_shape):
            raise ValueError(f"numeric feature {key} must declare a non-empty positive shape")

        storage_feature = dataset.hf_dataset.features.get(key)
        storage_dtype, storage_shape = _storage_dtype_and_shape(storage_feature)
        if not _is_numeric_dtype(storage_dtype):
            raise ValueError(f"numeric feature {key} has non-numeric frame storage")
        try:
            storage_matches = np.dtype(storage_dtype) == np.dtype(declared_dtype)
        except TypeError as exc:
            raise ValueError(f"numeric feature {key} declares an unsupported dtype") from exc
        if not storage_matches:
            raise ValueError(
                f"numeric feature {key} storage dtype {storage_dtype} "
                f"does not match declared dtype {declared_dtype}"
            )

        scalar_storage = storage_shape == () and declared_shape == (1,)
        if storage_shape != declared_shape and not scalar_storage:
            raise ValueError(
                f"numeric feature {key} storage shape {storage_shape} "
                f"does not match declared shape {declared_shape}"
            )

        values = np.asarray(loaded[key])
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"numeric feature {key} has a non-numeric loaded representation")
        expected_shape = (total_frames, *declared_shape)
        loaded_shape_matches = values.shape == expected_shape
        if scalar_storage and values.shape == (total_frames,):
            loaded_shape_matches = True
        if not loaded_shape_matches:
            raise ValueError(
                f"numeric feature {key} values must have shape {expected_shape}, got {values.shape}"
            )
        if declared_dtype.startswith("float") and not np.isfinite(values).all():
            raise ValueError(f"numeric feature {key} contains NaN or Inf")


def _validate_camera_sample(
    value: Any,
    *,
    key: str,
    declared_shape: tuple[int, ...],
    frame_index: int,
) -> None:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        try:
            array = np.asarray(value)
        except Exception as exc:
            raise ValueError(f"camera {key} frame {frame_index} is not array-like") from exc

    channel_first_shape = (declared_shape[2], declared_shape[0], declared_shape[1])
    if tuple(array.shape) not in (declared_shape, channel_first_shape):
        raise ValueError(
            f"camera {key} frame {frame_index} has shape {array.shape}; "
            f"expected {declared_shape} or {channel_first_shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"camera {key} frame {frame_index} has non-numeric dtype {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"camera {key} frame {frame_index} contains NaN or Inf")
    if np.issubdtype(array.dtype, np.floating) and array.dtype != np.float32:
        raise ValueError(f"camera {key} frame {frame_index} must decode to float32 or integer pixels")


def _validate_video_file(
    path: Path,
    *,
    key: str,
    declared_shape: tuple[int, ...],
    expected_frames: int,
) -> None:
    decoded_frames = 0
    try:
        with av.open(str(path), mode="r") as container:
            if not container.streams.video:
                raise ValueError(f"camera {key} file has no video stream: {path}")
            stream = container.streams.video[0]
            for decoded_frames, frame in enumerate(container.decode(stream), start=1):
                _validate_camera_sample(
                    frame.to_ndarray(format="rgb24"),
                    key=key,
                    declared_shape=declared_shape,
                    frame_index=decoded_frames - 1,
                )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"failed to decode camera {key} video: {path}") from exc

    if decoded_frames != expected_frames:
        raise ValueError(
            f"camera {key} video {path} decoded {decoded_frames} frames; expected {expected_frames}"
        )


def _validate_media(dataset: LeRobotDataset, *, skip_image_decode: bool) -> tuple[str, ...]:
    camera_keys = tuple(sorted(dataset.meta.camera_keys))
    if not camera_keys:
        raise ValueError("dataset contains no camera features")

    columns = set(dataset.hf_dataset.column_names)
    video_files: dict[tuple[str, Path], int] = {}
    for key in camera_keys:
        feature = dataset.meta.features[key]
        dtype = feature.get("dtype")
        shape = tuple(feature.get("shape", ()))
        if dtype not in {"image", "video"} or len(shape) != 3:
            raise ValueError(f"camera {key} must declare a three-dimensional image/video feature")
        if dtype == "image" and key not in columns:
            raise ValueError(f"dataset is missing image column {key}")
        if dtype == "image":
            image_column = dataset.hf_dataset.data.column(key)
            for chunk in image_column.chunks:
                if chunk.null_count:
                    raise ValueError(f"camera {key} has a missing embedded image payload")
                try:
                    has_payload = pc.or_(
                        pc.is_valid(chunk.field("bytes")),
                        pc.is_valid(chunk.field("path")),
                    )
                except (KeyError, TypeError) as exc:
                    raise ValueError(f"camera {key} has invalid embedded image storage") from exc
                if not bool(pc.all(has_payload).as_py()):
                    raise ValueError(f"camera {key} has a missing embedded image payload")
        if dtype == "video":
            for episode_index in range(dataset.meta.total_episodes):
                relative_path = dataset.meta.get_video_file_path(episode_index, key)
                path = dataset.root / relative_path
                if not path.is_file() or path.stat().st_size <= 0:
                    raise ValueError(f"missing or empty video for {key}: {path}")
                episode_length = int(dataset.meta.episodes[episode_index]["length"])
                file_key = (key, path)
                video_files[file_key] = video_files.get(file_key, 0) + episode_length

    if skip_image_decode:
        return camera_keys

    image_keys = tuple(key for key in camera_keys if dataset.meta.features[key]["dtype"] == "image")
    if image_keys:
        for frame_index in range(len(dataset)):
            try:
                frame = dataset[frame_index]
            except Exception as exc:
                raise ValueError(f"failed to decode camera media at frame {frame_index}") from exc
            for key in image_keys:
                if key not in frame:
                    raise ValueError(f"decoded frame {frame_index} is missing camera {key}")
                _validate_camera_sample(
                    frame[key],
                    key=key,
                    declared_shape=tuple(dataset.meta.features[key]["shape"]),
                    frame_index=frame_index,
                )

    total_video_files = len(video_files)
    for video_index, ((key, path), expected_frames) in enumerate(video_files.items(), start=1):
        print(
            f"Media decode [{dataset.repo_id}]: video {video_index}/{total_video_files} "
            f"({expected_frames} frames) {path.name}",
            flush=True,
        )
        _validate_video_file(
            path,
            key=key,
            declared_shape=tuple(dataset.meta.features[key]["shape"]),
            expected_frames=expected_frames,
        )
    return camera_keys


def validate_dataset(
    dataset: LeRobotDataset,
    *,
    expected_floors: set[int] | frozenset[int],
    skip_image_decode: bool = False,
) -> DatasetValidationReport:
    expected = frozenset(expected_floors)
    if not expected:
        raise ValueError("expected_floors must not be empty")
    for floor in expected:
        encode_target_floor(floor)

    features = dataset.meta.features
    _validate_environment_schema(features)
    _validate_environment_storage(dataset)
    state_dim = _validate_vector_schema(features, OBS_STATE)
    action_dim = _validate_vector_schema(features, ACTION)

    numeric_keys = (
        FLOOR_CONDITION_KEY,
        OBS_STATE,
        ACTION,
        "episode_index",
        "frame_index",
        "index",
    )
    values = _numeric_batch(dataset, numeric_keys)
    rows = _validate_episode_structure(dataset, values)
    total_frames = int(dataset.meta.total_frames)
    _validate_numeric_values(
        values,
        state_dim=state_dim,
        action_dim=action_dim,
        total_frames=total_frames,
    )
    floor_counts = _validate_conditions(values[FLOOR_CONDITION_KEY], rows, expected)
    _validate_all_numeric_features(dataset, total_frames=total_frames)
    camera_keys = _validate_media(dataset, skip_image_decode=skip_image_decode)

    return DatasetValidationReport(
        repo_id=dataset.repo_id,
        root=dataset.root.resolve(),
        total_episodes=int(dataset.meta.total_episodes),
        total_frames=total_frames,
        floor_episode_counts=floor_counts,
        state_dim=state_dim,
        action_dim=action_dim,
        camera_keys=camera_keys,
        feature_schema=dict(features),
    )


def _load_and_validate_spec(
    spec: DatasetSpec,
    *,
    skip_image_decode: bool,
) -> tuple[LeRobotDataset, DatasetValidationReport]:
    if not (spec.root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"dataset does not exist at {spec.root}")
    dataset = LeRobotDataset(
        repo_id=spec.repo_id,
        root=spec.root,
        return_uint8=True,
    )
    return dataset, validate_dataset(
        dataset,
        expected_floors={spec.floor},
        skip_image_decode=skip_image_decode,
    )


def validate_multifloor_datasets(
    specs: Sequence[DatasetSpec],
    *,
    skip_image_decode: bool = False,
) -> MultiFloorValidationReport:
    if not specs:
        raise ValueError("at least one dataset is required")

    reports: list[DatasetValidationReport] = []
    reference_schema: Any | None = None
    for spec in specs:
        _, report = _load_and_validate_spec(spec, skip_image_decode=skip_image_decode)
        normalised_schema = _normalise_schema(report.feature_schema)
        if reference_schema is None:
            reference_schema = normalised_schema
        elif normalised_schema != reference_schema:
            raise ValueError("source feature schemas differ; all fields and camera info must match exactly")
        reports.append(report)

    reference = reports[0]
    for report in reports[1:]:
        if report.state_dim != reference.state_dim or report.action_dim != reference.action_dim:
            raise ValueError("source observation.state/action dimensions differ")
        if report.camera_keys != reference.camera_keys:
            raise ValueError("source camera keys differ")

    floor_counts: Counter[int] = Counter()
    for report in reports:
        floor_counts.update(report.floor_episode_counts)

    return MultiFloorValidationReport(
        sources=tuple(reports),
        total_episodes=sum(report.total_episodes for report in reports),
        total_frames=sum(report.total_frames for report in reports),
        floor_episode_counts=dict(sorted(floor_counts.items())),
        state_dim=reference.state_dim,
        action_dim=reference.action_dim,
        camera_keys=reference.camera_keys,
    )


def _preflight_merge(
    specs: Sequence[DatasetSpec],
    output_root: str | Path,
    *,
    expected_floors: set[int] | frozenset[int],
) -> Path:
    expected = frozenset(expected_floors)
    if not expected:
        raise ValueError("merge expected_floors must not be empty")
    for floor in expected:
        encode_target_floor(floor)
    observed = frozenset(spec.floor for spec in specs)
    if observed != expected:
        raise ValueError(f"merge sources must cover exactly floors {_format_floor_list(expected)}")
    source_roots = [spec.root for spec in specs]
    if len(set(source_roots)) != len(source_roots):
        raise ValueError("duplicate source dataset root")

    resolved_output = Path(output_root).expanduser().resolve()
    if resolved_output.exists() or resolved_output.is_symlink():
        raise FileExistsError(f"output root already exists: {resolved_output}")
    for spec in specs:
        if resolved_output == spec.root or spec.root in resolved_output.parents:
            raise ValueError("merge output root must be outside every source dataset")
    return resolved_output


def _format_floor_list(floors: set[int] | frozenset[int]) -> str:
    values = [str(floor) for floor in sorted(floors)]
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def merge_validated_datasets(
    specs: Sequence[DatasetSpec],
    *,
    output_repo_id: str = DEFAULT_OUTPUT_REPO_ID,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    skip_image_decode: bool = False,
    expected_floors: set[int] | frozenset[int] = frozenset(TRAINED_ROLLOUT_FLOORS),
) -> MergeReport:
    expected = frozenset(expected_floors)
    resolved_output = _preflight_merge(specs, output_root, expected_floors=expected)
    source_report = validate_multifloor_datasets(specs, skip_image_decode=skip_image_decode)
    datasets = [LeRobotDataset(repo_id=spec.repo_id, root=spec.root) for spec in specs]

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    owned_temp = Path(
        tempfile.mkdtemp(
            prefix=f".{resolved_output.name}.merge-",
            dir=resolved_output.parent,
        )
    )
    staging_root = owned_temp / "dataset"
    try:
        merged = merge_datasets(
            datasets,
            output_repo_id=output_repo_id,
            output_dir=staging_root,
        )
        merged_report = validate_dataset(
            merged,
            expected_floors=expected,
            skip_image_decode=skip_image_decode,
        )
        if merged_report.total_episodes != source_report.total_episodes:
            raise RuntimeError("official merge changed the total episode count")
        if merged_report.total_frames != source_report.total_frames:
            raise RuntimeError("official merge changed the total frame count")
        if merged_report.floor_episode_counts != source_report.floor_episode_counts:
            raise RuntimeError("official merge changed the per-floor episode counts")
        if _normalise_schema(merged_report.feature_schema) != _normalise_schema(
            source_report.sources[0].feature_schema
        ):
            raise RuntimeError("official merge changed the feature schema")

        if resolved_output.exists() or resolved_output.is_symlink():
            raise FileExistsError(f"output root appeared during merge: {resolved_output}")
        staging_root.rename(resolved_output)
        return MergeReport(
            output_repo_id=output_repo_id,
            output_root=resolved_output,
            total_episodes=merged_report.total_episodes,
            total_frames=merged_report.total_frames,
            floor_episode_counts=merged_report.floor_episode_counts,
            state_dim=merged_report.state_dim,
            action_dim=merged_report.action_dim,
            camera_keys=merged_report.camera_keys,
        )
    finally:
        shutil.rmtree(owned_temp, ignore_errors=True)


def smoke_test_act_training(
    *,
    repo_id: str,
    root: str | Path,
) -> ACTSmokeTestReport:
    resolved_root = Path(root).expanduser().resolve()
    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=resolved_root)
    policy_features = dataset_to_policy_features(metadata.features)
    input_features = {
        key: feature for key, feature in policy_features.items() if feature.type is not FeatureType.ACTION
    }
    output_features = {
        key: feature for key, feature in policy_features.items() if feature.type is FeatureType.ACTION
    }
    if FLOOR_CONDITION_KEY not in input_features:
        raise ValueError(f"ACT smoke dataset is missing {FLOOR_CONDITION_KEY}")
    if OBS_STATE not in input_features or ACTION not in output_features:
        raise ValueError("ACT smoke dataset must contain observation.state and action")

    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        use_amp=False,
        push_to_hub=False,
        chunk_size=2,
        n_action_steps=2,
        pretrained_backbone_weights=None,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
    )
    delta_timestamps = resolve_delta_timestamps(config, metadata)
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=resolved_root,
        delta_timestamps=delta_timestamps,
        return_uint8=True,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    for camera_key in dataset.meta.camera_keys:
        if camera_key in batch and batch[camera_key].dtype == torch.uint8:
            batch[camera_key] = batch[camera_key].to(dtype=torch.float32) / 255.0

    policy = make_policy(config, ds_meta=metadata)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        dataset_stats=metadata.stats,
    )
    batch = preprocessor(batch)
    environment_shape = tuple(batch[FLOOR_CONDITION_KEY].shape)
    state_shape = tuple(batch[OBS_STATE].shape)
    action_shape = tuple(batch[ACTION].shape)
    if environment_shape != (1, NUM_ELEVATOR_FLOORS):
        raise RuntimeError(f"ACT environment batch must have shape (1, 5), got {environment_shape}")
    if batch[FLOOR_CONDITION_KEY].dtype != torch.float32:
        raise RuntimeError("ACT environment batch must be torch.float32")
    if state_shape[-1] != tuple(metadata.features[OBS_STATE]["shape"])[-1]:
        raise RuntimeError("ACT preprocessing changed observation.state dimension")
    if action_shape[-1] != tuple(metadata.features[ACTION]["shape"])[-1]:
        raise RuntimeError("ACT preprocessing changed action dimension")

    policy.train()
    loss, _ = policy.forward(batch)
    loss_value = float(loss.detach().cpu().item())
    if not math.isfinite(loss_value):
        raise RuntimeError(f"ACT smoke loss is not finite: {loss_value}")
    return ACTSmokeTestReport(
        repo_id=repo_id,
        root=resolved_root,
        environment_shape=environment_shape,
        state_shape=state_shape,
        action_shape=action_shape,
        loss=loss_value,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly validate conditioned floor datasets, optionally merge and smoke-test ACT."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=3,
        metavar=("FLOOR", "REPO_ID", "ROOT"),
        required=True,
        help="Repeat once per source dataset.",
    )
    parser.add_argument("--merge", action="store_true")
    parser.add_argument(
        "--expected-floors",
        type=int,
        nargs="+",
        default=list(TRAINED_ROLLOUT_FLOORS),
        metavar="FLOOR",
        help="Exact floor set required in merge sources; defaults to 1 4 5.",
    )
    parser.add_argument("--output-repo-id", default=DEFAULT_OUTPUT_REPO_ID)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-image-decode", action="store_true")
    return parser


def _parse_dataset_specs(raw_specs: Sequence[Sequence[str]]) -> tuple[DatasetSpec, ...]:
    specs = []
    for raw_floor, repo_id, root in raw_specs:
        try:
            floor = int(raw_floor)
        except ValueError as exc:
            raise ValueError(f"dataset floor must be an integer, got {raw_floor!r}") from exc
        specs.append(DatasetSpec(floor=floor, repo_id=repo_id, root=Path(root)))
    return tuple(specs)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    specs = _parse_dataset_specs(args.dataset)
    target_repo_id: str | None = None
    target_root: Path | None = None
    if args.merge:
        merge_report = merge_validated_datasets(
            specs,
            output_repo_id=args.output_repo_id,
            output_root=args.output_root,
            skip_image_decode=args.skip_image_decode,
            expected_floors=set(args.expected_floors),
        )
        target_repo_id = merge_report.output_repo_id
        target_root = merge_report.output_root
        print(
            "Source and official merge validation passed: "
            f"episodes={merge_report.total_episodes}, frames={merge_report.total_frames}, "
            f"floors={merge_report.floor_episode_counts}, output={merge_report.output_root}"
        )
    else:
        source_report = validate_multifloor_datasets(
            specs,
            skip_image_decode=args.skip_image_decode,
        )
        print(
            "Source validation passed: "
            f"episodes={source_report.total_episodes}, frames={source_report.total_frames}, "
            f"floors={source_report.floor_episode_counts}, "
            f"state_dim={source_report.state_dim}, action_dim={source_report.action_dim}"
        )

    if args.smoke_test:
        if target_root is None:
            if len(specs) != 1:
                raise ValueError("--smoke-test without --merge requires exactly one --dataset")
            target_repo_id = specs[0].repo_id
            target_root = specs[0].root
        smoke_report = smoke_test_act_training(repo_id=target_repo_id, root=target_root)
        print(
            "ACT smoke test passed: "
            f"environment={smoke_report.environment_shape}, state={smoke_report.state_shape}, "
            f"action={smoke_report.action_shape}, loss={smoke_report.loss:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
