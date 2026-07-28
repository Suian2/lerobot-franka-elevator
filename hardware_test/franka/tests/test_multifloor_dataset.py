from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from datasets import Image, List as DatasetList, Value

import hardware_test.franka.migrate_add_target_floor as migration_module
import hardware_test.franka.validate_multifloor_dataset as validator_module
from hardware_test.franka.floor_condition import (
    FLOOR_CONDITION_FEATURE,
    FLOOR_CONDITION_KEY,
    encode_target_floor,
)
from hardware_test.franka.migrate_add_target_floor import (
    _verify_video_files,
    build_arg_parser,
    migrate_dataset,
)
from hardware_test.franka.validate_multifloor_dataset import (
    DEFAULT_OUTPUT_REPO_ID,
    DatasetSpec,
    build_arg_parser as build_validation_arg_parser,
    merge_validated_datasets,
    smoke_test_act_training,
    validate_dataset,
    validate_multifloor_datasets,
)
from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

CAMERA_KEY = "observation.images.test"
SOURCE_REPO_ID = "local/source_floor_1"
OUTPUT_REPO_ID = "local/output_floor_1"


def _create_tiny_dataset(
    root: Path,
    *,
    repo_id: str = SOURCE_REPO_ID,
    conditioned_floor: int | None = None,
    state_dim: int = 3,
    action_dim: int = 2,
    image_shape: tuple[int, int, int] = (8, 8, 3),
    episode_lengths: tuple[int, ...] = (2, 3),
) -> LeRobotDataset:
    features = {
        OBS_STATE: {"dtype": "float32", "shape": (state_dim,), "names": None},
        ACTION: {"dtype": "float32", "shape": (action_dim,), "names": None},
        CAMERA_KEY: {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
    }
    if conditioned_floor is not None:
        features[FLOOR_CONDITION_KEY] = dict(FLOOR_CONDITION_FEATURE)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=10,
        features=features,
        robot_type="test_franka",
        use_videos=False,
    )
    for episode_index, episode_length in enumerate(episode_lengths):
        for frame_index in range(episode_length):
            value = float(episode_index * 10 + frame_index)
            frame = {
                OBS_STATE: np.arange(state_dim, dtype=np.float32) + value,
                ACTION: np.arange(action_dim, dtype=np.float32) - value,
                CAMERA_KEY: np.full(image_shape, int(value), dtype=np.uint8),
                "task": "press elevator button",
            }
            if conditioned_floor is not None:
                frame[FLOOR_CONDITION_KEY] = encode_target_floor(conditioned_floor)
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
    return LeRobotDataset(repo_id=repo_id, root=root)


def _create_tiny_video_dataset(
    root: Path,
    *,
    repo_id: str = SOURCE_REPO_ID,
    conditioned_floor: int = 4,
    episode_lengths: tuple[int, ...] = (2, 3),
) -> LeRobotDataset:
    image_shape = (8, 8, 3)
    features = {
        OBS_STATE: {"dtype": "float32", "shape": (8,), "names": None},
        ACTION: {"dtype": "float32", "shape": (7,), "names": None},
        CAMERA_KEY: {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        FLOOR_CONDITION_KEY: dict(FLOOR_CONDITION_FEATURE),
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=10,
        features=features,
        robot_type="test_franka",
        use_videos=True,
        streaming_encoding=True,
        encoder_threads=1,
    )
    for episode_index, episode_length in enumerate(episode_lengths):
        for frame_index in range(episode_length):
            value = episode_index * 10 + frame_index
            dataset.add_frame(
                {
                    OBS_STATE: np.full(8, value, dtype=np.float32),
                    ACTION: np.full(7, -value, dtype=np.float32),
                    CAMERA_KEY: np.full(image_shape, value, dtype=np.uint8),
                    FLOOR_CONDITION_KEY: encode_target_floor(conditioned_floor),
                    "task": "press elevator button",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    return LeRobotDataset(repo_id=repo_id, root=root, return_uint8=True)


def _create_conditioned_sources(root: Path) -> tuple[DatasetSpec, ...]:
    specs = []
    for floor in (1, 4, 5):
        dataset_root = root / f"floor-{floor}"
        repo_id = f"local/test-floor-{floor}"
        _create_tiny_dataset(
            dataset_root,
            repo_id=repo_id,
            conditioned_floor=floor,
            state_dim=8,
            action_dim=7,
            image_shape=(64, 64, 3),
        )
        specs.append(DatasetSpec(floor=floor, repo_id=repo_id, root=dataset_root))
    return tuple(specs)


def _load_spec(spec: DatasetSpec) -> LeRobotDataset:
    return LeRobotDataset(repo_id=spec.repo_id, root=spec.root, return_uint8=True)


def _replace_numeric_column(
    dataset: LeRobotDataset,
    key: str,
    values: np.ndarray,
) -> None:
    raw = dataset.hf_dataset.with_format(None)
    dataset.reader.hf_dataset = raw.remove_columns(key)
    _add_numeric_column(dataset, key, values)


def _add_numeric_column(
    dataset: LeRobotDataset,
    key: str,
    values: np.ndarray,
) -> None:
    raw = dataset.hf_dataset.with_format(None)
    dtype = str(values.dtype)
    feature = Value(dtype) if values.ndim == 1 else DatasetList(Value(dtype), length=values.shape[1])
    dataset.reader.hf_dataset = raw.add_column(
        key,
        values.tolist(),
        feature=feature,
    )


def _mutate_info(spec: DatasetSpec, mutate) -> None:
    info_path = spec.root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    mutate(info)
    info_path.write_text(json.dumps(info))


@pytest.fixture(scope="module")
def merged_multifloor_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("conditioned-merge")
    specs = _create_conditioned_sources(root / "sources")
    output_root = root / "merged"
    report = merge_validated_datasets(
        specs,
        output_repo_id="local/test-floors-1-4-5",
        output_root=output_root,
    )
    return specs, report


def _tree_hashes(root: Path) -> dict[Path, str]:
    return {
        path.relative_to(root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _numeric_batch(dataset: LeRobotDataset, keys: list[str]) -> dict[str, np.ndarray]:
    return dataset.hf_dataset.select_columns(keys).with_format("numpy")[:]


def _migration_temp_dirs(output_root: Path) -> list[Path]:
    return list(output_root.parent.glob(f".{output_root.name}.migration-*"))


def test_migrate_dataset_adds_floor_condition_without_changing_source(tmp_path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "new-parent" / "output"
    source = _create_tiny_dataset(source_root)
    source_hashes_before = _tree_hashes(source_root)

    report = migrate_dataset(
        source_repo_id=SOURCE_REPO_ID,
        source_root=source_root,
        output_repo_id=OUTPUT_REPO_ID,
        output_root=output_root,
        target_floor=1,
    )

    destination = LeRobotDataset(repo_id=OUTPUT_REPO_ID, root=output_root)
    assert source.root.resolve() != destination.root.resolve()
    assert FLOOR_CONDITION_KEY not in source.meta.features
    assert destination.meta.features[FLOOR_CONDITION_KEY] == FLOOR_CONDITION_FEATURE
    assert FLOOR_CONDITION_KEY in destination.meta.stats
    assert report.total_episodes == 2
    assert report.total_frames == 5
    assert report.total_videos == 0
    assert report.target_floor == 1
    assert report.output_root == output_root.resolve()
    assert output_root.is_dir()
    assert _migration_temp_dirs(output_root) == []
    assert _tree_hashes(source_root) == source_hashes_before

    assert destination.meta.total_episodes == source.meta.total_episodes
    assert destination.meta.total_frames == source.meta.total_frames
    assert [
        episode["dataset_to_index"] - episode["dataset_from_index"] for episode in destination.meta.episodes
    ] == [episode["dataset_to_index"] - episode["dataset_from_index"] for episode in source.meta.episodes]

    preserved_keys = [
        OBS_STATE,
        ACTION,
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    source_values = _numeric_batch(source, preserved_keys)
    destination_values = _numeric_batch(destination, preserved_keys)
    for key in preserved_keys:
        np.testing.assert_array_equal(destination_values[key], source_values[key])

    conditions = _numeric_batch(destination, [FLOOR_CONDITION_KEY])[FLOOR_CONDITION_KEY]
    assert conditions.dtype == np.float32
    assert conditions.shape == (5, 5)
    np.testing.assert_array_equal(
        conditions,
        np.repeat(encode_target_floor(1)[None, :], repeats=5, axis=0),
    )


@pytest.mark.parametrize("target_floor", [2, 3, 4, 5])
def test_migrate_dataset_uses_requested_canonical_floor(tmp_path, target_floor):
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _create_tiny_dataset(source_root)

    migrate_dataset(
        source_repo_id=SOURCE_REPO_ID,
        source_root=source_root,
        output_repo_id=OUTPUT_REPO_ID,
        output_root=output_root,
        target_floor=target_floor,
    )

    destination = LeRobotDataset(repo_id=OUTPUT_REPO_ID, root=output_root)
    conditions = _numeric_batch(destination, [FLOOR_CONDITION_KEY])[FLOOR_CONDITION_KEY]
    np.testing.assert_array_equal(
        conditions,
        np.repeat(encode_target_floor(target_floor)[None, :], repeats=5, axis=0),
    )


def test_migrate_dataset_rejects_same_resolved_root(tmp_path):
    source_root = tmp_path / "source"
    _create_tiny_dataset(source_root)

    with pytest.raises(ValueError, match="outside the source"):
        migrate_dataset(
            source_repo_id=SOURCE_REPO_ID,
            source_root=source_root,
            output_repo_id=OUTPUT_REPO_ID,
            output_root=source_root / ".." / "source",
        )


def test_migrate_dataset_rejects_output_nested_under_source(tmp_path):
    source_root = tmp_path / "source"
    output_root = source_root / "conditioned" / "output"
    _create_tiny_dataset(source_root)
    source_hashes_before = _tree_hashes(source_root)

    with pytest.raises(ValueError, match="outside the source"):
        migrate_dataset(
            source_repo_id=SOURCE_REPO_ID,
            source_root=source_root,
            output_repo_id=OUTPUT_REPO_ID,
            output_root=output_root,
        )

    assert not output_root.exists()
    assert _tree_hashes(source_root) == source_hashes_before


def test_migrate_dataset_rejects_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="source dataset"):
        migrate_dataset(
            source_repo_id=SOURCE_REPO_ID,
            source_root=tmp_path / "missing",
            output_repo_id=OUTPUT_REPO_ID,
            output_root=tmp_path / "output",
        )


def test_migrate_dataset_rejects_existing_output(tmp_path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _create_tiny_dataset(source_root)
    output_root.mkdir()

    with pytest.raises(FileExistsError, match="output root"):
        migrate_dataset(
            source_repo_id=SOURCE_REPO_ID,
            source_root=source_root,
            output_repo_id=OUTPUT_REPO_ID,
            output_root=output_root,
        )


def test_migrate_dataset_rejects_already_conditioned_source(tmp_path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _create_tiny_dataset(source_root, conditioned_floor=1)

    with pytest.raises(ValueError, match="already contains"):
        migrate_dataset(
            source_repo_id=SOURCE_REPO_ID,
            source_root=source_root,
            output_repo_id=OUTPUT_REPO_ID,
            output_root=output_root,
        )
    assert not output_root.exists()


@pytest.mark.parametrize("failure_stage", ["add_features", "recompute_stats", "verify_migration"])
def test_migrate_dataset_cleans_transaction_after_failure_and_can_retry(tmp_path, monkeypatch, failure_stage):
    source_root = tmp_path / "source"
    output_root = tmp_path / "new-parent" / "output"
    _create_tiny_dataset(source_root)
    sentinel = output_root.parent / "unrelated-sibling.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("must survive cleanup")
    original = getattr(migration_module, failure_stage)

    if failure_stage == "add_features":

        def fail_after_creating_partial_output(*args, output_dir, **kwargs):
            del args, kwargs
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True)
            (output_dir / "partial").write_text("incomplete")
            raise RuntimeError("injected add_features failure")

        failure = fail_after_creating_partial_output
    else:

        def fail_after_stage_exists(*args, **kwargs):
            del args, kwargs
            raise RuntimeError(f"injected {failure_stage} failure")

        failure = fail_after_stage_exists

    monkeypatch.setattr(migration_module, failure_stage, failure)
    with pytest.raises(RuntimeError, match="injected"):
        migrate_dataset(
            source_repo_id=SOURCE_REPO_ID,
            source_root=source_root,
            output_repo_id=OUTPUT_REPO_ID,
            output_root=output_root,
        )

    assert not output_root.exists()
    assert _migration_temp_dirs(output_root) == []
    assert sentinel.read_text() == "must survive cleanup"

    monkeypatch.setattr(migration_module, failure_stage, original)
    report = migrate_dataset(
        source_repo_id=SOURCE_REPO_ID,
        source_root=source_root,
        output_repo_id=OUTPUT_REPO_ID,
        output_root=output_root,
    )
    assert report.output_root == output_root.resolve()
    assert output_root.is_dir()
    assert _migration_temp_dirs(output_root) == []
    assert sentinel.read_text() == "must survive cleanup"


@pytest.mark.parametrize("target_floor", [0, 6, True, 1.0])
def test_migrate_dataset_rejects_invalid_floor_before_creating_output(tmp_path, target_floor):
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    _create_tiny_dataset(source_root)

    with pytest.raises((TypeError, ValueError)):
        migrate_dataset(
            source_repo_id=SOURCE_REPO_ID,
            source_root=source_root,
            output_repo_id=OUTPUT_REPO_ID,
            output_root=output_root,
            target_floor=target_floor,
        )
    assert not output_root.exists()


def test_verify_video_files_requires_identical_relative_paths_and_bytes(tmp_path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    relative_path = Path("videos/chunk-000/camera/file-000.mp4")
    (source_root / relative_path).parent.mkdir(parents=True)
    (output_root / relative_path).parent.mkdir(parents=True)
    (source_root / relative_path).write_bytes(b"identical video bytes")
    (output_root / relative_path).write_bytes(b"identical video bytes")

    assert _verify_video_files(source_root, output_root) == 1

    (output_root / relative_path).write_bytes(b"different video bytes")
    with pytest.raises(RuntimeError, match="differs byte-for-byte"):
        _verify_video_files(source_root, output_root)

    (output_root / relative_path).unlink()
    with pytest.raises(RuntimeError, match="relative video paths"):
        _verify_video_files(source_root, output_root)


def test_migration_parser_requires_dataset_locations_and_defaults_to_floor_one():
    args = build_arg_parser().parse_args(
        [
            "--source-repo-id",
            SOURCE_REPO_ID,
            "--source-root",
            "/tmp/source",
            "--output-repo-id",
            OUTPUT_REPO_ID,
            "--output-root",
            "/tmp/output",
        ]
    )

    assert args.target_floor == 1


def test_validate_multifloor_datasets_accepts_strict_conditioned_sources(tmp_path):
    specs = _create_conditioned_sources(tmp_path)

    report = validate_multifloor_datasets(specs)

    assert report.total_episodes == 6
    assert report.total_frames == 15
    assert report.floor_episode_counts == {1: 2, 4: 2, 5: 2}
    assert report.state_dim == 8
    assert report.action_dim == 7
    assert report.camera_keys == (CAMERA_KEY,)


def test_validate_multifloor_datasets_requires_completely_equal_feature_schemas(tmp_path):
    specs = _create_conditioned_sources(tmp_path)
    _mutate_info(
        specs[1],
        lambda info: info["features"][CAMERA_KEY].update(
            {"info": {"video.codec": "unexpected-camera-schema"}}
        ),
    )

    with pytest.raises(ValueError, match="feature schemas differ"):
        validate_multifloor_datasets(specs)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dtype", "float64", "dtype float32"),
        ("shape", [4], r"shape \(5,\)"),
        ("names", ["a", "b", "c", "d", "e"], "names must be null"),
    ],
)
def test_validate_dataset_rejects_noncanonical_environment_schema(tmp_path, field, value, message):
    spec = _create_conditioned_sources(tmp_path)[0]
    dataset = _load_spec(spec)
    dataset.meta.features[FLOOR_CONDITION_KEY][field] = value

    with pytest.raises(ValueError, match=message):
        validate_dataset(dataset, expected_floors={1})


def test_validate_dataset_rejects_loaded_environment_float64(tmp_path):
    spec = _create_conditioned_sources(tmp_path)[0]
    dataset = _load_spec(spec)
    conditions = _numeric_batch(dataset, [FLOOR_CONDITION_KEY])[FLOOR_CONDITION_KEY].astype(np.float64)
    _replace_numeric_column(dataset, FLOOR_CONDITION_KEY, conditions)

    with pytest.raises(ValueError, match="values must be float32"):
        validate_dataset(dataset, expected_floors={1})


def test_validate_dataset_rejects_loaded_environment_wrong_shape(tmp_path):
    spec = _create_conditioned_sources(tmp_path)[0]
    dataset = _load_spec(spec)
    conditions = _numeric_batch(dataset, [FLOOR_CONDITION_KEY])[FLOOR_CONDITION_KEY][:, :4]
    _replace_numeric_column(dataset, FLOOR_CONDITION_KEY, conditions)

    with pytest.raises(ValueError, match="values must have shape"):
        validate_dataset(dataset, expected_floors={1})


def test_validate_dataset_rejects_invalid_one_hot(tmp_path):
    spec = _create_conditioned_sources(tmp_path)[0]
    dataset = _load_spec(spec)
    conditions = _numeric_batch(dataset, [FLOOR_CONDITION_KEY])[FLOOR_CONDITION_KEY]
    conditions[0] = 0.0
    _replace_numeric_column(dataset, FLOOR_CONDITION_KEY, conditions)

    with pytest.raises(ValueError, match="one-hot"):
        validate_dataset(dataset, expected_floors={1})


def test_validate_dataset_rejects_floor_change_within_episode(tmp_path):
    spec = _create_conditioned_sources(tmp_path)[0]
    dataset = _load_spec(spec)
    conditions = _numeric_batch(dataset, [FLOOR_CONDITION_KEY])[FLOOR_CONDITION_KEY]
    conditions[1] = encode_target_floor(4)
    _replace_numeric_column(dataset, FLOOR_CONDITION_KEY, conditions)

    with pytest.raises(ValueError, match="changes within episode"):
        validate_dataset(dataset, expected_floors={1})


def test_validate_dataset_rejects_floor_different_from_source_declaration(tmp_path):
    spec = _create_conditioned_sources(tmp_path)[0]
    dataset = _load_spec(spec)
    conditions = np.repeat(encode_target_floor(4)[None, :], len(dataset), axis=0)
    _replace_numeric_column(dataset, FLOOR_CONDITION_KEY, conditions)

    with pytest.raises(ValueError, match="expected floors.*1.*observed.*4"):
        validate_dataset(dataset, expected_floors={1})


@pytest.mark.parametrize(("key", "bad_value"), [(OBS_STATE, np.nan), (ACTION, np.inf)])
def test_validate_dataset_rejects_nonfinite_state_or_action(tmp_path, key, bad_value):
    spec = _create_conditioned_sources(tmp_path)[0]
    dataset = _load_spec(spec)
    values = _numeric_batch(dataset, [key])[key]
    values[0, 0] = bad_value
    _replace_numeric_column(dataset, key, values)

    with pytest.raises(ValueError, match=f"{key}.*NaN or Inf"):
        validate_dataset(dataset, expected_floors={1})


def test_validate_dataset_rejects_nonfinite_timestamp(tmp_path):
    spec = _create_conditioned_sources(tmp_path)[0]
    dataset = _load_spec(spec)
    timestamps = _numeric_batch(dataset, ["timestamp"])["timestamp"]
    timestamps[0] = np.nan
    _replace_numeric_column(dataset, "timestamp", timestamps)

    with pytest.raises(ValueError, match="timestamp.*NaN or Inf"):
        validate_dataset(dataset, expected_floors={1}, skip_image_decode=True)


def test_validate_dataset_rejects_nonfinite_additional_numeric_feature(tmp_path):
    spec = _create_conditioned_sources(tmp_path)[0]
    dataset = _load_spec(spec)
    feature_key = "diagnostics.force"
    dataset.meta.features[feature_key] = {"dtype": "float32", "shape": (2,), "names": None}
    values = np.zeros((len(dataset), 2), dtype=np.float32)
    values[-1, -1] = np.inf
    _add_numeric_column(dataset, feature_key, values)

    with pytest.raises(ValueError, match="diagnostics.force.*NaN or Inf"):
        validate_dataset(dataset, expected_floors={1}, skip_image_decode=True)


def test_validate_dataset_rejects_missing_declared_numeric_column(tmp_path):
    spec = _create_conditioned_sources(tmp_path)[0]
    dataset = _load_spec(spec)
    dataset.meta.features["diagnostics.missing"] = {
        "dtype": "float32",
        "shape": (2,),
        "names": None,
    }

    with pytest.raises(ValueError, match="numeric feature diagnostics.missing.*missing frame column"):
        validate_dataset(dataset, expected_floors={1}, skip_image_decode=True)


def test_validate_dataset_accepts_declared_integer_features_including_task_index(tmp_path):
    spec = _create_conditioned_sources(tmp_path)[0]
    dataset = _load_spec(spec)
    feature_key = "diagnostics.flags"
    dataset.meta.features[feature_key] = {"dtype": "int32", "shape": (2,), "names": None}
    values = np.arange(len(dataset) * 2, dtype=np.int32).reshape(len(dataset), 2)
    _add_numeric_column(dataset, feature_key, values)

    report = validate_dataset(dataset, expected_floors={1}, skip_image_decode=True)

    assert report.total_frames == 5
    assert dataset.meta.features["task_index"]["dtype"] == "int64"


def test_validate_dataset_requires_one_dimensional_state_and_action_schema(tmp_path):
    spec = _create_conditioned_sources(tmp_path)[0]
    dataset = _load_spec(spec)
    dataset.meta.features[OBS_STATE]["shape"] = (8, 1)

    with pytest.raises(ValueError, match="observation.state.*one-dimensional"):
        validate_dataset(dataset, expected_floors={1})


@pytest.mark.parametrize("metadata_failure", ["missing", "empty", "gapped", "zero-length"])
def test_validate_dataset_rejects_invalid_episode_metadata(tmp_path, metadata_failure):
    spec = _create_conditioned_sources(tmp_path)[0]
    dataset = _load_spec(spec)
    if metadata_failure == "missing":
        dataset.meta.episodes = None
    elif metadata_failure == "empty":
        dataset.meta.episodes = dataset.meta.episodes.select([])
    elif metadata_failure == "gapped":
        dataset.meta.episodes = dataset.meta.episodes.map(
            lambda row: {"episode_index": row["episode_index"] + 1}
        )
    else:
        dataset.meta.episodes = dataset.meta.episodes.map(
            lambda row, index: {"dataset_to_index": row["dataset_from_index"]} if index == 0 else {},
            with_indices=True,
        )

    with pytest.raises(ValueError, match="episode metadata|empty episode"):
        validate_dataset(dataset, expected_floors={1})


def test_validate_dataset_rejects_missing_image_column(tmp_path):
    spec = _create_conditioned_sources(tmp_path)[0]
    dataset = _load_spec(spec)
    dataset.reader.hf_dataset = dataset.hf_dataset.remove_columns(CAMERA_KEY)

    with pytest.raises(ValueError, match="missing image column"):
        validate_dataset(dataset, expected_floors={1})


def test_validate_dataset_rejects_corrupt_image_decode(tmp_path, monkeypatch):
    spec = _create_conditioned_sources(tmp_path)[0]
    dataset = _load_spec(spec)
    original_get_item = dataset.reader.get_item

    def fail_first_decode(index):
        if index == 0:
            raise OSError("corrupt image bytes")
        return original_get_item(index)

    monkeypatch.setattr(dataset.reader, "get_item", fail_first_decode)

    with pytest.raises(ValueError, match="failed to decode.*frame 0"):
        validate_dataset(dataset, expected_floors={1})


def test_validate_video_dataset_uses_sequential_video_decode(tmp_path, monkeypatch):
    dataset = _create_tiny_video_dataset(tmp_path / "video")

    def reject_per_frame_random_access(index):
        pytest.fail(f"video validation used random dataset access at frame {index}")

    monkeypatch.setattr(dataset.reader, "get_item", reject_per_frame_random_access)

    report = validate_dataset(dataset, expected_floors={4})

    assert report.total_episodes == 2
    assert report.total_frames == 5
    assert report.camera_keys == (CAMERA_KEY,)


def test_validate_dataset_skip_image_decode_still_checks_embedded_media_column(tmp_path):
    spec = _create_conditioned_sources(tmp_path)[0]
    dataset = _load_spec(spec)

    report = validate_dataset(dataset, expected_floors={1}, skip_image_decode=True)

    assert report.total_frames == 5
    assert report.camera_keys == (CAMERA_KEY,)


def test_validate_dataset_skip_image_decode_rejects_missing_embedded_payload(tmp_path):
    spec = _create_conditioned_sources(tmp_path)[0]
    dataset = _load_spec(spec)
    raw = dataset.hf_dataset.with_format(None)
    dataset.reader.hf_dataset = raw.remove_columns(CAMERA_KEY).add_column(
        CAMERA_KEY,
        [None] * len(dataset),
        feature=Image(),
    )

    with pytest.raises(ValueError, match="missing embedded image payload"):
        validate_dataset(dataset, expected_floors={1}, skip_image_decode=True)


def test_merge_validated_datasets_uses_official_merge_and_preserves_floor_counts(
    merged_multifloor_dataset,
):
    specs, report = merged_multifloor_dataset

    assert validator_module.merge_datasets.__module__ == "lerobot.datasets.dataset_tools"
    assert report.output_root.is_dir()
    assert report.total_episodes == 6
    assert report.total_frames == 15
    assert report.floor_episode_counts == {1: 2, 4: 2, 5: 2}
    merged = LeRobotDataset(repo_id=report.output_repo_id, root=report.output_root)
    episode_indices = np.asarray(
        merged.hf_dataset.select_columns(["episode_index"]).with_format("numpy")[:]["episode_index"]
    )
    assert sorted(set(episode_indices.tolist())) == list(range(6))
    assert {spec.floor for spec in specs} == {1, 4, 5}


def test_merge_validated_datasets_aggregates_multiple_single_episode_roots_for_one_floor(tmp_path):
    specs = []
    for index in (1, 2):
        root = tmp_path / f"floor-4-{index:02d}"
        repo_id = f"local/floor-4-{index:02d}"
        _create_tiny_dataset(
            root,
            repo_id=repo_id,
            conditioned_floor=4,
            state_dim=8,
            action_dim=7,
            episode_lengths=(2,),
        )
        specs.append(DatasetSpec(floor=4, repo_id=repo_id, root=root))

    report = merge_validated_datasets(
        specs,
        output_repo_id="local/floor-4-conditioned",
        output_root=tmp_path / "floor-4-conditioned",
        expected_floors={4},
    )

    assert report.total_episodes == 2
    assert report.total_frames == 4
    assert report.floor_episode_counts == {4: 2}


def test_merge_validated_datasets_rejects_duplicate_resolved_source_roots(tmp_path):
    root = tmp_path / "floor-4-01"
    repo_id = "local/floor-4-01"
    _create_tiny_dataset(
        root,
        repo_id=repo_id,
        conditioned_floor=4,
        state_dim=8,
        action_dim=7,
        episode_lengths=(2,),
    )
    spec = DatasetSpec(floor=4, repo_id=repo_id, root=root)

    with pytest.raises(ValueError, match="duplicate source dataset root"):
        merge_validated_datasets(
            [spec, spec],
            output_repo_id="local/floor-4-duplicate-test",
            output_root=tmp_path / "output",
            expected_floors={4},
        )


@pytest.mark.parametrize(
    "floors",
    [(1, 4), (1, 4, 4), (1, 2, 5)],
)
def test_merge_validated_datasets_requires_exact_unique_floor_set(tmp_path, floors):
    specs = tuple(
        DatasetSpec(floor=floor, repo_id=f"local/floor-{index}", root=tmp_path / str(index))
        for index, floor in enumerate(floors)
    )

    with pytest.raises(ValueError, match="cover exactly floors 1, 4, and 5"):
        merge_validated_datasets(specs, output_root=tmp_path / "output")


def test_merge_validated_datasets_rejects_existing_or_nested_output(tmp_path):
    specs = _create_conditioned_sources(tmp_path / "sources")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="output root"):
        merge_validated_datasets(specs, output_root=existing)

    nested = specs[0].root / "nested-output"
    with pytest.raises(ValueError, match="outside every source"):
        merge_validated_datasets(specs, output_root=nested)


def test_smoke_test_act_training_runs_real_finite_forward(merged_multifloor_dataset):
    _, merged_report = merged_multifloor_dataset

    report = smoke_test_act_training(
        repo_id=merged_report.output_repo_id,
        root=merged_report.output_root,
    )

    assert report.environment_shape == (1, 5)
    assert report.state_shape[-1] == 8
    assert report.action_shape[-1] == 7
    assert math.isfinite(report.loss)


def test_validation_parser_supports_repeatable_datasets_and_requested_defaults(tmp_path):
    args = build_validation_arg_parser().parse_args(
        [
            "--dataset",
            "1",
            "local/floor-1",
            str(tmp_path / "floor-1"),
            "--dataset",
            "4",
            "local/floor-4",
            str(tmp_path / "floor-4"),
            "--dataset",
            "5",
            "local/floor-5",
            str(tmp_path / "floor-5"),
            "--merge",
            "--smoke-test",
            "--skip-image-decode",
        ]
    )

    assert args.dataset == [
        ["1", "local/floor-1", str(tmp_path / "floor-1")],
        ["4", "local/floor-4", str(tmp_path / "floor-4")],
        ["5", "local/floor-5", str(tmp_path / "floor-5")],
    ]
    assert args.output_repo_id == DEFAULT_OUTPUT_REPO_ID
    assert args.expected_floors == [1, 4, 5]
    assert args.merge is True
    assert args.smoke_test is True
    assert args.skip_image_decode is True

    floor_parts_args = build_validation_arg_parser().parse_args(
        [
            "--dataset",
            "4",
            "local/floor-4-01",
            str(tmp_path / "floor-4-01"),
            "--expected-floors",
            "4",
            "--merge",
        ]
    )
    assert floor_parts_args.expected_floors == [4]
