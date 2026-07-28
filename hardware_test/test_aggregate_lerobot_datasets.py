import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import hardware_test.aggregate_lerobot_datasets as aggregate_module
from hardware_test.aggregate_lerobot_datasets import (
    _validate_dataset_integrity,
    _validate_local_dataset_root,
    build_arg_parser,
    build_source_roots,
    main,
    merge_recording_group,
    parse_source_name,
    resolve_output_root,
)


class FakeHFDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows)

    def __len__(self):
        return len(next(iter(self.rows.values())))

    def select_columns(self, columns):
        return FakeHFDataset({column: self.rows[column] for column in columns})

    def with_format(self, _format):
        return self

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.rows
        return {key: values[index] for key, values in self.rows.items()}


class FakeMeta:
    def __init__(self, *, total_episodes, total_frames, video_keys=()):
        self.total_episodes = total_episodes
        self.total_frames = total_frames
        self.video_keys = video_keys
        self.episodes = [{"length": total_frames}] if total_episodes == 1 else None

    def get_data_file_path(self, episode_index):
        return Path(f"data-{episode_index}.parquet")

    def get_video_file_path(self, episode_index, video_key):
        return Path(f"{video_key}-{episode_index}.mp4")


def make_fake_dataset(tmp_path, *, rows, total_episodes=1, total_frames=None, video_keys=()):
    total_frames = len(rows["index"]) if total_frames is None else total_frames
    for episode_index in range(total_episodes):
        (tmp_path / f"data-{episode_index}.parquet").write_bytes(b"parquet")
    for video_key in video_keys:
        for episode_index in range(total_episodes):
            (tmp_path / f"{video_key}-{episode_index}.mp4").write_bytes(b"video")
    return SimpleNamespace(
        repo_id="local/fake",
        root=tmp_path,
        meta=FakeMeta(
            total_episodes=total_episodes,
            total_frames=total_frames,
            video_keys=video_keys,
        ),
        hf_dataset=FakeHFDataset(rows),
    )


def stub_orchestration_validation(monkeypatch):
    monkeypatch.setattr(aggregate_module, "_validate_local_dataset_root", lambda root: None)
    monkeypatch.setattr(
        aggregate_module,
        "_validate_dataset_integrity",
        lambda dataset, *, expected_episodes: (
            dataset.meta.total_episodes,
            dataset.meta.total_frames,
        ),
    )


def create_preflight_source(root, *, referenced_data=True, referenced_video=True):
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "camera" / "chunk-000").mkdir(parents=True)
    info = {
        "total_episodes": 1,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {"camera": {"dtype": "video"}},
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "stats.json").write_text("{}")
    (root / "meta" / "tasks.parquet").write_bytes(b"tasks")
    (root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").write_bytes(b"episodes")
    data_name = "file-000.parquet" if referenced_data else "unreferenced.parquet"
    (root / "data" / "chunk-000" / data_name).write_bytes(b"data")
    if referenced_video:
        (root / "videos" / "camera" / "chunk-000" / "file-000.mp4").write_bytes(b"video")


def test_parse_source_name_extracts_prefix_from_first_numbered_recording():
    group = parse_source_name("press_button_01")

    assert group.prefix == "press_button_"
    assert group.width == 2


@pytest.mark.parametrize("source_name", ["press_button_", "press_button_02", "press_button_1"])
def test_parse_source_name_requires_a_two_digit_01_suffix(source_name):
    with pytest.raises(ValueError, match="end with '01'"):
        parse_source_name(source_name)


@pytest.mark.parametrize("source_name", ["../press_01", "nested/press_01", "/tmp/press_01"])
def test_parse_source_name_keeps_sources_under_datasets_root(source_name):
    with pytest.raises(ValueError, match="single directory name"):
        parse_source_name(source_name)


def test_build_source_roots_returns_01_through_30_in_numeric_order(tmp_path):
    expected = []
    for index in range(1, 31):
        root = tmp_path / f"recording_{index:02d}"
        root.mkdir()
        expected.append(root.resolve())

    roots = build_source_roots(tmp_path, "recording_01")

    assert roots == tuple(expected)


def test_build_source_roots_uses_cli_configurable_count(tmp_path):
    expected = []
    for index in range(1, 5):
        root = tmp_path / f"recording_{index:02d}"
        root.mkdir()
        expected.append(root.resolve())

    roots = build_source_roots(tmp_path, "recording_01", count=4)

    assert roots == tuple(expected)


@pytest.mark.parametrize("count", [0, -1])
def test_build_source_roots_rejects_non_positive_count(tmp_path, count):
    with pytest.raises(ValueError, match="count must be positive"):
        build_source_roots(tmp_path, "recording_01", count=count)


def test_build_source_roots_reports_every_missing_number(tmp_path):
    for index in range(1, 31):
        if index not in {4, 17}:
            (tmp_path / f"press_{index:02d}").mkdir()

    with pytest.raises(FileNotFoundError, match=r"press_04.*press_17"):
        build_source_roots(tmp_path, "press_01")


def test_validate_local_dataset_root_rejects_an_incomplete_directory(tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("{}")

    with pytest.raises(ValueError, match="incomplete local LeRobot dataset"):
        _validate_local_dataset_root(tmp_path)


@pytest.mark.parametrize(
    ("missing_kind", "expected_error"),
    [("data", "missing or empty referenced data"), ("video", "missing or empty referenced video")],
)
def test_merge_rejects_missing_metadata_references_before_calling_loader(
    tmp_path, monkeypatch, missing_kind, expected_error
):
    for index in range(1, 31):
        create_preflight_source(
            tmp_path / f"press_{index:02d}",
            referenced_data=missing_kind != "data" or index != 1,
            referenced_video=missing_kind != "video" or index != 1,
        )

    monkeypatch.setattr(
        aggregate_module,
        "_read_episode_records",
        lambda paths: [
            {
                "episode_index": 0,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "videos/camera/chunk_index": 0,
                "videos/camera/file_index": 0,
            }
        ],
    )
    loader_calls = []
    monkeypatch.setattr(aggregate_module, "_load_dataset", lambda *args, **kwargs: loader_calls.append(args))

    with pytest.raises(ValueError, match=expected_error):
        merge_recording_group(tmp_path, "press_01", "train")

    assert loader_calls == []


def test_validate_dataset_integrity_checks_actual_rows_indices_and_media(tmp_path):
    dataset = make_fake_dataset(
        tmp_path,
        rows={
            "episode_index": [0, 0],
            "frame_index": [0, 1],
            "index": [0, 1],
        },
        video_keys=("camera",),
    )

    assert _validate_dataset_integrity(dataset, expected_episodes=1) == (1, 2)


def test_validate_dataset_integrity_rejects_multi_episode_source(tmp_path):
    dataset = make_fake_dataset(
        tmp_path,
        rows={
            "episode_index": [0, 1],
            "frame_index": [0, 0],
            "index": [0, 1],
        },
        total_episodes=2,
    )

    with pytest.raises(ValueError, match="expected 1 episode"):
        _validate_dataset_integrity(dataset, expected_episodes=1)


def test_validate_dataset_integrity_rejects_metadata_frame_count_mismatch(tmp_path):
    dataset = make_fake_dataset(
        tmp_path,
        rows={
            "episode_index": [0],
            "frame_index": [0],
            "index": [0],
        },
        total_frames=2,
    )

    with pytest.raises(ValueError, match="metadata reports 2 frames.*actual parquet data has 1"):
        _validate_dataset_integrity(dataset, expected_episodes=1)


def test_validate_dataset_integrity_rejects_non_contiguous_indices(tmp_path):
    dataset = make_fake_dataset(
        tmp_path,
        rows={
            "episode_index": [0, 0],
            "frame_index": [0, 2],
            "index": [0, 3],
        },
    )

    with pytest.raises(ValueError, match="non-contiguous global index"):
        _validate_dataset_integrity(dataset, expected_episodes=1)


def test_validate_dataset_integrity_rejects_missing_referenced_video(tmp_path):
    dataset = make_fake_dataset(
        tmp_path,
        rows={
            "episode_index": [0],
            "frame_index": [0],
            "index": [0],
        },
        video_keys=("camera",),
    )
    (tmp_path / "camera-0.mp4").unlink()

    with pytest.raises(ValueError, match="missing or empty video"):
        _validate_dataset_integrity(dataset, expected_episodes=1)


@pytest.mark.parametrize("output_name", ["", ".", "../train", "nested/train", "/tmp/train"])
def test_resolve_output_root_keeps_output_under_datasets_root(tmp_path, output_name):
    with pytest.raises(ValueError, match="single directory name"):
        resolve_output_root(tmp_path, output_name)


def test_resolve_output_root_rejects_an_existing_output(tmp_path):
    output = tmp_path / "press_train"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        resolve_output_root(tmp_path, "press_train")


def test_resolve_output_root_accepts_a_new_cli_output_name(tmp_path):
    assert resolve_output_root(tmp_path, "my_training_set") == (tmp_path / "my_training_set").resolve()


def test_merge_recording_group_uses_official_merge_and_publishes_verified_output(tmp_path, monkeypatch):
    stub_orchestration_validation(monkeypatch)
    source_roots = []
    frames_by_root = {}
    for index in range(1, 5):
        root = tmp_path / f"press_{index:02d}"
        root.mkdir()
        source_roots.append(root.resolve())
        frames_by_root[root.resolve()] = index

    class FakeDataset:
        def __init__(self, repo_id, root):
            self.repo_id = repo_id
            self.root = Path(root).resolve()
            self.meta = SimpleNamespace(total_episodes=1, total_frames=frames_by_root[self.root])

    merge_calls = []

    def fake_merge(datasets, output_repo_id, output_dir):
        merge_calls.append((datasets, output_repo_id, Path(output_dir)))
        Path(output_dir).mkdir(parents=True)
        return SimpleNamespace(
            meta=SimpleNamespace(
                total_episodes=sum(dataset.meta.total_episodes for dataset in datasets),
                total_frames=sum(dataset.meta.total_frames for dataset in datasets),
            )
        )

    monkeypatch.setattr(aggregate_module, "_load_dataset", FakeDataset)
    monkeypatch.setattr(aggregate_module, "_run_official_merge", fake_merge)

    report = merge_recording_group(tmp_path, "press_01", "press_train", count=4)

    datasets, output_repo_id, staging_root = merge_calls[0]
    assert [dataset.root for dataset in datasets] == source_roots
    assert [dataset.repo_id for dataset in datasets] == [f"local/press_{index:02d}" for index in range(1, 5)]
    assert output_repo_id == "local/press_train"
    assert not staging_root.exists()
    assert report.output_root == (tmp_path / "press_train").resolve()
    assert report.output_root.is_dir()
    assert report.total_episodes == 4
    assert report.total_frames == sum(range(1, 5))


def test_merge_recording_group_removes_staging_data_when_merge_fails(tmp_path, monkeypatch):
    stub_orchestration_validation(monkeypatch)
    for index in range(1, 31):
        (tmp_path / f"press_{index:02d}").mkdir()

    class FakeDataset:
        def __init__(self, repo_id, root):
            self.repo_id = repo_id
            self.root = Path(root)
            self.meta = SimpleNamespace(total_episodes=1, total_frames=1)

    def failing_merge(datasets, output_repo_id, output_dir):
        Path(output_dir).mkdir(parents=True)
        (Path(output_dir) / "partial").write_text("incomplete")
        raise RuntimeError("merge failed")

    monkeypatch.setattr(aggregate_module, "_load_dataset", FakeDataset)
    monkeypatch.setattr(aggregate_module, "_run_official_merge", failing_merge)

    with pytest.raises(RuntimeError, match="merge failed"):
        merge_recording_group(tmp_path, "press_01", "press_train")

    assert not (tmp_path / "press_train").exists()
    assert not list(tmp_path.glob(".press_train.merge-*"))


def test_merge_recording_group_reports_staging_cleanup_failure(tmp_path, monkeypatch):
    stub_orchestration_validation(monkeypatch)
    for index in range(1, 31):
        (tmp_path / f"press_{index:02d}").mkdir()

    class FakeDataset:
        def __init__(self, repo_id, root):
            self.repo_id = repo_id
            self.root = Path(root)
            self.meta = SimpleNamespace(total_episodes=1, total_frames=1)

    def failing_merge(datasets, output_repo_id, output_dir):
        Path(output_dir).mkdir(parents=True)
        raise RuntimeError("merge failed")

    monkeypatch.setattr(aggregate_module, "_load_dataset", FakeDataset)
    monkeypatch.setattr(aggregate_module, "_run_official_merge", failing_merge)
    monkeypatch.setattr(
        aggregate_module.shutil,
        "rmtree",
        lambda path: (_ for _ in ()).throw(OSError("read-only filesystem")),
    )

    with pytest.raises(RuntimeError, match="failed to remove staging directory"):
        merge_recording_group(tmp_path, "press_01", "press_train")


def test_merge_recording_group_rejects_changed_episode_or_frame_totals(tmp_path, monkeypatch):
    stub_orchestration_validation(monkeypatch)
    for index in range(1, 31):
        (tmp_path / f"press_{index:02d}").mkdir()

    class FakeDataset:
        def __init__(self, repo_id, root):
            self.repo_id = repo_id
            self.root = Path(root)
            self.meta = SimpleNamespace(total_episodes=1, total_frames=2)

    def lossy_merge(datasets, output_repo_id, output_dir):
        Path(output_dir).mkdir(parents=True)
        return SimpleNamespace(meta=SimpleNamespace(total_episodes=29, total_frames=59))

    monkeypatch.setattr(aggregate_module, "_load_dataset", FakeDataset)
    monkeypatch.setattr(aggregate_module, "_run_official_merge", lossy_merge)

    with pytest.raises(RuntimeError, match="changed dataset totals"):
        merge_recording_group(tmp_path, "press_01", "press_train")

    assert not (tmp_path / "press_train").exists()


def test_official_merge_keeps_individual_data_and_video_files(tmp_path, monkeypatch):
    calls = []

    def fake_merge(datasets, **kwargs):
        calls.append((datasets, kwargs))
        return "merged"

    monkeypatch.setitem(sys.modules, "lerobot.datasets", SimpleNamespace(merge_datasets=fake_merge))

    result = aggregate_module._run_official_merge(["one", "two"], "local/train", tmp_path / "train")

    assert result == "merged"
    assert calls == [
        (
            ["one", "two"],
            {
                "output_repo_id": "local/train",
                "output_dir": tmp_path / "train",
                "concatenate_videos": False,
                "concatenate_data": False,
            },
        )
    ]


def test_cli_requires_source_and_output_names_and_defaults_to_hardware_outputs():
    args = build_arg_parser().parse_args(
        ["--source-name", "press_button_01", "--output-name", "press_button_train_new"]
    )

    assert args.datasets_root == Path("outputs/hardware_test")
    assert args.source_name == "press_button_01"
    assert args.output_name == "press_button_train_new"
    assert args.count == 30
    assert args.dry_run is False


def test_cli_accepts_custom_dataset_count():
    args = build_arg_parser().parse_args(
        [
            "--source-name",
            "press_button_01",
            "--output-name",
            "press_button_train_new",
            "--count",
            "12",
        ]
    )

    assert args.count == 12


def test_main_dry_run_checks_the_group_without_creating_output(tmp_path, capsys):
    for index in range(1, 31):
        (tmp_path / f"new_task_{index:02d}").mkdir()

    exit_code = main(
        [
            "--datasets-root",
            str(tmp_path),
            "--source-name",
            "new_task_01",
            "--output-name",
            "new_task_train",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert not (tmp_path / "new_task_train").exists()
    output = capsys.readouterr().out
    assert "30 source datasets" in output
    assert str((tmp_path / "new_task_train").resolve()) in output


def test_main_dry_run_uses_custom_dataset_count(tmp_path, capsys):
    for index in range(1, 6):
        (tmp_path / f"short_task_{index:02d}").mkdir()

    exit_code = main(
        [
            "--datasets-root",
            str(tmp_path),
            "--source-name",
            "short_task_01",
            "--output-name",
            "short_task_train",
            "--count",
            "5",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert "5 source datasets" in capsys.readouterr().out


def test_main_reports_expected_cli_errors_without_a_traceback(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--datasets-root",
                str(tmp_path),
                "--source-name",
                "missing_01",
                "--output-name",
                "train",
                "--dry-run",
            ]
        )

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "missing source dataset directories" in error
    assert "Traceback" not in error
