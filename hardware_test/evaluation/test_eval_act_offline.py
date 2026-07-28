from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hardware_test.evaluation.eval_act_offline import (
    EXPECTED_ACTION_NAMES,
    EXPECTED_STATE_NAMES,
    EvaluationRuntime,
    compute_episode_metrics,
    evaluate_policy_on_dataset,
    format_terminal_summary,
    load_evaluation_runtime,
    parse_args,
    rotation_geodesic_degrees,
    run_evaluation,
    save_evaluation_outputs,
    summarize_episode_metrics,
    validate_act_dataset_contract,
    validate_runtime_paths,
)


def _chunks(frames: int, horizon: int, value: float) -> np.ndarray:
    chunks = np.zeros((frames, horizon, 7), dtype=np.float64)
    chunks[..., 0] = value
    return chunks


def test_compute_episode_metrics_pairs_horizons_and_ignores_padding() -> None:
    pred_physical = np.zeros((1, 3, 7), dtype=np.float64)
    gt_physical = np.zeros_like(pred_physical)
    pred_physical[0, :, 0] = [0.001, 0.002, 100.0]
    pred_physical[0, :, 6] = [0.4, 0.6, 100.0]

    pred_normalized = np.zeros_like(pred_physical)
    gt_normalized = np.zeros_like(pred_physical)
    pred_normalized[0, :, 0] = [1.0, 3.0, 10_000.0]
    action_is_pad = np.array([[False, False, True]])

    metrics = compute_episode_metrics(
        episode_id=7,
        pred_physical=pred_physical,
        gt_physical=gt_physical,
        pred_normalized=pred_normalized,
        gt_normalized=gt_normalized,
        action_is_pad=action_is_pad,
    )

    assert metrics.episode_id == 7
    assert metrics.num_frames == 1
    assert metrics.valid_pairs == 2
    assert metrics.eval_normalized_l1 == pytest.approx(2.0 / 7.0)
    assert metrics.translation_error_mm == pytest.approx(1.5)
    assert metrics.rotation_error_deg == pytest.approx(0.0)
    assert metrics.ade_mm == pytest.approx(2.0)
    assert metrics.fde_mm == pytest.approx(3.0)
    assert metrics.xyz_rmse_mm == pytest.approx(math.sqrt(2.5))
    assert metrics.action_mae[0] == pytest.approx(0.0015)
    assert metrics.gripper_mae == pytest.approx(0.5)
    assert metrics.gripper_accuracy == pytest.approx(0.5)
    np.testing.assert_allclose(metrics.horizon_translation_error_mm[:2], [1.0, 2.0])
    np.testing.assert_allclose(metrics.horizon_trajectory_error_mm[:2], [1.0, 3.0])
    assert math.isnan(metrics.horizon_translation_error_mm[2])
    np.testing.assert_array_equal(metrics.horizon_valid_pairs, [1, 1, 0])


def test_rotation_geodesic_degrees_uses_rpy_rotation_matrices() -> None:
    zero = np.zeros((3, 3), dtype=np.float64)
    pred = zero.copy()
    pred[0, 0] = math.pi / 2
    pred[1, 1] = math.pi
    pred[2, 2] = -math.pi / 4

    result = rotation_geodesic_degrees(pred, zero)

    np.testing.assert_allclose(result, [90.0, 180.0, 45.0], atol=1e-7)


def test_rotation_geodesic_degrees_preserves_combined_rpy_composition_order() -> None:
    pred = np.array([[0.3, -0.4, 0.7]])
    gt = np.array([[-0.2, 0.5, -0.1]])

    result = rotation_geodesic_degrees(pred, gt)

    assert result[0] == pytest.approx(72.12041810401082)


def test_compute_episode_metrics_rejects_non_prefix_padding() -> None:
    chunks = _chunks(frames=1, horizon=3, value=0.0)

    with pytest.raises(ValueError, match="contiguous valid prefix"):
        compute_episode_metrics(
            episode_id=0,
            pred_physical=chunks,
            gt_physical=chunks,
            pred_normalized=chunks,
            gt_normalized=chunks,
            action_is_pad=np.array([[False, True, False]]),
        )


def test_compute_episode_metrics_rejects_nan() -> None:
    pred = _chunks(frames=1, horizon=2, value=0.0)
    pred[0, 0, 0] = np.nan
    gt = _chunks(frames=1, horizon=2, value=0.0)

    with pytest.raises(ValueError, match="pred_physical contains NaN or Inf"):
        compute_episode_metrics(
            episode_id=0,
            pred_physical=pred,
            gt_physical=gt,
            pred_normalized=gt,
            gt_normalized=gt,
            action_is_pad=np.zeros((1, 2), dtype=bool),
        )


def test_compute_episode_metrics_rejects_non_binary_expert_gripper() -> None:
    pred = _chunks(frames=1, horizon=2, value=0.0)
    gt = pred.copy()
    gt[0, 0, 6] = 0.25

    with pytest.raises(ValueError, match="expert gripper"):
        compute_episode_metrics(
            episode_id=0,
            pred_physical=pred,
            gt_physical=gt,
            pred_normalized=pred,
            gt_normalized=gt,
            action_is_pad=np.zeros((1, 2), dtype=bool),
        )


def test_summary_weights_episodes_equally_instead_of_weighting_frames() -> None:
    short_gt = _chunks(frames=1, horizon=1, value=0.0)
    short_pred = _chunks(frames=1, horizon=1, value=0.001)
    long_gt = _chunks(frames=10, horizon=1, value=0.0)
    long_pred = _chunks(frames=10, horizon=1, value=0.009)

    short = compute_episode_metrics(
        0, short_pred, short_gt, short_pred, short_gt, np.zeros((1, 1), dtype=bool)
    )
    long = compute_episode_metrics(1, long_pred, long_gt, long_pred, long_gt, np.zeros((10, 1), dtype=bool))

    summary = summarize_episode_metrics([short, long])

    assert summary.metrics["translation_error_mm"].mean == pytest.approx(5.0)
    assert summary.metrics["translation_error_mm"].std == pytest.approx(math.sqrt(32.0))
    assert summary.horizon_translation_mean_mm[0] == pytest.approx(5.0)
    assert summary.horizon_episode_count[0] == 2
    assert summary.horizon_valid_pairs[0] == 11


def _valid_policy_config(horizon: int = 3) -> SimpleNamespace:
    def feature(shape: tuple[int, ...]) -> SimpleNamespace:
        return SimpleNamespace(shape=shape)

    return SimpleNamespace(
        type="act",
        chunk_size=horizon,
        input_features={
            "observation.state": feature((8,)),
            "observation.images.l515": feature((3, 540, 960)),
        },
        output_features={"action": feature((7,))},
    )


def _valid_dataset_features() -> dict[str, dict]:
    return {
        "action": {"shape": (7,), "names": list(EXPECTED_ACTION_NAMES)},
        "observation.state": {"shape": (8,), "names": list(EXPECTED_STATE_NAMES)},
        "observation.images.l515": {"shape": (540, 960, 3), "names": ["height", "width", "channel"]},
    }


def test_validate_contract_accepts_exact_recording_and_checkpoint_schema() -> None:
    validate_act_dataset_contract(_valid_policy_config(), _valid_dataset_features())


def test_validate_contract_rejects_unknown_action_semantics() -> None:
    features = _valid_dataset_features()
    features["action"]["names"][0] = "ee_pose.x"

    with pytest.raises(ValueError, match="action names"):
        validate_act_dataset_contract(_valid_policy_config(), features)


def test_parse_args_requires_unique_nonnegative_episode_ids(tmp_path: Path) -> None:
    common = [
        "--checkpoint",
        str(tmp_path / "checkpoint"),
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--output-dir",
        str(tmp_path / "out"),
        "--episodes",
    ]

    args = parse_args([*common, "4", "1", "9", "--device", "cpu"])
    assert args.episodes == [4, 1, 9]
    assert args.device == "cpu"

    with pytest.raises(ValueError, match="unique"):
        parse_args([*common, "4", "4"])
    with pytest.raises(ValueError, match="non-negative"):
        parse_args([*common, "-1"])


def test_save_outputs_writes_requested_files_and_explicit_metric_schema(tmp_path: Path) -> None:
    gt = _chunks(frames=2, horizon=2, value=0.0)
    pred_a = _chunks(frames=2, horizon=2, value=0.001)
    pred_b = _chunks(frames=2, horizon=2, value=0.003)
    pad = np.zeros((2, 2), dtype=bool)
    episodes = [
        compute_episode_metrics(3, pred_a, gt, pred_a, gt, pad),
        compute_episode_metrics(8, pred_b, gt, pred_b, gt, pad),
    ]
    summary = summarize_episode_metrics(episodes)

    paths = save_evaluation_outputs(
        output_dir=tmp_path,
        episode_metrics=episodes,
        summary=summary,
        checkpoint="/checkpoints/100000/pretrained_model",
        dataset_root="/datasets/fixed_test",
        requested_episode_ids=[3, 8],
        fps=30.0,
        chunk_size=2,
        device="cuda",
        action_names=EXPECTED_ACTION_NAMES,
    )

    assert set(paths) == {"per_episode", "summary", "horizon", "plot"}
    assert (
        (tmp_path / "per_episode_metrics.csv")
        .read_text()
        .splitlines()[0]
        .startswith("episode_id,num_frames,valid_pairs,eval_normalized_l1")
    )
    summary_text = (tmp_path / "summary_metrics.json").read_text()
    assert '"aggregation": "episode_mean_then_unweighted_mean_std"' in summary_text
    assert '"translation_error_mm"' in summary_text
    assert '"xyz_rmse_mm": "sqrt(mean' in summary_text
    assert (tmp_path / "horizon_translation_error.csv").read_text().splitlines()[0] == (
        "horizon,time_s,translation_error_mean_mm,translation_error_std_mm,"
        "trajectory_error_mean_mm,trajectory_error_std_mm,episode_count,valid_pairs"
    )
    assert (tmp_path / "horizon_error.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


class _FakeArrowColumn:
    def __init__(self, values: list[int]) -> None:
        self.values = np.asarray(values)

    def to_numpy(self) -> np.ndarray:
        return self.values


class _FakeArrowData:
    def __init__(self, episode_ids: list[int]) -> None:
        self.episode_ids = episode_ids

    def column(self, name: str) -> _FakeArrowColumn:
        assert name == "episode_index"
        return _FakeArrowColumn(self.episode_ids)


class _FakeHFDataset:
    def __init__(self, episode_ids: list[int]) -> None:
        self.data = _FakeArrowData(episode_ids)


class _FakeDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.episode_ids = [2, 2, 5]
        self.hf_dataset = _FakeHFDataset(self.episode_ids)

    def __len__(self) -> int:
        return len(self.episode_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        action = torch.zeros((2, 7), dtype=torch.float32)
        action[:, 0] = 0.001 * (index + 1)
        is_pad = torch.tensor([False, index == 1])
        return {
            "observation.state": torch.full((8,), float(index)),
            "observation.images.l515": torch.zeros((3, 2, 2)),
            "action": action,
            "action_is_pad": is_pad,
            "episode_index": torch.tensor(self.episode_ids[index]),
        }


class _IntegerPaddingDataset(_FakeDataset):
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(index)
        item["action_is_pad"] = item["action_is_pad"].to(torch.int64)
        return item


class _ScalingPreprocessor:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = dict(batch)
        result["action"] = batch["action"] * 10.0
        return result


class _ScalingPostprocessor:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def __call__(self, action: torch.Tensor) -> torch.Tensor:
        return action / 10.0


class _FakeChunkPolicy:
    def __init__(self) -> None:
        self.eval_calls = 0
        self.reset_calls = 0
        self.inference_mode_flags: list[bool] = []

    def eval(self) -> _FakeChunkPolicy:
        self.eval_calls += 1
        return self

    def reset(self) -> None:
        self.reset_calls += 1

    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.inference_mode_flags.append(torch.is_inference_mode_enabled())
        batch_size = batch["observation.state"].shape[0]
        return torch.zeros((batch_size, 2, 7), dtype=torch.float32)


def test_evaluation_orchestration_resets_each_episode_and_uses_inference_chunk_api() -> None:
    dataset = _FakeDataset()
    policy = _FakeChunkPolicy()
    preprocessor = _ScalingPreprocessor()
    postprocessor = _ScalingPostprocessor()

    results = evaluate_policy_on_dataset(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        episode_ids=[2, 5],
        input_feature_keys=("observation.state", "observation.images.l515"),
        chunk_size=2,
        batch_size=2,
        num_workers=0,
    )

    assert [item.episode_id for item in results] == [2, 5]
    assert [item.num_frames for item in results] == [2, 1]
    assert [item.valid_pairs for item in results] == [3, 2]
    assert policy.eval_calls == 1
    assert policy.reset_calls == 2
    assert preprocessor.reset_calls == 2
    assert postprocessor.reset_calls == 2
    assert policy.inference_mode_flags and all(policy.inference_mode_flags)
    assert results[0].translation_error_mm > 0


def test_evaluation_orchestration_rejects_non_boolean_padding_before_casting() -> None:
    with pytest.raises(ValueError, match="action_is_pad must be boolean"):
        evaluate_policy_on_dataset(
            policy=_FakeChunkPolicy(),
            preprocessor=_ScalingPreprocessor(),
            postprocessor=_ScalingPostprocessor(),
            dataset=_IntegerPaddingDataset(),
            episode_ids=[2],
            input_feature_keys=("observation.state", "observation.images.l515"),
            chunk_size=2,
            batch_size=2,
            num_workers=0,
        )


def test_validate_runtime_paths_requires_pretrained_model_and_dataset_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    for name in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        (checkpoint / name).touch()
    dataset_root = tmp_path / "dataset"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text("{}")

    resolved_checkpoint, resolved_dataset = validate_runtime_paths(
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        output_dir=tmp_path / "output",
        device="cpu",
    )

    assert resolved_checkpoint == checkpoint.resolve()
    assert resolved_dataset == dataset_root.resolve()

    (checkpoint / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="model.safetensors"):
        validate_runtime_paths(checkpoint, dataset_root, tmp_path / "output", "cpu")


def test_terminal_summary_reports_episode_balanced_mean_and_std() -> None:
    gt = _chunks(frames=1, horizon=1, value=0.0)
    first = compute_episode_metrics(
        0, _chunks(1, 1, 0.001), gt, _chunks(1, 1, 0.001), gt, np.zeros((1, 1), dtype=bool)
    )
    second = compute_episode_metrics(
        1, _chunks(1, 1, 0.003), gt, _chunks(1, 1, 0.003), gt, np.zeros((1, 1), dtype=bool)
    )
    summary = summarize_episode_metrics([first, second])

    text = format_terminal_summary("/checkpoint", 2, summary)

    assert "Checkpoint: /checkpoint" in text
    assert "Episodes: 2" in text
    assert "Normalized L1:" in text
    assert "Translation error: 2.00 ± 1.41 mm" in text
    assert "Rotation error: 0.00 ± 0.00 deg" in text
    assert "ADE: 2.00 ± 1.41 mm" in text
    assert "FDE: 2.00 ± 1.41 mm" in text


def test_run_evaluation_connects_loaded_runtime_to_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    policy = _FakeChunkPolicy()
    runtime = EvaluationRuntime(
        checkpoint=tmp_path / "pretrained_model",
        dataset_root=tmp_path / "dataset",
        device="cpu",
        config=_valid_policy_config(horizon=2),
        policy=policy,
        preprocessor=_ScalingPreprocessor(),
        postprocessor=_ScalingPostprocessor(),
        dataset=_FakeDataset(),
        fps=30.0,
        action_names=EXPECTED_ACTION_NAMES,
    )
    args = SimpleNamespace(
        episodes=[2, 5],
        output_dir=tmp_path / "outputs",
        batch_size=2,
        num_workers=0,
    )

    result = run_evaluation(args, runtime=runtime)

    assert len(result.episode_metrics) == 2
    assert result.output_paths["summary"].is_file()
    assert "Episodes: 2" in capsys.readouterr().out


def test_runtime_loader_requires_strict_checkpoint_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    for name in (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        (checkpoint / name).touch()
    dataset_root = tmp_path / "dataset"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text("{}")

    config = _valid_policy_config(horizon=2)
    captured: dict[str, object] = {}

    class FakeLoadedPolicy:
        def to(self, device: str) -> FakeLoadedPolicy:
            return self

        def eval(self) -> FakeLoadedPolicy:
            return self

    class FakePolicyClass:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> FakeLoadedPolicy:
            captured.update(kwargs)
            return FakeLoadedPolicy()

    class FakeMetadata:
        def __init__(self, repo_id: str, root: Path) -> None:
            self.features = _valid_dataset_features()
            self.total_episodes = 1
            self.fps = 30

    class FakeLoadedDataset:
        def __init__(self, **kwargs: object) -> None:
            self.features = _valid_dataset_features()
            self.fps = 30

    import lerobot.datasets as lerobot_datasets
    from lerobot import policies as lerobot_policies
    from lerobot.configs import PreTrainedConfig

    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", lambda *args, **kwargs: config)
    monkeypatch.setattr(lerobot_policies, "get_policy_class", lambda policy_type: FakePolicyClass)
    monkeypatch.setattr(
        lerobot_policies,
        "make_pre_post_processors",
        lambda **kwargs: (_ScalingPreprocessor(), _ScalingPostprocessor()),
    )
    monkeypatch.setattr(lerobot_datasets, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(lerobot_datasets, "LeRobotDataset", FakeLoadedDataset)
    args = SimpleNamespace(
        checkpoint=checkpoint,
        dataset_root=dataset_root,
        output_dir=tmp_path / "outputs",
        device="cpu",
        dataset_repo_id=None,
        episodes=[0],
        video_backend=None,
    )

    load_evaluation_runtime(args)

    assert captured["strict"] is True
