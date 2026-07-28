from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from hardware_test.franka.handeye import solve_eye_to_hand, validate_eye_to_hand
from hardware_test.franka.handeye.handeye_utils import HAND_EYE_METHODS, HandEyeSampleStore

CONFIG_PATH = Path(__file__).parent / "handeye" / "config" / "l515_eye_to_hand.yaml"
WIDTH = 8
HEIGHT = 6
SERIAL = "L515-synthetic"
METHOD_NAMES = tuple(HAND_EYE_METHODS)
SOLVE_PATH = Path(__file__).parent / "handeye" / "solve_eye_to_hand.py"
VALIDATE_PATH = Path(__file__).parent / "handeye" / "validate_eye_to_hand.py"
COORDINATE_DIRECTION = "Maps points from the L515 color optical camera frame to the Franka base frame."


def _axis_angle_rotation(axis: object, degrees: float) -> np.ndarray:
    axis_vector = np.asarray(axis, dtype=np.float64)
    axis_vector /= np.linalg.norm(axis_vector)
    radians = np.deg2rad(degrees)
    cross_product = np.array(
        [
            [0.0, -axis_vector[2], axis_vector[1]],
            [axis_vector[2], 0.0, -axis_vector[0]],
            [-axis_vector[1], axis_vector[0], 0.0],
        ]
    )
    return (
        np.eye(3) * np.cos(radians)
        + (1.0 - np.cos(radians)) * np.outer(axis_vector, axis_vector)
        + np.sin(radians) * cross_product
    )


def _transform(axis: object, degrees: float, translation: object) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _axis_angle_rotation(axis, degrees)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def _synthetic_transforms() -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    base_camera = _transform([0.3, -0.5, 0.8], 23.0, [0.72, -0.31, 0.88])
    ee_board = _transform([-0.4, 0.7, 0.2], -17.0, [0.08, 0.01, 0.14])
    camera_base = np.linalg.inv(base_camera)
    axes = (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, -1.0, 0.5],
        [-0.5, 1.0, 1.0],
        [1.0, 0.4, -0.8],
        [-0.7, 0.2, 1.0],
        [0.3, -1.0, 0.6],
        [0.8, 0.7, 0.2],
    )
    angles = (-42.0, -31.0, -24.0, -13.0, -7.0, 9.0, 16.0, 22.0, 29.0, 37.0, 45.0, 53.0)
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for index, (axis, angle) in enumerate(zip(axes, angles, strict=True)):
        base_ee = _transform(
            axis,
            angle,
            [
                0.34 + 0.025 * (index % 4),
                -0.22 + 0.031 * ((index * 2) % 5),
                0.39 + 0.019 * ((index * 3) % 7),
            ],
        )
        pairs.append((base_ee, camera_base @ base_ee @ ee_board))
    return pairs, base_camera


def _sample_record(base_ee: np.ndarray, camera_board: np.ndarray) -> dict[str, object]:
    rvec, _ = cv2.Rodrigues(camera_board[:3, :3])
    return {
        "camera_timestamp_ms": 1234.5,
        "robot_timestamp": 8765,
        "image_width": WIDTH,
        "image_height": HEIGHT,
        "charuco_ids": [0, 1, 6, 7],
        "charuco_corners_px": [[0.25, 0.25], [1.25, 0.25], [0.25, 1.25], [1.25, 1.25]],
        "num_charuco_corners": 4,
        "rvec_camera_board": rvec.reshape(3).tolist(),
        "tvec_camera_board_m": camera_board[:3, 3].tolist(),
        "T_camera_board": camera_board.tolist(),
        "T_base_ee": base_ee.tolist(),
        "robot_pose_raw": base_ee.tolist(),
        "reprojection_error_px": 0.19,
        "opencv_version": "5.0.0-test",
        "realsense_serial": SERIAL,
        "robot_pose_name": "T_base_ee",
        "translation_unit": "meter",
        "matrix_storage_source": "existing_franka_client",
    }


def _write_intrinsics(input_dir: Path) -> None:
    intrinsics = {
        "width": WIDTH,
        "height": HEIGHT,
        "fps": 30,
        "fx": 604.5,
        "fy": 603.25,
        "ppx": 4.0,
        "ppy": 3.0,
        "distortion_model": "brown_conrady",
        "distortion_coefficients": [0.0] * 5,
        "serial": SERIAL,
        "camera_matrix": [[604.5, 0.0, 4.0], [0.0, 603.25, 3.0], [0.0, 0.0, 1.0]],
    }
    (input_dir / "camera_intrinsics.json").write_text(
        json.dumps(intrinsics, allow_nan=False),
        encoding="utf-8",
    )


def _make_store(input_dir: Path, *, sample_count: int = 12) -> np.ndarray:
    shutil.copyfile(CONFIG_PATH, input_dir / "config_used.yaml")
    _write_intrinsics(input_dir)
    pairs, base_camera = _synthetic_transforms()
    store = HandEyeSampleStore(input_dir)
    rgb: np.ndarray = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    overlay = np.full_like(rgb, 31)
    for base_ee, camera_board in pairs[:sample_count]:
        store.save(_sample_record(base_ee, camera_board), rgb, overlay)
    return base_camera


class _FakeCv2(SimpleNamespace):
    def __init__(self, base_camera: np.ndarray, *, failed_methods: set[int] | None = None) -> None:
        super().__init__(__version__="5.0.0-test")
        self.base_camera = base_camera
        self.failed_methods = set() if failed_methods is None else failed_methods
        self.calls: list[int] = []

    def calibrateHandEye(  # noqa: N802 - mirrors OpenCV's API.
        self,
        _rotations_gripper_to_base: object,
        _translations_gripper_to_base: object,
        _rotations_target_to_camera: object,
        _translations_target_to_camera: object,
        *,
        method: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(method)
        if method in self.failed_methods:
            raise RuntimeError(f"injected failure for OpenCV method {method}")
        return self.base_camera[:3, :3].copy(), self.base_camera[:3, 3:4].copy()


@pytest.mark.parametrize("script_path", [SOLVE_PATH, VALIDATE_PATH])
def test_direct_script_help_has_no_input_or_hardware_side_effects(tmp_path: Path, script_path: Path):
    untouched_input = tmp_path / "does-not-exist"
    completed = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert completed.returncode == 0, completed.stderr
    assert "--input-dir" in completed.stdout
    assert not untouched_input.exists()


@pytest.mark.parametrize(
    ("script_path", "error_prefix"),
    [(SOLVE_PATH, "Solver error:"), (VALIDATE_PATH, "Validation error:")],
)
def test_direct_script_missing_input_reports_one_line_error_without_traceback(
    tmp_path: Path,
    script_path: Path,
    error_prefix: str,
):
    missing_input = tmp_path / "missing-capture"
    completed = subprocess.run(
        [sys.executable, str(script_path), "--input-dir", str(missing_input)],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr.startswith(error_prefix)
    assert str(missing_input) in completed.stderr
    assert len(completed.stderr.splitlines()) == 1
    assert "Traceback" not in completed.stderr


@pytest.mark.parametrize(
    ("script_path", "error_prefix"),
    [(SOLVE_PATH, "Solver error:"), (VALIDATE_PATH, "Validation error:")],
)
def test_direct_script_flattens_multiline_invalid_config_error(
    tmp_path: Path,
    script_path: Path,
    error_prefix: str,
):
    invalid_input = tmp_path / "invalid-capture"
    invalid_input.mkdir()
    (invalid_input / "config_used.yaml").write_text("charuco: [\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script_path), "--input-dir", str(invalid_input)],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert completed.stderr.startswith(error_prefix)
    assert "config_used.yaml" in completed.stderr
    assert len(completed.stderr.splitlines()) == 1
    assert "Traceback" not in completed.stderr


def test_solve_and_validate_round_trip_all_methods_without_mutating_calibration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    base_camera = _make_store(tmp_path)
    fake_cv2 = _FakeCv2(base_camera)

    assert solve_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2) == 0

    calibration_path = tmp_path / "result" / "T_base_camera.yaml"
    report_path = tmp_path / "result" / "validation_report.json"
    calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    first_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert calibration["parent_frame"] == "franka_base"
    assert calibration["child_frame"] == "l515_color_optical_frame"
    assert calibration["transform_name"] == "T_base_camera"
    assert calibration["translation_unit"] == "meter"
    assert calibration["method"] in METHOD_NAMES
    np.testing.assert_allclose(calibration["matrix"], base_camera, atol=1e-12)
    assert tuple(calibration["methods"]) == METHOD_NAMES
    assert all(calibration["methods"][name]["status"] == "success" for name in METHOD_NAMES)
    assert calibration["coordinate_direction"] == COORDINATE_DIRECTION
    assert first_report["coordinate_direction"] == COORDINATE_DIRECTION
    assert first_report["sample_count"] == 12
    assert first_report["sample_ids"] == list(range(12))
    assert calibration["sample_ids"] == list(range(12))
    provenance = calibration["calibration_input_provenance"]
    assert provenance == first_report["calibration_input_provenance"]
    assert provenance["schema"] == "lerobot.handeye.calibration-inputs"
    assert provenance["schema_version"] == 1
    assert provenance["digest_algorithm"] == "sha256"
    assert provenance["canonicalization"] == "json-utf8-sort-keys-compact-finite-v1"
    assert provenance["sample_ids"] == list(range(12))
    assert len(provenance["digest"]) == 64
    assert set(provenance["digest"]) <= set("0123456789abcdef")
    persisted_samples = [
        json.loads(line) for line in (tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    canonical_inputs = [
        {
            "sample_id": sample["sample_id"],
            "T_base_ee": sample["T_base_ee"],
            "T_camera_board": sample["T_camera_board"],
        }
        for sample in persisted_samples
    ]
    canonical_json = json.dumps(
        canonical_inputs,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert provenance["digest"] == hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    assert tuple(first_report["methods"]) == METHOD_NAMES
    assert first_report["outlier_ids"] == []
    assert first_report["validation_outlier_ids"] == []
    assert first_report["leave_one_out_influential_ids"] == []
    assert (
        first_report["config_thresholds"]
        == yaml.safe_load((tmp_path / "config_used.yaml").read_text(encoding="utf-8"))["validation"]
    )
    assert set(first_report["recommendation_terms"]) == {
        "validation_translation",
        "validation_rotation",
        "leave_one_out_translation",
        "leave_one_out_rotation",
    }

    protected_paths = [
        tmp_path / "config_used.yaml",
        tmp_path / "camera_intrinsics.json",
        tmp_path / "samples.jsonl",
        calibration_path,
        *sorted((tmp_path / "images").iterdir()),
        *sorted((tmp_path / "overlays").iterdir()),
    ]
    protected_bytes = {path: path.read_bytes() for path in protected_paths}

    assert validate_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2) == 0

    assert {path: path.read_bytes() for path in protected_paths} == protected_bytes
    assert json.loads(report_path.read_text(encoding="utf-8")) == first_report
    validated_report_bytes = report_path.read_bytes()
    assert validate_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2) == 0
    assert report_path.read_bytes() == validated_report_bytes
    output = capsys.readouterr().out
    for method_name in METHOD_NAMES:
        assert method_name in output


def _rewrite_sample(input_dir: Path, sample_index: int, **updates: object) -> None:
    manifest_path = input_dir / "samples.jsonl"
    samples = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    samples[sample_index].update(updates)
    manifest_path.write_text(
        "".join(json.dumps(sample, allow_nan=False, separators=(",", ":")) + "\n" for sample in samples),
        encoding="utf-8",
    )


def test_solve_requires_four_samples_for_leave_one_out(tmp_path: Path):
    base_camera = _make_store(tmp_path, sample_count=3)

    with pytest.raises(ValueError, match=r"samples\.jsonl.*at least 4.*got 3"):
        solve_eye_to_hand.main(
            ["--input-dir", str(tmp_path)],
            cv2_module=_FakeCv2(base_camera),
        )


def test_solve_accepts_collector_wrapped_config_and_nested_intrinsics(tmp_path: Path):
    base_camera = _make_store(tmp_path)
    config_path = tmp_path / "config_used.yaml"
    resolved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_path.write_text(
        yaml.safe_dump(
            {
                "resolved_config": resolved_config,
                "cli_capture": {"width": WIDTH, "height": HEIGHT, "fps": 30},
                "opencv_version": "5.0.0-test",
                "legacy_pattern": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    intrinsics_path = tmp_path / "camera_intrinsics.json"
    flat = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    nested = {
        "resolution": {"width": flat["width"], "height": flat["height"]},
        "fps": flat["fps"],
        "camera_matrix": flat["camera_matrix"],
        "distortion": {
            "model": flat["distortion_model"],
            "coefficients": flat["distortion_coefficients"],
        },
        "realsense_serial": flat["serial"],
    }
    intrinsics_path.write_text(json.dumps(nested, allow_nan=False), encoding="utf-8")

    assert (
        solve_eye_to_hand.main(
            ["--input-dir", str(tmp_path)],
            cv2_module=_FakeCv2(base_camera),
        )
        == 0
    )
    report = json.loads((tmp_path / "result" / "validation_report.json").read_text(encoding="utf-8"))
    assert report["config_thresholds"] == resolved_config["validation"]


def test_solve_rejects_sample_dimensions_that_disagree_with_actual_intrinsics(tmp_path: Path):
    base_camera = _make_store(tmp_path)
    _rewrite_sample(tmp_path, 5, image_width=WIDTH + 1)

    with pytest.raises(ValueError, match=r"samples\.jsonl.*sample_id 5.*9x6.*intrinsics.*8x6"):
        solve_eye_to_hand.main(
            ["--input-dir", str(tmp_path)],
            cv2_module=_FakeCv2(base_camera),
        )


def test_solve_rejects_sample_serial_that_disagrees_with_actual_intrinsics(tmp_path: Path):
    base_camera = _make_store(tmp_path)
    _rewrite_sample(tmp_path, 7, realsense_serial="different-L515")

    with pytest.raises(
        ValueError,
        match=r"samples\.jsonl.*sample_id 7.*different-L515.*camera_intrinsics\.json.*L515-synthetic",
    ):
        solve_eye_to_hand.main(
            ["--input-dir", str(tmp_path)],
            cv2_module=_FakeCv2(base_camera),
        )


def test_solve_preflights_missing_calibrate_hand_eye_capability_actionably(tmp_path: Path):
    _make_store(tmp_path)
    cv2_without_hand_eye = SimpleNamespace(__version__="5.0.0", calibrateHandEye=None)

    with pytest.raises(
        RuntimeError,
        match=r"cv2\.calibrateHandEye.*5\.0\.0.*OpenCV package conflict.*never install or upgrade",
    ):
        solve_eye_to_hand.main(
            ["--input-dir", str(tmp_path)],
            cv2_module=cv2_without_hand_eye,
        )


def test_failed_solve_method_is_retained_by_solve_and_validate(tmp_path: Path):
    base_camera = _make_store(tmp_path)
    fake_cv2 = _FakeCv2(base_camera, failed_methods={HAND_EYE_METHODS["PARK"]})

    assert solve_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2) == 0
    calibration = yaml.safe_load((tmp_path / "result" / "T_base_camera.yaml").read_text(encoding="utf-8"))
    solve_report = json.loads((tmp_path / "result" / "validation_report.json").read_text(encoding="utf-8"))
    assert calibration["methods"]["PARK"]["status"] == "error"
    assert "injected failure" in calibration["methods"]["PARK"]["error"]
    assert solve_report["methods"]["PARK"]["status"] == "error"

    assert validate_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2) == 0
    validate_report = json.loads((tmp_path / "result" / "validation_report.json").read_text(encoding="utf-8"))
    assert validate_report["methods"]["PARK"] == solve_report["methods"]["PARK"]


def test_validate_reports_malformed_saved_matrices_with_method_and_path_context(tmp_path: Path):
    base_camera = _make_store(tmp_path)
    fake_cv2 = _FakeCv2(base_camera)
    solve_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2)
    calibration_path = tmp_path / "result" / "T_base_camera.yaml"
    calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    for method_name in METHOD_NAMES:
        calibration["methods"][method_name]["matrix"] = [[1.0, 0.0], [0.0, 1.0]]
    calibration_path.write_text(yaml.safe_dump(calibration, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match=r"T_base_camera\.yaml.*no valid method matrices.*TSAI.*shape \(4, 4\)",
    ):
        validate_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2)


def test_validate_rejects_malformed_top_level_recommended_matrix(tmp_path: Path):
    base_camera = _make_store(tmp_path)
    fake_cv2 = _FakeCv2(base_camera)
    solve_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2)
    calibration_path = tmp_path / "result" / "T_base_camera.yaml"
    calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    calibration["matrix"] = [[1.0, 0.0], [0.0, 1.0]]
    calibration_path.write_text(yaml.safe_dump(calibration, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=r"T_base_camera\.yaml\.matrix.*shape \(4, 4\)"):
        validate_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2)


def test_validate_rejects_changed_transform_with_same_sample_ids_and_count(tmp_path: Path):
    base_camera = _make_store(tmp_path)
    fake_cv2 = _FakeCv2(base_camera)
    solve_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2)
    samples = [
        json.loads(line) for line in (tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    changed_camera_board = np.asarray(samples[5]["T_camera_board"]) @ _transform(
        [0.2, -0.7, 0.4],
        1.0,
        [0.001, -0.002, 0.0005],
    )
    _rewrite_sample(tmp_path, 5, T_camera_board=changed_camera_board.tolist())

    with pytest.raises(
        ValueError,
        match=r"T_base_camera\.yaml.*provenance.*samples changed.*rerun solve",
    ):
        validate_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2)


def test_validate_checks_changed_sample_provenance_before_saved_matrices(tmp_path: Path):
    base_camera = _make_store(tmp_path)
    fake_cv2 = _FakeCv2(base_camera)
    solve_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2)
    samples = [
        json.loads(line) for line in (tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    changed_camera_board = np.asarray(samples[3]["T_camera_board"]) @ _transform(
        [0.6, 0.1, -0.3],
        0.5,
        [0.0005, 0.0, -0.0005],
    )
    _rewrite_sample(tmp_path, 3, T_camera_board=changed_camera_board.tolist())
    calibration_path = tmp_path / "result" / "T_base_camera.yaml"
    calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    calibration["matrix"] = [[1.0, 0.0], [0.0, 1.0]]
    calibration_path.write_text(yaml.safe_dump(calibration, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"T_base_camera\.yaml.*provenance.*samples changed.*rerun solve",
    ):
        validate_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2)


def test_validate_reports_missing_calibration_provenance_actionably(tmp_path: Path):
    base_camera = _make_store(tmp_path)
    fake_cv2 = _FakeCv2(base_camera)
    solve_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2)
    calibration_path = tmp_path / "result" / "T_base_camera.yaml"
    calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    calibration.pop("calibration_input_provenance")
    calibration_path.write_text(yaml.safe_dump(calibration, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"T_base_camera\.yaml.*missing calibration input provenance.*samples.*rerun solve",
    ):
        validate_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2)


def test_validate_reports_malformed_provenance_digest_actionably(tmp_path: Path):
    base_camera = _make_store(tmp_path)
    fake_cv2 = _FakeCv2(base_camera)
    solve_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2)
    calibration_path = tmp_path / "result" / "T_base_camera.yaml"
    calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    calibration["calibration_input_provenance"]["digest"] = "not-a-sha256"
    calibration_path.write_text(yaml.safe_dump(calibration, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"T_base_camera\.yaml.*provenance.*digest.*64 lowercase hexadecimal.*rerun solve",
    ):
        validate_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2)


def test_validate_rejects_delete_add_with_same_count_but_different_sample_id(tmp_path: Path):
    base_camera = _make_store(tmp_path)
    fake_cv2 = _FakeCv2(base_camera)
    solve_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2)
    store = HandEyeSampleStore(tmp_path)
    deleted = store.delete_last()
    assert deleted is not None and deleted["sample_id"] == 11
    shutil.copyfile(
        tmp_path / "images" / "sample_000.png",
        tmp_path / "images" / "sample_099.png",
    )
    pairs, _ = _synthetic_transforms()
    rgb: np.ndarray = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    replacement = store.save(_sample_record(*pairs[-1]), rgb, np.full_like(rgb, 31))
    assert replacement["sample_id"] == 100
    assert len(store.samples) == 12

    with pytest.raises(
        ValueError,
        match=r"T_base_camera\.yaml.*provenance.*sample IDs.*changed.*rerun solve",
    ):
        validate_eye_to_hand.main(["--input-dir", str(tmp_path)], cv2_module=fake_cv2)
