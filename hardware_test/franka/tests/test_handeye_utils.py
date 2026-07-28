from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

import hardware_test.franka.handeye.handeye_utils as handeye_utils
from hardware_test.franka.handeye.handeye_utils import (
    HAND_EYE_METHODS,
    BoardPoseEstimate,
    CharucoDetection,
    CharucoDetectorCompat,
    FrankaPoseReader,
    HandEyeSampleStore,
    RobotStillnessMonitor,
    assess_pose_diversity,
    calibrate_eye_to_hand,
    create_charuco_board,
    draw_detection_overlay,
    estimate_board_pose,
    invert_transform,
    laplacian_blur_score,
    leave_one_out_validation,
    load_handeye_config,
    make_transform,
    match_charuco_image_points,
    max_charuco_corners,
    mean_rigid_transform,
    opencv_distortion_coefficients,
    parse_franka_state_pose,
    pose_delta,
    rotation_delta_deg,
    solve_all_methods,
    validate_eye_to_hand_result,
    validate_homogeneous_transform,
)

CONFIG_PATH = Path(__file__).parent / "handeye" / "config" / "l515_eye_to_hand.yaml"
EXPECTED_CHARUCO_CONFIG = {
    "dictionary": "DICT_5X5_100",
    "squares_x": 7,
    "squares_y": 5,
    "square_length_m": 0.035,
    "marker_length_m": 0.026,
    "legacy_pattern": False,
}


@pytest.fixture
def shipped_config() -> dict:
    return load_handeye_config(CONFIG_PATH)


def _write_config(tmp_path: Path, config: object) -> Path:
    path = tmp_path / "handeye.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _rotation_z(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_shipped_config_has_exact_charuco_board(shipped_config: dict):
    assert shipped_config["charuco"] == EXPECTED_CHARUCO_CONFIG
    assert max_charuco_corners(shipped_config["charuco"]) == 24


def test_shipped_config_has_capture_validation_defaults(shipped_config: dict):
    assert shipped_config["capture_validation"] == {
        "min_charuco_corners": 12,
        "warning_reprojection_error_px": 1.0,
        "max_reprojection_error_px": 2.0,
        "min_laplacian_variance": 100.0,
    }


def test_shipped_config_has_motion_and_validation_defaults(shipped_config: dict):
    assert shipped_config["robot_stillness"] == {
        "window_s": 1.0,
        "max_translation_m": 0.001,
        "max_rotation_deg": 0.5,
    }
    assert shipped_config["pose_similarity"] == {
        "translation_m": 0.010,
        "rotation_deg": 5.0,
    }
    assert shipped_config["validation"] == {
        "target_scatter_translation_m": 0.005,
        "target_scatter_rotation_deg": 2.0,
        "leave_one_out_translation_m": 0.010,
        "leave_one_out_rotation_deg": 3.0,
        "robust_mad_multiplier": 3.5,
    }


def test_load_config_rejects_wrong_charuco_dictionary(tmp_path: Path, shipped_config: dict):
    config = deepcopy(shipped_config)
    config["charuco"]["dictionary"] = "DICT_4X4_50"

    with pytest.raises(ValueError, match="DICT_5X5_100"):
        load_handeye_config(_write_config(tmp_path, config))


@pytest.mark.parametrize(
    ("key", "invalid_value"),
    [
        ("square_length_m", 0.0),
        ("square_length_m", -0.035),
        ("marker_length_m", 0.0),
        ("marker_length_m", -0.026),
        ("marker_length_m", "0.026"),
    ],
)
def test_load_config_rejects_invalid_board_lengths(
    tmp_path: Path, shipped_config: dict, key: str, invalid_value: object
):
    config = deepcopy(shipped_config)
    config["charuco"][key] = invalid_value

    with pytest.raises(ValueError, match=key):
        load_handeye_config(_write_config(tmp_path, config))


@pytest.mark.parametrize("marker_length_m", [0.035, 0.040])
def test_load_config_requires_marker_shorter_than_square(
    tmp_path: Path, shipped_config: dict, marker_length_m: float
):
    config = deepcopy(shipped_config)
    config["charuco"]["marker_length_m"] = marker_length_m

    with pytest.raises(ValueError, match="marker_length_m"):
        load_handeye_config(_write_config(tmp_path, config))


@pytest.mark.parametrize(
    ("key", "invalid_value"),
    [("squares_x", 1), ("squares_y", 1), ("squares_x", 7.0), ("squares_y", True)],
)
def test_load_config_rejects_invalid_board_dimensions(
    tmp_path: Path, shipped_config: dict, key: str, invalid_value: object
):
    config = deepcopy(shipped_config)
    config["charuco"][key] = invalid_value

    with pytest.raises(ValueError, match=key):
        load_handeye_config(_write_config(tmp_path, config))


def test_load_config_rejects_corner_minimum_above_board_capacity(tmp_path: Path, shipped_config: dict):
    config = deepcopy(shipped_config)
    config["capture_validation"]["min_charuco_corners"] = 25

    with pytest.raises(ValueError, match="min_charuco_corners"):
        load_handeye_config(_write_config(tmp_path, config))


@pytest.mark.parametrize("document", [None, [], "not a mapping"])
def test_load_config_rejects_non_mapping_yaml(tmp_path: Path, document: object):
    with pytest.raises(ValueError, match="mapping"):
        load_handeye_config(_write_config(tmp_path, document))


def test_validate_homogeneous_transform_returns_float64_copy():
    source = np.eye(4, dtype=np.float32)

    validated = validate_homogeneous_transform(source, name="T_base_ee")

    assert validated.dtype == np.float64
    assert not np.shares_memory(validated, source)
    validated[0, 3] = 1.0
    assert source[0, 3] == 0.0


def _invalid_transform(case: str) -> np.ndarray:
    transform = np.eye(4)
    if case == "shape":
        return np.eye(3)
    if case == "nan":
        transform[0, 3] = np.nan
    elif case == "bad_rotation":
        transform[0, 0] = 2.0
    elif case == "bad_last_row":
        transform[3, 0] = 0.1
    elif case == "reflection":
        transform[0, 0] = -1.0
    else:  # pragma: no cover - protects this test helper from silent misuse
        raise AssertionError(f"Unknown case: {case}")
    return transform


@pytest.mark.parametrize("case", ["shape", "nan", "bad_rotation", "bad_last_row", "reflection"])
def test_validate_homogeneous_transform_rejects_invalid_matrices(case: str):
    with pytest.raises(ValueError, match="T_bad"):
        validate_homogeneous_transform(_invalid_transform(case), name="T_bad")


def test_make_and_invert_transform_use_rigid_transform_math():
    transform = make_transform(_rotation_z(37.0), [0.42, -0.17, 0.63], name="T_base_camera")

    inverse = invert_transform(transform, name="T_base_camera")

    np.testing.assert_allclose(transform @ inverse, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(inverse @ transform, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(inverse[:3, :3], transform[:3, :3].T, atol=1e-12)
    np.testing.assert_allclose(inverse[:3, 3], -transform[:3, :3].T @ transform[:3, 3], atol=1e-12)


def test_make_transform_validates_rotation_and_translation_shape():
    with pytest.raises(ValueError, match="T_invalid"):
        make_transform(np.diag([-1.0, 1.0, 1.0]), [0.0, 0.0, 0.0], name="T_invalid")

    with pytest.raises(ValueError, match="translation"):
        make_transform(np.eye(3), [[0.0], [0.0], [0.0]], name="T_invalid")


def test_rotation_delta_deg_returns_known_geodesic_angle():
    assert rotation_delta_deg(np.eye(3), _rotation_z(90.0)) == pytest.approx(90.0)
    assert rotation_delta_deg(_rotation_z(-45.0), _rotation_z(45.0)) == pytest.approx(90.0)


def test_pose_delta_returns_translation_metres_and_rotation_degrees():
    pose_a = make_transform(np.eye(3), [0.0, 0.0, 0.0], name="T_base_ee_a")
    pose_b = make_transform(_rotation_z(90.0), [0.3, -0.4, 0.0], name="T_base_ee_b")

    translation_m, rotation_deg = pose_delta(pose_a, pose_b)

    assert translation_m == pytest.approx(0.5)
    assert rotation_deg == pytest.approx(90.0)


def test_parse_franka_state_pose_preserves_nested_existing_client_contract():
    T_base_ee = make_transform(  # noqa: N806 - coordinate-frame notation aids the test.
        _rotation_z(27.0),
        [0.41, -0.18, 0.63],
        name="T_base_ee",
    )
    robot_pose_raw = T_base_ee.tolist()

    reading = parse_franka_state_pose(
        {"is_ok": 1, "ee": robot_pose_raw},
        local_monotonic_s=42.5,
        request_latency_ms=7.25,
    )

    np.testing.assert_allclose(reading.T_base_ee, T_base_ee)
    assert reading.robot_pose_raw == robot_pose_raw
    assert reading.robot_timestamp is None
    assert reading.local_monotonic_s == pytest.approx(42.5)
    assert reading.request_latency_ms == pytest.approx(7.25)
    assert reading.robot_pose_name == "T_base_ee"
    assert reading.translation_unit == "meter"
    assert reading.matrix_storage_source == "existing_franka_client"
    assert reading.matrix_storage_format == "nested_4x4"


def test_parse_franka_state_pose_reconstructs_flat_libfranka_column_major_pose():
    T_base_ee = make_transform(  # noqa: N806 - coordinate-frame notation aids the test.
        _rotation_z(-38.0),
        [-0.12, 0.34, 0.78],
        name="T_base_ee",
    )
    robot_pose_raw = T_base_ee.reshape(-1, order="F").tolist()

    reading = parse_franka_state_pose({"O_T_EE": robot_pose_raw})

    np.testing.assert_allclose(reading.T_base_ee, T_base_ee)
    assert reading.robot_pose_raw == robot_pose_raw
    assert reading.matrix_storage_source == "existing_franka_client"
    assert reading.matrix_storage_format == "flat_16_column_major"


@pytest.mark.parametrize(
    "state",
    [
        {"is_ok": 1},
        {"ee": [[1.0, 0.0], [0.0, 1.0]]},
        {"ee": [["not", "numeric"]] * 4},
        {"O_T_EE": [1.0] * 15},
        {"O_T_EE": [1.0] * 16 + [[2.0]]},
    ],
)
def test_parse_franka_state_pose_rejects_missing_or_malformed_values(state: dict):
    with pytest.raises(ValueError, match="ee|O_T_EE|pose"):
        parse_franka_state_pose(state)


@pytest.mark.parametrize("case", ["bad_rotation", "bad_last_row"])
def test_parse_franka_state_pose_rejects_non_rigid_nested_pose(case: str):
    with pytest.raises(ValueError, match="T_base_ee"):
        parse_franka_state_pose({"ee": _invalid_transform(case).tolist()})


@pytest.mark.parametrize(
    ("extra_state", "expected_timestamp"),
    [
        ({"timestamp": 123.0, "time": "server-time"}, None),
        ({"robot_timestamp": 123.25}, 123.25),
        ({"robot_timestamp": "robot-clock-42"}, "robot-clock-42"),
        ({"robot_timestamp": {"seconds": 123}}, None),
    ],
)
def test_parse_franka_state_pose_preserves_only_explicit_json_scalar_robot_timestamp(
    extra_state: dict, expected_timestamp: float | str | None
):
    state = {"ee": np.eye(4).tolist(), **extra_state}

    reading = parse_franka_state_pose(state)

    assert reading.robot_timestamp == expected_timestamp


class _ReadOnlyFakeFrankaClient:
    def __init__(self, state: dict):
        self.state = state
        self.calls: list[tuple[str, float | None]] = []

    def get_curr(self, timeout: float | None = None) -> dict:
        self.calls.append(("get_curr", timeout))
        return self.state

    def close(self) -> None:
        self.calls.append(("close", None))

    def recover(self):
        raise AssertionError("FrankaPoseReader must never recover the robot")

    def cartesian_velocity_control(self, _data):
        raise AssertionError("FrankaPoseReader must never send Cartesian commands")

    def stop_cartesian_velocity_control(self):
        raise AssertionError("FrankaPoseReader must never stop Cartesian control")

    def joint_position_control(self, _joints):
        raise AssertionError("FrankaPoseReader must never send joint commands")

    def stop_joint_position_control(self):
        raise AssertionError("FrankaPoseReader must never stop joint control")

    def gripper_open(self):
        raise AssertionError("FrankaPoseReader must never control the gripper")

    def gripper_close(self):
        raise AssertionError("FrankaPoseReader must never control the gripper")

    def move_home(self):
        raise AssertionError("FrankaPoseReader must never home the robot")


def test_franka_pose_reader_only_reads_and_close_only_closes_bare_client():
    client = _ReadOnlyFakeFrankaClient({"is_ok": 1, "ee": np.eye(4).tolist()})
    clock_values = iter([10.0, 10.04])
    reader = FrankaPoseReader(client, timeout_s=0.35, clock=lambda: next(clock_values))

    reading = reader.read()

    assert client.calls == [("get_curr", 0.35)]
    assert reading.local_monotonic_s == pytest.approx(10.02)
    assert reading.request_latency_ms == pytest.approx(40.0)

    reader.close()

    assert client.calls == [("get_curr", 0.35), ("close", None)]


def _pose_at(*, x_m: float = 0.0, rotation_deg: float = 0.0) -> np.ndarray:
    return make_transform(_rotation_z(rotation_deg), [x_m, 0.0, 0.0], name="T_base_ee")


def _monitor_with_thresholds() -> RobotStillnessMonitor:
    return RobotStillnessMonitor(window_s=1.0, max_translation_m=0.001, max_rotation_deg=0.5)


def test_robot_stillness_monitor_accepts_stationary_full_window():
    monitor = _monitor_with_thresholds()
    monitor.add(5.0, _pose_at())
    monitor.add(5.5, _pose_at(x_m=0.0004, rotation_deg=0.2))
    monitor.add(6.0, _pose_at(x_m=0.0002, rotation_deg=0.1))

    status = monitor.status()

    assert status.is_still is True
    assert status.history_span_s == pytest.approx(1.0)
    assert status.max_translation_m == pytest.approx(0.0004)
    assert status.max_rotation_deg == pytest.approx(0.2)
    assert status.reason == "still"


def test_robot_stillness_monitor_rejects_history_shorter_than_window():
    monitor = _monitor_with_thresholds()
    monitor.add(5.0, _pose_at())
    monitor.add(5.99, _pose_at())

    status = monitor.status()

    assert status.is_still is False
    assert status.history_span_s == pytest.approx(0.99)
    assert status.reason == "insufficient_history"


def test_robot_stillness_monitor_retains_sample_bracketing_non_aligned_window():
    monitor = _monitor_with_thresholds()
    sample_period_s = 0.101
    for index in range(100):
        monitor.add(5.0 + index * sample_period_s, _pose_at())

    status = monitor.status()

    assert status.reason == "still"
    assert status.is_still is True
    assert 1.0 <= status.history_span_s < 1.0 + sample_period_s


def test_robot_stillness_monitor_checks_translation_across_all_window_poses():
    monitor = _monitor_with_thresholds()
    monitor.add(5.0, _pose_at())
    monitor.add(5.4, _pose_at(x_m=0.002))
    monitor.add(5.8, _pose_at())
    monitor.add(6.0, _pose_at())

    status = monitor.status()

    assert status.is_still is False
    assert status.max_translation_m == pytest.approx(0.002)
    assert status.reason == "translation_motion_exceeded"


def test_robot_stillness_monitor_checks_rotation_across_all_window_poses():
    monitor = _monitor_with_thresholds()
    monitor.add(5.0, _pose_at())
    monitor.add(5.4, _pose_at(rotation_deg=1.0))
    monitor.add(5.8, _pose_at())
    monitor.add(6.0, _pose_at())

    status = monitor.status()

    assert status.is_still is False
    assert status.max_rotation_deg == pytest.approx(1.0)
    assert status.reason == "rotation_motion_exceeded"


def test_create_charuco_board_builds_exact_real_opencv_board(shipped_config: dict):
    board = create_charuco_board(shipped_config["charuco"])

    assert board.getChessboardSize() == (7, 5)
    assert board.getSquareLength() == pytest.approx(0.035)
    assert board.getMarkerLength() == pytest.approx(0.026)
    assert board.getChessboardCorners().shape == (24, 3)
    assert board.getDictionary().markerSize == 5
    assert board.getDictionary().bytesList.shape[0] == 100
    if callable(getattr(board, "getLegacyPattern", None)):
        assert board.getLegacyPattern() is False


class _RecordingCharucoBoard:
    def __init__(self, constructor_args: tuple[object, ...]) -> None:
        self.constructor_args = constructor_args
        self.legacy_pattern: bool | None = None

    def setLegacyPattern(self, legacy_pattern: bool) -> None:  # noqa: N802 - mirrors OpenCV.
        self.legacy_pattern = legacy_pattern


class _ModernBoardAruco:
    DICT_5X5_100 = 5100

    def __init__(self) -> None:
        self.dictionary_id: int | None = None

    def getPredefinedDictionary(self, dictionary_id: int) -> object:  # noqa: N802 - mirrors OpenCV.
        self.dictionary_id = dictionary_id
        return "modern-dictionary"

    def CharucoBoard(self, *args: object) -> _RecordingCharucoBoard:  # noqa: N802 - mirrors OpenCV.
        return _RecordingCharucoBoard(args)

    def Dictionary_get(self, _dictionary_id: int) -> object:  # noqa: N802 - mirrors OpenCV.
        raise AssertionError("The modern dictionary factory must be preferred")

    def CharucoBoard_create(self, *_args: object) -> object:  # noqa: N802 - mirrors OpenCV.
        raise AssertionError("The modern board constructor must be preferred")


def test_create_charuco_board_prefers_modern_factories_and_sets_legacy_pattern():
    aruco = _ModernBoardAruco()
    config = {**EXPECTED_CHARUCO_CONFIG, "legacy_pattern": True}

    board = create_charuco_board(config, cv2_module=SimpleNamespace(aruco=aruco))

    assert aruco.dictionary_id == aruco.DICT_5X5_100
    assert board.constructor_args == ((7, 5), 0.035, 0.026, "modern-dictionary")
    assert board.legacy_pattern is True


class _LegacyBoardAruco:
    DICT_5X5_100 = 5100

    def __init__(self) -> None:
        self.dictionary_id: int | None = None

    def Dictionary_get(self, dictionary_id: int) -> object:  # noqa: N802 - mirrors OpenCV.
        self.dictionary_id = dictionary_id
        return "legacy-dictionary"

    def CharucoBoard_create(self, *args: object) -> _RecordingCharucoBoard:  # noqa: N802 - mirrors OpenCV.
        return _RecordingCharucoBoard(args)


def test_create_charuco_board_uses_legacy_factories_when_modern_api_is_absent():
    aruco = _LegacyBoardAruco()

    board = create_charuco_board(EXPECTED_CHARUCO_CONFIG, cv2_module=SimpleNamespace(aruco=aruco))

    assert aruco.dictionary_id == aruco.DICT_5X5_100
    assert board.constructor_args == (7, 5, 0.035, 0.026, "legacy-dictionary")
    assert board.legacy_pattern is False


class _DetectorBoard:
    def getDictionary(self) -> str:  # noqa: N802 - mirrors OpenCV.
        return "board-dictionary"


class _LegacyDetectorBoard:
    dictionary = "board-dictionary"


class _ModernCharucoDetector:
    def __init__(self, owner: _ModernDetectorAruco) -> None:
        self._owner = owner

    def detectBoard(self, gray_image: np.ndarray) -> tuple[object, ...]:  # noqa: N802 - mirrors OpenCV.
        self._owner.gray_image = gray_image.copy()
        marker_corners = [np.arange(8, dtype=np.float32).reshape(1, 4, 2)]
        marker_ids = np.array([[9]], dtype=np.int32)
        charuco_corners = np.arange(8, dtype=np.float32).reshape(4, 1, 2)
        charuco_ids = np.array([[0], [1], [6], [7]], dtype=np.int32)
        return charuco_corners, charuco_ids, marker_corners, marker_ids


class _ModernDetectorAruco:
    def __init__(self) -> None:
        self.board: object | None = None
        self.gray_image: np.ndarray | None = None

    def CharucoDetector(self, board: object) -> _ModernCharucoDetector:  # noqa: N802 - mirrors OpenCV.
        self.board = board
        return _ModernCharucoDetector(self)

    def detectMarkers(self, *_args: object) -> object:  # noqa: N802 - mirrors OpenCV.
        raise AssertionError("Legacy marker detection must not run when CharucoDetector exists")

    def interpolateCornersCharuco(self, *_args: object) -> object:  # noqa: N802 - mirrors OpenCV.
        raise AssertionError("Legacy interpolation must not run when CharucoDetector exists")


class _DetectorCv2:
    COLOR_RGB2GRAY = 91

    def __init__(self, aruco: object) -> None:
        self.aruco = aruco
        self.conversion_code: int | None = None

    def cvtColor(self, rgb_image: np.ndarray, conversion_code: int) -> np.ndarray:  # noqa: N802
        self.conversion_code = conversion_code
        return rgb_image[..., 0].copy()


def test_charuco_detector_compat_prefers_modern_detector_and_normalizes_shapes():
    aruco = _ModernDetectorAruco()
    fake_cv2 = _DetectorCv2(aruco)
    board = _DetectorBoard()
    rgb_image = np.zeros((8, 10, 3), dtype=np.uint8)

    detector = CharucoDetectorCompat(board, cv2_module=fake_cv2)
    detection = detector.detect(rgb_image)

    assert detector.api_name == "CharucoDetector.detectBoard"
    assert detection.api_name == detector.api_name
    assert detection.num_charuco_corners == 4
    assert detection.charuco_corners.shape == (4, 2)
    assert detection.charuco_ids.shape == (4,)
    assert len(detection.marker_corners) == 1
    assert detection.marker_corners[0].shape == (4, 2)
    assert detection.marker_ids.shape == (1,)
    assert fake_cv2.conversion_code == fake_cv2.COLOR_RGB2GRAY
    assert aruco.board is board


class _NoDetectionCharucoDetector:
    def detectBoard(self, _gray_image: np.ndarray) -> tuple[None, None, tuple[()], None]:  # noqa: N802
        return None, None, (), None


class _NoDetectionAruco:
    def CharucoDetector(self, _board: object) -> _NoDetectionCharucoDetector:  # noqa: N802
        return _NoDetectionCharucoDetector()


def test_charuco_detector_compat_normalizes_modern_no_detection_result():
    detector = CharucoDetectorCompat(_DetectorBoard(), cv2_module=_DetectorCv2(_NoDetectionAruco()))

    detection = detector.detect(np.zeros((4, 5, 3), dtype=np.uint8))

    assert detection.num_charuco_corners == 0
    assert detection.charuco_corners.shape == (0, 2)
    assert detection.charuco_ids.shape == (0,)
    assert detection.marker_corners == ()
    assert detection.marker_ids.shape == (0,)


class _LegacyDetectorAruco:
    def __init__(self) -> None:
        self.detect_dictionary: object | None = None
        self.interpolate_arguments: tuple[object, ...] | None = None

    def detectMarkers(self, gray_image: np.ndarray, dictionary: object) -> tuple[object, ...]:  # noqa: N802
        self.detect_dictionary = dictionary
        marker_corners = (np.arange(8, dtype=np.float32).reshape(1, 4, 2),)
        marker_ids = np.array([[3]], dtype=np.int32)
        return marker_corners, marker_ids, ["rejected"]

    def interpolateCornersCharuco(  # noqa: N802 - mirrors OpenCV.
        self,
        marker_corners: object,
        marker_ids: object,
        gray_image: np.ndarray,
        board: object,
        **calibration: np.ndarray | None,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        self.interpolate_arguments = (
            marker_corners,
            marker_ids,
            gray_image,
            board,
            calibration["cameraMatrix"],
            calibration["distCoeffs"],
        )
        corners = np.arange(8, dtype=np.float32).reshape(4, 1, 2)
        ids = np.array([[0], [1], [6], [7]], dtype=np.int32)
        return 4, corners, ids


def test_charuco_detector_compat_uses_complete_legacy_fallback():
    aruco = _LegacyDetectorAruco()
    fake_cv2 = _DetectorCv2(aruco)
    board = _LegacyDetectorBoard()
    camera_matrix = np.eye(3)
    distortion_coefficients = np.zeros(5)

    detector = CharucoDetectorCompat(
        board,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        cv2_module=fake_cv2,
    )
    detection = detector.detect(np.zeros((8, 10, 3), dtype=np.uint8))

    assert detector.api_name == "detectMarkers+interpolateCornersCharuco"
    assert detection.num_charuco_corners == 4
    assert detection.charuco_corners.shape == (4, 2)
    assert detection.charuco_ids.shape == (4,)
    assert aruco.detect_dictionary == "board-dictionary"
    assert aruco.interpolate_arguments is not None
    assert aruco.interpolate_arguments[3] is board
    np.testing.assert_array_equal(aruco.interpolate_arguments[4], camera_matrix)
    np.testing.assert_array_equal(aruco.interpolate_arguments[5], distortion_coefficients)


@pytest.mark.parametrize("missing_name", ["detectMarkers", "interpolateCornersCharuco"])
def test_charuco_detector_compat_requires_both_legacy_functions(missing_name: str):
    aruco = _LegacyDetectorAruco()
    setattr(aruco, missing_name, None)

    with pytest.raises(RuntimeError, match="detectMarkers.*interpolateCornersCharuco"):
        CharucoDetectorCompat(_DetectorBoard(), cv2_module=_DetectorCv2(aruco))


def test_charuco_detector_compat_detects_a_real_modern_board(shipped_config: dict):
    board = create_charuco_board(shipped_config["charuco"])
    gray_image = board.generateImage((700, 500), marginSize=20, borderBits=1)
    rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

    detector = CharucoDetectorCompat(board)
    detection = detector.detect(rgb_image)

    assert detector.api_name == "CharucoDetector.detectBoard"
    assert detection.num_charuco_corners == 24
    assert detection.charuco_corners.shape == (24, 2)
    assert detection.charuco_ids.shape == (24,)
    assert len(detection.marker_corners) == 17
    assert detection.marker_ids.shape == (17,)


class _MatchImagePointsBoard:
    def __init__(self) -> None:
        self.match_calls = 0

    def getChessboardSize(self) -> tuple[int, int]:  # noqa: N802 - mirrors OpenCV.
        return 7, 5

    def checkCharucoCornersCollinear(self, _ids: np.ndarray) -> bool:  # noqa: N802
        return False

    def matchImagePoints(self, corners: np.ndarray, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa: N802
        self.match_calls += 1
        object_points = np.column_stack((ids.reshape(-1), np.zeros((len(ids), 2)))).reshape(-1, 1, 3)
        return object_points.astype(np.float32), corners.reshape(-1, 1, 2).astype(np.float32)

    def getChessboardCorners(self) -> np.ndarray:  # noqa: N802 - mirrors OpenCV.
        raise AssertionError("Indexed fallback must not run when matchImagePoints is available")


def test_match_charuco_image_points_prefers_match_method_and_returns_float64():
    board = _MatchImagePointsBoard()
    corners = np.arange(8, dtype=np.float32).reshape(4, 1, 2)
    ids = np.array([[0], [1], [6], [7]], dtype=np.int32)

    object_points, image_points = match_charuco_image_points(board, corners, ids)

    assert board.match_calls == 1
    assert object_points.shape == (4, 3)
    assert image_points.shape == (4, 2)
    assert object_points.dtype == np.float64
    assert image_points.dtype == np.float64
    np.testing.assert_array_equal(object_points[:, 0], ids.reshape(-1))
    np.testing.assert_array_equal(image_points, corners.reshape(-1, 2))


class _IndexedChessboardBoard:
    matchImagePoints = None  # noqa: N815 - mirrors an unavailable OpenCV method.

    def __init__(self, *, collinear: bool = False) -> None:
        self.chessboard_corners = np.arange(72, dtype=np.float32).reshape(24, 3)
        self.collinear = collinear

    def getChessboardCorners(self) -> np.ndarray:  # noqa: N802 - mirrors OpenCV.
        return self.chessboard_corners

    def checkCharucoCornersCollinear(self, _ids: np.ndarray) -> bool:  # noqa: N802
        return self.collinear


def test_match_charuco_image_points_indexes_chessboard_fallback():
    board = _IndexedChessboardBoard()
    ids = np.array([5, 0, 7, 2], dtype=np.int32)
    corners = np.arange(8, dtype=np.float32).reshape(4, 2)

    object_points, image_points = match_charuco_image_points(board, corners, ids)

    np.testing.assert_array_equal(object_points, board.chessboard_corners[ids].astype(np.float64))
    np.testing.assert_array_equal(image_points, corners.astype(np.float64))


@pytest.mark.parametrize(
    ("corners", "ids", "error"),
    [
        (np.zeros((3, 2)), np.array([0, 1, 2]), "at least 4"),
        (np.zeros((4, 2)), np.array([0, 1, 2]), "same length"),
        (np.zeros((4, 2)), np.array([0, 1, 1, 2]), "unique"),
        (np.zeros((4, 2)), np.array([-1, 0, 1, 2]), "bounds"),
        (np.zeros((4, 2)), np.array([0, 1, 2, 24]), "bounds"),
    ],
)
def test_match_charuco_image_points_rejects_invalid_correspondences(
    corners: np.ndarray, ids: np.ndarray, error: str
):
    with pytest.raises(ValueError, match=error):
        match_charuco_image_points(_IndexedChessboardBoard(), corners, ids)


def test_match_charuco_image_points_rejects_collinear_corner_ids():
    with pytest.raises(ValueError, match="collinear"):
        match_charuco_image_points(
            _IndexedChessboardBoard(collinear=True),
            np.zeros((4, 2)),
            np.array([0, 1, 2, 3]),
        )


def test_estimate_board_pose_recovers_deterministic_synthetic_transform(shipped_config: dict):
    board = create_charuco_board(shipped_config["charuco"])
    object_points_board_m = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    charuco_ids = np.arange(len(object_points_board_m), dtype=np.int32)
    camera_matrix = np.array(
        [
            [850.0, 0.0, 320.0],
            [0.0, 830.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    distortion_coefficients = np.zeros(5)
    expected_rvec_camera_board = np.array([0.20, -0.12, 0.07])
    expected_tvec_camera_board_m = np.array([0.04, -0.03, 0.75])
    projected_points, _ = cv2.projectPoints(
        object_points_board_m,
        expected_rvec_camera_board,
        expected_tvec_camera_board_m,
        camera_matrix,
        distortion_coefficients,
    )
    expected_rotation_camera_board, _ = cv2.Rodrigues(expected_rvec_camera_board)
    expected_T_camera_board = make_transform(  # noqa: N806 - coordinate-frame notation aids the test.
        expected_rotation_camera_board,
        expected_tvec_camera_board_m,
        name="T_camera_board",
    )

    estimate = estimate_board_pose(
        board=board,
        charuco_corners=projected_points,
        charuco_ids=charuco_ids,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
    )

    np.testing.assert_allclose(estimate.rvec_camera_board, expected_rvec_camera_board, atol=1e-7)
    np.testing.assert_allclose(estimate.tvec_camera_board_m, expected_tvec_camera_board_m, atol=1e-7)
    np.testing.assert_allclose(estimate.T_camera_board, expected_T_camera_board, atol=1e-7)
    np.testing.assert_allclose(estimate.object_points_board_m, object_points_board_m)
    np.testing.assert_allclose(estimate.image_points_px, projected_points.reshape(-1, 2))
    assert estimate.reprojection_error_px < 1e-6


def test_estimate_board_pose_rejects_insufficient_correspondences(shipped_config: dict):
    board = create_charuco_board(shipped_config["charuco"])

    with pytest.raises(ValueError, match="at least 4"):
        estimate_board_pose(
            board=board,
            charuco_corners=np.zeros((3, 2)),
            charuco_ids=np.array([0, 1, 2]),
            camera_matrix=np.eye(3),
            distortion_coefficients=np.zeros(5),
        )


def test_estimate_board_pose_rejects_failed_solve_pnp():
    fake_cv2 = SimpleNamespace(solvePnP=lambda *_args: (False, None, None))

    with pytest.raises(ValueError, match="solvePnP"):
        estimate_board_pose(
            board=_IndexedChessboardBoard(),
            charuco_corners=np.zeros((4, 2)),
            charuco_ids=np.array([0, 1, 6, 7]),
            camera_matrix=np.eye(3),
            distortion_coefficients=np.zeros(5),
            cv2_module=fake_cv2,
        )


def test_laplacian_blur_score_is_higher_for_sharp_checker_pattern():
    rows, columns = np.indices((128, 128))
    checker = (((rows // 8) + (columns // 8)) % 2 * 255).astype(np.uint8)
    sharp_rgb = np.repeat(checker[..., None], 3, axis=2)
    blurred_rgb = cv2.GaussianBlur(sharp_rgb, (15, 15), 0.0)

    sharp_score = laplacian_blur_score(sharp_rgb)
    blurred_score = laplacian_blur_score(blurred_rgb)

    assert sharp_score > blurred_score


class _OverlayAruco:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def drawDetectedMarkers(  # noqa: N802 - mirrors OpenCV.
        self, image: np.ndarray, corners: tuple[np.ndarray, ...], ids: np.ndarray
    ) -> None:
        self.calls.append(("markers", corners[0].shape, ids.shape))
        image[0, 0] = [0, 0, 255]

    def drawDetectedCornersCharuco(  # noqa: N802 - mirrors OpenCV.
        self, image: np.ndarray, corners: np.ndarray, ids: np.ndarray
    ) -> None:
        self.calls.append(("charuco", corners.shape, ids.shape))
        image[0, 1] = [0, 255, 0]


class _OverlayCv2:
    COLOR_RGB2BGR = 1
    COLOR_BGR2RGB = 2

    def __init__(self) -> None:
        self.aruco = _OverlayAruco()
        self.axis_calls: list[tuple[tuple[int, ...], tuple[int, ...], float]] = []

    def cvtColor(self, image: np.ndarray, _conversion_code: int) -> np.ndarray:  # noqa: N802
        return image[..., ::-1].copy()

    def drawFrameAxes(  # noqa: N802 - mirrors OpenCV.
        self,
        image: np.ndarray,
        _camera_matrix: np.ndarray,
        _distortion_coefficients: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        axis_length_m: float,
    ) -> None:
        self.axis_calls.append((rvec.shape, tvec.shape, axis_length_m))
        image[0, 2] = [255, 0, 0]


def test_draw_detection_overlay_preserves_rgb_input_and_draws_all_annotations():
    rgb_image = np.full((12, 14, 3), [11, 22, 33], dtype=np.uint8)
    original_rgb_image = rgb_image.copy()
    detection = CharucoDetection(
        marker_corners=(np.arange(8, dtype=np.float32).reshape(4, 2),),
        marker_ids=np.array([4]),
        charuco_corners=np.arange(8, dtype=np.float32).reshape(4, 2),
        charuco_ids=np.array([0, 1, 6, 7]),
        api_name="test",
    )
    estimate = BoardPoseEstimate(
        rvec_camera_board=np.array([0.1, 0.2, 0.3]),
        tvec_camera_board_m=np.array([0.0, 0.0, 0.7]),
        T_camera_board=np.eye(4),
        object_points_board_m=np.zeros((4, 3)),
        image_points_px=np.zeros((4, 2)),
        reprojection_error_px=0.1,
    )
    fake_cv2 = _OverlayCv2()

    overlay = draw_detection_overlay(
        rgb_image,
        detection,
        estimate,
        np.eye(3),
        np.zeros(5),
        0.08,
        cv2_module=fake_cv2,
    )

    np.testing.assert_array_equal(rgb_image, original_rgb_image)
    assert overlay.shape == rgb_image.shape
    assert not np.shares_memory(overlay, rgb_image)
    assert not np.array_equal(overlay, rgb_image)
    assert fake_cv2.aruco.calls == [
        ("markers", (1, 4, 2), (1, 1)),
        ("charuco", (4, 1, 2), (4, 1)),
    ]
    assert fake_cv2.axis_calls == [((3,), (3,), 0.08)]


@pytest.mark.parametrize("model", [None, "none", "distortion.none", "RS2_DISTORTION_NONE"])
def test_opencv_distortion_coefficients_accepts_realsense_none_models(model: object):
    result = opencv_distortion_coefficients(model, [0.1, 0.2, 0.3, 0.4, 0.5])

    np.testing.assert_array_equal(result, np.zeros(5))
    assert result.dtype == np.float64


@pytest.mark.parametrize(
    "model",
    ["brown_conrady", "distortion.brown_conrady", "RS2_DISTORTION_BROWN_CONRADY"],
)
def test_opencv_distortion_coefficients_accepts_realsense_brown_conrady(model: str):
    coefficients = [0.1, -0.2, 0.003, -0.004, 0.05]

    result = opencv_distortion_coefficients(model, coefficients)

    np.testing.assert_array_equal(result, np.array(coefficients, dtype=np.float64))


@pytest.mark.parametrize(
    "model",
    [
        "inverse_brown_conrady",
        "modified_brown_conrady",
        "fisheye",
        "ftheta",
        "kannala_brandt4",
        "unknown",
    ],
)
def test_opencv_distortion_coefficients_rejects_unsupported_models(model: str):
    with pytest.raises(ValueError, match="unsupported.*distortion|Unsupported.*distortion"):
        opencv_distortion_coefficients(model, np.zeros(5))


@pytest.mark.parametrize("coefficients", [[0.0] * 4, [0.0] * 6, [0.0, 0.0, np.nan, 0.0, 0.0]])
def test_opencv_distortion_coefficients_requires_five_finite_values(coefficients: list[float]):
    with pytest.raises(ValueError, match="five finite|5 finite"):
        opencv_distortion_coefficients("brown_conrady", coefficients)


SAMPLE_REQUIRED_FIELDS = {
    "sample_id",
    "image_path",
    "overlay_path",
    "camera_timestamp_ms",
    "robot_timestamp",
    "image_width",
    "image_height",
    "charuco_ids",
    "charuco_corners_px",
    "num_charuco_corners",
    "rvec_camera_board",
    "tvec_camera_board_m",
    "T_camera_board",
    "T_base_ee",
    "robot_pose_raw",
    "reprojection_error_px",
    "opencv_version",
    "realsense_serial",
    "robot_pose_name",
    "translation_unit",
    "matrix_storage_source",
}


def _sample_record(
    *,
    T_base_ee: np.ndarray | None = None,  # noqa: N803 - coordinate-frame notation aids the test.
) -> dict:
    robot_pose = np.eye(4) if T_base_ee is None else T_base_ee
    T_camera_board = make_transform(  # noqa: N806 - coordinate-frame notation aids the test.
        _rotation_z(4.0),
        [0.02, -0.03, 0.72],
        name="T_camera_board",
    )
    return {
        "camera_timestamp_ms": np.float64(1234.5),
        "robot_timestamp": np.int64(8765),
        "image_width": np.int32(3),
        "image_height": np.int32(2),
        "charuco_ids": np.array([0, 1, 6, 7], dtype=np.int32),
        "charuco_corners_px": np.array(
            [[0.25, 0.25], [1.25, 0.25], [0.25, 1.25], [1.25, 1.25]],
            dtype=np.float32,
        ),
        "num_charuco_corners": np.int32(4),
        "rvec_camera_board": np.array([0.01, -0.02, 0.03], dtype=np.float64),
        "tvec_camera_board_m": np.array([0.02, -0.03, 0.72], dtype=np.float64),
        "T_camera_board": T_camera_board,
        "T_base_ee": robot_pose,
        "robot_pose_raw": robot_pose.copy(),
        "reprojection_error_px": np.float32(0.19),
        "opencv_version": cv2.__version__,
        "realsense_serial": "f0123456",
        "robot_pose_name": "T_base_ee",
        "translation_unit": "meter",
        "matrix_storage_source": "existing_franka_client",
        "blur_score": np.float32(321.5),
    }


def _sample_images() -> tuple[np.ndarray, np.ndarray]:
    rgb_image = np.array(
        [
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[17, 33, 65], [90, 45, 10], [1, 2, 3]],
        ],
        dtype=np.uint8,
    )
    rgb_overlay = np.flip(rgb_image, axis=1).copy()
    return rgb_image, rgb_overlay


def test_sample_store_creates_layout_and_round_trips_rgb_png_channels(tmp_path: Path):
    store = HandEyeSampleStore(tmp_path)
    rgb_image, rgb_overlay = _sample_images()

    saved = store.save(_sample_record(), rgb_image, rgb_overlay)

    assert (tmp_path / "images").is_dir()
    assert (tmp_path / "overlays").is_dir()
    assert saved["sample_id"] == 0
    assert saved["image_path"] == "images/sample_000.png"
    assert saved["overlay_path"] == "overlays/sample_000.png"
    decoded_image = cv2.cvtColor(cv2.imread(str(tmp_path / saved["image_path"])), cv2.COLOR_BGR2RGB)
    decoded_overlay = cv2.cvtColor(cv2.imread(str(tmp_path / saved["overlay_path"])), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(decoded_image, rgb_image)
    np.testing.assert_array_equal(decoded_overlay, rgb_overlay)


def test_sample_store_writes_strict_jsonl_schema_and_preserves_extra_metadata(tmp_path: Path):
    store = HandEyeSampleStore(tmp_path)
    rgb_image, rgb_overlay = _sample_images()

    saved = store.save(_sample_record(), rgb_image, rgb_overlay)

    manifest_record = json.loads((tmp_path / "samples.jsonl").read_text(encoding="utf-8"))
    assert manifest_record.keys() >= SAMPLE_REQUIRED_FIELDS
    assert manifest_record == saved
    assert manifest_record["robot_pose_name"] == "T_base_ee"
    assert manifest_record["translation_unit"] == "meter"
    assert manifest_record["matrix_storage_source"] == "existing_franka_client"
    assert manifest_record["blur_score"] == pytest.approx(321.5)
    assert manifest_record["charuco_ids"] == [0, 1, 6, 7]
    assert manifest_record["num_charuco_corners"] == 4
    assert isinstance(manifest_record["T_base_ee"], list)


def test_sample_store_resumes_without_overwriting_existing_bundles(tmp_path: Path):
    rgb_image, rgb_overlay = _sample_images()
    store = HandEyeSampleStore(tmp_path)
    first = store.save(_sample_record(), rgb_image, rgb_overlay)
    second = store.save(_sample_record(), rgb_overlay, rgb_image)
    first_image_bytes = (tmp_path / first["image_path"]).read_bytes()
    second_overlay_bytes = (tmp_path / second["overlay_path"]).read_bytes()

    resumed = HandEyeSampleStore(tmp_path)

    assert [sample["sample_id"] for sample in resumed.samples] == [0, 1]
    assert resumed.next_sample_id == 2
    third = resumed.save(_sample_record(), rgb_image, rgb_overlay)
    assert third["sample_id"] == 2
    assert (tmp_path / first["image_path"]).read_bytes() == first_image_bytes
    assert (tmp_path / second["overlay_path"]).read_bytes() == second_overlay_bytes


def test_sample_store_rolls_back_new_images_when_manifest_commit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = HandEyeSampleStore(tmp_path)
    rgb_image, rgb_overlay = _sample_images()
    real_replace = handeye_utils.os.replace

    def fail_manifest_replace(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == tmp_path / "samples.jsonl":
            raise OSError("injected manifest failure")
        real_replace(source, destination)

    monkeypatch.setattr(handeye_utils.os, "replace", fail_manifest_replace)

    with pytest.raises(OSError, match="injected manifest failure"):
        store.save(_sample_record(), rgb_image, rgb_overlay)

    assert store.samples == []
    assert not (tmp_path / "samples.jsonl").exists()
    assert list((tmp_path / "images").iterdir()) == []
    assert list((tmp_path / "overlays").iterdir()) == []


def test_sample_store_removes_first_temporary_image_when_second_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = HandEyeSampleStore(tmp_path)
    rgb_image, rgb_overlay = _sample_images()
    real_write_temporary_bytes = handeye_utils._write_temporary_bytes
    write_count = 0

    def fail_second_temporary_write(directory: Path, *, prefix: str, payload: bytes) -> Path:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise OSError("injected image write failure")
        return real_write_temporary_bytes(directory, prefix=prefix, payload=payload)

    monkeypatch.setattr(handeye_utils, "_write_temporary_bytes", fail_second_temporary_write)

    with pytest.raises(OSError, match="injected image write failure"):
        store.save(_sample_record(), rgb_image, rgb_overlay)

    assert not (tmp_path / "samples.jsonl").exists()
    assert list((tmp_path / "images").iterdir()) == []
    assert list((tmp_path / "overlays").iterdir()) == []


def test_sample_store_delete_last_rewrites_manifest_and_removes_only_highest_sample(tmp_path: Path):
    rgb_image, rgb_overlay = _sample_images()
    store = HandEyeSampleStore(tmp_path)
    first = store.save(_sample_record(), rgb_image, rgb_overlay)
    second = store.save(_sample_record(), rgb_overlay, rgb_image)

    deleted = store.delete_last()

    assert deleted == second
    assert [sample["sample_id"] for sample in store.samples] == [0]
    assert json.loads((tmp_path / "samples.jsonl").read_text(encoding="utf-8")) == first
    assert (tmp_path / first["image_path"]).exists()
    assert (tmp_path / first["overlay_path"]).exists()
    assert not (tmp_path / second["image_path"]).exists()
    assert not (tmp_path / second["overlay_path"]).exists()
    assert store.delete_last() == first
    assert store.delete_last() is None
    assert (tmp_path / "samples.jsonl").read_text(encoding="utf-8") == ""


def test_sample_store_rejects_malformed_jsonl_with_line_context(tmp_path: Path):
    (tmp_path / "samples.jsonl").write_text("\n{not-json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"samples\.jsonl line 2.*malformed JSON"):
        HandEyeSampleStore(tmp_path)


def test_sample_store_rejects_non_object_jsonl_record_with_line_context(tmp_path: Path):
    (tmp_path / "samples.jsonl").write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"samples\.jsonl line 1.*object"):
        HandEyeSampleStore(tmp_path)


def test_sample_store_rejects_duplicate_sample_ids_with_line_context(tmp_path: Path):
    rgb_image, rgb_overlay = _sample_images()
    store = HandEyeSampleStore(tmp_path)
    store.save(_sample_record(), rgb_image, rgb_overlay)
    row = (tmp_path / "samples.jsonl").read_text(encoding="utf-8").strip()
    (tmp_path / "samples.jsonl").write_text(f"{row}\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"samples\.jsonl line 2.*duplicate sample_id 0"):
        HandEyeSampleStore(tmp_path)


def test_sample_store_rejects_noncanonical_image_path_with_line_context(tmp_path: Path):
    rgb_image, rgb_overlay = _sample_images()
    store = HandEyeSampleStore(tmp_path)
    saved = store.save(_sample_record(), rgb_image, rgb_overlay)
    saved["image_path"] = "images/wrong-name.png"
    (tmp_path / "samples.jsonl").write_text(json.dumps(saved) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"samples\.jsonl line 1.*image_path.*images/sample_000\.png"):
        HandEyeSampleStore(tmp_path)


def test_sample_store_reserves_canonical_orphan_ids_from_both_image_directories(tmp_path: Path):
    store = HandEyeSampleStore(tmp_path)
    (tmp_path / "images" / "sample_007.png").write_bytes(b"orphan image")
    (tmp_path / "overlays" / "sample_012.png").write_bytes(b"orphan overlay")

    assert store.next_sample_id == 13

    rgb_image, rgb_overlay = _sample_images()
    saved = store.save(_sample_record(), rgb_image, rgb_overlay)
    assert saved["sample_id"] == 13
    assert (tmp_path / "images" / "sample_007.png").read_bytes() == b"orphan image"
    assert (tmp_path / "overlays" / "sample_012.png").read_bytes() == b"orphan overlay"


def test_sample_store_loads_gapped_manifest_in_sample_id_order(tmp_path: Path):
    rgb_image, rgb_overlay = _sample_images()
    store = HandEyeSampleStore(tmp_path)
    saved = store.save(_sample_record(), rgb_image, rgb_overlay)
    image_bytes = (tmp_path / saved["image_path"]).read_bytes()
    overlay_bytes = (tmp_path / saved["overlay_path"]).read_bytes()
    records = []
    for sample_id in (9, 3):
        record = {
            **saved,
            "sample_id": sample_id,
            "image_path": f"images/sample_{sample_id:03d}.png",
            "overlay_path": f"overlays/sample_{sample_id:03d}.png",
        }
        (tmp_path / record["image_path"]).write_bytes(image_bytes)
        (tmp_path / record["overlay_path"]).write_bytes(overlay_bytes)
        records.append(record)
    (tmp_path / saved["image_path"]).unlink()
    (tmp_path / saved["overlay_path"]).unlink()
    (tmp_path / "samples.jsonl").write_text(
        "\n" + "\n\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    resumed = HandEyeSampleStore(tmp_path)

    assert [sample["sample_id"] for sample in resumed.samples] == [3, 9]
    assert resumed.next_sample_id == 10


def test_sample_store_reports_nearest_existing_robot_pose_delta(tmp_path: Path):
    store = HandEyeSampleStore(tmp_path)
    assert store.nearest_pose_delta(_pose_at()) is None
    rgb_image, rgb_overlay = _sample_images()
    store.save(_sample_record(T_base_ee=_pose_at(x_m=0.30, rotation_deg=30.0)), rgb_image, rgb_overlay)
    store.save(_sample_record(T_base_ee=_pose_at(x_m=0.05, rotation_deg=5.0)), rgb_image, rgb_overlay)

    nearest = store.nearest_pose_delta(_pose_at())

    assert nearest is not None
    assert nearest.sample_id == 1
    assert nearest.translation_m == pytest.approx(0.05)
    assert nearest.rotation_deg == pytest.approx(5.0)


VALIDATION_CONFIG = {
    "target_scatter_translation_m": 0.005,
    "target_scatter_rotation_deg": 2.0,
    "leave_one_out_translation_m": 0.010,
    "leave_one_out_rotation_deg": 3.0,
    "robust_mad_multiplier": 3.5,
}


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


def _synthetic_eye_to_hand_dataset() -> tuple[list[dict], np.ndarray, np.ndarray]:
    base_camera = make_transform(
        _axis_angle_rotation([0.3, -0.5, 0.8], 23.0),
        [0.72, -0.31, 0.88],
        name="T_base_camera",
    )
    ee_board = make_transform(
        _axis_angle_rotation([-0.4, 0.7, 0.2], -17.0),
        [0.08, 0.01, 0.14],
        name="T_ee_board",
    )
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
    angles_deg = (-42.0, -31.0, -24.0, -13.0, -7.0, 9.0, 16.0, 22.0, 29.0, 37.0, 45.0, 53.0)
    samples: list[dict] = []
    camera_base = invert_transform(base_camera, name="T_base_camera")
    for index, (axis, angle_deg) in enumerate(zip(axes, angles_deg, strict=True)):
        base_ee = make_transform(
            _axis_angle_rotation(axis, angle_deg),
            [
                0.34 + 0.025 * (index % 4),
                -0.22 + 0.031 * ((index * 2) % 5),
                0.39 + 0.019 * ((index * 3) % 7),
            ],
            name="T_base_ee",
        )
        camera_board = camera_base @ base_ee @ ee_board
        samples.append(
            {
                "sample_id": 100 + index,
                "T_base_ee": base_ee,
                "T_camera_board": camera_board,
            }
        )
    return samples, base_camera, ee_board


class _ReturningHandEyeCv2:
    def __init__(self, returned_transform: np.ndarray, *, failed_methods: set[int] | None = None) -> None:
        self.returned_transform = returned_transform
        self.failed_methods = set() if failed_methods is None else failed_methods
        self.calls: list[dict[str, object]] = []

    def calibrateHandEye(  # noqa: N802 - mirrors OpenCV's public API.
        self,
        rotations_gripper_to_base: object,
        translations_gripper_to_base: object,
        rotations_target_to_camera: object,
        translations_target_to_camera: object,
        *,
        method: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(
            {
                "rotations_gripper_to_base": rotations_gripper_to_base,
                "translations_gripper_to_base": translations_gripper_to_base,
                "rotations_target_to_camera": rotations_target_to_camera,
                "translations_target_to_camera": translations_target_to_camera,
                "method": method,
            }
        )
        if method in self.failed_methods:
            raise RuntimeError(f"injected method failure {method}")
        return self.returned_transform[:3, :3].copy(), self.returned_transform[:3, 3:4].copy()


def test_hand_eye_method_mapping_resolves_all_four_opencv_constants():
    assert HAND_EYE_METHODS == {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }


def test_calibrate_eye_to_hand_passes_inverse_robot_poses_and_direct_camera_board_poses():
    samples, base_camera, _ = _synthetic_eye_to_hand_dataset()
    fake_cv2 = _ReturningHandEyeCv2(base_camera)

    result = calibrate_eye_to_hand(samples[:3], method="PARK", cv2_module=fake_cv2)

    np.testing.assert_allclose(result, base_camera, atol=1e-12)
    assert len(fake_cv2.calls) == 1
    call = fake_cv2.calls[0]
    assert call["method"] == cv2.CALIB_HAND_EYE_PARK
    for index, sample in enumerate(samples[:3]):
        ee_base = invert_transform(sample["T_base_ee"], name="T_base_ee")
        np.testing.assert_allclose(call["rotations_gripper_to_base"][index], ee_base[:3, :3])
        np.testing.assert_allclose(call["translations_gripper_to_base"][index], ee_base[:3, 3])
        np.testing.assert_allclose(
            call["rotations_target_to_camera"][index], sample["T_camera_board"][:3, :3]
        )
        np.testing.assert_allclose(
            call["translations_target_to_camera"][index], sample["T_camera_board"][:3, 3]
        )


def test_calibrate_eye_to_hand_reports_missing_opencv_capability():
    samples, _, _ = _synthetic_eye_to_hand_dataset()

    with pytest.raises(RuntimeError, match=r"cv2\.calibrateHandEye.*unavailable|calibrateHandEye.*callable"):
        calibrate_eye_to_hand(samples[:3], method="TSAI", cv2_module=SimpleNamespace(__version__="5.0.0"))


@pytest.mark.parametrize(
    ("samples_slice", "method", "error"),
    [
        (slice(0, 2), "TSAI", "at least 3"),
        (slice(0, 3), "UNKNOWN", "method"),
        (slice(0, 3), True, "method"),
    ],
)
def test_calibrate_eye_to_hand_rejects_invalid_sample_count_or_method(
    samples_slice: slice, method: object, error: str
):
    samples, base_camera, _ = _synthetic_eye_to_hand_dataset()

    with pytest.raises(ValueError, match=error):
        calibrate_eye_to_hand(
            samples[samples_slice],
            method=method,
            cv2_module=_ReturningHandEyeCv2(base_camera),
        )


def test_calibrate_eye_to_hand_validates_both_named_sample_transforms():
    samples, base_camera, _ = _synthetic_eye_to_hand_dataset()
    missing_camera_board = [dict(sample) for sample in samples[:3]]
    missing_camera_board[1].pop("T_camera_board")

    with pytest.raises(ValueError, match=r"samples\[1\]\.T_camera_board"):
        calibrate_eye_to_hand(
            missing_camera_board,
            method=cv2.CALIB_HAND_EYE_TSAI,
            cv2_module=_ReturningHandEyeCv2(base_camera),
        )


def test_mean_rigid_transform_projects_arithmetic_rotation_mean_to_so3():
    transforms = [
        make_transform(np.eye(3), [0.0, 0.0, 0.0], name="T_0"),
        make_transform(_axis_angle_rotation([1, 0, 0], 180.0), [0.3, 0.0, 0.0], name="T_1"),
        make_transform(_axis_angle_rotation([0, 1, 0], 180.0), [0.0, 0.6, 0.0], name="T_2"),
    ]

    mean_transform = mean_rigid_transform(transforms)

    np.testing.assert_allclose(mean_transform[:3, :3].T @ mean_transform[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(mean_transform[:3, :3]) == pytest.approx(1.0)
    np.testing.assert_allclose(mean_transform[:3, 3], [0.1, 0.2, 0.0], atol=1e-12)


def test_exact_closure_validates_constant_target_and_retains_all_method_results():
    samples, base_camera, ee_board = _synthetic_eye_to_hand_dataset()
    fake_cv2 = _ReturningHandEyeCv2(base_camera)

    diversity = assess_pose_diversity(samples)
    report = solve_all_methods(samples, VALIDATION_CONFIG, cv2_module=fake_cv2)

    assert diversity["is_diverse"] is True
    assert diversity["pose_count"] == 12
    assert diversity["usable_relative_rotation_axis_count"] >= 2
    assert set(report["methods"]) == set(HAND_EYE_METHODS)
    assert report["recommended_method"] == "TSAI"
    np.testing.assert_allclose(report["recommended_T_base_camera"], base_camera, atol=1e-12)
    assert {call["method"] for call in fake_cv2.calls} == set(HAND_EYE_METHODS.values())
    for method_name in HAND_EYE_METHODS:
        method_report = report["methods"][method_name]
        assert method_report["status"] == "success"
        np.testing.assert_allclose(method_report["T_base_camera"], base_camera, atol=1e-12)
        validation = method_report["validation"]
        np.testing.assert_allclose(validation["mean_T_ee_board"], ee_board, atol=1e-12)
        np.testing.assert_allclose(validation["mean_translation_m"], ee_board[:3, 3], atol=1e-12)
        np.testing.assert_allclose(validation["translation_component_std_m"], np.zeros(3), atol=1e-12)
        np.testing.assert_allclose(validation["mean_rotation_matrix"], ee_board[:3, :3], atol=1e-12)
        assert validation["translation_error_m"]["max"] < 1e-12
        assert validation["rotation_geodesic_error_deg"]["max"] < 1e-6
        assert validation["per_sample_errors"].keys() == {str(sample["sample_id"]) for sample in samples}
        assert validation["statutory_outlier_ids"] == []
        assert validation["robust_outlier_ids"] == []
        assert validation["outlier_ids"] == []
        leave_one_out = method_report["leave_one_out"]
        assert len(leave_one_out["per_omission"]) == len(samples)
        assert leave_one_out["calibration_stability_translation_m"]["max"] < 1e-12
        assert leave_one_out["calibration_stability_rotation_deg"]["max"] < 1e-6
        assert leave_one_out["configured_influential_ids"] == []
        assert leave_one_out["robust_influential_ids"] == []
        assert leave_one_out["influential_ids"] == []
        terms = method_report["recommendation_terms"]
        assert set(terms) == {
            "validation_translation",
            "validation_rotation",
            "leave_one_out_translation",
            "leave_one_out_rotation",
        }
        for term in terms.values():
            assert set(term) == {"value", "threshold", "normalized", "unit"}


def test_validation_and_leave_one_out_flag_a_perturbed_sample():
    samples, base_camera, _ = _synthetic_eye_to_hand_dataset()
    bad_sample_id = samples[-1]["sample_id"]
    board_perturbation = make_transform(
        _axis_angle_rotation([0.2, 0.9, -0.3], 12.0),
        [0.08, -0.04, 0.03],
        name="T_board_perturbation",
    )
    samples[-1] = {
        **samples[-1],
        "T_camera_board": samples[-1]["T_camera_board"] @ board_perturbation,
    }

    validation = validate_eye_to_hand_result(samples, base_camera, VALIDATION_CONFIG)
    leave_one_out = leave_one_out_validation(
        samples,
        base_camera,
        "HORAUD",
        VALIDATION_CONFIG,
        cv2_module=_ReturningHandEyeCv2(base_camera),
    )

    assert bad_sample_id in validation["statutory_outlier_ids"]
    assert bad_sample_id in validation["robust_outlier_ids"]
    assert bad_sample_id in validation["outlier_ids"]
    assert bad_sample_id in leave_one_out["configured_influential_ids"]
    assert bad_sample_id in leave_one_out["robust_influential_ids"]
    assert bad_sample_id in leave_one_out["influential_ids"]
    bad_error = validation["per_sample_errors"][str(bad_sample_id)]
    assert bad_error["translation_m"] > VALIDATION_CONFIG["target_scatter_translation_m"]
    assert bad_error["rotation_deg"] > VALIDATION_CONFIG["target_scatter_rotation_deg"]


def test_assess_pose_diversity_distinguishes_nonparallel_axes_from_degenerate_sets():
    samples, _, _ = _synthetic_eye_to_hand_dataset()
    diverse = assess_pose_diversity(samples)
    single_axis = assess_pose_diversity(
        [
            {
                "sample_id": index,
                "T_base_ee": make_transform(
                    _rotation_z(index * 15.0), [index * 0.03, 0.0, 0.0], name="T_base_ee"
                ),
            }
            for index in range(5)
        ]
    )
    translation_only = assess_pose_diversity(
        [
            {
                "sample_id": index,
                "T_base_ee": make_transform(np.eye(3), [index * 0.04, 0.0, 0.0], name="T_base_ee"),
            }
            for index in range(5)
        ]
    )
    near_identical = assess_pose_diversity(
        [
            {
                "sample_id": index,
                "T_base_ee": make_transform(
                    _axis_angle_rotation([1.0, index + 1.0, 0.2], index * 0.2),
                    [index * 0.0001, 0.0, 0.0],
                    name="T_base_ee",
                ),
            }
            for index in range(5)
        ]
    )

    assert diverse["is_diverse"] is True
    assert (
        diverse["max_nonparallel_axis_separation_deg"]
        >= diverse["thresholds"]["min_nonparallel_axis_separation_deg"]
    )
    assert single_axis["is_diverse"] is False
    assert single_axis["translation_span_m"] > 0.0
    assert single_axis["max_nonparallel_axis_separation_deg"] == pytest.approx(0.0, abs=1e-6)
    assert translation_only["is_diverse"] is False
    assert translation_only["usable_relative_rotation_axis_count"] == 0
    assert near_identical["is_diverse"] is False
    assert (
        near_identical["relative_rotation_span_deg"]
        < near_identical["thresholds"]["min_relative_rotation_deg"]
    )


def test_solve_all_methods_isolates_failures_and_exposes_dimensionless_recommendation_terms():
    samples, base_camera, _ = _synthetic_eye_to_hand_dataset()
    fake_cv2 = _ReturningHandEyeCv2(
        base_camera,
        failed_methods={HAND_EYE_METHODS["PARK"]},
    )

    report = solve_all_methods(samples, VALIDATION_CONFIG, cv2_module=fake_cv2)

    assert report["methods"]["PARK"]["status"] == "error"
    assert "injected method failure" in report["methods"]["PARK"]["error"]
    assert report["recommended_method"] in {"TSAI", "HORAUD", "DANIILIDIS"}
    for method_name in ("TSAI", "HORAUD", "DANIILIDIS"):
        method_report = report["methods"][method_name]
        assert method_report["status"] == "success"
        for term in method_report["recommendation_terms"].values():
            assert term["unit"] in {"meter", "degree"}
            assert term["normalized"] == pytest.approx(term["value"] / term["threshold"])
        assert method_report["recommendation_score"] == pytest.approx(
            np.mean([term["normalized"] for term in method_report["recommendation_terms"].values()])
        )


def test_solve_all_methods_raises_a_clear_error_when_every_method_fails():
    samples, base_camera, _ = _synthetic_eye_to_hand_dataset()
    fake_cv2 = _ReturningHandEyeCv2(base_camera, failed_methods=set(HAND_EYE_METHODS.values()))

    with pytest.raises(
        RuntimeError, match=r"All hand-eye calibration methods failed.*TSAI.*PARK.*HORAUD.*DANIILIDIS"
    ):
        solve_all_methods(samples, VALIDATION_CONFIG, cv2_module=fake_cv2)


@pytest.mark.skipif(
    not callable(getattr(cv2, "calibrateHandEye", None)),
    reason="installed OpenCV build has no calibrateHandEye capability",
)
@pytest.mark.parametrize("method_name", tuple(HAND_EYE_METHODS))
def test_real_opencv_hand_eye_methods_recover_exact_synthetic_transform(method_name: str):
    samples, base_camera, _ = _synthetic_eye_to_hand_dataset()

    result = calibrate_eye_to_hand(samples, method=method_name)

    translation_m, rotation_deg = pose_delta(result, base_camera)
    assert translation_m < 5e-5
    assert rotation_deg < 0.05
