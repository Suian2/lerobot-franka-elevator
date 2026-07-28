from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

CHARUCO_DICTIONARY = "DICT_5X5_100"

HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

# Pose-diversity gates are deliberately conservative and independent of the
# capture-time pose-similarity warning. A solve needs at least three poses, a
# five-degree usable relative rotation, and two rotation axes separated by at
# least fifteen degrees. Translation span is reported but is not a substitute
# for rotational excitation in hand-eye calibration.
_DIVERSITY_MIN_POSE_COUNT = 3
_DIVERSITY_MIN_RELATIVE_ROTATION_DEG = 5.0
_DIVERSITY_MIN_NONPARALLEL_AXIS_SEPARATION_DEG = 15.0


def _require_mapping(mapping: Mapping[str, Any], key: str, *, context: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}.{key} must be a mapping")
    return value


def _require_int(mapping: Mapping[str, Any], key: str, *, context: str, minimum: int) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context}.{key} must be an integer >= {minimum}")
    return value


def _require_real(
    mapping: Mapping[str, Any],
    key: str,
    *,
    context: str,
    minimum: float,
    inclusive: bool = False,
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context}.{key} must be a finite number")

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{context}.{key} must be a finite number")
    meets_minimum = numeric_value >= minimum if inclusive else numeric_value > minimum
    if not meets_minimum:
        operator = ">=" if inclusive else ">"
        raise ValueError(f"{context}.{key} must be {operator} {minimum}")
    return numeric_value


def max_charuco_corners(charuco_config: Mapping[str, Any]) -> int:
    """Return the number of internal chessboard corners for configured dimensions."""
    if not isinstance(charuco_config, Mapping):
        raise ValueError("charuco must be a mapping")
    squares_x = _require_int(charuco_config, "squares_x", context="charuco", minimum=2)
    squares_y = _require_int(charuco_config, "squares_y", context="charuco", minimum=2)
    return (squares_x - 1) * (squares_y - 1)


def _validate_charuco_config(charuco: Mapping[str, Any]) -> None:
    dictionary = charuco.get("dictionary")
    if dictionary != CHARUCO_DICTIONARY:
        raise ValueError(f"charuco.dictionary must be exactly {CHARUCO_DICTIONARY}, got {dictionary!r}")

    max_charuco_corners(charuco)
    square_length_m = _require_real(charuco, "square_length_m", context="charuco", minimum=0.0)
    marker_length_m = _require_real(charuco, "marker_length_m", context="charuco", minimum=0.0)
    if marker_length_m >= square_length_m:
        raise ValueError("charuco.marker_length_m must be smaller than charuco.square_length_m")
    if not isinstance(charuco.get("legacy_pattern"), bool):
        raise ValueError("charuco.legacy_pattern must be a boolean")


def _validate_capture_config(capture: Mapping[str, Any], charuco: Mapping[str, Any]) -> None:
    minimum_corners = _require_int(capture, "min_charuco_corners", context="capture_validation", minimum=1)
    corner_capacity = max_charuco_corners(charuco)
    if minimum_corners > corner_capacity:
        raise ValueError(
            f"capture_validation.min_charuco_corners cannot exceed the board maximum of {corner_capacity}"
        )

    warning_px = _require_real(
        capture,
        "warning_reprojection_error_px",
        context="capture_validation",
        minimum=0.0,
        inclusive=True,
    )
    maximum_px = _require_real(
        capture,
        "max_reprojection_error_px",
        context="capture_validation",
        minimum=0.0,
    )
    if warning_px > maximum_px:
        raise ValueError(
            "capture_validation.warning_reprojection_error_px cannot exceed "
            "capture_validation.max_reprojection_error_px"
        )
    _require_real(
        capture,
        "min_laplacian_variance",
        context="capture_validation",
        minimum=0.0,
        inclusive=True,
    )


def _validate_motion_config(config: Mapping[str, Any]) -> None:
    stillness = _require_mapping(config, "robot_stillness", context="config")
    _require_real(stillness, "window_s", context="robot_stillness", minimum=0.0)
    _require_real(stillness, "max_translation_m", context="robot_stillness", minimum=0.0)
    _require_real(stillness, "max_rotation_deg", context="robot_stillness", minimum=0.0)

    similarity = _require_mapping(config, "pose_similarity", context="config")
    _require_real(similarity, "translation_m", context="pose_similarity", minimum=0.0)
    _require_real(similarity, "rotation_deg", context="pose_similarity", minimum=0.0)


def _validate_result_config(config: Mapping[str, Any]) -> None:
    validation = _require_mapping(config, "validation", context="config")
    for key in (
        "target_scatter_translation_m",
        "target_scatter_rotation_deg",
        "leave_one_out_translation_m",
        "leave_one_out_rotation_deg",
        "robust_mad_multiplier",
    ):
        _require_real(validation, key, context="validation", minimum=0.0)


def load_handeye_config(path: str | Path) -> dict[str, Any]:
    """Safely load and validate the hand-eye calibration configuration."""
    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in hand-eye config {config_path}") from exc

    if not isinstance(loaded, dict):
        raise ValueError("Hand-eye config root must be a mapping")

    charuco = _require_mapping(loaded, "charuco", context="config")
    _validate_charuco_config(charuco)
    capture = _require_mapping(loaded, "capture_validation", context="config")
    _validate_capture_config(capture, charuco)
    _validate_motion_config(loaded)
    _validate_result_config(loaded)
    return loaded


@dataclass(frozen=True)
class CharucoDetection:
    """Canonical ChArUco detection output independent of OpenCV API generation."""

    marker_corners: tuple[np.ndarray, ...]
    marker_ids: np.ndarray
    charuco_corners: np.ndarray
    charuco_ids: np.ndarray
    api_name: str

    @property
    def num_charuco_corners(self) -> int:
        return int(np.asarray(self.charuco_ids).size)


@dataclass(frozen=True)
class BoardPoseEstimate:
    """Pose of a metric board in the camera optical frame."""

    rvec_camera_board: np.ndarray
    tvec_camera_board_m: np.ndarray
    T_camera_board: np.ndarray  # noqa: N815 - coordinate-frame notation is the public API.
    object_points_board_m: np.ndarray
    image_points_px: np.ndarray
    reprojection_error_px: float


def create_charuco_board(
    charuco_config: Mapping[str, Any],
    cv2_module: Any = cv2,
) -> Any:
    """Build the configured board through either modern or legacy ArUco factories."""
    if not isinstance(charuco_config, Mapping):
        raise ValueError("charuco must be a mapping")
    _validate_charuco_config(charuco_config)

    aruco = getattr(cv2_module, "aruco", None)
    if aruco is None:
        raise RuntimeError("OpenCV was built without the aruco module")

    dictionary_name = str(charuco_config["dictionary"])
    dictionary_id = getattr(aruco, dictionary_name, None)
    if dictionary_id is None:
        raise RuntimeError(f"OpenCV aruco does not provide {dictionary_name}")

    modern_dictionary_factory = getattr(aruco, "getPredefinedDictionary", None)
    legacy_dictionary_factory = getattr(aruco, "Dictionary_get", None)
    if callable(modern_dictionary_factory):
        dictionary = modern_dictionary_factory(dictionary_id)
    elif callable(legacy_dictionary_factory):
        dictionary = legacy_dictionary_factory(dictionary_id)
    else:
        raise RuntimeError("OpenCV aruco provides neither getPredefinedDictionary nor Dictionary_get")

    squares_x = int(charuco_config["squares_x"])
    squares_y = int(charuco_config["squares_y"])
    square_length_m = float(charuco_config["square_length_m"])
    marker_length_m = float(charuco_config["marker_length_m"])
    modern_board_constructor = getattr(aruco, "CharucoBoard", None)
    legacy_board_factory = getattr(aruco, "CharucoBoard_create", None)
    if callable(modern_board_constructor):
        board = modern_board_constructor(
            (squares_x, squares_y),
            square_length_m,
            marker_length_m,
            dictionary,
        )
    elif callable(legacy_board_factory):
        board = legacy_board_factory(
            squares_x,
            squares_y,
            square_length_m,
            marker_length_m,
            dictionary,
        )
    else:
        raise RuntimeError("OpenCV aruco provides neither CharucoBoard nor CharucoBoard_create")

    set_legacy_pattern = getattr(board, "setLegacyPattern", None)
    if callable(set_legacy_pattern):
        set_legacy_pattern(bool(charuco_config["legacy_pattern"]))
    return board


def _normalized_ids(value: Any, *, name: str, strict: bool) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.int32)

    raw_ids = np.asarray(value)
    if raw_ids.size == 0:
        return np.empty((0,), dtype=np.int32)
    if strict and (raw_ids.dtype.kind not in "iu" or raw_ids.dtype.kind == "b"):
        raise ValueError(f"{name} must contain integer IDs")
    try:
        normalized = np.asarray(raw_ids, dtype=np.int32).reshape(-1).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer IDs") from exc
    return normalized


def _normalized_image_points(
    value: Any,
    *,
    name: str,
    dtype: type[np.floating[Any]],
) -> np.ndarray:
    if value is None:
        return np.empty((0, 2), dtype=dtype)
    try:
        points = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric image points") from exc
    if points.size == 0:
        return np.empty((0, 2), dtype=dtype)
    if points.size % 2 != 0:
        raise ValueError(f"{name} must contain pairs of image coordinates")
    normalized = points.reshape(-1, 2).copy()
    if not np.isfinite(normalized).all():
        raise ValueError(f"{name} must contain only finite image coordinates")
    return normalized


def _normalized_marker_corners(value: Any) -> tuple[np.ndarray, ...]:
    if value is None:
        return ()
    try:
        corners = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("marker_corners must contain numeric image points") from exc
    if corners.size == 0:
        return ()
    if corners.size % 8 != 0:
        raise ValueError("Each detected marker must contain exactly four image corners")
    normalized = corners.reshape(-1, 4, 2)
    if not np.isfinite(normalized).all():
        raise ValueError("marker_corners must contain only finite image coordinates")
    return tuple(marker.copy() for marker in normalized)


def _build_charuco_detection(
    *,
    marker_corners: Any,
    marker_ids: Any,
    charuco_corners: Any,
    charuco_ids: Any,
    api_name: str,
) -> CharucoDetection:
    normalized_marker_corners = _normalized_marker_corners(marker_corners)
    normalized_marker_ids = _normalized_ids(marker_ids, name="marker_ids", strict=False)
    normalized_charuco_corners = _normalized_image_points(
        charuco_corners,
        name="charuco_corners",
        dtype=np.float32,
    )
    normalized_charuco_ids = _normalized_ids(charuco_ids, name="charuco_ids", strict=False)

    if len(normalized_marker_corners) != len(normalized_marker_ids):
        raise ValueError("marker_corners and marker_ids must have the same length")
    if len(normalized_charuco_corners) != len(normalized_charuco_ids):
        raise ValueError("charuco_corners and charuco_ids must have the same length")
    return CharucoDetection(
        marker_corners=normalized_marker_corners,
        marker_ids=normalized_marker_ids,
        charuco_corners=normalized_charuco_corners,
        charuco_ids=normalized_charuco_ids,
        api_name=api_name,
    )


def _optional_float64_array(value: Any, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = _as_float64_array(value, name=name)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


class CharucoDetectorCompat:
    """Select the newest available ChArUco detection API and normalize its result."""

    def __init__(
        self,
        board: Any,
        camera_matrix: Any | None = None,
        distortion_coefficients: Any | None = None,
        cv2_module: Any = cv2,
    ) -> None:
        self._board = board
        self._cv2 = cv2_module
        self._camera_matrix = _optional_float64_array(camera_matrix, name="camera_matrix")
        self._distortion_coefficients = _optional_float64_array(
            distortion_coefficients,
            name="distortion_coefficients",
        )
        aruco = getattr(cv2_module, "aruco", None)
        if aruco is None:
            raise RuntimeError("OpenCV was built without the aruco module")
        self._aruco = aruco
        self._detector: Any | None = None

        modern_detector_factory = getattr(aruco, "CharucoDetector", None)
        if callable(modern_detector_factory):
            detector = self._construct_modern_detector(modern_detector_factory)
            if callable(getattr(detector, "detectBoard", None)):
                self._detector = detector
                self.api_name = "CharucoDetector.detectBoard"
                return

        detect_markers = getattr(aruco, "detectMarkers", None)
        interpolate_corners = getattr(aruco, "interpolateCornersCharuco", None)
        if not callable(detect_markers) or not callable(interpolate_corners):
            raise RuntimeError(
                "OpenCV ChArUco fallback requires callable detectMarkers and interpolateCornersCharuco"
            )
        self.api_name = "detectMarkers+interpolateCornersCharuco"

    def _construct_modern_detector(self, detector_factory: Callable[..., Any]) -> Any:
        parameter_factory = getattr(self._aruco, "CharucoParameters", None)
        has_calibration = self._camera_matrix is not None or self._distortion_coefficients is not None
        if not has_calibration or not callable(parameter_factory):
            return detector_factory(self._board)

        parameters = parameter_factory()
        if self._camera_matrix is not None:
            parameters.cameraMatrix = self._camera_matrix
        if self._distortion_coefficients is not None:
            parameters.distCoeffs = self._distortion_coefficients
        try:
            return detector_factory(self._board, parameters)
        except TypeError:
            detector = detector_factory(self._board)
            set_parameters = getattr(detector, "setCharucoParameters", None)
            if callable(set_parameters):
                set_parameters(parameters)
            return detector

    def detect(self, rgb_image: np.ndarray) -> CharucoDetection:
        image = np.asarray(rgb_image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"rgb_image must have shape (height, width, 3), got {image.shape}")
        gray_image = self._cv2.cvtColor(image, self._cv2.COLOR_RGB2GRAY)

        if self._detector is not None:
            charuco_corners, charuco_ids, marker_corners, marker_ids = self._detector.detectBoard(gray_image)
        else:
            get_dictionary = getattr(self._board, "getDictionary", None)
            dictionary = (
                get_dictionary() if callable(get_dictionary) else getattr(self._board, "dictionary", None)
            )
            if dictionary is None:
                raise RuntimeError(
                    "Legacy ChArUco detection requires board.getDictionary() or board.dictionary"
                )
            detection_result = self._aruco.detectMarkers(gray_image, dictionary)
            marker_corners, marker_ids = detection_result[:2]
            if marker_ids is None or np.asarray(marker_ids).size == 0:
                charuco_corners = None
                charuco_ids = None
            else:
                try:
                    interpolation_result = self._aruco.interpolateCornersCharuco(
                        marker_corners,
                        marker_ids,
                        gray_image,
                        self._board,
                        cameraMatrix=self._camera_matrix,
                        distCoeffs=self._distortion_coefficients,
                    )
                except TypeError:
                    interpolation_result = self._aruco.interpolateCornersCharuco(
                        marker_corners,
                        marker_ids,
                        gray_image,
                        self._board,
                        self._camera_matrix,
                        self._distortion_coefficients,
                    )
                _, charuco_corners, charuco_ids = interpolation_result[:3]

        return _build_charuco_detection(
            marker_corners=marker_corners,
            marker_ids=marker_ids,
            charuco_corners=charuco_corners,
            charuco_ids=charuco_ids,
            api_name=self.api_name,
        )


def _charuco_corner_capacity(board: Any, *, allow_corner_lookup: bool) -> int | None:
    get_chessboard_size = getattr(board, "getChessboardSize", None)
    if callable(get_chessboard_size):
        size = tuple(get_chessboard_size())
        if len(size) == 2:
            squares_x, squares_y = (int(value) for value in size)
            return (squares_x - 1) * (squares_y - 1)
    if allow_corner_lookup:
        get_chessboard_corners = getattr(board, "getChessboardCorners", None)
        if callable(get_chessboard_corners):
            return int(np.asarray(get_chessboard_corners()).reshape(-1, 3).shape[0])
    return None


def match_charuco_image_points(
    board: Any,
    corners: Any,
    ids: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated metric board points paired with canonical pixel points."""
    image_points_px = _normalized_image_points(corners, name="charuco_corners", dtype=np.float64)
    charuco_ids = _normalized_ids(ids, name="charuco_ids", strict=True)
    if len(image_points_px) != len(charuco_ids):
        raise ValueError("charuco_corners and charuco_ids must have the same length")
    if len(charuco_ids) < 4:
        raise ValueError("at least 4 ChArUco correspondences are required")
    if len(np.unique(charuco_ids)) != len(charuco_ids):
        raise ValueError("charuco_ids must be unique")
    if np.any(charuco_ids < 0):
        raise ValueError("charuco_ids are outside board corner bounds")

    match_image_points = getattr(board, "matchImagePoints", None)
    capacity = _charuco_corner_capacity(board, allow_corner_lookup=not callable(match_image_points))
    if capacity is not None and np.any(charuco_ids >= capacity):
        raise ValueError("charuco_ids are outside board corner bounds")

    check_collinear = getattr(board, "checkCharucoCornersCollinear", None)
    if callable(check_collinear) and bool(check_collinear(charuco_ids.astype(np.int32).reshape(-1, 1))):
        raise ValueError("ChArUco correspondences are collinear")

    if callable(match_image_points):
        matched_object_points, matched_image_points = match_image_points(
            image_points_px.astype(np.float32).reshape(-1, 1, 2),
            charuco_ids.astype(np.int32).reshape(-1, 1),
        )
        try:
            object_points_board_m = np.asarray(matched_object_points, dtype=np.float64).reshape(-1, 3)
        except (TypeError, ValueError) as exc:
            raise ValueError("Matched ChArUco object points must have three coordinates") from exc
        matched_image_points_px = _normalized_image_points(
            matched_image_points,
            name="matched image points",
            dtype=np.float64,
        )
        # matchImagePoints requires float32 input even on recent OpenCV. Its image
        # output therefore loses precision from synthetic or sub-pixel float64
        # observations. IDs and corners are already paired in the same order, so
        # retain the validated source pixels after checking the matched lengths.
        if len(matched_image_points_px) == len(image_points_px):
            matched_image_points_px = image_points_px.copy()
    else:
        get_chessboard_corners = getattr(board, "getChessboardCorners", None)
        if not callable(get_chessboard_corners):
            raise RuntimeError("ChArUco board provides neither matchImagePoints nor getChessboardCorners")
        try:
            all_object_points_board_m = np.asarray(
                get_chessboard_corners(),
                dtype=np.float64,
            ).reshape(-1, 3)
        except (TypeError, ValueError) as exc:
            raise ValueError("Board chessboard corners must have three coordinates") from exc
        if np.any(charuco_ids >= len(all_object_points_board_m)):
            raise ValueError("charuco_ids are outside board corner bounds")
        object_points_board_m = all_object_points_board_m[charuco_ids].copy()
        matched_image_points_px = image_points_px.copy()

    if len(object_points_board_m) != len(charuco_ids) or len(matched_image_points_px) != len(charuco_ids):
        raise ValueError("Matched object, image, and ID arrays must have the same length")
    if not np.isfinite(object_points_board_m).all():
        raise ValueError("Matched ChArUco object points must contain only finite values")
    if not np.isfinite(matched_image_points_px).all():
        raise ValueError("Matched ChArUco image points must contain only finite values")
    return object_points_board_m.copy(), matched_image_points_px.copy()


def _validated_camera_matrix(camera_matrix: Any) -> np.ndarray:
    matrix = _as_float64_array(camera_matrix, name="camera_matrix")
    if matrix.shape != (3, 3):
        raise ValueError(f"camera_matrix must have shape (3, 3), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("camera_matrix must contain only finite values")
    return matrix


def _validated_distortion_coefficients(distortion_coefficients: Any) -> np.ndarray:
    coefficients = _as_float64_array(
        distortion_coefficients,
        name="distortion_coefficients",
    ).reshape(-1)
    if coefficients.size not in {4, 5, 8, 12, 14}:
        raise ValueError("distortion_coefficients must use an OpenCV-supported coefficient count")
    if not np.isfinite(coefficients).all():
        raise ValueError("distortion_coefficients must contain only finite values")
    return coefficients


def estimate_board_pose(
    *,
    board: Any,
    charuco_corners: Any,
    charuco_ids: Any,
    camera_matrix: Any,
    distortion_coefficients: Any,
    cv2_module: Any = cv2,
) -> BoardPoseEstimate:
    """Estimate ``T_camera_board`` from validated ChArUco correspondences."""
    object_points_board_m, image_points_px = match_charuco_image_points(
        board,
        charuco_corners,
        charuco_ids,
    )
    resolved_camera_matrix = _validated_camera_matrix(camera_matrix)
    resolved_distortion_coefficients = _validated_distortion_coefficients(distortion_coefficients)

    solved, rvec_camera_board, tvec_camera_board_m = cv2_module.solvePnP(
        object_points_board_m,
        image_points_px,
        resolved_camera_matrix,
        resolved_distortion_coefficients,
    )
    if not solved or rvec_camera_board is None or tvec_camera_board_m is None:
        raise ValueError("OpenCV solvePnP failed to estimate T_camera_board")

    rvec_camera_board = _as_float64_array(
        rvec_camera_board,
        name="rvec_camera_board",
    ).reshape(-1)
    tvec_camera_board_m = _as_float64_array(
        tvec_camera_board_m,
        name="tvec_camera_board_m",
    ).reshape(-1)
    if rvec_camera_board.shape != (3,) or not np.isfinite(rvec_camera_board).all():
        raise ValueError("OpenCV solvePnP returned an invalid rvec_camera_board")
    if tvec_camera_board_m.shape != (3,) or not np.isfinite(tvec_camera_board_m).all():
        raise ValueError("OpenCV solvePnP returned an invalid tvec_camera_board_m")

    rotation_camera_board, _ = cv2_module.Rodrigues(rvec_camera_board)
    T_camera_board = make_transform(  # noqa: N806 - coordinate-frame notation is the public API.
        rotation_camera_board,
        tvec_camera_board_m,
        name="T_camera_board",
    )
    reprojected_points, _ = cv2_module.projectPoints(
        object_points_board_m,
        rvec_camera_board,
        tvec_camera_board_m,
        resolved_camera_matrix,
        resolved_distortion_coefficients,
    )
    reprojected_points_px = _normalized_image_points(
        reprojected_points,
        name="reprojected image points",
        dtype=np.float64,
    )
    residuals_px = reprojected_points_px - image_points_px
    reprojection_error_px = float(np.sqrt(np.mean(np.sum(residuals_px**2, axis=1))))
    return BoardPoseEstimate(
        rvec_camera_board=rvec_camera_board.copy(),
        tvec_camera_board_m=tvec_camera_board_m.copy(),
        T_camera_board=T_camera_board,
        object_points_board_m=object_points_board_m.copy(),
        image_points_px=image_points_px.copy(),
        reprojection_error_px=reprojection_error_px,
    )


def laplacian_blur_score(rgb_image: np.ndarray, cv2_module: Any = cv2) -> float:
    """Return Laplacian variance, where a larger value indicates sharper edges."""
    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"rgb_image must have shape (height, width, 3), got {image.shape}")
    gray_image = cv2_module.cvtColor(image, cv2_module.COLOR_RGB2GRAY)
    laplacian = cv2_module.Laplacian(gray_image, cv2_module.CV_64F)
    return float(np.var(laplacian))


def draw_detection_overlay(
    rgb_image: np.ndarray,
    detection: CharucoDetection,
    estimate: BoardPoseEstimate | None,
    camera_matrix: Any,
    distortion_coefficients: Any,
    axis_length_m: float,
    cv2_module: Any = cv2,
) -> np.ndarray:
    """Draw detections in OpenCV's BGR convention and return an independent RGB image."""
    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"rgb_image must have shape (height, width, 3), got {image.shape}")
    bgr_overlay = np.array(
        cv2_module.cvtColor(image, cv2_module.COLOR_RGB2BGR),
        copy=True,
    )

    marker_corners = _normalized_marker_corners(detection.marker_corners)
    marker_ids = _normalized_ids(detection.marker_ids, name="marker_ids", strict=True)
    if len(marker_corners) != len(marker_ids):
        raise ValueError("marker_corners and marker_ids must have the same length")
    if marker_corners:
        cv2_module.aruco.drawDetectedMarkers(
            bgr_overlay,
            tuple(corner.astype(np.float32).reshape(1, 4, 2) for corner in marker_corners),
            marker_ids.astype(np.int32).reshape(-1, 1),
        )

    charuco_corners = _normalized_image_points(
        detection.charuco_corners,
        name="charuco_corners",
        dtype=np.float32,
    )
    charuco_ids = _normalized_ids(detection.charuco_ids, name="charuco_ids", strict=True)
    if len(charuco_corners) != len(charuco_ids):
        raise ValueError("charuco_corners and charuco_ids must have the same length")
    if len(charuco_corners):
        cv2_module.aruco.drawDetectedCornersCharuco(
            bgr_overlay,
            charuco_corners.reshape(-1, 1, 2),
            charuco_ids.astype(np.int32).reshape(-1, 1),
        )

    if estimate is not None:
        numeric_axis_length_m = float(axis_length_m)
        if not math.isfinite(numeric_axis_length_m) or numeric_axis_length_m <= 0.0:
            raise ValueError("axis_length_m must be a finite positive number")
        cv2_module.drawFrameAxes(
            bgr_overlay,
            _validated_camera_matrix(camera_matrix),
            _validated_distortion_coefficients(distortion_coefficients),
            _as_float64_array(estimate.rvec_camera_board, name="rvec_camera_board").reshape(3),
            _as_float64_array(
                estimate.tvec_camera_board_m,
                name="tvec_camera_board_m",
            ).reshape(3),
            numeric_axis_length_m,
        )

    rgb_overlay = cv2_module.cvtColor(bgr_overlay, cv2_module.COLOR_BGR2RGB)
    if np.asarray(rgb_overlay).shape != image.shape:
        raise ValueError("OpenCV color conversion changed the overlay image shape")
    return np.array(rgb_overlay, copy=True)


def _realsense_distortion_model_name(model: Any) -> str:
    if model is None:
        return "none"
    enum_name = getattr(model, "name", None)
    raw_name = enum_name if isinstance(enum_name, str) else str(model)
    normalized = raw_name.strip().lower().replace("-", "_").replace(" ", "_")
    for prefix in (
        "pyrealsense2.distortion.",
        "rs.distortion.",
        "distortion.",
        "rs2_distortion_",
    ):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    return normalized


def opencv_distortion_coefficients(model: Any, coefficients: Any) -> np.ndarray:
    """Translate supported RealSense color distortion metadata to OpenCV coefficients."""
    try:
        resolved_coefficients = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("RealSense distortion coefficients must be five finite values") from exc
    if resolved_coefficients.shape != (5,) or not np.isfinite(resolved_coefficients).all():
        raise ValueError("RealSense distortion coefficients must be five finite values")

    model_name = _realsense_distortion_model_name(model)
    if model_name == "none":
        return np.zeros(5, dtype=np.float64)
    if model_name == "brown_conrady":
        return resolved_coefficients.copy()
    raise ValueError(f"Unsupported RealSense distortion model for OpenCV PnP: {model_name!r}")


def _as_float64_array(value: Any, *, name: str) -> np.ndarray:
    try:
        return np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc


def _validate_atol(atol: float) -> float:
    if isinstance(atol, bool) or not isinstance(atol, Real):
        raise ValueError("atol must be a finite non-negative number")
    numeric_atol = float(atol)
    if not math.isfinite(numeric_atol) or numeric_atol < 0.0:
        raise ValueError("atol must be a finite non-negative number")
    return numeric_atol


def _validate_rotation(rotation: Any, *, name: str, atol: float = 1e-6) -> np.ndarray:
    numeric_atol = _validate_atol(atol)
    matrix = _as_float64_array(rotation, name=name)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=numeric_atol, rtol=0.0):
        raise ValueError(f"{name} must be orthonormal")
    determinant = float(np.linalg.det(matrix))
    if not math.isclose(determinant, 1.0, abs_tol=numeric_atol, rel_tol=0.0):
        raise ValueError(f"{name} must have determinant +1, got {determinant}")
    return matrix


def validate_homogeneous_transform(transform: Any, *, name: str, atol: float = 1e-6) -> np.ndarray:
    """Return a validated, independent float64 copy of a rigid 4x4 transform."""
    numeric_atol = _validate_atol(atol)
    matrix = _as_float64_array(transform, name=name)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    expected_last_row = np.array([0.0, 0.0, 0.0, 1.0])
    if not np.allclose(matrix[3], expected_last_row, atol=numeric_atol, rtol=0.0):
        raise ValueError(f"{name} must end with homogeneous row [0, 0, 0, 1]")
    _validate_rotation(matrix[:3, :3], name=f"{name} rotation", atol=numeric_atol)
    return matrix


def make_transform(rotation: Any, translation_m: Any, *, name: str) -> np.ndarray:
    """Build a validated homogeneous transform from an SO(3) rotation and metres."""
    rotation_matrix = _validate_rotation(rotation, name=f"{name} rotation")
    translation = _as_float64_array(translation_m, name=f"{name} translation")
    if translation.shape != (3,):
        raise ValueError(f"{name} translation must have shape (3,), got {translation.shape}")
    if not np.isfinite(translation).all():
        raise ValueError(f"{name} translation must contain only finite values")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return validate_homogeneous_transform(transform, name=name)


def invert_transform(transform: Any, *, name: str) -> np.ndarray:
    """Return the rigid inverse of a named homogeneous transform."""
    matrix = validate_homogeneous_transform(transform, name=name)
    rotation_inverse = matrix[:3, :3].T
    translation_inverse = -rotation_inverse @ matrix[:3, 3]
    return make_transform(rotation_inverse, translation_inverse, name=f"inverse of {name}")


def rotation_delta_deg(rotation_a: Any, rotation_b: Any) -> float:
    """Return the SO(3) geodesic distance between rotations in degrees."""
    matrix_a = _validate_rotation(rotation_a, name="rotation_a")
    matrix_b = _validate_rotation(rotation_b, name="rotation_b")
    relative_rotation = matrix_a.T @ matrix_b
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    if math.isclose(cosine, 1.0, abs_tol=10.0 * np.finfo(np.float64).eps, rel_tol=0.0):
        return 0.0
    return math.degrees(math.acos(cosine))


def pose_delta(
    T_base_ee_a: Any,  # noqa: N803 - coordinate-frame notation is the public API.
    T_base_ee_b: Any,  # noqa: N803 - coordinate-frame notation is the public API.
) -> tuple[float, float]:
    """Return base-frame translation distance in metres and SO(3) distance in degrees."""
    pose_a = validate_homogeneous_transform(T_base_ee_a, name="T_base_ee_a")
    pose_b = validate_homogeneous_transform(T_base_ee_b, name="T_base_ee_b")
    translation_m = float(np.linalg.norm(pose_b[:3, 3] - pose_a[:3, 3]))
    rotation_deg = rotation_delta_deg(pose_a[:3, :3], pose_b[:3, :3])
    return translation_m, rotation_deg


def _resolved_hand_eye_method(method: str | int) -> tuple[str, int]:
    if isinstance(method, str):
        method_name = method.strip().upper()
        if method_name in HAND_EYE_METHODS:
            return method_name, HAND_EYE_METHODS[method_name]
    elif isinstance(method, int) and not isinstance(method, bool):
        for method_name, method_constant in HAND_EYE_METHODS.items():
            if method == method_constant:
                return method_name, method_constant
    accepted_names = ", ".join(HAND_EYE_METHODS)
    raise ValueError(f"method must be one of {accepted_names} or its OpenCV integer constant")


def _require_hand_eye_capability(cv2_module: Any) -> Callable[..., Any]:
    calibrate_hand_eye = getattr(cv2_module, "calibrateHandEye", None)
    if not callable(calibrate_hand_eye):
        version = getattr(cv2_module, "__version__", "unknown")
        raise RuntimeError(
            "cv2.calibrateHandEye is unavailable or not callable in OpenCV "
            f"{version}; use an OpenCV build that provides the hand-eye calibration API"
        )
    return calibrate_hand_eye


def _validated_eye_to_hand_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    minimum: int,
    require_camera_board: bool = True,
) -> list[tuple[int, np.ndarray, np.ndarray | None]]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise ValueError("samples must be a sequence of sample mappings")
    if len(samples) < minimum:
        raise ValueError(f"at least {minimum} hand-eye samples are required, got {len(samples)}")

    validated: list[tuple[int, np.ndarray, np.ndarray | None]] = []
    seen_sample_ids: set[int] = set()
    for index, sample in enumerate(samples):
        context = f"samples[{index}]"
        if not isinstance(sample, Mapping):
            raise ValueError(f"{context} must be a mapping")
        sample_id = _sample_int(sample, "sample_id", context=context)
        if sample_id in seen_sample_ids:
            raise ValueError(f"{context}.sample_id duplicates sample ID {sample_id}")
        seen_sample_ids.add(sample_id)

        if "T_base_ee" not in sample:
            raise ValueError(f"{context}.T_base_ee is required")
        try:
            T_base_ee = validate_homogeneous_transform(  # noqa: N806
                sample["T_base_ee"],
                name=f"{context}.T_base_ee",
            )
        except ValueError as exc:
            raise ValueError(f"{context}.T_base_ee is invalid: {exc}") from exc

        T_camera_board: np.ndarray | None = None  # noqa: N806
        if require_camera_board:
            if "T_camera_board" not in sample:
                raise ValueError(f"{context}.T_camera_board is required")
            try:
                T_camera_board = validate_homogeneous_transform(  # noqa: N806
                    sample["T_camera_board"],
                    name=f"{context}.T_camera_board",
                )
            except ValueError as exc:
                raise ValueError(f"{context}.T_camera_board is invalid: {exc}") from exc
        validated.append((sample_id, T_base_ee, T_camera_board))
    return validated


def calibrate_eye_to_hand(
    samples: Sequence[Mapping[str, Any]],
    *,
    method: str | int,
    cv2_module: Any = cv2,
) -> np.ndarray:
    """Solve and return ``T_base_camera`` using a named or OpenCV hand-eye method."""
    _, method_constant = _resolved_hand_eye_method(method)
    calibrate_hand_eye = _require_hand_eye_capability(cv2_module)
    validated_samples = _validated_eye_to_hand_samples(samples, minimum=3)

    rotations_gripper_to_base: list[np.ndarray] = []
    translations_gripper_to_base: list[np.ndarray] = []
    rotations_target_to_camera: list[np.ndarray] = []
    translations_target_to_camera: list[np.ndarray] = []
    for _, T_base_ee, T_camera_board in validated_samples:  # noqa: N806
        if T_camera_board is None:  # pragma: no cover - enforced by the validator contract
            raise AssertionError("T_camera_board validation was unexpectedly skipped")
        T_ee_base = invert_transform(T_base_ee, name="T_base_ee")  # noqa: N806
        rotations_gripper_to_base.append(T_ee_base[:3, :3].copy())
        translations_gripper_to_base.append(T_ee_base[:3, 3].copy())
        rotations_target_to_camera.append(T_camera_board[:3, :3].copy())
        translations_target_to_camera.append(T_camera_board[:3, 3].copy())

    input_lengths = {
        len(rotations_gripper_to_base),
        len(translations_gripper_to_base),
        len(rotations_target_to_camera),
        len(translations_target_to_camera),
    }
    if len(input_lengths) != 1:
        raise ValueError("OpenCV hand-eye rotation and translation inputs must have equal lengths")

    # Transform convention: ^A T_B maps coordinates from frame B into frame A.
    # With G_i = ^base T_ee, C_i = ^camera T_board,
    # X = ^base T_camera, and Z = ^ee T_board, rigid closure is
    #
    #     G_i Z = X C_i.
    #
    # OpenCV normally consumes gripper2base and target2cam and returns
    # cam2gripper. Supplying G_i^-1 = ^ee T_base in the gripper2base argument
    # relabels OpenCV's logical gripper frame as the physical Franka base. Its
    # returned cam2gripper is therefore physically X = ^base T_camera. It is
    # already T_base_camera and must not be inverted.
    calibration_result = calibrate_hand_eye(
        rotations_gripper_to_base,
        translations_gripper_to_base,
        rotations_target_to_camera,
        translations_target_to_camera,
        method=method_constant,
    )
    if not isinstance(calibration_result, (tuple, list)) or len(calibration_result) != 2:
        raise ValueError("cv2.calibrateHandEye returned an invalid rotation/translation result")
    rotation_base_camera = _as_float64_array(
        calibration_result[0],
        name="cv2.calibrateHandEye rotation for T_base_camera",
    )
    translation_base_camera = _as_float64_array(
        calibration_result[1],
        name="cv2.calibrateHandEye translation for T_base_camera",
    ).reshape(-1)
    if translation_base_camera.shape != (3,):
        raise ValueError(
            "cv2.calibrateHandEye translation for T_base_camera must contain exactly three values"
        )
    try:
        return make_transform(
            rotation_base_camera,
            translation_base_camera,
            name="T_base_camera",
        )
    except ValueError as exc:
        raise ValueError(f"cv2.calibrateHandEye returned invalid T_base_camera: {exc}") from exc


def mean_rigid_transform(transforms: Sequence[Any]) -> np.ndarray:
    """Average translations arithmetically and project the rotation mean onto SO(3)."""
    if isinstance(transforms, (str, bytes)) or not isinstance(transforms, Sequence) or not transforms:
        raise ValueError("transforms must be a nonempty sequence")
    matrices = [
        validate_homogeneous_transform(transform, name=f"transforms[{index}]")
        for index, transform in enumerate(transforms)
    ]
    mean_translation_m = np.mean([matrix[:3, 3] for matrix in matrices], axis=0)
    arithmetic_rotation_mean = np.mean([matrix[:3, :3] for matrix in matrices], axis=0)
    left_vectors, _, right_vectors_transposed = np.linalg.svd(arithmetic_rotation_mean)
    mean_rotation = left_vectors @ right_vectors_transposed
    if np.linalg.det(mean_rotation) < 0.0:
        left_vectors[:, -1] *= -1.0
        mean_rotation = left_vectors @ right_vectors_transposed
    return make_transform(mean_rotation, mean_translation_m, name="mean rigid transform")


def _validated_result_thresholds(validation_config: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(validation_config, Mapping):
        raise ValueError("validation_config must be a mapping")
    thresholds: dict[str, float] = {}
    for key in (
        "target_scatter_translation_m",
        "target_scatter_rotation_deg",
        "leave_one_out_translation_m",
        "leave_one_out_rotation_deg",
        "robust_mad_multiplier",
    ):
        thresholds[key] = _require_real(
            validation_config,
            key,
            context="validation_config",
            minimum=0.0,
        )
    return thresholds


def _scalar_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("summary values must be a nonempty finite one-dimensional sequence")
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "max": float(np.max(array)),
    }


def _robust_outlier_ids(
    sample_ids: Sequence[int],
    values: Sequence[float],
    *,
    mad_multiplier: float,
    numerical_floor: float,
) -> tuple[list[int], float]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    median_absolute_deviation = float(np.median(np.abs(array - median)))
    robust_sigma = max(1.4826 * median_absolute_deviation, numerical_floor)
    threshold = median + mad_multiplier * robust_sigma
    return [
        sample_id for sample_id, value in zip(sample_ids, array, strict=True) if value > threshold
    ], threshold


def _target_transforms(
    validated_samples: Sequence[tuple[int, np.ndarray, np.ndarray | None]],
    T_base_camera: np.ndarray,  # noqa: N803 - coordinate-frame notation is the public API.
) -> list[tuple[int, np.ndarray]]:
    targets: list[tuple[int, np.ndarray]] = []
    for sample_id, T_base_ee, T_camera_board in validated_samples:  # noqa: N806
        if T_camera_board is None:  # pragma: no cover - enforced by the validator contract
            raise AssertionError("T_camera_board validation was unexpectedly skipped")
        T_ee_board = (  # noqa: N806
            invert_transform(T_base_ee, name=f"T_base_ee sample {sample_id}") @ T_base_camera @ T_camera_board
        )
        targets.append(
            (
                sample_id,
                validate_homogeneous_transform(T_ee_board, name=f"T_ee_board sample {sample_id}"),
            )
        )
    return targets


def validate_eye_to_hand_result(
    samples: Sequence[Mapping[str, Any]],
    T_base_camera: Any,  # noqa: N803 - coordinate-frame notation is the public API.
    validation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Report rigid-target scatter for one eye-to-hand result, with units kept separate."""
    thresholds = _validated_result_thresholds(validation_config)
    validated_samples = _validated_eye_to_hand_samples(samples, minimum=3)
    resolved_base_camera = validate_homogeneous_transform(T_base_camera, name="T_base_camera")
    targets = _target_transforms(validated_samples, resolved_base_camera)
    mean_target = mean_rigid_transform([target for _, target in targets])

    sample_ids = [sample_id for sample_id, _ in targets]
    translations = np.asarray([target[:3, 3] for _, target in targets])
    translation_errors = [float(np.linalg.norm(target[:3, 3] - mean_target[:3, 3])) for _, target in targets]
    rotation_errors = [rotation_delta_deg(mean_target[:3, :3], target[:3, :3]) for _, target in targets]
    per_sample_errors = {
        str(sample_id): {
            "translation_m": translation_error,
            "rotation_deg": rotation_error,
        }
        for sample_id, translation_error, rotation_error in zip(
            sample_ids,
            translation_errors,
            rotation_errors,
            strict=True,
        )
    }

    statutory_outlier_ids = [
        sample_id
        for sample_id, translation_error, rotation_error in zip(
            sample_ids,
            translation_errors,
            rotation_errors,
            strict=True,
        )
        if translation_error > thresholds["target_scatter_translation_m"]
        or rotation_error > thresholds["target_scatter_rotation_deg"]
    ]
    robust_translation_ids, robust_translation_threshold_m = _robust_outlier_ids(
        sample_ids,
        translation_errors,
        mad_multiplier=thresholds["robust_mad_multiplier"],
        numerical_floor=1e-12,
    )
    robust_rotation_ids, robust_rotation_threshold_deg = _robust_outlier_ids(
        sample_ids,
        rotation_errors,
        mad_multiplier=thresholds["robust_mad_multiplier"],
        numerical_floor=1e-9,
    )
    robust_outlier_ids = sorted(set(robust_translation_ids) | set(robust_rotation_ids))
    outlier_ids = sorted(set(statutory_outlier_ids) | set(robust_outlier_ids))

    return {
        "sample_count": len(targets),
        "mean_T_ee_board": mean_target.tolist(),
        "mean_translation_m": mean_target[:3, 3].tolist(),
        "translation_component_std_m": np.std(translations, axis=0).tolist(),
        "translation_error_m": _scalar_summary(translation_errors),
        "mean_rotation_matrix": mean_target[:3, :3].tolist(),
        "rotation_geodesic_error_deg": _scalar_summary(rotation_errors),
        "per_sample_errors": per_sample_errors,
        "statutory_outlier_ids": sorted(statutory_outlier_ids),
        "robust_outlier_ids": robust_outlier_ids,
        "outlier_ids": outlier_ids,
        "thresholds": {
            "target_scatter_translation_m": thresholds["target_scatter_translation_m"],
            "target_scatter_rotation_deg": thresholds["target_scatter_rotation_deg"],
            "robust_mad_multiplier": thresholds["robust_mad_multiplier"],
            "robust_translation_threshold_m": robust_translation_threshold_m,
            "robust_rotation_threshold_deg": robust_rotation_threshold_deg,
        },
    }


def _leave_one_out_metric_values(
    per_omission: Mapping[str, Mapping[str, Any]],
    key: str,
) -> list[float]:
    return [float(omission[key]) for omission in per_omission.values()]


def leave_one_out_validation(
    samples: Sequence[Mapping[str, Any]],
    full_T_base_camera: Any,  # noqa: N803 - coordinate-frame notation is the public API.
    method: str | int,
    validation_config: Mapping[str, Any],
    cv2_module: Any = cv2,
) -> dict[str, Any]:
    """Re-solve each omission and report calibration and held-target stability."""
    method_name, method_constant = _resolved_hand_eye_method(method)
    thresholds = _validated_result_thresholds(validation_config)
    validated_samples = _validated_eye_to_hand_samples(samples, minimum=4)
    resolved_full_transform = validate_homogeneous_transform(
        full_T_base_camera,
        name="full_T_base_camera",
    )
    sample_ids = [sample_id for sample_id, _, _ in validated_samples]
    per_omission: dict[str, dict[str, Any]] = {}

    for omitted_index, (omitted_id, omitted_base_ee, omitted_camera_board) in enumerate(validated_samples):
        if omitted_camera_board is None:  # pragma: no cover - enforced by the validator contract
            raise AssertionError("T_camera_board validation was unexpectedly skipped")
        training_samples = [sample for index, sample in enumerate(samples) if index != omitted_index]
        loo_base_camera = calibrate_eye_to_hand(
            training_samples,
            method=method_constant,
            cv2_module=cv2_module,
        )
        stability_translation_m, stability_rotation_deg = pose_delta(
            resolved_full_transform,
            loo_base_camera,
        )

        training_validated = [
            sample for index, sample in enumerate(validated_samples) if index != omitted_index
        ]
        training_targets = _target_transforms(training_validated, loo_base_camera)
        mean_training_target = mean_rigid_transform([target for _, target in training_targets])
        held_out_target = (
            invert_transform(omitted_base_ee, name=f"T_base_ee sample {omitted_id}")
            @ loo_base_camera
            @ omitted_camera_board
        )
        held_out_target = validate_homogeneous_transform(
            held_out_target,
            name=f"held-out T_ee_board sample {omitted_id}",
        )
        held_translation_m, held_rotation_deg = pose_delta(mean_training_target, held_out_target)
        per_omission[str(omitted_id)] = {
            "sample_id": omitted_id,
            "T_base_camera": loo_base_camera.tolist(),
            "calibration_stability_translation_m": stability_translation_m,
            "calibration_stability_rotation_deg": stability_rotation_deg,
            "held_out_target_translation_m": held_translation_m,
            "held_out_target_rotation_deg": held_rotation_deg,
            "training_mean_T_ee_board": mean_training_target.tolist(),
        }

    metric_units = {
        "calibration_stability_translation_m": ("meter", 1e-12),
        "calibration_stability_rotation_deg": ("degree", 1e-9),
        "held_out_target_translation_m": ("meter", 1e-12),
        "held_out_target_rotation_deg": ("degree", 1e-9),
    }
    metric_values = {key: _leave_one_out_metric_values(per_omission, key) for key in metric_units}
    configured_influential_ids = [
        sample_id
        for sample_id, omission in zip(sample_ids, per_omission.values(), strict=True)
        if max(
            float(omission["calibration_stability_translation_m"]),
            float(omission["held_out_target_translation_m"]),
        )
        > thresholds["leave_one_out_translation_m"]
        or max(
            float(omission["calibration_stability_rotation_deg"]),
            float(omission["held_out_target_rotation_deg"]),
        )
        > thresholds["leave_one_out_rotation_deg"]
    ]

    robust_influential_ids: set[int] = set()
    robust_metric_thresholds: dict[str, float] = {}
    for key, (_, numerical_floor) in metric_units.items():
        metric_outlier_ids, robust_threshold = _robust_outlier_ids(
            sample_ids,
            metric_values[key],
            mad_multiplier=thresholds["robust_mad_multiplier"],
            numerical_floor=numerical_floor,
        )
        robust_influential_ids.update(metric_outlier_ids)
        robust_metric_thresholds[key] = robust_threshold

    sorted_robust_ids = sorted(robust_influential_ids)
    influential_ids = sorted(set(configured_influential_ids) | robust_influential_ids)
    return {
        "sample_count": len(validated_samples),
        "method": method_name,
        "per_omission": per_omission,
        **{key: _scalar_summary(values) for key, values in metric_values.items()},
        "configured_influential_ids": sorted(configured_influential_ids),
        "robust_influential_ids": sorted_robust_ids,
        "influential_ids": influential_ids,
        "thresholds": {
            "leave_one_out_translation_m": thresholds["leave_one_out_translation_m"],
            "leave_one_out_rotation_deg": thresholds["leave_one_out_rotation_deg"],
            "robust_mad_multiplier": thresholds["robust_mad_multiplier"],
            "robust_metric_thresholds": robust_metric_thresholds,
        },
    }


def _rotation_axis(rotation: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eig(rotation)
    axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
    norm = float(np.linalg.norm(axis))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("relative rotation has no usable real rotation axis")
    return axis / norm


def assess_pose_diversity(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Assess rotational excitation using first-pose-relative axes in one common frame."""
    validated_samples = _validated_eye_to_hand_samples(
        samples,
        minimum=0,
        require_camera_board=False,
    )
    translations = [T_base_ee[:3, 3] for _, T_base_ee, _ in validated_samples]  # noqa: N806
    translation_span_m = max(
        (
            float(np.linalg.norm(translation_b - translation_a))
            for index, translation_a in enumerate(translations)
            for translation_b in translations[index + 1 :]
        ),
        default=0.0,
    )

    usable_axes: list[dict[str, Any]] = []
    relative_rotation_angles: list[float] = []
    if validated_samples:
        reference_id, reference_pose, _ = validated_samples[0]
        for sample_id, pose, _ in validated_samples[1:]:
            relative_rotation = reference_pose[:3, :3].T @ pose[:3, :3]
            relative_angle_deg = rotation_delta_deg(np.eye(3), relative_rotation)
            relative_rotation_angles.append(relative_angle_deg)
            if relative_angle_deg >= _DIVERSITY_MIN_RELATIVE_ROTATION_DEG:
                usable_axes.append(
                    {
                        "reference_sample_id": reference_id,
                        "sample_id": sample_id,
                        "relative_rotation_deg": relative_angle_deg,
                        "axis": _rotation_axis(relative_rotation).tolist(),
                    }
                )

    axis_separations = [
        math.degrees(
            math.acos(
                float(
                    np.clip(
                        abs(np.dot(first["axis"], second["axis"])),
                        0.0,
                        1.0,
                    )
                )
            )
        )
        for index, first in enumerate(usable_axes)
        for second in usable_axes[index + 1 :]
    ]
    max_axis_separation_deg = max(axis_separations, default=0.0)
    relative_rotation_span_deg = max(relative_rotation_angles, default=0.0)
    is_diverse = (
        len(validated_samples) >= _DIVERSITY_MIN_POSE_COUNT
        and len(usable_axes) >= 2
        and relative_rotation_span_deg >= _DIVERSITY_MIN_RELATIVE_ROTATION_DEG
        and max_axis_separation_deg >= _DIVERSITY_MIN_NONPARALLEL_AXIS_SEPARATION_DEG
    )
    reasons: list[str] = []
    if len(validated_samples) < _DIVERSITY_MIN_POSE_COUNT:
        reasons.append("insufficient_pose_count")
    if relative_rotation_span_deg < _DIVERSITY_MIN_RELATIVE_ROTATION_DEG:
        reasons.append("insufficient_relative_rotation")
    if len(usable_axes) < 2:
        reasons.append("fewer_than_two_usable_rotation_axes")
    elif max_axis_separation_deg < _DIVERSITY_MIN_NONPARALLEL_AXIS_SEPARATION_DEG:
        reasons.append("rotation_axes_are_nearly_parallel")

    return {
        "pose_count": len(validated_samples),
        "translation_span_m": translation_span_m,
        "relative_rotation_span_deg": relative_rotation_span_deg,
        "usable_relative_rotation_axes": usable_axes,
        "usable_relative_rotation_axis_count": len(usable_axes),
        "max_nonparallel_axis_separation_deg": max_axis_separation_deg,
        "is_diverse": is_diverse,
        "reasons": reasons,
        "thresholds": {
            "min_pose_count": _DIVERSITY_MIN_POSE_COUNT,
            "min_relative_rotation_deg": _DIVERSITY_MIN_RELATIVE_ROTATION_DEG,
            "min_nonparallel_axis_separation_deg": (_DIVERSITY_MIN_NONPARALLEL_AXIS_SEPARATION_DEG),
        },
    }


def _recommendation_terms(
    validation: Mapping[str, Any],
    leave_one_out: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, dict[str, float | str]]:
    validation_translation_m = float(validation["translation_error_m"]["max"])
    validation_rotation_deg = float(validation["rotation_geodesic_error_deg"]["max"])
    leave_one_out_translation_m = max(
        float(leave_one_out["calibration_stability_translation_m"]["max"]),
        float(leave_one_out["held_out_target_translation_m"]["max"]),
    )
    leave_one_out_rotation_deg = max(
        float(leave_one_out["calibration_stability_rotation_deg"]["max"]),
        float(leave_one_out["held_out_target_rotation_deg"]["max"]),
    )
    raw_terms = {
        "validation_translation": (
            validation_translation_m,
            thresholds["target_scatter_translation_m"],
            "meter",
        ),
        "validation_rotation": (
            validation_rotation_deg,
            thresholds["target_scatter_rotation_deg"],
            "degree",
        ),
        "leave_one_out_translation": (
            leave_one_out_translation_m,
            thresholds["leave_one_out_translation_m"],
            "meter",
        ),
        "leave_one_out_rotation": (
            leave_one_out_rotation_deg,
            thresholds["leave_one_out_rotation_deg"],
            "degree",
        ),
    }
    return {
        name: {
            "value": value,
            "threshold": threshold,
            "normalized": value / threshold,
            "unit": unit,
        }
        for name, (value, threshold, unit) in raw_terms.items()
    }


def solve_all_methods(
    samples: Sequence[Mapping[str, Any]],
    validation_config: Mapping[str, Any],
    cv2_module: Any = cv2,
) -> dict[str, Any]:
    """Solve every supported method independently and recommend the lowest normalized score."""
    thresholds = _validated_result_thresholds(validation_config)
    _validated_eye_to_hand_samples(samples, minimum=4)
    method_reports: dict[str, dict[str, Any]] = {}

    for method_name, method_constant in HAND_EYE_METHODS.items():
        try:
            T_base_camera = calibrate_eye_to_hand(  # noqa: N806
                samples,
                method=method_constant,
                cv2_module=cv2_module,
            )
            validation = validate_eye_to_hand_result(samples, T_base_camera, thresholds)
            leave_one_out = leave_one_out_validation(
                samples,
                T_base_camera,
                method_constant,
                thresholds,
                cv2_module=cv2_module,
            )
            recommendation_terms = _recommendation_terms(validation, leave_one_out, thresholds)
            recommendation_score = float(
                np.mean([float(term["normalized"]) for term in recommendation_terms.values()])
            )
            method_reports[method_name] = {
                "method": method_constant,
                "status": "success",
                "T_base_camera": T_base_camera.tolist(),
                "validation": validation,
                "leave_one_out": leave_one_out,
                "recommendation_terms": recommendation_terms,
                "recommendation_score": recommendation_score,
            }
        except Exception as exc:
            method_reports[method_name] = {
                "method": method_constant,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    successful_methods = [
        method_name for method_name, report in method_reports.items() if report["status"] == "success"
    ]
    if not successful_methods:
        failure_summary = "; ".join(
            f"{method_name}: {method_reports[method_name]['error']}" for method_name in HAND_EYE_METHODS
        )
        raise RuntimeError(f"All hand-eye calibration methods failed: {failure_summary}")

    recommended_method = min(
        successful_methods,
        key=lambda method_name: float(method_reports[method_name]["recommendation_score"]),
    )
    recommended_transform = method_reports[recommended_method]["T_base_camera"]
    return {
        "methods": method_reports,
        "recommended_method": recommended_method,
        "recommended_T_base_camera": recommended_transform,
    }


_SAMPLE_REQUIRED_FIELDS = frozenset(
    {
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
)


@dataclass(frozen=True)
class PoseSimilarity:
    """Distance from a named query pose to one persisted robot pose."""

    sample_id: int
    translation_m: float
    rotation_deg: float


def _json_safe(value: Any, *, context: str) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), context=context)
    if isinstance(value, np.generic):
        return _json_safe(value.item(), context=context)
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, nested_value in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context} contains a non-string JSON object key")
            converted[key] = _json_safe(nested_value, context=f"{context}.{key}")
        return converted
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, context=context) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Real):
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"{context} contains a non-finite number")
        return numeric_value
    raise ValueError(f"{context} contains non-JSON value of type {type(value).__name__}")


def _sample_int(record: Mapping[str, Any], key: str, *, context: str, minimum: int = 0) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context}.{key} must be an integer >= {minimum}")
    return value


def _sample_real(record: Mapping[str, Any], key: str, *, context: str, minimum: float = 0.0) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context}.{key} must be a finite number >= {minimum}")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value < minimum:
        raise ValueError(f"{context}.{key} must be a finite number >= {minimum}")
    return numeric_value


def _sample_vector(record: Mapping[str, Any], key: str, *, context: str) -> np.ndarray:
    raw_value = record.get(key)
    raw_array = np.asarray(raw_value)
    if raw_array.dtype.kind not in "iuf" or raw_array.dtype.kind == "b":
        raise ValueError(f"{context}.{key} must contain three finite numbers")
    vector = np.asarray(raw_array, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{context}.{key} must contain three finite numbers")
    return vector


def _canonical_sample_paths(sample_id: int) -> tuple[str, str]:
    filename = f"sample_{sample_id:03d}.png"
    return f"images/{filename}", f"overlays/{filename}"


def _validate_sample_record(
    record: Mapping[str, Any],
    *,
    context: str,
    artifact_root: Path | None = None,
) -> None:
    missing_fields = sorted(_SAMPLE_REQUIRED_FIELDS - record.keys())
    if missing_fields:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing_fields)}")

    sample_id = _sample_int(record, "sample_id", context=context)
    expected_image_path, expected_overlay_path = _canonical_sample_paths(sample_id)
    for key, expected_path in (
        ("image_path", expected_image_path),
        ("overlay_path", expected_overlay_path),
    ):
        if record.get(key) != expected_path:
            raise ValueError(f"{context}.{key} must be exactly {expected_path!r}")

    _sample_real(record, "camera_timestamp_ms", context=context)
    robot_timestamp = record["robot_timestamp"]
    if robot_timestamp is not None and not isinstance(robot_timestamp, (bool, int, float, str)):
        raise ValueError(f"{context}.robot_timestamp must be a JSON scalar or null")
    if isinstance(robot_timestamp, float) and not math.isfinite(robot_timestamp):
        raise ValueError(f"{context}.robot_timestamp must be finite")

    _sample_int(record, "image_width", context=context, minimum=1)
    _sample_int(record, "image_height", context=context, minimum=1)
    corner_count = _sample_int(record, "num_charuco_corners", context=context)

    charuco_ids = record["charuco_ids"]
    if not isinstance(charuco_ids, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in charuco_ids
    ):
        raise ValueError(f"{context}.charuco_ids must be a list of nonnegative integers")
    if len(set(charuco_ids)) != len(charuco_ids):
        raise ValueError(f"{context}.charuco_ids must be unique")

    raw_corners = record["charuco_corners_px"]
    try:
        corners_array = np.asarray(raw_corners)
    except ValueError as exc:
        raise ValueError(f"{context}.charuco_corners_px must contain finite [x, y] pairs") from exc
    if corner_count == 0 and corners_array.shape == (0,):
        corners_array = np.empty((0, 2), dtype=np.float64)
    elif corners_array.dtype.kind not in "iuf" or corners_array.dtype.kind == "b":
        raise ValueError(f"{context}.charuco_corners_px must contain finite [x, y] pairs")
    else:
        corners_array = np.asarray(corners_array, dtype=np.float64)
    if corners_array.shape != (corner_count, 2) or not np.isfinite(corners_array).all():
        raise ValueError(
            f"{context}.charuco_corners_px must contain exactly {corner_count} finite [x, y] pairs"
        )
    if len(charuco_ids) != corner_count:
        raise ValueError(f"{context}.num_charuco_corners must equal the charuco_ids length")

    _sample_vector(record, "rvec_camera_board", context=context)
    _sample_vector(record, "tvec_camera_board_m", context=context)
    for transform_name in ("T_camera_board", "T_base_ee"):
        try:
            validate_homogeneous_transform(record[transform_name], name=transform_name)
        except ValueError as exc:
            raise ValueError(f"{context}.{transform_name} is invalid: {exc}") from exc

    _sample_real(record, "reprojection_error_px", context=context)
    for key in ("opencv_version", "realsense_serial"):
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError(f"{context}.{key} must be a nonempty string")

    required_metadata = {
        "robot_pose_name": "T_base_ee",
        "translation_unit": "meter",
        "matrix_storage_source": "existing_franka_client",
    }
    for key, expected_value in required_metadata.items():
        if record[key] != expected_value:
            raise ValueError(f"{context}.{key} must be exactly {expected_value!r}")

    if artifact_root is not None:
        for key in ("image_path", "overlay_path"):
            if not (artifact_root / str(record[key])).is_file():
                raise ValueError(f"{context}.{key} does not reference an existing file")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def load_samples(input_dir: str | Path) -> list[dict[str, Any]]:
    """Strictly load, validate, and order persisted hand-eye sample records."""
    root = Path(input_dir)
    manifest_path = root / "samples.jsonl"
    if not manifest_path.exists():
        return []

    samples: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    with manifest_path.open("r", encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            context = f"samples.jsonl line {line_number}"
            try:
                record = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_json_keys,
                    parse_constant=_reject_nonfinite_json_constant,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{context} contains malformed JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{context} must contain one JSON object")
            _validate_sample_record(record, context=context, artifact_root=root)
            sample_id = int(record["sample_id"])
            if sample_id in seen_ids:
                raise ValueError(f"{context} contains duplicate sample_id {sample_id}")
            seen_ids.add(sample_id)
            samples.append(record)
    return sorted(samples, key=lambda sample: int(sample["sample_id"]))


def _canonical_artifact_id(path: Path) -> int | None:
    name = path.name
    if not name.startswith("sample_") or not name.endswith(".png"):
        return None
    raw_id = name[len("sample_") : -len(".png")]
    if not raw_id.isdigit():
        return None
    sample_id = int(raw_id)
    return sample_id if name == f"sample_{sample_id:03d}.png" else None


def next_sample_id(input_dir: str | Path, samples: Sequence[Mapping[str, Any]]) -> int:
    """Return an ID above all manifest and canonical orphan artifact IDs."""
    reserved_ids: set[int] = set()
    for index, sample in enumerate(samples):
        reserved_ids.add(_sample_int(sample, "sample_id", context=f"samples[{index}]"))
    root = Path(input_dir)
    for directory_name in ("images", "overlays"):
        directory = root / directory_name
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            artifact_id = _canonical_artifact_id(path)
            if artifact_id is not None:
                reserved_ids.add(artifact_id)
    return max(reserved_ids, default=-1) + 1


def _encode_rgb_png(image: Any, *, name: str, width: int, height: int) -> bytes:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.shape != (height, width, 3):
        raise ValueError(f"{name} must be a uint8 RGB image with shape ({height}, {width}, 3)")
    bgr_image = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    encoded, png_buffer = cv2.imencode(".png", bgr_image)
    if not encoded:
        raise OSError(f"Failed to encode {name} as PNG")
    return png_buffer.tobytes()


def _write_temporary_bytes(directory: Path, *, prefix: str, payload: bytes) -> Path:
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{prefix}.", suffix=".tmp", dir=directory)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _manifest_bytes(samples: Sequence[Mapping[str, Any]]) -> bytes:
    lines = [
        json.dumps(sample, ensure_ascii=False, allow_nan=False, separators=(",", ":")) for sample in samples
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode()


def _atomic_write_manifest(output_dir: Path, samples: Sequence[Mapping[str, Any]]) -> None:
    temporary_path = _write_temporary_bytes(
        output_dir,
        prefix="samples.jsonl",
        payload=_manifest_bytes(samples),
    )
    try:
        os.replace(temporary_path, output_dir / "samples.jsonl")
    finally:
        temporary_path.unlink(missing_ok=True)


def nearest_pose_delta(
    T_base_ee: Any,  # noqa: N803 - coordinate-frame notation is the public API.
    samples: Sequence[Mapping[str, Any]],
) -> PoseSimilarity | None:
    """Return the persisted robot pose nearest in translation, then rotation."""
    query_pose = validate_homogeneous_transform(T_base_ee, name="T_base_ee_query")
    nearest: PoseSimilarity | None = None
    for index, sample in enumerate(samples):
        sample_id = _sample_int(sample, "sample_id", context=f"samples[{index}]")
        if "T_base_ee" not in sample:
            raise ValueError(f"samples[{index}].T_base_ee is required")
        sample_pose = validate_homogeneous_transform(
            sample["T_base_ee"],
            name=f"T_base_ee_sample_{sample_id}",
        )
        translation_m, rotation_deg = pose_delta(query_pose, sample_pose)
        candidate = PoseSimilarity(sample_id, translation_m, rotation_deg)
        if nearest is None or (
            candidate.translation_m,
            candidate.rotation_deg,
            candidate.sample_id,
        ) < (nearest.translation_m, nearest.rotation_deg, nearest.sample_id):
            nearest = candidate
    return nearest


class HandEyeSampleStore:
    """Crash-safe owner of hand-eye JSONL records and paired RGB PNG files."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "overlays").mkdir(exist_ok=True)
        self._samples = load_samples(self.output_dir)

    @property
    def samples(self) -> list[dict[str, Any]]:
        return list(self._samples)

    @property
    def next_sample_id(self) -> int:
        return next_sample_id(self.output_dir, self._samples)

    def _refresh(self) -> None:
        self._samples = load_samples(self.output_dir)

    def save(
        self,
        record: Mapping[str, Any],
        rgb_image: np.ndarray,
        rgb_overlay: np.ndarray,
    ) -> dict[str, Any]:
        """Atomically make a validated sample bundle visible in the manifest."""
        self._refresh()
        sample = _json_safe(record, context="sample")
        if not isinstance(sample, dict):
            raise ValueError("sample must be a mapping")

        assigned_id = self.next_sample_id
        supplied_id = sample.get("sample_id", assigned_id)
        if isinstance(supplied_id, bool) or not isinstance(supplied_id, int):
            raise ValueError("sample.sample_id must be a nonnegative integer")
        if supplied_id != assigned_id:
            raise ValueError(f"sample.sample_id must equal the next sample ID {assigned_id}")
        sample["sample_id"] = assigned_id

        image_path, overlay_path = _canonical_sample_paths(assigned_id)
        for key, expected_path in (("image_path", image_path), ("overlay_path", overlay_path)):
            supplied_path = sample.get(key, expected_path)
            if supplied_path != expected_path:
                raise ValueError(f"sample.{key} must be exactly {expected_path!r}")
            sample[key] = expected_path
        _validate_sample_record(sample, context="sample")

        width = int(sample["image_width"])
        height = int(sample["image_height"])
        image_png = _encode_rgb_png(rgb_image, name="rgb_image", width=width, height=height)
        overlay_png = _encode_rgb_png(rgb_overlay, name="rgb_overlay", width=width, height=height)

        final_paths = (self.output_dir / image_path, self.output_dir / overlay_path)
        for final_path in final_paths:
            if final_path.exists():
                raise FileExistsError(f"Refusing to overwrite existing sample artifact {final_path}")

        temporary_paths: list[Path] = []
        installed_paths: list[Path] = []
        try:
            temporary_paths.append(
                _write_temporary_bytes(final_paths[0].parent, prefix=final_paths[0].name, payload=image_png)
            )
            temporary_paths.append(
                _write_temporary_bytes(final_paths[1].parent, prefix=final_paths[1].name, payload=overlay_png)
            )
            for temporary_path, final_path in zip(temporary_paths, final_paths, strict=True):
                if final_path.exists():
                    raise FileExistsError(f"Refusing to overwrite existing sample artifact {final_path}")
                os.replace(temporary_path, final_path)
                installed_paths.append(final_path)
            new_samples = sorted([*self._samples, sample], key=lambda item: int(item["sample_id"]))
            _atomic_write_manifest(self.output_dir, new_samples)
        except BaseException:
            for installed_path in reversed(installed_paths):
                installed_path.unlink(missing_ok=True)
            raise
        finally:
            for temporary_path in temporary_paths:
                temporary_path.unlink(missing_ok=True)

        self._samples = new_samples
        return dict(sample)

    def delete_last(self) -> dict[str, Any] | None:
        """Remove the highest-ID manifest sample, committing the manifest first."""
        self._refresh()
        if not self._samples:
            return None

        deleted = self._samples[-1]
        remaining = self._samples[:-1]
        _atomic_write_manifest(self.output_dir, remaining)
        self._samples = remaining

        first_error: OSError | None = None
        for key in ("image_path", "overlay_path"):
            try:
                (self.output_dir / str(deleted[key])).unlink(missing_ok=True)
            except OSError as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        return dict(deleted)

    def nearest_pose_delta(
        self,
        T_base_ee: Any,  # noqa: N803 - coordinate-frame notation is the public API.
    ) -> PoseSimilarity | None:
        return nearest_pose_delta(T_base_ee, self._samples)


def save_sample_bundle(
    output_dir: str | Path,
    sample: Mapping[str, Any],
    rgb_image: np.ndarray,
    rgb_overlay: np.ndarray,
) -> dict[str, Any]:
    """Save one sample through a short functional compatibility API."""
    return HandEyeSampleStore(output_dir).save(sample, rgb_image, rgb_overlay)


def delete_last_sample(output_dir: str | Path) -> dict[str, Any] | None:
    """Delete the highest-ID sample through a short functional compatibility API."""
    return HandEyeSampleStore(output_dir).delete_last()


@dataclass(frozen=True)
class RobotPoseReading:
    """One validated read of the Franka end-effector pose."""

    T_base_ee: np.ndarray  # noqa: N815 - coordinate-frame notation is the public API.
    robot_pose_raw: Any
    robot_timestamp: bool | int | float | str | None
    local_monotonic_s: float
    request_latency_ms: float
    robot_pose_name: str = "T_base_ee"
    translation_unit: str = "meter"
    matrix_storage_source: str = "existing_franka_client"
    matrix_storage_format: str = "nested_4x4"


def _explicit_robot_timestamp(state: Mapping[str, Any]) -> bool | int | float | str | None:
    if "robot_timestamp" not in state:
        return None

    value = state["robot_timestamp"]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def parse_franka_state_pose(
    state: Mapping[str, Any],
    *,
    local_monotonic_s: float | None = None,
    request_latency_ms: float = 0.0,
) -> RobotPoseReading:
    """Parse the passive Franka state pose without guessing its matrix layout."""
    if not isinstance(state, Mapping):
        raise ValueError("Franka state must be a mapping")

    if "ee" in state:
        robot_pose_raw = state["ee"]
        T_base_ee = validate_homogeneous_transform(robot_pose_raw, name="T_base_ee")  # noqa: N806
        matrix_storage_format = "nested_4x4"
    elif "O_T_EE" in state:
        robot_pose_raw = state["O_T_EE"]
        flat_pose = _as_float64_array(robot_pose_raw, name="O_T_EE")
        if flat_pose.shape != (16,):
            raise ValueError(f"O_T_EE pose must contain 16 flat values, got shape {flat_pose.shape}")
        T_base_ee = validate_homogeneous_transform(  # noqa: N806
            flat_pose.reshape((4, 4), order="F"),
            name="T_base_ee",
        )
        matrix_storage_format = "flat_16_column_major"
    else:
        raise ValueError("Franka state is missing pose field 'ee' or flat 'O_T_EE'")

    local_timestamp = time.monotonic() if local_monotonic_s is None else float(local_monotonic_s)
    return RobotPoseReading(
        T_base_ee=T_base_ee,
        robot_pose_raw=robot_pose_raw,
        robot_timestamp=_explicit_robot_timestamp(state),
        local_monotonic_s=local_timestamp,
        request_latency_ms=float(request_latency_ms),
        matrix_storage_format=matrix_storage_format,
    )


class FrankaPoseReader:
    """Read poses through the bare Franka client's read-only endpoint."""

    def __init__(
        self,
        client: Any,
        *,
        timeout_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._timeout_s = float(timeout_s)
        self._clock = clock

    def read(self) -> RobotPoseReading:
        request_started_s = self._clock()
        state = self._client.get_curr(timeout=self._timeout_s)
        request_finished_s = self._clock()
        return parse_franka_state_pose(
            state,
            local_monotonic_s=(request_started_s + request_finished_s) / 2.0,
            request_latency_ms=(request_finished_s - request_started_s) * 1000.0,
        )

    def close(self) -> None:
        self._client.close()


@dataclass(frozen=True)
class StillnessStatus:
    """Current motion bounds across the retained stillness window."""

    is_still: bool
    history_span_s: float
    max_translation_m: float
    max_rotation_deg: float
    reason: str


class RobotStillnessMonitor:
    """Track whether every robot pose in a recent time window is stationary."""

    def __init__(self, window_s: float, max_translation_m: float, max_rotation_deg: float) -> None:
        self.window_s = float(window_s)
        self.max_translation_m = float(max_translation_m)
        self.max_rotation_deg = float(max_rotation_deg)
        self._history: deque[tuple[float, np.ndarray]] = deque()

    def add(
        self,
        timestamp_s: float,
        T_base_ee: Any,  # noqa: N803 - coordinate-frame notation is the public API.
    ) -> None:
        timestamp = float(timestamp_s)
        pose = validate_homogeneous_transform(T_base_ee, name="T_base_ee")
        if self._history and timestamp < self._history[-1][0]:
            raise ValueError("Robot stillness timestamps must be nondecreasing")

        self._history.append((timestamp, pose))
        cutoff_s = timestamp - self.window_s
        while len(self._history) >= 2 and self._history[1][0] <= cutoff_s:
            self._history.popleft()

    def status(self) -> StillnessStatus:
        history_span_s = self._history[-1][0] - self._history[0][0] if len(self._history) >= 2 else 0.0
        max_translation_m = 0.0
        max_rotation_deg = 0.0
        history = list(self._history)
        for index, (_, pose_a) in enumerate(history):
            for _, pose_b in history[index + 1 :]:
                translation_m, rotation_deg = pose_delta(pose_a, pose_b)
                max_translation_m = max(max_translation_m, translation_m)
                max_rotation_deg = max(max_rotation_deg, rotation_deg)

        if history_span_s < self.window_s:
            reason = "insufficient_history"
            is_still = False
        elif max_translation_m > self.max_translation_m:
            reason = "translation_motion_exceeded"
            is_still = False
        elif max_rotation_deg > self.max_rotation_deg:
            reason = "rotation_motion_exceeded"
            is_still = False
        else:
            reason = "still"
            is_still = True

        return StillnessStatus(
            is_still=is_still,
            history_span_s=history_span_s,
            max_translation_m=max_translation_m,
            max_rotation_deg=max_rotation_deg,
            reason=reason,
        )
