#!/usr/bin/env python3
"""Passively collect RealSense color/Franka pose bundles for Eye-to-Hand calibration.

The module deliberately imports only the Python standard library at import time.
Runtime vision, SDK, robot, and hand-eye imports happen after argument parsing so
``--help`` is safe on machines without the hardware stack. The only project
import at module load is the standard-library-only shared control-host config.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware_test.franka.defaults import get_control_host  # noqa: E402

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "l515_eye_to_hand.yaml"
DEFAULT_OUTPUT_DIR = Path("outputs/handeye/l515_eye_to_hand")
WINDOW_NAME = "Franka L515 Eye-to-Hand collector"
WARMUP_VALID_COLOR_FRAMES = 30
ROBOT_READ_TIMEOUT_S = 2.0
COMMAND_DURATION_MS = 300
LEGACY_PATTERN_WARNING = (
    "OpenCV 4.6 changed the ChArUco board coordinate convention; legacy_pattern must match "
    "the physical board used for every capture."
)


@dataclass(frozen=True)
class RuntimeDependencies:
    """Runtime-only modules loaded after CLI parsing."""

    cv2: Any
    numpy: Any
    realsense: Any
    yaml: Any


@dataclass(frozen=True)
class ColorCameraInfo:
    """Intrinsics taken from the active RealSense color video profile."""

    width: int
    height: int
    fps: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    distortion_model: str
    distortion_coefficients: tuple[float, float, float, float, float]
    serial: str
    camera_matrix: Any


@dataclass(frozen=True)
class ColorCamera:
    """Started color-only pipeline plus its authoritative active profile."""

    pipeline: Any
    active_profile: Any
    info: ColorCameraInfo


@dataclass(frozen=True)
class ColorFrame:
    """One RGB8 color frame and its SDK timestamp."""

    rgb: Any
    timestamp_ms: float


@dataclass(frozen=True)
class CaptureEligibility:
    """Pure decision result for one proposed sample."""

    eligible: bool
    status: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    can_force_similarity: bool


@dataclass(frozen=True)
class PassiveRobotAccess:
    """The existing bare HTTP client paired with its read-only pose adapter."""

    client: Any
    pose_reader: Any

    def close(self) -> None:
        """Close client resources without invoking any robot endpoint."""
        self.client.close()


@dataclass(frozen=True)
class RobotPollResult:
    """A state read plus the monitor that remains valid after that read."""

    reading: Any | None
    stillness: Any | None
    monitor: Any
    error: str | None


@dataclass(frozen=True)
class CaptureAttempt:
    """Result of freezing a frame, reading a fresh pose, and optionally saving."""

    eligibility: CaptureEligibility
    monitor: Any
    saved_record: dict[str, Any] | None
    message: str


@dataclass(frozen=True)
class SavedPoseDelta:
    """Distance from the current pose to one persisted sample."""

    sample_id: int
    translation_m: float
    rotation_deg: float
    normalized_max_distance: float


@dataclass(frozen=True)
class SavedPoseComparison:
    """Nearest, jointly similar, and most recently saved pose comparisons."""

    nearest: SavedPoseDelta | None
    similar: SavedPoseDelta | None
    previous: SavedPoseDelta | None


def positive_int(value: str) -> int:
    """Argparse type accepting integers strictly greater than zero."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the hardware-lazy collector command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Collect passive L515 RGB + Franka T_base_ee samples for fixed-camera Eye-to-Hand calibration."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--width", type=positive_int, default=960)
    parser.add_argument("--height", type=positive_int, default=540)
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--camera-serial", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=positive_int, default=20)
    parser.add_argument("--control-host", default=get_control_host())
    return parser


def _required_import(
    import_module: Callable[[str], Any],
    module_name: str,
    missing_message: str,
) -> Any:
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise RuntimeError(missing_message) from exc
        raise RuntimeError(
            f"{module_name} is installed but a dependency failed to import ({exc}). "
            "Repair that environment before collecting; this program never auto-installs packages."
        ) from exc
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"{module_name} is installed but could not initialize ({exc}). "
            "Repair its native runtime before collecting; this program never auto-installs packages."
        ) from exc


def check_runtime_dependencies(
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
) -> RuntimeDependencies:
    """Load and audit collector dependencies without opening a device or GUI window."""
    cv2 = _required_import(
        import_module,
        "cv2",
        "OpenCV (cv2) is required; install a build containing contrib/aruco. "
        "This collector never auto-installs packages.",
    )
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise RuntimeError(
            "cv2.aruco is unavailable; an OpenCV build containing contrib/aruco is needed. "
            "This collector will never auto-install packages."
        )

    missing_geometry_apis = [
        name for name in ("solvePnP", "projectPoints") if not callable(getattr(cv2, name, None))
    ]
    if missing_geometry_apis:
        raise RuntimeError(
            "OpenCV collection requires callable "
            + ", ".join(missing_geometry_apis)
            + "; install a compatible OpenCV contrib build."
        )

    detector_factory = getattr(aruco, "CharucoDetector", None)
    has_modern_detector = callable(detector_factory) and callable(
        getattr(detector_factory, "detectBoard", None)
    )
    has_legacy_detector = callable(getattr(aruco, "detectMarkers", None)) and callable(
        getattr(aruco, "interpolateCornersCharuco", None)
    )
    if not has_modern_detector and not has_legacy_detector:
        raise RuntimeError(
            "OpenCV ChArUco support requires modern CharucoDetector.detectBoard or the complete "
            "legacy detectMarkers + interpolateCornersCharuco pair."
        )

    numpy = _required_import(
        import_module,
        "numpy",
        "numpy is required for color frames and calibration matrices; install it in this environment.",
    )
    yaml = _required_import(
        import_module,
        "yaml",
        "yaml (PyYAML) is required to load and persist the calibration configuration.",
    )
    realsense = _required_import(
        import_module,
        "pyrealsense2",
        "pyrealsense2 is required; install the matching Intel RealSense SDK Python bindings.",
    )

    pipeline_factory = getattr(realsense, "pipeline", None)
    if not callable(pipeline_factory):
        raise RuntimeError(
            "pyrealsense2 is installed but does not expose rs.pipeline; repair the RealSense SDK install."
        )
    try:
        pipeline_factory()
    except Exception as exc:
        raise RuntimeError(
            "pyrealsense2 is installed but failed to initialize. This is an SDK/device/udev "
            f"or permissions problem, not a missing Python package: {exc}"
        ) from exc

    return RuntimeDependencies(cv2=cv2, numpy=numpy, realsense=realsense, yaml=yaml)


def _distortion_model_name(model: Any) -> str:
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
            return normalized[len(prefix) :]
    return normalized


def _extract_active_color_info(rs: Any, active_profile: Any) -> ColorCameraInfo:
    numpy = importlib.import_module("numpy")
    try:
        video_profile = active_profile.get_stream(rs.stream.color).as_video_stream_profile()
        width = int(video_profile.width())
        height = int(video_profile.height())
        fps = int(video_profile.fps())
        intrinsics = video_profile.get_intrinsics()
        intrinsic_width = int(getattr(intrinsics, "width", width))
        intrinsic_height = int(getattr(intrinsics, "height", height))
        if (intrinsic_width, intrinsic_height) != (width, height):
            raise RuntimeError(
                "active color intrinsics resolution does not match the active video profile: "
                f"{intrinsic_width}x{intrinsic_height} versus {width}x{height}"
            )

        coefficients = tuple(float(value) for value in intrinsics.coeffs)
        if len(coefficients) != 5 or not all(math.isfinite(value) for value in coefficients):
            raise RuntimeError("active RealSense color profile must provide five distortion coefficients")

        fx = float(intrinsics.fx)
        fy = float(intrinsics.fy)
        ppx = float(intrinsics.ppx)
        ppy = float(intrinsics.ppy)
        if not all(math.isfinite(value) for value in (fx, fy, ppx, ppy)) or fx <= 0.0 or fy <= 0.0:
            raise RuntimeError("active RealSense color profile returned invalid focal intrinsics")

        serial = str(active_profile.get_device().get_info(rs.camera_info.serial_number))
        if not serial:
            raise RuntimeError("active RealSense device returned an empty serial number")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"failed to read the active RealSense color profile: {exc}") from exc

    camera_matrix = numpy.array(
        [[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]],
        dtype=numpy.float64,
    )
    return ColorCameraInfo(
        width=width,
        height=height,
        fps=fps,
        fx=fx,
        fy=fy,
        ppx=ppx,
        ppy=ppy,
        distortion_model=_distortion_model_name(intrinsics.model),
        distortion_coefficients=coefficients,
        serial=serial,
        camera_matrix=camera_matrix,
    )


def _discard_valid_color_frames(pipeline: Any, *, count: int) -> None:
    discarded = 0
    consecutive_missing = 0
    while discarded < count:
        try:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
        except Exception as exc:
            raise RuntimeError(f"RealSense camera disconnected during color warmup: {exc}") from exc
        if not color_frame:
            consecutive_missing += 1
            if consecutive_missing >= 60:
                raise RuntimeError("RealSense color stream returned no valid frame during warmup")
            continue
        consecutive_missing = 0
        discarded += 1


def start_color_camera(
    rs: Any,
    *,
    width: int,
    height: int,
    fps: int,
    camera_serial: str | None = None,
) -> ColorCamera:
    """Start exactly one RGB8 color stream and record its active profile."""
    try:
        pipeline = rs.pipeline()
        rs_config = rs.config()
    except Exception as exc:
        raise RuntimeError(
            "RealSense SDK initialization failed; check the SDK install, device connection, and udev "
            f"permissions: {exc}"
        ) from exc

    if camera_serial:
        rs_config.enable_device(camera_serial)
    rs_config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

    try:
        active_profile = pipeline.start(rs_config)
    except Exception as exc:
        raise RuntimeError(
            "Unable to start the RealSense RGB8 color stream. Check that the requested L515/serial "
            f"is connected and accessible through SDK/udev permissions: {exc}"
        ) from exc

    try:
        _discard_valid_color_frames(pipeline, count=WARMUP_VALID_COLOR_FRAMES)
        info = _extract_active_color_info(rs, active_profile)
    except Exception:
        with suppress(Exception):
            pipeline.stop()
        raise
    return ColorCamera(pipeline=pipeline, active_profile=active_profile, info=info)


def read_color_frame(camera: ColorCamera) -> ColorFrame:
    """Read one RGB8 frame and reject any mismatch with active-profile intrinsics."""
    numpy = importlib.import_module("numpy")
    try:
        frames = camera.pipeline.wait_for_frames()
        sdk_frame = frames.get_color_frame()
    except Exception as exc:
        raise RuntimeError(f"RealSense camera disconnected while waiting for a color frame: {exc}") from exc
    if not sdk_frame:
        raise RuntimeError("RealSense camera returned no color frame; the device may be disconnected")

    try:
        rgb = numpy.asanyarray(sdk_frame.get_data())
        timestamp_ms = float(sdk_frame.get_timestamp())
    except Exception as exc:
        raise RuntimeError(f"Unable to decode the RealSense RGB8 color frame: {exc}") from exc

    expected_shape = (camera.info.height, camera.info.width, 3)
    if rgb.shape != expected_shape:
        actual_height = int(rgb.shape[0]) if rgb.ndim >= 1 else 0
        actual_width = int(rgb.shape[1]) if rgb.ndim >= 2 else 0
        raise RuntimeError(
            f"RealSense frame is {actual_width}x{actual_height}, but active profile intrinsics are "
            f"{camera.info.width}x{camera.info.height}; frames are not resized and mismatched "
            "intrinsics are never reused"
        )
    if rgb.dtype != numpy.uint8:
        raise RuntimeError(f"RealSense RGB8 frame has unexpected dtype {rgb.dtype}")
    if not math.isfinite(timestamp_ms):
        raise RuntimeError("RealSense color frame returned a non-finite SDK timestamp")
    return ColorFrame(rgb=rgb, timestamp_ms=timestamp_ms)


def _freeze_color_frame(frame: ColorFrame) -> ColorFrame:
    """Detach an acquired frame from the camera SDK's reusable image buffer."""
    numpy = importlib.import_module("numpy")
    return ColorFrame(
        rgb=numpy.array(frame.rgb, copy=True),
        timestamp_ms=float(frame.timestamp_ms),
    )


def evaluate_capture_eligibility(
    *,
    detection_success: bool,
    num_charuco_corners: int,
    min_charuco_corners: int,
    pnp_success: bool,
    reprojection_error_px: float | None,
    warning_reprojection_error_px: float,
    max_reprojection_error_px: float,
    laplacian_score: float | None,
    min_laplacian_variance: float,
    robot_pose_fresh: bool,
    robot_still: bool,
    stillness_history_s: float,
    required_stillness_s: float = 1.0,
    similar_pose: bool = False,
    force_similar: bool = False,
) -> CaptureEligibility:
    """Apply every capture gate without hardware, filesystem, or mutable state."""
    reasons: list[str] = []
    warnings: list[str] = []

    if not detection_success:
        reasons.append("detection_failed")
    if num_charuco_corners < min_charuco_corners:
        reasons.append("insufficient_charuco_corners")
    if not pnp_success:
        reasons.append("pnp_failed")

    if reprojection_error_px is None or not math.isfinite(float(reprojection_error_px)):
        reasons.append("reprojection_unavailable")
    else:
        reprojection = float(reprojection_error_px)
        if reprojection > max_reprojection_error_px:
            reasons.append("reprojection_error_exceeded")
        elif reprojection > warning_reprojection_error_px:
            warnings.append("reprojection_warning")

    if laplacian_score is None or not math.isfinite(float(laplacian_score)):
        reasons.append("blur_score_unavailable")
    elif float(laplacian_score) < min_laplacian_variance:
        reasons.append("image_too_blurry")

    if not robot_pose_fresh:
        reasons.append("robot_pose_not_fresh")
    else:
        if not robot_still:
            reasons.append("robot_not_still")
        if stillness_history_s < required_stillness_s:
            reasons.append("stillness_history_too_short")

    can_force_similarity = False
    if not reasons and similar_pose:
        if force_similar:
            warnings.append("similar_pose_forced")
        else:
            reasons.append("similar_pose_confirmation_required")
            warnings.append("similar_pose")
            can_force_similarity = True

    if reasons and not can_force_similarity:
        status = "red"
    elif warnings or can_force_similarity:
        status = "yellow"
    else:
        status = "green"
    return CaptureEligibility(
        eligible=not reasons,
        status=status,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        can_force_similarity=can_force_similarity,
    )


def _reject_on_vision_error(
    eligibility: CaptureEligibility,
    vision_error: str | None,
) -> CaptureEligibility:
    """Make any incomplete vision product, including its overlay, non-persistable."""
    if vision_error is None:
        return eligibility
    reasons = eligibility.reasons
    if "vision_processing_failed" not in reasons:
        reasons = (*reasons, "vision_processing_failed")
    return CaptureEligibility(
        eligible=False,
        status="red",
        reasons=reasons,
        warnings=eligibility.warnings,
        can_force_similarity=False,
    )


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)


def build_robot_access(
    control_host: str,
    *,
    client_factory: Callable[..., Any] | None = None,
    pose_reader_factory: Callable[..., Any] | None = None,
    timeout_s: float = ROBOT_READ_TIMEOUT_S,
) -> PassiveRobotAccess:
    """Construct the existing bare client and read-only pose reader only."""
    numeric_timeout = float(timeout_s)
    if not math.isfinite(numeric_timeout) or numeric_timeout <= 0.0:
        raise ValueError("robot timeout_s must be a finite positive value")

    if client_factory is None or pose_reader_factory is None:
        _ensure_repo_root_on_path()
    if client_factory is None:
        robot_module = importlib.import_module("hardware_test.franka.franka_robot")
        client_factory = robot_module.FrankaControlClient
    if pose_reader_factory is None:
        utilities = importlib.import_module("hardware_test.franka.handeye.handeye_utils")
        pose_reader_factory = utilities.FrankaPoseReader

    client = client_factory(
        base_url=None,
        control_host=control_host,
        velocity_transport="http",
        zmq_url=None,
        timeout_s=numeric_timeout,
        command_duration_ms=COMMAND_DURATION_MS,
    )
    try:
        pose_reader = pose_reader_factory(client, timeout_s=numeric_timeout)
    except Exception:
        client.close()
        raise
    return PassiveRobotAccess(client=client, pose_reader=pose_reader)


def poll_robot_pose(
    pose_reader: Any,
    stillness_monitor: Any,
    *,
    monitor_factory: Callable[[], Any],
) -> RobotPollResult:
    """Read one pose, or replace the monitor so failed reads cannot leave it valid."""
    try:
        reading = pose_reader.read()
        stillness_monitor.add(reading.local_monotonic_s, reading.T_base_ee)
        stillness = stillness_monitor.status()
    except Exception as exc:
        replacement = monitor_factory()
        return RobotPollResult(
            reading=None,
            stillness=None,
            monitor=replacement,
            error=f"Robot state read failed; stillness history reset: {exc}",
        )
    return RobotPollResult(
        reading=reading,
        stillness=stillness,
        monitor=stillness_monitor,
        error=None,
    )


def compare_saved_poses(
    T_base_ee: Any,  # noqa: N803 - coordinate-frame notation is the public API.
    samples: Sequence[Mapping[str, Any]],
    *,
    pose_delta_fn: Callable[[Any, Any], tuple[float, float]],
    translation_threshold_m: float,
    rotation_threshold_deg: float,
) -> SavedPoseComparison:
    """Scan every saved pose so a joint threshold match cannot be hidden."""
    translation_threshold = float(translation_threshold_m)
    rotation_threshold = float(rotation_threshold_deg)
    if (
        not math.isfinite(translation_threshold)
        or translation_threshold <= 0.0
        or not math.isfinite(rotation_threshold)
        or rotation_threshold <= 0.0
    ):
        raise ValueError("pose similarity thresholds must be finite positive values")

    candidates: list[SavedPoseDelta] = []
    for index, sample in enumerate(samples):
        sample_id = sample.get("sample_id")
        if isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id < 0:
            raise ValueError(f"samples[{index}].sample_id must be a nonnegative integer")
        if "T_base_ee" not in sample:
            raise ValueError(f"samples[{index}].T_base_ee is required")
        translation_m, rotation_deg = pose_delta_fn(T_base_ee, sample["T_base_ee"])
        translation_m = float(translation_m)
        rotation_deg = float(rotation_deg)
        if (
            not math.isfinite(translation_m)
            or translation_m < 0.0
            or not math.isfinite(rotation_deg)
            or rotation_deg < 0.0
        ):
            raise ValueError(f"samples[{index}] produced an invalid pose delta")
        candidates.append(
            SavedPoseDelta(
                sample_id=sample_id,
                translation_m=translation_m,
                rotation_deg=rotation_deg,
                normalized_max_distance=max(
                    translation_m / translation_threshold,
                    rotation_deg / rotation_threshold,
                ),
            )
        )

    def distance_key(delta: SavedPoseDelta) -> tuple[float, float, float, int]:
        return (
            delta.normalized_max_distance,
            delta.translation_m,
            delta.rotation_deg,
            delta.sample_id,
        )

    nearest = min(candidates, key=distance_key, default=None)
    similar_candidates = [
        candidate
        for candidate in candidates
        if candidate.translation_m < translation_threshold and candidate.rotation_deg < rotation_threshold
    ]
    similar = min(similar_candidates, key=distance_key, default=None)
    previous = max(candidates, key=lambda candidate: candidate.sample_id, default=None)
    return SavedPoseComparison(nearest=nearest, similar=similar, previous=previous)


def _empty_pose_comparison() -> SavedPoseComparison:
    return SavedPoseComparison(nearest=None, similar=None, previous=None)


def build_sample_record(
    *,
    sample_id: int,
    color_frame: ColorFrame,
    detection: Any,
    estimate: Any,
    fresh_pose: Any,
    blur_score: float,
    camera_info: Any,
    opencv_version: str,
    legacy_pattern: bool,
    force_similar: bool,
    previous_pose_delta: SavedPoseDelta | None,
) -> dict[str, Any]:
    """Build one explicit frame/pose record for ``HandEyeSampleStore``."""
    numpy = importlib.import_module("numpy")
    charuco_ids = numpy.asarray(detection.charuco_ids, dtype=numpy.int64).reshape(-1)
    charuco_corners = numpy.asarray(detection.charuco_corners, dtype=numpy.float64).reshape(-1, 2)
    if len(charuco_ids) != len(charuco_corners):
        raise ValueError("detected ChArUco IDs and corners have different lengths")

    return {
        "sample_id": int(sample_id),
        "camera_timestamp_ms": float(color_frame.timestamp_ms),
        "robot_timestamp": fresh_pose.robot_timestamp,
        "image_width": int(camera_info.width),
        "image_height": int(camera_info.height),
        "charuco_ids": [int(value) for value in charuco_ids],
        "charuco_corners_px": charuco_corners.tolist(),
        "num_charuco_corners": int(len(charuco_ids)),
        "rvec_camera_board": numpy.asarray(estimate.rvec_camera_board, dtype=numpy.float64)
        .reshape(3)
        .tolist(),
        "tvec_camera_board_m": numpy.asarray(estimate.tvec_camera_board_m, dtype=numpy.float64)
        .reshape(3)
        .tolist(),
        "T_camera_board": numpy.asarray(estimate.T_camera_board, dtype=numpy.float64).reshape(4, 4).tolist(),
        "T_base_ee": numpy.asarray(fresh_pose.T_base_ee, dtype=numpy.float64).reshape(4, 4).tolist(),
        "robot_pose_raw": fresh_pose.robot_pose_raw,
        "reprojection_error_px": float(estimate.reprojection_error_px),
        "opencv_version": str(opencv_version),
        "realsense_serial": str(camera_info.serial),
        "robot_pose_name": fresh_pose.robot_pose_name,
        "translation_unit": fresh_pose.translation_unit,
        "matrix_storage_source": fresh_pose.matrix_storage_source,
        "matrix_storage_format": fresh_pose.matrix_storage_format,
        "robot_request_local_monotonic_s": float(fresh_pose.local_monotonic_s),
        "robot_request_latency_ms": float(fresh_pose.request_latency_ms),
        "blur_score": float(blur_score),
        "detection_api": str(detection.api_name),
        "legacy_pattern": bool(legacy_pattern),
        "similar_pose_forced": bool(force_similar),
        "translation_delta_to_previous_m": (
            None if previous_pose_delta is None else float(previous_pose_delta.translation_m)
        ),
        "rotation_delta_to_previous_deg": (
            None if previous_pose_delta is None else float(previous_pose_delta.rotation_deg)
        ),
    }


def capture_bound_sample(
    *,
    color_frame: ColorFrame,
    overlay_rgb: Any,
    detection: Any,
    estimate: Any,
    blur_score: float,
    pose_reader: Any,
    stillness_monitor: Any,
    monitor_factory: Callable[[], Any],
    store: Any,
    camera_info: Any,
    opencv_version: str,
    legacy_pattern: bool,
    min_charuco_corners: int,
    warning_reprojection_error_px: float,
    max_reprojection_error_px: float,
    min_laplacian_variance: float,
    required_stillness_s: float,
    similarity_translation_m: float,
    similarity_rotation_deg: float,
    force_similar: bool = False,
    pose_delta_fn: Callable[[Any, Any], tuple[float, float]] | None = None,
    robot_poll: RobotPollResult | None = None,
    vision_error: str | None = None,
) -> CaptureAttempt:
    """Freeze one analyzed image and bind it to a supplied or newly read pose."""
    numpy = importlib.import_module("numpy")
    frozen_rgb = numpy.array(color_frame.rgb, copy=True)
    frozen_overlay = numpy.array(overlay_rgb, copy=True)
    frozen_frame = ColorFrame(rgb=frozen_rgb, timestamp_ms=float(color_frame.timestamp_ms))

    if detection is None:
        frozen_detection = None
        corner_count = 0
    else:
        frozen_detection = type("FrozenDetection", (), {})()
        frozen_detection.charuco_ids = numpy.array(detection.charuco_ids, copy=True)
        frozen_detection.charuco_corners = numpy.array(detection.charuco_corners, copy=True)
        frozen_detection.num_charuco_corners = int(detection.num_charuco_corners)
        frozen_detection.api_name = str(detection.api_name)
        corner_count = frozen_detection.num_charuco_corners

    if estimate is None:
        frozen_estimate = None
        reprojection = None
    else:
        frozen_estimate = type("FrozenEstimate", (), {})()
        frozen_estimate.rvec_camera_board = numpy.array(estimate.rvec_camera_board, copy=True)
        frozen_estimate.tvec_camera_board_m = numpy.array(estimate.tvec_camera_board_m, copy=True)
        frozen_estimate.T_camera_board = numpy.array(estimate.T_camera_board, copy=True)
        frozen_estimate.reprojection_error_px = float(estimate.reprojection_error_px)
        reprojection = frozen_estimate.reprojection_error_px

    poll = robot_poll
    if poll is None:
        poll = poll_robot_pose(
            pose_reader,
            stillness_monitor,
            monitor_factory=monitor_factory,
        )
    if poll.reading is None:
        eligibility = evaluate_capture_eligibility(
            detection_success=frozen_detection is not None and corner_count > 0,
            num_charuco_corners=corner_count,
            min_charuco_corners=min_charuco_corners,
            pnp_success=frozen_estimate is not None,
            reprojection_error_px=reprojection,
            warning_reprojection_error_px=warning_reprojection_error_px,
            max_reprojection_error_px=max_reprojection_error_px,
            laplacian_score=blur_score,
            min_laplacian_variance=min_laplacian_variance,
            robot_pose_fresh=False,
            robot_still=False,
            stillness_history_s=0.0,
            required_stillness_s=required_stillness_s,
            similar_pose=False,
            force_similar=force_similar,
        )
        eligibility = _reject_on_vision_error(eligibility, vision_error)
        message = poll.error or "Robot state read failed"
        if vision_error is not None:
            message = f"{message}. {vision_error}"
        return CaptureAttempt(eligibility, poll.monitor, None, message)

    if pose_delta_fn is None:
        utilities = importlib.import_module("hardware_test.franka.handeye.handeye_utils")
        pose_delta_fn = utilities.pose_delta
    pose_comparison = compare_saved_poses(
        poll.reading.T_base_ee,
        store.samples,
        pose_delta_fn=pose_delta_fn,
        translation_threshold_m=similarity_translation_m,
        rotation_threshold_deg=similarity_rotation_deg,
    )
    similar_pose = pose_comparison.similar is not None
    if poll.stillness is None:  # A successful poll always returns its monitor status.
        raise AssertionError("fresh robot pose is missing stillness status")
    eligibility = evaluate_capture_eligibility(
        detection_success=frozen_detection is not None and corner_count > 0,
        num_charuco_corners=corner_count,
        min_charuco_corners=min_charuco_corners,
        pnp_success=frozen_estimate is not None,
        reprojection_error_px=reprojection,
        warning_reprojection_error_px=warning_reprojection_error_px,
        max_reprojection_error_px=max_reprojection_error_px,
        laplacian_score=blur_score,
        min_laplacian_variance=min_laplacian_variance,
        robot_pose_fresh=True,
        robot_still=bool(poll.stillness.is_still),
        stillness_history_s=float(poll.stillness.history_span_s),
        required_stillness_s=required_stillness_s,
        similar_pose=similar_pose,
        force_similar=force_similar,
    )
    eligibility = _reject_on_vision_error(eligibility, vision_error)
    if not eligibility.eligible:
        if eligibility.can_force_similarity:
            message = "Similar saved pose: press S again to force this otherwise-valid sample"
        else:
            message = "Sample rejected: " + ", ".join(eligibility.reasons)
        if vision_error is not None:
            message = f"{message}. {vision_error}"
        return CaptureAttempt(eligibility, poll.monitor, None, message)

    if frozen_detection is None or frozen_estimate is None:
        raise AssertionError("eligible capture is missing frozen vision data")
    record = build_sample_record(
        sample_id=store.next_sample_id,
        color_frame=frozen_frame,
        detection=frozen_detection,
        estimate=frozen_estimate,
        fresh_pose=poll.reading,
        blur_score=blur_score,
        camera_info=camera_info,
        opencv_version=opencv_version,
        legacy_pattern=legacy_pattern,
        force_similar=similar_pose and force_similar,
        previous_pose_delta=pose_comparison.previous,
    )
    saved = store.save(record, frozen_rgb, frozen_overlay)
    return CaptureAttempt(
        eligibility=eligibility,
        monitor=poll.monitor,
        saved_record=saved,
        message=f"Saved sample {saved['sample_id']}",
    )


def decode_key(key_code: int, *, similarity_confirmation_armed: bool) -> str:
    """Decode the exact S/D/Q/Escape/R collector keyboard contract."""
    if key_code < 0:
        return "none"
    key = key_code & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        return "quit"
    if key in (ord("d"), ord("D")):
        return "delete"
    if key in (ord("r"), ord("R")):
        return "rebuild_detector"
    if key in (ord("s"), ord("S")):
        return "force_save" if similarity_confirmation_armed else "save"
    return "none"


def target_completion_banner(saved_count: int, target_count: int) -> str | None:
    """Return a non-terminal completion banner; the event loop stays open."""
    if saved_count < target_count:
        return None
    return f"TARGET COMPLETE: {saved_count}/{target_count} saved (Q/Esc exits)"


def create_live_window(cv2: Any, window_name: str = WINDOW_NAME) -> None:
    """Create HighGUI state at point-of-use with a clear headless diagnostic."""
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except Exception as exc:
        raise RuntimeError(
            "OpenCV HighGUI display is unavailable. On a headless host, run with a display; otherwise "
            f"install GUI-enabled OpenCV and verify DISPLAY/Wayland access: {exc}"
        ) from exc


def _display_rgb_frame(cv2: Any, rgb_image: Any, *, window_name: str = WINDOW_NAME) -> int:
    """Convert RGB to BGR only at the OpenCV display boundary."""
    try:
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, bgr_image)
        return int(cv2.waitKey(1))
    except Exception as exc:
        raise RuntimeError(
            "OpenCV HighGUI failed while displaying the live frame. Check headless display access and "
            f"use a GUI-enabled OpenCV build: {exc}"
        ) from exc


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_collection_metadata(
    *,
    output_dir: Path,
    resolved_config: Mapping[str, Any],
    args: argparse.Namespace,
    camera_info: ColorCameraInfo,
    opencv_version: str,
    yaml_module: Any,
) -> None:
    """Persist authoritative intrinsics and the fully resolved capture settings."""
    intrinsics = {
        "schema_version": 1,
        "camera_frame": "l515_color_optical_frame",
        "frame_directions": {
            "T_camera_board": "charuco_board -> l515_color_optical_frame",
            "T_base_ee": "franka_end_effector -> franka_base",
            "T_base_camera": "l515_color_optical_frame -> franka_base",
        },
        "resolution": {"width": camera_info.width, "height": camera_info.height},
        "fps": camera_info.fps,
        "realsense_serial": camera_info.serial,
        "camera_matrix": camera_info.camera_matrix.tolist(),
        "fx": camera_info.fx,
        "fy": camera_info.fy,
        "ppx": camera_info.ppx,
        "ppy": camera_info.ppy,
        "distortion": {
            "model": camera_info.distortion_model,
            "coefficients": list(camera_info.distortion_coefficients),
            "coefficient_order": ["k1", "k2", "p1", "p2", "k3"],
        },
        "opencv_version": opencv_version,
    }
    _atomic_write_text(
        output_dir / "camera_intrinsics.json",
        json.dumps(intrinsics, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )

    config_used = {
        "resolved_config": dict(resolved_config),
        "cli_capture": {
            "config_path": str(Path(args.config).resolve()),
            "requested_width": args.width,
            "requested_height": args.height,
            "requested_fps": args.fps,
            "camera_serial_requested": args.camera_serial,
            "active_width": camera_info.width,
            "active_height": camera_info.height,
            "active_fps": camera_info.fps,
            "camera_serial_active": camera_info.serial,
            "output_dir": str(args.output_dir),
            "num_samples": args.num_samples,
            "control_host": args.control_host,
        },
        "opencv_version": opencv_version,
        "legacy_pattern": bool(resolved_config["charuco"]["legacy_pattern"]),
        "legacy_pattern_warning": LEGACY_PATTERN_WARNING,
    }
    yaml_text = yaml_module.safe_dump(config_used, sort_keys=False)
    _atomic_write_text(output_dir / "config_used.yaml", yaml_text)


def _resume_metadata_error(detail: str) -> RuntimeError:
    return RuntimeError(f"resume metadata is incompatible: {detail}")


def _load_resume_intrinsics(path: Path) -> Mapping[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _resume_metadata_error(f"cannot read camera_intrinsics.json: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise _resume_metadata_error("camera_intrinsics.json root must be a mapping")
    return loaded


def _load_resume_config(path: Path, yaml_module: Any) -> Mapping[str, Any]:
    try:
        loaded = yaml_module.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, Exception) as exc:
        if isinstance(exc, OSError) or exc.__class__.__module__.startswith("yaml"):
            raise _resume_metadata_error(f"cannot read config_used.yaml: {exc}") from exc
        raise
    if not isinstance(loaded, Mapping):
        raise _resume_metadata_error("config_used.yaml root must be a mapping")
    nested = loaded.get("resolved_config", loaded.get("config", loaded))
    if not isinstance(nested, Mapping):
        raise _resume_metadata_error("config_used.yaml resolved config must be a mapping")
    return nested


def _normalized_resume_intrinsics(metadata: Mapping[str, Any]) -> dict[str, Any]:
    resolution = metadata.get("resolution")
    if isinstance(resolution, Mapping):
        width = resolution.get("width")
        height = resolution.get("height")
    else:
        width = metadata.get("width")
        height = metadata.get("height")

    distortion = metadata.get("distortion")
    if isinstance(distortion, Mapping):
        distortion_model = distortion.get("model")
        distortion_coefficients = distortion.get("coefficients")
    else:
        distortion_model = metadata.get("distortion_model")
        distortion_coefficients = metadata.get("distortion_coefficients")

    return {
        "width": width,
        "height": height,
        "serial": metadata.get("realsense_serial", metadata.get("serial")),
        "camera_matrix": metadata.get("camera_matrix"),
        "distortion_model": distortion_model,
        "distortion_coefficients": distortion_coefficients,
    }


def _validate_resume_intrinsics(metadata: Mapping[str, Any], camera_info: ColorCameraInfo) -> None:
    numpy = importlib.import_module("numpy")
    existing = _normalized_resume_intrinsics(metadata)
    for key, active_value in (("width", camera_info.width), ("height", camera_info.height)):
        if existing[key] != active_value:
            raise _resume_metadata_error(
                f"active {key} {active_value!r} does not match stored {key} {existing[key]!r}"
            )
    if existing["serial"] != camera_info.serial:
        raise _resume_metadata_error(
            f"active serial {camera_info.serial!r} does not match stored serial {existing['serial']!r}"
        )

    try:
        stored_matrix = numpy.asarray(existing["camera_matrix"], dtype=numpy.float64)
        active_matrix = numpy.asarray(camera_info.camera_matrix, dtype=numpy.float64)
    except (TypeError, ValueError) as exc:
        raise _resume_metadata_error(f"camera_matrix is not numeric: {exc}") from exc
    if (
        stored_matrix.shape != (3, 3)
        or active_matrix.shape != (3, 3)
        or not numpy.isfinite(stored_matrix).all()
        or not numpy.allclose(stored_matrix, active_matrix, atol=1e-9, rtol=0.0)
    ):
        raise _resume_metadata_error("active camera_matrix does not match stored camera_matrix")

    stored_model = _distortion_model_name(existing["distortion_model"])
    active_model = _distortion_model_name(camera_info.distortion_model)
    if stored_model != active_model:
        raise _resume_metadata_error(
            f"active distortion model {active_model!r} does not match stored distortion model {stored_model!r}"
        )
    try:
        stored_coefficients = numpy.asarray(
            existing["distortion_coefficients"],
            dtype=numpy.float64,
        ).reshape(-1)
        active_coefficients = numpy.asarray(
            camera_info.distortion_coefficients,
            dtype=numpy.float64,
        ).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise _resume_metadata_error(f"distortion coefficients are not numeric: {exc}") from exc
    if (
        stored_coefficients.shape != (5,)
        or active_coefficients.shape != (5,)
        or not numpy.isfinite(stored_coefficients).all()
        or not numpy.allclose(stored_coefficients, active_coefficients, atol=1e-12, rtol=0.0)
    ):
        raise _resume_metadata_error(
            "active distortion coefficients do not match stored distortion coefficients"
        )


def _validate_resume_config(
    stored_config: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
) -> None:
    for section in (
        "charuco",
        "capture_validation",
        "robot_stillness",
        "pose_similarity",
    ):
        if stored_config.get(section) != resolved_config.get(section):
            raise _resume_metadata_error(
                f"stored {section} does not match the resolved {section} configuration"
            )


def prepare_or_validate_collection_metadata(
    *,
    output_dir: Path,
    store: Any,
    resolved_config: Mapping[str, Any],
    args: argparse.Namespace,
    camera_info: ColorCameraInfo,
    opencv_version: str,
    yaml_module: Any,
) -> None:
    """Write metadata for a new store, or validate a resume without rewriting it."""
    if not store.samples:
        write_collection_metadata(
            output_dir=output_dir,
            resolved_config=resolved_config,
            args=args,
            camera_info=camera_info,
            opencv_version=opencv_version,
            yaml_module=yaml_module,
        )
        return

    intrinsics_path = output_dir / "camera_intrinsics.json"
    config_path = output_dir / "config_used.yaml"
    missing = [path.name for path in (intrinsics_path, config_path) if not path.is_file()]
    if missing:
        raise _resume_metadata_error("missing " + ", ".join(missing))

    stored_intrinsics = _load_resume_intrinsics(intrinsics_path)
    stored_config = _load_resume_config(config_path, yaml_module)
    _validate_resume_intrinsics(stored_intrinsics, camera_info)
    _validate_resume_config(stored_config, resolved_config)


def _empty_detection(utilities: Any, detector_api_name: str) -> Any:
    numpy = importlib.import_module("numpy")
    return utilities.CharucoDetection(
        marker_corners=(),
        marker_ids=numpy.empty((0,), dtype=numpy.int32),
        charuco_corners=numpy.empty((0, 2), dtype=numpy.float32),
        charuco_ids=numpy.empty((0,), dtype=numpy.int32),
        api_name=detector_api_name,
    )


def _append_operator_error(current: str | None, new_error: str) -> str:
    return new_error if current is None else f"{current} | {new_error}"


def _legacy_pattern_guidance(*, legacy_pattern: bool, opencv_version: str) -> str:
    return (
        f"Check legacy_pattern={legacy_pattern!r} for this board. Running OpenCV {opencv_version}. "
        f"{LEGACY_PATTERN_WARNING}"
    )


def board_region_laplacian_score(
    rgb_image: Any,
    detection: Any,
    *,
    cv2_module: Any,
) -> float:
    """Measure focus inside an eroded convex hull of the detected board."""
    numpy = importlib.import_module("numpy")
    rgb = numpy.asarray(rgb_image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 1 or rgb.shape[1] < 1:
        raise ValueError("invalid board region: expected a non-empty RGB image")

    marker_point_groups: list[Any] = []
    for marker_corners in getattr(detection, "marker_corners", ()):
        marker_points = numpy.asarray(marker_corners, dtype=numpy.float32).reshape(-1, 2)
        if marker_points.size:
            marker_point_groups.append(marker_points)
    charuco_points = numpy.asarray(
        getattr(detection, "charuco_corners", ()),
        dtype=numpy.float32,
    ).reshape(-1, 2)
    point_groups = marker_point_groups or ([charuco_points] if charuco_points.size else [])
    if not point_groups:
        raise ValueError("invalid board region: no detected board corners")

    points = numpy.concatenate(point_groups, axis=0)
    if len(points) < 3 or not numpy.isfinite(points).all():
        raise ValueError("invalid board region: detected corners are non-finite or degenerate")
    height, width = rgb.shape[:2]
    if (
        numpy.any(points[:, 0] < 0.0)
        or numpy.any(points[:, 0] >= width)
        or numpy.any(points[:, 1] < 0.0)
        or numpy.any(points[:, 1] >= height)
    ):
        raise ValueError("invalid board region: detected corners fall outside the image")

    hull = cv2_module.convexHull(points.reshape(-1, 1, 2))
    if len(hull) < 3 or float(cv2_module.contourArea(hull)) < 64.0:
        raise ValueError("detected board region is too small or invalid for blur scoring")

    mask = numpy.zeros((height, width), dtype=numpy.uint8)
    integer_hull = numpy.rint(hull).astype(numpy.int32)
    cv2_module.fillConvexPoly(mask, integer_hull, 255)
    erosion_kernel = numpy.ones((5, 5), dtype=numpy.uint8)
    interior_mask = cv2_module.erode(mask, erosion_kernel, iterations=1) > 0
    if int(numpy.count_nonzero(interior_mask)) < 25:
        raise ValueError("detected board region is too small after boundary erosion")

    gray = cv2_module.cvtColor(rgb, cv2_module.COLOR_RGB2GRAY)
    laplacian = cv2_module.Laplacian(gray, cv2_module.CV_64F)
    score = float(numpy.var(laplacian[interior_mask]))
    if not math.isfinite(score):
        raise ValueError("invalid board region Laplacian score")
    return score


def _analyze_frame(
    *,
    frame: ColorFrame,
    detector: Any,
    board: Any,
    camera_info: ColorCameraInfo,
    distortion_coefficients: Any,
    utilities: Any,
    cv2: Any,
    axis_length_m: float,
    legacy_pattern: bool,
    opencv_version: str,
) -> tuple[Any, Any | None, float, Any, str | None]:
    detection_error: str | None = None
    legacy_guidance = _legacy_pattern_guidance(
        legacy_pattern=legacy_pattern,
        opencv_version=opencv_version,
    )
    try:
        detection = detector.detect(frame.rgb)
    except Exception as exc:
        detection = _empty_detection(utilities, getattr(detector, "api_name", "unknown"))
        detection_error = f"ChArUco detection failed: {exc}. {legacy_guidance}"

    if detection.num_charuco_corners == 0 and detection_error is None:
        detection_error = f"No ChArUco corners detected. {legacy_guidance}"

    estimate = None
    if detection.num_charuco_corners >= 4:
        try:
            estimate = utilities.estimate_board_pose(
                board=board,
                charuco_corners=detection.charuco_corners,
                charuco_ids=detection.charuco_ids,
                camera_matrix=camera_info.camera_matrix,
                distortion_coefficients=distortion_coefficients,
                cv2_module=cv2,
            )
        except Exception as exc:
            detection_error = _append_operator_error(detection_error, f"PnP failed: {exc}")

    try:
        blur_score = board_region_laplacian_score(frame.rgb, detection, cv2_module=cv2)
    except Exception as exc:
        blur_score = float("nan")
        detection_error = _append_operator_error(
            detection_error,
            f"Board-region blur scoring failed: {exc}",
        )

    try:
        overlay = utilities.draw_detection_overlay(
            frame.rgb,
            detection,
            estimate,
            camera_info.camera_matrix,
            distortion_coefficients,
            axis_length_m,
            cv2_module=cv2,
        )
    except Exception as exc:
        overlay = importlib.import_module("numpy").array(frame.rgb, copy=True)
        detection_error = _append_operator_error(
            detection_error,
            f"Detection/axis overlay failed: {exc}. {legacy_guidance}",
        )
    return detection, estimate, blur_score, overlay, detection_error


def capture_triggered_sample(
    *,
    camera: Any,
    pose_reader: Any,
    stillness_monitor: Any,
    monitor_factory: Callable[[], Any],
    store: Any,
    camera_info: Any,
    opencv_version: str,
    legacy_pattern: bool,
    min_charuco_corners: int,
    warning_reprojection_error_px: float,
    max_reprojection_error_px: float,
    min_laplacian_variance: float,
    required_stillness_s: float,
    similarity_translation_m: float,
    similarity_rotation_deg: float,
    force_similar: bool = False,
    pose_delta_fn: Callable[[Any, Any], tuple[float, float]] | None = None,
    read_frame_fn: Callable[[Any], ColorFrame] = read_color_frame,
    freeze_frame_fn: Callable[[ColorFrame], ColorFrame] = _freeze_color_frame,
    analyze_frame_fn: Callable[..., tuple[Any, Any | None, float, Any, str | None]] = _analyze_frame,
    analyze_frame_kwargs: Mapping[str, Any] | None = None,
) -> CaptureAttempt:
    """Run one S-triggered frame/pose transaction in strict acquisition order."""
    fresh_frame = read_frame_fn(camera)
    frozen_frame = freeze_frame_fn(fresh_frame)
    robot_poll = poll_robot_pose(
        pose_reader,
        stillness_monitor,
        monitor_factory=monitor_factory,
    )
    if robot_poll.reading is None:
        return capture_bound_sample(
            color_frame=frozen_frame,
            overlay_rgb=frozen_frame.rgb,
            detection=None,
            estimate=None,
            blur_score=float("nan"),
            pose_reader=pose_reader,
            stillness_monitor=stillness_monitor,
            monitor_factory=monitor_factory,
            store=store,
            camera_info=camera_info,
            opencv_version=opencv_version,
            legacy_pattern=legacy_pattern,
            min_charuco_corners=min_charuco_corners,
            warning_reprojection_error_px=warning_reprojection_error_px,
            max_reprojection_error_px=max_reprojection_error_px,
            min_laplacian_variance=min_laplacian_variance,
            required_stillness_s=required_stillness_s,
            similarity_translation_m=similarity_translation_m,
            similarity_rotation_deg=similarity_rotation_deg,
            force_similar=force_similar,
            pose_delta_fn=pose_delta_fn,
            robot_poll=robot_poll,
        )

    detection, estimate, blur_score, overlay_rgb, vision_error = analyze_frame_fn(
        frame=frozen_frame,
        **dict(analyze_frame_kwargs or {}),
    )
    attempt = capture_bound_sample(
        color_frame=frozen_frame,
        overlay_rgb=overlay_rgb,
        detection=detection,
        estimate=estimate,
        blur_score=blur_score,
        pose_reader=pose_reader,
        stillness_monitor=stillness_monitor,
        monitor_factory=monitor_factory,
        store=store,
        camera_info=camera_info,
        opencv_version=opencv_version,
        legacy_pattern=legacy_pattern,
        min_charuco_corners=min_charuco_corners,
        warning_reprojection_error_px=warning_reprojection_error_px,
        max_reprojection_error_px=max_reprojection_error_px,
        min_laplacian_variance=min_laplacian_variance,
        required_stillness_s=required_stillness_s,
        similarity_translation_m=similarity_translation_m,
        similarity_rotation_deg=similarity_rotation_deg,
        force_similar=force_similar,
        pose_delta_fn=pose_delta_fn,
        robot_poll=robot_poll,
        vision_error=vision_error,
    )
    return attempt


def _live_eligibility(
    *,
    detection: Any,
    estimate: Any | None,
    blur_score: float,
    robot_poll: RobotPollResult,
    similar_pose: bool,
    capture_config: Mapping[str, Any],
    required_stillness_s: float,
    vision_error: str | None = None,
) -> CaptureEligibility:
    stillness = robot_poll.stillness
    corner_count = int(detection.num_charuco_corners)
    eligibility = evaluate_capture_eligibility(
        detection_success=detection is not None and corner_count > 0,
        num_charuco_corners=corner_count,
        min_charuco_corners=int(capture_config["min_charuco_corners"]),
        pnp_success=estimate is not None,
        reprojection_error_px=None if estimate is None else float(estimate.reprojection_error_px),
        warning_reprojection_error_px=float(capture_config["warning_reprojection_error_px"]),
        max_reprojection_error_px=float(capture_config["max_reprojection_error_px"]),
        laplacian_score=blur_score,
        min_laplacian_variance=float(capture_config["min_laplacian_variance"]),
        robot_pose_fresh=robot_poll.reading is not None,
        robot_still=bool(stillness is not None and stillness.is_still),
        stillness_history_s=0.0 if stillness is None else float(stillness.history_span_s),
        required_stillness_s=required_stillness_s,
        similar_pose=similar_pose,
        force_similar=False,
    )
    return _reject_on_vision_error(eligibility, vision_error)


def _put_lines(cv2: Any, rgb_image: Any, lines: Sequence[tuple[str, tuple[int, int, int]]]) -> Any:
    numpy = importlib.import_module("numpy")
    annotated = numpy.array(rgb_image, copy=True)
    for index, (line, color) in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (12, 28 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated


def _pose_delta_status(label: str, delta: SavedPoseDelta | None) -> str:
    if delta is None:
        return f"{label}: none"
    return (
        f"{label} sample {delta.sample_id}: {delta.translation_m * 1000.0:.1f} mm / "
        f"{delta.rotation_deg:.2f} deg"
    )


def _annotate_live_frame(
    *,
    cv2: Any,
    overlay_rgb: Any,
    detection: Any,
    estimate: Any | None,
    blur_score: float,
    stillness: Any | None,
    eligibility: CaptureEligibility,
    saved_count: int,
    target_count: int,
    similar_pose: bool,
    pose_comparison: SavedPoseComparison,
    opencv_version: str,
    legacy_pattern: bool,
    robot_error: str | None,
    vision_error: str | None,
    operator_message: str | None,
) -> Any:
    green = (0, 255, 0)
    yellow = (255, 255, 0)
    red = (255, 0, 0)
    white = (255, 255, 255)
    status_color = {"green": green, "yellow": yellow, "red": red}[eligibility.status]
    reprojection = "n/a" if estimate is None else f"{estimate.reprojection_error_px:.3f}px"
    stillness_text = (
        "invalid" if stillness is None else f"{stillness.reason} ({stillness.history_span_s:.2f}s)"
    )
    eligibility_text = "eligible" if eligibility.eligible else ",".join(eligibility.reasons)
    lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"corners: {detection.num_charuco_corners}", white),
        (f"saved: {saved_count}/{target_count}", white),
        (
            f"reprojection: {reprojection}",
            yellow if "reprojection_warning" in eligibility.warnings else status_color,
        ),
        (f"blur (Laplacian variance): {blur_score:.1f}", white),
        (
            f"robot stillness: {stillness_text}",
            green if stillness is not None and stillness.is_still else red,
        ),
        (f"save: {eligibility_text}", status_color),
        (_pose_delta_status("nearest", pose_comparison.nearest), white),
        (_pose_delta_status("previous", pose_comparison.previous), white),
        (
            f"OpenCV {opencv_version} | legacy_pattern={legacy_pattern} | check if axes look wrong",
            yellow,
        ),
        ("keys: S save | D delete last | R rebuild detector | Q/Esc quit", white),
    ]
    if similar_pose:
        lines.append(("WARNING: pose is similar; first S arms, second S forces", yellow))
    for error in (robot_error, vision_error):
        if error:
            lines.append((error, red))
    if operator_message:
        lines.append((operator_message, yellow))
    banner = target_completion_banner(saved_count, target_count)
    if banner:
        lines.append((banner, green))
    return _put_lines(cv2, overlay_rgb, lines)


def _safe_stop_pipeline(camera: ColorCamera | None) -> None:
    if camera is None:
        return
    try:
        camera.pipeline.stop()
    except Exception as exc:
        print(f"Warning: failed to stop RealSense pipeline cleanly: {exc}", file=sys.stderr)


def _safe_close_robot(access: PassiveRobotAccess | None) -> None:
    if access is None:
        return
    try:
        access.close()
    except Exception as exc:
        print(f"Warning: failed to close passive Franka client cleanly: {exc}", file=sys.stderr)


def _safe_destroy_windows(cv2: Any, *, created: bool) -> None:
    if not created:
        return
    try:
        cv2.destroyAllWindows()
    except Exception as exc:
        print(f"Warning: failed to destroy OpenCV windows cleanly: {exc}", file=sys.stderr)


def run_collection(args: argparse.Namespace, dependencies: RuntimeDependencies) -> int:
    """Run the color-only passive collector until Q/Escape or a fatal device error."""
    _ensure_repo_root_on_path()
    utilities = importlib.import_module("hardware_test.franka.handeye.handeye_utils")
    config = utilities.load_handeye_config(args.config)
    output_dir = Path(args.output_dir)

    camera: ColorCamera | None = None
    robot: PassiveRobotAccess | None = None
    window_created = False
    try:
        camera = start_color_camera(
            dependencies.realsense,
            width=args.width,
            height=args.height,
            fps=args.fps,
            camera_serial=args.camera_serial,
        )
        store = utilities.HandEyeSampleStore(output_dir)
        prepare_or_validate_collection_metadata(
            output_dir=output_dir,
            store=store,
            resolved_config=config,
            args=args,
            camera_info=camera.info,
            opencv_version=str(dependencies.cv2.__version__),
            yaml_module=dependencies.yaml,
        )

        distortion_coefficients = utilities.opencv_distortion_coefficients(
            camera.info.distortion_model,
            camera.info.distortion_coefficients,
        )
        board = utilities.create_charuco_board(config["charuco"], cv2_module=dependencies.cv2)

        def build_detector() -> Any:
            return utilities.CharucoDetectorCompat(
                board,
                camera_matrix=camera.info.camera_matrix,
                distortion_coefficients=distortion_coefficients,
                cv2_module=dependencies.cv2,
            )

        detector = build_detector()
        robot = build_robot_access(
            args.control_host,
            pose_reader_factory=utilities.FrankaPoseReader,
        )
        stillness_config = config["robot_stillness"]

        def monitor_factory() -> Any:
            return utilities.RobotStillnessMonitor(
                window_s=float(stillness_config["window_s"]),
                max_translation_m=float(stillness_config["max_translation_m"]),
                max_rotation_deg=float(stillness_config["max_rotation_deg"]),
            )

        stillness_monitor = monitor_factory()
        create_live_window(dependencies.cv2)
        window_created = True
        similarity_armed = False
        operator_message: str | None = None
        capture_config = config["capture_validation"]
        similarity_config = config["pose_similarity"]
        axis_length_m = float(config["charuco"]["square_length_m"]) * 2.0
        required_stillness_s = max(1.0, float(stillness_config["window_s"]))

        while True:
            frame = read_color_frame(camera)
            robot_poll = poll_robot_pose(
                robot.pose_reader,
                stillness_monitor,
                monitor_factory=monitor_factory,
            )
            stillness_monitor = robot_poll.monitor
            detection, estimate, blur_score, vision_overlay, vision_error = _analyze_frame(
                frame=frame,
                detector=detector,
                board=board,
                camera_info=camera.info,
                distortion_coefficients=distortion_coefficients,
                utilities=utilities,
                cv2=dependencies.cv2,
                axis_length_m=axis_length_m,
                legacy_pattern=bool(config["charuco"]["legacy_pattern"]),
                opencv_version=str(dependencies.cv2.__version__),
            )

            pose_comparison = (
                _empty_pose_comparison()
                if robot_poll.reading is None
                else compare_saved_poses(
                    robot_poll.reading.T_base_ee,
                    store.samples,
                    pose_delta_fn=utilities.pose_delta,
                    translation_threshold_m=float(similarity_config["translation_m"]),
                    rotation_threshold_deg=float(similarity_config["rotation_deg"]),
                )
            )
            similar_pose = pose_comparison.similar is not None
            eligibility = _live_eligibility(
                detection=detection,
                estimate=estimate,
                blur_score=blur_score,
                robot_poll=robot_poll,
                similar_pose=similar_pose,
                capture_config=capture_config,
                required_stillness_s=required_stillness_s,
                vision_error=vision_error,
            )
            display_rgb = _annotate_live_frame(
                cv2=dependencies.cv2,
                overlay_rgb=vision_overlay,
                detection=detection,
                estimate=estimate,
                blur_score=blur_score,
                stillness=robot_poll.stillness,
                eligibility=eligibility,
                saved_count=len(store.samples),
                target_count=args.num_samples,
                similar_pose=similar_pose,
                pose_comparison=pose_comparison,
                opencv_version=str(dependencies.cv2.__version__),
                legacy_pattern=bool(config["charuco"]["legacy_pattern"]),
                robot_error=robot_poll.error,
                vision_error=vision_error,
                operator_message=operator_message,
            )
            action = decode_key(
                _display_rgb_frame(dependencies.cv2, display_rgb),
                similarity_confirmation_armed=similarity_armed,
            )
            if action == "quit":
                break
            if action == "delete":
                deleted = store.delete_last()
                operator_message = (
                    "No complete sample bundle to delete"
                    if deleted is None
                    else f"Deleted complete sample bundle {deleted['sample_id']}"
                )
                similarity_armed = False
                continue
            if action == "rebuild_detector":
                detector = build_detector()
                operator_message = f"Rebuilt detector using {detector.api_name}"
                similarity_armed = False
                continue
            if action not in {"save", "force_save"}:
                continue

            attempt = capture_triggered_sample(
                camera=camera,
                pose_reader=robot.pose_reader,
                stillness_monitor=stillness_monitor,
                monitor_factory=monitor_factory,
                store=store,
                camera_info=camera.info,
                opencv_version=str(dependencies.cv2.__version__),
                legacy_pattern=bool(config["charuco"]["legacy_pattern"]),
                min_charuco_corners=int(capture_config["min_charuco_corners"]),
                warning_reprojection_error_px=float(capture_config["warning_reprojection_error_px"]),
                max_reprojection_error_px=float(capture_config["max_reprojection_error_px"]),
                min_laplacian_variance=float(capture_config["min_laplacian_variance"]),
                required_stillness_s=required_stillness_s,
                similarity_translation_m=float(similarity_config["translation_m"]),
                similarity_rotation_deg=float(similarity_config["rotation_deg"]),
                force_similar=action == "force_save",
                pose_delta_fn=utilities.pose_delta,
                analyze_frame_fn=_analyze_frame,
                analyze_frame_kwargs={
                    "detector": detector,
                    "board": board,
                    "camera_info": camera.info,
                    "distortion_coefficients": distortion_coefficients,
                    "utilities": utilities,
                    "cv2": dependencies.cv2,
                    "axis_length_m": axis_length_m,
                    "legacy_pattern": bool(config["charuco"]["legacy_pattern"]),
                    "opencv_version": str(dependencies.cv2.__version__),
                },
            )
            stillness_monitor = attempt.monitor
            operator_message = attempt.message
            similarity_armed = attempt.eligibility.can_force_similarity
            if attempt.saved_record is not None:
                similarity_armed = False
                print(attempt.message)
    finally:
        _safe_stop_pipeline(camera)
        _safe_close_robot(robot)
        _safe_destroy_windows(dependencies.cv2, created=window_created)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; argument help exits before any runtime import or hardware access."""
    args = build_arg_parser().parse_args(argv)
    try:
        dependencies = check_runtime_dependencies()
        return run_collection(args, dependencies)
    except KeyboardInterrupt:
        print("Collection interrupted by operator", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Collector error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
