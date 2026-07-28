from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from hardware_test.franka.defaults import DEFAULT_CONTROL_HOST
from hardware_test.franka.handeye import collect_eye_to_hand as collector

COLLECTOR_PATH = Path(__file__).parent / "handeye" / "collect_eye_to_hand.py"
CONFIG_PATH = Path(__file__).parent / "handeye" / "config" / "l515_eye_to_hand.yaml"


def test_import_is_lazy_about_runtime_and_hardware_dependencies():
    source = """
import builtins
real_import = builtins.__import__
forbidden = {'cv2', 'pyrealsense2',
                 'hardware_test.franka.handeye.handeye_utils'}
def guarded_import(name, *args, **kwargs):
    if name in forbidden:
        raise AssertionError(f'forbidden import during module load: {name}')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import hardware_test.franka.handeye.collect_eye_to_hand
print('lazy-import-ok')
"""

    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "lazy-import-ok"


def test_direct_script_help_needs_only_repo_pythonpath_and_starts_no_runtime():
    completed = subprocess.run(
        [sys.executable, str(COLLECTOR_PATH), "--help"],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert completed.returncode == 0, completed.stderr
    assert "--camera-serial" in completed.stdout
    assert "--control-host" in completed.stdout
    assert "--num-samples" in completed.stdout


def test_parser_has_exact_safe_defaults():
    args = collector.build_arg_parser().parse_args([])

    assert args.config == CONFIG_PATH
    assert args.width == 960
    assert args.height == 540
    assert args.fps == 30
    assert args.camera_serial is None
    assert args.output_dir == Path("outputs/handeye/l515_eye_to_hand")
    assert args.num_samples == 20
    assert args.control_host == DEFAULT_CONTROL_HOST


def test_parser_all_requested_flags_override_defaults(tmp_path: Path):
    config = tmp_path / "custom.yaml"
    output = tmp_path / "captures"

    args = collector.build_arg_parser().parse_args(
        [
            "--config",
            str(config),
            "--width",
            "1280",
            "--height",
            "720",
            "--fps",
            "15",
            "--camera-serial",
            "L515-123",
            "--output-dir",
            str(output),
            "--num-samples",
            "37",
            "--control-host",
            "10.0.0.8",
        ]
    )

    assert vars(args) == {
        "config": config,
        "width": 1280,
        "height": 720,
        "fps": 15,
        "camera_serial": "L515-123",
        "output_dir": output,
        "num_samples": 37,
        "control_host": "10.0.0.8",
    }


@pytest.mark.parametrize("flag", ["--width", "--height", "--fps", "--num-samples"])
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "not-an-int"])
def test_parser_rejects_non_positive_integers(flag: str, value: str):
    with pytest.raises(SystemExit):
        collector.build_arg_parser().parse_args([flag, value])


class _ModernCharucoDetector:
    def __init__(self, _board: object) -> None:
        pass

    def detectBoard(self, _image: object) -> tuple[object, ...]:  # noqa: N802 - OpenCV API.
        return (), (), (), ()


def _runtime_modules() -> dict[str, object]:
    aruco = SimpleNamespace(CharucoDetector=_ModernCharucoDetector)
    cv2 = SimpleNamespace(
        __version__="4.test",
        aruco=aruco,
        solvePnP=lambda *_args: None,
        projectPoints=lambda *_args: None,
    )
    rs = SimpleNamespace(
        pipeline=lambda: object(),
        config=lambda: object(),
        stream=SimpleNamespace(color="color"),
        format=SimpleNamespace(rgb8="rgb8"),
    )
    return {
        "cv2": cv2,
        "numpy": SimpleNamespace(__version__="test"),
        "pyrealsense2": rs,
        "yaml": SimpleNamespace(__version__="test"),
    }


def _importer_for(modules: dict[str, object]):
    def import_module(name: str) -> object:
        value = modules[name]
        if isinstance(value, BaseException):
            raise value
        return value

    return import_module


def _missing_module(name: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(f"No module named {name!r}", name=name)


def test_dependency_check_returns_all_runtime_modules_and_checks_no_highgui():
    modules = _runtime_modules()
    cv2 = modules["cv2"]
    cv2.namedWindow = lambda *_args: (_ for _ in ()).throw(AssertionError("HighGUI checked early"))

    dependencies = collector.check_runtime_dependencies(import_module=_importer_for(modules))

    assert dependencies.cv2 is modules["cv2"]
    assert dependencies.numpy is modules["numpy"]
    assert dependencies.realsense is modules["pyrealsense2"]
    assert dependencies.yaml is modules["yaml"]


def test_dependency_check_reports_missing_opencv_actionably():
    modules = _runtime_modules() | {"cv2": _missing_module("cv2")}

    with pytest.raises(RuntimeError, match=r"OpenCV.*cv2.*required.*contrib/aruco"):
        collector.check_runtime_dependencies(import_module=_importer_for(modules))


def test_dependency_check_distinguishes_missing_aruco_contrib_build():
    modules = _runtime_modules()
    modules["cv2"] = SimpleNamespace(
        __version__="4.test",
        solvePnP=lambda *_args: None,
        projectPoints=lambda *_args: None,
    )

    with pytest.raises(RuntimeError, match=r"cv2\.aruco.*contrib/aruco.*never auto-install"):
        collector.check_runtime_dependencies(import_module=_importer_for(modules))


@pytest.mark.parametrize("missing_api", ["solvePnP", "projectPoints"])
def test_dependency_check_requires_pnp_and_projection(missing_api: str):
    modules = _runtime_modules()
    delattr(modules["cv2"], missing_api)

    with pytest.raises(RuntimeError, match=missing_api):
        collector.check_runtime_dependencies(import_module=_importer_for(modules))


def test_dependency_check_accepts_complete_legacy_charuco_api():
    modules = _runtime_modules()
    modules["cv2"].aruco = SimpleNamespace(
        detectMarkers=lambda *_args: None,
        interpolateCornersCharuco=lambda *_args: None,
    )

    collector.check_runtime_dependencies(import_module=_importer_for(modules))


@pytest.mark.parametrize("missing_api", ["detectMarkers", "interpolateCornersCharuco"])
def test_dependency_check_rejects_incomplete_legacy_charuco_api(missing_api: str):
    modules = _runtime_modules()
    legacy = {
        "detectMarkers": lambda *_args: None,
        "interpolateCornersCharuco": lambda *_args: None,
    }
    legacy.pop(missing_api)
    modules["cv2"].aruco = SimpleNamespace(**legacy)

    with pytest.raises(RuntimeError, match=r"CharucoDetector.*detectMarkers.*interpolateCornersCharuco"):
        collector.check_runtime_dependencies(import_module=_importer_for(modules))


def test_dependency_check_reports_missing_realsense_package():
    modules = _runtime_modules() | {"pyrealsense2": _missing_module("pyrealsense2")}

    with pytest.raises(RuntimeError, match=r"pyrealsense2.*required.*RealSense SDK"):
        collector.check_runtime_dependencies(import_module=_importer_for(modules))


def test_dependency_check_distinguishes_installed_realsense_initialization_failure():
    modules = _runtime_modules()
    modules["pyrealsense2"].pipeline = lambda: (_ for _ in ()).throw(
        RuntimeError("failed to set power state: permission denied")
    )

    with pytest.raises(RuntimeError, match=r"installed.*initialize.*SDK/device/udev.*permission denied"):
        collector.check_runtime_dependencies(import_module=_importer_for(modules))


@pytest.mark.parametrize("name", ["numpy", "yaml"])
def test_dependency_check_reports_other_missing_runtime_packages(name: str):
    modules = _runtime_modules() | {name: _missing_module(name)}

    with pytest.raises(RuntimeError, match=rf"{name}.*required"):
        collector.check_runtime_dependencies(import_module=_importer_for(modules))


def test_highgui_failure_is_checked_only_when_window_is_used():
    cv2 = SimpleNamespace(
        WINDOW_NORMAL=0,
        namedWindow=lambda *_args: (_ for _ in ()).throw(RuntimeError("no display backend")),
    )

    with pytest.raises(RuntimeError, match=r"HighGUI.*headless.*GUI-enabled OpenCV.*no display backend"):
        collector.create_live_window(cv2, "collector")


class _FakeColorFrame:
    def __init__(self, image: np.ndarray | None = None, timestamp_ms: float = 12.5) -> None:
        self.image = image
        self.timestamp_ms = timestamp_ms

    def __bool__(self) -> bool:
        return True

    def get_data(self) -> np.ndarray:
        assert self.image is not None
        return self.image

    def get_timestamp(self) -> float:
        return self.timestamp_ms


class _FakeFrameSet:
    def __init__(self, color_frame: _FakeColorFrame) -> None:
        self.color_frame = color_frame

    def get_color_frame(self) -> _FakeColorFrame:
        return self.color_frame


class _FakePipeline:
    def __init__(self, profile: object) -> None:
        self.profile = profile
        self.started_with: object | None = None
        self.wait_count = 0
        self.stopped = False
        self.next_frame = _FakeColorFrame()

    def start(self, config: object) -> object:
        self.started_with = config
        return self.profile

    def wait_for_frames(self) -> _FakeFrameSet:
        self.wait_count += 1
        return _FakeFrameSet(self.next_frame)

    def stop(self) -> None:
        self.stopped = True


class _FakeRsConfig:
    def __init__(self) -> None:
        self.enabled_devices: list[str] = []
        self.enabled_streams: list[tuple[object, ...]] = []

    def enable_device(self, serial: str) -> None:
        self.enabled_devices.append(serial)

    def enable_stream(self, *args: object) -> None:
        self.enabled_streams.append(args)


class _FakeVideoProfile:
    def __init__(self) -> None:
        self.intrinsics = SimpleNamespace(
            width=848,
            height=480,
            fx=604.5,
            fy=603.25,
            ppx=422.0,
            ppy=239.5,
            model=SimpleNamespace(name="brown_conrady"),
            coeffs=[0.1, -0.2, 0.003, -0.004, 0.05],
        )

    def as_video_stream_profile(self) -> _FakeVideoProfile:
        return self

    def width(self) -> int:
        return 848

    def height(self) -> int:
        return 480

    def fps(self) -> int:
        return 15

    def get_intrinsics(self) -> object:
        return self.intrinsics


class _FakeActiveProfile:
    def __init__(self, video_profile: _FakeVideoProfile, color_token: object, serial_token: object) -> None:
        self.video_profile = video_profile
        self.color_token = color_token
        self.serial_token = serial_token
        self.requested_streams: list[object] = []

    def get_stream(self, stream: object) -> _FakeVideoProfile:
        self.requested_streams.append(stream)
        assert stream == self.color_token
        return self.video_profile

    def get_device(self) -> object:
        serial_token = self.serial_token

        class Device:
            def get_info(self, token: object) -> str:
                assert token == serial_token
                return "actual-L515-serial"

        return Device()


@dataclass
class _FakeRsBundle:
    module: object
    pipeline: _FakePipeline
    config: _FakeRsConfig
    profile: _FakeActiveProfile


def _fake_realsense() -> _FakeRsBundle:
    color = object()
    rgb8 = object()
    serial = object()
    profile = _FakeActiveProfile(_FakeVideoProfile(), color, serial)
    pipeline = _FakePipeline(profile)
    config = _FakeRsConfig()
    module = SimpleNamespace(
        pipeline=lambda: pipeline,
        config=lambda: config,
        stream=SimpleNamespace(color=color),
        format=SimpleNamespace(rgb8=rgb8),
        camera_info=SimpleNamespace(serial_number=serial),
    )
    return _FakeRsBundle(module, pipeline, config, profile)


def test_start_color_camera_enables_only_one_rgb8_color_stream_and_discards_30_frames():
    fake = _fake_realsense()

    camera = collector.start_color_camera(
        fake.module,
        width=960,
        height=540,
        fps=30,
        camera_serial="requested-serial",
    )

    assert fake.config.enabled_devices == ["requested-serial"]
    assert fake.config.enabled_streams == [(fake.module.stream.color, 960, 540, fake.module.format.rgb8, 30)]
    assert fake.pipeline.started_with is fake.config
    assert fake.pipeline.wait_count == 30
    assert fake.profile.requested_streams == [fake.module.stream.color]
    assert camera.pipeline is fake.pipeline
    assert camera.active_profile is fake.profile


def test_start_color_camera_uses_actual_active_profile_intrinsics_and_serial():
    fake = _fake_realsense()

    camera = collector.start_color_camera(fake.module, width=960, height=540, fps=30)

    assert camera.info.width == 848
    assert camera.info.height == 480
    assert camera.info.fps == 15
    assert camera.info.fx == pytest.approx(604.5)
    assert camera.info.fy == pytest.approx(603.25)
    assert camera.info.ppx == pytest.approx(422.0)
    assert camera.info.ppy == pytest.approx(239.5)
    assert camera.info.distortion_model == "brown_conrady"
    assert camera.info.distortion_coefficients == pytest.approx((0.1, -0.2, 0.003, -0.004, 0.05))
    assert camera.info.serial == "actual-L515-serial"
    np.testing.assert_allclose(
        camera.info.camera_matrix,
        [[604.5, 0.0, 422.0], [0.0, 603.25, 239.5], [0.0, 0.0, 1.0]],
    )


def test_start_color_camera_stops_pipeline_after_startup_failure():
    fake = _fake_realsense()
    fake.profile.video_profile.intrinsics.coeffs = [0.1, 0.2]

    with pytest.raises(RuntimeError, match=r"five distortion coefficients"):
        collector.start_color_camera(fake.module, width=960, height=540, fps=30)

    assert fake.pipeline.stopped


def test_read_color_frame_uses_actual_resolution_and_never_resizes():
    fake = _fake_realsense()
    camera = collector.start_color_camera(fake.module, width=960, height=540, fps=30)
    expected = np.zeros((480, 848, 3), dtype=np.uint8)
    fake.pipeline.next_frame = _FakeColorFrame(expected, timestamp_ms=991.25)

    frame = collector.read_color_frame(camera)

    assert frame.timestamp_ms == pytest.approx(991.25)
    assert frame.rgb.shape == (480, 848, 3)
    assert np.shares_memory(frame.rgb, expected)

    fake.pipeline.next_frame = _FakeColorFrame(np.zeros((540, 960, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match=r"960x540.*active profile.*848x480.*not resized"):
        collector.read_color_frame(camera)


def _eligible_inputs() -> dict[str, object]:
    return {
        "detection_success": True,
        "num_charuco_corners": 12,
        "min_charuco_corners": 12,
        "pnp_success": True,
        "reprojection_error_px": 0.5,
        "warning_reprojection_error_px": 1.0,
        "max_reprojection_error_px": 2.0,
        "laplacian_score": 150.0,
        "min_laplacian_variance": 100.0,
        "robot_pose_fresh": True,
        "robot_still": True,
        "stillness_history_s": 1.0,
        "required_stillness_s": 1.0,
        "similar_pose": False,
        "force_similar": False,
    }


def test_capture_eligibility_accepts_every_threshold_at_its_valid_boundary():
    inputs = _eligible_inputs() | {
        "reprojection_error_px": 2.0,
        "laplacian_score": 100.0,
    }

    result = collector.evaluate_capture_eligibility(**inputs)

    assert result.eligible
    assert result.status == "yellow"
    assert result.reasons == ()
    assert result.warnings == ("reprojection_warning",)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"detection_success": False}, "detection_failed"),
        ({"num_charuco_corners": 11}, "insufficient_charuco_corners"),
        ({"pnp_success": False}, "pnp_failed"),
        ({"reprojection_error_px": None}, "reprojection_unavailable"),
        ({"reprojection_error_px": 2.0001}, "reprojection_error_exceeded"),
        ({"laplacian_score": None}, "blur_score_unavailable"),
        ({"laplacian_score": 99.999}, "image_too_blurry"),
        ({"robot_pose_fresh": False}, "robot_pose_not_fresh"),
        ({"robot_still": False}, "robot_not_still"),
        ({"stillness_history_s": 0.999}, "stillness_history_too_short"),
    ],
)
def test_capture_eligibility_exposes_each_rejection_reason(overrides: dict[str, object], reason: str):
    result = collector.evaluate_capture_eligibility(**(_eligible_inputs() | overrides))

    assert not result.eligible
    assert result.status == "red"
    assert reason in result.reasons
    assert not result.can_force_similarity


def test_similarity_is_the_only_forceable_rejection_and_needs_second_save():
    first_press = collector.evaluate_capture_eligibility(**(_eligible_inputs() | {"similar_pose": True}))

    assert not first_press.eligible
    assert first_press.status == "yellow"
    assert first_press.reasons == ("similar_pose_confirmation_required",)
    assert first_press.warnings == ("similar_pose",)
    assert first_press.can_force_similarity

    second_press = collector.evaluate_capture_eligibility(
        **(_eligible_inputs() | {"similar_pose": True, "force_similar": True})
    )
    assert second_press.eligible
    assert second_press.status == "yellow"
    assert second_press.reasons == ()
    assert second_press.warnings == ("similar_pose_forced",)

    still_invalid = collector.evaluate_capture_eligibility(
        **(
            _eligible_inputs()
            | {
                "similar_pose": True,
                "force_similar": True,
                "robot_pose_fresh": False,
            }
        )
    )
    assert not still_invalid.eligible
    assert still_invalid.reasons == ("robot_pose_not_fresh",)
    assert not still_invalid.can_force_similarity


class _NoCommandClient:
    instances: list[_NoCommandClient] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls: list[str] = []
        self.closed = False
        self.instances.append(self)

    def get_curr(self, *, timeout: float) -> dict[str, object]:
        self.calls.append("get_curr")
        return {"ee": np.eye(4).tolist(), "timeout": timeout}

    def close(self) -> None:
        self.closed = True

    def __getattr__(self, name: str) -> Any:
        if any(token in name for token in ("move", "control", "recover", "gripper", "stop", "home")):
            raise AssertionError(f"forbidden robot command accessed: {name}")
        raise AttributeError(name)


class _FakePoseReader:
    def __init__(self, client: object, *, timeout_s: float) -> None:
        self.client = client
        self.timeout_s = timeout_s


def test_build_robot_access_constructs_only_bare_passive_client_with_exact_arguments():
    _NoCommandClient.instances.clear()

    access = collector.build_robot_access(
        "10.2.3.4",
        client_factory=_NoCommandClient,
        pose_reader_factory=_FakePoseReader,
    )

    client = _NoCommandClient.instances[-1]
    assert client.kwargs == {
        "base_url": None,
        "control_host": "10.2.3.4",
        "velocity_transport": "http",
        "zmq_url": None,
        "timeout_s": 2.0,
        "command_duration_ms": 300,
    }
    assert access.client is client
    assert access.pose_reader.client is client
    assert access.pose_reader.timeout_s == 2.0
    assert client.calls == []

    access.close()
    assert client.closed
    assert client.calls == []


class _FakeMonitor:
    def __init__(self, *, status: object | None = None) -> None:
        self.added: list[tuple[float, np.ndarray]] = []
        self._status = status or SimpleNamespace(is_still=True, history_span_s=1.1, reason="still")

    def add(self, timestamp_s: float, pose: np.ndarray) -> None:
        self.added.append((timestamp_s, np.array(pose, copy=True)))

    def status(self) -> object:
        return self._status


def test_failed_robot_poll_replaces_monitor_and_invalidates_stillness():
    original = _FakeMonitor()
    replacement = _FakeMonitor()
    reader = SimpleNamespace(read=lambda: (_ for _ in ()).throw(ConnectionError("robot offline")))

    result = collector.poll_robot_pose(reader, original, monitor_factory=lambda: replacement)

    assert result.reading is None
    assert result.stillness is None
    assert result.monitor is replacement
    assert "Robot state read failed" in result.error
    assert "robot offline" in result.error


class _MutatingPoseReader:
    def __init__(self, frame_to_mutate: np.ndarray, reading: object) -> None:
        self.frame_to_mutate = frame_to_mutate
        self.reading = reading
        self.calls = 0

    def read(self) -> object:
        self.calls += 1
        self.frame_to_mutate[:] = 255
        return self.reading


class _FakeStore:
    def __init__(
        self,
        similarity: object | None = None,
        samples: list[dict[str, object]] | None = None,
    ) -> None:
        self.next_sample_id = 7
        self.similarity = similarity
        self.samples = [] if samples is None else samples
        self.saved: list[tuple[dict[str, object], np.ndarray, np.ndarray]] = []

    def nearest_pose_delta(self, _pose: np.ndarray) -> object | None:
        return self.similarity

    def save(self, record: dict[str, object], rgb: np.ndarray, overlay: np.ndarray) -> dict[str, object]:
        bound = (record, np.array(rgb, copy=True), np.array(overlay, copy=True))
        self.saved.append(bound)
        return record


def _capture_objects() -> tuple[object, ...]:
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    overlay = np.full((2, 3, 3), 19, dtype=np.uint8)
    frame = collector.ColorFrame(rgb=rgb, timestamp_ms=1234.5)
    detection = SimpleNamespace(
        charuco_ids=np.array([0, 1, 2, 3], dtype=np.int32),
        charuco_corners=np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32),
        num_charuco_corners=4,
        api_name="CharucoDetector.detectBoard",
    )
    estimate = SimpleNamespace(
        rvec_camera_board=np.array([0.1, 0.2, 0.3]),
        tvec_camera_board_m=np.array([0.4, 0.5, 0.6]),
        T_camera_board=np.eye(4),
        reprojection_error_px=0.5,
    )
    pose = np.eye(4)
    pose[0, 3] = 0.42
    reading = SimpleNamespace(
        T_base_ee=pose,
        robot_pose_raw=pose.tolist(),
        robot_timestamp="robot-clock-17",
        local_monotonic_s=88.0,
        request_latency_ms=4.25,
        robot_pose_name="T_base_ee",
        translation_unit="meter",
        matrix_storage_source="existing_franka_client",
        matrix_storage_format="nested_4x4",
    )
    camera_info = SimpleNamespace(width=3, height=2, serial="L515-real")
    return rgb, overlay, frame, detection, estimate, reading, camera_info


def test_capture_bundle_freezes_frame_then_binds_one_fresh_pose_and_all_metadata():
    rgb, overlay, frame, detection, estimate, reading, camera_info = _capture_objects()
    pose_reader = _MutatingPoseReader(rgb, reading)
    monitor = _FakeMonitor()
    store = _FakeStore()

    attempt = collector.capture_bound_sample(
        color_frame=frame,
        overlay_rgb=overlay,
        detection=detection,
        estimate=estimate,
        blur_score=222.0,
        pose_reader=pose_reader,
        stillness_monitor=monitor,
        monitor_factory=_FakeMonitor,
        store=store,
        camera_info=camera_info,
        opencv_version="4.11.0",
        legacy_pattern=False,
        min_charuco_corners=4,
        warning_reprojection_error_px=1.0,
        max_reprojection_error_px=2.0,
        min_laplacian_variance=100.0,
        required_stillness_s=1.0,
        similarity_translation_m=0.01,
        similarity_rotation_deg=5.0,
    )

    assert pose_reader.calls == 1
    assert attempt.saved_record is store.saved[0][0]
    record, saved_rgb, saved_overlay = store.saved[0]
    assert not saved_rgb.any(), "the frame must be frozen before the fresh robot GET mutates live state"
    np.testing.assert_array_equal(saved_overlay, np.full((2, 3, 3), 19, dtype=np.uint8))
    np.testing.assert_allclose(record["T_base_ee"], reading.T_base_ee)
    assert record["robot_pose_raw"] == reading.robot_pose_raw
    assert record["camera_timestamp_ms"] == pytest.approx(1234.5)
    assert record["robot_timestamp"] == "robot-clock-17"
    assert record["robot_pose_name"] == "T_base_ee"
    assert record["translation_unit"] == "meter"
    assert record["matrix_storage_source"] == "existing_franka_client"
    assert record["matrix_storage_format"] == "nested_4x4"
    assert record["robot_request_local_monotonic_s"] == pytest.approx(88.0)
    assert record["robot_request_latency_ms"] == pytest.approx(4.25)
    assert record["blur_score"] == pytest.approx(222.0)
    assert record["detection_api"] == "CharucoDetector.detectBoard"
    assert record["legacy_pattern"] is False
    assert record["image_width"] == 3
    assert record["image_height"] == 2
    assert record["realsense_serial"] == "L515-real"
    assert record["opencv_version"] == "4.11.0"


def test_capture_bundle_robot_failure_saves_nothing_and_returns_replacement_monitor():
    _, overlay, frame, detection, estimate, _, camera_info = _capture_objects()
    replacement = _FakeMonitor()
    store = _FakeStore()
    pose_reader = SimpleNamespace(read=lambda: (_ for _ in ()).throw(TimeoutError("GET timed out")))

    attempt = collector.capture_bound_sample(
        color_frame=frame,
        overlay_rgb=overlay,
        detection=detection,
        estimate=estimate,
        blur_score=222.0,
        pose_reader=pose_reader,
        stillness_monitor=_FakeMonitor(),
        monitor_factory=lambda: replacement,
        store=store,
        camera_info=camera_info,
        opencv_version="4.11.0",
        legacy_pattern=False,
        min_charuco_corners=4,
        warning_reprojection_error_px=1.0,
        max_reprojection_error_px=2.0,
        min_laplacian_variance=100.0,
        required_stillness_s=1.0,
        similarity_translation_m=0.01,
        similarity_rotation_deg=5.0,
    )

    assert attempt.saved_record is None
    assert attempt.monitor is replacement
    assert attempt.eligibility.reasons == ("robot_pose_not_fresh",)
    assert "Robot state read failed" in attempt.message
    assert store.saved == []


def test_capture_bundle_cannot_force_non_similarity_failure():
    _, overlay, frame, detection, estimate, reading, camera_info = _capture_objects()
    monitor = _FakeMonitor(status=SimpleNamespace(is_still=False, history_span_s=2.0, reason="moving"))
    store = _FakeStore(similarity=SimpleNamespace(translation_m=0.0, rotation_deg=0.0))

    attempt = collector.capture_bound_sample(
        color_frame=frame,
        overlay_rgb=overlay,
        detection=detection,
        estimate=estimate,
        blur_score=222.0,
        pose_reader=SimpleNamespace(read=lambda: reading),
        stillness_monitor=monitor,
        monitor_factory=_FakeMonitor,
        store=store,
        camera_info=camera_info,
        opencv_version="4.11.0",
        legacy_pattern=False,
        min_charuco_corners=4,
        warning_reprojection_error_px=1.0,
        max_reprojection_error_px=2.0,
        min_laplacian_variance=100.0,
        required_stillness_s=1.0,
        similarity_translation_m=0.01,
        similarity_rotation_deg=5.0,
        force_similar=True,
    )

    assert not attempt.eligibility.eligible
    assert attempt.eligibility.reasons == ("robot_not_still",)
    assert store.saved == []


@pytest.mark.parametrize(
    ("key", "armed", "expected"),
    [
        (ord("s"), False, "save"),
        (ord("S"), True, "force_save"),
        (ord("d"), False, "delete"),
        (ord("D"), True, "delete"),
        (ord("r"), False, "rebuild_detector"),
        (ord("R"), True, "rebuild_detector"),
        (ord("q"), False, "quit"),
        (ord("Q"), True, "quit"),
        (27, False, "quit"),
        (-1, False, "none"),
        (ord("x"), True, "none"),
    ],
)
def test_key_semantics_are_exact(key: int, armed: bool, expected: str):
    assert collector.decode_key(key, similarity_confirmation_armed=armed) == expected


def test_target_count_only_changes_banner_and_never_synthesizes_quit():
    assert collector.target_completion_banner(19, 20) is None
    assert "20/20" in collector.target_completion_banner(20, 20)
    assert "target complete" in collector.target_completion_banner(25, 20).lower()
    assert collector.decode_key(-1, similarity_confirmation_armed=False) == "none"


def test_collector_source_contains_no_forbidden_camera_or_robot_control_paths():
    source = COLLECTOR_PATH.read_text(encoding="utf-8")

    for forbidden in (
        "VideoCapture",
        "FrankaRobot(",
        "cartesian_velocity_control(",
        "joint_position_control(",
        "gripper_open(",
        "gripper_close(",
        ".recover(",
        "calibrateHandEye",
        "rs.stream.depth",
        "enable_stream(rs.stream.depth",
        "rospy",
        "rclpy",
        "import zmq",
    ):
        assert forbidden not in source


class _StrictPutTextCv2:
    FONT_HERSHEY_SIMPLEX = 1
    LINE_AA = 16

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def putText(  # noqa: N802 - mirrors OpenCV.
        self,
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        font_face: int,
        font_scale: float,
        color: tuple[int, int, int],
        thickness: int,
        line_type: int,
        /,
    ) -> np.ndarray:
        assert isinstance(thickness, int)
        assert thickness == 2
        assert line_type == self.LINE_AA
        self.calls.append((image, text, origin, font_face, font_scale, color, thickness, line_type))
        return image


def test_put_lines_uses_the_standard_opencv_put_text_signature():
    cv2 = _StrictPutTextCv2()
    source = np.zeros((20, 30, 3), dtype=np.uint8)

    result = collector._put_lines(cv2, source, [("status", (1, 2, 3))])

    assert len(cv2.calls) == 1
    assert cv2.calls[0][5:] == ((1, 2, 3), 2, cv2.LINE_AA)
    assert result is not source


def _resolved_handeye_config() -> dict[str, object]:
    loaded = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _metadata_camera_info() -> collector.ColorCameraInfo:
    return collector.ColorCameraInfo(
        width=848,
        height=480,
        fps=15,
        fx=604.5,
        fy=603.25,
        ppx=422.0,
        ppy=239.5,
        distortion_model="brown_conrady",
        distortion_coefficients=(0.1, -0.2, 0.003, -0.004, 0.05),
        serial="actual-L515-serial",
        camera_matrix=np.array(
            [[604.5, 0.0, 422.0], [0.0, 603.25, 239.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
    )


def _metadata_args(output_dir: Path) -> object:
    args = collector.build_arg_parser().parse_args(["--output-dir", str(output_dir)])
    args.num_samples = 91
    return args


def _write_nested_metadata(output_dir: Path) -> tuple[bytes, bytes]:
    collector.write_collection_metadata(
        output_dir=output_dir,
        resolved_config=_resolved_handeye_config(),
        args=_metadata_args(output_dir),
        camera_info=_metadata_camera_info(),
        opencv_version="4.11.0",
        yaml_module=yaml,
    )
    return (
        (output_dir / "camera_intrinsics.json").read_bytes(),
        (output_dir / "config_used.yaml").read_bytes(),
    )


def test_prepare_resume_accepts_nested_metadata_without_rewriting_bytes(tmp_path: Path):
    before_intrinsics, before_config = _write_nested_metadata(tmp_path)
    store = SimpleNamespace(samples=[{"sample_id": 3}], save=lambda *_args: pytest.fail("must not save"))

    collector.prepare_or_validate_collection_metadata(
        output_dir=tmp_path,
        store=store,
        resolved_config=_resolved_handeye_config(),
        args=_metadata_args(tmp_path),
        camera_info=_metadata_camera_info(),
        opencv_version="4.12.0-different-but-compatible",
        yaml_module=yaml,
    )

    assert (tmp_path / "camera_intrinsics.json").read_bytes() == before_intrinsics
    assert (tmp_path / "config_used.yaml").read_bytes() == before_config


def test_prepare_resume_accepts_flat_legacy_metadata_without_rewriting_bytes(tmp_path: Path):
    camera = _metadata_camera_info()
    flat_intrinsics = {
        "width": camera.width,
        "height": camera.height,
        "fps": camera.fps,
        "fx": camera.fx,
        "fy": camera.fy,
        "ppx": camera.ppx,
        "ppy": camera.ppy,
        "distortion_model": camera.distortion_model,
        "distortion_coefficients": list(camera.distortion_coefficients),
        "serial": camera.serial,
        "camera_matrix": camera.camera_matrix.tolist(),
    }
    (tmp_path / "camera_intrinsics.json").write_text(json.dumps(flat_intrinsics), encoding="utf-8")
    (tmp_path / "config_used.yaml").write_text(
        yaml.safe_dump(_resolved_handeye_config(), sort_keys=False),
        encoding="utf-8",
    )
    before_intrinsics = (tmp_path / "camera_intrinsics.json").read_bytes()
    before_config = (tmp_path / "config_used.yaml").read_bytes()

    collector.prepare_or_validate_collection_metadata(
        output_dir=tmp_path,
        store=SimpleNamespace(samples=[{"sample_id": 0}]),
        resolved_config=_resolved_handeye_config(),
        args=_metadata_args(tmp_path),
        camera_info=camera,
        opencv_version="4.11.0",
        yaml_module=yaml,
    )

    assert (tmp_path / "camera_intrinsics.json").read_bytes() == before_intrinsics
    assert (tmp_path / "config_used.yaml").read_bytes() == before_config


def test_prepare_empty_collection_writes_metadata_normally(tmp_path: Path):
    collector.prepare_or_validate_collection_metadata(
        output_dir=tmp_path,
        store=SimpleNamespace(samples=[]),
        resolved_config=_resolved_handeye_config(),
        args=_metadata_args(tmp_path),
        camera_info=_metadata_camera_info(),
        opencv_version="4.11.0",
        yaml_module=yaml,
    )

    assert (tmp_path / "camera_intrinsics.json").is_file()
    assert (tmp_path / "config_used.yaml").is_file()


@pytest.mark.parametrize(
    ("mutate", "expected_detail"),
    [
        (lambda intrinsics, _config: intrinsics["resolution"].update(width=640), "width"),
        (lambda intrinsics, _config: intrinsics.update(realsense_serial="other-camera"), "serial"),
        (lambda intrinsics, _config: intrinsics["camera_matrix"][0].__setitem__(0, 999.0), "camera_matrix"),
        (
            lambda intrinsics, _config: intrinsics["distortion"].update(model="none"),
            "distortion model",
        ),
        (
            lambda intrinsics, _config: intrinsics["distortion"]["coefficients"].__setitem__(0, 9.0),
            "distortion coefficients",
        ),
        (
            lambda _intrinsics, config: config["resolved_config"]["charuco"].update(legacy_pattern=True),
            "charuco",
        ),
        (
            lambda _intrinsics, config: config["resolved_config"]["capture_validation"].update(
                min_charuco_corners=4
            ),
            "capture_validation",
        ),
        (
            lambda _intrinsics, config: config["resolved_config"]["robot_stillness"].update(window_s=3.0),
            "robot_stillness",
        ),
        (
            lambda _intrinsics, config: config["resolved_config"]["pose_similarity"].update(
                translation_m=0.5
            ),
            "pose_similarity",
        ),
    ],
)
def test_prepare_resume_rejects_incompatible_metadata_before_save(
    tmp_path: Path,
    mutate: Any,
    expected_detail: str,
):
    _write_nested_metadata(tmp_path)
    intrinsics = json.loads((tmp_path / "camera_intrinsics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((tmp_path / "config_used.yaml").read_text(encoding="utf-8"))
    mutate(intrinsics, config)
    (tmp_path / "camera_intrinsics.json").write_text(json.dumps(intrinsics), encoding="utf-8")
    (tmp_path / "config_used.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    store = SimpleNamespace(samples=[{"sample_id": 3}], save=lambda *_args: pytest.fail("must not save"))

    with pytest.raises(RuntimeError, match=rf"resume metadata.*{expected_detail}"):
        collector.prepare_or_validate_collection_metadata(
            output_dir=tmp_path,
            store=store,
            resolved_config=_resolved_handeye_config(),
            args=_metadata_args(tmp_path),
            camera_info=_metadata_camera_info(),
            opencv_version="4.11.0",
            yaml_module=yaml,
        )


def test_prepare_resume_requires_both_metadata_files(tmp_path: Path):
    (tmp_path / "camera_intrinsics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"resume metadata.*config_used\.yaml"):
        collector.prepare_or_validate_collection_metadata(
            output_dir=tmp_path,
            store=SimpleNamespace(samples=[{"sample_id": 0}]),
            resolved_config=_resolved_handeye_config(),
            args=_metadata_args(tmp_path),
            camera_info=_metadata_camera_info(),
            opencv_version="4.11.0",
            yaml_module=yaml,
        )


def test_compare_saved_poses_finds_a_joint_threshold_match_not_translation_lexicographic():
    samples = [
        {"sample_id": 99, "T_base_ee": [[1.0]]},
        {"sample_id": 3, "T_base_ee": [[2.0]]},
        {"sample_id": 5, "T_base_ee": [[3.0]]},
    ]
    deltas = {
        1.0: (0.005, 8.0),  # translation-nearest but outside the rotation gate
        2.0: (0.009, 4.0),
        3.0: (0.008, 1.0),  # smallest normalized max-distance matching both gates
    }

    comparison = collector.compare_saved_poses(
        [[0.0]],
        samples,
        pose_delta_fn=lambda _query, stored: deltas[float(stored[0][0])],
        translation_threshold_m=0.01,
        rotation_threshold_deg=5.0,
    )

    assert comparison.similar is not None
    assert comparison.similar.sample_id == 5
    assert comparison.similar.translation_m == pytest.approx(0.008)
    assert comparison.similar.rotation_deg == pytest.approx(1.0)
    assert comparison.previous is not None
    assert comparison.previous.sample_id == 99
    assert comparison.previous.translation_m == pytest.approx(0.005)
    assert comparison.previous.rotation_deg == pytest.approx(8.0)


def test_capture_persists_exact_delta_to_highest_id_previous_sample():
    rgb, overlay, frame, detection, estimate, reading, camera_info = _capture_objects()
    previous_pose = np.eye(4)
    previous_pose[0, 3] = 0.32
    older_pose = np.eye(4)
    older_pose[0, 3] = -0.5
    store = _FakeStore(
        samples=[
            {"sample_id": 6, "T_base_ee": previous_pose.tolist()},
            {"sample_id": 2, "T_base_ee": older_pose.tolist()},
        ]
    )

    attempt = collector.capture_bound_sample(
        color_frame=frame,
        overlay_rgb=overlay,
        detection=detection,
        estimate=estimate,
        blur_score=222.0,
        pose_reader=SimpleNamespace(read=lambda: reading),
        stillness_monitor=_FakeMonitor(),
        monitor_factory=_FakeMonitor,
        store=store,
        camera_info=camera_info,
        opencv_version="4.11.0",
        legacy_pattern=False,
        min_charuco_corners=4,
        warning_reprojection_error_px=1.0,
        max_reprojection_error_px=2.0,
        min_laplacian_variance=100.0,
        required_stillness_s=1.0,
        similarity_translation_m=0.01,
        similarity_rotation_deg=5.0,
    )

    assert attempt.saved_record is not None
    assert attempt.saved_record["translation_delta_to_previous_m"] == pytest.approx(0.10)
    assert attempt.saved_record["rotation_delta_to_previous_deg"] == pytest.approx(0.0)


def test_first_capture_persists_null_previous_pose_deltas():
    rgb, overlay, frame, detection, estimate, reading, camera_info = _capture_objects()
    store = _FakeStore()

    attempt = collector.capture_bound_sample(
        color_frame=frame,
        overlay_rgb=overlay,
        detection=detection,
        estimate=estimate,
        blur_score=222.0,
        pose_reader=SimpleNamespace(read=lambda: reading),
        stillness_monitor=_FakeMonitor(),
        monitor_factory=_FakeMonitor,
        store=store,
        camera_info=camera_info,
        opencv_version="4.11.0",
        legacy_pattern=False,
        min_charuco_corners=4,
        warning_reprojection_error_px=1.0,
        max_reprojection_error_px=2.0,
        min_laplacian_variance=100.0,
        required_stillness_s=1.0,
        similarity_translation_m=0.01,
        similarity_rotation_deg=5.0,
    )

    assert attempt.saved_record is not None
    assert attempt.saved_record["translation_delta_to_previous_m"] is None
    assert attempt.saved_record["rotation_delta_to_previous_deg"] is None


class _AnalysisUtilities:
    CharucoDetection = staticmethod(
        lambda **kwargs: SimpleNamespace(
            **kwargs,
            num_charuco_corners=int(np.asarray(kwargs["charuco_ids"]).size),
        )
    )

    @staticmethod
    def laplacian_blur_score(*_args: object, **_kwargs: object) -> float:
        return 321.0

    @staticmethod
    def draw_detection_overlay(rgb: np.ndarray, *_args: object, **_kwargs: object) -> np.ndarray:
        return np.array(rgb, copy=True)


@pytest.mark.parametrize("detector_raises", [False, True])
def test_zero_corner_or_failed_detection_reports_legacy_convention_guidance(detector_raises: bool):
    empty = SimpleNamespace(
        marker_corners=(),
        marker_ids=np.empty((0,), dtype=np.int32),
        charuco_corners=np.empty((0, 2), dtype=np.float32),
        charuco_ids=np.empty((0,), dtype=np.int32),
        num_charuco_corners=0,
        api_name="CharucoDetector.detectBoard",
    )
    detector = SimpleNamespace(
        api_name="CharucoDetector.detectBoard",
        detect=(
            (lambda _rgb: (_ for _ in ()).throw(RuntimeError("detector exploded")))
            if detector_raises
            else (lambda _rgb: empty)
        ),
    )

    _, _, _, _, error = collector._analyze_frame(
        frame=collector.ColorFrame(np.zeros((2, 3, 3), dtype=np.uint8), 1.0),
        detector=detector,
        board=object(),
        camera_info=SimpleNamespace(camera_matrix=np.eye(3)),
        distortion_coefficients=np.zeros(5),
        utilities=_AnalysisUtilities,
        cv2=object(),
        axis_length_m=0.07,
        legacy_pattern=False,
        opencv_version="4.8.1",
    )

    assert error is not None
    assert "legacy_pattern=False" in error
    assert "OpenCV 4.8.1" in error
    assert "OpenCV 4.6" in error
    assert "coordinate convention" in error


def test_live_eligibility_marks_empty_normalized_detection_as_failed():
    detection = SimpleNamespace(num_charuco_corners=0)
    stillness = SimpleNamespace(is_still=True, history_span_s=1.1)

    result = collector._live_eligibility(
        detection=detection,
        estimate=None,
        blur_score=200.0,
        robot_poll=collector.RobotPollResult(object(), stillness, object(), None),
        similar_pose=False,
        capture_config={
            "min_charuco_corners": 4,
            "warning_reprojection_error_px": 1.0,
            "max_reprojection_error_px": 2.0,
            "min_laplacian_variance": 100.0,
        },
        required_stillness_s=1.0,
    )

    assert "detection_failed" in result.reasons


def test_live_eligibility_rejects_overlay_generation_error():
    detection = SimpleNamespace(num_charuco_corners=12)
    estimate = SimpleNamespace(reprojection_error_px=0.5)
    stillness = SimpleNamespace(is_still=True, history_span_s=1.1)

    result = collector._live_eligibility(
        detection=detection,
        estimate=estimate,
        blur_score=200.0,
        robot_poll=collector.RobotPollResult(object(), stillness, object(), None),
        similar_pose=False,
        capture_config={
            "min_charuco_corners": 4,
            "warning_reprojection_error_px": 1.0,
            "max_reprojection_error_px": 2.0,
            "min_laplacian_variance": 100.0,
        },
        required_stillness_s=1.0,
        vision_error="Detection/axis overlay failed: simulated draw error",
    )

    assert not result.eligible
    assert result.status == "red"
    assert result.reasons == ("vision_processing_failed",)


def test_live_overlay_displays_nearest_and_previous_pose_deltas():
    cv2 = _StrictPutTextCv2()
    comparison = SimpleNamespace(
        nearest=SimpleNamespace(sample_id=4, translation_m=0.0123, rotation_deg=6.5),
        similar=SimpleNamespace(sample_id=7, translation_m=0.004, rotation_deg=1.25),
        previous=SimpleNamespace(sample_id=9, translation_m=0.023, rotation_deg=8.75),
    )

    collector._annotate_live_frame(
        cv2=cv2,
        overlay_rgb=np.zeros((300, 400, 3), dtype=np.uint8),
        detection=SimpleNamespace(num_charuco_corners=12),
        estimate=SimpleNamespace(reprojection_error_px=0.5),
        blur_score=200.0,
        stillness=SimpleNamespace(is_still=True, history_span_s=1.2, reason="still"),
        eligibility=collector.CaptureEligibility(True, "green", (), (), False),
        saved_count=2,
        target_count=20,
        similar_pose=True,
        pose_comparison=comparison,
        opencv_version="4.11.0",
        legacy_pattern=False,
        robot_error=None,
        vision_error=None,
        operator_message=None,
    )

    rendered_text = [str(call[1]) for call in cv2.calls]
    assert any("nearest" in text and "12.3 mm" in text and "6.50 deg" in text for text in rendered_text)
    assert any("previous" in text and "23.0 mm" in text and "8.75 deg" in text for text in rendered_text)
    assert any(
        "OpenCV 4.11.0" in text and "legacy_pattern=False" in text and "check if axes look wrong" in text
        for text in rendered_text
    )


def _trigger_capture_kwargs(
    *,
    camera: object,
    pose_reader: object,
    store: _FakeStore,
    read_frame_fn: Any,
    freeze_frame_fn: Any,
    analyze_frame_fn: Any,
) -> dict[str, object]:
    return {
        "camera": camera,
        "pose_reader": pose_reader,
        "stillness_monitor": _FakeMonitor(),
        "monitor_factory": _FakeMonitor,
        "store": store,
        "camera_info": SimpleNamespace(width=3, height=2, serial="fresh-L515"),
        "opencv_version": "4.11.0",
        "legacy_pattern": False,
        "min_charuco_corners": 4,
        "warning_reprojection_error_px": 1.0,
        "max_reprojection_error_px": 2.0,
        "min_laplacian_variance": 100.0,
        "required_stillness_s": 1.0,
        "similarity_translation_m": 0.01,
        "similarity_rotation_deg": 5.0,
        "read_frame_fn": read_frame_fn,
        "freeze_frame_fn": freeze_frame_fn,
        "analyze_frame_fn": analyze_frame_fn,
        "analyze_frame_kwargs": {},
    }


def test_s_trigger_acquires_freezes_reads_pose_then_runs_vision_and_binds_fresh_data():
    events: list[str] = []
    fresh_rgb = np.full((2, 3, 3), 73, dtype=np.uint8)
    frozen_rgb: np.ndarray | None = None
    _, overlay, _, detection, estimate, reading, _ = _capture_objects()
    store = _FakeStore()

    def read_frame(_camera: object) -> collector.ColorFrame:
        events.append("fresh_frame_read")
        return collector.ColorFrame(fresh_rgb, 9876.5)

    def freeze_frame(frame: collector.ColorFrame) -> collector.ColorFrame:
        nonlocal frozen_rgb
        events.append("rgb_frozen")
        frozen_rgb = np.array(frame.rgb, copy=True)
        return collector.ColorFrame(frozen_rgb, frame.timestamp_ms)

    class OrderedPoseReader:
        calls = 0

        def read(self) -> object:
            self.calls += 1
            events.append("fresh_pose_read")
            assert frozen_rgb is not None
            fresh_rgb[:] = 255
            return reading

    pose_reader = OrderedPoseReader()

    def analyze_frame(*, frame: collector.ColorFrame, **_kwargs: object) -> tuple[object, ...]:
        events.append("detection_pnp_overlay")
        assert pose_reader.calls == 1
        assert not np.shares_memory(frame.rgb, fresh_rgb)
        np.testing.assert_array_equal(frame.rgb, np.full((2, 3, 3), 73, dtype=np.uint8))
        return detection, estimate, 222.0, overlay, None

    attempt = collector.capture_triggered_sample(
        **_trigger_capture_kwargs(
            camera=object(),
            pose_reader=pose_reader,
            store=store,
            read_frame_fn=read_frame,
            freeze_frame_fn=freeze_frame,
            analyze_frame_fn=analyze_frame,
        )
    )

    assert events == ["fresh_frame_read", "rgb_frozen", "fresh_pose_read", "detection_pnp_overlay"]
    assert pose_reader.calls == 1
    assert attempt.saved_record is not None
    assert attempt.saved_record["camera_timestamp_ms"] == pytest.approx(9876.5)
    np.testing.assert_allclose(attempt.saved_record["T_base_ee"], reading.T_base_ee)
    np.testing.assert_array_equal(store.saved[0][1], np.full((2, 3, 3), 73, dtype=np.uint8))


def test_s_trigger_robot_failure_runs_no_vision_and_saves_nothing():
    events: list[str] = []
    store = _FakeStore()

    def read_frame(_camera: object) -> collector.ColorFrame:
        events.append("fresh_frame_read")
        return collector.ColorFrame(np.zeros((2, 3, 3), dtype=np.uint8), 25.0)

    def freeze_frame(frame: collector.ColorFrame) -> collector.ColorFrame:
        events.append("rgb_frozen")
        return collector.ColorFrame(np.array(frame.rgb, copy=True), frame.timestamp_ms)

    def fail_pose() -> object:
        events.append("fresh_pose_read")
        raise TimeoutError("fresh GET failed")

    def forbidden_vision(**_kwargs: object) -> tuple[object, ...]:
        pytest.fail("vision must not run between/after a failed fresh pose read")

    attempt = collector.capture_triggered_sample(
        **_trigger_capture_kwargs(
            camera=object(),
            pose_reader=SimpleNamespace(read=fail_pose),
            store=store,
            read_frame_fn=read_frame,
            freeze_frame_fn=freeze_frame,
            analyze_frame_fn=forbidden_vision,
        )
    )

    assert events == ["fresh_frame_read", "rgb_frozen", "fresh_pose_read"]
    assert attempt.saved_record is None
    assert store.saved == []
    assert "fresh GET failed" in attempt.message


def test_s_trigger_overlay_failure_rejects_sample_without_store_write():
    _, _, _, detection, estimate, reading, _ = _capture_objects()
    store = _FakeStore()
    raw_fallback = np.full((2, 3, 3), 73, dtype=np.uint8)

    def read_frame(_camera: object) -> collector.ColorFrame:
        return collector.ColorFrame(raw_fallback, 9876.5)

    def analyze_frame(*, frame: collector.ColorFrame, **_kwargs: object) -> tuple[object, ...]:
        return (
            detection,
            estimate,
            222.0,
            np.array(frame.rgb, copy=True),
            "Detection/axis overlay failed: simulated draw error",
        )

    attempt = collector.capture_triggered_sample(
        **_trigger_capture_kwargs(
            camera=object(),
            pose_reader=SimpleNamespace(read=lambda: reading),
            store=store,
            read_frame_fn=read_frame,
            freeze_frame_fn=collector._freeze_color_frame,
            analyze_frame_fn=analyze_frame,
        )
    )

    assert attempt.saved_record is None
    assert not attempt.eligibility.eligible
    assert "vision_processing_failed" in attempt.eligibility.reasons
    assert "Detection/axis overlay failed" in attempt.message
    assert store.saved == []


def test_run_loop_routes_both_save_actions_through_triggered_capture():
    run_source = inspect.getsource(collector.run_collection)

    assert "capture_triggered_sample(" in run_source
    assert "attempt = capture_bound_sample(" not in run_source


def _board_detection(x0: int = 45, y0: int = 30, x1: int = 115, y1: int = 90) -> object:
    hull = np.array([[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]], dtype=np.float32)
    return SimpleNamespace(
        marker_corners=(hull,),
        charuco_corners=np.array(
            [[x0 + 10, y0 + 10], [x1 - 11, y0 + 10], [x1 - 11, y1 - 11], [x0 + 10, y1 - 11]],
            dtype=np.float32,
        ),
    )


def _checkerboard(height: int, width: int, cell: int = 3) -> np.ndarray:
    rows, columns = np.indices((height, width))
    mono = (((rows // cell) + (columns // cell)) % 2 * 255).astype(np.uint8)
    return np.repeat(mono[..., None], 3, axis=2)


def test_board_blur_score_ignores_sharp_static_background_outside_detected_board():
    detection = _board_detection()
    board = cv2.GaussianBlur(_checkerboard(60, 70), (15, 15), 0)
    smooth_background = np.full((120, 160, 3), 127, dtype=np.uint8)
    sharp_background = _checkerboard(120, 160)
    smooth_background[30:90, 45:115] = board
    sharp_background[30:90, 45:115] = board

    smooth_score = collector.board_region_laplacian_score(smooth_background, detection, cv2_module=cv2)
    sharp_score = collector.board_region_laplacian_score(sharp_background, detection, cv2_module=cv2)

    assert smooth_score == pytest.approx(sharp_score, rel=0.01, abs=0.1)


def test_board_blur_score_is_higher_for_sharp_board_than_blurred_board():
    detection = _board_detection()
    sharp = np.full((120, 160, 3), 127, dtype=np.uint8)
    blurred = sharp.copy()
    sharp_board = _checkerboard(60, 70)
    sharp[30:90, 45:115] = sharp_board
    blurred[30:90, 45:115] = cv2.GaussianBlur(sharp_board, (15, 15), 0)

    sharp_score = collector.board_region_laplacian_score(sharp, detection, cv2_module=cv2)
    blurred_score = collector.board_region_laplacian_score(blurred, detection, cv2_module=cv2)

    assert sharp_score > blurred_score * 10.0


def test_board_blur_score_rejects_too_small_detected_region():
    tiny_detection = _board_detection(10, 10, 13, 13)

    with pytest.raises(ValueError, match=r"board region.*too small|invalid board region"):
        collector.board_region_laplacian_score(
            np.zeros((40, 40, 3), dtype=np.uint8),
            tiny_detection,
            cv2_module=cv2,
        )
