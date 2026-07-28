from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Event

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware_test.cameras.ros2_image_bridge import DEFAULT_IMAGE_ZMQ_ENDPOINT  # noqa: E402
from hardware_test.franka.defaults import get_control_host  # noqa: E402
from hardware_test.franka.floor_condition import encode_target_floor  # noqa: E402
from hardware_test.franka.franka_robot import FrankaRobot, FrankaRobotConfig  # noqa: E402
from hardware_test.franka.franka_spacemouse_teleop import (  # noqa: E402
    FrankaSpaceMouseTeleop,
    FrankaSpaceMouseTeleopConfig,
)
from hardware_test.franka.record_lerobot_dataset import (  # noqa: E402
    create_lerobot_dataset,
    record_lerobot_episode,
)

DEFAULT_RECORD_FPS = 30
L515_COLOR_WIDTH = 960
L515_COLOR_HEIGHT = 540
L515_COLOR_FPS = 30
L515_COLOR_FOURCC = "YUYV"
L515_COLOR_PROFILES = ((960, 540), (1280, 720), (1920, 1080))
L515_VIDEO0_GREY_PROFILES = ((480, 640), (768, 1024), (240, 320))


def build_arg_parser(*, require_target_floor: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record Franka + L515 SpaceMouse demos as LeRobotDataset.")

    parser.add_argument("--repo-id", default="local/franka_l515_smoke")
    parser.add_argument("--root", default="outputs/hardware_test/franka_l515_smoke")
    parser.add_argument("--task", default="Franka SpaceMouse teleoperation")
    parser.add_argument("--fps", type=int, default=DEFAULT_RECORD_FPS)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--reset-time-s", type=float, default=0.0)
    if require_target_floor:
        parser.add_argument("--target-floor", type=int, choices=range(1, 6), required=True)

    parser.add_argument("--control-host", default=get_control_host())
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--velocity-transport", choices=("zmq", "http"), default="zmq")
    parser.add_argument("--zmq-url", default=None)
    parser.add_argument("--action-mode", choices=("delta_ee_pose", "joint"), default="delta_ee_pose")
    parser.add_argument("--cartesian-action-units", choices=("delta", "velocity"), default="delta")
    parser.add_argument("--max-linear-velocity", type=float, default=0.05)
    parser.add_argument("--max-angular-velocity", type=float, default=0.40)
    parser.add_argument("--command-duration-ms", type=int, default=300)
    parser.add_argument("--state-poll-hz", type=float, default=30.0)
    parser.add_argument("--state-timeout-s", type=float, default=0.2)
    parser.add_argument("--max-state-age-s", type=float, default=0.25)
    parser.add_argument("--max-state-wait-s", type=float, default=1.0)
    parser.add_argument("--state-max-consecutive-misses", type=int, default=60)
    parser.add_argument("--state-retry-sleep-s", type=float, default=0.01)
    parser.add_argument("--max-camera-age-s", type=float, default=0.25)

    parser.add_argument(
        "--camera-backend", choices=("realsense", "ros2_zmq", "opencv", "none"), default="realsense"
    )
    parser.add_argument("--camera-name", default="l515")
    parser.add_argument("--image-zmq", default=DEFAULT_IMAGE_ZMQ_ENDPOINT)
    parser.add_argument("--camera-serial-or-name", default="Intel RealSense L515")
    parser.add_argument("--camera-width", type=int, default=L515_COLOR_WIDTH)
    parser.add_argument("--camera-height", type=int, default=L515_COLOR_HEIGHT)
    parser.add_argument("--camera-fps", type=int, default=L515_COLOR_FPS)
    parser.add_argument("--camera-warmup-s", type=int, default=1)
    parser.add_argument("--opencv-index-or-path", default="auto")
    parser.add_argument("--opencv-fourcc", default=None)
    parser.add_argument("--spnav-lib-path", default=None)
    parser.add_argument("--spnav-axis-scale", type=float, default=500.0)
    parser.add_argument("--deadband", type=float, default=0.05)
    parser.add_argument("--motion-timeout", type=float, default=0.2)
    parser.add_argument("--linear-scale", type=float, default=None)
    parser.add_argument("--angular-scale", type=float, default=None)
    parser.add_argument("--mirror", action="store_true")

    parser.add_argument("--no-video", dest="use_videos", action="store_false")
    parser.set_defaults(use_videos=True)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads-per-camera", type=int, default=4)
    parser.add_argument("--streaming-encoding", action="store_true")
    parser.add_argument("--encoder-queue-maxsize", type=int, default=30)
    parser.add_argument("--encoder-threads", type=int, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run-config", action="store_true")

    return parser


def build_camera_configs(args: argparse.Namespace):
    if args.camera_backend in {"none", "ros2_zmq"}:
        camera_shapes = {}
        if args.camera_backend == "ros2_zmq":
            camera_shapes = {args.camera_name: (args.camera_height, args.camera_width, 3)}
        return {}, camera_shapes
    if args.camera_backend == "opencv":
        from lerobot.cameras.opencv import OpenCVCameraConfig

        index_or_path, width, height, fourcc = _resolve_opencv_capture_settings(args)
        camera_config = OpenCVCameraConfig(
            index_or_path=index_or_path,
            fps=args.camera_fps,
            width=width,
            height=height,
            warmup_s=args.camera_warmup_s,
            fourcc=fourcc,
        )
        camera_shapes = {args.camera_name: (height, width, 3)}
        return {args.camera_name: camera_config}, camera_shapes

    from lerobot.cameras.realsense import RealSenseCameraConfig

    camera_config = RealSenseCameraConfig(
        serial_number_or_name=args.camera_serial_or_name,
        fps=args.camera_fps,
        width=args.camera_width,
        height=args.camera_height,
        warmup_s=args.camera_warmup_s,
        use_rgb=True,
        use_depth=False,
    )
    camera_shapes = {args.camera_name: (args.camera_height, args.camera_width, 3)}
    return {args.camera_name: camera_config}, camera_shapes


def _resolve_opencv_capture_settings(args: argparse.Namespace):
    width = int(args.camera_width)
    height = int(args.camera_height)
    fourcc = args.opencv_fourcc
    index_or_path = str(args.opencv_index_or_path)

    if _should_auto_select_l515_color(args, width, height):
        index_or_path = _find_l515_opencv_color_device()
        if fourcc is None:
            fourcc = L515_COLOR_FOURCC

    resolved_index: int | str = int(index_or_path) if index_or_path.isdigit() else index_or_path
    return resolved_index, width, height, fourcc


def _should_auto_select_l515_color(args: argparse.Namespace, width: int, height: int) -> bool:
    if args.camera_name != "l515":
        return False

    requested = str(args.opencv_index_or_path)
    if requested in {"auto", "l515"}:
        return True

    return requested == "/dev/video0" and (width, height) not in L515_VIDEO0_GREY_PROFILES


def _find_l515_opencv_color_device() -> str:
    detected = _find_v4l2_yuyv_color_device()
    if detected is not None:
        return detected
    return "/dev/video6"


def _find_v4l2_yuyv_color_device() -> str | None:
    if shutil.which("v4l2-ctl") is None:
        return None

    for device in sorted(Path("/dev").glob("video*"), key=_video_device_sort_key):
        try:
            result = subprocess.run(
                ["v4l2-ctl", f"--device={device}", "--list-formats-ext"],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue

        formats = result.stdout
        if "YUYV" not in formats:
            continue
        if any(f"{width}x{height}" in formats for width, height in L515_COLOR_PROFILES):
            return str(device)

    return None


def _video_device_sort_key(path: Path):
    suffix = path.name.removeprefix("video")
    return int(suffix) if suffix.isdigit() else 10_000


def build_cameras(args: argparse.Namespace):
    if args.camera_backend == "ros2_zmq":
        from hardware_test.cameras.ros2_image_bridge import ZmqRgbImageClient

        return {
            args.camera_name: ZmqRgbImageClient(
                args.image_zmq,
                max_age_ms=max(1, int(args.max_camera_age_s * 1000)),
            )
        }
    return None


def build_robot_config(args: argparse.Namespace) -> FrankaRobotConfig:
    camera_configs, camera_shapes = build_camera_configs(args)
    return FrankaRobotConfig(
        action_mode=args.action_mode,
        cartesian_action_units=args.cartesian_action_units,
        control_hz=float(args.fps),
        max_linear_velocity=args.max_linear_velocity,
        max_angular_velocity=args.max_angular_velocity,
        command_duration_ms=args.command_duration_ms,
        base_url=args.base_url,
        control_host=args.control_host,
        velocity_transport=args.velocity_transport,
        zmq_url=args.zmq_url,
        state_cache_enabled=True,
        state_poll_hz=args.state_poll_hz,
        state_timeout_s=args.state_timeout_s,
        max_state_age_s=args.max_state_age_s,
        cameras=camera_configs,
        camera_shapes=camera_shapes,
        camera_read_mode="latest",
        max_camera_age_s=args.max_camera_age_s,
    )


def build_teleop_config(args: argparse.Namespace) -> FrankaSpaceMouseTeleopConfig:
    fps = float(args.fps)
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    linear_scale = args.linear_scale if args.linear_scale is not None else args.max_linear_velocity / fps
    angular_scale = args.angular_scale if args.angular_scale is not None else args.max_angular_velocity / fps
    return FrankaSpaceMouseTeleopConfig(
        spnav_lib_path=args.spnav_lib_path,
        spnav_axis_scale=args.spnav_axis_scale,
        deadband=args.deadband,
        motion_timeout=args.motion_timeout,
        pose_scaler=(linear_scale, angular_scale),
        mirror=args.mirror,
    )


def install_signal_handlers(stop_event: Event) -> None:
    def _request_stop(signum, frame):  # noqa: ARG001
        stop_event.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)


def describe_config(args: argparse.Namespace, robot_config: FrankaRobotConfig) -> str:
    cameras = _describe_cameras(args, robot_config)
    target_floor = getattr(args, "target_floor", None)
    target_floor_config = f" target_floor={target_floor}" if target_floor is not None else ""
    return (
        f"repo_id={args.repo_id} root={args.root} fps={args.fps} episodes={args.num_episodes} "
        f"duration_s={args.duration_s} control_host={args.control_host} "
        f"velocity_transport={args.velocity_transport} camera_backend={args.camera_backend} cameras={cameras}"
        f"{target_floor_config}"
    )


def _describe_cameras(args: argparse.Namespace, robot_config: FrankaRobotConfig) -> str:
    if not robot_config.camera_shapes:
        return "none"

    camera_parts = []
    for name, shape in robot_config.camera_shapes.items():
        height, width, channels = shape
        source = args.camera_backend
        camera_config = robot_config.cameras.get(name)
        if camera_config is not None and hasattr(camera_config, "index_or_path"):
            source = str(camera_config.index_or_path)
        elif args.camera_backend == "ros2_zmq":
            source = args.image_zmq
        camera_parts.append(f"{name}={width}x{height}x{channels}@{args.camera_fps}Hz:{source}")
    return ", ".join(camera_parts)


def main(argv: list[str] | None = None, *, require_target_floor: bool = False) -> int:
    args = build_arg_parser(require_target_floor=require_target_floor).parse_args(argv)
    target_floor = getattr(args, "target_floor", None)
    robot_config = build_robot_config(args)
    teleop_config = build_teleop_config(args)

    print(describe_config(args, robot_config), flush=True)
    if args.dry_run_config:
        print("dry-run-config: no hardware connection attempted", flush=True)
        return 0

    stop_event = Event()
    install_signal_handlers(stop_event)

    robot = FrankaRobot(robot_config, cameras=build_cameras(args))
    teleop = FrankaSpaceMouseTeleop(teleop_config)
    dataset = None

    try:
        robot.connect()
        teleop.connect()

        dataset = create_lerobot_dataset(
            repo_id=args.repo_id,
            root=args.root,
            fps=args.fps,
            robot=robot,
            teleop=teleop,
            task=args.task,
            use_videos=args.use_videos,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads_per_camera=args.image_writer_threads_per_camera,
            streaming_encoding=args.streaming_encoding,
            encoder_queue_maxsize=args.encoder_queue_maxsize,
            encoder_threads=args.encoder_threads,
            include_environment_state=target_floor is not None,
        )

        for episode_idx in range(args.num_episodes):
            if stop_event.is_set():
                break
            environment_state = encode_target_floor(target_floor) if target_floor is not None else None
            print(f"episode {episode_idx}: recording", flush=True)
            try:
                frames = record_lerobot_episode(
                    robot=robot,
                    teleop=teleop,
                    dataset=dataset,
                    fps=args.fps,
                    duration_s=args.duration_s,
                    task=args.task,
                    stop_event=stop_event,
                    max_consecutive_state_misses=args.state_max_consecutive_misses,
                    max_state_wait_s=args.max_state_wait_s,
                    state_retry_sleep_s=args.state_retry_sleep_s,
                    environment_state=environment_state,
                    tolerate_robot_faults=target_floor is not None and args.action_mode == "delta_ee_pose",
                )
                if frames == 0:
                    dataset.clear_episode_buffer(delete_images=True)
                    print(f"episode {episode_idx}: no frames recorded; discarded", flush=True)
                    break
                dataset.save_episode()
                print(f"episode {episode_idx}: saved {frames} frames", flush=True)
            except BaseException:
                if dataset is not None and dataset.has_pending_frames():
                    dataset.clear_episode_buffer(delete_images=True)
                    print(f"episode {episode_idx}: discarded pending frames after error", flush=True)
                raise

            if args.reset_time_s > 0 and episode_idx + 1 < args.num_episodes:
                print(f"reset window: {args.reset_time_s:.1f}s", flush=True)
                reset_start = time.perf_counter()
                while time.perf_counter() - reset_start < args.reset_time_s:
                    if stop_event.is_set():
                        break
                    time.sleep(0.05)
    finally:
        if dataset is not None:
            dataset.finalize()
        with _SuppressDisconnectErrors():
            teleop.disconnect()
        with _SuppressDisconnectErrors():
            robot.disconnect()

    if args.push_to_hub and dataset is not None:
        dataset.push_to_hub(private=args.private)

    print(f"dataset root: {args.root}", flush=True)
    return 0


class _SuppressDisconnectErrors:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return True


if __name__ == "__main__":
    raise SystemExit(main())
