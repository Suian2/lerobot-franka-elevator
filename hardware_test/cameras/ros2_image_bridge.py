from __future__ import annotations

import argparse
import json
import signal
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_IMAGE_ZMQ_ENDPOINT = "tcp://127.0.0.1:5557"
DEFAULT_L515_COLOR_TOPIC = "/l515/color/image_raw"


@dataclass(frozen=True)
class ZmqImageSample:
    image: np.ndarray
    seq: int
    height: int
    width: int
    encoding: str
    source_timestamp_s: float | None
    bridge_monotonic_s: float
    received_monotonic_s: float
    image_age_ms: float
    dropped_frame_count: int


class ZmqRgbImageClient:
    """Latest-frame ZMQ image client for LeRobot/conda processes.

    This class intentionally does not import ROS2 modules. It consumes RGB
    frames produced by this file's ROS2 bridge process.
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_IMAGE_ZMQ_ENDPOINT,
        *,
        max_age_ms: int = 250,
        startup_timeout_ms: int = 2000,
    ):
        self.endpoint = endpoint
        self.max_age_ms = int(max_age_ms)
        self.startup_timeout_ms = int(startup_timeout_ms)
        self._socket = None
        self._last_seq: int | None = None
        self._dropped_frame_count = 0
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def dropped_frame_count(self) -> int:
        return self._dropped_frame_count

    def connect(self) -> None:
        if self._is_connected:
            return
        try:
            import zmq
        except Exception as exc:
            raise ImportError("ZmqRgbImageClient requires pyzmq in the LeRobot environment.") from exc

        context = zmq.Context.instance()
        self._socket = context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(self.endpoint)
        self._is_connected = True

    def disconnect(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        self._is_connected = False

    def read(self) -> np.ndarray:
        return self.latest().image

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        return self.latest(timeout_ms=int(timeout_ms)).image

    def read_latest(self, max_age_ms: int = 250) -> np.ndarray:
        return self.latest(max_age_ms=max_age_ms).image

    def latest(self, *, max_age_ms: int | None = None, timeout_ms: int = 50) -> ZmqImageSample:
        if not self._is_connected:
            self.connect()
        if self._socket is None:
            raise RuntimeError("ZMQ image client is not connected")

        allowed_age_ms = self.max_age_ms if max_age_ms is None else int(max_age_ms)
        effective_timeout_ms = max(0, int(timeout_ms))
        if self._last_seq is None:
            effective_timeout_ms = max(effective_timeout_ms, self.startup_timeout_ms)
        deadline_s = time.monotonic() + effective_timeout_ms / 1000.0
        stale_error: TimeoutError | None = None

        while True:
            remaining_ms = max(0, int((deadline_s - time.monotonic()) * 1000))
            try:
                message = self._recv_latest_message(timeout_ms=remaining_ms)
            except TimeoutError:
                if stale_error is not None:
                    raise stale_error from None
                raise

            received_monotonic_s = time.monotonic()
            metadata, payload = decode_rgb_frame_message(message)
            frame = decode_rgb_frame_payload(metadata, payload)
            seq = int(metadata["seq"])
            if self._last_seq is not None and seq > self._last_seq + 1:
                self._dropped_frame_count += seq - self._last_seq - 1
            self._last_seq = seq

            bridge_monotonic_s = float(metadata["local_monotonic_s"])
            image_age_ms = (received_monotonic_s - bridge_monotonic_s) * 1000.0
            if image_age_ms <= allowed_age_ms:
                return ZmqImageSample(
                    image=frame,
                    seq=seq,
                    height=int(metadata["height"]),
                    width=int(metadata["width"]),
                    encoding=str(metadata["encoding"]),
                    source_timestamp_s=metadata.get("source_timestamp_s"),
                    bridge_monotonic_s=bridge_monotonic_s,
                    received_monotonic_s=received_monotonic_s,
                    image_age_ms=image_age_ms,
                    dropped_frame_count=self._dropped_frame_count,
                )

            stale_error = TimeoutError(
                f"latest ZMQ image is too old: {image_age_ms:.1f}ms "
                f"(max allowed: {allowed_age_ms}ms)"
            )
            if time.monotonic() >= deadline_s:
                raise stale_error

    def _recv_latest_message(self, *, timeout_ms: int) -> bytes:
        try:
            import zmq
        except Exception as exc:
            raise ImportError("ZmqRgbImageClient requires pyzmq in the LeRobot environment.") from exc

        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        events = dict(poller.poll(timeout_ms))
        if self._socket not in events:
            raise TimeoutError(f"timed out waiting for ZMQ image on {self.endpoint} after {timeout_ms}ms")

        latest = self._socket.recv(flags=zmq.NOBLOCK)
        while True:
            try:
                latest = self._socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                return latest


def encode_rgb_frame_message(
    frame: np.ndarray,
    *,
    seq: int,
    source_timestamp_s: float | None,
    local_monotonic_s: float,
    original_encoding: str,
) -> bytes:
    image = np.asarray(frame)
    if image.dtype != np.uint8:
        raise TypeError(f"expected uint8 RGB frame, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected H,W,3 RGB frame, got shape={image.shape}")
    image = np.ascontiguousarray(image)
    height, width, _ = image.shape
    metadata = {
        "seq": int(seq),
        "height": int(height),
        "width": int(width),
        "encoding": "rgb8",
        "dtype": "uint8",
        "channels": 3,
        "source_timestamp_s": source_timestamp_s,
        "local_monotonic_s": float(local_monotonic_s),
        "original_encoding": str(original_encoding),
    }
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    return struct.pack("!I", len(header)) + header + image.tobytes()


def decode_rgb_frame_message(message: bytes) -> tuple[dict[str, Any], bytes]:
    if len(message) < 4:
        raise ValueError(f"ZMQ image message too short: {len(message)} bytes")
    header_len = struct.unpack("!I", message[:4])[0]
    header_end = 4 + header_len
    if header_end > len(message):
        raise ValueError(f"ZMQ image header length {header_len} exceeds message size {len(message)}")
    metadata = json.loads(message[4:header_end].decode("utf-8"))
    return metadata, message[header_end:]


def decode_rgb_frame_payload(metadata: dict[str, Any], payload: bytes) -> np.ndarray:
    if metadata.get("dtype") != "uint8" or metadata.get("encoding") != "rgb8":
        raise ValueError(f"unsupported image metadata: {metadata}")
    height = int(metadata["height"])
    width = int(metadata["width"])
    channels = int(metadata.get("channels", 3))
    if channels != 3:
        raise ValueError(f"expected 3 channels, got {channels}")
    expected_bytes = height * width * channels
    if len(payload) != expected_bytes:
        raise ValueError(f"expected {expected_bytes} image bytes, got {len(payload)}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(height, width, channels).copy()


@dataclass
class _RawRosImage:
    payload: bytes
    source_timestamp_s: float | None
    local_monotonic_s: float
    encoding: str
    height: int
    width: int
    step: int


class _LatestSlot:
    def __init__(self):
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._latest: _RawRosImage | None = None
        self._received = 0
        self._taken = 0

    @property
    def received_count(self) -> int:
        with self._lock:
            return self._received

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return max(0, self._received - self._taken - 1)

    def put(self, image: _RawRosImage) -> None:
        with self._lock:
            self._latest = image
            self._received += 1
            self._event.set()

    def take(self, timeout_s: float) -> _RawRosImage | None:
        if not self._event.wait(timeout=timeout_s):
            return None
        with self._lock:
            image = self._latest
            self._latest = None
            self._taken = self._received
            self._event.clear()
            return image


class Ros2ImageZmqBridge:
    """ROS2 subscriber + ZMQ publisher for latest-only RGB image streaming."""

    def __init__(
        self,
        *,
        topic: str = DEFAULT_L515_COLOR_TOPIC,
        endpoint: str = DEFAULT_IMAGE_ZMQ_ENDPOINT,
        stats_interval_s: float = 1.0,
    ):
        if topic.endswith("/compressed"):
            raise ValueError(
                "Compressed ROS2 image topics are intentionally unsupported in this ROS2+ZMQ recording path. "
                "Use a raw sensor_msgs/Image topic such as /l515/l515/color/image_raw."
            )
        self.topic = topic
        self.endpoint = endpoint
        self.stats_interval_s = float(stats_interval_s)
        self._slot = _LatestSlot()
        self._stop = threading.Event()
        self._seq = 0
        self._published = 0

    def run(self) -> None:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import Image
            import zmq
        except Exception as exc:
            raise ImportError(
                "Ros2ImageZmqBridge must run in a system ROS2 Python environment with "
                "rclpy, sensor_msgs, numpy, and pyzmq available."
            ) from exc

        context = zmq.Context.instance()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDHWM, 1)
        socket.bind(self.endpoint)

        if not rclpy.ok():
            rclpy.init(args=None)

        qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        node = rclpy.create_node("lerobot_ros2_image_zmq_bridge")
        node.create_subscription(Image, self.topic, self._image_callback, qos)
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        publisher = threading.Thread(target=self._publish_loop, args=(socket,), daemon=True)
        publisher.start()
        try:
            while not self._stop.is_set():
                executor.spin_once(timeout_sec=0.1)
        finally:
            self._stop.set()
            publisher.join(timeout=1.0)
            executor.shutdown()
            node.destroy_node()
            socket.close(linger=0)

    def stop(self) -> None:
        self._stop.set()

    def _image_callback(self, msg: Any) -> None:
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        source_timestamp_s = None
        if stamp is not None:
            source_timestamp_s = float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9
        payload = bytes(msg.data)
        encoding = str(getattr(msg, "encoding", ""))
        height = int(getattr(msg, "height"))
        width = int(getattr(msg, "width"))
        step = int(getattr(msg, "step", width * 3))
        self._slot.put(
            _RawRosImage(
                payload=payload,
                source_timestamp_s=source_timestamp_s,
                local_monotonic_s=time.monotonic(),
                encoding=encoding,
                height=height,
                width=width,
                step=step,
            )
        )

    def _publish_loop(self, socket: Any) -> None:
        try:
            import zmq
        except Exception as exc:
            raise ImportError("Ros2ImageZmqBridge requires pyzmq for publishing.") from exc

        last_stats_t = time.monotonic()
        last_published = 0
        while not self._stop.is_set():
            raw = self._slot.take(timeout_s=0.1)
            if raw is None:
                continue
            try:
                frame = self._raw_ros_image_to_rgb(raw)
            except Exception as exc:
                print(f"image bridge decode error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                continue
            self._seq += 1
            try:
                socket.send(
                    encode_rgb_frame_message(
                        frame,
                        seq=self._seq,
                        source_timestamp_s=raw.source_timestamp_s,
                        local_monotonic_s=raw.local_monotonic_s,
                        original_encoding=raw.encoding,
                    ),
                    flags=zmq.NOBLOCK,
                )
            except zmq.Again:
                continue
            self._published += 1

            now = time.monotonic()
            if now - last_stats_t >= self.stats_interval_s:
                fps = (self._published - last_published) / max(now - last_stats_t, 1e-9)
                age_ms = (now - raw.local_monotonic_s) * 1000.0
                print(
                    f"bridge topic={self.topic} endpoint={self.endpoint} fps={fps:.1f} "
                    f"seq={self._seq} dropped={self._slot.dropped_count} latest_age_ms={age_ms:.1f}",
                    flush=True,
                )
                last_stats_t = now
                last_published = self._published

    def _raw_ros_image_to_rgb(self, raw: _RawRosImage) -> np.ndarray:
        encoding = raw.encoding.lower()
        if encoding in {"rgb8", "bgr8", "8uc3"}:
            channels = 3
        elif encoding in {"rgba8", "bgra8", "8uc4"}:
            channels = 4
        else:
            raise ValueError(f"unsupported raw ROS image encoding: {raw.encoding}")

        min_step = raw.width * channels
        if raw.step < min_step:
            raise ValueError(f"ROS image step {raw.step} is too small for {raw.width}x{channels}")
        expected_bytes = raw.height * raw.step
        if len(raw.payload) < expected_bytes:
            raise ValueError(f"expected at least {expected_bytes} image bytes, got {len(raw.payload)}")

        rows = np.frombuffer(raw.payload, dtype=np.uint8, count=expected_bytes).reshape(raw.height, raw.step)
        frame = rows[:, :min_step].reshape(raw.height, raw.width, channels)
        if encoding in {"bgr8", "bgra8"}:
            frame = frame[:, :, 2::-1]
        else:
            frame = frame[:, :, :3]
        return np.ascontiguousarray(frame, dtype=np.uint8)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge a ROS2 image topic to latest-only ZMQ RGB frames.")
    parser.add_argument("--topic", default=DEFAULT_L515_COLOR_TOPIC)
    parser.add_argument("--endpoint", default=DEFAULT_IMAGE_ZMQ_ENDPOINT)
    parser.add_argument("--stats-interval-s", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    bridge = Ros2ImageZmqBridge(
        topic=args.topic,
        endpoint=args.endpoint,
        stats_interval_s=args.stats_interval_s,
    )

    def _stop(signum, frame):  # noqa: ARG001
        bridge.stop()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    bridge.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
