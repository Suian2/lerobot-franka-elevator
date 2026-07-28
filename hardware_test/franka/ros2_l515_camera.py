from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np


class Ros2L515Camera:
    """LeRobot-style camera wrapper for the existing ROS2 L515 image topic."""

    def __init__(
        self,
        *,
        color_topic: str = "/camera/color/image_raw",
        compressed: bool | None = None,
        qos: str = "sensor_data",
        width: int = 640,
        height: int = 480,
        warmup_s: float = 1.0,
    ):
        self.color_topic = color_topic
        self.compressed = color_topic.endswith("/compressed") if compressed is None else bool(compressed)
        self.qos = qos
        self.width = int(width)
        self.height = int(height)
        self.warmup_s = float(warmup_s)
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._stop = threading.Event()
        self._latest_frame: np.ndarray | None = None
        self._latest_timestamp: float | None = None
        self._node = None
        self._executor = None
        self._thread: threading.Thread | None = None
        self._bridge = None
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, warmup: bool = True) -> None:
        if self._is_connected:
            return
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from sensor_msgs.msg import CompressedImage, Image
        except Exception as exc:
            raise ImportError(
                "ROS2 camera backend requires rclpy and sensor_msgs. "
                "Run this script from a shell that sourced the ROS2 setup file, e.g. "
                "`source /opt/ros/humble/setup.bash`."
            ) from exc

        if not self.compressed:
            try:
                from cv_bridge import CvBridge
            except Exception as exc:
                raise ImportError("ROS2 raw Image backend requires cv_bridge.") from exc
            self._bridge = CvBridge()

        if not rclpy.ok():
            rclpy.init(args=None)

        node_name = f"lerobot_franka_l515_{id(self) & 0xffff:x}"
        self._node = rclpy.create_node(node_name)
        msg_type = CompressedImage if self.compressed else Image
        self._node.create_subscription(msg_type, self.color_topic, self._image_callback, _make_qos(self.qos))
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, name=node_name, daemon=True)
        self._thread.start()
        self._is_connected = True

        if warmup:
            self.async_read(timeout_ms=max(1, int(self.warmup_s * 1000)))

    def read(self) -> np.ndarray:
        return self.async_read(timeout_ms=1000)

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        if not self._new_frame.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for ROS2 image on {self.color_topic} after {timeout_ms}ms")
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            self._new_frame.clear()
        if frame is None:
            raise RuntimeError(f"ROS2 image event fired but no frame is available for {self.color_topic}")
        return frame

    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        with self._lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
            timestamp = self._latest_timestamp
        if frame is None or timestamp is None:
            raise RuntimeError(f"No ROS2 image has been received yet on {self.color_topic}")
        age_ms = (time.monotonic() - timestamp) * 1000.0
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"latest ROS2 image is too old on {self.color_topic}: {age_ms:.1f}ms "
                f"(max allowed: {max_age_ms}ms)"
            )
        return frame

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            if self._executor is not None:
                self._executor.shutdown()
        except Exception:
            pass
        try:
            if self._node is not None:
                self._node.destroy_node()
        except Exception:
            pass
        self._executor = None
        self._node = None
        self._is_connected = False

    def _spin(self) -> None:
        while not self._stop.is_set():
            if self._executor is not None:
                self._executor.spin_once(timeout_sec=0.1)

    def _image_callback(self, msg: Any) -> None:
        try:
            if self.compressed:
                import cv2

                encoded = np.frombuffer(msg.data, dtype=np.uint8)
                frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError("cv2.imdecode returned None")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            frame = np.ascontiguousarray(frame)
            if frame.shape[:2] != (self.height, self.width):
                import cv2

                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        except Exception:
            return

        with self._lock:
            self._latest_frame = frame.copy()
            self._latest_timestamp = time.monotonic()
            self._new_frame.set()


def _make_qos(mode: str):
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.qos import qos_profile_sensor_data

    mode = mode.strip().lower()
    if mode in {"sensor", "sensor_data", "best_effort"}:
        return qos_profile_sensor_data
    if mode in {"reliable_transient", "reliable_transient_local", "transient_local"}:
        return QoSProfile(
            depth=10,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
    if mode == "reliable":
        return QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE)
    raise ValueError(f"Unsupported ROS2 image QoS mode: {mode}")
