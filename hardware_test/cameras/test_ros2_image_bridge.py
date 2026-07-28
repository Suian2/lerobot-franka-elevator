from __future__ import annotations

import sys
import threading
import time
from types import ModuleType

import numpy as np
import pytest

from hardware_test.cameras.ros2_image_bridge import (
    Ros2ImageZmqBridge,
    ZmqRgbImageClient,
    _RawRosImage,
    encode_rgb_frame_message,
)


def test_zmq_image_client_does_not_import_ros2_modules():
    assert "rclpy" not in sys.modules
    assert "cv_bridge" not in sys.modules
    assert "cv2" not in sys.modules


def test_ros2_bridge_rejects_compressed_topics_to_avoid_opencv_path():
    with pytest.raises(ValueError, match="Compressed ROS2 image topics"):
        Ros2ImageZmqBridge(topic="/l515/color/image_raw/compressed")


def test_ros2_bridge_uses_reliable_latest_frame_qos(monkeypatch):
    captured = {}

    class FakeSocket:
        def setsockopt(self, *args):
            pass

        def bind(self, endpoint):
            captured["endpoint"] = endpoint

        def close(self, *, linger):
            captured["linger"] = linger

    class FakeContext:
        def socket(self, socket_type):
            captured["socket_type"] = socket_type
            return FakeSocket()

    class FakeContextFactory:
        @staticmethod
        def instance():
            return FakeContext()

    class FakeQoSProfile:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

    class FakeHistoryPolicy:
        KEEP_LAST = object()

    class FakeReliabilityPolicy:
        BEST_EFFORT = object()
        RELIABLE = object()

    class FakeDurabilityPolicy:
        VOLATILE = object()

    class FakeNode:
        def create_subscription(self, message_type, topic, callback, qos):
            captured["message_type"] = message_type
            captured["topic"] = topic
            captured["callback"] = callback
            captured["qos"] = qos

        def destroy_node(self):
            captured["node_destroyed"] = True

    class FakeExecutor:
        def add_node(self, node):
            captured["node"] = node

        def shutdown(self):
            captured["executor_shutdown"] = True

    rclpy = ModuleType("rclpy")
    rclpy.__path__ = []
    rclpy.ok = lambda: True
    rclpy.create_node = lambda name: FakeNode()
    executors = ModuleType("rclpy.executors")
    executors.SingleThreadedExecutor = FakeExecutor
    qos_module = ModuleType("rclpy.qos")
    qos_module.QoSProfile = FakeQoSProfile
    qos_module.HistoryPolicy = FakeHistoryPolicy
    qos_module.ReliabilityPolicy = FakeReliabilityPolicy
    qos_module.DurabilityPolicy = FakeDurabilityPolicy
    sensor_msgs = ModuleType("sensor_msgs")
    sensor_msgs.__path__ = []
    sensor_msgs_msg = ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.Image = object()
    zmq = ModuleType("zmq")
    zmq.Context = FakeContextFactory
    zmq.PUB = object()
    zmq.LINGER = object()
    zmq.SNDHWM = object()

    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.executors", executors)
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos_module)
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs_msg)
    monkeypatch.setitem(sys.modules, "zmq", zmq)

    bridge = Ros2ImageZmqBridge()
    bridge.stop()
    bridge.run()

    captured_qos = captured["qos"]
    assert captured["topic"] == "/l515/color/image_raw"
    assert captured_qos.depth == 1
    assert captured_qos.history is FakeHistoryPolicy.KEEP_LAST
    assert captured_qos.reliability is FakeReliabilityPolicy.RELIABLE
    assert captured_qos.durability is FakeDurabilityPolicy.VOLATILE


def test_ros2_bridge_converts_padded_bgr8_raw_image_without_cv2():
    bridge = Ros2ImageZmqBridge()
    bgr_payload = bytes(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            99,
            99,
            10,
            20,
            30,
            40,
            50,
            60,
            88,
            88,
        ]
    )
    raw = _RawRosImage(
        payload=bgr_payload,
        source_timestamp_s=1.0,
        local_monotonic_s=time.monotonic(),
        encoding="bgr8",
        height=2,
        width=2,
        step=8,
    )

    frame = bridge._raw_ros_image_to_rgb(raw)

    assert frame.shape == (2, 2, 3)
    np.testing.assert_array_equal(frame[0, 0], np.array([3, 2, 1], dtype=np.uint8))
    np.testing.assert_array_equal(frame[1, 1], np.array([60, 50, 40], dtype=np.uint8))
    assert "cv2" not in sys.modules


def test_zmq_image_client_reads_latest_rgb_frame_and_counts_drops():
    zmq = pytest.importorskip("zmq")

    context = zmq.Context.instance()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    client = ZmqRgbImageClient(f"tcp://127.0.0.1:{port}", max_age_ms=500)
    client.connect()
    time.sleep(0.1)

    first = np.zeros((2, 3, 3), dtype=np.uint8)
    second = np.full((2, 3, 3), 7, dtype=np.uint8)
    publisher.send(
        encode_rgb_frame_message(
            first,
            seq=1,
            source_timestamp_s=10.0,
            local_monotonic_s=time.monotonic(),
            original_encoding="rgb8",
        )
    )
    first_sample = client.latest(timeout_ms=500)

    publisher.send(
        encode_rgb_frame_message(
            second,
            seq=3,
            source_timestamp_s=10.1,
            local_monotonic_s=time.monotonic(),
            original_encoding="rgb8",
        )
    )

    sample = client.latest(timeout_ms=500)

    np.testing.assert_array_equal(first_sample.image, first)
    assert first_sample.seq == 1
    np.testing.assert_array_equal(sample.image, second)
    assert sample.seq == 3
    assert sample.height == 2
    assert sample.width == 3
    assert sample.encoding == "rgb8"
    assert sample.dropped_frame_count == 1
    assert sample.image_age_ms < 500

    client.disconnect()
    publisher.close(linger=0)


def test_zmq_image_client_rejects_stale_latest_frame():
    zmq = pytest.importorskip("zmq")

    context = zmq.Context.instance()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    client = ZmqRgbImageClient(f"tcp://127.0.0.1:{port}", max_age_ms=10)
    client.connect()
    time.sleep(0.1)

    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    publisher.send(
        encode_rgb_frame_message(
            frame,
            seq=1,
            source_timestamp_s=10.0,
            local_monotonic_s=time.monotonic() - 1.0,
            original_encoding="rgb8",
        )
    )

    with pytest.raises(TimeoutError, match="too old"):
        client.latest(timeout_ms=500)

    client.disconnect()
    publisher.close(linger=0)


def test_zmq_image_client_waits_for_fresh_frame_after_stale_startup_sample():
    zmq = pytest.importorskip("zmq")

    context = zmq.Context.instance()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    client = ZmqRgbImageClient(f"tcp://127.0.0.1:{port}", max_age_ms=100)
    client.connect()
    time.sleep(0.1)

    stale = np.zeros((2, 3, 3), dtype=np.uint8)
    fresh = np.full((2, 3, 3), 9, dtype=np.uint8)
    publisher.send(
        encode_rgb_frame_message(
            stale,
            seq=1,
            source_timestamp_s=10.0,
            local_monotonic_s=time.monotonic() - 1.0,
            original_encoding="rgb8",
        )
    )

    def publish_fresh() -> None:
        time.sleep(0.05)
        publisher.send(
            encode_rgb_frame_message(
                fresh,
                seq=2,
                source_timestamp_s=10.1,
                local_monotonic_s=time.monotonic(),
                original_encoding="rgb8",
            )
        )

    thread = threading.Thread(target=publish_fresh)
    thread.start()
    try:
        sample = client.latest(max_age_ms=100, timeout_ms=500)
    finally:
        thread.join(timeout=1.0)
        client.disconnect()
        publisher.close(linger=0)

    np.testing.assert_array_equal(sample.image, fresh)
    assert sample.seq == 2
    assert sample.image_age_ms < 100


def test_zmq_image_client_uses_longer_timeout_only_for_first_frame():
    client = ZmqRgbImageClient("tcp://127.0.0.1:1", max_age_ms=250)
    client._is_connected = True
    client._socket = object()
    timeout_calls = []

    def recv_message(*, timeout_ms: int) -> bytes:
        timeout_calls.append(timeout_ms)
        return encode_rgb_frame_message(
            np.zeros((2, 3, 3), dtype=np.uint8),
            seq=len(timeout_calls),
            source_timestamp_s=None,
            local_monotonic_s=time.monotonic(),
            original_encoding="rgb8",
        )

    client._recv_latest_message = recv_message

    client.latest(timeout_ms=50)
    client.latest(timeout_ms=50)

    assert 1900 <= timeout_calls[0] <= 2000
    assert 1 <= timeout_calls[1] <= 50
