# ROS2 Image Bridge Reliable QoS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Python ROS2-to-ZMQ image bridge receive the live L515 RELIABLE image publisher consistently while retaining latest-frame semantics and a clean rollback point.

**Architecture:** Keep the existing bridge threads, queue depth, durability, image encoding, and ZMQ protocol. Add a behavioral setup test with fake ROS2/ZMQ modules, then change only the subscription reliability enum and validate it against the live 30 Hz camera diagnostics.

**Tech Stack:** Python 3.10/3.12, pytest, rclpy QoS, pyzmq, Ruff, ROS2 Humble, RealSense L515.

---

### Task 1: Lock the bridge subscription QoS behavior

**Files:**
- Modify: `hardware_test/cameras/test_ros2_image_bridge.py`
- Test: `hardware_test/cameras/test_ros2_image_bridge.py`

- [ ] **Step 1: Add a behavioral setup test**

Add `ModuleType` to the imports and add this test. It stops the bridge before
calling `run()` so setup executes without entering the spin loop, then observes
the real QoS passed to `create_subscription`:

```python
from types import ModuleType


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
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
source /home/yanrihong/rs_modes/env_lerobot_sdk.sh
pytest -q hardware_test/cameras/test_ros2_image_bridge.py::test_ros2_bridge_uses_reliable_latest_frame_qos
```

Expected: one assertion failure showing the actual reliability is
`BEST_EFFORT`, proving the test catches the live configuration defect.

### Task 2: Apply the minimal QoS correction

**Files:**
- Modify: `hardware_test/cameras/ros2_image_bridge.py:318`
- Test: `hardware_test/cameras/test_ros2_image_bridge.py`

- [ ] **Step 1: Change only the reliability policy**

Replace:

```python
reliability=ReliabilityPolicy.BEST_EFFORT,
```

with:

```python
reliability=ReliabilityPolicy.RELIABLE,
```

- [ ] **Step 2: Verify GREEN and run the focused suite**

Run:

```bash
source /home/yanrihong/rs_modes/env_lerobot_sdk.sh
pytest -q hardware_test/cameras/test_ros2_image_bridge.py
ruff check hardware_test/cameras/ros2_image_bridge.py hardware_test/cameras/test_ros2_image_bridge.py
```

Expected: all bridge tests pass and Ruff reports no errors.

### Task 3: Validate on the live camera and archive the result

**Files:**
- No production file changes unless rollback is required.

- [ ] **Step 1: Confirm the existing camera launch is healthy**

Check that `realsense2_camera_node` is running and its current launch log has no
new `Frames didn't arrived`, `No such device`, `resource busy`, or `ERROR`
entries.

- [ ] **Step 2: Measure the modified bridge**

Run the bridge for 26 seconds with `--stats-interval-s 5` against
`/l515/color/image_raw` and `tcp://127.0.0.1:5557`. Expected: sustained windows
near 27-30 Hz and no regression below the previous 13-24 Hz range.

- [ ] **Step 3: Keep or roll back**

If the reliable bridge is not better, restore both changed files from baseline
commit `6f38b6c7` and rerun the focused tests. If it improves, commit only the
bridge source and bridge test with a Lore-format message recording measured
rates and the baseline rollback commit.
