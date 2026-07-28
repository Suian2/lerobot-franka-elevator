
import time
import sys
from pathlib import Path

# 添加路径（如果脚本不在根目录运行，可保留）
REPO_ROOT = Path(__file__).resolve().parents[2]  # 根据实际层级调整
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware_test.franka.franka_robot import FrankaRobot, FrankaRobotConfig

# 使用与录制相同的配置，请根据实际参数调整
config = FrankaRobotConfig(
    control_host="192.168.1.11",   # 改为实际 IP，或使用 get_control_host() 导入
    base_url=None,
    velocity_transport="http",      # 或 "zmq"
    state_poll_hz=100,              # 轮询频率高一些
    state_timeout_s=0.2,
    max_state_age_s=0.25,
    # 相机配置可忽略，因为我们只测状态
)
robot = FrankaRobot(config)
robot.connect()

# 预热（丢弃前几个）
for _ in range(5):
    robot.get_observation()

N = 100
times = []
for _ in range(N):
    t0 = time.perf_counter()
    obs = robot.get_observation()   # 或 get_observation_sample()
    times.append(time.perf_counter() - t0)

avg_ms = sum(times) / N * 1000
print(f"Average latency: {avg_ms:.2f} ms")
print(f"Min: {min(times)*1000:.2f} ms, Max: {max(times)*1000:.2f} ms")
robot.disconnect()