"""Franka hardware-test adapters for LeRobot.

These modules are intentionally not wired into LeRobot's global factories yet.
Importing this package makes the local config classes available for manual use
and tests.
"""

from .franka_robot import FrankaRobot, FrankaRobotConfig
from .franka_spacemouse_teleop import FrankaSpaceMouseTeleop, FrankaSpaceMouseTeleopConfig
from .record_lerobot_dataset import build_lerobot_features, create_lerobot_dataset, make_lerobot_frame
from .state_cache import FrankaStateCache, FrankaStateSnapshot, StaleFrankaStateError

__all__ = [
    "FrankaRobot",
    "FrankaRobotConfig",
    "FrankaSpaceMouseTeleop",
    "FrankaSpaceMouseTeleopConfig",
    "FrankaStateCache",
    "FrankaStateSnapshot",
    "StaleFrankaStateError",
    "build_lerobot_features",
    "create_lerobot_dataset",
    "make_lerobot_frame",
]
