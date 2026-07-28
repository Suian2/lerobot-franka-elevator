from __future__ import annotations

import numpy as np

from lerobot.utils.constants import OBS_ENV_STATE

FLOOR_CONDITION_KEY = OBS_ENV_STATE
NUM_ELEVATOR_FLOORS = 5
TRAINED_ROLLOUT_FLOORS = (1, 4, 5)
FLOOR_CONDITION_FEATURE = {"dtype": "float32", "shape": (5,), "names": None}


def encode_target_floor(floor: int) -> np.ndarray:
    if isinstance(floor, bool) or not isinstance(floor, int):
        raise TypeError("floor must be an integer")
    if not 1 <= floor <= NUM_ELEVATOR_FLOORS:
        raise ValueError(f"floor must be between 1 and {NUM_ELEVATOR_FLOORS}")

    condition = np.zeros(NUM_ELEVATOR_FLOORS, dtype=np.float32)
    condition[floor - 1] = 1.0
    return condition
