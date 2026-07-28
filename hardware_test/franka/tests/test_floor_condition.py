import numpy as np
import pytest

from hardware_test.franka.floor_condition import (
    FLOOR_CONDITION_FEATURE,
    FLOOR_CONDITION_KEY,
    NUM_ELEVATOR_FLOORS,
    TRAINED_ROLLOUT_FLOORS,
    encode_target_floor,
)
from lerobot.utils.constants import OBS_ENV_STATE


def test_floor_condition_contract_uses_canonical_environment_state():
    assert FLOOR_CONDITION_KEY == OBS_ENV_STATE
    assert NUM_ELEVATOR_FLOORS == 5
    assert TRAINED_ROLLOUT_FLOORS == (1, 4, 5)
    assert FLOOR_CONDITION_FEATURE == {"dtype": "float32", "shape": (5,), "names": None}


@pytest.mark.parametrize(
    ("floor", "expected"),
    [
        (1, [1, 0, 0, 0, 0]),
        (2, [0, 1, 0, 0, 0]),
        (3, [0, 0, 1, 0, 0]),
        (4, [0, 0, 0, 1, 0]),
        (5, [0, 0, 0, 0, 1]),
    ],
)
def test_encode_target_floor_returns_exact_float32_one_hot(floor, expected):
    encoded = encode_target_floor(floor)

    np.testing.assert_array_equal(encoded, np.asarray(expected, dtype=np.float32))
    assert encoded.shape == (5,)
    assert encoded.dtype == np.float32


def test_encode_target_floor_returns_a_fresh_array_per_call():
    first = encode_target_floor(1)
    second = encode_target_floor(1)

    assert first is not second
    first[0] = 0
    np.testing.assert_array_equal(second, np.asarray([1, 0, 0, 0, 0], dtype=np.float32))


@pytest.mark.parametrize("floor", [0, 6, True, 1.0, "1"])
def test_encode_target_floor_rejects_invalid_values(floor):
    with pytest.raises((TypeError, ValueError)):
        encode_target_floor(floor)
