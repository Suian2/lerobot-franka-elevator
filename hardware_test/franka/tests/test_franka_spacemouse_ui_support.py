from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import hardware_test.franka.franka_spacemouse_teleop as spacemouse
from hardware_test.franka.franka_spacemouse_teleop import (
    SPNAV_EVENT_ANY,
    SPNAV_EVENT_BUTTON,
    SPNAV_EVENT_MOTION,
    FrankaSpaceMouseTeleop,
    FrankaSpaceMouseTeleopConfig,
    SpacenavReader,
    _SpnavLibrary,
)


class FakeSpnavLibrary:
    def __init__(self, events=()):
        self.events = list(events)
        self.remove_events_calls = 0
        self.closed = False

    def open(self) -> None:
        pass

    def remove_events(self) -> None:
        self.remove_events_calls += 1
        self.events.clear()

    def poll_event(self):
        if not self.events:
            return 0, SimpleNamespace()
        event = self.events.pop(0)
        return event.type, event

    def close(self) -> None:
        self.closed = True


class ClearableReader:
    def __init__(self):
        self.action = np.ones(6, dtype=np.float64)
        self.buttons = [1, 1, 0, 0]
        self.clear_calls = 0

    def open(self) -> None:
        pass

    def get_action(self):
        return self.action.copy(), list(self.buttons)

    def clear(self) -> None:
        self.clear_calls += 1
        self.action.fill(0.0)
        self.buttons = [0, 0, 0, 0]


def motion_event(*, y: int):
    return SimpleNamespace(
        type=SPNAV_EVENT_MOTION,
        motion=SimpleNamespace(x=0, y=y, z=0, rx=0, ry=0, rz=0),
    )


def button_event(*, index: int, pressed: bool):
    return SimpleNamespace(
        type=SPNAV_EVENT_BUTTON,
        button=SimpleNamespace(bnum=index, press=int(pressed)),
    )


def make_reader(monkeypatch, library: FakeSpnavLibrary, **kwargs) -> SpacenavReader:
    monkeypatch.setattr(spacemouse, "_SpnavLibrary", lambda _path=None: library)
    return SpacenavReader(**kwargs)


def test_spnav_library_remove_events_requests_all_event_types():
    calls = []
    library = _SpnavLibrary.__new__(_SpnavLibrary)
    library._lib = SimpleNamespace(spnav_remove_events=lambda event_type: calls.append(event_type))

    library.remove_events()

    assert calls == [SPNAV_EVENT_ANY]


def test_spacenav_reader_clear_drops_queued_events_and_cached_input(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(spacemouse.time, "monotonic", lambda: now[0])
    library = FakeSpnavLibrary([motion_event(y=500), button_event(index=0, pressed=True)])
    reader = make_reader(monkeypatch, library, axis_scale=500.0, deadband=0.0, motion_timeout=0.0)

    action, buttons = reader.get_action()
    assert np.any(action)
    assert buttons[0] == 1

    library.events.extend([motion_event(y=-500), button_event(index=1, pressed=True)])
    now[0] = 20.0
    reader.clear()
    cleared_action, cleared_buttons = reader.get_action()

    assert library.remove_events_calls == 1
    assert library.events == []
    np.testing.assert_array_equal(reader._raw_motion, np.zeros(6))
    np.testing.assert_array_equal(reader._action, np.zeros(6))
    np.testing.assert_array_equal(cleared_action, np.zeros(6))
    assert cleared_buttons == [0, 0, 0, 0]
    assert reader._last_motion_time == 20.0
    assert library.closed is False


def test_teleop_clear_input_preserves_read_only_gripper_target_without_stale_edges():
    reader = ClearableReader()
    teleop = FrankaSpaceMouseTeleop(FrankaSpaceMouseTeleopConfig(), reader=reader)
    teleop.connect()

    initial_action = teleop.get_action()
    assert initial_action["gripper_cmd_bin"] == 0.0
    assert initial_action["reset_requested"] is True
    gripper_target = teleop.gripper_command

    teleop.clear_input()
    assert teleop._last_buttons == []
    cleared_action = teleop.get_action()

    assert reader.clear_calls == 1
    assert teleop.gripper_command == gripper_target
    assert cleared_action["gripper_cmd_bin"] == gripper_target
    assert cleared_action["reset_requested"] is False
    with pytest.raises(AttributeError):
        teleop.gripper_command = 1.0


def test_teleop_clear_input_allows_reader_without_clear():
    reader = SimpleNamespace(get_action=lambda: (np.zeros(6), [0, 0]))
    teleop = FrankaSpaceMouseTeleop(FrankaSpaceMouseTeleopConfig(), reader=reader)

    teleop.clear_input()

    assert teleop._last_buttons == []


@pytest.mark.parametrize(("raw_y", "expected_robot_z"), [(-500, -1.0), (500, 1.0)])
def test_vita_axis_mapping_preserves_both_robot_z_directions(monkeypatch, raw_y, expected_robot_z):
    library = FakeSpnavLibrary([motion_event(y=raw_y)])
    reader = make_reader(monkeypatch, library, axis_scale=500.0, deadband=0.0)
    teleop = FrankaSpaceMouseTeleop(
        FrankaSpaceMouseTeleopConfig(pose_scaler=(1.0, 1.0), use_gripper=False),
        reader=reader,
    )
    teleop.connect()

    action = teleop.get_action()

    assert action["delta_ee_pose.z"] == expected_robot_z
