from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from hardware_test.franka.franka_recording_controller import (
    DatasetSpec,
    FrankaRecorderSession,
    RecorderOptions,
    RecorderState,
)
from hardware_test.franka.state_cache import StaleFrankaStateError

ACTION_KEYS = tuple(f"delta_ee_pose.{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz"))


def _zero_action(**extra):
    action = dict.fromkeys(ACTION_KEYS, 0.0)
    action["gripper_cmd_bin"] = 1.0
    action["reset_requested"] = False
    action.update(extra)
    return action


class FakeRobot:
    name = "fake_franka"
    observation_features = {"joint_1.pos": float}
    action_features = dict.fromkeys((*ACTION_KEYS, "gripper_cmd_bin"), float)

    def __init__(self, *, samples=None, motion_results=None, recover_error=None):
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.sample_index = 0
        self.sample_calls = 0
        self.samples = deque(samples or [])
        self.sent_actions = []
        self.zero_calls = 0
        self.quiesce_calls = 0
        self.start_home_calls = 0
        self.stop_home_calls = 0
        self.recover_calls = 0
        self.motion_results = deque(motion_results or [True])
        self.recover_error = recover_error
        self.quiesce_error = None
        self.config = SimpleNamespace(home_stable_samples=2, home_timeout_s=20.0)

    def connect(self):
        self.connect_calls += 1

    def disconnect(self):
        self.disconnect_calls += 1

    def get_observation_sample(self):
        self.sample_calls += 1
        if self.samples:
            sample = self.samples.popleft()
            if isinstance(sample, BaseException):
                raise sample
            return sample
        self.sample_index += 1
        return SimpleNamespace(
            observation={"joint_1.pos": float(self.sample_index)},
            state={"joint": [0.0] * 7, "ee": _matrix_at(float(self.sample_index) / 100.0)},
            state_timestamp_s=float(self.sample_index) / 30.0,
        )

    def send_action(self, action):
        sent = {key: value for key, value in action.items() if key != "reset_requested"}
        self.sent_actions.append(sent)
        return sent

    def send_zero_cartesian_velocity(self):
        self.zero_calls += 1

    def quiesce_cartesian_velocity_control(self):
        self.quiesce_calls += 1
        if self.quiesce_error is not None:
            raise self.quiesce_error

    def start_home_async(self):
        self.start_home_calls += 1

    def stop_home_motion(self):
        self.stop_home_calls += 1

    def recover(self):
        self.recover_calls += 1
        if self.recover_error is not None:
            raise self.recover_error

    def motion_complete(self):
        if len(self.motion_results) > 1:
            return self.motion_results.popleft()
        return self.motion_results[0]

    def is_home_state(self, state):
        return bool(state.get("home", False))


class FakeTeleop:
    action_features = dict.fromkeys((*ACTION_KEYS, "gripper_cmd_bin"), float)

    def __init__(self, actions=None):
        self.actions = deque(actions or [])
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.gripper_command = 1.0
        self.clear_calls = 0

    def connect(self):
        self.connect_calls += 1

    def disconnect(self):
        self.disconnect_calls += 1

    def get_action(self):
        return dict(self.actions.popleft()) if self.actions else _zero_action()

    def clear_input(self):
        self.clear_calls += 1


class FakeDataset:
    features = {}

    def __init__(self):
        self.frames = []
        self.saved_episodes = []
        self.clear_calls = 0
        self.finalize_calls = 0
        self.add_error = None
        self.save_error = None
        self.finalize_error = None

    def add_frame(self, frame):
        if self.add_error is not None:
            raise self.add_error
        self.frames.append(frame)

    def save_episode(self):
        if self.save_error is not None:
            raise self.save_error
        self.saved_episodes.append(list(self.frames))
        self.frames.clear()

    def clear_episode_buffer(self, *, delete_images):
        assert delete_images is True
        self.clear_calls += 1
        self.frames.clear()

    def has_pending_frames(self):
        return bool(self.frames)

    def finalize(self):
        self.finalize_calls += 1
        if self.finalize_error is not None:
            raise self.finalize_error


class Rig:
    def __init__(
        self,
        *,
        duration_s=0.0,
        num_episodes=0,
        samples=None,
        actions=None,
        motion_results=None,
        recover_error=None,
    ):
        self.robot = FakeRobot(
            samples=samples,
            motion_results=motion_results,
            recover_error=recover_error,
        )
        self.teleop = FakeTeleop(actions=actions)
        self.dataset = FakeDataset()
        self.dataset_factory_calls = []
        self.session = FrankaRecorderSession(
            robot=self.robot,
            teleop=self.teleop,
            dataset_factory=self._make_dataset,
            options=RecorderOptions(
                fps=30,
                duration_s=duration_s,
                num_episodes=num_episodes,
                cartesian_action_units="delta",
            ),
            frame_builder=lambda features, observation, action, *, task: {
                "observation": dict(observation),
                "action": dict(action),
                "task": task,
            },
        )

    def _make_dataset(self, spec):
        self.dataset_factory_calls.append(spec)
        return self.dataset


def _matrix_at(x):
    return [
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _sample(index, *, x, home):
    return SimpleNamespace(
        observation={"joint_1.pos": float(index)},
        state={"joint": [0.0] * 7, "ee": _matrix_at(x), "home": home},
        state_timestamp_s=float(index) / 30.0,
    )


def test_session_connects_hardware_once_and_saves_multiple_episodes():
    rig = Rig()
    spec = DatasetSpec("local/test", "/tmp/franka-test", "pick")
    rig.session.connect()

    rig.session.start_recording(spec)
    rig.session.tick()
    rig.session.end_recording()
    rig.session.start_recording(spec)
    rig.session.tick()
    rig.session.end_recording()

    assert rig.robot.connect_calls == 1
    assert rig.teleop.connect_calls == 1
    assert rig.dataset_factory_calls == [spec]
    assert len(rig.dataset.saved_episodes) == 2
    assert rig.session.snapshot.saved_episodes == 2
    assert rig.session.snapshot.dataset_locked is True
    assert rig.session.state is RecorderState.IDLE


def test_session_rejects_dataset_field_changes_after_first_start():
    rig = Rig()
    rig.session.connect()
    rig.session.start_recording(DatasetSpec("local/test", "/tmp/franka-test", "pick"))
    rig.session.end_recording()

    with pytest.raises(ValueError, match="dataset fields are locked"):
        rig.session.start_recording(DatasetSpec("local/other", "/tmp/franka-test", "pick"))


def test_session_discards_episode_stopped_before_first_frame():
    rig = Rig()
    rig.session.connect()
    rig.session.start_recording(DatasetSpec("local/test", "/tmp/franka-test", "pick"))

    rig.session.end_recording()

    assert rig.dataset.clear_calls == 1
    assert rig.dataset.saved_episodes == []
    assert rig.session.state is RecorderState.IDLE


def test_session_idle_teleoperation_sends_actions_without_writing_dataset():
    rig = Rig()
    rig.session.connect()

    rig.session.tick()

    assert len(rig.robot.sent_actions) == 1
    assert rig.dataset.frames == []


def test_session_duration_limit_saves_after_expected_frame_count():
    rig = Rig(duration_s=2 / 30)
    rig.session.connect()
    rig.session.start_recording(DatasetSpec("local/test", "/tmp/franka-test", "pick"))

    rig.session.tick()
    assert rig.session.state is RecorderState.RECORDING
    rig.session.tick()

    assert rig.session.state is RecorderState.IDLE
    assert len(rig.dataset.saved_episodes) == 1
    assert len(rig.dataset.saved_episodes[0]) == 2


def test_session_retries_transient_stale_state_before_sending_action():
    rig = Rig(
        samples=[
            StaleFrankaStateError("stale 1"),
            StaleFrankaStateError("stale 2"),
            _sample(1, x=0.0, home=False),
        ]
    )
    rig.session.connect()

    rig.session.tick()

    assert rig.robot.sample_calls == 3
    assert len(rig.robot.sent_actions) == 1


def test_session_episode_limit_disables_another_start():
    rig = Rig(num_episodes=1)
    spec = DatasetSpec("local/test", "/tmp/franka-test", "pick")
    rig.session.connect()
    rig.session.start_recording(spec)
    rig.session.tick()
    rig.session.end_recording()

    with pytest.raises(RuntimeError, match="episode limit"):
        rig.session.start_recording(spec)


def test_recorded_home_records_transitions_and_auto_saves_after_fresh_final_sample():
    samples = [
        _sample(1, x=0.00, home=False),
        _sample(2, x=0.01, home=False),
        _sample(3, x=0.02, home=True),
        _sample(4, x=0.03, home=True),
        _sample(5, x=0.04, home=True),
        _sample(6, x=0.05, home=True),
    ]
    rig = Rig(samples=samples, motion_results=[False, True])
    rig.session.connect()
    rig.session.start_recording(DatasetSpec("local/test", "/tmp/franka-test", "pick"))

    rig.session.home()
    for _ in range(5):
        rig.session.tick()

    assert rig.robot.quiesce_calls == 1
    assert rig.robot.start_home_calls == 1
    assert len(rig.dataset.saved_episodes) == 1
    saved = rig.dataset.saved_episodes[0]
    assert any(frame["action"]["delta_ee_pose.x"] > 0.0 for frame in saved)
    assert all(saved[-1]["action"][key] == 0.0 for key in ACTION_KEYS)
    assert saved[-1]["observation"]["joint_1.pos"] == 6.0
    assert rig.session.state is RecorderState.IDLE


def test_spacemouse_reset_routes_to_async_home_without_send_action():
    rig = Rig(
        samples=[_sample(1, x=0.0, home=False), _sample(2, x=0.0, home=False)],
        actions=[_zero_action(reset_requested=True)],
    )
    rig.session.connect()

    rig.session.tick()

    assert rig.robot.sent_actions == []
    assert rig.robot.start_home_calls == 1
    assert rig.session.state is RecorderState.HOMING_IDLE


def test_clear_fault_success_continues_same_episode():
    rig = Rig()
    rig.session.connect()
    rig.session.start_recording(DatasetSpec("local/test", "/tmp/franka-test", "pick"))
    rig.session.tick()

    rig.session.clear_fault()
    rig.session.tick()

    assert rig.robot.recover_calls == 1
    assert rig.teleop.clear_calls == 1
    assert rig.session.state is RecorderState.RECORDING
    assert len(rig.dataset.frames) == 2


def test_clear_fault_failure_discards_pending_episode():
    rig = Rig(recover_error=RuntimeError("recover failed"))
    rig.session.connect()
    rig.session.start_recording(DatasetSpec("local/test", "/tmp/franka-test", "pick"))
    rig.session.tick()

    with pytest.raises(RuntimeError, match="recover failed"):
        rig.session.clear_fault()

    assert rig.dataset.clear_calls == 1
    assert rig.dataset.frames == []
    assert rig.session.state is RecorderState.FATAL_ERROR


def test_close_pauses_motion_then_saves_valid_recording_and_disconnects():
    rig = Rig()
    rig.session.connect()
    rig.session.start_recording(DatasetSpec("local/test", "/tmp/franka-test", "pick"))
    rig.session.tick()

    rig.session.prepare_close()
    assert rig.session.state is RecorderState.PAUSED_CLOSE
    assert rig.session.can_save_on_close is True
    assert rig.robot.quiesce_calls == 1

    rig.session.close(save_pending=True)

    assert len(rig.dataset.saved_episodes) == 1
    assert rig.dataset.finalize_calls == 1
    assert rig.teleop.disconnect_calls == 1
    assert rig.robot.disconnect_calls == 1
    assert rig.session.state is RecorderState.CLOSED


def test_close_during_recorded_home_cannot_save_incomplete_trajectory():
    rig = Rig(samples=[_sample(1, x=0.0, home=False)])
    rig.session.connect()
    rig.session.start_recording(DatasetSpec("local/test", "/tmp/franka-test", "pick"))
    rig.session.home()

    rig.session.prepare_close()

    assert rig.robot.stop_home_calls == 1
    assert rig.session.can_save_on_close is False

    rig.session.close(save_pending=False)
    assert rig.dataset.saved_episodes == []
    assert rig.dataset.finalize_calls == 1


def test_runtime_fault_during_home_stops_joint_motion():
    rig = Rig(samples=[_sample(1, x=0.0, home=False)])
    rig.session.connect()
    rig.session.home()

    rig.session.mark_runtime_fault(RuntimeError("state stream failed"))

    assert rig.robot.stop_home_calls == 1
    assert rig.session.state is RecorderState.FAULTED


def test_close_disconnects_hardware_even_when_dataset_finalize_fails():
    rig = Rig()
    rig.session.connect()
    rig.session.start_recording(DatasetSpec("local/test", "/tmp/franka-test", "pick"))
    rig.session.end_recording()
    rig.dataset.finalize_error = RuntimeError("finalize failed")
    rig.session.prepare_close()

    with pytest.raises(RuntimeError, match="finalize failed"):
        rig.session.close(save_pending=False)

    assert rig.teleop.disconnect_calls == 1
    assert rig.robot.disconnect_calls == 1
    assert rig.session.state is RecorderState.CLOSED


def test_clear_fault_reconfirms_old_home_motion_stopped_after_recovery():
    rig = Rig(samples=[_sample(1, x=0.0, home=False), _sample(2, x=0.0, home=False)])
    rig.session.connect()
    rig.session.home()
    stop_calls = 0

    def stop_home_motion():
        nonlocal stop_calls
        stop_calls += 1
        if rig.robot.recover_calls == 0:
            raise RuntimeError("controller still faulted")

    rig.robot.stop_home_motion = stop_home_motion
    rig.session.mark_runtime_fault(RuntimeError("joint fault"))

    rig.session.clear_fault()

    assert stop_calls == 3
    assert rig.robot.start_home_calls == 2
    assert rig.session.state is RecorderState.HOMING_IDLE


def test_prepare_close_failure_enters_fault_instead_of_sticking_in_transition():
    rig = Rig()
    rig.session.connect()
    rig.robot.quiesce_error = RuntimeError("stop timed out")

    with pytest.raises(RuntimeError, match="stop timed out"):
        rig.session.prepare_close()

    assert rig.session.state is RecorderState.FAULTED


def test_dataset_add_failure_is_fatal_and_cannot_be_cleared_as_hardware_fault():
    rig = Rig()
    rig.session.connect()
    rig.session.start_recording(DatasetSpec("local/test", "/tmp/franka-test", "pick"))
    rig.dataset.add_error = RuntimeError("encoder failed")

    with pytest.raises(RuntimeError, match="encoder failed"):
        rig.session.tick()

    assert rig.session.state is RecorderState.FATAL_ERROR
    assert rig.session.snapshot.pending_valid is False
    with pytest.raises(RuntimeError, match="fatal_error"):
        rig.session.clear_fault()


def test_dataset_save_failure_is_fatal_and_pending_episode_can_only_be_discarded():
    rig = Rig()
    rig.session.connect()
    rig.session.start_recording(DatasetSpec("local/test", "/tmp/franka-test", "pick"))
    rig.session.tick()
    rig.dataset.save_error = RuntimeError("disk full")

    with pytest.raises(RuntimeError, match="disk full"):
        rig.session.end_recording()

    assert rig.session.state is RecorderState.FATAL_ERROR
    assert rig.session.snapshot.pending_valid is False
    rig.session.prepare_close()
    assert rig.session.can_save_on_close is False
    rig.session.close(save_pending=False)
    assert rig.dataset.clear_calls == 1
