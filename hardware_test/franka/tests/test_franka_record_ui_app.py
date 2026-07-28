from __future__ import annotations

import queue
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

from hardware_test.franka.franka_recording_controller import (
    DatasetSpec,
    RecorderCommand,
    RecorderEvent,
    RecorderSnapshot,
    RecorderState,
    RecorderWorker,
)
from hardware_test.franka.run_record_ui import (
    ButtonPolicy,
    FrankaRecorderApp,
    build_ui_arg_parser,
    button_policy,
    main,
)


class FakeWorkerSession:
    def __init__(self):
        self.state = RecorderState.CONNECTING
        self.thread_ids = set()
        self.closed = threading.Event()
        self.saved_episodes = 0
        self.frame_count = 0

    @property
    def snapshot(self):
        return RecorderSnapshot(
            state=self.state,
            message=self.state.value,
            frame_count=self.frame_count,
            saved_episodes=self.saved_episodes,
            dataset_locked=self.state is not RecorderState.CONNECTING,
            pending_valid=self.frame_count > 0,
            wall_elapsed_s=0.0,
            can_save_on_close=False,
        )

    def _touch(self):
        self.thread_ids.add(threading.get_ident())

    def connect(self):
        self._touch()
        self.state = RecorderState.IDLE

    def tick(self):
        self._touch()

    def start_recording(self, spec):
        self._touch()
        assert spec.task == "pick"
        self.state = RecorderState.RECORDING

    def end_recording(self):
        self._touch()
        self.state = RecorderState.IDLE

    def home(self):
        self._touch()
        self.state = RecorderState.HOMING_IDLE

    def clear_fault(self):
        self._touch()
        self.state = RecorderState.IDLE

    def prepare_close(self):
        self._touch()
        self.state = RecorderState.PAUSED_CLOSE

    def cancel_close(self):
        self._touch()
        self.state = RecorderState.IDLE

    def close(self, *, save_pending):
        self._touch()
        assert save_pending is False
        self.state = RecorderState.CLOSED
        self.closed.set()


def test_worker_thread_is_the_only_command_owner():
    session = FakeWorkerSession()
    worker = RecorderWorker(session, fps=100)
    worker.start()
    worker.submit(RecorderCommand.start(DatasetSpec("local/test", "/tmp/test", "pick")))
    worker.submit(RecorderCommand.end())
    worker.submit(RecorderCommand.prepare_close())
    worker.submit(RecorderCommand.close(save_pending=False))

    assert session.closed.wait(timeout=2.0)
    worker.join(timeout=2.0)

    assert session.thread_ids == {worker.thread_ident}
    assert worker.is_alive() is False


def test_ui_parser_uses_manual_defaults_and_existing_hardware_flags():
    args = build_ui_arg_parser().parse_args(["--camera-backend", "none", "--control-host", "test-controller"])

    assert args.duration_s == 0.0
    assert args.num_episodes == 0
    assert args.control_host == "test-controller"


def test_button_policy_matches_recording_and_homing_states():
    assert button_policy(RecorderState.RECORDING) == ButtonPolicy(False, True, True, True)
    assert button_policy(RecorderState.HOMING_RECORDING) == ButtonPolicy(False, False, False, True)
    assert button_policy(RecorderState.FATAL_ERROR) == ButtonPolicy(False, False, False, False)


def test_ui_dry_run_does_not_construct_tk_or_hardware(monkeypatch):
    monkeypatch.setattr(
        "hardware_test.franka.run_record_ui._build_runtime",
        lambda args: (_ for _ in ()).throw(AssertionError("runtime should not be built")),
    )

    assert main(["--dry-run-config", "--camera-backend", "none"]) == 0


def test_launcher_exposes_vita_style_lifecycle_commands():
    script = Path("hardware_test/franka/scripts/start_franka_record_ui.sh")

    result = subprocess.run(["bash", str(script), "help"], capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert "start-ui" in result.stdout
    assert "stop-ui" in result.stdout
    assert "status" in result.stdout


class _DeadWorker:
    def get_event_nowait(self):
        raise queue.Empty

    def is_alive(self):
        return False


class _FakeRoot:
    def __init__(self):
        self.destroyed = False

    def after_idle(self, callback):
        callback()

    def destroy(self):
        self.destroyed = True


class _Var:
    def set(self, value):
        self.value = value


def test_closing_ui_destroys_after_connection_worker_has_exited():
    app = object.__new__(FrankaRecorderApp)
    app.worker = _DeadWorker()
    app.root = _FakeRoot()
    app._closing = True

    app._poll_worker()

    assert app.root.destroyed is True


def test_close_command_error_reenables_ui_instead_of_leaving_it_stuck():
    app = object.__new__(FrankaRecorderApp)
    app.root = _FakeRoot()
    app.args = SimpleNamespace(fps=30, num_episodes=0)
    app.connection_var = _Var()
    app.state_var = _Var()
    app.episode_var = _Var()
    app.frame_var = _Var()
    app.elapsed_var = _Var()
    app._closing = True
    app._last_log_line = ""
    app._apply_policy = lambda policy: None
    app._set_dataset_locked = lambda locked: None
    app._append_log = lambda message, level="info": None
    snapshot = RecorderSnapshot(
        state=RecorderState.FAULTED,
        message="stop failed",
        frame_count=0,
        saved_episodes=0,
        dataset_locked=False,
        pending_valid=False,
        wall_elapsed_s=0.0,
        can_save_on_close=False,
    )

    app._apply_event(RecorderEvent(snapshot=snapshot, level="error", message="stop failed"))

    assert app._closing is False
