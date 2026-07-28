from __future__ import annotations

import json

from hardware_test.franka import go_home, recover_fault
from hardware_test.franka.franka_robot import VITA_HOME_JOINTS


class FakeControlClient:
    def __init__(self):
        self.calls = []

    def recover(self):
        self.calls.append(("recover",))
        return {"is_ok": 1}

    def velocity_loop_status(self):
        self.calls.append(("velocity_loop_status",))
        return {"is_ok": 1, "faulted": False, "last_error": ""}

    def joint_position_control(self, joints, *, mode, is_async, timeout):
        self.calls.append(("joint_position_control", list(joints), mode, is_async, timeout))
        return {"is_ok": 1}

    def close(self):
        self.calls.append(("close",))


class FailingControlClient(FakeControlClient):
    def recover(self):
        raise RuntimeError("controller unavailable")

    def joint_position_control(self, joints, *, mode, is_async, timeout):
        raise RuntimeError("motion rejected")


def test_recover_fault_recovers_robot_then_reports_velocity_loop(capsys):
    client = FakeControlClient()

    exit_code = recover_fault.main([], client_factory=lambda _args: client)

    assert exit_code == 0
    assert client.calls == [("recover",), ("velocity_loop_status",), ("close",)]
    output = capsys.readouterr().out
    assert json.loads(output.splitlines()[0].removeprefix("recover: ")) == {"is_ok": 1}
    assert json.loads(output.splitlines()[1].removeprefix("velocity_loop: ")) == {
        "faulted": False,
        "is_ok": 1,
        "last_error": "",
    }


def test_go_home_sends_vita_home_as_synchronous_absolute_motion(capsys):
    client = FakeControlClient()

    exit_code = go_home.main(
        ["--control-host", "test-controller", "--timeout-s", "42"],
        client_factory=lambda args: client,
    )

    assert exit_code == 0
    assert client.calls == [
        ("joint_position_control", list(VITA_HOME_JOINTS), "absolute", False, 42.0),
        ("close",),
    ]
    output = capsys.readouterr().out
    assert "control_host=test-controller" in output
    assert json.loads(output.splitlines()[-1].removeprefix("home: ")) == {"is_ok": 1}


def test_recover_fault_returns_nonzero_and_prints_error(capsys):
    exit_code = recover_fault.main([], client_factory=lambda _args: FailingControlClient())

    assert exit_code == 1
    assert "controller unavailable" in capsys.readouterr().err


def test_go_home_returns_nonzero_and_prints_error(capsys):
    exit_code = go_home.main([], client_factory=lambda _args: FailingControlClient())

    assert exit_code == 1
    assert "motion rejected" in capsys.readouterr().err
