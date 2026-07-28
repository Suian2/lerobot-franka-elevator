from __future__ import annotations

from typing import Any

import pytest

import hardware_test.franka.franka_robot as franka_robot
from hardware_test.franka.franka_robot import FrankaControlClient


class FakeResponse:
    def __init__(self, payload: Any):
        self.payload = payload
        self.raise_for_status_calls = 0

    def raise_for_status(self) -> None:
        self.raise_for_status_calls += 1

    def json(self) -> Any:
        return self.payload


class FakeRequestsSession:
    def __init__(self, *payloads: Any):
        self.responses = [FakeResponse(payload) for payload in payloads]
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("GET", url, kwargs))
        return self.responses.pop(0)

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(("POST", url, kwargs))
        return self.responses.pop(0)


def _http_client(*payloads: Any) -> tuple[FrankaControlClient, FakeRequestsSession]:
    session = FakeRequestsSession(*payloads)
    client = FrankaControlClient(
        base_url="http://franka.test:29000/ctl",
        control_host="unused",
        velocity_transport="http",
        zmq_url=None,
        timeout_s=2.0,
        command_duration_ms=300,
    )
    client._session = session
    return client, session


def test_joint_position_control_posts_async_flag() -> None:
    client, session = _http_client({"is_ok": 1})

    client.joint_position_control([0.1, 0.2], mode="absolute", is_async=True)

    method, url, kwargs = session.calls[0]
    assert method == "POST"
    assert url == "http://franka.test:29000/ctl/joint_position_control"
    assert kwargs["json"] == {
        "joints_lst": [0.1, 0.2],
        "mode": "absolute",
        "is_async": 1,
    }


@pytest.mark.parametrize(
    ("method_name", "path"),
    [
        ("recover", "recover"),
        ("velocity_loop_status", "velocity_ws_status"),
        ("stop_cartesian_velocity_control_direct", "stop_cartesian_velocity_control"),
        ("stop_joint_position_control", "stop_joint_position_control"),
    ],
)
def test_safety_requests_use_expected_get_endpoint(method_name: str, path: str) -> None:
    client, session = _http_client({"is_ok": 1})

    getattr(client, method_name)()

    assert session.calls == [("GET", f"http://franka.test:29000/ctl/{path}", {"timeout": 2.0})]


def test_join_motion_posts_timeout_and_returns_true_when_complete() -> None:
    client, session = _http_client({"is_ok": 1})

    is_complete = client.join_motion(timeout_s=0.0)

    assert is_complete is True
    assert session.calls == [
        (
            "POST",
            "http://franka.test:29000/ctl/join",
            {"json": {"timeout": 0.0}, "timeout": 2.0},
        )
    ]


def test_join_motion_returns_false_for_incomplete_reply_without_error() -> None:
    client, _ = _http_client({"is_ok": 0})

    assert client.join_motion(timeout_s=0.0) is False


def test_safety_request_raises_for_application_error_reply() -> None:
    client, _ = _http_client({"is_ok": 0, "error_type": "RuntimeError", "error": "fault"})

    with pytest.raises(franka_robot.FrankaControlError, match="RuntimeError: fault"):
        client.recover()


def test_request_rejects_non_mapping_json_reply() -> None:
    client, _ = _http_client(["unexpected"])

    with pytest.raises(franka_robot.FrankaControlError):
        client.recover()
