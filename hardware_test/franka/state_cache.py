from __future__ import annotations

import inspect
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class FrankaStateSnapshot:
    state: dict[str, Any]
    gripper_state: dict[str, Any]
    timestamp_s: float


class StaleFrankaStateError(TimeoutError):
    pass


class FrankaStateCache:
    """Background cache for robot state so observation reads do not block on HTTP."""

    def __init__(
        self,
        client: Any,
        *,
        poll_hz: float,
        timeout_s: float,
        max_age_s: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        if poll_hz <= 0.0:
            raise ValueError("poll_hz must be positive")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if max_age_s <= 0.0:
            raise ValueError("max_age_s must be positive")
        self.client = client
        self.poll_hz = float(poll_hz)
        self.timeout_s = float(timeout_s)
        self.max_age_s = float(max_age_s)
        self._clock = clock
        self._interval_s = 1.0 / self.poll_hz
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._snapshot: FrankaStateSnapshot | None = None
        self._last_error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.refresh_once()
        self._thread = threading.Thread(target=self._run, name="franka-state-cache", daemon=True)
        self._thread.start()

    def refresh_once(self) -> FrankaStateSnapshot:
        state = _call_with_optional_timeout(self.client.get_curr, self.timeout_s)
        gripper_getter = getattr(self.client, "gripper_get_state", None)
        gripper_state = {}
        if callable(gripper_getter):
            gripper_state = _call_with_optional_timeout(gripper_getter, self.timeout_s)
            if not isinstance(gripper_state, dict):
                gripper_state = {}
        snapshot = FrankaStateSnapshot(
            state=dict(state),
            gripper_state=dict(gripper_state),
            timestamp_s=self._clock(),
        )
        with self._lock:
            self._snapshot = snapshot
            self._last_error = None
        return snapshot

    def latest(self, *, max_age_s: float | None = None) -> FrankaStateSnapshot:
        with self._lock:
            snapshot = self._snapshot
            last_error = self._last_error
        if snapshot is None:
            if last_error is not None:
                raise RuntimeError("No Franka state snapshot is available") from last_error
            raise RuntimeError("No Franka state snapshot is available")
        allowed_age = self.max_age_s if max_age_s is None else float(max_age_s)
        age_s = self._clock() - snapshot.timestamp_s
        if age_s > allowed_age:
            raise StaleFrankaStateError(
                f"latest Franka state is too old: {age_s:.3f}s (max allowed: {allowed_age:.3f}s)"
            )
        return snapshot

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                self.refresh_once()
            except BaseException as exc:
                with self._lock:
                    self._last_error = exc


def _call_with_optional_timeout(method: Callable[..., Any], timeout_s: float) -> Any:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and "timeout" in signature.parameters:
        return method(timeout=timeout_s)
    return method()
