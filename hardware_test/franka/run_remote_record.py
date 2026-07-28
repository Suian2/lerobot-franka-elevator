from __future__ import annotations

import argparse
import inspect
import time
from dataclasses import dataclass, replace
from threading import Event
from typing import Any

from hardware_test.franka import run_record
from hardware_test.franka.franka_robot import FrankaRobotConfig
from hardware_test.franka.record_lerobot_dataset import make_lerobot_frame, precise_sleep
from hardware_test.franka.state_cache import FrankaStateSnapshot, StaleFrankaStateError


@dataclass(frozen=True)
class PreflightStateResult:
    ok: bool
    attempts: int
    snapshot: FrankaStateSnapshot | None = None
    error: BaseException | None = None
    gripper_error: BaseException | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = run_record.build_arg_parser()
    parser.add_argument("--state-preflight-retries", type=int, default=5)
    parser.set_defaults(
        streaming_encoding=True,
        encoder_threads=2,
        max_state_age_s=1.0,
        state_max_consecutive_misses=5,
        state_retry_sleep_s=0.02,
    )
    return parser


def build_robot_config(args: argparse.Namespace) -> FrankaRobotConfig:
    return replace(
        run_record.build_robot_config(args),
        state_cache_enabled=False,
        validate_connection=False,
        max_state_age_s=args.max_state_age_s,
    )


def preflight_state_client(
    client: Any,
    *,
    retries: int,
    timeout_s: float,
    sleep_s: float,
    gripper_state_optional: bool,
) -> PreflightStateResult:
    last_error: BaseException | None = None
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            state = _call_with_optional_timeout(client.get_curr, timeout_s)
        except BaseException as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(max(0.0, float(sleep_s)))
                continue
            return PreflightStateResult(ok=False, attempts=attempt, error=exc)

        gripper_state: dict[str, Any] = {}
        gripper_error: BaseException | None = None
        gripper_getter = getattr(client, "gripper_get_state", None)
        if callable(gripper_getter):
            try:
                maybe_gripper_state = _call_with_optional_timeout(gripper_getter, timeout_s)
                if isinstance(maybe_gripper_state, dict):
                    gripper_state = dict(maybe_gripper_state)
            except BaseException as exc:
                gripper_error = exc
                if not gripper_state_optional:
                    return PreflightStateResult(ok=False, attempts=attempt, error=exc, gripper_error=exc)

        snapshot = FrankaStateSnapshot(
            state=dict(state),
            gripper_state=gripper_state,
            timestamp_s=time.monotonic(),
        )
        return PreflightStateResult(
            ok=True,
            attempts=attempt,
            snapshot=snapshot,
            error=last_error,
            gripper_error=gripper_error,
        )

    return PreflightStateResult(ok=False, attempts=attempts, error=last_error)


def record_remote_episode(
    *,
    robot: Any,
    teleop: Any,
    dataset: Any,
    fps: int,
    duration_s: float,
    task: str,
    max_consecutive_state_misses: int,
    state_retry_sleep_s: float,
    stop_event: Event | None = None,
) -> int:
    features = dataset.features
    control_interval_s = 1.0 / float(fps)
    frames = 0
    consecutive_state_misses = 0
    start_t = time.perf_counter()

    while time.perf_counter() - start_t < duration_s:
        loop_t = time.perf_counter()
        if stop_event is not None and stop_event.is_set():
            break

        try:
            observation = robot.get_observation()
            consecutive_state_misses = 0
        except StaleFrankaStateError:
            consecutive_state_misses += 1
            if consecutive_state_misses > max_consecutive_state_misses:
                raise
            time.sleep(max(0.0, float(state_retry_sleep_s)))
            continue

        action = teleop.get_action()
        sent_action = robot.send_action(action)
        dataset.add_frame(make_lerobot_frame(features, observation, sent_action, task=task))
        frames += 1

        dt_s = time.perf_counter() - loop_t
        precise_sleep(max(0.0, control_interval_s - dt_s))

    return frames


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_record.main(_namespace_to_argv(args))


def _call_with_optional_timeout(method: Any, timeout_s: float) -> Any:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and "timeout" in signature.parameters:
        return method(timeout=timeout_s)
    return method()


def _namespace_to_argv(args: argparse.Namespace) -> list[str]:
    argv: list[str] = []
    for key, value in vars(args).items():
        if key in {"state_max_consecutive_misses", "state_preflight_retries", "state_retry_sleep_s"}:
            continue
        option = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                argv.append(option)
            continue
        if value is None:
            continue
        argv.extend([option, str(value)])
    return argv
