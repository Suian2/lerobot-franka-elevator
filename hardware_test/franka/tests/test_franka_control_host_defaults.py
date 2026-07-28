from __future__ import annotations

import argparse
import importlib
import importlib.util
from collections.abc import Callable

from hardware_test.franka import (
    go_home,
    recover_fault,
    run_record,
    run_record_ui,
    run_remote_record,
    run_teleop,
)
from hardware_test.franka.defaults import CONTROL_HOST_ENV_VAR, DEFAULT_CONTROL_HOST
from hardware_test.franka.franka_robot import FrankaRobotConfig
from hardware_test.franka.handeye import collect_eye_to_hand

ParserCase = tuple[Callable[[], argparse.ArgumentParser], list[str]]


def _parser_cases() -> tuple[ParserCase, ...]:
    cases = (
        (go_home.build_arg_parser, []),
        (recover_fault.build_arg_parser, []),
        (run_record.build_arg_parser, []),
        (run_record_ui.build_ui_arg_parser, []),
        (run_remote_record.build_arg_parser, []),
        (run_teleop.build_arg_parser, []),
        (collect_eye_to_hand.build_arg_parser, []),
    )
    module_name = "hardware_test.franka.run_act_rollout_realsense"
    if importlib.util.find_spec(module_name) is None:
        return cases
    rollout_realsense = importlib.import_module(module_name)
    return (
        *cases,
        (
            rollout_realsense.build_arg_parser,
            ["--policy-path", "policy", "--target-floor", "1"],
        ),
    )


def test_franka_defaults_share_the_requested_control_host(monkeypatch):
    monkeypatch.delenv(CONTROL_HOST_ENV_VAR, raising=False)

    assert FrankaRobotConfig(validate_connection=False).control_host == DEFAULT_CONTROL_HOST
    for parser_factory, argv in _parser_cases():
        assert parser_factory().parse_args(argv).control_host == DEFAULT_CONTROL_HOST


def test_environment_overrides_every_implicit_control_host(monkeypatch):
    override = "test-controller"
    monkeypatch.setenv(CONTROL_HOST_ENV_VAR, override)

    assert FrankaRobotConfig(validate_connection=False).control_host == override
    for parser_factory, argv in _parser_cases():
        assert parser_factory().parse_args(argv).control_host == override


def test_blank_environment_value_falls_back_to_checked_in_default(monkeypatch):
    monkeypatch.setenv(CONTROL_HOST_ENV_VAR, "   ")

    assert FrankaRobotConfig(validate_connection=False).control_host == DEFAULT_CONTROL_HOST


def test_explicit_cli_control_host_wins_over_environment(monkeypatch):
    monkeypatch.setenv(CONTROL_HOST_ENV_VAR, "environment-controller")

    args = run_record.build_arg_parser().parse_args(["--control-host", "cli-controller"])

    assert args.control_host == "cli-controller"
