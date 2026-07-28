#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (SRC_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hardware_test.franka.franka_robot import VITA_HOME_JOINTS  # noqa: E402
from hardware_test.franka.maintenance_cli import (  # noqa: E402
    ClientFactory,
    add_connection_arguments,
    build_control_client,
    run_client_operation,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Move Franka to the VITA UI home joint pose.")
    add_connection_arguments(parser)
    parser.add_argument("--timeout-s", type=float, default=60.0, help="Motion request timeout in seconds.")
    return parser


def main(argv: list[str] | None = None, *, client_factory: ClientFactory = build_control_client) -> int:
    args = build_arg_parser().parse_args(argv)
    print(
        f"moving home: control_host={args.control_host} joints={list(VITA_HOME_JOINTS)}",
        flush=True,
    )

    def move_home(client):
        reply = client.joint_position_control(
            list(VITA_HOME_JOINTS),
            mode="absolute",
            is_async=False,
            timeout=args.timeout_s,
        )
        return [("home", reply)]

    return run_client_operation("go_home", args, move_home, client_factory)


if __name__ == "__main__":
    raise SystemExit(main())
