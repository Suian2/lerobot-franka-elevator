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

from hardware_test.franka.maintenance_cli import (  # noqa: E402
    ClientFactory,
    add_connection_arguments,
    build_control_client,
    run_client_operation,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clear Franka and VITA velocity-loop faults.")
    add_connection_arguments(parser)
    parser.add_argument("--timeout-s", type=float, dest="request_timeout_s", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None, *, client_factory: ClientFactory = build_control_client) -> int:
    args = build_arg_parser().parse_args(argv)

    def recover(client):
        return [
            ("recover", client.recover()),
            ("velocity_loop", client.velocity_loop_status()),
        ]

    return run_client_operation("recover_fault", args, recover, client_factory)


if __name__ == "__main__":
    raise SystemExit(main())
