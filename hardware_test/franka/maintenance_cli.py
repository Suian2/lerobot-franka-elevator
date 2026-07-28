from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Any

from hardware_test.franka.defaults import get_control_host
from hardware_test.franka.franka_robot import FrankaControlClient


ClientFactory = Callable[[argparse.Namespace], Any]
ClientOperation = Callable[[Any], list[tuple[str, dict[str, Any]]]]


def add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--control-host", default=get_control_host())
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--request-timeout-s", type=float, default=2.0)


def build_control_client(args: argparse.Namespace) -> FrankaControlClient:
    return FrankaControlClient(
        base_url=args.base_url,
        control_host=args.control_host,
        velocity_transport="http",
        zmq_url=None,
        timeout_s=args.request_timeout_s,
        command_duration_ms=300,
    )


def run_client_operation(
    command_name: str,
    args: argparse.Namespace,
    operation: ClientOperation,
    client_factory: ClientFactory,
) -> int:
    client = None
    try:
        client = client_factory(args)
        for label, reply in operation(client):
            print(f"{label}: {json.dumps(reply, sort_keys=True)}", flush=True)
        return 0
    except Exception as exc:
        print(f"{command_name} failed: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()
