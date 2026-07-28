from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from hardware_test.franka.defaults import CONTROL_HOST_ENV_VAR, DEFAULT_CONTROL_HOST

REPO_ROOT = Path(__file__).parents[2]
SCRIPT_PATH = Path(__file__).parent / "scripts" / "start_franka_control_server.sh"


def _write_fake_command(bin_dir: Path, name: str) -> None:
    path = bin_dir / name
    path.write_text(f"#!/usr/bin/env bash\nprintf '{name} %s\\n' \"$*\"\n")
    path.chmod(0o755)


def _run_launcher(
    tmp_path: Path,
    command: str,
    *,
    control_host: str | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_command(bin_dir, "ssh")
    _write_fake_command(bin_dir, "curl")

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PYTHON_BIN": sys.executable,
    }
    if control_host is None:
        env.pop(CONTROL_HOST_ENV_VAR, None)
    else:
        env[CONTROL_HOST_ENV_VAR] = control_host

    return subprocess.run(
        ["bash", str(SCRIPT_PATH), command],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_start_control_resolves_the_checked_in_host_and_vita_server_command(tmp_path: Path):
    completed = _run_launcher(tmp_path, "start-control")

    assert completed.returncode == 0, completed.stderr
    assert f"franka@{DEFAULT_CONTROL_HOST}" in completed.stdout
    assert f"http://{DEFAULT_CONTROL_HOST}:29000/ctl/config" in completed.stdout
    assert "vita-franka-server:zmq-franky-tuned" in completed.stdout
    assert "FRANKA_VELOCITY_BACKEND=franky" in completed.stdout
    assert "run_franka_server_docker_control_machine.sh" in completed.stdout


def test_launcher_honors_the_shared_control_host_environment_override(tmp_path: Path):
    completed = _run_launcher(tmp_path, "status", control_host="control.test")

    assert completed.returncode == 0, completed.stderr
    assert "franka@control.test" in completed.stdout
    assert "http://control.test:29000/ctl/config" in completed.stdout
