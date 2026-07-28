# Franka Control Server Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `192.168.1.11` the single Franka control-machine default and provide a LeRobot-local command for the remote VITA Docker server.

**Architecture:** `hardware_test/franka/defaults.py` remains the sole checked-in address source. A standalone Bash lifecycle script asks the Python resolver for that host, then uses system `ssh` and `curl` to manage and verify the already-deployed remote Docker server.

**Tech Stack:** Python 3.12, Bash, argparse, pytest, Ruff, OpenSSH, curl, remote tmux/Docker

---

### Task 1: Lock the single-host contract

**Files:**
- Modify: `hardware_test/franka/test_franka_control_host_defaults.py`
- Modify: `hardware_test/franka/test_handeye_cli.py`

- [x] Add `collect_eye_to_hand.build_arg_parser` to the shared parser cases.
- [x] Assert the requested checked-in address is `192.168.1.11` while other tests compare through `DEFAULT_CONTROL_HOST`.
- [x] Run `uv run pytest hardware_test/franka/test_franka_control_host_defaults.py hardware_test/franka/test_handeye_cli.py -q` and verify failures report the old default and the hand-eye parser ignoring `FRANKA_CONTROL_HOST`.

### Task 2: Lock the server launcher contract

**Files:**
- Create: `hardware_test/franka/test_franka_control_server_script.py`
- Create: `hardware_test/franka/scripts/start_franka_control_server.sh`

- [x] Write a subprocess test that prepends fake `ssh` and `curl` commands, runs `start-control`, and asserts the output and remote command use the shared default host, current image, franky backend, and health URL.
- [x] Run `uv run pytest hardware_test/franka/test_franka_control_server_script.py -q` and verify it fails because the launcher does not exist.

### Task 3: Implement the shared default and launcher

**Files:**
- Modify: `hardware_test/franka/defaults.py`
- Modify: `hardware_test/franka/handeye/collect_eye_to_hand.py`
- Create: `hardware_test/franka/scripts/start_franka_control_server.sh`

- [x] Change `DEFAULT_CONTROL_HOST` to `192.168.1.11` and use `get_control_host()` for the hand-eye CLI default.
- [x] Implement `start-control`, `status`, and `stop-control` with VITA-compatible remote Docker, tmux, tuning, health, proxy-bypass, and diagnostic behavior.
- [x] Run all three new/changed test modules and verify they pass.

### Task 4: Document and verify

**Files:**
- Modify: `hardware_test/franka/README.md`

- [x] Add launcher prerequisites, copy-paste lifecycle commands, configuration precedence, and the single-edit IP rule.
- [x] Run `bash -n hardware_test/franka/scripts/start_franka_control_server.sh`.
- [x] Run targeted Franka tests and Ruff checks for changed Python files.
- [x] Scan active `hardware_test` code for competing `192.168.1.5` or `192.168.1.11` literals; only `defaults.py` may own the requested address.
- [x] Run `git diff --check`, inspect `git diff`, and report any hardware verification not performed.
