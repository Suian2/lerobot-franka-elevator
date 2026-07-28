# Franka Maintenance Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide separate terminal commands for Franka fault recovery and VITA-aligned home motion.

**Architecture:** Thin Python CLIs reuse `FrankaControlClient`; shared argument/client/error helpers avoid duplicated protocol code. The existing home tuple becomes a named constant used by both configuration and CLI.

**Tech Stack:** Python 3.12, argparse, pytest, existing requests-based Franka client.

---

### Task 1: Lock CLI behavior with failing tests

**Files:**
- Create: `hardware_test/franka/test_maintenance_scripts.py`

- [ ] Write tests that inject a fake client into `recover_fault.main()` and `go_home.main()`.
- [ ] Assert recovery calls `recover()` then `velocity_loop_status()`.
- [ ] Assert home sends the exact seven VITA joints with `mode=\"absolute\"`, `is_async=False`, and the configured timeout.
- [ ] Assert exceptions produce exit code 1 and stderr output.
- [ ] Run `uv run pytest hardware_test/franka/test_maintenance_scripts.py -q` and verify collection fails because the scripts do not exist.

### Task 2: Implement the scripts

**Files:**
- Create: `hardware_test/franka/maintenance_cli.py`
- Create: `hardware_test/franka/recover_fault.py`
- Create: `hardware_test/franka/go_home.py`
- Modify: `hardware_test/franka/franka_robot.py`

- [ ] Extract the current `home_joints` tuple to `VITA_HOME_JOINTS` without changing its values.
- [ ] Add shared `--control-host`, `--base-url`, and client construction helpers.
- [ ] Implement recovery followed by velocity-loop status output.
- [ ] Implement immediate absolute synchronous home motion with a 60-second default timeout.
- [ ] Run `uv run pytest hardware_test/franka/test_maintenance_scripts.py -q` and verify all new tests pass.

### Task 3: Document and verify

**Files:**
- Modify: `hardware_test/franka/README.md`

- [ ] Add copy-paste commands for both scripts and document their options and immediate-motion behavior.
- [ ] Run `uv run pytest hardware_test/franka/test_maintenance_scripts.py hardware_test/franka/test_franka_adapters.py -q`.
- [ ] Run Ruff on the files changed for this feature.
- [ ] Inspect `git diff --check` and the scoped diff before reporting completion.
