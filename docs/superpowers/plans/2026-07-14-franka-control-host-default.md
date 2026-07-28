# Franka Control Host Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every Franka hardware-test entry point use one configurable control-host default while preserving command-line overrides.

**Architecture:** A small `hardware_test.franka.defaults` module owns the checked-in address, environment-variable name, and resolver. Franka dataclass and parser defaults call the resolver; tests and documentation refer to that shared contract instead of copying runtime address literals.

**Tech Stack:** Python 3.12, dataclasses, argparse, pytest, Ruff, uv

---

### Task 1: Lock the shared-default contract with regression tests

**Files:**
- Create: `hardware_test/franka/test_franka_control_host_defaults.py`

- [x] **Step 1: Write the failing test module**

```python
from __future__ import annotations

import argparse
from collections.abc import Callable

import pytest

from hardware_test.franka import go_home, run_act_rollout, run_act_rollout_realsense, run_record, run_teleop
from hardware_test.franka.defaults import CONTROL_HOST_ENV_VAR, DEFAULT_CONTROL_HOST, get_control_host
from hardware_test.franka.franka_robot import FrankaRobotConfig


ParserFactory = Callable[[], argparse.ArgumentParser]


@pytest.fixture
def parser_cases() -> tuple[tuple[ParserFactory, list[str]], ...]:
    return (
        (go_home.build_arg_parser, []),
        (run_record.build_arg_parser, []),
        (run_teleop.build_arg_parser, []),
        (run_act_rollout.build_arg_parser, ["--policy-path", "policy"]),
        (run_act_rollout_realsense.build_arg_parser, ["--policy-path", "policy"]),
    )


def test_franka_defaults_share_one_control_host(monkeypatch, parser_cases):
    monkeypatch.delenv(CONTROL_HOST_ENV_VAR, raising=False)

    assert get_control_host() == DEFAULT_CONTROL_HOST
    assert FrankaRobotConfig(validate_connection=False).control_host == DEFAULT_CONTROL_HOST
    for parser_factory, argv in parser_cases:
        assert parser_factory().parse_args(argv).control_host == DEFAULT_CONTROL_HOST


def test_environment_overrides_every_implicit_control_host(monkeypatch, parser_cases):
    override = "test-controller"
    monkeypatch.setenv(CONTROL_HOST_ENV_VAR, override)

    assert get_control_host() == override
    assert FrankaRobotConfig(validate_connection=False).control_host == override
    for parser_factory, argv in parser_cases:
        assert parser_factory().parse_args(argv).control_host == override


def test_empty_environment_value_falls_back_to_checked_in_default(monkeypatch):
    monkeypatch.setenv(CONTROL_HOST_ENV_VAR, "   ")

    assert get_control_host() == DEFAULT_CONTROL_HOST


def test_explicit_cli_control_host_wins_over_environment(monkeypatch):
    monkeypatch.setenv(CONTROL_HOST_ENV_VAR, "environment-controller")

    args = run_record.build_arg_parser().parse_args(["--control-host", "cli-controller"])

    assert args.control_host == "cli-controller"
```

- [x] **Step 2: Run the new test and verify the red state**

Run:

```bash
uv run pytest hardware_test/franka/test_franka_control_host_defaults.py -q
```

Expected: collection fails because `hardware_test.franka.defaults` does not yet exist.

Execution note: the actual RED test temporarily asserted the requested address
directly so it failed on the old value rather than stopping at collection. Once
GREEN, the assertion was refactored to import `DEFAULT_CONTROL_HOST`.

### Task 2: Add the shared resolver and wire active Franka entry points

**Files:**
- Create: `hardware_test/franka/defaults.py`
- Modify: `hardware_test/franka/franka_robot.py`
- Modify: `hardware_test/franka/maintenance_cli.py`
- Modify: `hardware_test/franka/run_teleop.py`
- Modify: `hardware_test/franka/run_record.py`
- Modify: `hardware_test/franka/run_act_rollout.py`
- Modify: `hardware_test/franka/run_act_rollout_realsense.py`

- [x] **Step 1: Add the central default and environment resolver**

```python
from __future__ import annotations

import os
from typing import Final


CONTROL_HOST_ENV_VAR: Final[str] = "FRANKA_CONTROL_HOST"
DEFAULT_CONTROL_HOST: Final[str] = "192.168.1.5"


def get_control_host() -> str:
    """Return the configured Franka control-service host."""
    return os.getenv(CONTROL_HOST_ENV_VAR, "").strip() or DEFAULT_CONTROL_HOST
```

- [x] **Step 2: Make direct robot configuration use the resolver**

Add this import to `hardware_test/franka/franka_robot.py`:

```python
from .defaults import get_control_host
```

Replace its field default with:

```python
control_host: str = field(default_factory=get_control_host)
```

- [x] **Step 3: Make every active parser use the resolver**

Each parser module imports:

```python
from hardware_test.franka.defaults import get_control_host
```

Each implicit parser default becomes:

```python
parser.add_argument("--control-host", default=get_control_host())
```

Apply this to `maintenance_cli.py`, `run_teleop.py`, `run_record.py`,
`run_act_rollout.py`, and `run_act_rollout_realsense.py`. Preserve the existing
path-bootstrap ordering and `# noqa: E402` markers where present.

- [x] **Step 4: Run the new tests and verify the green state**

Run:

```bash
uv run pytest hardware_test/franka/test_franka_control_host_defaults.py -q
```

Expected: `4 passed`.

### Task 3: Remove stale copies from tests and documentation

**Files:**
- Modify: `hardware_test/franka/test_franka_adapters.py`
- Modify: `hardware_test/franka/test_act_rollout.py`
- Modify: `hardware_test/franka/test_act_rollout_realsense.py`
- Modify: `hardware_test/franka/test_franka_record_ui_app.py`
- Modify: `hardware_test/franka/test_maintenance_scripts.py`
- Modify: `hardware_test/franka/README.md`
- Modify: `docs/superpowers/specs/2026-07-10-franka-maintenance-scripts-design.md`
- Modify: `docs/superpowers/specs/2026-07-10-franka-act-rollout-design.md`
- Modify: `docs/superpowers/specs/2026-07-12-yolo11-elevator-button-design.md`
- Modify: `docs/superpowers/plans/2026-07-10-franka-recorder-ui.md`
- Modify: `docs/superpowers/plans/2026-07-10-franka-act-rollout.md`

- [x] **Step 1: Replace test expectations with the shared contract**

Import `DEFAULT_CONTROL_HOST` where a test verifies an implicit default:

```python
from hardware_test.franka.defaults import DEFAULT_CONTROL_HOST
```

Use `DEFAULT_CONTROL_HOST` in default-value fixtures and assertions. For tests
that intentionally verify explicit override propagation, use the hostname
`"test-controller"` in both input and assertion instead of a real address.

- [x] **Step 2: Document the single edit and override points**

Add this current README guidance and remove redundant `--control-host` flags
from commands that use the default:

````markdown
## Control Host Configuration

The default Franka control-service host lives in `hardware_test/franka/defaults.py`.
Change `DEFAULT_CONTROL_HOST` there once for a persistent project-wide change,
or override it for one shell without editing files:

```bash
export FRANKA_CONTROL_HOST=controller.example
```

An explicit `--control-host` still takes precedence for a single command.
````

Use symbolic shared-default wording in dated plans/specs so those records no
longer look like additional runtime configuration sources.

- [x] **Step 3: Verify the stale address is gone outside OMX worktrees**

Run:

```bash
rg -n --hidden -g '!.git/**' -g '!.omx/**' '192\.168\.1\.11' .
```

Expected: no matches.

### Task 4: Run regression and quality verification

**Files:**
- Verify all files changed in Tasks 1-3

- [x] **Step 1: Run focused Franka tests**

Run:

```bash
uv run pytest \
  hardware_test/franka/test_franka_control_host_defaults.py \
  hardware_test/franka/test_maintenance_scripts.py \
  hardware_test/franka/test_act_rollout.py \
  hardware_test/franka/test_act_rollout_realsense.py \
  hardware_test/franka/test_franka_adapters.py \
  hardware_test/franka/test_franka_record_ui_app.py -q
```

Execution result: the exact staged snapshot passed 43 tests with 4 RealSense
driver-loading tests deselected, and the host suite passed all 5 tests again
with `FRANKA_CONTROL_HOST=ci-controller`. The deselected tests currently stop
inside `pyrealsense2` while initializing the local udev monitor, before any
control-host assertion; that hardware-driver condition is outside this change.

- [x] **Step 2: Run Ruff on changed Python files**

Run:

```bash
uv run ruff check --select F <all-staged-python-files>
uv run ruff check --select F,I \
  hardware_test/franka/defaults.py \
  hardware_test/franka/test_franka_control_host_defaults.py
uv run ruff format --check \
  hardware_test/franka/defaults.py \
  hardware_test/franka/test_franka_control_host_defaults.py
```

Expected: fatal-error checks pass for every staged Python file, and full
import/format checks pass for the newly added files. Repository-wide Ruff
still reports unrelated pre-existing style findings.

- [x] **Step 3: Check repository invariants and user-change preservation**

Run:

```bash
git diff --check
git diff -- hardware_test/franka/run_act_rollout.py
git status --short
```

Expected: no whitespace errors; the pre-existing `MAX_ACTION_SCALE = 0.8`
change remains; only task-related files plus the user's pre-existing untracked
files are present.

- [x] **Step 4: Commit the implementation with a Lore message**

Stage only task-owned files. Use a commit message whose intent is that future
Franka network changes require one configuration edit, with `Tested:` trailers
listing the exact pytest, Ruff, scan, and diff checks that passed.
