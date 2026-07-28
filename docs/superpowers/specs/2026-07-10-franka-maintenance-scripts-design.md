# Franka Maintenance Scripts Design

## Goal

Add two independently runnable terminal scripts under `hardware_test/franka` that reproduce the VITA teleoperation UI's Franka fault recovery and hard-coded home motion.

## Interface

- `recover_fault.py` calls the VITA-compatible `GET /ctl/recover` endpoint, then reads and prints the shared velocity-loop status.
- `go_home.py` sends the VITA UI's seven joint targets to `POST /ctl/joint_position_control` with absolute, synchronous motion and a 60-second default timeout.
- Both scripts use the shared `DEFAULT_CONTROL_HOST`, accept `--control-host` and `--base-url`, execute without interactive confirmation, print successful JSON replies, and return a nonzero status on errors.

## Architecture

Both scripts reuse `FrankaControlClient` from `hardware_test/franka/franka_robot.py`. A small shared CLI helper constructs the client and formats success/error output, keeping request validation consistent with the hardware-test adapter. The VITA UI home joints are exposed as a named constant so the robot configuration and terminal script cannot drift apart.

## Safety and Errors

The home script states the target and immediately issues the motion, as requested. HTTP failures and `{\"is_ok\": 0}` replies are caught at the CLI boundary, printed to stderr, and converted to exit status 1. No retry is performed because repeating a physical motion command implicitly would be unsafe.

## Tests

Unit tests inject a fake client and verify recovery/status calls, exact home joints, absolute synchronous motion, timeout propagation, success output, and nonzero failure behavior. Existing Franka adapter tests remain the regression suite.
