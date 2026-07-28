# Franka Control Server Launcher Design

## Goal

Start and inspect the VITA-compatible Franka Docker control server from this
repository while keeping the control-machine address owned by
`hardware_test/franka/defaults.py`.

## Scope

- Change the checked-in Franka control host to `192.168.1.11`.
- Move the remaining active hand-eye collector default onto the shared resolver.
- Add a hardware-test shell command for starting, inspecting, and stopping the
  already-deployed Docker server on the realtime control machine.
- Preserve temporary overrides through `FRANKA_CONTROL_HOST`,
  `FRANKA_CONTROL_REMOTE`, and the existing server tuning environment variables.
- Document the remote prerequisite; do not copy the Docker image or the remote
  runtime into this repository.

## Design

`hardware_test.franka.defaults.get_control_host()` remains the only checked-in
source for the control-machine address. The shell launcher obtains its default
host by invoking that resolver with the repository root on `PYTHONPATH`, then
constructs the default SSH target as `franka@<control-host>`. It never embeds a
second IP literal.

The launcher ports only the control-server portion of VITA's
`start_franka_teleop_stack.sh`. `start-control` replaces stale remote Docker and
tmux instances, starts the physically tested `zmq-franky-tuned` image through
the remote `run_franka_server_docker_control_machine.sh`, waits for `/config`,
and prints the returned configuration. `status` reports the remote process and
HTTP/ZMQ health. `stop-control` removes only that remote container and tmux
session. SSH explicitly bypasses proxies, matching the current VITA path.

## Error Handling and Safety

The script fails on missing local commands, SSH failures, or a server that does
not become healthy within the configured wait. On a health timeout it prints
the tail of the remote server log. `stop-control` is explicit; merely asking for
help or status does not stop or replace the server.

## Alternatives Considered

- Calling the VITA workspace script directly would avoid duplicated lifecycle
  commands but would make LeRobot depend on a separate checkout and path.
- Copying `192.168.1.11` into Bash is simpler but violates the requested
  one-edit configuration contract.
- A Python SSH implementation would be easier to unit test but either adds a
  dependency or recreates mature `ssh` behavior unnecessarily.

## Verification

- Parser regression tests cover the hand-eye collector under default and
  `FRANKA_CONTROL_HOST` override conditions.
- A subprocess test runs the shell launch path with fake `ssh` and `curl`
  commands and verifies it resolves the shared default and emits the expected
  remote Docker start command without touching hardware.
- Shell syntax, targeted pytest, Ruff, repository scans, and diff checks run
  before completion.
