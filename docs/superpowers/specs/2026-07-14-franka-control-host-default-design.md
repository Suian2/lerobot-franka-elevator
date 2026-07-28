# Franka Control Host Default Design

## Goal

Keep the Franka control-service host in one runtime configuration module so a
network change no longer requires editing every command, adapter, test, and
example independently.

## Scope

- Add one canonical Franka control-host default under `hardware_test/franka/`.
- Allow `FRANKA_CONTROL_HOST` to override that default without editing files.
- Make every active Franka CLI and `FrankaRobotConfig` resolve its implicit
  value through the shared configuration.
- Keep the existing `--control-host` and `base_url` overrides working.
- Update the checked-in Franka backup and documentation so they do not retain a
  competing control-host literal.
- Leave loopback endpoints and other robots' network addresses unchanged.

## Design

`hardware_test/franka/defaults.py` owns `DEFAULT_CONTROL_HOST`, the environment
variable name, and `get_control_host()`. CLI parser factories call the helper
when constructing their arguments. `FrankaRobotConfig` uses it as a dataclass
default factory, so both direct Python construction and command-line entry
points follow the same source of truth.

Explicit command-line arguments remain highest priority, followed by the
environment variable, then the checked-in default. An empty environment value
is ignored so an accidentally exported empty variable cannot produce malformed
control URLs.

## Alternatives Considered

- A TOML or YAML file would also centralize the value, but adds path resolution,
  parsing, and failure modes for one setting.
- An environment-only setting avoids a checked-in default, but makes every new
  shell require setup and weakens out-of-the-box behavior.
- Importing the dataclass field from each parser couples CLI construction to a
  larger hardware module and still does not provide an environment override.

## Verification

- Regression tests assert direct config construction and all CLI families use
  the shared default.
- Tests cover environment and explicit CLI precedence.
- A repository scan confirms no competing Franka control-host address remains.
- Existing targeted Franka tests, lint, and static checks run after the change.
