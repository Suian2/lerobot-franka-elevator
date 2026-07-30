# Franka + L515 Hardware Test Adapters

This folder keeps local Franka experiments outside `src/lerobot` so they can be
iterated without changing LeRobot's global robot/teleop registries.

## Components

- `FrankaRobot`: LeRobot `Robot` adapter for a 7-axis Franka arm, Robotiq-style
  gripper command, and L515 RGB observations.
- `FrankaSpaceMouseTeleop`: LeRobot `Teleoperator` adapter that reuses VITA's
  SpaceMouse ideas: spnav/libspnav input, deadband, motion timeout, axis mapping,
  gripper toggle, and home/reset button.
- `FrankaStateCache`: background state polling cache. `get_observation()` reads
  cached state instead of doing synchronous HTTP on the recording loop.
- `record_lerobot_dataset.py`: helpers to build LeRobotDataset features and
  frame dictionaries directly from the local robot/teleop adapters.

## Control Path

The default command path is VITA-compatible:

- high-rate Cartesian velocity commands use latest-only ZMQ;
- low-rate operations such as joint home, gripper, and recover-style endpoints
  remain HTTP-compatible;
- state polling is moved to a background cache with short timeouts.

## Control Host Configuration

The default Franka control-service host lives in
`hardware_test/franka/defaults.py`. Change `DEFAULT_CONTROL_HOST` there once for
a persistent project-wide change, or override it for one shell without editing
files:

```bash
export FRANKA_CONTROL_HOST=controller.example
```

An explicit `--control-host` still takes precedence for a single command.

## Control Server Lifecycle

The control-machine Docker runtime must already be deployed at
`/home/franka/franka_ws/base/teleop/docker`, including the executable
`run_franka_server_docker_control_machine.sh`. From the LeRobot repository,
start the same VITA-compatible `franky` + ZMQ server with:

```bash
./hardware_test/franka/scripts/start_franka_control_server.sh start-control
```

Inspect health or stop only the remote control server with:

```bash
./hardware_test/franka/scripts/start_franka_control_server.sh status
./hardware_test/franka/scripts/start_franka_control_server.sh stop-control
```

The launcher imports `get_control_host()` from `defaults.py`, so it does not
carry another checked-in IP address. It connects to `franka@<control-host>` by
default and bypasses SSH/HTTP proxies. Temporary overrides remain available:

```bash
export FRANKA_CONTROL_HOST=controller.example
export FRANKA_CONTROL_REMOTE=operator@controller.example
export FRANKA_CONTROL_PASSWORD='...'  # omit when SSH keys work
```

The image, remote directory, tmux session, container name, and franky dynamics
can also be overridden with the environment variables printed by the script's
`--help` output.

## Fault Recovery and Home Scripts

Clear Franka errors and reset the shared VITA velocity-loop fault state:

```bash
PYTHONPATH=src /home/yanrihong/miniconda3/envs/lerobot/bin/python \
  hardware_test/franka/recover_fault.py
```

Move immediately to the same seven-joint absolute home pose used by the VITA
teleoperation UI:

```bash
PYTHONPATH=src /home/yanrihong/miniconda3/envs/lerobot/bin/python \
  hardware_test/franka/go_home.py
```

`go_home.py` deliberately has no interactive confirmation. It sends a
synchronous motion request as soon as it starts and waits up to 60 seconds by
default. Override that wait with `--timeout-s`, or use `--base-url` when the
control endpoint is not `http://<control-host>:29000/ctl`.

## Recording Behavior

Camera reads default to `read_latest(max_age_ms=...)`. If the L515 stops
producing frames, `get_observation()` raises instead of returning the previous
image again. This prevents an episode from silently recording hundreds of
duplicate stale frames.

The intended write path is LeRobotDataset v3:

- scalar state/action data goes to Parquet;
- RGB observations go to image/video storage through LeRobot's writer;
- action/image/state alignment is fixed when one frame dict is passed to
  `dataset.add_frame()`, not when images are physically written to disk.

## Minimal Use

```python
from hardware_test.franka import (
    FrankaRobot,
    FrankaRobotConfig,
    FrankaSpaceMouseTeleop,
    FrankaSpaceMouseTeleopConfig,
)
from hardware_test.franka.record_lerobot_dataset import (
    build_lerobot_features,
    make_lerobot_frame,
)

robot = FrankaRobot(
    FrankaRobotConfig(
        camera_shapes={"l515": (540, 960, 3)},
    )
)
teleop = FrankaSpaceMouseTeleop(FrankaSpaceMouseTeleopConfig())
```

## Smoke Recording Command

Dry-run the startup path without connecting to hardware:

```bash
PYTHONPATH=src /home/yanrihong/miniconda3/envs/lerobot/bin/python \
  hardware_test/franka/run_record.py \
  --dry-run-config \
  --camera-backend none
```

Record one 10 second L515 RGB episode with the direct RealSense backend:

```bash
PYTHONPATH=src python \
  hardware_test/franka/run_record.py \
  --repo-id local/franka_l515_smoke \
  --root outputs/hardware_test/franka_l515_smoke \
  --task "Franka SpaceMouse teleoperation" \
  --duration-s 10 \
  --num-episodes 1 \
  --camera-backend realsense \
  --camera-serial-or-name "Intel RealSense L515" \
  --streaming-encoding \
  --encoder-threads 2
```

## Recorder UI

The recorder UI follows the same ownership boundary as VITA:

- the shell script starts and monitors the application;
- Tkinter only displays state and submits local commands;
- one `RecorderWorker` owns the robot, SpaceMouse, cameras, control-server
  connection, and LeRobot dataset for the whole UI session.

Start the UI with the same hardware arguments used by `run_record.py`:

```bash
./hardware_test/franka/scripts/start_franka_record_ui.sh start-ui \
  --repo-id local/franka_l515_smoke \
  --root outputs/hardware_test/franka_l515_smoke \
  --task "Franka SpaceMouse teleoperation" \
  --duration-s 30 \
  --num-episodes 1 \
  --camera-backend realsense \
  --camera-serial-or-name "Intel RealSense L515" \
  --streaming-encoding \
  --encoder-threads 2
```

Lifecycle commands:

```bash
./hardware_test/franka/scripts/start_franka_record_ui.sh status
./hardware_test/franka/scripts/start_franka_record_ui.sh stop-ui
```

`stop-ui` sends a graceful interrupt and never force-kills the recorder while a
save/discard dialog is waiting on screen.

UI behavior:

- **开始录制** creates the dataset on first use and starts a new episode. The
  repo ID, root, and task fields are then locked for the rest of the session.
- **结束录制** stops Cartesian velocity and saves the current non-empty
  episode. A later Start creates another episode in the same dataset.
- **回到原位** outside recording moves Home without writing data. During
  recording it keeps capturing camera/state frames, labels the joint-controlled
  trajectory with measured end-effector deltas, and saves automatically after
  Home completes.
- **清除 Fault** pauses SpaceMouse input and calls the recorder backend's
  recovery operation. Success resumes the same episode; failure discards the
  pending episode.

The UI supports `delta_ee_pose` recording. It deliberately rejects joint action
mode and Hub upload. Dataset roots from a previous process are still rejected;
choose a new root for each UI session.

The default recording profile is tuned for low-stutter collection:

- collection/control loop: `30 Hz`
- L515 RGB frames read directly through the RealSense SDK
- color profile metadata: `960x540@30Hz`, RGB uint8
- stale camera guard: latest-frame reads fail if frames are too old

The default camera backend reads the L515 directly through the RealSense SDK:

```text
--camera-backend realsense --camera-serial-or-name <serial-or-name>
```

This requires `pyrealsense2` in the active Python environment. The legacy
OpenCV backend remains available only as an explicit opt-in with
`--camera-backend opencv`.

These modules are not wired into `lerobot-record` CLI automatically. Use them
from a local script, or import this package before constructing configs manually.
