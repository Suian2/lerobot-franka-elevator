# Franka ACT Unconditioned RealSense Rollout Design

## Goal

Add a standalone rollout command for the existing unconditioned ACT checkpoint. The runner must read the L515 directly through LeRobot's `RealSenseCameraConfig`, control the Franka through the existing HTTP/ZMQ adapter, and never require or inject a target-floor feature.

## Scope

- Add `hardware_test/franka/run_act_rollout_realsense_unconditioned.py`.
- Preserve `run_act_rollout_realsense.py` as the multi-floor conditioned runner.
- Reuse the conditioned runner's policy loading, direct RealSense construction, bounded execution loop, action scaling, velocity limits, gripper suppression, signal handling, and dry-run safety gate.
- Remove only floor-specific behavior: the floor-conditioning imports, `--target-floor`, one-hot encoding, observation injection, floor logging, and the requirement that checkpoint input features contain `observation.environment_state`.
- Reject a conditioned checkpoint with a clear schema error rather than silently running it without its required input.
- Do not use ROS2 image topics, the ROS2-to-ZMQ image bridge, or `ZmqRgbImageClient`.

## Command Interface

The new command accepts the same arguments as the current direct-RealSense runner except `--target-floor`:

```bash
PYTHONPATH=src python hardware_test/franka/run_act_rollout_realsense_unconditioned.py \
  --policy-path /path/to/pretrained_model \
  --control-host 192.168.1.11 \
  --camera-serial-or-name "Intel RealSense L515" \
  --fps 30 \
  --duration-s 200 \
  --action-scale 0.8 \
  --max-linear-velocity 0.08 \
  --max-angular-velocity 0.10 \
  --execute
```

## Data Flow

1. Construct the L515 using `RealSenseCameraConfig` and attach it to `FrankaRobot` as camera key `l515`.
2. Read the eight-element robot state and `observation.images.l515`.
3. Validate that the checkpoint expects the state and image inputs and does not require the floor-conditioning feature.
4. Run the saved preprocessor, ACT policy, and saved postprocessor.
5. Suppress the gripper output and send the six bounded Cartesian commands through the existing Franka transport.

## Error Handling and Safety

- Keep all existing frequency, duration, scale, CUDA-only, velocity, stale-camera, transport, and `--execute` checks.
- Fail before robot execution when the checkpoint requires `observation.environment_state` or has incompatible state/image/action shapes.
- Keep teardown behavior unchanged so velocity control is stopped and the robot/camera are disconnected after errors or interruption.

## Verification

- Add focused tests proving the parser has no target-floor argument.
- Prove observation construction contains only state and image policy inputs.
- Prove unconditioned checkpoint features pass validation and conditioned checkpoint features fail clearly.
- Prove the robot builder uses direct `RealSenseCameraConfig` and no ROS2/ZMQ image client.
- Run the existing conditioned rollout tests unchanged to verify the multi-floor runner is unaffected.
- Run Ruff and Python compilation on the new files.

