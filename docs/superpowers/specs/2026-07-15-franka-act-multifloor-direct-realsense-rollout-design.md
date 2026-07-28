# Franka ACT Multi-Floor Direct RealSense Rollout Design

## Goal

Run the trained floor-conditioned ACT checkpoint on the physical Franka for
floors 1, 4, and 5 while acquiring L515 RGB frames directly in the rollout
process. The rollout must not publish, subscribe to, import, or depend on ROS
image topics, the ROS2 image bridge, or the ZMQ image client.

## Selected approach

Keep `run_act_rollout_realsense.py` as a dedicated physical entry point. It
uses LeRobot's `RealSenseCamera`, whose implementation uses `pyrealsense2` for
capture and OpenCV/NumPy for image processing. This is narrower and safer for
the immediate hardware test than adding another camera backend to the existing
ROS/ZMQ rollout entry point.

The direct entry point preserves the conditioned rollout contract:

- require `--target-floor` and accept only 1, 4, or 5;
- encode the target through the shared `encode_target_floor` function;
- insert `observation.environment_state` before preprocessing;
- require a processed finite `torch.float32` tensor with shape `(1, 5)`;
- reject checkpoints missing the `(5,)` environment feature;
- reset the policy and processors at the start of every rollout;
- print the floor, one-hot vector, checkpoint, and direct camera identifier
  before robot connection.

## Camera path

`FrankaRobot` owns one `RealSenseCamera` configured for L515 RGB at
960x540, 30 Hz, with depth disabled. The unique device name
`Intel RealSense L515` is passed explicitly in hardware commands. The local
RealSense wrapper treats alphanumeric serial `f1381152` as a name, so passing
the detected unique device name is the compatible selection path. Frames remain
RGB `uint8` arrays compatible with the training feature
`observation.images.l515`.

## Safety and failure behavior

Dry-run remains the default and performs one real observation and inference
without sending the policy action. Physical motion requires `--execute`.
Every exit path attempts zero Cartesian velocity and disconnects the camera and
robot. Invalid floors, camera frames, checkpoint features, processed condition
shape/dtype, and non-finite actions fail before policy actions are sent.

The first physical commands use bounded parameters already accepted by the
rollout safety contract. Floors are run as separate processes so every change
of target necessarily resets ACT's cached action chunk.

## Verification

Automated tests cover parser floor restrictions, canonical condition
propagation, checkpoint feature validation, reset behavior, direct RealSense
configuration, and absence of ROS/ZMQ image dependencies. Before exposing
`--execute` commands, run a real checkpoint dry-run with the detected L515 and
verify the printed target, one-hot value, camera shape, processed `(1, 5)`
condition, and finite selected action.
