# Franka + L515 Eye-to-Hand Calibration Design

## Goal and scope

Add an independent, non-ROS calibration workflow under
`hardware_test/franka/handeye/` that computes `T_base_camera`, the transform
from the Intel RealSense L515 color optical frame to the fixed Franka base
frame. The camera is fixed outside the robot and a 7 x 5 ChArUco target is
rigidly attached to the end effector.

The workflow only reads robot state. It must not instantiate a control
lifecycle that sends stop, zero-velocity, homing, gripper, or other motion
commands. It does not modify LeRobot datasets or the existing Franka service.

## Approaches considered

### Selected: independent CLIs with shared, testable utilities

The collector owns the RealSense pipeline and interactive UI. A shared utility
module owns configuration validation, transform math, ChArUco compatibility,
Franka pose parsing, sample persistence, calibration, and validation. The solve
and validate scripts are thin CLIs over those utilities.

This keeps the hardware loop easy to audit while allowing all mathematical and
persistence behavior to be tested without a camera or robot.

### Rejected: reuse the complete `FrankaRobot` and `RealSenseCamera` objects

`FrankaRobot.connect()` initializes state/control lifecycle pieces and
`disconnect()` sends a zero Cartesian command. That conflicts with the strict
read-only requirement. The generic RealSense wrapper also does not expose all
active color-profile intrinsics required by this calibration artifact.

### Rejected: add a new robot-state service or protocol

The existing `FrankaControlClient.get_curr()` already performs the required
read-only `GET /ctl/get_curr`. A second protocol would duplicate a proven
contract and violate the task boundary.

## Existing Franka state contract

The collector constructs the existing `FrankaControlClient` but calls only
`get_curr()` and `close()`. It never constructs `FrankaRobot`.

The active VITA server builds the response field as
`state.O_T_EE.matrix.tolist()`. libfranka defines `O_T_EE` as the measured end
effector pose in the base frame, so the nested HTTP value is `T_base_ee`: an
end-effector-frame point is mapped into the Franka base frame. The service has
already converted libfranka's column-major 16-value storage into a nested 4 x 4
matrix. Existing LeRobot hardware-test code reads translation from its fourth
column without scaling, confirming meter units.

Every read is validated before use:

- shape is nested 4 x 4, or a documented libfranka 16-value fallback;
- all values are finite;
- `R.T @ R` is approximately identity;
- `det(R)` is approximately one;
- the last row is approximately `[0, 0, 0, 1]`.

The original HTTP pose value is retained in `robot_pose_raw`. For a flat
libfranka fallback, column-major storage is used explicitly and recorded; the
parser never guesses from a variable name.

## Configuration

`config/l515_eye_to_hand.yaml` is the single source for board geometry and all
quality thresholds. It contains:

- exact `DICT_5X5_100`, 7 x 5 square counts, meter lengths, and
  `legacy_pattern`;
- ChArUco corner and reprojection thresholds;
- Laplacian blur threshold;
- one-second robot-still window and allowable motion within that window;
- pose-similarity warning thresholds;
- validation and leave-one-out outlier thresholds.

The loader rejects a non-`DICT_5X5_100` dictionary, invalid board dimensions,
non-meter/negative lengths, a marker not smaller than a square, and a minimum
corner count above the board maximum `(squares_x - 1) * (squares_y - 1)`.

CLI width, height, fps, serial, output directory, sample target, and control
host override runtime values without rewriting the source YAML. The fully
resolved values and OpenCV version are saved as `config_used.yaml`.

## RealSense acquisition and intrinsics

The collector imports `cv2`, `numpy`, `yaml`, and `pyrealsense2` at startup with
separate error messages. Missing `cv2.aruco` reports that an OpenCV build with
contrib modules is required and never installs anything. A present but
uninitializable RealSense SDK is reported as an SDK/device/udev failure rather
than as a missing package.

Only an RGB8 color stream is enabled through `pyrealsense2.pipeline`; no depth
stream, OpenCV capture device, ROS, topic, or image bridge is used. After 30
discarded frames, the active color video profile supplies actual width, height,
fps, `fx`, `fy`, `ppx`, `ppy`, distortion model and coefficients. The device
supplies the serial number. Those exact values and the corresponding 3 x 3
camera matrix are persisted in `camera_intrinsics.json`.

The collector rejects an active resolution that disagrees with the captured
frame. It does not resize frames or reuse intrinsics from another resolution.
Only RealSense distortion models that are safely compatible with OpenCV's
Brown-Conrady coefficient convention are accepted for pose estimation; an
unsupported model produces an explicit error instead of silently applying the
wrong coefficients.

## ChArUco compatibility and pose estimation

The board is built from the YAML geometry. `setLegacyPattern()` is applied when
requested and available. The running OpenCV version and legacy choice are
recorded, and detection/axis failures point the operator to that setting.

Detection prefers `cv2.aruco.CharucoDetector.detectBoard()`. If unavailable it
uses the legacy `detectMarkers()` plus `interpolateCornersCharuco()` path. Both
produce a common detection record containing marker corners/IDs and ChArUco
corners/IDs.

The board's `matchImagePoints()` maps detected ChArUco IDs to board-space meter
points. Older OpenCV builds fall back to indexing
`getChessboardCorners()` by the returned IDs. `solvePnP()` yields
`T_camera_board`, mapping board points into the L515 color optical frame.
Reprojection RMSE is computed with the same object/image correspondences.

The live overlay shows marker outlines, ChArUco corners and IDs, PnP axes,
corner count, reprojection RMSE, blur score, robot-still state, saved count, and
save eligibility. Errors between warning and maximum reprojection thresholds
are yellow; errors above the maximum are rejected.

## Safe sample capture and persistence

The live loop polls only `get_curr()` to maintain a time window of validated
`T_base_ee` values. A pose is considered stopped only when the history spans at
least one second and stays below configured translation and rotation motion
limits. Severe blur, stale/failed robot reads, insufficient corners, failed
PnP, or excessive reprojection error disables saving.

When `S` is pressed, the current RealSense color frame is frozen and a fresh
robot state is read immediately. The fresh pose is added to the stillness
window and all save conditions are checked again. Robot-read failure aborts the
whole sample. The original image, overlay, camera-board pose, and robot pose use
one sample ID and are committed to `samples.jsonl` as one logical bundle.

Existing samples are loaded on startup; new IDs never overwrite existing
artifacts. `D` removes the final manifest row and its paired files. `Q`/Escape
exits and `R` rebuilds the detector. Reaching the requested sample count only
shows completion; collection remains open.

The current pose is compared with existing robot poses. If both translation is
under 10 mm and rotation is under 5 degrees, the UI warns that it is too
similar. The first `S` arms confirmation and a second `S` forces the save, so
similarity never silently blocks an intentional sample.

## Eye-to-Hand solve

For each sample, the known transforms satisfy

```text
T_base_ee * T_ee_board = T_base_camera * T_camera_board
```

OpenCV's normal hand-eye inputs are `T_base_gripper` and
`T_camera_target`, and its output is `T_gripper_camera`. For this fixed-camera
Eye-to-Hand arrangement, frames are relabeled by passing
`T_ee_base = inverse(T_base_ee)` in the `gripper2base` argument position. Under
that relabeling OpenCV's logical gripper frame is the physical Franka base, so
its `cam2gripper` output is `T_base_camera`, not an end-effector-to-camera
transform. `T_camera_board` is passed unchanged as `target2cam`.

The solver runs TSAI, PARK, HORAUD, and DANIILIDIS independently. Invalid or
non-finite method outputs are reported without discarding successful methods.
The result YAML keeps every method matrix while exposing the recommended
method and matrix at the top level with explicit frame and meter metadata.

## Validation and recommendation

For every method and sample:

```text
T_ee_board(i) = inverse(T_base_ee(i))
                * T_base_camera
                * T_camera_board(i)
```

A rigid target makes these transforms constant. Translation uses the arithmetic
mean; rotation uses an SVD-projected mean on SO(3). The report includes mean
translation, per-axis and scalar translation standard deviation, maximum
translation deviation, mean rotation, rotation-deviation standard deviation,
maximum rotation deviation, per-sample errors, and configured outliers.

Leave-one-out validation recalibrates each method after excluding each sample
and compares that estimate with the full-data `T_base_camera` using translation
and geodesic rotation deltas. Influential sample IDs are identified by configured
absolute thresholds and robust median/MAD thresholds.

The recommendation score combines target-rigidity scatter and leave-one-out
stability in millimetres and degrees; lower is better. Every raw component is
reported so the choice is auditable.

## Testing and verification

Hardware-independent tests cover:

- config defaults and validation;
- nested and column-major Franka pose parsing plus invalid matrices;
- transform inversion/composition and pose deltas;
- synthetic PnP/reprojection behavior;
- synthetic Eye-to-Hand recovery for all four OpenCV methods;
- rigid-target statistics, outlier detection, and leave-one-out stability;
- resume/delete sample persistence;
- CLI defaults and safe dependency failures;
- the Franka reader calling only `get_curr()`.

Final verification uses the existing `lerobot` conda Python for tests, Ruff,
bytecode compilation, and `--help`/dependency checks. Physical L515 display and
robot-state smoke tests remain hardware-gated and are reported separately.
