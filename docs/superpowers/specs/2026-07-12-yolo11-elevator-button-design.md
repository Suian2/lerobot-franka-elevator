# YOLO11 Elevator Button Perception and Franka Pressing Design

Date: 2026-07-12

Status: Approved in conversation as "方案 A"

## Context

The objective is to build a conventional, inspectable perception-and-control
pipeline that lets the existing Franka FR3 setup find an elevator button,
interpret its floor or function, estimate a safe three-dimensional contact
target, and press it. The implementation must be based primarily on maintained
open-source projects rather than a bespoke perception stack.

The local machine already provides the major hardware and control assets:

- Ubuntu 22.04, ROS 2 Humble, and an RTX 4090 with NVIDIA driver 580.126.09;
- an Intel RealSense L515 with an existing isolated ROS launch setup (the
  camera is physically online, but its ROS node is not currently running);
- an FR3 whose single active FCI owner is the Franky service on the shared
  Franka control host;
- a latest-only ROS-to-ZMQ image bridge and existing HTTP/ZMQ robot clients;
- 29 recent button-press episodes with 24,281 RGB frames at 960x540, plus 31
  older RGB-D demonstrations with 7,983 RGB frames at 640x480.

The current `yolo11_franka` Conda environment exists at
`/home/yanrihong/miniconda3/envs/yolo11_franka`, but it contains only its base
Python and packaging tools. A previous PyTorch installation stopped during a
Pillow download because the network stream broke before package installation.

## Goals

The system must:

- detect elevator buttons as a general object instead of encoding every floor
  as a separate detector class;
- recognize arbitrary floor labels with OCR;
- retain semantic detector classes for non-text icons such as door open, door
  close, alarm, phone, stop, call up, and call down;
- use aligned depth and camera calibration to estimate a button contact point
  and panel normal;
- transform the target into the live FR3 base frame;
- reuse the existing remote Franky control service without creating a second
  FCI owner;
- fail closed at every perception, calibration, and motion boundary;
- minimize manual annotation by using public labels, automatic format
  conversion, temporal pseudo-labels, and the existing local demonstrations;
- keep training, ROS perception, OCR, and real-time robot control dependency
  boundaries explicit and reproducible.

## Non-goals

- Do not replace the current Franky real-time controller.
- Do not start local `franka_ros2`, MoveIt controllers, or another libfranka
  client while the configured Franka control host owns FCI.
- Do not use an end-to-end imitation policy as the first button-selection and
  pressing implementation.
- Do not command the real robot as part of environment setup, model training,
  automated tests, or the first ROS perception smoke test.
- Do not claim zero human validation for a physical system that can press the
  wrong target. The goal is no bulk hand-drawing of boxes, not no safety audit.
- Do not upgrade the L515 stack merely to use the newest RealSense release.

## Open-source Baseline

### YOLO ROS integration

Use [`mgonzs13/yolo_ros`](https://github.com/mgonzs13/yolo_ros) release 4.6.1
as the perception baseline. It already
supports ROS 2 Humble, YOLO11, tracking, debug images, synchronized depth,
camera information, TF transforms, and 3D detections. Preserve its GPL-3.0
notices and keep the upstream source as a separately traceable Git remote.

Create a separate ROS workspace at:

`/home/yanrihong/yolo11_franka_ws`

Pin the upstream release rather than following `main`. Keep local changes in a
small integration branch. The local patch is limited to dependency locking,
L515 launch parameters, topic/frame configuration, and interfaces needed by
the target-fusion package. It must not fork or rewrite the detector core
without a demonstrated defect.

The upstream launch currently runs `uv sync` at launch time and pins
`ultralytics==8.4.6`. The local integration will remove launch-time dependency
mutation, use a checked-in lock, and test a current headless Ultralytics build
before changing the upstream pin. A failed compatibility test falls back to
the upstream 8.4.6 pin rather than forcing an upgrade.

Training and deployment must use the same Ultralytics version before the first
custom checkpoint is trained. Test 8.4.92 in both environments first. If the
upstream ROS integration is incompatible, roll both environments back to
8.4.6; do not train on 8.4.92 and silently deploy on 8.4.6.

### OCR

Use [`PaddleOCR`](https://github.com/PaddlePaddle/PaddleOCR) 3.7 with
PP-OCRv6 for OCR. PP-OCRv6 is Apache-2.0 and is designed
for multilingual scene text, industrial text, dot-matrix characters, and
digital displays. Start with the small recognition model. Run OCR on a stable
panel crop or expanded button crop, not on every full-resolution frame.

OCR runs in a separate process and dependency environment. The first runtime
uses CPU/ONNX inference because OCR is event-driven and the cropped workload is
small; this avoids adding a second CUDA framework to the YOLO environment.
Only move OCR to GPU after profiling shows that it blocks the required control
latency.

OCR does not classify graphical icons or Braille. Those remain explicit
detector classes. Text recognition and icon classification are merged into one
normalized semantic target vocabulary.

### Camera and robot interfaces

Keep the existing L515 ROS setup and its tested wrapper. Standardize the first
integration on:

- `/l515/color/image_raw`;
- `/l515/color/camera_info`;
- `/l515/aligned_depth_to_color/image_raw`;
- `ROS_DOMAIN_ID=87`;
- `ROS_LOCALHOST_ONLY=1`;
- `rmw_fastrtps_cpp`.

The first profile is 960x540 RGB at 30 Hz with aligned depth. This matches the
newest local demonstrations and the currently measured color intrinsics. If
the median character height inside the panel crop is below 16 pixels, OCR is
not accepted as a model failure. The camera must instead be moved closer or
the entire color/calibration path must switch together to 1280x720. Intrinsics,
hand-eye results, labels, and pixels from different resolutions must never be
mixed.

The L515 currently runs firmware 1.5.4.1 and has prior depth/USB watchdog
failures. Before perception integration, measure RGB plus aligned-depth
stability with the current isolated stack. Firmware changes are a separate,
explicitly approved recovery action; they are not an automatic setup step.

Keep the remote Franky HTTP/ZMQ service as the only motion interface. The local
FR3 ROS and MoveIt installations are references and simulation tools only in
this phase.

## Process and Dependency Boundaries

The system is split into four processes:

1. **ROS camera process** — system ROS 2 Python/C++ publishes RGB, aligned
   depth, camera info, and TF.
2. **YOLO ROS process** — the pinned `yolo_ros` workspace consumes ROS images
   and publishes 2D/3D detections and debug output.
3. **OCR and target-fusion process** — receives stable detections, recognizes
   text/icons, estimates the panel plane and contact target, and publishes a
   validated `ButtonTarget` or a rejection reason.
4. **Press coordinator** — consumes only validated targets and talks to the
   existing Franky HTTP/ZMQ service through a narrow state machine.

Training uses the isolated `yolo11_franka` Conda environment. ROS binary
packages are not installed into that environment, and ROS `PYTHONPATH` and
`LD_LIBRARY_PATH` are removed from all training/install commands. Model
artifacts, dataset manifests, and a machine-readable class map are the
interface between training and deployment.

## YOLO Training Environment

Use the existing absolute interpreter:

`/home/yanrihong/miniconda3/envs/yolo11_franka/bin/python`

The initial lock is:

- Python 3.10.20;
- NumPy 1.26.4;
- Pillow 12.2.0;
- PyTorch 2.7.1 with CUDA 12.6 wheels;
- torchvision 0.22.1 with CUDA 12.6 wheels;
- OpenCV headless 4.11.0.86;
- `ultralytics-opencv-headless` 8.4.92.

Every environment command must:

- use the absolute Python path rather than a bare `pip`;
- set `PYTHONNOUSERSITE=1`;
- remove `PYTHONPATH` and `LD_LIBRARY_PATH` inherited from ROS overlays;
- keep pip caching enabled;
- use `--timeout 600 --retries 20 --resume-retries 20` for network installs.

System CUDA Toolkit and `nvcc` are not required because this design uses
prebuilt PyTorch wheels and no custom CUDA extensions.

Before installation, export an explicit package snapshot of the empty target
environment. After installation, save `pip freeze`, `pip inspect`, and a wheel
or download manifest so the environment can be reconstructed without relying
on future resolver behavior.

## Detection and Semantic Model

Start from `yolo11s.pt`. The initial class map is deliberately small:

- `button`;
- `door_open`;
- `door_close`;
- `alarm`;
- `phone`;
- `stop`;
- `call_up`;
- `call_down`.

All numeric, alphabetic, basement, and named floor buttons map to `button`.
Their meaning comes from OCR. This avoids hundreds of sparse detector classes
and permits floors absent from the training set.

Use object detection for the first model because the selected public datasets
provide bounding boxes. Do not pretend bounding boxes are masks. Instance
segmentation is a later measurable upgrade only if automatically generated
masks improve the 3D contact estimate on the held-out target-panel set.

The inference pipeline requires temporal agreement. A target is stable only
when the same tracked button and normalized semantic label appear in at least
five of the latest seven usable frames. Default gates are:

- YOLO confidence at least 0.60;
- OCR confidence at least 0.85 for text-selected buttons;
- no conflicting semantic result in the stability window;
- image, depth, and camera-info timestamps within the configured synchronization
  tolerance;
- source data age no greater than 250 ms at target publication.

Thresholds are configuration values, but lowering them below these initial
gates is rejected in real-motion mode.

## Public and Local Data Strategy

Use the following public data as seeds, retaining attribution and original
license metadata with every imported manifest:

1. [Sun Moon University Elevator Button Recognition](https://universe.roboflow.com/sun-moon-university/elevator_button_recognition/dataset/1):
   2,019 images, 368 bounding-box classes, published as CC BY 4.0, with YOLO
   export.
2. [School Elevator Button Detection v9](https://universe.roboflow.com/school-3rzal/elevator-button-detection-miikd/dataset/9):
   921 original images and 11 classes, published as CC BY 4.0. Do not count
   augmented v10 derivatives as new independent images.
3. [ACC Elevator Button](https://universe.roboflow.com/acc-stwam/elevator-button-jinxe-qb0gs):
   381 images and 38 classes, published as CC BY 4.0. This is an optional
   supplement only. Import it only if a versioned, reproducible export is
   available; baseline delivery and acceptance do not depend on it.

The old CUHK dataset is not an implementation dependency. Its original
download has expired, its repository license covers code rather than clearly
licensing the crawled image data, and its Python 2/TensorFlow implementation is
obsolete.

The Roboflow uploads do not fully document original image provenance. They are
acceptable for this research prototype with attribution, but any commercial
distribution requires a separate provenance and license audit.

Automatically transform public labels into the reduced class map. Preserve the
original class string as a candidate OCR transcription when it represents a
floor label. Split train/validation/test data by physical panel or source video,
never by random frame, so near-duplicate frames cannot leak across splits.

Use the local recordings as target-domain data:

- sample video frames at a bounded rate to avoid thousands of near duplicates;
- run the public-data seed detector and OCR teacher;
- accept pseudo-labels only when detector confidence, tracker consistency, and
  OCR consistency all pass;
- use robot action timestamps and before/after image pairs as weak evidence for
  pressed or illuminated state, never as unquestioned ground truth;
- route disagreements and rare icons to a review queue.

The user does not need to draw the full dataset from scratch. A 200-500-frame
panel-stratified audit set is still required for release and real motion. It is
primarily accept/reject/correct review. If that audit is skipped, training and
offline inference may continue, but the press coordinator remains disarmed.

## OCR and Button-to-Text Association

Run PaddleOCR on a rectified panel crop after YOLO has produced a stable panel
region. Associate recognized text boxes with button boxes using containment,
nearest-neighbor distance, row/column ordering, and one-to-one assignment.
Also try an expanded per-button crop for labels engraved inside or immediately
beside a button.

Normalize common results such as `B1`, `B2`, `G`, `LG`, signed floor numbers,
and Chinese/English floor strings. Apply an allowlist supplied by the requested
destination and visible panel results, but never replace a low-confidence OCR
result merely because one allowed value is close.

Use multi-frame consensus before accepting the association. A disagreement
between OCR and an explicit icon class is a fault, not a tie to resolve
silently.

## Three-dimensional Target Fusion

The upstream `yolo_ros` 3D box is diagnostic input, not the final pressing
point. The target-fusion component performs these steps:

1. Select valid depth samples from a shrunken central region of the button box.
2. Reject zero, saturated, stale, and median-absolute-deviation outliers.
3. Fit the surrounding panel plane while excluding button/bezel pixels.
4. Intersect the button-center camera ray with the fitted plane.
5. Orient the plane normal toward the camera and derive a pre-contact point
   40 mm away from the surface.
6. Transform the contact point, normal, and pre-contact point using the live,
   resolution-matched `T_base_camera`.
7. Track the result over time and require position standard deviation below
   5 mm in the accepted stability window.

The live base frame name is a launch parameter discovered from the running TF
tree; it is not hard-coded. Target publication is disabled unless the complete
camera-to-base transform is available at the image timestamp.

## Press Coordinator

The press coordinator is a fail-closed state machine:

`DISARMED -> TARGET_STABLE -> PRECONTACT -> APPROACH -> PRESS -> RETRACT -> VERIFY`

Any failed gate transitions to `FAULT`, sends a stop/zero command through the
existing control interface, and requires explicit re-arming.

Real motion remains disabled until all of the following artifacts exist and
pass validation:

- a resolution-matched camera intrinsic calibration;
- a final eye-to-hand `T_base_camera` result;
- a measured end-effector-to-contact-point transform;
- stable aligned depth at the selected camera profile;
- a physically cleared workspace and reachable pre-contact target;
- a working stop path to the Franky service;
- an operator-reviewed target audit set.

Initial motion limits are conservative hard maxima:

- 5 mm/s straight-line approach/press speed;
- 40 mm nominal pre-contact offset;
- 15 mm maximum commanded travel after reaching the panel estimate;
- 4 second approach/press timeout;
- immediate stop on stale data, target loss, non-finite geometry, HTTP/ZMQ
  failure, or robot fault;
- mandatory retract along the accepted panel normal after success or a
  recoverable abort.

Force/torque may become an additional stop condition only after the live Franky
API is verified to expose a fresh, correctly framed wrench signal. Displacement
and timeout limits remain mandatory even when force sensing is added.

The coordinator never sends gripper commands. The current Robotiq communication
fault must be repaired or explicitly shown irrelevant to a fixed, mechanically
safe pressing tool before physical execution.

## Error Handling

- Missing or stale RGB/depth/camera-info data: publish a structured rejection
  and do not retain the previous target.
- OCR ambiguity or multiple matching buttons: remain disarmed and request a
  new stable observation.
- Invalid depth or plane fit: reject the 3D target rather than substituting a
  nominal depth.
- Missing TF or mismatched calibration resolution: reject target publication.
- Network/controller error: send stop if possible, transition to `FAULT`, and
  never retry motion automatically.
- Model download or dependency failure: preserve completed files in cache or a
  wheelhouse and resume; never rebuild the Conda environment in place as an
  untracked workaround.
- L515 watchdog/USB failure: stop perception and require camera recovery before
  re-arming. Do not silently fall back from RGB-D to RGB-only motion.

## Verification Strategy

### Environment verification

- `pip check` reports no broken requirements.
- Imports resolve from the target environment, not ROS overlays or user site.
- PyTorch reports version 2.7.1, CUDA 12.6, the RTX 4090, and compute capability
  8.9.
- A CUDA tensor operation and torchvision NMS complete successfully.
- Ultralytics loads a YOLO11 checkpoint and performs inference on a recorded
  local frame without touching ROS or robot control.

### Offline perception verification

- Dataset manifests are reproducible and preserve source/license metadata.
- Panel-disjoint detector metrics are reported for `button` and every icon
  class.
- OCR accuracy is measured on the panel-stratified audit set.
- Semantic association is tested on text inside, above, below, and beside
  buttons.
- Recorded videos exercise temporal consensus, occlusion, reflections, robot
  self-occlusion, and illuminated/unilluminated states.

### ROS dry-run verification

- The configured L515 topics are live in domain 87 at the expected rate.
- RGB and aligned depth have consistent timestamps, dimensions, and camera
  model.
- YOLO publishes detections and debug images without downloading or mutating
  dependencies at launch.
- Target fusion publishes camera-frame diagnostics while robot output remains
  disabled.
- Missing depth, stale data, missing TF, and camera restart tests all fail
  closed.

### Geometry verification

Compare predicted contact points against independently measured reference
points across the usable panel. Real motion requires median absolute position
error below 5 mm and 95th-percentile error below 10 mm. A system that misses
these gates remains a perception demo.

### Motion verification

Motion testing is a later, explicitly armed phase. It progresses through fake
targets, live perception dry-run, pre-contact-only motion, compliant surface
tests away from the elevator, one low-speed button press, and repeated presses.
Every stage verifies stop, timeout, retract, and target-loss behavior before the
next stage.

## Delivery Phases

1. Install and verify the isolated YOLO11 training environment.
2. Create the separate ROS workspace and pin the upstream `yolo_ros` release.
3. Acquire, attribute, convert, and audit the public seed datasets.
4. Train the reduced-class YOLO11 detector and evaluate panel-disjoint data.
5. Add PP-OCRv6 and semantic button-to-text association.
6. Integrate live L515 RGB-D in ROS perception-only mode.
7. Complete and validate camera intrinsics, eye-to-hand, and contact TCP.
8. Add target fusion and offline/live geometry tests.
9. Add the disarmed press coordinator and test every failure path.
10. Conduct separately approved staged physical validation.

## Acceptance Criteria

The environment phase is accepted when the pinned packages install in the
correct Conda environment, CUDA inference succeeds on the RTX 4090, package
integrity checks pass, and an offline YOLO11 inference succeeds.

The perception phase is accepted when a requested floor or icon is selected by
detector plus OCR across panel-disjoint tests, live RGB-D produces a stable
camera-frame target, and every ambiguity/staleness condition rejects output.

The geometry phase is accepted only when calibrated base-frame targets meet the
specified physical error limits.

The real-robot phase is accepted only after the safety gates, stop behavior,
pre-contact-only test, low-speed press, retract, and post-press visual
verification all pass. Environment or perception success alone is never
reported as successful elevator-button pressing.

## Rollback and Isolation

- The existing `yolo11_franka` environment is snapshotted before package
  installation.
- New ROS perception work lives under `/home/yanrihong/yolo11_franka_ws` and
  does not modify `franka_ros2_ws` or the current L515 isolated workspace.
- Upstream repositories retain upstream remotes and pinned tags.
- Dataset downloads, converted data, and trained weights live outside Git with
  checksummed manifests.
- The existing Franky server, LeRobot recording path, ACT checkpoints, and
  robot configuration remain unchanged during the environment and perception
  phases.
- Any local open-source modification is kept in a small revertible commit with
  its upstream version and rationale recorded.

## Licensing and Attribution

`yolo_ros` is GPL-3.0 and Ultralytics YOLO11 is AGPL-3.0 unless a commercial
license is obtained. PaddleOCR, Franka ROS 2, and the RealSense ROS wrapper use
permissive licenses, but their notices must still be preserved. Public dataset
license and provenance metadata must accompany every generated manifest.

This design targets a research prototype. Commercial deployment requires a
separate review of Ultralytics licensing, GPL/AGPL integration obligations,
dataset provenance, and attribution before distribution.
