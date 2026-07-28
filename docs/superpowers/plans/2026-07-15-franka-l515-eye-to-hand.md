# Franka + L515 Eye-to-Hand Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a passive, non-ROS workflow that binds L515 color images to validated Franka `T_base_ee` states and solves, validates, and persists `T_base_camera` with four OpenCV hand-eye methods.

**Architecture:** Four executable files live in `hardware_test/franka/handeye/`. `collect_eye_to_hand.py` owns hardware acquisition and UI; `solve_eye_to_hand.py` and `validate_eye_to_hand.py` are thin offline CLIs; `handeye_utils.py` contains pure/configurable transform, ChArUco, persistence, calibration, and validation logic. Robot access reuses only `FrankaControlClient.get_curr()` so no control command enters the calibration path.

**Tech Stack:** Python 3.12, NumPy, OpenCV contrib/aruco, PyYAML, pyrealsense2, pytest, Ruff.

---

## File map

- Create `hardware_test/franka/handeye/__init__.py`: package marker only.
- Create `hardware_test/franka/handeye/config/l515_eye_to_hand.yaml`: board and quality thresholds.
- Create `hardware_test/franka/handeye/handeye_utils.py`: shared contracts, math, vision compatibility, robot parsing, persistence, solve, and validation.
- Create `hardware_test/franka/handeye/collect_eye_to_hand.py`: RealSense pipeline, live UI, keyboard flow, safe capture.
- Create `hardware_test/franka/handeye/solve_eye_to_hand.py`: four-method calibration and result/report persistence.
- Create `hardware_test/franka/handeye/validate_eye_to_hand.py`: independent recomputation of validation report.
- Create `hardware_test/franka/test_handeye_utils.py`: pure math, parsing, calibration, validation, and persistence tests.
- Create `hardware_test/franka/test_handeye_cli.py`: parser, dependency diagnostics, and passive-client tests.

No existing control service, dataset schema, camera wrapper, or robot class is modified.

### Task 1: Lock configuration and transform contracts

**Files:**
- Create: `hardware_test/franka/test_handeye_utils.py`
- Create: `hardware_test/franka/handeye/__init__.py`
- Create: `hardware_test/franka/handeye/config/l515_eye_to_hand.yaml`
- Create: `hardware_test/franka/handeye/handeye_utils.py`

- [ ] **Step 1: Write failing tests for the exact board and transform validation**

Tests load the shipped YAML and assert:

```python
config = load_handeye_config(CONFIG_PATH)
assert config["charuco"] == {
    "dictionary": "DICT_5X5_100",
    "squares_x": 7,
    "squares_y": 5,
    "square_length_m": 0.035,
    "marker_length_m": 0.026,
    "legacy_pattern": False,
}
assert max_charuco_corners(config["charuco"]) == 24
```

They also assert that wrong dictionaries, marker lengths greater than square
length, and corner thresholds above 24 raise `ValueError`; invalid SO(3), a
non-homogeneous final row, NaN values, and wrong shapes are rejected.

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=src /home/yanrihong/miniconda3/envs/lerobot/bin/python \
  -m pytest hardware_test/franka/test_handeye_utils.py -q
```

Expected: collection fails because `hardware_test.franka.handeye.handeye_utils`
does not exist.

- [ ] **Step 3: Add the YAML and minimal config/transform implementation**

Define these public interfaces with explicit coordinate names:

```python
def load_handeye_config(path: str | Path) -> dict[str, Any]: ...
def max_charuco_corners(charuco_config: Mapping[str, Any]) -> int: ...
def validate_homogeneous_transform(
    transform: Any, *, name: str, atol: float = 1e-6
) -> np.ndarray: ...
def make_transform(rotation: Any, translation_m: Any, *, name: str) -> np.ndarray: ...
def invert_transform(transform: Any, *, name: str) -> np.ndarray: ...
def rotation_delta_deg(rotation_a: Any, rotation_b: Any) -> float: ...
def pose_delta(T_base_ee_a: Any, T_base_ee_b: Any) -> tuple[float, float]: ...
```

The YAML includes exact task defaults plus `robot_stillness`,
`pose_similarity`, and `validation` thresholds. Every length key ends in `_m`
and every angle key ends in `_deg`.

- [ ] **Step 4: Run GREEN**

Run the Task 1 pytest command and expect all Task 1 tests to pass.

### Task 2: Lock the existing Franka pose contract and stillness monitor

**Files:**
- Modify: `hardware_test/franka/test_handeye_utils.py`
- Modify: `hardware_test/franka/handeye/handeye_utils.py`

- [ ] **Step 1: Add failing tests for nested and libfranka raw poses**

Exercise this API:

```python
reading = parse_franka_state_pose({"ee": T_base_ee.tolist()})
assert reading.robot_pose_name == "T_base_ee"
assert reading.translation_unit == "meter"
assert reading.matrix_storage_source == "existing_franka_client"
np.testing.assert_allclose(reading.T_base_ee, T_base_ee)
```

Also flatten the same matrix with `order="F"` and prove the documented
libfranka fallback reconstructs it. Test that missing `ee`, malformed raw data,
bad rotation, and bad final row fail clearly.

Add a fake client with counters and verify `FrankaPoseReader.read()` calls only
`get_curr(timeout=...)`; no method whose name contains `send`, `stop`, `move`,
`control`, `recover`, `gripper`, or `home` is invoked.

Test `RobotStillnessMonitor` with a one-second stationary window, a too-short
window, a translation excursion, and a rotation excursion.

- [ ] **Step 2: Run RED**

Run the targeted tests and expect missing symbol failures.

- [ ] **Step 3: Implement the passive reader and monitor**

Add:

```python
@dataclass(frozen=True)
class RobotPoseReading:
    T_base_ee: np.ndarray
    robot_pose_raw: Any
    robot_timestamp: float | str | None
    local_monotonic_s: float
    request_latency_ms: float
    robot_pose_name: str = "T_base_ee"
    translation_unit: str = "meter"
    matrix_storage_source: str = "existing_franka_client"

def parse_franka_state_pose(
    state: Mapping[str, Any], *, local_monotonic_s: float | None = None,
    request_latency_ms: float = 0.0,
) -> RobotPoseReading: ...

class FrankaPoseReader:
    def read(self) -> RobotPoseReading: ...
    def close(self) -> None: ...

class RobotStillnessMonitor:
    def add(self, timestamp_s: float, T_base_ee: Any) -> None: ...
    def status(self, now_s: float | None = None) -> StillnessStatus: ...
```

The reader keeps an undocumented timestamp as `None` unless the payload has an
explicitly recognized timestamp field; its local monotonic time is never
mislabelled as robot time.

- [ ] **Step 4: Run GREEN**

Run the targeted tests and then all of `test_handeye_utils.py`.

### Task 3: Lock ChArUco detection and PnP behavior

**Files:**
- Modify: `hardware_test/franka/test_handeye_utils.py`
- Modify: `hardware_test/franka/handeye/handeye_utils.py`

- [ ] **Step 1: Add failing vision tests**

Assert the shipped board has 24 chessboard corners and uses
`DICT_5X5_100`. Use `cv2.projectPoints()` on known board points to create image
points, then verify:

```python
estimate = estimate_board_pose(
    board=board,
    charuco_corners=projected_points,
    charuco_ids=ids,
    camera_matrix=camera_matrix,
    distortion_coefficients=np.zeros(5),
)
np.testing.assert_allclose(estimate.T_camera_board, expected, atol=1e-7)
assert estimate.reprojection_error_px < 1e-6
```

Test insufficient points, failed PnP, blur scoring, and both adapter branches by
using small fake aruco objects that expose only the new or legacy methods.

- [ ] **Step 2: Run RED**

Run the vision tests and confirm missing implementation failures.

- [ ] **Step 3: Implement the compatibility adapter and pose estimate**

Add:

```python
def create_charuco_board(charuco_config: Mapping[str, Any]) -> Any: ...

class CharucoDetectorCompat:
    api_name: str
    def detect(self, rgb_image: np.ndarray) -> CharucoDetection: ...

def match_charuco_image_points(
    board: Any, corners: np.ndarray, ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]: ...

def estimate_board_pose(
    *, board: Any, charuco_corners: np.ndarray, charuco_ids: np.ndarray,
    camera_matrix: np.ndarray, distortion_coefficients: np.ndarray,
) -> BoardPoseEstimate: ...

def draw_detection_overlay(
    rgb_image: np.ndarray, detection: CharucoDetection,
    estimate: BoardPoseEstimate | None, camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray, axis_length_m: float,
) -> np.ndarray: ...
```

`CharucoDetector` is selected first. The fallback calls
`detectMarkers`/`interpolateCornersCharuco`. `matchImagePoints()` is selected
first, with indexed chessboard points as the fallback. The overlay returned by
the utility is RGB; conversion to BGR happens only at OpenCV display/save
boundaries.

- [ ] **Step 4: Run GREEN**

Run all utility tests and expect pass.

### Task 4: Lock sample persistence, resume, similarity, and deletion

**Files:**
- Modify: `hardware_test/franka/test_handeye_utils.py`
- Modify: `hardware_test/franka/handeye/handeye_utils.py`

- [ ] **Step 1: Add failing persistence tests**

Within `tmp_path`, save two bundles and assert each manifest row points to an
original image and an overlay with the same ID. Reload the store and assert the
next ID does not overwrite either. Delete the last sample and assert only its
two image files and manifest row disappear.

Test `nearest_pose_delta()` against several existing `T_base_ee` values and
confirm the warning requires both translation and rotation thresholds.

- [ ] **Step 2: Run RED**

Run the persistence tests and confirm missing symbol failures.

- [ ] **Step 3: Implement transactional bundle helpers**

Add:

```python
def load_samples(input_dir: str | Path) -> list[dict[str, Any]]: ...
def next_sample_id(input_dir: str | Path, samples: Sequence[Mapping[str, Any]]) -> int: ...
def save_sample_bundle(
    output_dir: str | Path, sample: Mapping[str, Any],
    rgb_image: np.ndarray, rgb_overlay: np.ndarray,
) -> dict[str, Any]: ...
def delete_last_sample(output_dir: str | Path) -> dict[str, Any] | None: ...
def nearest_pose_delta(
    T_base_ee: Any, samples: Sequence[Mapping[str, Any]]
) -> PoseSimilarity | None: ...
```

PNG bytes are encoded before either final file is installed. The JSONL manifest
is rewritten through a temporary file and `os.replace`; exceptions roll back
new image files. Startup ignores no manifest error: malformed JSON, duplicate
IDs, or missing paired files is reported explicitly.

- [ ] **Step 4: Run GREEN**

Run all utility tests and expect pass.

### Task 5: Lock four-method Eye-to-Hand solve and validation

**Files:**
- Modify: `hardware_test/franka/test_handeye_utils.py`
- Modify: `hardware_test/franka/handeye/handeye_utils.py`

- [ ] **Step 1: Add a deterministic synthetic calibration test**

Choose fixed `T_base_camera` and `T_ee_board`, then generate at least 12 diverse
`T_base_ee(i)` poses. Compute exact observations with:

```python
T_camera_board_i = (
    invert_transform(T_base_camera, name="T_base_camera")
    @ T_base_ee_i
    @ T_ee_board
)
```

For TSAI, PARK, HORAUD, and DANIILIDIS, assert recovered translation and
geodesic rotation are within method-appropriate numerical tolerances. Assert
the validation recovers a constant `T_ee_board`, zero outliers, and stable
leave-one-out results. Add one perturbed board pose and assert its sample ID is
reported as an outlier/influential sample.

- [ ] **Step 2: Run RED**

Run only the synthetic solver tests and verify missing symbol failures.

- [ ] **Step 3: Implement the solve with the full coordinate derivation**

Add:

```python
HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}

def calibrate_eye_to_hand(
    samples: Sequence[Mapping[str, Any]], *, method: int
) -> np.ndarray: ...
def validate_eye_to_hand_result(
    samples: Sequence[Mapping[str, Any]], T_base_camera: Any,
    validation_config: Mapping[str, Any],
) -> dict[str, Any]: ...
def leave_one_out_validation(
    samples: Sequence[Mapping[str, Any]], full_T_base_camera: Any,
    method: int, validation_config: Mapping[str, Any],
) -> dict[str, Any]: ...
def solve_all_methods(
    samples: Sequence[Mapping[str, Any]], validation_config: Mapping[str, Any]
) -> dict[str, Any]: ...
```

Inside `calibrate_eye_to_hand`, explicitly compute
`T_ee_base = inverse(T_base_ee)` and pass its rotation/translation in OpenCV's
`gripper2base` slots. Pass `T_camera_board` in `target2cam`. A multi-line source
comment derives why the returned `cam2gripper` is physically
`T_base_camera` under this frame relabeling.

Use an SVD-projected SO(3) mean and geodesic deviations for validation.
Recommendation scoring reports its terms and retains every method.

- [ ] **Step 4: Run GREEN**

Run all utility tests and expect pass.

### Task 6: Implement the passive RealSense collector test-first

**Files:**
- Create: `hardware_test/franka/test_handeye_cli.py`
- Create: `hardware_test/franka/handeye/collect_eye_to_hand.py`

- [ ] **Step 1: Add failing parser and dependency tests**

Assert defaults are width 960, height 540, fps 30, target 20, control host
`192.168.1.5`, and the required output path. Verify `--camera-serial` and every
other requested option override defaults.

Inject import functions that simulate:

- missing `cv2`;
- OpenCV without `aruco`;
- missing `pyrealsense2`;
- present pyrealsense2 whose initialization raises a udev/device error.

Assert each message is actionable and no installer is invoked.

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=src /home/yanrihong/miniconda3/envs/lerobot/bin/python \
  -m pytest hardware_test/franka/test_handeye_cli.py -q
```

Expected: import failure for the missing collector module.

- [ ] **Step 3: Implement parser, dependency check, and pipeline profile capture**

The collector must use only:

```python
pipeline = rs.pipeline()
rs_config = rs.config()
if camera_serial:
    rs_config.enable_device(camera_serial)
rs_config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
profile = pipeline.start(rs_config)
```

Discard exactly 30 frames. Read active color profile width, height, fps,
intrinsics, distortion, coefficients, device serial, and SDK frame timestamp.
Persist the resolved camera matrix and never resize frames.

- [ ] **Step 4: Implement and test capture eligibility**

Factor a pure `evaluate_capture_eligibility()` that requires detection, minimum
corners, PnP success, reprojection under maximum, blur over threshold, a fresh
robot read, and one-second stillness. Unit-test every rejection reason and the
yellow reprojection warning state.

- [ ] **Step 5: Implement the live loop and key behavior**

The loop detects every frame, draws all requested diagnostics, and handles:

- `S`: freeze frame, immediately read/validate `T_base_ee`, then save the one
  bound bundle;
- second `S`: force-save a pose previously rejected only for similarity;
- `D`: delete the last complete bundle;
- `R`: rebuild the compatibility detector;
- `Q`/Escape: cleanly stop the RealSense pipeline and close the passive client.

Robot-state failure never writes an image. Reaching the target count adds a
completion banner but does not exit.

- [ ] **Step 6: Run GREEN**

Run both hand-eye test files and expect pass.

### Task 7: Implement solve and validation CLIs test-first

**Files:**
- Modify: `hardware_test/franka/test_handeye_cli.py`
- Create: `hardware_test/franka/handeye/solve_eye_to_hand.py`
- Create: `hardware_test/franka/handeye/validate_eye_to_hand.py`

- [ ] **Step 1: Add failing CLI integration tests**

Create a temporary input directory with resolved config, intrinsics, and
synthetic JSONL samples. Invoke both `main([...])` functions and assert:

- solve writes `result/T_base_camera.yaml` and
  `result/validation_report.json`;
- YAML top-level metadata is `franka_base`, `l515_color_optical_frame`,
  `T_base_camera`, `meter`, and the recommended method/matrix;
- YAML retains TSAI, PARK, HORAUD, and DANIILIDIS results;
- validation can reload that YAML and independently replace the report;
- too few samples and malformed transforms fail with clear messages.

- [ ] **Step 2: Run RED**

Run only the new integration tests and confirm missing module failures.

- [ ] **Step 3: Implement the two thin CLIs**

`solve_eye_to_hand.py` loads every valid sample, checks image dimensions against
the saved intrinsics, evaluates pose diversity, solves all methods, writes the
complete YAML and JSON report, and prints a concise per-method summary.

`validate_eye_to_hand.py` loads all stored method matrices, recomputes rigid
target and leave-one-out metrics, reruns recommendation, and writes the same
report schema without modifying the calibration samples.

- [ ] **Step 4: Run GREEN**

Run both hand-eye test files and expect pass.

### Task 8: Final verification and acceptance audit

**Files:**
- Review all files listed above.

- [ ] **Step 1: Run the complete focused suite**

```bash
PYTHONPATH=src /home/yanrihong/miniconda3/envs/lerobot/bin/python \
  -m pytest hardware_test/franka/test_handeye_utils.py \
  hardware_test/franka/test_handeye_cli.py -vv
```

Expected: zero failures.

- [ ] **Step 2: Run nearby Franka regressions**

```bash
PYTHONPATH=src /home/yanrihong/miniconda3/envs/lerobot/bin/python \
  -m pytest hardware_test/franka/test_franka_measured_home.py \
  hardware_test/franka/test_franka_control_host_defaults.py -q
```

Expected: zero failures.

- [ ] **Step 3: Run static verification**

```bash
uv run ruff check hardware_test/franka/handeye \
  hardware_test/franka/test_handeye_utils.py hardware_test/franka/test_handeye_cli.py
uv run ruff format --check hardware_test/franka/handeye \
  hardware_test/franka/test_handeye_utils.py hardware_test/franka/test_handeye_cli.py
PYTHONPATH=src /home/yanrihong/miniconda3/envs/lerobot/bin/python \
  -m compileall -q hardware_test/franka/handeye
```

Expected: all commands exit zero.

- [ ] **Step 4: Run no-hardware CLI smoke checks**

```bash
PYTHONPATH=src /home/yanrihong/miniconda3/envs/lerobot/bin/python \
  hardware_test/franka/handeye/collect_eye_to_hand.py --help
PYTHONPATH=src /home/yanrihong/miniconda3/envs/lerobot/bin/python \
  hardware_test/franka/handeye/solve_eye_to_hand.py --help
PYTHONPATH=src /home/yanrihong/miniconda3/envs/lerobot/bin/python \
  hardware_test/franka/handeye/validate_eye_to_hand.py --help
```

Expected: all requested flags appear and all commands exit zero without opening
a camera or contacting the robot.

- [ ] **Step 5: Audit forbidden behavior and output schema**

Search the new collector for `send_action`, velocity/joint/gripper control,
ROS, ZMQ image bridge, `VideoCapture`, and depth enablement. Confirm the only
Franka call is `get_curr()` and the only RealSense stream is color. Re-read the
15 acceptance requirements against the implementation and report any physical
hardware checks that remain unrun.

No git commit is created automatically in the shared dirty worktree. If the
user later requests a commit, it must use the repository's Lore trailer format.
