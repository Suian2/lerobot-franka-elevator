# Existing `yolo11_franka` Environment Installation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete and verify the existing `/home/yanrihong/miniconda3/envs/yolo11_franka` environment for YOLO11 GPU training and inference without creating another Conda environment or mixing in ROS packages.

**Architecture:** Treat the existing Conda prefix as the only mutable Python environment in this phase. Every command enters it through a checked-in wrapper that removes ROS and user-site contamination; large CUDA wheels are downloaded to a persistent wheelhouse before offline installation, and a standalone verifier proves package pins, RTX 4090 CUDA execution, torchvision CUDA operators, YOLO11 inference, and the absence of ROS paths.

**Tech Stack:** Python 3.10.20, pip 26.1.2, PyTorch 2.7.1+cu126, torchvision 0.22.1+cu126, NumPy 1.26.4, Pillow 12.2.0, OpenCV headless 4.11.0.86, `ultralytics-opencv-headless` 8.4.92, YOLO11s.

---

## Scope and file map

This plan implements only the training/inference environment described in the
approved design. It does not create a Conda environment, install `rclpy`, source
ROS 2, start the L515, install PaddleOCR, clone `yolo_ros`, connect to Franky, or
move the FR3. Those are separate, independently testable plans after this one.

Files created by this plan:

- `hardware_test/elevator_button/requirements-torch-cu126.txt` — direct CUDA
  PyTorch wheel pins installed only from the PyTorch CUDA 12.6 index.
- `hardware_test/elevator_button/requirements-yolo.txt` — direct YOLO runtime
  pins installed from PyPI after CUDA PyTorch is present.
- `hardware_test/elevator_button/run_in_yolo_env.sh` — the single entry point
  that selects the existing interpreter and removes inherited ROS paths.
- `hardware_test/elevator_button/verify_yolo_env.py` — executable acceptance
  test for isolation, versions, CUDA kernels, torchvision NMS, and YOLO11s.
- `hardware_test/elevator_button/README.md` — short operator instructions and
  explicit phase boundaries.

Machine-local evidence is written beneath
`outputs/hardware_test/yolo11_env/` and reusable downloads beneath
`/home/yanrihong/.cache/yolo11_franka/`. Neither location defines source code.

### Task 1: Add the isolation contract before installing packages

**Files:**
- Create: `hardware_test/elevator_button/requirements-torch-cu126.txt`
- Create: `hardware_test/elevator_button/requirements-yolo.txt`
- Create: `hardware_test/elevator_button/run_in_yolo_env.sh`
- Create: `hardware_test/elevator_button/verify_yolo_env.py`
- Create: `hardware_test/elevator_button/README.md`

- [ ] **Step 1: Pin the two package layers**

Create `hardware_test/elevator_button/requirements-torch-cu126.txt` with exactly:

```text
# Install only with https://download.pytorch.org/whl/cu126 as index-url.
torch==2.7.1+cu126
torchvision==0.22.1+cu126
```

Create `hardware_test/elevator_button/requirements-yolo.txt` with exactly:

```text
# Install after the CUDA PyTorch layer is verified.
numpy==1.26.4
pillow==12.2.0
opencv-python-headless==4.11.0.86
ultralytics-opencv-headless==8.4.92
```

- [ ] **Step 2: Add the environment entry-point wrapper**

Create `hardware_test/elevator_button/run_in_yolo_env.sh` with exactly:

```bash
#!/usr/bin/env bash
set -euo pipefail

readonly ENV_PREFIX="/home/yanrihong/miniconda3/envs/yolo11_franka"
readonly PYTHON_BIN="${ENV_PREFIX}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing target interpreter: ${PYTHON_BIN}" >&2
  exit 2
fi

unset PYTHONPATH
unset LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1

exec "${PYTHON_BIN}" "$@"
```

Make only this wrapper executable:

```bash
chmod 0755 hardware_test/elevator_button/run_in_yolo_env.sh
```

- [ ] **Step 3: Write the acceptance verifier before installation**

Create `hardware_test/elevator_button/verify_yolo_env.py` with exactly:

```python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path


EXPECTED_PREFIX = Path("/home/yanrihong/miniconda3/envs/yolo11_franka")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prefix = Path(sys.prefix).resolve()
    require(prefix == EXPECTED_PREFIX, f"wrong Python prefix: {prefix}")
    require(os.environ.get("PYTHONPATH") is None, "PYTHONPATH was not removed")
    require(
        os.environ.get("LD_LIBRARY_PATH") is None,
        "LD_LIBRARY_PATH was not removed",
    )
    require(os.environ.get("PYTHONNOUSERSITE") == "1", "user site is enabled")

    contaminated_paths = [
        entry
        for entry in sys.path
        if entry
        and (entry.startswith("/opt/ros/") or "/.local/lib/python" in entry)
    ]
    require(not contaminated_paths, f"contaminated sys.path: {contaminated_paths}")
    require(importlib.util.find_spec("rclpy") is None, "rclpy leaked into YOLO env")

    import cv2
    import numpy as np
    import PIL
    import torch
    import torchvision
    import ultralytics
    from torchvision.ops import nms
    from ultralytics import YOLO

    expected_versions = {
        "torch": "2.7.1+cu126",
        "torchvision": "0.22.1+cu126",
        "numpy": "1.26.4",
        "pillow": "12.2.0",
        "opencv": "4.11.0",
        "ultralytics": "8.4.92",
    }
    actual_versions = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "opencv": cv2.__version__,
        "ultralytics": ultralytics.__version__,
    }
    require(actual_versions == expected_versions, f"version drift: {actual_versions}")
    require(torch.version.cuda == "12.6", f"unexpected CUDA runtime: {torch.version.cuda}")
    require(torch.cuda.is_available(), "torch.cuda.is_available() is false")

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    require("4090" in device_name, f"unexpected GPU: {device_name}")
    require(capability == (8, 9), f"unexpected compute capability: {capability}")

    torch.manual_seed(0)
    matrix = torch.randn((512, 512), device="cuda")
    checksum = (matrix @ matrix.T).mean().item()
    torch.cuda.synchronize()
    require(math.isfinite(checksum), "CUDA matrix result is not finite")

    boxes = torch.tensor(
        [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]],
        dtype=torch.float32,
        device="cuda",
    )
    scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
    kept = nms(boxes, scores, 0.5).cpu().tolist()
    require(kept == [0, 2], f"unexpected torchvision NMS result: {kept}")

    weights = args.weights.expanduser().resolve()
    require(weights.is_file(), f"missing YOLO weights: {weights}")
    model = YOLO(str(weights))
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model.predict(source=image, imgsz=640, device=0, verbose=False)
    require(len(results) == 1, f"unexpected YOLO result count: {len(results)}")

    report = {
        "prefix": str(prefix),
        "versions": actual_versions,
        "cuda_runtime": torch.version.cuda,
        "gpu": device_name,
        "compute_capability": list(capability),
        "cuda_checksum": checksum,
        "torchvision_nms_kept": kept,
        "weights": str(weights),
        "yolo_result_count": len(results),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Document the one-environment boundary**

Create `hardware_test/elevator_button/README.md` with exactly:

````markdown
# Elevator-button YOLO environment

This directory bootstraps the existing Conda prefix:

`/home/yanrihong/miniconda3/envs/yolo11_franka`

It does not create another environment. Invoke Python and pip through
`run_in_yolo_env.sh` so ROS 2 overlays and user-site packages cannot leak in.

```bash
hardware_test/elevator_button/run_in_yolo_env.sh -m pip --version
hardware_test/elevator_button/run_in_yolo_env.sh \
  hardware_test/elevator_button/verify_yolo_env.py \
  --weights /home/yanrihong/.cache/yolo11_franka/models/yolo11s.pt
```

ROS 2, the L515 driver, PaddleOCR, `yolo_ros`, Franky control, and robot motion
are outside this installation phase. Do not source `/opt/ros/humble/setup.bash`
before running these commands.
````

- [ ] **Step 5: Verify the test is RED for the expected reason**

Run:

```bash
hardware_test/elevator_button/run_in_yolo_env.sh \
  hardware_test/elevator_button/verify_yolo_env.py \
  --weights /home/yanrihong/.cache/yolo11_franka/models/yolo11s.pt
```

Expected: failure at `import torch` with `ModuleNotFoundError`. Any prefix,
`PYTHONPATH`, or `LD_LIBRARY_PATH` failure must be fixed before installation.

- [ ] **Step 6: Commit the isolation harness**

```bash
git add hardware_test/elevator_button
git commit -m "Keep YOLO dependencies inside the existing training prefix" \
  -m "Add pinned package layers and a sanitized entry point so CUDA training cannot inherit ROS or user-site Python paths." \
  -m "Constraint: Reuse /home/yanrihong/miniconda3/envs/yolo11_franka; do not create another environment
Rejected: Install through an activated ROS shell | PYTHONPATH and LD_LIBRARY_PATH contamination
Confidence: high
Scope-risk: narrow
Directive: Keep ROS, OCR deployment, and robot control out of this training prefix
Tested: Wrapper selects the existing Python prefix and verifier fails before torch installation
Not-tested: CUDA and YOLO inference until package installation completes"
```

### Task 2: Snapshot the existing prefix and install CUDA PyTorch from a persistent wheelhouse

**Files:**
- Runtime output: `outputs/hardware_test/yolo11_env/before-pip-freeze.txt`
- Runtime output: `outputs/hardware_test/yolo11_env/before-pip-inspect.json`
- Runtime output: `outputs/hardware_test/yolo11_env/cu126-wheel-sha256.txt`
- Wheel cache: `/home/yanrihong/.cache/yolo11_franka/wheelhouse/cu126/`

- [ ] **Step 1: Prove pip supports resumable downloads and record capacity**

Run:

```bash
hardware_test/elevator_button/run_in_yolo_env.sh -m pip install --help
df -h /home/yanrihong
```

Expected: pip help contains `--resume-retries`; `/home/yanrihong` has at least
15 GB free. The observed baseline is pip 26.1.2 and roughly 198 GB free.

- [ ] **Step 2: Capture the clean pre-install snapshot**

Run:

```bash
mkdir -p outputs/hardware_test/yolo11_env
hardware_test/elevator_button/run_in_yolo_env.sh -m pip list --format=freeze | tee outputs/hardware_test/yolo11_env/before-pip-freeze.txt
hardware_test/elevator_button/run_in_yolo_env.sh -m pip inspect | tee outputs/hardware_test/yolo11_env/before-pip-inspect.json
```

Expected: the freeze contains packaging tools only and does not contain
`torch`, `torchvision`, `ultralytics`, `opencv`, `numpy`, or `pillow`.

- [ ] **Step 3: Download all CUDA wheels without disabling pip's cache**

Run:

```bash
mkdir -p /home/yanrihong/.cache/yolo11_franka/wheelhouse/cu126
hardware_test/elevator_button/run_in_yolo_env.sh -m pip download \
  --dest /home/yanrihong/.cache/yolo11_franka/wheelhouse/cu126 \
  --index-url https://download.pytorch.org/whl/cu126 \
  --timeout 600 \
  --retries 20 \
  --resume-retries 20 \
  -r hardware_test/elevator_button/requirements-torch-cu126.txt
```

Expected: exit code 0 and complete local wheels for torch, torchvision, Triton,
and the CUDA 12 dependencies. If the network drops, run the identical command
again; completed files remain in the wheelhouse and partial downloads can
resume. Do not add `--no-cache-dir`.

- [ ] **Step 4: Hash the wheelhouse before installation**

Run:

```bash
find /home/yanrihong/.cache/yolo11_franka/wheelhouse/cu126 -type f -name '*.whl' -print0 | sort -z | xargs -0 sha256sum | tee outputs/hardware_test/yolo11_env/cu126-wheel-sha256.txt
```

Expected: one SHA-256 row per wheel, including torch 2.7.1+cu126 and
torchvision 0.22.1+cu126.

- [ ] **Step 5: Install only from the completed local wheelhouse**

Run:

```bash
hardware_test/elevator_button/run_in_yolo_env.sh -m pip install \
  --no-index \
  --find-links /home/yanrihong/.cache/yolo11_franka/wheelhouse/cu126 \
  -r hardware_test/elevator_button/requirements-torch-cu126.txt
```

Expected: exit code 0 with no network access during installation.

- [ ] **Step 6: Run the CUDA-layer smoke check**

Run:

```bash
hardware_test/elevator_button/run_in_yolo_env.sh -c 'import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); assert torch.cuda.is_available()'
```

Expected output includes `2.7.1+cu126`, `0.22.1+cu126`, `12.6`,
`NVIDIA GeForce RTX 4090`, and `(8, 9)`.

### Task 3: Install the pinned headless YOLO layer and cache YOLO11s

**Files:**
- Model cache: `/home/yanrihong/.cache/yolo11_franka/models/yolo11s.pt`

- [ ] **Step 1: Install direct YOLO pins after CUDA PyTorch is present**

Run:

```bash
hardware_test/elevator_button/run_in_yolo_env.sh -m pip install \
  --index-url https://pypi.org/simple \
  --timeout 600 \
  --retries 20 \
  --resume-retries 20 \
  --prefer-binary \
  -r hardware_test/elevator_button/requirements-yolo.txt
```

Expected: pip keeps the already installed CUDA torch/torchvision builds,
installs the exact direct pins, and exits 0. Re-run the identical command after
a broken connection; pip's cache remains enabled.

- [ ] **Step 2: Check dependency consistency before downloading a model**

Run:

```bash
hardware_test/elevator_button/run_in_yolo_env.sh -m pip check
```

Expected: `No broken requirements found.`

- [ ] **Step 3: Download and load the upstream YOLO11s checkpoint in the cache directory**

Run from `/home/yanrihong/.cache/yolo11_franka/models`:

```bash
mkdir -p /home/yanrihong/.cache/yolo11_franka/models
cd /home/yanrihong/.cache/yolo11_franka/models
/home/yanrihong/lerobot/lerobot/hardware_test/elevator_button/run_in_yolo_env.sh -c 'from ultralytics import YOLO; model = YOLO("yolo11s.pt"); print(model.model.yaml.get("model", "yolo11s"))'
```

Expected: `yolo11s.pt` exists and is loadable by Ultralytics 8.4.92. This is an
upstream Ultralytics asset, not a custom checkpoint.

### Task 4: Prove the environment is isolated and GPU-ready

**Files:**
- Runtime output: `outputs/hardware_test/yolo11_env/verification.json`
- Runtime output: `outputs/hardware_test/yolo11_env/after-pip-freeze.txt`
- Runtime output: `outputs/hardware_test/yolo11_env/after-pip-inspect.json`
- Runtime output: `outputs/hardware_test/yolo11_env/pip-check.txt`
- Runtime output: `outputs/hardware_test/yolo11_env/model-sha256.txt`

- [ ] **Step 1: Run the full acceptance verifier and make it GREEN**

Run:

```bash
hardware_test/elevator_button/run_in_yolo_env.sh \
  hardware_test/elevator_button/verify_yolo_env.py \
  --weights /home/yanrihong/.cache/yolo11_franka/models/yolo11s.pt | tee outputs/hardware_test/yolo11_env/verification.json
```

Expected: exit code 0; JSON identifies the existing Conda prefix, all exact
versions, CUDA 12.6, RTX 4090, compute capability 8.9, NMS result `[0, 2]`, and
one YOLO result object.

- [ ] **Step 2: Prove the wrapper defeats a deliberately contaminated shell**

Run:

```bash
PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages LD_LIBRARY_PATH=/opt/ros/humble/lib hardware_test/elevator_button/run_in_yolo_env.sh -c 'import os, sys; assert os.environ.get("PYTHONPATH") is None; assert os.environ.get("LD_LIBRARY_PATH") is None; assert not any(path.startswith("/opt/ros/") for path in sys.path); print(sys.prefix)'
```

Expected: `/home/yanrihong/miniconda3/envs/yolo11_franka` and exit code 0.

- [ ] **Step 3: Archive final package and model provenance**

Run:

```bash
hardware_test/elevator_button/run_in_yolo_env.sh -m pip freeze | tee outputs/hardware_test/yolo11_env/after-pip-freeze.txt
hardware_test/elevator_button/run_in_yolo_env.sh -m pip inspect | tee outputs/hardware_test/yolo11_env/after-pip-inspect.json
hardware_test/elevator_button/run_in_yolo_env.sh -m pip check | tee outputs/hardware_test/yolo11_env/pip-check.txt
sha256sum /home/yanrihong/.cache/yolo11_franka/models/yolo11s.pt | tee outputs/hardware_test/yolo11_env/model-sha256.txt
```

Expected: `pip check` reports no broken requirements; the freeze contains all
six direct pins at the planned versions; the model has a recorded SHA-256.

- [ ] **Step 4: Inspect the final repository diff and status**

Run:

```bash
git diff --check
git status --short --branch
```

Expected: the plan's source commit contains only the five files under
`hardware_test/elevator_button/`; machine-local evidence under `outputs/` is
ignored and the working tree is clean after the Task 1 commit.

## Completion criteria

This phase is complete only when all of the following are simultaneously true:

- the interpreter prefix remains the pre-existing `yolo11_franka` directory;
- no other Conda environment or virtualenv was created;
- the package versions exactly match the approved design;
- `pip check` is clean;
- CUDA matrix multiplication and torchvision CUDA NMS pass on the RTX 4090;
- YOLO11s performs one GPU inference;
- `rclpy`, `/opt/ros`, user-site Python, and inherited ROS library paths are
  absent from the process;
- before/after package metadata, wheel hashes, model hash, and verification JSON
  are archived;
- no camera, ROS controller, Franky client, or physical robot motion was started.
