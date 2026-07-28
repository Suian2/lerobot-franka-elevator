#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PARENT_FRAME = "franka_base"
CHILD_FRAME = "l515_color_optical_frame"
TRANSFORM_NAME = "T_base_camera"
TRANSLATION_UNIT = "meter"
COORDINATE_DIRECTION = "Maps points from the L515 color optical camera frame to the Franka base frame."
TRANSFORM_MEANING = (
    "T_base_camera transforms point coordinates expressed in the L515 color optical camera "
    "frame into point coordinates expressed in the Franka base frame."
)
PROVENANCE_SCHEMA = "lerobot.handeye.calibration-inputs"
PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_DIGEST_ALGORITHM = "sha256"
PROVENANCE_CANONICALIZATION = "json-utf8-sort-keys-compact-finite-v1"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve an L515-to-Franka eye-to-hand calibration from persisted samples."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Capture directory containing config_used.yaml, camera_intrinsics.json, and samples.jsonl.",
    )
    return parser


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    try:
        loaded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except OSError as exc:
        raise OSError(f"Failed to read {path}: {exc}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a top-level JSON object")
    return loaded


def _finite_number(value: Any, *, context: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{context} must be a finite number")
    if positive and numeric <= 0.0:
        raise ValueError(f"{context} must be greater than zero")
    return numeric


def _positive_integer(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return int(value)


def _matrix_3x3(value: Any, *, context: str) -> list[list[float]]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
        or any(
            isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != 3 for row in value
        )
    ):
        raise ValueError(f"{context} must be a finite 3x3 matrix")
    return [
        [
            _finite_number(element, context=f"{context}[{row_index}][{column_index}]")
            for column_index, element in enumerate(row)
        ]
        for row_index, row in enumerate(value)
    ]


def _normalized_intrinsics(path: Path) -> dict[str, Any]:
    raw = _load_json_mapping(path)
    resolution = raw.get("resolution")
    if resolution is not None and not isinstance(resolution, Mapping):
        raise ValueError(f"{path}.resolution must be an object")

    nested_width = resolution.get("width") if isinstance(resolution, Mapping) else None
    nested_height = resolution.get("height") if isinstance(resolution, Mapping) else None
    width_value = nested_width if nested_width is not None else raw.get("width")
    height_value = nested_height if nested_height is not None else raw.get("height")
    width = _positive_integer(width_value, context=f"{path} actual width")
    height = _positive_integer(height_value, context=f"{path} actual height")
    if nested_width is not None and "width" in raw and raw["width"] != nested_width:
        raise ValueError(f"{path} has conflicting flat and resolution.width values")
    if nested_height is not None and "height" in raw and raw["height"] != nested_height:
        raise ValueError(f"{path} has conflicting flat and resolution.height values")

    nested_serial = raw.get("realsense_serial")
    flat_serial = raw.get("serial")
    serial = nested_serial if nested_serial is not None else flat_serial
    if not isinstance(serial, str) or not serial:
        raise ValueError(f"{path} must record a nonempty RealSense serial")
    if nested_serial is not None and flat_serial is not None and nested_serial != flat_serial:
        raise ValueError(f"{path} has conflicting serial and realsense_serial values")

    camera_matrix = _matrix_3x3(raw.get("camera_matrix"), context=f"{path}.camera_matrix")
    fps = raw.get("fps")
    if fps is None and isinstance(resolution, Mapping):
        fps = resolution.get("fps")
    if fps is not None:
        _positive_integer(fps, context=f"{path}.fps")

    distortion = raw.get("distortion")
    if distortion is not None and not isinstance(distortion, Mapping):
        raise ValueError(f"{path}.distortion must be an object")
    distortion_model = (
        distortion.get("model") if isinstance(distortion, Mapping) else raw.get("distortion_model")
    )
    distortion_coefficients = (
        distortion.get("coefficients")
        if isinstance(distortion, Mapping)
        else raw.get("distortion_coefficients")
    )
    if not isinstance(distortion_model, str) or not distortion_model:
        raise ValueError(f"{path} must record a nonempty distortion model")
    if (
        isinstance(distortion_coefficients, (str, bytes))
        or not isinstance(distortion_coefficients, Sequence)
        or len(distortion_coefficients) != 5
    ):
        raise ValueError(f"{path} must record exactly five distortion coefficients")
    coefficients = [
        _finite_number(value, context=f"{path} distortion coefficient {index}")
        for index, value in enumerate(distortion_coefficients)
    ]
    return {
        "width": width,
        "height": height,
        "serial": serial,
        "camera_matrix": camera_matrix,
        "distortion_model": distortion_model,
        "distortion_coefficients": coefficients,
    }


def _load_resolved_config(path: Path) -> dict[str, Any]:
    from hardware_test.franka.handeye.handeye_utils import load_handeye_config

    if not path.is_file():
        raise FileNotFoundError(f"Required hand-eye config does not exist: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OSError(f"Failed to read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a top-level mapping")

    if "resolved_config" not in document:
        try:
            return load_handeye_config(path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid hand-eye config {path}: {exc}") from exc

    resolved = document["resolved_config"]
    if not isinstance(resolved, dict):
        raise ValueError(f"{path}.resolved_config must be a mapping")
    with tempfile.TemporaryDirectory(prefix="lerobot-handeye-config-") as temporary_directory:
        extracted_path = Path(temporary_directory) / "resolved_config.yaml"
        try:
            extracted_path.write_text(
                yaml.safe_dump(
                    _to_builtin_finite(resolved, context=f"{path}.resolved_config"), sort_keys=False
                ),
                encoding="utf-8",
            )
            return load_handeye_config(extracted_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise ValueError(f"Invalid hand-eye config {path}.resolved_config: {exc}") from exc


def _load_inputs(
    input_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    from hardware_test.franka.handeye.handeye_utils import assess_pose_diversity, load_samples

    root = input_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Hand-eye input directory does not exist: {root}")

    config_path = root / "config_used.yaml"
    intrinsics_path = root / "camera_intrinsics.json"
    samples_path = root / "samples.jsonl"
    config = _load_resolved_config(config_path)
    validation_config = config.get("validation")
    if not isinstance(validation_config, Mapping):
        raise ValueError(f"{config_path} must contain a top-level validation config")
    intrinsics = _normalized_intrinsics(intrinsics_path)
    if not samples_path.is_file():
        raise FileNotFoundError(f"Required sample manifest does not exist: {samples_path}")
    try:
        samples = load_samples(root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Invalid sample store {samples_path}: {exc}") from exc
    if len(samples) < 4:
        raise ValueError(
            f"{samples_path}: at least 4 valid samples are required for leave-one-out validation, "
            f"got {len(samples)}"
        )

    expected_width = int(intrinsics["width"])
    expected_height = int(intrinsics["height"])
    expected_serial = str(intrinsics["serial"])
    for sample in samples:
        sample_id = int(sample["sample_id"])
        sample_width = int(sample["image_width"])
        sample_height = int(sample["image_height"])
        if (sample_width, sample_height) != (expected_width, expected_height):
            raise ValueError(
                f"{samples_path}: sample_id {sample_id} image dimensions "
                f"{sample_width}x{sample_height} do not match actual intrinsics "
                f"{expected_width}x{expected_height} from {intrinsics_path}"
            )
        sample_serial = sample.get("realsense_serial")
        if sample_serial is not None and sample_serial != expected_serial:
            raise ValueError(
                f"{samples_path}: sample_id {sample_id} recorded RealSense serial "
                f"{sample_serial!r}, which does not match {intrinsics_path} serial {expected_serial!r}"
            )

    diversity = assess_pose_diversity(samples)
    if not diversity["is_diverse"]:
        reasons = ", ".join(diversity["reasons"]) or "unknown diversity failure"
        raise ValueError(f"{samples_path}: robot poses are not diverse enough for calibration: {reasons}")
    return config, intrinsics, samples, diversity


def _resolve_cv2(cv2_module: Any | None) -> Any:
    if cv2_module is not None:
        return cv2_module
    try:
        return importlib.import_module("cv2")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is required for offline hand-eye calibration, but cv2 could not be imported. "
            "Install one compatible OpenCV build manually; this CLI never installs or upgrades packages."
        ) from exc


def _require_hand_eye_capability(cv2_module: Any) -> None:
    if callable(getattr(cv2_module, "calibrateHandEye", None)):
        return
    version = getattr(cv2_module, "__version__", "unknown")
    raise RuntimeError(
        "cv2.calibrateHandEye is unavailable or not callable in OpenCV "
        f"{version}. This commonly indicates an OpenCV package conflict or a build without the "
        "hand-eye API. Select one compatible OpenCV build that provides calibrateHandEye; this "
        "CLI will never install or upgrade packages automatically."
    )


def _recommendation_terms(
    validation: Mapping[str, Any],
    leave_one_out: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, dict[str, float | str]]:
    raw_terms = {
        "validation_translation": (
            float(validation["translation_error_m"]["max"]),
            float(thresholds["target_scatter_translation_m"]),
            "meter",
        ),
        "validation_rotation": (
            float(validation["rotation_geodesic_error_deg"]["max"]),
            float(thresholds["target_scatter_rotation_deg"]),
            "degree",
        ),
        "leave_one_out_translation": (
            max(
                float(leave_one_out["calibration_stability_translation_m"]["max"]),
                float(leave_one_out["held_out_target_translation_m"]["max"]),
            ),
            float(thresholds["leave_one_out_translation_m"]),
            "meter",
        ),
        "leave_one_out_rotation": (
            max(
                float(leave_one_out["calibration_stability_rotation_deg"]["max"]),
                float(leave_one_out["held_out_target_rotation_deg"]["max"]),
            ),
            float(thresholds["leave_one_out_rotation_deg"]),
            "degree",
        ),
    }
    return {
        name: {
            "value": value,
            "threshold": threshold,
            "normalized": value / threshold,
            "unit": unit,
        }
        for name, (value, threshold, unit) in raw_terms.items()
    }


def _recommendation_score(terms: Mapping[str, Mapping[str, Any]]) -> float:
    return sum(float(term["normalized"]) for term in terms.values()) / len(terms)


def _successful_method_names(method_reports: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [name for name, report in method_reports.items() if report.get("status") == "success"]


def _select_recommended_method(method_reports: Mapping[str, Mapping[str, Any]]) -> str:
    successful = _successful_method_names(method_reports)
    if not successful:
        errors = "; ".join(
            f"{name}: {report.get('error', 'unknown failure')}" for name, report in method_reports.items()
        )
        raise RuntimeError(f"No hand-eye method validated successfully: {errors}")
    return min(successful, key=lambda name: float(method_reports[name]["recommendation_score"]))


def _to_builtin_finite(value: Any, *, context: str) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{context} contains a non-finite number")
        return numeric
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context} contains a non-string mapping key")
            converted[key] = _to_builtin_finite(nested, context=f"{context}.{key}")
        return converted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _to_builtin_finite(nested, context=f"{context}[{index}]") for index, nested in enumerate(value)
        ]
    if hasattr(value, "tolist"):
        return _to_builtin_finite(value.tolist(), context=context)
    if hasattr(value, "item"):
        return _to_builtin_finite(value.item(), context=context)
    raise ValueError(f"{context} contains unsupported value type {type(value).__name__}")


def _calibration_input_provenance(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from hardware_test.franka.handeye.handeye_utils import validate_homogeneous_transform

    canonical_inputs: list[dict[str, Any]] = []
    sample_ids: list[int] = []
    for index, sample in enumerate(samples):
        sample_id = int(sample["sample_id"])
        sample_ids.append(sample_id)
        canonical_inputs.append(
            {
                "sample_id": sample_id,
                "T_base_ee": validate_homogeneous_transform(
                    sample["T_base_ee"],
                    name=f"samples[{index}] sample_id {sample_id} T_base_ee",
                ).tolist(),
                "T_camera_board": validate_homogeneous_transform(
                    sample["T_camera_board"],
                    name=f"samples[{index}] sample_id {sample_id} T_camera_board",
                ).tolist(),
            }
        )
    canonical_json = json.dumps(
        canonical_inputs,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema": PROVENANCE_SCHEMA,
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "digest_algorithm": PROVENANCE_DIGEST_ALGORITHM,
        "canonicalization": PROVENANCE_CANONICALIZATION,
        "digest": hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
        "sample_ids": sample_ids,
    }


def _calibration_document(
    method_reports: Mapping[str, Mapping[str, Any]],
    recommended_method: str,
    *,
    sample_count: int,
    provenance: Mapping[str, Any],
    opencv_version: str,
) -> dict[str, Any]:
    methods: dict[str, dict[str, Any]] = {}
    for method_name, report in method_reports.items():
        if report["status"] == "success":
            methods[method_name] = {
                "status": "success",
                "matrix": report["T_base_camera"],
                "score": report["recommendation_score"],
            }
        else:
            methods[method_name] = {
                "status": "error",
                "error": report["error"],
            }
    return _to_builtin_finite(
        {
            "parent_frame": PARENT_FRAME,
            "child_frame": CHILD_FRAME,
            "transform_name": TRANSFORM_NAME,
            "translation_unit": TRANSLATION_UNIT,
            "coordinate_direction": COORDINATE_DIRECTION,
            "meaning": TRANSFORM_MEANING,
            "method": recommended_method,
            "matrix": method_reports[recommended_method]["T_base_camera"],
            "sample_count": sample_count,
            "sample_ids": provenance["sample_ids"],
            "calibration_input_provenance": provenance,
            "opencv_version": opencv_version,
            "methods": methods,
        },
        context="T_base_camera.yaml",
    )


def _validation_report(
    method_reports: Mapping[str, Mapping[str, Any]],
    recommended_method: str,
    *,
    samples: Sequence[Mapping[str, Any]],
    diversity: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    opencv_version: str,
) -> dict[str, Any]:
    validation_outlier_ids: set[int] = set()
    leave_one_out_influential_ids: set[int] = set()
    for report in method_reports.values():
        if report.get("status") != "success":
            continue
        validation_outlier_ids.update(int(value) for value in report["validation"]["outlier_ids"])
        leave_one_out_influential_ids.update(
            int(value) for value in report["leave_one_out"]["influential_ids"]
        )
    combined_outlier_ids = sorted(validation_outlier_ids | leave_one_out_influential_ids)
    recommended_report = method_reports[recommended_method]
    return _to_builtin_finite(
        {
            "parent_frame": PARENT_FRAME,
            "child_frame": CHILD_FRAME,
            "transform_name": TRANSFORM_NAME,
            "translation_unit": TRANSLATION_UNIT,
            "coordinate_direction": COORDINATE_DIRECTION,
            "meaning": TRANSFORM_MEANING,
            "sample_count": len(samples),
            "sample_ids": provenance["sample_ids"],
            "calibration_input_provenance": provenance,
            "opencv_version": opencv_version,
            "config_thresholds": validation_config,
            "pose_diversity": diversity,
            "methods": method_reports,
            "recommended_method": recommended_method,
            "recommendation_terms": recommended_report["recommendation_terms"],
            "recommendation_score": recommended_report["recommendation_score"],
            "validation_outlier_ids": sorted(validation_outlier_ids),
            "leave_one_out_influential_ids": sorted(leave_one_out_influential_ids),
            "outlier_ids": combined_outlier_ids,
        },
        context="validation_report.json",
    )


def _json_text(document: Mapping[str, Any]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _yaml_text(document: Mapping[str, Any]) -> str:
    return yaml.safe_dump(document, sort_keys=False, allow_unicode=True)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(text)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _print_summary(method_reports: Mapping[str, Mapping[str, Any]], recommended_method: str) -> None:
    for method_name, report in method_reports.items():
        if report["status"] == "success":
            print(f"{method_name}: success, normalized score={float(report['recommendation_score']):.6g}")
        else:
            print(f"{method_name}: error, {report['error']}")
    print(f"Recommended method: {recommended_method}")


def main(argv: Sequence[str] | None = None, *, cv2_module: Any | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config, _intrinsics, samples, diversity = _load_inputs(args.input_dir)
    resolved_cv2 = _resolve_cv2(cv2_module)
    _require_hand_eye_capability(resolved_cv2)

    from hardware_test.franka.handeye.handeye_utils import solve_all_methods

    validation_config = config["validation"]
    solve_report = solve_all_methods(samples, validation_config, cv2_module=resolved_cv2)
    method_reports = solve_report["methods"]
    recommended_method = str(solve_report["recommended_method"])
    opencv_version = str(getattr(resolved_cv2, "__version__", "unknown"))
    provenance = _calibration_input_provenance(samples)
    calibration = _calibration_document(
        method_reports,
        recommended_method,
        sample_count=len(samples),
        provenance=provenance,
        opencv_version=opencv_version,
    )
    report = _validation_report(
        method_reports,
        recommended_method,
        samples=samples,
        diversity=diversity,
        validation_config=validation_config,
        provenance=provenance,
        opencv_version=opencv_version,
    )

    result_dir = args.input_dir.expanduser().resolve() / "result"
    calibration_text = _yaml_text(calibration)
    report_text = _json_text(report)
    _atomic_write_text(result_dir / "T_base_camera.yaml", calibration_text)
    _atomic_write_text(result_dir / "validation_report.json", report_text)
    _print_summary(method_reports, recommended_method)
    return 0


def _run_cli() -> int:
    try:
        return main()
    except Exception as exc:
        print(f"Solver error: {' '.join(str(exc).split())}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_run_cli())
