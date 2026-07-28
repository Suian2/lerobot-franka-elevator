#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware_test.franka.handeye import solve_eye_to_hand as shared  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute validation for a saved L515-to-Franka eye-to-hand calibration."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Capture directory containing the sample store and result/T_base_camera.yaml.",
    )
    return parser


def _load_calibration_document(path: Path) -> dict[str, Any]:
    from hardware_test.franka.handeye.handeye_utils import HAND_EYE_METHODS

    if not path.is_file():
        raise FileNotFoundError(f"Saved calibration does not exist: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise OSError(f"Failed to read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a top-level mapping")

    expected_metadata = {
        "parent_frame": shared.PARENT_FRAME,
        "child_frame": shared.CHILD_FRAME,
        "transform_name": shared.TRANSFORM_NAME,
        "translation_unit": shared.TRANSLATION_UNIT,
        "coordinate_direction": shared.COORDINATE_DIRECTION,
    }
    for key, expected in expected_metadata.items():
        if document.get(key) != expected:
            raise ValueError(f"{path}.{key} must be exactly {expected!r}")
    methods = document.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError(f"{path}.methods must be a mapping")
    recommended_method = document.get("method")
    if recommended_method not in HAND_EYE_METHODS:
        raise ValueError(f"{path}.method must name one of {', '.join(HAND_EYE_METHODS)}")
    recommended_entry = methods.get(recommended_method)
    if not isinstance(recommended_entry, Mapping) or recommended_entry.get("status") != "success":
        raise ValueError(f"{path}.method must reference a successful methods entry")
    return document


def _validate_recommended_matrix(path: Path, document: Mapping[str, Any]) -> None:
    import numpy as np

    from hardware_test.franka.handeye.handeye_utils import validate_homogeneous_transform

    methods = document["methods"]
    recommended_method = document["method"]
    assert isinstance(methods, Mapping)
    recommended_entry = methods[recommended_method]
    assert isinstance(recommended_entry, Mapping)
    recommended_matrix = validate_homogeneous_transform(
        document.get("matrix"),
        name=f"{path}.matrix",
    )
    try:
        entry_matrix = validate_homogeneous_transform(
            recommended_entry.get("matrix"),
            name=f"{path}.methods.{recommended_method}.matrix",
        )
    except ValueError:
        # The per-method loader aggregates invalid matrices so operators see every
        # affected OpenCV method in one error instead of fixing them one at a time.
        entry_matrix = None
    if entry_matrix is not None and not np.allclose(
        recommended_matrix,
        entry_matrix,
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError(f"{path}.matrix must match methods.{recommended_method}.matrix")


def _require_matching_calibration_provenance(
    calibration_path: Path,
    calibration: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if "calibration_input_provenance" not in calibration:
        raise ValueError(
            f"{calibration_path}: missing calibration input provenance; samples may have changed. "
            "rerun solve_eye_to_hand.py before validation."
        )
    stored = calibration["calibration_input_provenance"]
    if not isinstance(stored, Mapping):
        raise ValueError(
            f"{calibration_path}: malformed calibration input provenance; expected a mapping. "
            "rerun solve_eye_to_hand.py before validation."
        )
    expected_metadata = {
        "schema": shared.PROVENANCE_SCHEMA,
        "schema_version": shared.PROVENANCE_SCHEMA_VERSION,
        "digest_algorithm": shared.PROVENANCE_DIGEST_ALGORITHM,
        "canonicalization": shared.PROVENANCE_CANONICALIZATION,
    }
    for key, expected in expected_metadata.items():
        if stored.get(key) != expected:
            raise ValueError(
                f"{calibration_path}: malformed calibration input provenance {key}; "
                f"expected {expected!r}. rerun solve_eye_to_hand.py before validation."
            )
    digest = stored.get("digest")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(
            f"{calibration_path}: malformed calibration input provenance digest; expected "
            "64 lowercase hexadecimal characters. rerun solve_eye_to_hand.py before validation."
        )
    stored_sample_ids = stored.get("sample_ids")
    if (
        not isinstance(stored_sample_ids, list)
        or any(
            isinstance(sample_id, bool) or not isinstance(sample_id, int) or sample_id < 0
            for sample_id in stored_sample_ids
        )
        or len(set(stored_sample_ids)) != len(stored_sample_ids)
    ):
        raise ValueError(
            f"{calibration_path}: malformed calibration input provenance sample_ids; expected "
            "an ordered list of unique nonnegative integers. rerun solve_eye_to_hand.py before validation."
        )
    current = shared._calibration_input_provenance(samples)
    if calibration.get("sample_ids") != current["sample_ids"] or stored_sample_ids != current["sample_ids"]:
        raise ValueError(
            f"{calibration_path}: calibration input provenance sample IDs do not match current "
            "samples; samples changed after solve. rerun solve_eye_to_hand.py before validation."
        )
    if digest != current["digest"]:
        raise ValueError(
            f"{calibration_path}: calibration input provenance does not match current samples; "
            "samples changed after solve. rerun solve_eye_to_hand.py before validation."
        )
    return current


def _reload_method_reports(
    calibration_path: Path,
    calibration: Mapping[str, Any],
    *,
    samples: Sequence[Mapping[str, Any]],
    validation_config: Mapping[str, Any],
    cv2_module: Any,
) -> dict[str, dict[str, Any]]:
    from hardware_test.franka.handeye.handeye_utils import (
        HAND_EYE_METHODS,
        leave_one_out_validation,
        validate_eye_to_hand_result,
        validate_homogeneous_transform,
    )

    raw_methods = calibration["methods"]
    assert isinstance(raw_methods, Mapping)
    expected_names = tuple(HAND_EYE_METHODS)
    missing_names = [name for name in expected_names if name not in raw_methods]
    unknown_names = [str(name) for name in raw_methods if name not in HAND_EYE_METHODS]
    if missing_names or unknown_names:
        details: list[str] = []
        if missing_names:
            details.append(f"missing {', '.join(missing_names)}")
        if unknown_names:
            details.append(f"unknown {', '.join(unknown_names)}")
        raise ValueError(
            f"{calibration_path}.methods must contain exactly four methods: {'; '.join(details)}"
        )

    reports: dict[str, dict[str, Any]] = {}
    valid_matrices: dict[str, Any] = {}
    for method_name, method_constant in HAND_EYE_METHODS.items():
        entry = raw_methods[method_name]
        context = f"{calibration_path}.methods.{method_name}"
        if not isinstance(entry, Mapping):
            reports[method_name] = {
                "method": method_constant,
                "status": "error",
                "error": f"ValueError: {context} must be a mapping",
            }
            continue
        status = entry.get("status")
        if status == "error":
            error = entry.get("error")
            if not isinstance(error, str) or not error:
                error = f"ValueError: {context}.error must be a nonempty string"
            reports[method_name] = {
                "method": method_constant,
                "status": "error",
                "error": error,
            }
            continue
        if status != "success":
            reports[method_name] = {
                "method": method_constant,
                "status": "error",
                "error": f"ValueError: {context}.status must be 'success' or 'error'",
            }
            continue
        try:
            valid_matrices[method_name] = validate_homogeneous_transform(
                entry.get("matrix"),
                name=f"{context}.matrix",
            )
        except ValueError as exc:
            reports[method_name] = {
                "method": method_constant,
                "status": "error",
                "error": f"ValueError: {exc}",
            }

    if not valid_matrices:
        failures = "; ".join(
            f"{method_name}: {reports[method_name]['error']}" for method_name in expected_names
        )
        raise RuntimeError(f"{calibration_path}: no valid method matrices are available: {failures}")

    shared._require_hand_eye_capability(cv2_module)
    for method_name, matrix in valid_matrices.items():
        method_constant = HAND_EYE_METHODS[method_name]
        try:
            validation = validate_eye_to_hand_result(samples, matrix, validation_config)
            leave_one_out = leave_one_out_validation(
                samples,
                matrix,
                method_constant,
                validation_config,
                cv2_module=cv2_module,
            )
            recommendation_terms = shared._recommendation_terms(
                validation,
                leave_one_out,
                validation_config,
            )
            reports[method_name] = {
                "method": method_constant,
                "status": "success",
                "T_base_camera": matrix.tolist(),
                "validation": validation,
                "leave_one_out": leave_one_out,
                "recommendation_terms": recommendation_terms,
                "recommendation_score": shared._recommendation_score(recommendation_terms),
            }
        except Exception as exc:
            reports[method_name] = {
                "method": method_constant,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    if not shared._successful_method_names(reports):
        failures = "; ".join(
            f"{method_name}: {reports[method_name]['error']}" for method_name in expected_names
        )
        raise RuntimeError(f"{calibration_path}: all saved method matrices failed validation: {failures}")
    return {method_name: reports[method_name] for method_name in expected_names}


def main(argv: Sequence[str] | None = None, *, cv2_module: Any | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    input_dir = args.input_dir.expanduser().resolve()
    config, _intrinsics, samples, diversity = shared._load_inputs(input_dir)
    calibration_path = input_dir / "result" / "T_base_camera.yaml"
    calibration = _load_calibration_document(calibration_path)
    provenance = _require_matching_calibration_provenance(calibration_path, calibration, samples)
    _validate_recommended_matrix(calibration_path, calibration)
    resolved_cv2 = shared._resolve_cv2(cv2_module)
    method_reports = _reload_method_reports(
        calibration_path,
        calibration,
        samples=samples,
        validation_config=config["validation"],
        cv2_module=resolved_cv2,
    )
    recommended_method = shared._select_recommended_method(method_reports)
    report = shared._validation_report(
        method_reports,
        recommended_method,
        samples=samples,
        diversity=diversity,
        validation_config=config["validation"],
        provenance=provenance,
        opencv_version=str(getattr(resolved_cv2, "__version__", "unknown")),
    )
    report_path = input_dir / "result" / "validation_report.json"
    report_text = shared._json_text(report)
    shared._atomic_write_text(report_path, report_text)
    shared._print_summary(method_reports, recommended_method)
    return 0


def _run_cli() -> int:
    try:
        return main()
    except Exception as exc:
        print(f"Validation error: {' '.join(str(exc).split())}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_run_cli())
