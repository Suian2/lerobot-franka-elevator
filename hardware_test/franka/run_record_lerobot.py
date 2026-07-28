from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hardware_test.franka.run_record as run_record  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return run_record.main(argv, require_target_floor=True)


if __name__ == "__main__":
    raise SystemExit(main())
