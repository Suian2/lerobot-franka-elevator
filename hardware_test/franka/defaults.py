from __future__ import annotations

import os
from typing import Final

CONTROL_HOST_ENV_VAR: Final[str] = "FRANKA_CONTROL_HOST"
DEFAULT_CONTROL_HOST: Final[str] = "192.168.1.11"


def get_control_host() -> str:
    """Return the configured Franka control-service host."""
    return os.getenv(CONTROL_HOST_ENV_VAR, "").strip() or DEFAULT_CONTROL_HOST
