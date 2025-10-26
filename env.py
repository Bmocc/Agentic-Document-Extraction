from __future__ import annotations

import os
from typing import Optional

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # optional dependency; README covers usage


def load_env(override: bool = False) -> None:
    """Load environment variables from a local .env file if present.

    Uses python-dotenv when available; otherwise this is a no-op.
    """
    if load_dotenv is None:
        return
    # Only load if a .env exists nearby
    # Respect existing env unless override=True
    load_dotenv(override=override)


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

