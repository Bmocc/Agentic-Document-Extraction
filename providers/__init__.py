from __future__ import annotations

import os
from typing import Optional

from .base import Provider


def get_provider(name: Optional[str] = None) -> Provider:
    provider_name = (name or os.getenv("PROVIDER") or "openai_agents").strip().lower()
    if provider_name in {"openai", "openai_agents", "openai-agents"}:
        from .openai_agents import OpenAIAgentsProvider  # lazy import
        return OpenAIAgentsProvider()

    raise ValueError(
        f"Unknown provider '{provider_name}'. Implement a Provider and register it in providers/__init__.py."
    )
