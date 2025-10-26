from __future__ import annotations

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel


ContentPart = Dict[str, Any]


class Provider:
    """Abstract interface for model providers.

    A provider must support structured output parsed by Pydantic models and
    optionally mixed text+image inputs.
    """

    async def run_structured_text(
        self,
        *,
        name: str,
        instructions: str,
        model: str,
        output_type: Type[BaseModel],
        input_text: str,
    ) -> BaseModel:
        raise NotImplementedError

    async def run_structured_messages(
        self,
        *,
        name: str,
        instructions: str,
        model: str,
        output_type: Type[BaseModel],
        messages: List[Dict[str, Any]],  # [{ role: 'user'|'system'|'assistant', content: [parts...] }]
    ) -> BaseModel:
        raise NotImplementedError
