from __future__ import annotations

from typing import Any, Dict, List, Type
from pydantic import BaseModel

from .base import Provider


class OpenAIAgentsProvider(Provider):
    def __init__(self) -> None:
        # Lazy import so that other providers can be used without this dep present
        try:
            from agents import Agent, Runner  # type: ignore
            self._Agent = Agent
            self._Runner = Runner
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "OpenAI Agents SDK ('agents') not installed. See README for installation or switch PROVIDER."
            ) from e

        try:
            from openai.types.responses import ResponseInputImageParam, ResponseInputTextParam  # type: ignore
            from openai.types.responses.response_input_item_param import Message  # type: ignore
            self._ResponseInputImageParam = ResponseInputImageParam
            self._ResponseInputTextParam = ResponseInputTextParam
            self._Message = Message
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "openai>=1.52.0 required for message payload types used by Agents SDK."
            ) from e

    async def _run(self, agent, input):
        return await self._Runner.run(agent, input=input)

    async def run_structured_text(
        self,
        *,
        name: str,
        instructions: str,
        model: str,
        output_type: Type[BaseModel],
        input_text: str,
    ) -> BaseModel:
        Agent = self._Agent
        agent = Agent(name=name, instructions=instructions, model=model, output_type=output_type)
        res = await self._run(agent, input=input_text)
        return res.final_output

    async def run_structured_messages(
        self,
        *,
        name: str,
        instructions: str,
        model: str,
        output_type: Type[BaseModel],
        messages: List[Dict[str, Any]],
    ) -> BaseModel:
        Agent = self._Agent
        Message = self._Message
        Text = self._ResponseInputTextParam
        Image = self._ResponseInputImageParam

        # Convert generic messages -> OpenAI message objects
        oa_msgs = []
        for m in messages:
            role = m.get("role", "user")
            parts = []
            for p in m.get("content", []):
                ptype = p.get("type")
                if ptype == "text":
                    parts.append(Text(type="input_text", text=p.get("text", "")))
                elif ptype == "image":
                    parts.append(
                        Image(type="input_image", image_url=p.get("image_url"), detail=p.get("detail", "low"))
                    )
            oa_msgs.append(Message(role=role, content=parts))

        agent = Agent(name=name, instructions=instructions, model=model, output_type=output_type)
        res = await self._run(agent, input=oa_msgs)
        return res.final_output
