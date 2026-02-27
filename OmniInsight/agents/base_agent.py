"""Base class for OmniInsight LLM agents."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Abstract JSON-in, JSON-out contract for all agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name used for tracing and metadata."""

    @abstractmethod
    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Run agent logic with structured JSON payload."""

    def has_openai_key(self) -> bool:
        """Return True when OPENAI_API_KEY is configured."""
        return bool(os.getenv("OPENAI_API_KEY"))

    def _call_openai_json(self, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Call OpenAI Responses API and parse JSON output."""
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"{system_prompt}\nInput JSON: {json.dumps(payload)}",
            temperature=0,
        )
        text = response.output_text.strip()
        return json.loads(text)

    def _mock_base(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Shared deterministic metadata for local mock outputs."""
        return {
            "agent": self.name,
            "mode": "mock",
            "source": "local",
            "input": payload,
        }
