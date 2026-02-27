"""Domain Mapping Agent implementation."""

from __future__ import annotations

from typing import Any

from .base_agent import BaseAgent


class DomainMappingAgent(BaseAgent):
    """Map model insights into domain-specific meaning."""

    @property
    def name(self) -> str:
        return "domain_mapping_agent"

    def _mock(self, payload: dict[str, Any]) -> dict[str, Any]:
        top_features = payload.get("interpretation", {}).get("top_features", [])
        mapped = [
            {
                "feature": f,
                "domain_concept": f"Potential domain driver: {f}",
                "confidence": 0.5,
            }
            for f in top_features[:5]
        ]
        return {
            "agent": self.name,
            "mode": "mock",
            "mapped_concepts": mapped,
            "caveats": [
                "Domain mappings are heuristic because no external ontology is configured.",
            ],
        }

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Produce domain mapping JSON."""
        if not self.has_openai_key():
            return self._mock(payload)

        try:
            return self._call_openai_json(
                (
                    "You are a domain mapping agent. Return valid JSON only with keys: "
                    "agent, mode, mapped_concepts, caveats."
                ),
                payload,
            )
        except Exception:
            return self._mock(payload)
