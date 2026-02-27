"""Feature Interpretation Agent implementation."""

from __future__ import annotations

from typing import Any

from .base_agent import BaseAgent


class FeatureInterpretationAgent(BaseAgent):
    """Translate raw importance scores into structured interpretation."""

    @property
    def name(self) -> str:
        return "feature_interpretation_agent"

    def _mock(self, payload: dict[str, Any]) -> dict[str, Any]:
        top_features = payload.get("top_features", [])
        importance = payload.get("feature_importance", {})
        narratives = [f"{feat} has influence score {importance.get(feat, 0.0):.4f}" for feat in top_features[:5]]
        return {
            "agent": self.name,
            "mode": "mock",
            "top_features": top_features,
            "interpretation": {
                "shap_status": payload.get("status", "unknown"),
                "explainer": payload.get("explainer", "unknown"),
                "feature_importance": importance,
                "narrative": narratives,
            },
        }

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Produce interpretation JSON from SHAP payload."""
        if not self.has_openai_key():
            return self._mock(payload)

        try:
            return self._call_openai_json(
                (
                    "You are a feature interpretation agent. Return valid JSON only with keys: "
                    "agent, mode, top_features, interpretation."
                ),
                payload,
            )
        except Exception:
            return self._mock(payload)
