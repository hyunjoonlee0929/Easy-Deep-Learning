"""Data Summary Agent implementation."""

from __future__ import annotations

from typing import Any

from .base_agent import BaseAgent


class DataSummaryAgent(BaseAgent):
    """Generate concise dataset summary from validation metadata."""

    @property
    def name(self) -> str:
        return "data_summary_agent"

    def _mock(self, payload: dict[str, Any]) -> dict[str, Any]:
        missing_total = int(sum(payload.get("missing_by_column", {}).values()))
        return {
            "agent": self.name,
            "mode": "mock",
            "summary_text": (
                f"Dataset has {payload.get('n_rows', 0)} rows and {payload.get('n_columns', 0)} columns. "
                f"Detected {len(payload.get('numeric_columns', []))} numeric and "
                f"{len(payload.get('categorical_columns', []))} categorical features."
            ),
            "highlights": {
                "target_column": payload.get("target_column"),
                "missing_total": missing_total,
                "missing_by_column": payload.get("missing_by_column", {}),
            },
        }

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Produce structured summary JSON."""
        if not self.has_openai_key():
            return self._mock(payload)

        try:
            return self._call_openai_json(
                (
                    "You are a data summary agent. Return valid JSON only with keys: "
                    "agent, mode, summary_text, highlights."
                ),
                payload,
            )
        except Exception:
            return self._mock(payload)
