"""Recommendation engine for improving model performance."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fallback_recommendations(metrics: dict[str, Any], model_info: dict[str, Any]) -> dict[str, Any]:
    task = model_info.get("task_type", "classification")
    recs = []
    if task == "classification":
        acc = float(metrics.get("accuracy", 0.0))
        if acc < 0.75:
            recs.append("Try stronger models (xgboost/gbm) or tune max_depth/n_estimators.")
        recs.append("Check class imbalance and consider class weighting.")
        recs.append("Inspect top errors for mislabeled or outlier samples.")
    else:
        r2 = float(metrics.get("r2", -1.0))
        if r2 < 0.5:
            recs.append("Add non-linear models (gbm/xgboost) or feature transformations.")
        recs.append("Inspect residual plots for heteroscedasticity.")
        recs.append("Consider log/scale transforms for skewed targets.")

    return {
        "summary": "Rule-based recommendations generated (no API key).",
        "recommendations": recs,
        "priority": recs[:3],
    }


def generate_model_recommendations(run_path: Path) -> Path:
    metrics = _load_json(run_path / "metrics.json") or {}
    model_info = _load_json(run_path / "model_info.json") or {}
    error_analysis = _load_json(run_path / "error_analysis.json") or {}
    explainability = _load_json(run_path / "explainability.json") or {}

    api_key = os.getenv("OPENAI_API_KEY")
    report = None
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI()
            prompt = (
                "Return STRICT JSON with keys: summary, recommendations (array), priority (array). "
                "Provide concise, actionable improvements."
            )
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {"role": "system", "content": "You are a ML improvement advisor."},
                    {"role": "user", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "metrics": metrics,
                                "model_info": model_info,
                                "error_analysis": error_analysis,
                                "explainability": explainability,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            )
            text = ""
            for item in response.output:
                if item.type == "output_text":
                    text += item.text
            report = json.loads(text) if text else None
        except Exception:
            report = None

    if report is None:
        report = _fallback_recommendations(metrics, model_info)

    out_path = run_path / "recommendations.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
