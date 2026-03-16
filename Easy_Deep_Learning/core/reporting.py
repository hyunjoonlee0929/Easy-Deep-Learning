"""HTML reporting utilities for Easy Deep Learning runs."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _table_rows(payload: dict[str, Any]) -> str:
    rows = []
    for key, value in payload.items():
        rows.append(f"<tr><th>{key}</th><td>{value}</td></tr>")
    return "\n".join(rows)


def _fallback_ai_report(metrics: dict[str, Any], model_info: dict[str, Any], top_features: list[str]) -> dict[str, Any]:
    task = model_info.get("task_type", "classification")
    summary = f"{model_info.get('model_type', 'model')} model trained for {task}."
    if metrics:
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        summary += f" Metrics: {metrics_str}."
    strengths = [
        "Auto preprocessing handled missing values and encoding.",
        "Run artifacts saved for reproducibility.",
    ]
    if top_features:
        strengths.append(f"Top features: {', '.join(top_features[:5])}.")
    risks = [
        "Validate performance on out-of-sample data.",
        "Check for class imbalance if accuracy looks misleading.",
    ]
    next_steps = [
        "Try alternative models or hyperparameters.",
        "Inspect PDP/ICE plots for key features.",
    ]
    return {
        "summary": summary,
        "strengths": strengths,
        "risks": risks,
        "next_steps": next_steps,
    }


def generate_ai_report(run_path: Path) -> Path:
    """Generate AI-assisted narrative report (OpenAI optional)."""
    metrics = _read_json(run_path / "metrics.json") or {}
    model_info = _read_json(run_path / "model_info.json") or {}
    top_features_payload = _read_json(run_path / "top_features.json") or []
    top_features = [f[0] for f in top_features_payload] if top_features_payload else []

    api_key = os.getenv("OPENAI_API_KEY")
    report = None
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI()
            prompt = (
                "Return STRICT JSON with keys: summary, strengths, risks, next_steps. "
                "Summarize the model run. Keep it concise."
            )
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {"role": "system", "content": "You generate concise ML reports."},
                    {"role": "user", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "metrics": metrics,
                                "model_info": model_info,
                                "top_features": top_features,
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
        report = _fallback_ai_report(metrics, model_info, top_features)

    report_path = run_path / "ai_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    text_path = run_path / "ai_report.txt"
    text_path.write_text(
        f"Summary: {report.get('summary')}\n"
        f"Strengths: {', '.join(report.get('strengths', []))}\n"
        f"Risks: {', '.join(report.get('risks', []))}\n"
        f"Next steps: {', '.join(report.get('next_steps', []))}\n",
        encoding="utf-8",
    )
    return report_path


def generate_html_report(run_path: Path) -> Path:
    """Generate a lightweight HTML report for a run."""
    metrics = _read_json(run_path / "metrics.json") or {}
    model_info = _read_json(run_path / "model_info.json") or {}
    model_params = _read_json(run_path / "model_params.json") or {}
    validation = _read_json(run_path / "validation_report.json") or {}
    predictions = _read_json(run_path / "predictions_preview.json") or {}
    ai_report = _read_json(run_path / "ai_report.json") or {}
    recommendations = _read_json(run_path / "recommendations.json") or {}
    error_analysis = _read_json(run_path / "error_analysis.json") or {}
    data_quality = _read_json(run_path / "data_quality.json") or {}
    drift_report = _read_json(run_path / "drift_report.json") or {}
    uncertainty = _read_json(run_path / "uncertainty.json") or {}
    has_cm = (run_path / "confusion_matrix.png").exists()
    has_roc = (run_path / "roc_curve.png").exists()
    has_scatter = (run_path / "prediction_scatter.png").exists()
    has_shap = (run_path / "shap_summary.png").exists()
    has_shap_inter = (run_path / "shap_interaction.png").exists()
    has_residuals = (run_path / "residuals.png").exists()
    interaction_pdp = list(run_path.glob("pdp_interaction_*.png"))

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Easy Deep Learning Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
    th {{ background: #f9fafb; width: 240px; }}
    .section {{ margin-top: 24px; }}
  </style>
</head>
<body>
  <h1>Easy Deep Learning Report</h1>
  <div>Run path: {run_path.resolve()}</div>
  <div>Generated: {datetime.now().isoformat(timespec="seconds")}</div>

  <div class="section">
    <h2>Model Info</h2>
    <table>{_table_rows(model_info)}</table>
  </div>

  <div class="section">
    <h2>Metrics</h2>
    <table>{_table_rows(metrics)}</table>
  </div>

  <div class="section">
    <h2>Model Params</h2>
    <table>{_table_rows(model_params)}</table>
  </div>

  <div class="section">
    <h2>Validation Summary</h2>
    <pre>{json.dumps(validation, indent=2)}</pre>
  </div>

  <div class="section">
    <h2>Prediction Preview</h2>
    <pre>{json.dumps(predictions, indent=2)}</pre>
  </div>

  <div class="section">
    <h2>AI Report</h2>
    <pre>{json.dumps(ai_report, indent=2, ensure_ascii=False)}</pre>
  </div>

  <div class="section">
    <h2>Data Quality</h2>
    <pre>{json.dumps(data_quality, indent=2, ensure_ascii=False)}</pre>
  </div>

  <div class="section">
    <h2>Drift Report</h2>
    <pre>{json.dumps(drift_report, indent=2, ensure_ascii=False)}</pre>
  </div>

  <div class="section">
    <h2>Uncertainty</h2>
    <pre>{json.dumps(uncertainty, indent=2, ensure_ascii=False)}</pre>
  </div>

  <div class="section">
    <h2>Error Analysis</h2>
    <pre>{json.dumps(error_analysis, indent=2, ensure_ascii=False)}</pre>
  </div>

  <div class="section">
    <h2>Recommendations</h2>
    <pre>{json.dumps(recommendations, indent=2, ensure_ascii=False)}</pre>
  </div>

  {"<div class='section'><h2>Confusion Matrix</h2><img src='confusion_matrix.png' style='max-width: 640px; width: 100%;'/></div>" if has_cm else ""}
  {"<div class='section'><h2>ROC Curve</h2><img src='roc_curve.png' style='max-width: 640px; width: 100%;'/></div>" if has_roc else ""}
  {"<div class='section'><h2>Prediction Scatter</h2><img src='prediction_scatter.png' style='max-width: 640px; width: 100%;'/></div>" if has_scatter else ""}
  {"<div class='section'><h2>SHAP Summary</h2><img src='shap_summary.png' style='max-width: 640px; width: 100%;'/></div>" if has_shap else ""}
  {"<div class='section'><h2>SHAP Interaction</h2><img src='shap_interaction.png' style='max-width: 640px; width: 100%;'/></div>" if has_shap_inter else ""}
  {"<div class='section'><h2>Residual Plot</h2><img src='residuals.png' style='max-width: 640px; width: 100%;'/></div>" if has_residuals else ""}
  {(
        "<div class='section'><h2>Interaction PDP</h2>"
        + "".join([f"<img src='{p.name}' style='max-width: 640px; width: 100%;'/>" for p in interaction_pdp])
        + "</div>"
    ) if interaction_pdp else ""}
</body>
</html>
"""
    report_path = run_path / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
