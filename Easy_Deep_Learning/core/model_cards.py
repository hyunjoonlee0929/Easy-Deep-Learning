"""Automatic data/model card generation per run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def generate_run_card(run_path: Path) -> tuple[Path, Path]:
    """Generate run_card.json and run_card.md from run artifacts."""
    metrics = _read_json(run_path / "metrics.json") or {}
    model_info = _read_json(run_path / "model_info.json") or {}
    metadata = _read_json(run_path / "run_metadata.standard.json") or {}
    top_features = _read_json(run_path / "top_features.json") or []
    ai_report = _read_json(run_path / "ai_report.json") or {}

    card_json = {
        "run_id": run_path.name,
        "run_type": metadata.get("run_type"),
        "task_type": metadata.get("task_type") or model_info.get("task_type"),
        "model_type": metadata.get("model_type") or model_info.get("model_type"),
        "model_signature": metadata.get("model_signature"),
        "dataset_hash": metadata.get("dataset_hash"),
        "metrics": metrics,
        "top_features": top_features[:10],
        "summary": ai_report.get("summary", ""),
        "strengths": ai_report.get("strengths", []),
        "risks": ai_report.get("risks", []),
        "next_steps": ai_report.get("next_steps", []),
    }

    json_path = run_path / "run_card.json"
    json_path.write_text(json.dumps(card_json, indent=2), encoding="utf-8")

    md = [
        f"# Run Card: {run_path.name}",
        "",
        f"- Run Type: `{card_json.get('run_type')}`",
        f"- Task Type: `{card_json.get('task_type')}`",
        f"- Model Type: `{card_json.get('model_type')}`",
        f"- Model Signature: `{card_json.get('model_signature')}`",
        f"- Dataset Hash: `{card_json.get('dataset_hash')}`",
        "",
        "## Metrics",
        json.dumps(metrics, indent=2),
        "",
        "## Summary",
        str(card_json.get("summary", "")),
        "",
        "## Strengths",
    ]
    for item in card_json.get("strengths", []):
        md.append(f"- {item}")
    md.extend(["", "## Risks"])
    for item in card_json.get("risks", []):
        md.append(f"- {item}")
    md.extend(["", "## Next Steps"])
    for item in card_json.get("next_steps", []):
        md.append(f"- {item}")

    md_path = run_path / "run_card.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return json_path, md_path
