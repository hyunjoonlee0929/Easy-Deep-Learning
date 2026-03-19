"""Model registry layer for run tagging and promotion workflows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _score(task_type: str, metrics: dict[str, Any]) -> float:
    if task_type == "classification":
        if "f1_weighted" in metrics:
            return float(metrics["f1_weighted"])
        return float(metrics.get("accuracy", 0.0))
    if "r2" in metrics:
        return float(metrics["r2"])
    if "rmse" in metrics:
        return -float(metrics["rmse"])
    return 0.0


class ModelRegistry:
    """Simple JSON-backed model registry for run tags."""

    def __init__(self, runs_dir: Path | str = Path("runs")) -> None:
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.runs_dir / "model_registry.json"

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"models": [], "tags": {}, "updated_at": _utc_now()}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, payload: dict[str, Any]) -> None:
        payload["updated_at"] = _utc_now()
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def register_run(
        self,
        run_id: str,
        run_type: str,
        model_type: str,
        task_type: str,
        metrics: dict[str, Any],
        dataset_hash: str,
        model_signature: str,
    ) -> dict[str, str]:
        payload = self._load()
        models = payload["models"]
        tags = payload["tags"]

        models = [m for m in models if m.get("run_id") != run_id]
        new_entry = {
            "run_id": run_id,
            "run_type": run_type,
            "model_type": model_type,
            "task_type": task_type,
            "metrics": metrics,
            "dataset_hash": dataset_hash,
            "model_signature": model_signature,
            "score": _score(task_type, metrics),
            "created_at": _utc_now(),
        }
        models.append(new_entry)
        payload["models"] = models

        latest_tag = f"latest:{run_type}:{task_type}:{model_type}"
        tags[latest_tag] = run_id

        scope_tag = f"best:{run_type}:{task_type}:{model_type}"
        scope_entries = [m for m in models if m["run_type"] == run_type and m["task_type"] == task_type and m["model_type"] == model_type]
        if scope_entries:
            best_scope = sorted(scope_entries, key=lambda x: x.get("score", 0.0), reverse=True)[0]
            tags[scope_tag] = best_scope["run_id"]

        task_tag = f"best:{run_type}:{task_type}"
        task_entries = [m for m in models if m["run_type"] == run_type and m["task_type"] == task_type]
        if task_entries:
            best_task = sorted(task_entries, key=lambda x: x.get("score", 0.0), reverse=True)[0]
            tags[task_tag] = best_task["run_id"]

        payload["tags"] = tags
        self._save(payload)
        return {"latest_tag": latest_tag, "best_scope_tag": scope_tag, "best_task_tag": task_tag}

    def promote_to_production(self, run_id: str, run_type: str, task_type: str, model_type: str | None = None) -> str:
        payload = self._load()
        tags = payload["tags"]
        if model_type:
            tag = f"production:{run_type}:{task_type}:{model_type}"
        else:
            tag = f"production:{run_type}:{task_type}"
        tags[tag] = run_id
        payload["tags"] = tags
        self._save(payload)
        return tag

    def resolve_tag(self, tag: str) -> str | None:
        payload = self._load()
        return payload.get("tags", {}).get(tag)

    def list_registry(self) -> dict[str, Any]:
        return self._load()

