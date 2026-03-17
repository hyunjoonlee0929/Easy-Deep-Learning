"""Experiment tracking and artifact persistence for Easy Deep Learning."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Persist run metadata, metrics, model artifacts, and reports."""

    def __init__(self, base_dir: Path | str = Path("runs")) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_run(self, model_type: str) -> tuple[str, Path]:
        """Create unique run directory and return (run_id, run_path)."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_id = f"{ts}_{model_type}"
        run_path = self.base_dir / run_id
        run_path.mkdir(parents=True, exist_ok=False)
        return run_id, run_path

    def config_hash(self, config_path: Path) -> str:
        """Compute deterministic SHA256 hash from config file bytes."""
        raw = config_path.read_bytes()
        return hashlib.sha256(raw).hexdigest()

    def file_hash(self, path: Path) -> str:
        """Compute SHA256 hash for a file."""
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def find_matching_run(self, model_type: str, metadata: dict[str, Any]) -> str | None:
        """Find a previous run that matches the provided metadata."""
        if not self.base_dir.exists():
            return None
        runs = sorted([p for p in self.base_dir.iterdir() if p.is_dir()], reverse=True)
        for run_path in runs:
            meta_path = run_path / "run_metadata.json"
            if not meta_path.exists():
                continue
            try:
                saved = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if saved.get("model_type") != model_type:
                continue
            if any(saved.get(k) != v for k, v in metadata.items()):
                continue
            model_info = run_path / "model_info.json"
            if model_info.exists():
                return run_path.name
        return None

    def save_yaml(self, path: Path, payload: dict[str, Any]) -> None:
        """Save dictionary as YAML."""
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)

    def save_json(self, path: Path, payload: dict[str, Any] | list[Any]) -> None:
        """Save JSON payload."""
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save_text(self, path: Path, text: str) -> None:
        """Save plain text."""
        path.write_text(text, encoding="utf-8")

    def save_model_artifact(self, model: Any, model_type: str, run_path: Path) -> Path:
        """Persist trained model according to model family conventions."""
        if model_type == "xgboost":
            if hasattr(model, "save_model"):
                model_path = run_path / "model.json"
                model.save_model(str(model_path))
                return model_path

            import joblib

            model_path = run_path / "model.model"
            joblib.dump(model, model_path)
            return model_path

        if model_type == "dnn":
            import joblib

            model_path = run_path / "model.pt"
            joblib.dump(model, model_path)
            return model_path

        import joblib

        model_path = run_path / "model.bin"
        joblib.dump(model, model_path)
        return model_path

    def load_run_snapshot(self, run_id: str) -> dict[str, Any]:
        """Load saved run snapshot YAML."""
        snap_path = self.base_dir / run_id / "config_snapshot.yaml"
        if not snap_path.exists():
            raise FileNotFoundError(f"Run snapshot not found: {snap_path}")
        with snap_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
