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
