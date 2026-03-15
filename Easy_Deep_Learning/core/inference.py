"""Inference utilities for saved tabular runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def _load_model(run_path: Path):
    if (run_path / "model.json").exists():
        model_path = run_path / "model.json"
        try:
            from xgboost import XGBClassifier

            model = XGBClassifier()
            model.load_model(str(model_path))
            return model
        except Exception:
            try:
                from xgboost import XGBRegressor

                model = XGBRegressor()
                model.load_model(str(model_path))
                return model
            except Exception:
                pass

    for path in [run_path / "model.model", run_path / "model.pt", run_path / "model.bin"]:
        if path.exists():
            return joblib.load(path)

    raise FileNotFoundError("No model artifact found in run directory.")


def load_run_bundle(run_id: str) -> dict[str, Any]:
    run_path = Path("runs") / run_id
    if not run_path.exists():
        raise FileNotFoundError(f"Run '{run_id}' does not exist.")

    model_info = json.loads((run_path / "model_info.json").read_text(encoding="utf-8"))
    preprocessor = joblib.load(run_path / "preprocessor.joblib")
    label_encoder = None
    if (run_path / "label_encoder.joblib").exists():
        label_encoder = joblib.load(run_path / "label_encoder.joblib")
    model = _load_model(run_path)

    return {
        "run_path": run_path,
        "model_info": model_info,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "model": model,
    }


def predict_from_dataframe(run_id: str, df: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    bundle = load_run_bundle(run_id)
    model = bundle["model"]
    model_info = bundle["model_info"]
    preprocessor = bundle["preprocessor"]
    label_encoder = bundle["label_encoder"]

    target = target_column or model_info.get("target_column")
    if target and target in df.columns:
        X_df = df.drop(columns=[target])
    else:
        X_df = df

    X = preprocessor.transform(X_df)
    X_infer = X.toarray() if hasattr(X, "toarray") else X
    y_pred = model.predict(X_infer)

    result: dict[str, Any] = {
        "run_id": run_id,
        "model_type": model_info.get("model_type"),
        "task_type": model_info.get("task_type"),
        "predictions": [],
    }

    if label_encoder is not None:
        y_pred = np.asarray(y_pred).astype(int)
        labels = label_encoder.inverse_transform(y_pred)
        result["predictions"] = [str(v) for v in labels]
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_infer)
                result["probabilities"] = proba.tolist()
                result["label_classes"] = [str(c) for c in label_encoder.classes_]
            except Exception:
                pass
    else:
        result["predictions"] = [float(v) for v in np.asarray(y_pred).ravel()]

    return result
