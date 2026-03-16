"""Error analysis utilities for tabular models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from Easy_Deep_Learning.core.experiment_tracker import ExperimentTracker


def _to_numpy(X: Any) -> np.ndarray:
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)

def _normalize_value(val: Any) -> Any:
    if isinstance(val, (np.integer, np.floating)):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: _normalize_value(v) for k, v in row.items()}

def _local_shap(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
) -> np.ndarray | None:
    try:
        import shap
    except Exception:
        return None

    try:
        explainer = None
        if hasattr(model, "get_booster") or "xgboost" in model.__class__.__name__.lower():
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, X, feature_names=feature_names)
        elif hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)

        if explainer is None:
            return None

        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_array = np.mean([np.abs(v) for v in shap_values], axis=0)
        else:
            shap_array = np.abs(shap_values)
        return shap_array
    except Exception:
        return None

def generate_error_analysis(
    run_path: Path,
    model: Any,
    X_test: Any,
    y_test: Any,
    task_type: str,
    label_encoder: Any | None = None,
    raw_df: Any | None = None,
    feature_names: list[str] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Generate error analysis artifacts and JSON summary."""
    tracker = ExperimentTracker(base_dir=Path("runs"))
    X = _to_numpy(X_test)
    y_true = np.asarray(y_test)
    y_pred = model.predict(X)

    payload: dict[str, Any] = {"task_type": task_type}

    if task_type == "classification":
        if label_encoder is not None:
            y_true_enc = label_encoder.transform(y_true)
            y_pred_enc = np.asarray(y_pred).astype(int)
            y_true_labels = label_encoder.inverse_transform(y_true_enc)
            y_pred_labels = label_encoder.inverse_transform(y_pred_enc)
        else:
            y_true_labels = y_true
            y_pred_labels = y_pred

        correct = np.asarray(y_true_labels) == np.asarray(y_pred_labels)
        errors_idx = np.where(~correct)[0]

        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
            except Exception:
                proba = None

        error_rows = []
        if errors_idx.size > 0:
            if proba is not None:
                conf = np.max(proba[errors_idx], axis=1)
                order = np.argsort(-conf)
                top_idx = errors_idx[order][:top_k]
                for idx in top_idx:
                    row = None
                    if raw_df is not None:
                        try:
                            row = _normalize_row(raw_df.iloc[int(idx)].to_dict())
                        except Exception:
                            row = None
                    error_rows.append(
                        {
                            "index": int(idx),
                            "y_true": str(y_true_labels[idx]),
                            "y_pred": str(y_pred_labels[idx]),
                            "confidence": float(np.max(proba[idx])),
                            "row": row,
                        }
                    )
            else:
                for idx in errors_idx[:top_k]:
                    row = None
                    if raw_df is not None:
                        try:
                            row = _normalize_row(raw_df.iloc[int(idx)].to_dict())
                        except Exception:
                            row = None
                    error_rows.append(
                        {
                            "index": int(idx),
                            "y_true": str(y_true_labels[idx]),
                            "y_pred": str(y_pred_labels[idx]),
                            "row": row,
                        }
                    )

        payload.update(
            {
                "total": int(len(y_true_labels)),
                "errors": int(len(errors_idx)),
                "error_rate": float(len(errors_idx) / max(1, len(y_true_labels))),
                "top_errors": error_rows,
            }
        )
    else:
        y_pred = np.asarray(y_pred).ravel()
        y_true = np.asarray(y_true).ravel()
        residuals = y_true - y_pred
        abs_res = np.abs(residuals)
        order = np.argsort(-abs_res)
        top_idx = order[:top_k]
        top_errors = [
            {
                "index": int(i),
                "y_true": float(y_true[i]),
                "y_pred": float(y_pred[i]),
                "error": float(residuals[i]),
                "abs_error": float(abs_res[i]),
                "row": _normalize_row(raw_df.iloc[int(i)].to_dict()) if raw_df is not None else None,
            }
            for i in top_idx
        ]

        payload.update(
            {
                "residual_mean": float(np.mean(residuals)),
                "residual_std": float(np.std(residuals)),
                "top_errors": top_errors,
            }
        )

    if feature_names and payload.get("top_errors"):
        indices = [row["index"] for row in payload["top_errors"] if "index" in row]
        if indices:
            X_sub = X[indices]
            shap_vals = _local_shap(model, X_sub, feature_names)
            if shap_vals is not None:
                for i, row in enumerate(payload["top_errors"]):
                    vals = shap_vals[i]
                    order = np.argsort(-np.abs(vals))[:5]
                    row["local_importance"] = [
                        {"feature": feature_names[idx], "value": float(vals[idx])}
                        for idx in order
                    ]
                    row["reason"] = "Top drivers: " + ", ".join([feature_names[idx] for idx in order])

                try:
                    import shap

                    first_vals = shap_vals[0]
                    base = np.mean(shap_vals, axis=0)
                    html = shap.force_plot(base, first_vals, feature_names=feature_names, matplotlib=False)
                    shap.save_html(str(run_path / "force_plot.html"), html)
                    payload["force_plot"] = "force_plot.html"
                except Exception:
                    pass

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(y_pred, residuals, alpha=0.6)
            ax.axhline(0, color="black", linewidth=1)
            ax.set_xlabel("y_pred")
            ax.set_ylabel("residual (y_true - y_pred)")
            ax.set_title("Residual Plot")
            fig.tight_layout()
            fig.savefig(run_path / "residuals.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass

    tracker.save_json(run_path / "error_analysis.json", payload)
    return payload
