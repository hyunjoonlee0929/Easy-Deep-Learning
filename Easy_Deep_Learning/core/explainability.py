"""Explainability utilities (SHAP + PDP + ICE) for tabular models."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from Easy_Deep_Learning.core.experiment_tracker import ExperimentTracker


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)[:64]


def _to_numpy(X: Any) -> np.ndarray:
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _get_feature_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    task_type: str,
) -> list[tuple[str, float]]:
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)
    else:
        try:
            from sklearn.inspection import permutation_importance

            scoring = "accuracy" if task_type == "classification" else "r2"
            perm = permutation_importance(
                model,
                X,
                y,
                n_repeats=5,
                random_state=42,
                scoring=scoring,
            )
            importances = np.asarray(perm.importances_mean, dtype=float)
        except Exception:
            importances = np.zeros(len(feature_names), dtype=float)

    feature_importance = list(zip(feature_names, importances.tolist()))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    return feature_importance


def _plot_pdp_ice(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    top_features: list[str],
    run_path: Path,
) -> list[str]:
    paths: list[str] = []
    try:
        import matplotlib.pyplot as plt
        from sklearn.inspection import PartialDependenceDisplay
    except Exception:
        return paths

    for feat in top_features:
        if feat not in feature_names:
            continue
        idx = feature_names.index(feat)

        fig, ax = plt.subplots(figsize=(5, 4))
        PartialDependenceDisplay.from_estimator(model, X, [idx], feature_names=feature_names, ax=ax, kind="average")
        fig.tight_layout()
        path = run_path / f"pdp_{_sanitize(feat)}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path.name))

        fig, ax = plt.subplots(figsize=(5, 4))
        PartialDependenceDisplay.from_estimator(model, X, [idx], feature_names=feature_names, ax=ax, kind="individual")
        fig.tight_layout()
        path = run_path / f"ice_{_sanitize(feat)}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path.name))

    return paths


def _try_shap(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    run_path: Path,
) -> dict[str, Any]:
    try:
        import shap
        import matplotlib.pyplot as plt
    except Exception:
        return {"enabled": False, "reason": "shap not installed"}

    try:
        explainer = None
        if hasattr(model, "get_booster") or "xgboost" in model.__class__.__name__.lower():
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, X, feature_names=feature_names)
        elif hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)

        if explainer is None:
            return {"enabled": False, "reason": "unsupported model for SHAP"}

        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_array = np.mean([np.abs(v) for v in shap_values], axis=0)
        else:
            shap_array = np.abs(shap_values)

        mean_abs = np.mean(shap_array, axis=0)
        shap_importance = list(zip(feature_names, mean_abs.tolist()))
        shap_importance.sort(key=lambda x: x[1], reverse=True)

        fig = plt.figure(figsize=(7, 5))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        fig.tight_layout()
        summary_path = run_path / "shap_summary.png"
        fig.savefig(summary_path, dpi=150)
        plt.close(fig)

        return {
            "enabled": True,
            "summary_plot": str(summary_path.name),
            "importance": shap_importance[:20],
        }
    except Exception as exc:
        return {"enabled": False, "reason": str(exc)}


def _try_shap_interactions(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    run_path: Path,
) -> dict[str, Any]:
    try:
        import shap
        import matplotlib.pyplot as plt
    except Exception:
        return {"enabled": False, "reason": "shap not installed"}

    try:
        if not (hasattr(model, "get_booster") or "xgboost" in model.__class__.__name__.lower()):
            return {"enabled": False, "reason": "interaction requires tree model"}

        explainer = shap.TreeExplainer(model)
        inter = explainer.shap_interaction_values(X)

        if isinstance(inter, list):
            inter_vals = np.mean([np.abs(v) for v in inter], axis=0)
        else:
            inter_vals = np.abs(inter)

        mean_inter = np.mean(inter_vals, axis=0)
        n = mean_inter.shape[0]
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append(((feature_names[i], feature_names[j]), float(mean_inter[i, j])))
        pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = [{"pair": list(p), "importance": v} for p, v in pairs[:10]]

        fig = plt.figure(figsize=(7, 5))
        shap.summary_plot(inter, X, feature_names=feature_names, show=False)
        fig.tight_layout()
        path = run_path / "shap_interaction.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)

        return {
            "enabled": True,
            "summary_plot": str(path.name),
            "top_interactions": top_pairs,
        }
    except Exception as exc:
        return {"enabled": False, "reason": str(exc)}


def generate_explainability_artifacts(
    run_path: Path,
    model: Any,
    X_test: Any,
    y_test: Any,
    feature_names: list[str],
    task_type: str,
    top_k: int = 6,
) -> dict[str, Any]:
    """Generate explainability artifacts (top features, SHAP, PDP/ICE)."""
    tracker = ExperimentTracker(base_dir=Path("runs"))
    X = _to_numpy(X_test)
    y = np.asarray(y_test)

    if len(X) > 800:
        X = X[:800]
        y = y[:800]

    feature_importance = _get_feature_importance(model, X, y, feature_names, task_type)
    top_features = [f for f, _ in feature_importance[:top_k]]
    tracker.save_json(run_path / "top_features.json", feature_importance[:top_k])

    pdp_ice_paths = _plot_pdp_ice(model, X, feature_names, top_features, run_path)

    shap_result = _try_shap(model, X, feature_names, run_path)
    shap_interactions = _try_shap_interactions(model, X, feature_names, run_path)
    tracker.save_json(run_path / "shap_summary.json", shap_result)
    tracker.save_json(run_path / "shap_interactions.json", shap_interactions)

    payload = {
        "top_features": top_features,
        "pdp_ice": pdp_ice_paths,
        "shap": shap_result,
        "shap_interactions": shap_interactions,
    }
    tracker.save_json(run_path / "explainability.json", payload)
    return payload
