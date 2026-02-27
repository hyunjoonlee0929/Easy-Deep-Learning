"""SHAP wrapper for model interpretation."""

from __future__ import annotations

from typing import Any

import numpy as np


class ShapEngine:
    """Compute SHAP values and extract top-k important features."""

    def _to_numpy(self, X: Any) -> np.ndarray:
        if hasattr(X, "toarray"):
            return X.toarray()
        return np.asarray(X)

    def _mean_abs_shap(self, shap_values: Any) -> np.ndarray:
        values = shap_values
        if isinstance(values, list):
            stacked = np.stack([np.abs(np.asarray(v)) for v in values], axis=0)
            return stacked.mean(axis=(0, 1))

        arr = np.asarray(values)
        if arr.ndim == 3:
            return np.abs(arr).mean(axis=(0, 2))
        return np.abs(arr).mean(axis=0)

    def _build_result(self, scores: np.ndarray, feature_names: list[str], top_k: int) -> dict[str, Any]:
        k = min(max(top_k, 1), len(feature_names))
        indices = np.argsort(scores)[::-1][:k]
        top_features = [feature_names[i] for i in indices]
        importance = {feature_names[i]: float(scores[i]) for i in indices}

        return {
            "status": "ok",
            "top_features": top_features,
            "feature_importance": importance,
        }

    def _prediction_permutation_scores(self, model: Any, X_eval: np.ndarray) -> np.ndarray:
        baseline = np.asarray(model.predict(X_eval)).reshape(-1)
        scores = np.zeros(X_eval.shape[1], dtype=float)

        for i in range(X_eval.shape[1]):
            perm = X_eval.copy()
            rng = np.random.default_rng(42 + i)
            perm[:, i] = rng.permutation(perm[:, i])
            pred = np.asarray(model.predict(perm)).reshape(-1)
            scores[i] = float(np.mean(np.abs(pred - baseline)))

        return scores

    def explain(
        self,
        model: Any,
        X_reference: Any,
        X_target: Any,
        feature_names: list[str],
        task_type: str,
        model_type: str,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Return SHAP top feature information for XGBoost and DNN models."""
        X_bg = self._to_numpy(X_reference)
        X_eval = self._to_numpy(X_target)

        bg_size = min(100, len(X_bg))
        eval_size = min(200, len(X_eval))
        X_bg = X_bg[:bg_size]
        X_eval = X_eval[:eval_size]

        if X_bg.size == 0 or X_eval.size == 0:
            return {
                "status": "error",
                "error": "Insufficient samples for SHAP explanation.",
                "top_features": [],
                "feature_importance": {},
            }

        try:
            import shap
        except ImportError:
            scores = self._prediction_permutation_scores(model, X_eval)
            result = self._build_result(scores=scores, feature_names=feature_names, top_k=top_k)
            result["status"] = "approximate"
            result["explainer"] = "PermutationFallback"
            result["error"] = "shap package is not installed."
            return result

        try:
            if model_type == "xgboost":
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_eval)
                    scores = self._mean_abs_shap(shap_values)
                    result = self._build_result(scores=scores, feature_names=feature_names, top_k=top_k)
                    result["explainer"] = "TreeExplainer"
                    return result
                except Exception:
                    explainer = shap.KernelExplainer(model.predict, X_bg)
                    shap_values = explainer.shap_values(X_eval, nsamples=min(200, X_bg.shape[0] * 2 + 20))
                    scores = self._mean_abs_shap(shap_values)
                    result = self._build_result(scores=scores, feature_names=feature_names, top_k=top_k)
                    result["explainer"] = "KernelExplainer"
                    return result

            if model_type == "dnn":
                if task_type == "classification" and hasattr(model, "predict_proba"):
                    predict_fn = model.predict_proba
                else:
                    predict_fn = model.predict

                explainer = shap.KernelExplainer(predict_fn, X_bg)
                shap_values = explainer.shap_values(X_eval, nsamples=min(200, X_bg.shape[0] * 2 + 20))
                scores = self._mean_abs_shap(shap_values)
                result = self._build_result(scores=scores, feature_names=feature_names, top_k=top_k)
                result["explainer"] = "KernelExplainer"
                return result

            return {
                "status": "error",
                "error": f"Unsupported model_type: {model_type}",
                "top_features": [],
                "feature_importance": {},
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": f"SHAP computation failed: {exc}",
                "top_features": [],
                "feature_importance": {},
            }
