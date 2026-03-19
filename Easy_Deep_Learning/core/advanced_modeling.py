"""Advanced tabular modeling helpers: imbalance, calibration, and intervals."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, f1_score


def imbalance_profile(y: np.ndarray) -> dict[str, Any]:
    """Return imbalance statistics for a classification label array."""
    values, counts = np.unique(y, return_counts=True)
    pairs = sorted(zip(values.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True)
    majority = int(pairs[0][1]) if pairs else 0
    minority = int(pairs[-1][1]) if pairs else 0
    ratio = float(minority / majority) if majority > 0 else 1.0
    return {
        "class_counts": {str(v): int(c) for v, c in pairs},
        "majority_count": majority,
        "minority_count": minority,
        "imbalance_ratio": ratio,
        "is_imbalanced": ratio < 0.6,
    }


def resolve_resampling_strategy(strategy: str, profile: dict[str, Any]) -> str:
    """Resolve resampling strategy from explicit or auto mode."""
    strategy = (strategy or "auto").lower()
    if strategy != "auto":
        return strategy
    return "oversample" if profile.get("is_imbalanced", False) else "none"


def compute_class_weight_map(y: np.ndarray) -> dict[int, float]:
    """Compute inverse-frequency class weights."""
    values, counts = np.unique(y, return_counts=True)
    total = float(np.sum(counts))
    n_cls = float(len(values))
    return {int(v): float(total / (n_cls * c)) for v, c in zip(values, counts)}


def build_sample_weight(y: np.ndarray, class_weight_map: dict[int, float] | None) -> np.ndarray | None:
    """Build sample weight vector from per-class weights."""
    if not class_weight_map:
        return None
    return np.asarray([class_weight_map.get(int(v), 1.0) for v in y], dtype=np.float64)


def resample_classification(
    X: Any,
    y: np.ndarray,
    strategy: str,
    seed: int,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Apply simple over/under-sampling for classification training data."""
    strategy = (strategy or "none").lower()
    if strategy == "none":
        return X, y, {"resampling_applied": "none"}

    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X, y, {"resampling_applied": "none", "reason": "single_class"}

    class_to_idx = {c: np.where(y == c)[0] for c in classes}
    if strategy == "oversample":
        target = int(np.max(counts))
        sampled = []
        for c in classes:
            idx = class_to_idx[c]
            sampled_idx = rng.choice(idx, size=target, replace=True)
            sampled.append(sampled_idx)
        final_idx = np.concatenate(sampled)
    elif strategy == "undersample":
        target = int(np.min(counts))
        sampled = []
        for c in classes:
            idx = class_to_idx[c]
            sampled_idx = rng.choice(idx, size=target, replace=False)
            sampled.append(sampled_idx)
        final_idx = np.concatenate(sampled)
    else:
        return X, y, {"resampling_applied": "none", "reason": "unknown_strategy"}

    rng.shuffle(final_idx)
    X_rs = X[final_idx]
    y_rs = y[final_idx]
    return X_rs, y_rs, {
        "resampling_applied": strategy,
        "before_count": int(len(y)),
        "after_count": int(len(y_rs)),
        "after_profile": imbalance_profile(y_rs),
    }


def fit_with_optional_sample_weight(model: Any, X: Any, y: np.ndarray, sample_weight: np.ndarray | None) -> Any:
    """Fit a model and use sample_weight if supported by estimator."""
    if sample_weight is None:
        model.fit(X, y)
        return model
    try:
        model.fit(X, y, sample_weight=sample_weight)
        return model
    except TypeError:
        model.fit(X, y)
        return model


def maybe_calibrate_classifier(
    model: Any,
    X_cal: Any,
    y_cal: np.ndarray,
    enabled: bool,
) -> tuple[Any, dict[str, Any]]:
    """Calibrate classifier probabilities with sigmoid calibration."""
    if not enabled or not hasattr(model, "predict_proba"):
        return model, {"calibration_applied": False}

    try:
        before = model.predict_proba(X_cal)
        # Use isotonic calibration to avoid optimizer instability in some scipy/macOS builds.
        model_cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        model_cal.fit(X_cal, y_cal)
        after = model_cal.predict_proba(X_cal)
        payload = {"calibration_applied": True}
        if before.shape[1] == 2:
            payload["brier_before"] = float(brier_score_loss(y_cal, before[:, 1]))
            payload["brier_after"] = float(brier_score_loss(y_cal, after[:, 1]))
        return model_cal, payload
    except Exception as exc:
        return model, {"calibration_applied": False, "calibration_error": str(exc)}


def tune_binary_threshold(
    y_true: np.ndarray,
    proba_pos: np.ndarray,
    enabled: bool,
) -> tuple[float, dict[str, Any]]:
    """Tune binary threshold by maximizing weighted F1."""
    if not enabled:
        return 0.5, {"threshold_tuning_applied": False}

    thresholds = np.linspace(0.1, 0.9, 17)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        pred = (proba_pos >= t).astype(int)
        score = float(f1_score(y_true, pred, average="weighted"))
        if score > best_f1:
            best_f1 = score
            best_t = float(t)

    return best_t, {
        "threshold_tuning_applied": True,
        "best_threshold": best_t,
        "best_f1_weighted": best_f1,
    }


def regression_interval_from_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """Build simple prediction interval summary from residual quantiles."""
    residuals = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    lo_q = float(np.quantile(residuals, alpha / 2.0))
    hi_q = float(np.quantile(residuals, 1.0 - alpha / 2.0))
    lower = np.asarray(y_pred).ravel() + lo_q
    upper = np.asarray(y_pred).ravel() + hi_q
    truth = np.asarray(y_true).ravel()
    coverage = float(np.mean((truth >= lower) & (truth <= upper)))
    return {
        "interval_alpha": float(alpha),
        "interval_confidence": float(1.0 - alpha),
        "residual_quantile_low": lo_q,
        "residual_quantile_high": hi_q,
        "empirical_coverage": coverage,
        "preview": {
            "y_pred": [float(v) for v in np.asarray(y_pred).ravel()[:50]],
            "lower": [float(v) for v in lower[:50]],
            "upper": [float(v) for v in upper[:50]],
        },
    }
