"""Lightweight hyperparameter tuning (random search)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import LabelEncoder

from Easy_Deep_Learning.core.automl import score_metrics
from Easy_Deep_Learning.core.model_registry import build_tabular_model
from Easy_Deep_Learning.core.preprocessing import AutoPreprocessor


@dataclass
class TuneResult:
    model_type: str
    best_params: dict[str, Any]
    best_metrics: dict[str, float]
    trials: list[dict[str, Any]]
    scores: list[float]


def _search_space(model_type: str, task_type: str) -> dict[str, list[Any]]:
    if model_type == "rf":
        return {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [None, 4, 6, 8, 12],
        }
    if model_type == "gbm":
        return {
            "n_estimators": [100, 200, 400],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4],
        }
    if model_type == "xgboost":
        return {
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
        }
    if model_type == "svm":
        return {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
        }
    if model_type == "knn":
        return {
            "n_neighbors": [3, 5, 7, 11],
        }
    if model_type == "lr":
        if task_type == "classification":
            return {"C": [0.1, 1.0, 10.0]}
        return {"alpha": [0.1, 1.0, 10.0]}

    raise ValueError(f"Unsupported model_type for tuning: {model_type}")


def run_auto_tuning(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    model_type: str,
    seed: int,
    max_trials: int = 10,
) -> TuneResult:
    """Run random search for a given model and return best params."""
    preprocessor = AutoPreprocessor(test_size=0.2, random_state=seed)
    processed = preprocessor.fit_transform(df=df, target_column=target_column, task_type=task_type)

    X_train = processed.X_train.toarray() if hasattr(processed.X_train, "toarray") else processed.X_train
    X_test = processed.X_test.toarray() if hasattr(processed.X_test, "toarray") else processed.X_test
    y_train = processed.y_train
    y_test = processed.y_test

    label_encoder = None
    if task_type == "classification":
        label_encoder = LabelEncoder()
        combined = np.concatenate([np.asarray(y_train), np.asarray(y_test)])
        label_encoder.fit(combined)
        y_train = label_encoder.transform(np.asarray(y_train))
        y_test = label_encoder.transform(np.asarray(y_test))

    space = _search_space(model_type, task_type)
    sampler = ParameterSampler(space, n_iter=max_trials, random_state=seed)

    trials: list[dict[str, Any]] = []
    scores: list[float] = []
    best_score = -1e9
    best_params: dict[str, Any] = {}
    best_metrics: dict[str, float] = {}

    for params in sampler:
        model = build_tabular_model(model_type=model_type, task_type=task_type, params=params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if task_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
            }
        else:
            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
            }

        score = score_metrics(task_type, metrics)
        scores.append(float(score))
        trials.append({"params": params, "metrics": metrics, "score": score})
        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

    return TuneResult(
        model_type=model_type,
        best_params=best_params,
        best_metrics=best_metrics,
        trials=trials,
        scores=scores,
    )
