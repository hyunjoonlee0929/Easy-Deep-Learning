"""Simple auto-model recommendation and leaderboard logic."""

from __future__ import annotations

from typing import Any

import pandas as pd


def recommend_model(df: pd.DataFrame, target_column: str, task_type: str) -> tuple[str, dict[str, Any]]:
    """Recommend a baseline model based on dataset size/type."""
    n_rows = len(df)
    n_cols = len(df.columns) - 1

    if task_type == "classification":
        if n_rows < 2000:
            return "rf", {"n_estimators": 300, "max_depth": 6}
        if n_cols > 200:
            return "lr", {"C": 1.0}
        return "gbm", {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3}

    # regression
    if n_rows < 2000:
        return "gbm", {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3}
    if n_cols > 200:
        return "lr", {"alpha": 1.0}
    return "rf", {"n_estimators": 300, "max_depth": 8}


def leaderboard_candidates(task_type: str) -> list[tuple[str, dict[str, Any]]]:
    """Return a small grid of candidate models for quick leaderboard runs."""
    if task_type == "classification":
        return [
            ("rf", {"n_estimators": 300, "max_depth": 8}),
            ("gbm", {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3}),
            ("svm", {"C": 1.0, "kernel": "rbf"}),
            ("knn", {"n_neighbors": 7}),
            ("lr", {"C": 1.0}),
            ("dnn", {"hidden_layers": [128, 64], "learning_rate": 1e-3, "max_epochs": 200, "patience": 20, "batch_size": 32}),
        ]

    return [
        ("rf", {"n_estimators": 300, "max_depth": 10}),
        ("gbm", {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3}),
        ("svm", {"C": 1.0, "kernel": "rbf"}),
        ("knn", {"n_neighbors": 7}),
        ("lr", {"alpha": 1.0}),
        ("dnn", {"hidden_layers": [128, 64], "learning_rate": 1e-3, "max_epochs": 200, "patience": 20, "batch_size": 32}),
    ]


def score_metrics(task_type: str, metrics: dict[str, float]) -> float:
    """Return a single scalar score to rank runs."""
    if task_type == "classification":
        if "f1_weighted" in metrics:
            return float(metrics["f1_weighted"])
        return float(metrics.get("accuracy", 0.0))

    return float(metrics.get("r2", -1e9))
