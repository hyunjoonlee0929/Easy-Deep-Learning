"""Model registry for tabular ML models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from Easy_Deep_Learning.core.model_engine import SklearnDNNClassifier, SklearnDNNRegressor


@dataclass
class ModelSpec:
    """Model specification with task-aware constructor."""

    name: str
    task_type: str
    model: Any


def build_tabular_model(model_type: str, task_type: str, params: dict[str, Any]) -> Any:
    """Instantiate a tabular model by type with parameters."""
    model_type = model_type.lower()

    if model_type == "dnn":
        hidden_layers = params.get("hidden_layers", [128, 64, 32])
        lr = float(params.get("learning_rate", 1e-3))
        epochs = int(params.get("max_epochs", 200))
        patience = int(params.get("patience", 20))
        batch_size = int(params.get("batch_size", 32))
        seed = int(params.get("random_state", 42))

        if task_type == "classification":
            return SklearnDNNClassifier(
                hidden_layers=hidden_layers,
                learning_rate=lr,
                max_epochs=epochs,
                patience=patience,
                batch_size=batch_size,
                random_state=seed,
            )
        return SklearnDNNRegressor(
            hidden_layers=hidden_layers,
            learning_rate=lr,
            max_epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            random_state=seed,
        )

    if model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        n_estimators = int(params.get("n_estimators", 200))
        max_depth = params.get("max_depth")
        if max_depth is not None:
            max_depth = int(max_depth)
        seed = int(params.get("random_state", 42))

        if task_type == "classification":
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=seed,
            )
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
        )

    if model_type == "svm":
        from sklearn.svm import SVC, SVR

        C = float(params.get("C", 1.0))
        kernel = params.get("kernel", "rbf")
        if task_type == "classification":
            return SVC(C=C, kernel=kernel, probability=True)
        return SVR(C=C, kernel=kernel)

    if model_type == "knn":
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

        n_neighbors = int(params.get("n_neighbors", 5))
        if task_type == "classification":
            return KNeighborsClassifier(n_neighbors=n_neighbors)
        return KNeighborsRegressor(n_neighbors=n_neighbors)

    if model_type == "lr":
        from sklearn.linear_model import LogisticRegression, Ridge

        if task_type == "classification":
            C = float(params.get("C", 1.0))
            return LogisticRegression(max_iter=1000, C=C, solver="liblinear")
        alpha = float(params.get("alpha", 1.0))
        return Ridge(alpha=alpha)

    if model_type == "gbm":
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        n_estimators = int(params.get("n_estimators", 200))
        learning_rate = float(params.get("learning_rate", 0.05))
        max_depth = int(params.get("max_depth", 3))
        if task_type == "classification":
            return GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
            )
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )

    if model_type == "xgboost":
        # Use sklearn fallback behavior from model_engine
        from Easy_Deep_Learning.core.model_engine import ModelEngine

        num_classes = int(params.get("num_classes", 0)) or None
        return ModelEngine().build_xgboost(task_type, num_classes=num_classes, random_state=int(params.get("random_state", 42)))

    raise ValueError(f"Unsupported model_type: {model_type}")
