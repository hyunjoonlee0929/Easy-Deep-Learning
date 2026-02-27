"""Training orchestration for OmniInsight models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from .model_engine import (
    DNNClassifier,
    DNNRegressor,
    ModelEngine,
    ModelResult,
    PredictableModel,
    SklearnDNNClassifier,
    SklearnDNNRegressor,
)


@dataclass
class TrainingConfig:
    """Configuration object used by Trainer."""

    task_type: str
    model_type: str
    dnn_hidden_layers: list[int]
    dnn_dropout: float = 0.2
    dnn_learning_rate: float = 1e-3
    dnn_max_epochs: int = 200
    dnn_patience: int = 20
    dnn_batch_size: int = 32
    dnn_val_split: float = 0.2
    random_state: int = 42


class Trainer:
    """Train either XGBoost or DNN models with unified interface."""

    def __init__(self, model_engine: ModelEngine | None = None) -> None:
        self.model_engine = model_engine or ModelEngine()

    def _build_dnn(self, X_train: Any, y_train: Any, cfg: TrainingConfig) -> PredictableModel:
        """Construct task-specific DNN model."""
        input_dim = int(X_train.shape[1])
        try:
            if cfg.task_type == "classification":
                n_classes = int(np.unique(y_train).shape[0])
                return DNNClassifier(
                    input_dim=input_dim,
                    n_classes=n_classes,
                    hidden_layers=cfg.dnn_hidden_layers,
                    dropout=cfg.dnn_dropout,
                    learning_rate=cfg.dnn_learning_rate,
                    max_epochs=cfg.dnn_max_epochs,
                    patience=cfg.dnn_patience,
                    batch_size=cfg.dnn_batch_size,
                    val_split=cfg.dnn_val_split,
                    random_state=cfg.random_state,
                )

            if cfg.task_type == "regression":
                return DNNRegressor(
                    input_dim=input_dim,
                    hidden_layers=cfg.dnn_hidden_layers,
                    dropout=cfg.dnn_dropout,
                    learning_rate=cfg.dnn_learning_rate,
                    max_epochs=cfg.dnn_max_epochs,
                    patience=cfg.dnn_patience,
                    batch_size=cfg.dnn_batch_size,
                    val_split=cfg.dnn_val_split,
                    random_state=cfg.random_state,
                )
        except ImportError:
            if cfg.task_type == "classification":
                return SklearnDNNClassifier(
                    hidden_layers=cfg.dnn_hidden_layers,
                    dropout=cfg.dnn_dropout,
                    learning_rate=cfg.dnn_learning_rate,
                    max_epochs=cfg.dnn_max_epochs,
                    patience=cfg.dnn_patience,
                    batch_size=cfg.dnn_batch_size,
                    random_state=cfg.random_state,
                )
            if cfg.task_type == "regression":
                return SklearnDNNRegressor(
                    hidden_layers=cfg.dnn_hidden_layers,
                    dropout=cfg.dnn_dropout,
                    learning_rate=cfg.dnn_learning_rate,
                    max_epochs=cfg.dnn_max_epochs,
                    patience=cfg.dnn_patience,
                    batch_size=cfg.dnn_batch_size,
                    random_state=cfg.random_state,
                )

        raise ValueError("task_type must be 'classification' or 'regression'.")

    def train(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        cfg: TrainingConfig,
    ) -> ModelResult:
        """Train configured model and return metrics + model instance."""
        model: PredictableModel
        label_encoder: LabelEncoder | None = None
        y_train_encoded: Any = y_train
        y_test_encoded: Any = y_test
        label_classes: list[str] | None = None

        if cfg.task_type == "classification":
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(np.asarray(y_train))
            y_test_encoded = label_encoder.transform(np.asarray(y_test))
            label_classes = [str(c) for c in label_encoder.classes_]

        if cfg.model_type == "xgboost":
            num_classes = int(np.unique(y_train_encoded).shape[0]) if cfg.task_type == "classification" else None
            model = self.model_engine.build_xgboost(cfg.task_type, num_classes=num_classes)
            X_train_fit = X_train.toarray() if hasattr(X_train, "toarray") else X_train
            X_test_fit = X_test.toarray() if hasattr(X_test, "toarray") else X_test
        elif cfg.model_type == "dnn":
            model = self._build_dnn(X_train, y_train_encoded, cfg)
            X_train_fit = X_train
            X_test_fit = X_test
        else:
            raise ValueError("model_type must be 'xgboost' or 'dnn'.")

        model.fit(X_train_fit, y_train_encoded)
        y_pred_encoded = model.predict(X_test_fit)

        if cfg.task_type == "classification":
            y_pred_encoded = np.asarray(y_pred_encoded).astype(int)
            metrics = {
                "accuracy": float(accuracy_score(y_test_encoded, y_pred_encoded)),
                "f1_weighted": float(f1_score(y_test_encoded, y_pred_encoded, average="weighted")),
            }
        else:
            y_pred = np.asarray(y_pred_encoded)
            metrics = self.model_engine.evaluate_regression(np.asarray(y_test_encoded), y_pred)

        metrics["n_train_samples"] = float(len(y_train))
        metrics["n_test_samples"] = float(len(y_test))

        return ModelResult(
            model_name=cfg.model_type,
            metrics=metrics,
            model=model,
            task_type=cfg.task_type,
            label_classes=label_classes,
        )
