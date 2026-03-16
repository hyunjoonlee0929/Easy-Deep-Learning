"""Cross-validation utilities for tabular models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

from Easy_Deep_Learning.core.model_registry import build_tabular_model
from Easy_Deep_Learning.core.preprocessing import AutoPreprocessor
from Easy_Deep_Learning.core.automl import score_metrics


@dataclass
class CVResult:
    model_type: str
    task_type: str
    metrics: list[dict[str, float]]
    scores: list[float]
    mean_metrics: dict[str, float]


def run_cross_validation(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    model_type: str,
    seed: int,
    model_params: dict[str, Any] | None = None,
    folds: int = 5,
) -> CVResult:
    model_params = model_params or {}
    X = df.drop(columns=[target_column])
    y = df[target_column]

    splitter = None
    if task_type == "classification":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)

    metrics_list: list[dict[str, float]] = []
    scores: list[float] = []

    for train_idx, test_idx in splitter.split(X, y):
        fold_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        preprocessor = AutoPreprocessor(test_size=0.2, random_state=seed)
        processed = preprocessor.fit_transform(df=fold_df, target_column=target_column, task_type=task_type)

        X_train = processed.X_train.toarray() if hasattr(processed.X_train, "toarray") else processed.X_train
        X_val = processed.X_test.toarray() if hasattr(processed.X_test, "toarray") else processed.X_test
        y_train = processed.y_train
        y_val = processed.y_test

        # override validation with actual split data for consistency
        X_val_df = test_df.drop(columns=[target_column])
        y_val = test_df[target_column]
        X_val = preprocessor.transform(X_val_df)
        X_val = X_val.toarray() if hasattr(X_val, "toarray") else X_val

        label_encoder = None
        if task_type == "classification":
            label_encoder = LabelEncoder()
            combined = np.concatenate([np.asarray(y_train), np.asarray(y_val)])
            label_encoder.fit(combined)
            y_train = label_encoder.transform(np.asarray(y_train))
            y_val = label_encoder.transform(np.asarray(y_val))

        model = build_tabular_model(model_type=model_type, task_type=task_type, params=model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if task_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "f1_weighted": float(f1_score(y_val, y_pred, average="weighted")),
            }
        else:
            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
                "mae": float(mean_absolute_error(y_val, y_pred)),
                "r2": float(r2_score(y_val, y_pred)),
            }
        score = score_metrics(task_type, metrics)
        metrics_list.append(metrics)
        scores.append(float(score))

    mean_metrics = {}
    if metrics_list:
        keys = metrics_list[0].keys()
        for key in keys:
            mean_metrics[key] = float(np.mean([m[key] for m in metrics_list]))

    return CVResult(
        model_type=model_type,
        task_type=task_type,
        metrics=metrics_list,
        scores=scores,
        mean_metrics=mean_metrics,
    )
