"""Automated preprocessing pipeline for tabular data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ProcessedData:
    """Container for preprocessed train/test splits."""

    X_train: object
    X_test: object
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    feature_names: list[str]


class AutoPreprocessor:
    """Handle imputation, scaling, encoding, and train-test split."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state

    def fit_transform(self, df: pd.DataFrame, target_column: str) -> ProcessedData:
        """Fit preprocessing pipeline and return processed splits."""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' does not exist.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ],
            remainder="drop",
        )

        can_stratify = y.nunique() > 1 and y.nunique() <= max(20, int(len(y) * 0.1))
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if can_stratify else None,
        )

        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)

        raw_feature_names = preprocessor.get_feature_names_out()
        feature_names = [str(name) for name in np.asarray(raw_feature_names).tolist()]

        return ProcessedData(
            X_train=X_train_t,
            X_test=X_test_t,
            y_train=y_train,
            y_test=y_test,
            preprocessor=preprocessor,
            feature_names=feature_names,
        )
