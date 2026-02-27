"""Model factory and trainable model wrappers for OmniInsight."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class PredictableModel(Protocol):
    """Protocol for trainable predictive models."""

    def fit(self, X: Any, y: Any) -> Any: ...

    def predict(self, X: Any) -> Any: ...


@dataclass
class ModelResult:
    """Training result and evaluation metrics."""

    model_name: str
    metrics: dict[str, float]
    model: PredictableModel
    task_type: str
    label_classes: list[str] | None = None


class _TorchDNNBase:
    """Shared implementation for PyTorch MLP models."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        dropout: float,
        learning_rate: float,
        max_epochs: int,
        patience: int,
        batch_size: int,
        val_split: float,
        random_state: int,
    ) -> None:
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError("PyTorch is required for DNN models.") from exc

        self.torch = torch
        self.nn = nn
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_split = val_split
        self.random_state = random_state
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout

        self.model = self._build_model()
        self.best_epoch: int = 0
        self.best_val_loss: float = float("inf")

    def _build_hidden_stack(self) -> list[Any]:
        nn = self.nn
        layers: list[Any] = []
        prev_dim = self.input_dim
        for h in self.hidden_layers:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(self.dropout)])
            prev_dim = h
        return layers

    def _build_model(self) -> Any:
        raise NotImplementedError

    def _to_numpy(self, X: Any) -> np.ndarray:
        if hasattr(X, "toarray"):
            return X.toarray().astype(np.float32)
        return np.asarray(X, dtype=np.float32)

    def _make_val_split(self, X_np: np.ndarray, y_np: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(X_np) < 5:
            return X_np, X_np, y_np, y_np

        split_idx = int(len(X_np) * (1.0 - self.val_split))
        split_idx = min(max(split_idx, 1), len(X_np) - 1)

        return (
            X_np[:split_idx],
            X_np[split_idx:],
            y_np[:split_idx],
            y_np[split_idx:],
        )


class DNNRegressor(_TorchDNNBase):
    """PyTorch DNN regressor with early stopping on validation loss."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 32,
        val_split: float = 0.2,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            val_split=val_split,
            random_state=random_state,
        )

    def _build_model(self) -> Any:
        nn = self.nn
        layers = self._build_hidden_stack()
        last_hidden = self.hidden_layers[-1] if self.hidden_layers else self.input_dim
        if not self.hidden_layers:
            layers = []
        layers.append(nn.Linear(last_hidden, 1))
        return nn.Sequential(*layers)

    def fit(self, X: Any, y: Any) -> "DNNRegressor":
        """Train model with mini-batch loop and early stopping."""
        torch = self.torch
        nn = self.nn
        torch.manual_seed(self.random_state)

        X_np = self._to_numpy(X)
        y_np = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        X_train, X_val, y_train, y_val = self._make_val_split(X_np, y_np)

        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=min(self.batch_size, len(train_ds)),
            shuffle=True,
        )

        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        best_state = copy.deepcopy(self.model.state_dict())
        no_improve = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val_t)
                val_loss = float(criterion(val_preds, y_val_t).item())

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                break

        self.model.load_state_dict(best_state)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict numeric outputs."""
        torch = self.torch
        self.model.eval()
        X_t = torch.from_numpy(self._to_numpy(X))
        with torch.no_grad():
            y_hat = self.model(X_t).cpu().numpy().ravel()
        return y_hat


class DNNClassifier(_TorchDNNBase):
    """PyTorch DNN classifier with early stopping on validation loss."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_layers: list[int],
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 32,
        val_split: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.n_classes = n_classes
        super().__init__(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            val_split=val_split,
            random_state=random_state,
        )

    def _build_model(self) -> Any:
        nn = self.nn
        layers = self._build_hidden_stack()
        last_hidden = self.hidden_layers[-1] if self.hidden_layers else self.input_dim
        if not self.hidden_layers:
            layers = []
        layers.append(nn.Linear(last_hidden, self.n_classes))
        return nn.Sequential(*layers)

    def fit(self, X: Any, y: Any) -> "DNNClassifier":
        """Train model with mini-batch loop and early stopping."""
        torch = self.torch
        nn = self.nn
        torch.manual_seed(self.random_state)

        X_np = self._to_numpy(X)
        y_np = np.asarray(y, dtype=np.int64)

        X_train, X_val, y_train, y_val = self._make_val_split(X_np, y_np)

        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=min(self.batch_size, len(train_ds)),
            shuffle=True,
        )

        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_state = copy.deepcopy(self.model.state_dict())
        no_improve = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_t)
                val_loss = float(criterion(val_logits, y_val_t).item())

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                break

        self.model.load_state_dict(best_state)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels."""
        torch = self.torch
        self.model.eval()
        X_t = torch.from_numpy(self._to_numpy(X))
        with torch.no_grad():
            logits = self.model(X_t)
            preds = logits.argmax(dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities."""
        torch = self.torch
        self.model.eval()
        X_t = torch.from_numpy(self._to_numpy(X))
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs


class SklearnDNNRegressor:
    """Fallback DNN regressor using scikit-learn MLP with early stopping."""

    def __init__(
        self,
        hidden_layers: list[int],
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 32,
        random_state: int = 42,
    ) -> None:
        from sklearn.neural_network import MLPRegressor

        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layers),
            activation="relu",
            solver="adam",
            alpha=max(dropout, 1e-6),
            batch_size="auto",
            learning_rate_init=learning_rate,
            max_iter=max_epochs,
            n_iter_no_change=patience,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=random_state,
        )

    def fit(self, X: Any, y: Any) -> "SklearnDNNRegressor":
        X_in = X.toarray() if hasattr(X, "toarray") else X
        self.model.fit(X_in, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_in = X.toarray() if hasattr(X, "toarray") else X
        return np.asarray(self.model.predict(X_in))


class SklearnDNNClassifier:
    """Fallback DNN classifier using scikit-learn MLP with early stopping."""

    def __init__(
        self,
        hidden_layers: list[int],
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 32,
        random_state: int = 42,
    ) -> None:
        from sklearn.neural_network import MLPClassifier

        self.model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layers),
            activation="relu",
            solver="adam",
            alpha=max(dropout, 1e-6),
            batch_size="auto",
            learning_rate_init=learning_rate,
            max_iter=max_epochs,
            n_iter_no_change=patience,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=random_state,
        )

    def fit(self, X: Any, y: Any) -> "SklearnDNNClassifier":
        X_in = X.toarray() if hasattr(X, "toarray") else X
        self.model.fit(X_in, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_in = X.toarray() if hasattr(X, "toarray") else X
        return np.asarray(self.model.predict(X_in))

    def predict_proba(self, X: Any) -> np.ndarray:
        X_in = X.toarray() if hasattr(X, "toarray") else X
        return np.asarray(self.model.predict_proba(X_in))


class ModelEngine:
    """Build baseline models for classification or regression."""

    def build_xgboost(self, task_type: str, num_classes: int | None = None) -> PredictableModel:
        """Return an XGBoost model instance for the selected task."""
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            logger.warning(
                "xgboost is not installed. Falling back to sklearn gradient boosting for model_type='xgboost'."
            )
            from sklearn.linear_model import LogisticRegression, Ridge

            if task_type == "classification":
                return LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    random_state=42,
                )
            if task_type == "regression":
                return Ridge(alpha=1.0, random_state=42)
            raise ValueError("task_type must be 'classification' or 'regression'.") from exc

        if task_type == "classification":
            if num_classes is not None and num_classes > 2:
                return XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    num_class=num_classes,
                    random_state=42,
                )
            return XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            )

        if task_type == "regression":
            return XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=42,
            )

        raise ValueError("task_type must be 'classification' or 'regression'.")

    def evaluate_regression(self, y_true: Any, y_pred: Any) -> dict[str, float]:
        """Compute regression metrics."""
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
        }
