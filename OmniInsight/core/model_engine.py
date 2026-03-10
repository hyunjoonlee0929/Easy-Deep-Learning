"""Model factory and trainable model wrappers for Easy Deep Learning."""

from __future__ import annotations

import copy
import logging
import os
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
    label_encoder: Any | None = None


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


class _NumpyDNNBase:
    """Simple NumPy feed-forward network with early stopping."""

    def __init__(
        self,
        hidden_layers: list[int],
        learning_rate: float,
        max_epochs: int,
        patience: int,
        batch_size: int,
        random_state: int,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.random_state = random_state
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

    def _to_numpy(self, X: Any) -> np.ndarray:
        return X.toarray().astype(np.float64) if hasattr(X, "toarray") else np.asarray(X, dtype=np.float64)

    def _init_params(self, input_dim: int, output_dim: int) -> None:
        rng = np.random.default_rng(self.random_state)
        dims = [input_dim] + self.hidden_layers + [output_dim]
        self.weights = []
        self.biases = []
        for i in range(len(dims) - 1):
            w = rng.normal(0.0, np.sqrt(2.0 / dims[i]), size=(dims[i], dims[i + 1]))
            b = np.zeros((1, dims[i + 1]), dtype=np.float64)
            self.weights.append(w)
            self.biases.append(b)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def _relu_grad(self, x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float64)

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        zs: list[np.ndarray] = []
        a = X
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            a = self._relu(z)
            zs.append(z)
            activations.append(a)
        z_out = a @ self.weights[-1] + self.biases[-1]
        zs.append(z_out)
        activations.append(z_out)
        return activations, zs


class SklearnDNNRegressor(_NumpyDNNBase):
    """NumPy DNN regressor with mini-batch SGD and early stopping."""

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
        super().__init__(hidden_layers, learning_rate, max_epochs, patience, batch_size, random_state)
        self.best_weights: list[np.ndarray] = []
        self.best_biases: list[np.ndarray] = []

    def fit(self, X: Any, y: Any) -> "SklearnDNNRegressor":
        X_np = self._to_numpy(X)
        y_np = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n = len(X_np)
        split = max(1, int(n * 0.8))
        X_train, X_val = X_np[:split], X_np[split:]
        y_train, y_val = y_np[:split], y_np[split:]
        if len(X_val) == 0:
            X_val, y_val = X_train, y_train

        self._init_params(input_dim=X_np.shape[1], output_dim=1)
        best_val = float("inf")
        no_improve = 0
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.max_epochs):
            idx = rng.permutation(len(X_train))
            for s in range(0, len(idx), self.batch_size):
                batch = idx[s : s + self.batch_size]
                xb = X_train[batch]
                yb = y_train[batch]

                acts, zs = self._forward(xb)
                pred = acts[-1]
                grad = 2.0 * (pred - yb) / len(xb)

                for i in range(len(self.weights) - 1, -1, -1):
                    dW = acts[i].T @ grad
                    dB = grad.sum(axis=0, keepdims=True)
                    if i > 0:
                        grad = (grad @ self.weights[i].T) * self._relu_grad(zs[i - 1])
                    self.weights[i] -= self.learning_rate * dW
                    self.biases[i] -= self.learning_rate * dB

            val_pred = self.predict(X_val)
            val_loss = float(np.mean((val_pred.reshape(-1, 1) - y_val) ** 2))
            if val_loss < best_val:
                best_val = val_loss
                self.best_weights = [w.copy() for w in self.weights]
                self.best_biases = [b.copy() for b in self.biases]
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                break

        if self.best_weights:
            self.weights = self.best_weights
            self.biases = self.best_biases
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_np = self._to_numpy(X)
        acts, _ = self._forward(X_np)
        return acts[-1].ravel()


class SklearnDNNClassifier(_NumpyDNNBase):
    """NumPy DNN classifier with mini-batch SGD and early stopping."""

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
        super().__init__(hidden_layers, learning_rate, max_epochs, patience, batch_size, random_state)
        self.n_classes = 0
        self.best_weights: list[np.ndarray] = []
        self.best_biases: list[np.ndarray] = []

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        z = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def fit(self, X: Any, y: Any) -> "SklearnDNNClassifier":
        X_np = self._to_numpy(X)
        y_np = np.asarray(y, dtype=np.int64)
        self.n_classes = int(np.max(y_np) + 1)
        y_onehot = np.eye(self.n_classes)[y_np]

        n = len(X_np)
        split = max(1, int(n * 0.8))
        X_train, X_val = X_np[:split], X_np[split:]
        y_train, y_val = y_onehot[:split], y_onehot[split:]
        y_val_idx = y_np[split:]
        if len(X_val) == 0:
            X_val, y_val = X_train, y_train
            y_val_idx = y_np[:split]

        self._init_params(input_dim=X_np.shape[1], output_dim=self.n_classes)
        best_val = float("inf")
        no_improve = 0
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.max_epochs):
            idx = rng.permutation(len(X_train))
            for s in range(0, len(idx), self.batch_size):
                batch = idx[s : s + self.batch_size]
                xb = X_train[batch]
                yb = y_train[batch]

                acts, zs = self._forward(xb)
                logits = acts[-1]
                probs = self._softmax(logits)
                grad = (probs - yb) / len(xb)

                for i in range(len(self.weights) - 1, -1, -1):
                    dW = acts[i].T @ grad
                    dB = grad.sum(axis=0, keepdims=True)
                    if i > 0:
                        grad = (grad @ self.weights[i].T) * self._relu_grad(zs[i - 1])
                    self.weights[i] -= self.learning_rate * dW
                    self.biases[i] -= self.learning_rate * dB

            val_probs = self.predict_proba(X_val)
            eps = 1e-12
            val_loss = float(-np.mean(np.log(val_probs[np.arange(len(y_val_idx)), y_val_idx] + eps)))
            if val_loss < best_val:
                best_val = val_loss
                self.best_weights = [w.copy() for w in self.weights]
                self.best_biases = [b.copy() for b in self.biases]
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                break

        if self.best_weights:
            self.weights = self.best_weights
            self.biases = self.best_biases
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X_np = self._to_numpy(X)
        acts, _ = self._forward(X_np)
        return self._softmax(acts[-1])

    def predict(self, X: Any) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class ModelEngine:
    """Build baseline models for classification or regression."""

    def build_xgboost(
        self,
        task_type: str,
        num_classes: int | None = None,
        random_state: int = 42,
    ) -> PredictableModel:
        """Return an XGBoost model instance for the selected task."""
        if os.getenv("EASY_DL_ENABLE_XGBOOST", "0") != "1":
            logger.warning(
                "Using sklearn fallback for model_type='xgboost'. Set EASY_DL_ENABLE_XGBOOST=1 to enable native xgboost."
            )
            from sklearn.linear_model import LogisticRegression, Ridge

            if task_type == "classification":
                return LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    random_state=random_state,
                )
            if task_type == "regression":
                return Ridge(alpha=1.0, random_state=random_state)
            raise ValueError("task_type must be 'classification' or 'regression'.")

        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            logger.warning(
                "xgboost import failed. Falling back to sklearn for model_type='xgboost'."
            )
            from sklearn.linear_model import LogisticRegression, Ridge

            if task_type == "classification":
                return LogisticRegression(max_iter=1000, solver="liblinear", random_state=random_state)
            if task_type == "regression":
                return Ridge(alpha=1.0, random_state=random_state)
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
                    random_state=random_state,
                )
            return XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
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
                random_state=random_state,
            )

        raise ValueError("task_type must be 'classification' or 'regression'.")

    def evaluate_regression(self, y_true: Any, y_pred: Any) -> dict[str, float]:
        """Compute regression metrics."""
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
        }
