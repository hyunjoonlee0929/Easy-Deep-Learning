"""Train/test workflows for Easy Deep Learning."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from Easy_Deep_Learning.core.automl import leaderboard_candidates, recommend_model, score_metrics
from Easy_Deep_Learning.core.data_validator import DataValidator
from Easy_Deep_Learning.core.experiment_tracker import ExperimentTracker
from Easy_Deep_Learning.core.model_registry import build_tabular_model
from Easy_Deep_Learning.core.preprocessing import AutoPreprocessor
from Easy_Deep_Learning.core.model_engine import ModelResult
from Easy_Deep_Learning.core.reporting import generate_ai_report, generate_html_report
from Easy_Deep_Learning.core.explainability import generate_explainability_artifacts
from Easy_Deep_Learning.core.error_analysis import generate_error_analysis
from Easy_Deep_Learning.core.recommendations import generate_model_recommendations
from Easy_Deep_Learning.core.tuning import run_auto_tuning
from Easy_Deep_Learning.core.trainer import Trainer, TrainingConfig

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("KMP_USE_SHM", "0")


@dataclass
class RunResult:
    """Container for persisted training run outputs."""

    run_id: str
    run_path: Path
    metrics: dict[str, float]


def set_global_seed(seed: int) -> None:
    """Set global random seed for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("KMP_USE_SHM", "0")
    random.seed(seed)
    np.random.seed(seed)

    if os.getenv("EASY_DL_ENABLE_TORCH", "0") == "1":
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file into dict."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_and_save(
    data_path: Path,
    config_path: Path,
    target_column: str,
    task_type: str,
    model_type: str,
    seed: int,
    model_params: dict[str, Any] | None = None,
) -> RunResult:
    """Run training pipeline and persist all artifacts."""
    set_global_seed(seed)

    cfg = load_yaml(config_path)
    dnn_cfg = cfg.get("dnn", {})

    df = pd.read_csv(data_path)
    if task_type == "regression":
        df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
        if df[target_column].isna().any():
            raise ValueError("Regression target must be numeric without missing values.")

    validator = DataValidator()
    validation = validator.validate(df=df, target_column=target_column)

    resolved_model_type = model_type
    auto_choice: dict[str, Any] | None = None
    params = model_params or {}
    if model_type == "auto":
        resolved_model_type, suggested_params = recommend_model(df, target_column, task_type)
        auto_choice = {
            "requested_model_type": "auto",
            "resolved_model_type": resolved_model_type,
            "suggested_params": suggested_params,
        }
        if not params:
            params = suggested_params

    preprocessor = AutoPreprocessor(test_size=0.2, random_state=seed)
    processed = preprocessor.fit_transform(df=df, target_column=target_column, task_type=task_type)

    if resolved_model_type == "dnn":
        params = {
            "hidden_layers": dnn_cfg.get("hidden_layers", [128, 64, 32]),
            "learning_rate": float(dnn_cfg.get("learning_rate", 1e-3)),
            "max_epochs": int(dnn_cfg.get("max_epochs", 200)),
            "patience": int(dnn_cfg.get("patience", 20)),
            "batch_size": int(dnn_cfg.get("batch_size", 32)),
            "random_state": seed,
            **params,
        }

    model = build_tabular_model(model_type=resolved_model_type, task_type=task_type, params=params)

    train_cfg = TrainingConfig(
        task_type=task_type,
        model_type=resolved_model_type,
        dnn_hidden_layers=params.get("hidden_layers", [128, 64, 32]),
        dnn_dropout=0.0,
        dnn_learning_rate=float(params.get("learning_rate", 1e-3)),
        dnn_max_epochs=int(params.get("max_epochs", 200)),
        dnn_patience=int(params.get("patience", 20)),
        dnn_batch_size=int(params.get("batch_size", 32)),
        random_state=seed,
    )

    if resolved_model_type == "dnn":
        trainer = Trainer()
        model_result = trainer.train(
            X_train=processed.X_train,
            y_train=processed.y_train,
            X_test=processed.X_test,
            y_test=processed.y_test,
            cfg=train_cfg,
        )
        model = model_result.model
    else:
        X_train = processed.X_train.toarray() if hasattr(processed.X_train, "toarray") else processed.X_train
        X_test = processed.X_test.toarray() if hasattr(processed.X_test, "toarray") else processed.X_test

        label_encoder = None
        y_train = processed.y_train
        y_test = processed.y_test
        if task_type == "classification":
            label_encoder = LabelEncoder()
            combined = np.concatenate([np.asarray(y_train), np.asarray(y_test)])
            label_encoder.fit(combined)
            y_train = label_encoder.transform(np.asarray(y_train))
            y_test = label_encoder.transform(np.asarray(y_test))

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

        model_result = ModelResult(
            model_name=resolved_model_type,
            metrics=metrics,
            model=model,
            task_type=task_type,
            label_classes=[str(c) for c in label_encoder.classes_] if label_encoder is not None else None,
            label_encoder=label_encoder,
        )

    tracker = ExperimentTracker(base_dir=Path("runs"))
    run_id, run_path = tracker.create_run(model_type=resolved_model_type)

    cfg_hash = tracker.config_hash(config_path)
    snapshot = {
        "project": "Easy Deep Learning",
        "run_id": run_id,
        "config_path": str(config_path.resolve()),
        "config_hash": cfg_hash,
        "seed": seed,
        "input": {
            "data_path": str(data_path.resolve()),
            "target_column": target_column,
            "task_type": task_type,
            "model_type": resolved_model_type,
        },
        "training_config": asdict(train_cfg),
        "model_params": params,
        "feature_names": processed.feature_names,
    }

    tracker.save_yaml(run_path / "config_snapshot.yaml", snapshot)
    tracker.save_json(run_path / "validation_report.json", validation.to_dict())
    tracker.save_json(run_path / "metrics.json", model_result.metrics)
    tracker.save_json(run_path / "feature_names.json", processed.feature_names)
    tracker.save_json(run_path / "model_params.json", params)
    if auto_choice is not None:
        tracker.save_json(run_path / "auto_recommendation.json", auto_choice)
    tracker.save_text(run_path / "config_hash.txt", cfg_hash)

    model_path = tracker.save_model_artifact(model=model_result.model, model_type=resolved_model_type, run_path=run_path)
    joblib.dump(processed.preprocessor, run_path / "preprocessor.joblib")
    if model_result.label_encoder is not None:
        joblib.dump(model_result.label_encoder, run_path / "label_encoder.joblib")

    tracker.save_json(
        run_path / "model_info.json",
        {
            "model_path": str(model_path.name),
            "model_type": resolved_model_type,
            "task_type": task_type,
            "target_column": target_column,
            "label_classes": model_result.label_classes,
        },
    )

    _save_prediction_artifacts(
        run_path=run_path,
        task_type=task_type,
        X_test=processed.X_test,
        y_test=processed.y_test,
        model=model_result.model,
        label_encoder=model_result.label_encoder,
    )

    try:
        generate_explainability_artifacts(
            run_path=run_path,
            model=model_result.model,
            X_test=processed.X_test,
            y_test=processed.y_test,
            feature_names=processed.feature_names,
            task_type=task_type,
        )
    except Exception:
        pass
    try:
        generate_error_analysis(
            run_path=run_path,
            model=model_result.model,
            X_test=processed.X_test,
            y_test=processed.y_test,
            task_type=task_type,
            label_encoder=model_result.label_encoder,
            raw_df=processed.X_test_raw,
        )
    except Exception:
        pass

    generate_ai_report(run_path)
    generate_model_recommendations(run_path)
    generate_html_report(run_path)
    return RunResult(run_id=run_id, run_path=run_path, metrics=model_result.metrics)


def auto_tune_and_train(
    data_path: Path,
    config_path: Path,
    target_column: str,
    task_type: str,
    model_type: str,
    seed: int,
    max_trials: int = 10,
) -> RunResult:
    """Run lightweight tuning and train the best model."""
    df = pd.read_csv(data_path)
    tune_result = run_auto_tuning(
        df=df,
        target_column=target_column,
        task_type=task_type,
        model_type=model_type,
        seed=seed,
        max_trials=max_trials,
    )

    result = train_and_save(
        data_path=data_path,
        config_path=config_path,
        target_column=target_column,
        task_type=task_type,
        model_type=model_type,
        seed=seed,
        model_params=tune_result.best_params,
    )

    tracker = ExperimentTracker(base_dir=Path("runs"))
    tracker.save_json(result.run_path / "tuning_results.json", tune_result.trials)
    tracker.save_json(result.run_path / "best_params.json", tune_result.best_params)
    tracker.save_json(result.run_path / "best_metrics.json", tune_result.best_metrics)
    return result


def run_leaderboard(
    data_path: Path,
    config_path: Path,
    target_column: str,
    task_type: str,
    seed: int,
    max_models: int | None = None,
) -> dict[str, Any]:
    """Train multiple models and save a leaderboard run."""
    set_global_seed(seed)
    df = pd.read_csv(data_path)

    tracker = ExperimentTracker(base_dir=Path("runs"))
    run_id, run_path = tracker.create_run(model_type="automl")
    cfg_hash = tracker.config_hash(config_path)

    candidates = leaderboard_candidates(task_type)
    if max_models is not None:
        candidates = candidates[: max_models]

    leaderboard: list[dict[str, Any]] = []
    for model_type, params in candidates:
        result = train_and_save(
            data_path=data_path,
            config_path=config_path,
            target_column=target_column,
            task_type=task_type,
            model_type=model_type,
            seed=seed,
            model_params=params,
        )
        score = score_metrics(task_type, result.metrics)
        leaderboard.append(
            {
                "run_id": result.run_id,
                "model_type": model_type,
                "metrics": result.metrics,
                "score": score,
            }
        )

    leaderboard = sorted(leaderboard, key=lambda x: x["score"], reverse=True)

    tracker.save_json(run_path / "leaderboard.json", leaderboard)
    tracker.save_json(run_path / "best_run.json", leaderboard[0] if leaderboard else {})
    tracker.save_text(run_path / "config_hash.txt", cfg_hash)
    tracker.save_yaml(
        run_path / "config_snapshot.yaml",
        {
            "project": "Easy Deep Learning",
            "run_id": run_id,
            "config_path": str(config_path.resolve()),
            "config_hash": cfg_hash,
            "seed": seed,
            "input": {
                "data_path": str(data_path.resolve()),
                "target_column": target_column,
                "task_type": task_type,
            },
        },
    )

    return {
        "project": "Easy Deep Learning",
        "run_id": run_id,
        "run_path": str(run_path.resolve()),
        "best_run": leaderboard[0] if leaderboard else None,
        "leaderboard": leaderboard,
        "config_hash": cfg_hash,
        "seed": seed,
    }


def _save_prediction_artifacts(
    run_path: Path,
    task_type: str,
    X_test: Any,
    y_test: Any,
    model: Any,
    label_encoder: LabelEncoder | None,
) -> None:
    """Persist prediction preview and confusion matrix image if applicable."""
    tracker = ExperimentTracker(base_dir=Path("runs"))
    X_infer = X_test.toarray() if hasattr(X_test, "toarray") else X_test
    y_pred_encoded = model.predict(X_infer)

    if task_type == "classification":
        if label_encoder is not None:
            y_true_encoded = label_encoder.transform(np.asarray(y_test))
            y_pred_encoded = np.asarray(y_pred_encoded).astype(int)
            y_true_labels = label_encoder.inverse_transform(y_true_encoded)
            y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
        else:
            y_true_labels = np.asarray(y_test)
            y_pred_labels = np.asarray(y_pred_encoded)

        preview = {
            "y_true": [str(v) for v in y_true_labels[:50]],
            "y_pred": [str(v) for v in y_pred_labels[:50]],
        }
        tracker.save_json(run_path / "predictions_preview.json", preview)

        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(y_true_labels, y_pred_labels, ax=ax)
            fig.tight_layout()
            fig.savefig(run_path / "confusion_matrix.png", dpi=150)
            plt.close(fig)

            if label_encoder is not None and len(label_encoder.classes_) == 2 and hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_infer)[:, 1]
                fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                RocCurveDisplay.from_predictions(y_true_encoded, y_score, ax=ax_roc)
                fig_roc.tight_layout()
                fig_roc.savefig(run_path / "roc_curve.png", dpi=150)
                plt.close(fig_roc)
        except Exception:
            pass
    else:
        preview = {
            "y_true": [float(v) for v in np.asarray(y_test).ravel()[:50]],
            "y_pred": [float(v) for v in np.asarray(y_pred_encoded).ravel()[:50]],
        }
        tracker.save_json(run_path / "predictions_preview.json", preview)
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(np.asarray(y_test).ravel(), np.asarray(y_pred_encoded).ravel(), alpha=0.6)
            ax.set_xlabel("y_true")
            ax.set_ylabel("y_pred")
            ax.set_title("Prediction Scatter")
            fig.tight_layout()
            fig.savefig(run_path / "prediction_scatter.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass


def _load_model(run_path: Path, model_type: str):
    """Load persisted model from run directory."""
    if (run_path / "model.json").exists():
        model_path = run_path / "model.json"
        try:
            from xgboost import XGBClassifier

            model = XGBClassifier()
            model.load_model(str(model_path))
            return model
        except Exception:
            try:
                from xgboost import XGBRegressor

                model = XGBRegressor()
                model.load_model(str(model_path))
                return model
            except Exception:
                pass

    for path in [run_path / "model.model", run_path / "model.pt", run_path / "model.bin"]:
        if path.exists():
            return joblib.load(path)

    raise FileNotFoundError("No model artifact found in run directory.")


def test_from_run(
    run_id: str,
    test_data_path: Path,
    target_column: str | None = None,
    save_artifacts: bool = False,
) -> dict[str, Any]:
    """Load saved run artifacts and evaluate on a new dataset."""
    run_path = Path("runs") / run_id
    if not run_path.exists():
        raise FileNotFoundError(f"Run '{run_id}' does not exist.")

    snapshot = load_yaml(run_path / "config_snapshot.yaml")
    model_info = json.loads((run_path / "model_info.json").read_text(encoding="utf-8"))

    model_type = model_info["model_type"]
    task_type = model_info["task_type"]
    expected_target = model_info["target_column"]
    target = target_column or expected_target

    df_test = pd.read_csv(test_data_path)
    if target not in df_test.columns:
        raise ValueError(f"Target column '{target}' not found in test data.")

    X_test_df = df_test.drop(columns=[target])
    y_true = df_test[target].to_numpy()
    if task_type == "regression":
        y_true = pd.to_numeric(pd.Series(y_true), errors="coerce").to_numpy()
        if np.isnan(y_true).any():
            raise ValueError("Regression target must be numeric without missing values.")

    preprocessor = joblib.load(run_path / "preprocessor.joblib")
    model = _load_model(run_path=run_path, model_type=model_type)

    X_test = preprocessor.transform(X_test_df)
    X_infer = X_test.toarray() if hasattr(X_test, "toarray") else X_test
    y_pred_encoded = model.predict(X_infer)

    label_encoder_path = run_path / "label_encoder.joblib"
    if label_encoder_path.exists():
        label_encoder = joblib.load(label_encoder_path)
        y_true_encoded = label_encoder.transform(y_true)
        y_pred = np.asarray(y_pred_encoded).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_true_encoded, y_pred)),
            "f1_weighted": float(f1_score(y_true_encoded, y_pred, average="weighted")),
        }
        preview = {
            "y_true": label_encoder.inverse_transform(y_true_encoded).tolist()[:10],
            "y_pred": label_encoder.inverse_transform(y_pred).tolist()[:10],
        }
    else:
        y_pred = np.asarray(y_pred_encoded)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics = {
            "rmse": rmse,
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
        preview = {
            "y_true": y_true.tolist()[:10],
            "y_pred": y_pred.tolist()[:10],
        }

    payload = {
        "project": "Easy Deep Learning",
        "run_id": run_id,
        "task_type": task_type,
        "model_type": model_type,
        "test_data_path": str(test_data_path.resolve()),
        "metrics": metrics,
        "prediction_preview": preview,
        "seed": snapshot.get("seed"),
        "config_hash": snapshot.get("config_hash"),
    }

    if save_artifacts:
        _save_prediction_artifacts(
            run_path=run_path,
            task_type=task_type,
            X_test=X_test,
            y_test=y_true,
            model=model,
            label_encoder=joblib.load(label_encoder_path) if label_encoder_path.exists() else None,
        )
        try:
            feature_names = json.loads((run_path / "feature_names.json").read_text(encoding="utf-8"))
            generate_explainability_artifacts(
                run_path=run_path,
                model=model,
                X_test=X_test,
                y_test=y_true,
                feature_names=feature_names,
                task_type=task_type,
            )
        except Exception:
            pass
        try:
            label_encoder = joblib.load(label_encoder_path) if label_encoder_path.exists() else None
            generate_error_analysis(
                run_path=run_path,
                model=model,
                X_test=X_test,
                y_test=y_true,
                task_type=task_type,
                label_encoder=label_encoder,
                raw_df=X_test_df,
            )
        except Exception:
            pass
        generate_ai_report(run_path)
        generate_model_recommendations(run_path)
        generate_html_report(run_path)

    return payload
