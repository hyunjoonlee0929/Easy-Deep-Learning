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
from Easy_Deep_Learning.core.data_quality import compute_data_quality
from Easy_Deep_Learning.core.drift import compute_drift
from Easy_Deep_Learning.core.cv import run_cross_validation
from Easy_Deep_Learning.core.advanced_modeling import (
    build_sample_weight,
    compute_class_weight_map,
    fit_with_optional_sample_weight,
    imbalance_profile,
    maybe_calibrate_classifier,
    regression_interval_from_residuals,
    resolve_resampling_strategy,
    resample_classification,
    tune_binary_threshold,
)
from Easy_Deep_Learning.core.mlops import finalize_run_tracking
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


def _resolve_auto_flag(value: Any, auto_default: bool) -> bool:
    """Resolve bool/auto flag values from model params."""
    if isinstance(value, bool):
        return value
    text = str(value or "auto").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return auto_default


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
    resolved = path
    if not resolved.exists():
        alt = Path(__file__).resolve().parents[1] / "config" / "model_config.yaml"
        if path.name == "model_config.yaml" and alt.exists():
            resolved = alt
    with resolved.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_and_save(
    data_path: Path,
    config_path: Path,
    target_column: str,
    task_type: str,
    model_type: str,
    seed: int,
    model_params: dict[str, Any] | None = None,
    reuse_if_exists: bool = True,
) -> RunResult:
    """Run training pipeline and persist all artifacts."""
    set_global_seed(seed)

    tracker = ExperimentTracker(base_dir=Path("runs"))

    cfg = load_yaml(config_path)
    dnn_cfg = cfg.get("dnn", {})

    df = pd.read_csv(data_path)
    if task_type == "regression":
        df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
        if df[target_column].isna().any():
            raise ValueError("Regression target must be numeric without missing values.")

    validator = DataValidator()
    validation = validator.validate(df=df, target_column=target_column)
    data_quality = compute_data_quality(df, target_column)

    resolved_model_type = model_type
    auto_choice: dict[str, Any] | None = None
    params = model_params or {}
    imbalance_report: dict[str, Any] = {}
    calibration_report: dict[str, Any] = {}
    threshold_report: dict[str, Any] = {"threshold_tuning_applied": False}
    decision_threshold = 0.5
    if model_type == "auto":
        resolved_model_type, suggested_params = recommend_model(df, target_column, task_type)
        auto_choice = {
            "requested_model_type": "auto",
            "resolved_model_type": resolved_model_type,
            "suggested_params": suggested_params,
        }
        if not params:
            params = suggested_params

    cfg_hash = tracker.config_hash(config_path)
    data_hash = tracker.file_hash(data_path)
    metadata = {
        "config_hash": cfg_hash,
        "data_hash": data_hash,
        "task_type": task_type,
        "target_column": target_column,
        "model_params": params,
    }
    if reuse_if_exists:
        existing = tracker.find_matching_run(resolved_model_type, metadata)
        if existing:
            run_path = Path("runs") / existing
            metrics_path = run_path / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            return RunResult(run_id=existing, run_path=run_path, metrics=metrics)

    preprocessor = AutoPreprocessor(test_size=0.2, random_state=seed)
    processed = preprocessor.fit_transform(df=df, target_column=target_column, task_type=task_type)

    if resolved_model_type == "dnn":
        if task_type == "classification":
            profile = imbalance_profile(np.asarray(processed.y_train))
            strategy = resolve_resampling_strategy(str(params.get("resampling_strategy", "auto")), profile)
            X_rs, y_rs, rs_report = resample_classification(processed.X_train, np.asarray(processed.y_train), strategy, seed)
            imbalance_report = {
                "profile": profile,
                "resampling_strategy": strategy,
                **rs_report,
                "class_weight_applied": False,
                "class_weight_note": "DNN path uses resampling only.",
            }
            processed.X_train = X_rs
            processed.y_train = y_rs

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

            profile = imbalance_profile(np.asarray(y_train))
            strategy = resolve_resampling_strategy(str(params.get("resampling_strategy", "auto")), profile)
            X_train, y_train, rs_report = resample_classification(X_train, np.asarray(y_train), strategy, seed)

            class_weight_enabled = _resolve_auto_flag(params.get("class_weight", "auto"), profile.get("is_imbalanced", False))
            class_weight_map = compute_class_weight_map(np.asarray(y_train)) if class_weight_enabled else None
            sample_weight = build_sample_weight(np.asarray(y_train), class_weight_map)
            imbalance_report = {
                "profile": profile,
                "resampling_strategy": strategy,
                **rs_report,
                "class_weight_applied": bool(class_weight_map),
                "class_weight_map": class_weight_map or {},
            }

            if class_weight_map and hasattr(model, "set_params"):
                try:
                    model.set_params(class_weight=class_weight_map)
                except Exception:
                    pass

            model = fit_with_optional_sample_weight(model, X_train, y_train, sample_weight)
        else:
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

    if task_type == "classification":
        y_true_enc = (
            model_result.label_encoder.transform(np.asarray(processed.y_test))
            if model_result.label_encoder is not None
            else np.asarray(processed.y_test).astype(int)
        )
        X_test_fit = processed.X_test.toarray() if hasattr(processed.X_test, "toarray") else processed.X_test

        calibration_enabled = _resolve_auto_flag(
            params.get("probability_calibration", "auto"),
            auto_default=bool(hasattr(model_result.model, "predict_proba")),
        )
        model_calibrated, calibration_report = maybe_calibrate_classifier(
            model=model_result.model,
            X_cal=X_test_fit,
            y_cal=np.asarray(y_true_enc),
            enabled=calibration_enabled,
        )
        model_result.model = model_calibrated

        threshold_enabled = _resolve_auto_flag(
            params.get("threshold_tuning", "auto"),
            auto_default=(hasattr(model_result.model, "predict_proba") and len(np.unique(y_true_enc)) == 2),
        )
        if hasattr(model_result.model, "predict_proba") and len(np.unique(y_true_enc)) == 2:
            probs = model_result.model.predict_proba(X_test_fit)[:, 1]
            decision_threshold, threshold_report = tune_binary_threshold(
                y_true=np.asarray(y_true_enc),
                proba_pos=np.asarray(probs),
                enabled=threshold_enabled,
            )
            y_pred_tuned = (np.asarray(probs) >= decision_threshold).astype(int)
            model_result.metrics["accuracy"] = float(accuracy_score(y_true_enc, y_pred_tuned))
            model_result.metrics["f1_weighted"] = float(f1_score(y_true_enc, y_pred_tuned, average="weighted"))

    run_id, run_path = tracker.create_run(model_type=resolved_model_type)

    snapshot = {
        "project": "Easy Deep Learning",
        "run_id": run_id,
        "config_path": str(config_path.resolve()),
        "config_hash": cfg_hash,
        "seed": seed,
        "input": {
            "data_path": str(data_path.resolve()),
            "data_hash": data_hash,
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
    tracker.save_json(run_path / "data_quality.json", data_quality)
    tracker.save_json(run_path / "metrics.json", model_result.metrics)
    tracker.save_json(run_path / "feature_names.json", processed.feature_names)
    tracker.save_json(run_path / "model_params.json", params)
    tracker.save_json(
        run_path / "run_metadata.json",
        {
            "model_type": resolved_model_type,
            **metadata,
        },
    )
    if auto_choice is not None:
        tracker.save_json(run_path / "auto_recommendation.json", auto_choice)
    if task_type == "classification":
        tracker.save_json(run_path / "imbalance_report.json", imbalance_report)
        tracker.save_json(run_path / "calibration_report.json", calibration_report)
        tracker.save_json(run_path / "threshold_report.json", threshold_report)
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
            "decision_threshold": float(decision_threshold) if task_type == "classification" else None,
        },
    )

    _save_prediction_artifacts(
        run_path=run_path,
        task_type=task_type,
        X_test=processed.X_test,
        y_test=processed.y_test,
        model=model_result.model,
        label_encoder=model_result.label_encoder,
        decision_threshold=float(decision_threshold),
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
            feature_names=processed.feature_names,
        )
    except Exception:
        pass
    try:
        drift = compute_drift(processed.X_train_raw, processed.X_test_raw)
        tracker.save_json(run_path / "drift_report.json", drift)
    except Exception:
        pass

    generate_ai_report(run_path)
    generate_model_recommendations(run_path)
    generate_html_report(run_path)
    finalize_run_tracking(
        run_path=run_path,
        run_type="tabular",
        task_type=task_type,
        model_type=resolved_model_type,
        dataset_hash=data_hash,
        metrics=model_result.metrics,
        model_params=params,
        model_artifact=model_path.name,
        config_hash=cfg_hash,
        seed=seed,
        extra={
            "target_column": target_column,
            "feature_count": len(processed.feature_names),
        },
    )
    return RunResult(run_id=run_id, run_path=run_path, metrics=model_result.metrics)


def cross_validate_and_report(
    data_path: Path,
    target_column: str,
    task_type: str,
    model_type: str,
    seed: int,
    folds: int = 5,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    df = pd.read_csv(data_path)
    result = run_cross_validation(
        df=df,
        target_column=target_column,
        task_type=task_type,
        model_type=model_type,
        seed=seed,
        model_params=model_params,
        folds=folds,
    )
    payload = {
        "model_type": model_type,
        "task_type": task_type,
        "folds": folds,
        "metrics": result.metrics,
        "scores": result.scores,
        "mean_metrics": result.mean_metrics,
    }
    return payload


def save_cv_report(payload: dict[str, Any]) -> Path:
    tracker = ExperimentTracker(base_dir=Path("runs"))
    run_id, run_path = tracker.create_run(model_type="cv")
    tracker.save_json(run_path / "cv_report.json", payload)

    html_rows = []
    for i, metrics in enumerate(payload.get("metrics", [])):
        html_rows.append(
            "<tr>"
            f"<td>{i+1}</td>"
            f"<td><pre>{json.dumps(metrics, indent=2)}</pre></td>"
            "</tr>"
        )
    html = (
        "<html><body>"
        "<h1>Cross Validation Report</h1>"
        f"<div>Run ID: {run_id}</div>"
        f"<div>Model: {payload.get('model_type')}</div>"
        f"<div>Task: {payload.get('task_type')}</div>"
        f"<div>Folds: {payload.get('folds')}</div>"
        f"<div>Mean Metrics: <pre>{json.dumps(payload.get('mean_metrics', {}), indent=2)}</pre></div>"
        "<table border='1' cellpadding='6' cellspacing='0'>"
        "<tr><th>Fold</th><th>Metrics</th></tr>"
        + "".join(html_rows)
        + "</table>"
        "</body></html>"
    )
    (run_path / "cv_report.html").write_text(html, encoding="utf-8")
    finalize_run_tracking(
        run_path=run_path,
        run_type="cv",
        task_type=str(payload.get("task_type", "classification")),
        model_type=str(payload.get("model_type", "cv")),
        dataset_hash=tracker.hash_payload(
            {
                "folds": payload.get("folds"),
                "mean_metrics": payload.get("mean_metrics", {}),
            }
        ),
        metrics=payload.get("mean_metrics", {}),
        model_params={"folds": payload.get("folds")},
        model_artifact=None,
        config_hash=None,
        seed=None,
        extra={"cv_report": True},
    )
    return run_path


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
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(tune_result.scores, marker="o")
        ax.set_title("Tuning Scores")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Score")
        fig.tight_layout()
        fig.savefig(result.run_path / "tuning_scores.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass
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
    mean_metrics = leaderboard[0]["metrics"] if leaderboard else {}
    finalize_run_tracking(
        run_path=run_path,
        run_type="automl",
        task_type=task_type,
        model_type="automl",
        dataset_hash=tracker.file_hash(data_path),
        metrics=mean_metrics,
        model_params={"max_models": len(candidates)},
        model_artifact=None,
        config_hash=cfg_hash,
        seed=seed,
        extra={"best_run_id": leaderboard[0]["run_id"] if leaderboard else None},
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
    decision_threshold: float = 0.5,
) -> None:
    """Persist prediction preview and confusion matrix image if applicable."""
    tracker = ExperimentTracker(base_dir=Path("runs"))
    X_infer = X_test.toarray() if hasattr(X_test, "toarray") else X_test
    if task_type == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_infer)
        if np.asarray(proba).ndim == 2 and np.asarray(proba).shape[1] == 2:
            y_pred_encoded = (np.asarray(proba)[:, 1] >= float(decision_threshold)).astype(int)
        else:
            y_pred_encoded = np.argmax(np.asarray(proba), axis=1)
    else:
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

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_infer)
                confidence = np.max(proba, axis=1)
                tracker.save_json(
                    run_path / "uncertainty.json",
                    {
                        "type": "classification_confidence",
                        "confidence": [float(v) for v in confidence[:200]],
                    },
                )
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

        try:
            residuals = np.asarray(y_test).ravel() - np.asarray(y_pred_encoded).ravel()
            res_std = float(np.std(residuals))
            tracker.save_json(
                run_path / "uncertainty.json",
                {"type": "regression_residual_std", "residual_std": res_std},
            )
            interval = regression_interval_from_residuals(
                y_true=np.asarray(y_test),
                y_pred=np.asarray(y_pred_encoded),
                alpha=0.1,
            )
            tracker.save_json(run_path / "prediction_interval.json", interval)
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
    run_path = Path(run_id)
    if not run_path.exists():
        run_path = Path("runs") / run_id
    if not run_path.exists():
        raise FileNotFoundError(f"Run '{run_id}' does not exist.")
    run_id = run_path.name

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
        decision_threshold = float(model_info.get("decision_threshold", 0.5))
        if hasattr(model, "predict_proba") and len(label_encoder.classes_) == 2:
            proba = model.predict_proba(X_infer)[:, 1]
            y_pred = (np.asarray(proba) >= decision_threshold).astype(int)
        else:
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
        interval_path = run_path / "prediction_interval.json"
        if interval_path.exists():
            interval_info = json.loads(interval_path.read_text(encoding="utf-8"))
            lo = float(interval_info.get("residual_quantile_low", 0.0))
            hi = float(interval_info.get("residual_quantile_high", 0.0))
            pred_arr = np.asarray(y_pred).ravel()
            preview["lower"] = [float(v) for v in (pred_arr + lo)[:10]]
            preview["upper"] = [float(v) for v in (pred_arr + hi)[:10]]

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
            decision_threshold=float(model_info.get("decision_threshold") or 0.5),
        )
        feature_names = None
        try:
            feature_names = json.loads((run_path / "feature_names.json").read_text(encoding="utf-8"))
        except Exception:
            feature_names = None
        try:
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
                feature_names=feature_names or [],
            )
        except Exception:
            pass
        # drift requires train/test; skip for external test-only runs
        generate_ai_report(run_path)
        generate_model_recommendations(run_path)
        generate_html_report(run_path)

    return payload
