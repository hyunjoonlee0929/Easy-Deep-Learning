"""Streamlit dashboard for Easy Deep Learning."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from Easy_Deep_Learning.core.logging_utils import configure_logging
from Easy_Deep_Learning.core.automl import recommend_model
from Easy_Deep_Learning.core.workflows import run_leaderboard, test_from_run, train_and_save, auto_tune_and_train

configure_logging("INFO")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Easy Deep Learning", layout="wide")
st.title("Easy Deep Learning")
st.caption("CSV 기반 분류/회귀 모델 학습, 저장, 재평가를 빠르게 수행합니다.")

st.sidebar.header("Quick Start")
quick_action = st.sidebar.radio(
    "Choose a quick workflow",
    options=["None", "Quick Classification", "Quick Regression", "Quick Image Demo", "Quick Text Demo"],
    index=0,
)
if quick_action != "None":
    st.sidebar.info("Quick Start presets loaded. Adjust if needed.")
if st.sidebar.button("Clear Quick Preset"):
    for key in ["train_source_default", "train_preset_default"]:
        if key in st.session_state:
            del st.session_state[key]

st.sidebar.header("OpenAI API Key")
api_key_input = st.sidebar.text_input("API Key", type="password", value=st.session_state.get("openai_api_key", ""))
if api_key_input:
    st.session_state["openai_api_key"] = api_key_input
    os.environ["OPENAI_API_KEY"] = api_key_input
    st.sidebar.success("API key set for this session.")
else:
    st.sidebar.info("키를 입력하면 챗봇/요약 기능에서 OpenAI 사용 가능")

st.sidebar.header("System Status")
def _dep_status(module_name: str) -> tuple[bool, str]:
    try:
        __import__(module_name)
        return True, "OK"
    except Exception as exc:
        return False, str(exc)

with st.sidebar.expander("Dependencies", expanded=False):
    for mod in ["fastapi", "uvicorn", "shap", "torch", "torchvision", "torchtext"]:
        ok, detail = _dep_status(mod)
        if ok:
            st.write(f"OK: {mod}")
        else:
            st.write(f"Missing: {mod} ({detail.splitlines()[0]})")

st.sidebar.header("Recent Runs")
runs_dir = Path("runs")
run_ids_sidebar = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()], reverse=True) if runs_dir.exists() else []
selected_sidebar_run = st.sidebar.selectbox("Run ID", options=[""] + run_ids_sidebar)
if selected_sidebar_run:
    st.sidebar.write(selected_sidebar_run)
    info_path = Path("runs") / selected_sidebar_run / "model_info.json"
    metrics_path = Path("runs") / selected_sidebar_run / "metrics.json"
    if info_path.exists():
        st.sidebar.subheader("Model Info")
        st.sidebar.json(json.loads(info_path.read_text(encoding="utf-8")))
    if metrics_path.exists():
        st.sidebar.subheader("Metrics")
        st.sidebar.json(json.loads(metrics_path.read_text(encoding="utf-8")))
    report_path = Path("runs") / selected_sidebar_run / "report.html"
    if report_path.exists():
        with report_path.open("rb") as f:
            st.sidebar.download_button("Download report.html", f, file_name="report.html", mime="text/html")

@st.cache_data
def load_preset_dataset(name: str) -> tuple[pd.DataFrame, str]:
    """Load a built-in dataset and return (df, target_column)."""
    from sklearn.datasets import (
        load_breast_cancer,
        load_diabetes,
        load_digits,
        load_iris,
        load_wine,
    )

    if name == "Iris (classification)":
        data = load_iris(as_frame=True)
    elif name == "Breast Cancer (classification)":
        data = load_breast_cancer(as_frame=True)
    elif name == "Wine (classification)":
        data = load_wine(as_frame=True)
    elif name == "Digits (classification)":
        data = load_digits(as_frame=True)
    elif name == "Diabetes (regression)":
        data = load_diabetes(as_frame=True)
    elif name == "California Housing (regression)":
        try:
            from sklearn.datasets import fetch_california_housing

            data = fetch_california_housing(as_frame=True)
        except Exception:
            data = load_diabetes(as_frame=True)
    else:
        raise ValueError(f"Unknown dataset preset: {name}")

    df = data.frame.copy()
    target_col = "target"
    if target_col not in df.columns:
        df[target_col] = data.target
    return df, target_col


def pick_data_source(prefix: str) -> tuple[pd.DataFrame | None, str | None]:
    """Pick dataset from upload/preset."""
    default_source = st.session_state.get(f"{prefix}_source_default")
    source_options = ["Upload CSV", "Preset Dataset"]
    default_index = source_options.index(default_source) if default_source in source_options else 0

    source = st.radio(
        "Data source",
        options=source_options,
        index=default_index,
        key=f"{prefix}_source",
        horizontal=True,
    )

    if source == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key=f"{prefix}_upload")
        if uploaded is None:
            return None, None
        df = pd.read_csv(uploaded)
        return df, None

    preset_options = [
        "Iris (classification)",
        "Breast Cancer (classification)",
        "Wine (classification)",
        "Digits (classification)",
        "Diabetes (regression)",
        "California Housing (regression)",
    ]
    default_preset = st.session_state.get(f"{prefix}_preset_default")
    preset_index = preset_options.index(default_preset) if default_preset in preset_options else 0

    preset = st.selectbox(
        "Preset Dataset",
        options=preset_options,
        index=preset_index,
        key=f"{prefix}_preset",
    )
    df, target = load_preset_dataset(preset)
    return df, target


def model_param_controls(model_type: str) -> dict[str, Any]:
    """Render model parameter controls and return params dict."""
    params: dict[str, Any] = {}

    if model_type == "dnn":
        hidden_layers = st.text_input("Hidden layers", value="128,64,32", key="tab_dnn_layers")
        params["hidden_layers"] = [int(x.strip()) for x in hidden_layers.split(",") if x.strip()]
        params["learning_rate"] = st.number_input("Learning rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f", key="tab_dnn_lr")
        params["max_epochs"] = st.number_input("Max epochs", min_value=10, max_value=2000, value=200, step=10, key="tab_dnn_epochs")
        params["patience"] = st.number_input("Early stop patience", min_value=3, max_value=200, value=20, step=1, key="tab_dnn_patience")
        params["batch_size"] = st.number_input("Batch size", min_value=4, max_value=1024, value=32, step=4, key="tab_dnn_batch")

    if model_type == "rf":
        params["n_estimators"] = st.number_input("n_estimators", min_value=10, max_value=1000, value=200, step=10, key="tab_rf_estimators")
        params["max_depth"] = st.number_input("max_depth (0 = None)", min_value=0, max_value=100, value=0, step=1, key="tab_rf_depth")
        if params["max_depth"] == 0:
            params["max_depth"] = None

    if model_type == "svm":
        params["C"] = st.number_input("C", min_value=0.01, max_value=100.0, value=1.0, step=0.1, key="tab_svm_c")
        params["kernel"] = st.selectbox("kernel", options=["rbf", "linear", "poly", "sigmoid"], key="tab_svm_kernel")

    if model_type == "knn":
        params["n_neighbors"] = st.number_input("n_neighbors", min_value=1, max_value=50, value=5, step=1, key="tab_knn_k")

    if model_type == "lr":
        params["C"] = st.number_input("C (classification)", min_value=0.01, max_value=100.0, value=1.0, step=0.1, key="tab_lr_c")
        params["alpha"] = st.number_input("alpha (regression)", min_value=0.01, max_value=100.0, value=1.0, step=0.1, key="tab_lr_alpha")

    if model_type == "gbm":
        params["n_estimators"] = st.number_input("n_estimators", min_value=50, max_value=1000, value=200, step=10, key="tab_gbm_estimators")
        params["learning_rate"] = st.number_input("learning_rate", min_value=0.001, max_value=0.5, value=0.05, format="%.3f", key="tab_gbm_lr")
        params["max_depth"] = st.number_input("max_depth", min_value=1, max_value=10, value=3, step=1, key="tab_gbm_depth")

    return params


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401

        return True
    except Exception:
        return False


def show_dataset_summary(df: pd.DataFrame, target_col: str | None) -> None:
    """Display dataset summary and basic stats."""
    st.subheader("Dataset Overview")
    st.write(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    st.dataframe(df.head(20), use_container_width=True)

    if target_col and target_col in df.columns:
        if pd.api.types.is_numeric_dtype(df[target_col]):
            st.subheader("Target Distribution (Summary)")
            stats = df[target_col].describe().to_frame(name="value")
            st.dataframe(stats, use_container_width=True)
        else:
            st.subheader("Target Distribution")
            dist = df[target_col].value_counts().rename_axis("label").reset_index(name="count")
            st.dataframe(dist, use_container_width=True)

    st.subheader("Missing Values")
    st.dataframe(df.isna().sum().reset_index().rename(columns={"index": "column", 0: "missing"}))


def show_data_profile(df: pd.DataFrame, target_col: str | None) -> None:
    """Show quick schema/profile hints for UX."""
    st.subheader("Data Profile")
    type_counts = df.dtypes.astype(str).value_counts().reset_index()
    type_counts.columns = ["dtype", "count"]
    st.dataframe(type_counts, use_container_width=True)

    st.subheader("Target Candidates")
    candidates = []
    for col in df.columns:
        if col == target_col:
            continue
        nunique = df[col].nunique(dropna=True)
        dtype = str(df[col].dtype)
        candidates.append({"column": col, "nunique": int(nunique), "dtype": dtype})
    if candidates:
        cand_df = pd.DataFrame(candidates).sort_values("nunique", ascending=True)
        st.dataframe(cand_df.head(20), use_container_width=True)


def plot_confusion_and_roc(result: dict[str, Any]) -> None:
    """Render confusion matrix and ROC if classification predictions are present."""
    metrics = result.get("metrics", {})
    if "accuracy" not in metrics and "f1_weighted" not in metrics:
        st.info("Classification metrics not found. Skipping confusion matrix/ROC.")
        return

    preview = result.get("prediction_preview", {})
    y_true = preview.get("y_true")
    y_pred = preview.get("y_pred")

    if not y_true or not y_pred:
        st.info("No prediction preview available for confusion matrix/ROC.")
        return

    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

    st.subheader("Confusion Matrix")
    fig_cm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    st.pyplot(fig_cm.figure_)

    try:
        st.subheader("ROC Curve")
        fig_roc = RocCurveDisplay.from_predictions(y_true, y_pred)
        st.pyplot(fig_roc.figure_)
    except Exception:
        st.info("ROC curve is only available for binary classification with probability scores.")


def show_run_artifacts(run_path: Path) -> None:
    """Display saved run artifacts if available."""
    report_path = run_path / "report.html"
    if report_path.exists():
        with report_path.open("rb") as f:
            st.download_button("Download report.html", f, file_name="report.html", mime="text/html")

    ai_report_path = run_path / "ai_report.json"
    if ai_report_path.exists():
        st.subheader("AI Report")
        st.json(json.loads(ai_report_path.read_text(encoding="utf-8")))

    rec_path = run_path / "recommendations.json"
    if rec_path.exists():
        st.subheader("Recommendations")
        rec_payload = json.loads(rec_path.read_text(encoding="utf-8"))
        priority = rec_payload.get("priority", [])
        if priority:
            st.write("Priority:")
            st.write(priority)
        with st.expander("All Recommendations", expanded=False):
            st.json(rec_payload)

    quality_path = run_path / "data_quality.json"
    if quality_path.exists():
        st.subheader("Data Quality")
        quality_payload = json.loads(quality_path.read_text(encoding="utf-8"))
        warnings = quality_payload.get("warnings", [])
        if warnings:
            for warn in warnings:
                level = warn.get("level", "warning")
                msg = warn.get("message", "")
                if level == "critical":
                    st.error(msg)
                elif level == "warning":
                    st.warning(msg)
                else:
                    st.info(msg)
        st.json(quality_payload)

    drift_path = run_path / "drift_report.json"
    if drift_path.exists():
        st.subheader("Drift Report")
        drift_payload = json.loads(drift_path.read_text(encoding="utf-8"))
        warnings = drift_payload.get("warnings", [])
        if warnings:
            for warn in warnings:
                level = warn.get("level", "warning")
                msg = warn.get("message", "")
                if level == "critical":
                    st.error(msg)
                elif level == "warning":
                    st.warning(msg)
                else:
                    st.info(msg)
        st.json(drift_payload)

    uncertainty_path = run_path / "uncertainty.json"
    if uncertainty_path.exists():
        st.subheader("Uncertainty")
        st.json(json.loads(uncertainty_path.read_text(encoding="utf-8")))

    best_params_path = run_path / "best_params.json"
    if best_params_path.exists():
        st.subheader("Best Params (Tuning)")
        st.json(json.loads(best_params_path.read_text(encoding="utf-8")))

    tuning_path = run_path / "tuning_results.json"
    if tuning_path.exists():
        st.subheader("Tuning Results")
        try:
            tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
            df_tune = pd.DataFrame(tuning)
            if "score" in df_tune.columns:
                df_tune = df_tune.sort_values("score", ascending=False)
            st.dataframe(df_tune, use_container_width=True)
        except Exception:
            st.json(json.loads(tuning_path.read_text(encoding="utf-8")))

    err_path = run_path / "error_analysis.json"
    if err_path.exists():
        st.subheader("Error Analysis")
        err_payload = json.loads(err_path.read_text(encoding="utf-8"))
        if err_payload.get("task_type") == "classification":
            cols = st.columns(3)
            cols[0].metric("Total", err_payload.get("total", 0))
            cols[1].metric("Errors", err_payload.get("errors", 0))
            cols[2].metric("Error Rate", f"{err_payload.get('error_rate', 0.0):.4f}")
        else:
            cols = st.columns(2)
            cols[0].metric("Residual Mean", f"{err_payload.get('residual_mean', 0.0):.4f}")
            cols[1].metric("Residual Std", f"{err_payload.get('residual_std', 0.0):.4f}")
        with st.expander("Raw Error Analysis JSON", expanded=False):
            st.json(err_payload)
        top_errors = err_payload.get("top_errors")
        if top_errors:
            try:
                st.dataframe(pd.DataFrame(top_errors), use_container_width=True)
            except Exception:
                pass

    cm_path = run_path / "confusion_matrix.png"
    if cm_path.exists():
        st.subheader("Confusion Matrix (saved)")
        st.image(str(cm_path))

    roc_path = run_path / "roc_curve.png"
    if roc_path.exists():
        st.subheader("ROC Curve (saved)")
        st.image(str(roc_path))

    scatter_path = run_path / "prediction_scatter.png"
    if scatter_path.exists():
        st.subheader("Prediction Scatter (saved)")
        st.image(str(scatter_path))

    residual_path = run_path / "residuals.png"
    if residual_path.exists():
        st.subheader("Residual Plot (saved)")
        st.image(str(residual_path))

    shap_path = run_path / "shap_summary.png"
    if shap_path.exists():
        st.subheader("SHAP Summary (saved)")
        st.image(str(shap_path))

    shap_inter_path = run_path / "shap_interaction.png"
    if shap_inter_path.exists():
        st.subheader("SHAP Interaction (saved)")
        st.image(str(shap_inter_path))

    interaction_pdp = sorted(run_path.glob("pdp_interaction_*.png"))
    if interaction_pdp:
        st.subheader("Interaction PDP (saved)")
        for path in interaction_pdp:
            st.image(str(path))

    pdp_paths = sorted(run_path.glob("pdp_*.png"))
    if pdp_paths:
        st.subheader("PDP (saved)")
        for path in pdp_paths:
            st.image(str(path))

    ice_paths = sorted(run_path.glob("ice_*.png"))
    if ice_paths:
        st.subheader("ICE (saved)")
        for path in ice_paths:
            st.image(str(path))

    pred_path = run_path / "predictions_preview.json"
    if pred_path.exists():
        st.subheader("Prediction Preview (saved)")
        pred = json.loads(pred_path.read_text(encoding="utf-8"))
        st.dataframe(pd.DataFrame(pred), use_container_width=True)

    force_plot_path = run_path / "force_plot.html"
    if force_plot_path.exists():
        st.subheader("SHAP Force Plot (HTML)")
        with force_plot_path.open("r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=400, scrolling=True)

    force_plot_paths = sorted(run_path.glob("force_plot_*.html"))
    if force_plot_paths:
        st.subheader("SHAP Force Plot Gallery")
        selected_force = st.selectbox(
            "Select force plot",
            options=[p.name for p in force_plot_paths],
            key="force_plot_select",
        )
        sel_path = run_path / selected_force
        if sel_path.exists():
            with sel_path.open("r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=320, scrolling=True)

    img_paths = sorted([p for p in run_path.glob("*.png")])
    if img_paths:
        st.subheader("Artifacts Gallery")
        cols = st.columns(3)
        for i, path in enumerate(img_paths):
            cols[i % 3].image(str(path), caption=path.name)

    leader_path = run_path / "leaderboard.json"
    if leader_path.exists():
        st.subheader("Leaderboard (saved)")
        leader = json.loads(leader_path.read_text(encoding="utf-8"))
        st.dataframe(pd.DataFrame(leader), use_container_width=True)


def quick_preset_state(action: str) -> dict[str, Any]:
    """Return preset UI defaults for quick workflows."""
    if action == "Quick Classification":
        return {
            "task_type": "classification",
            "model_type": "rf",
            "preset": "Breast Cancer (classification)",
        }
    if action == "Quick Regression":
        return {
            "task_type": "regression",
            "model_type": "gbm",
            "preset": "Diabetes (regression)",
        }
    if action == "Quick Image Demo":
        return {
            "image_dataset": "MNIST",
            "image_arch": "cnn",
        }
    if action == "Quick Text Demo":
        return {
            "text_dataset": "SST2_SAMPLE",
            "text_arch": "gru",
        }
    return {}


tabular_tab, image_tab, text_tab, audio_tab, video_tab, image_det_tab, video_det_tab, agent_tab, rag_tab, mm_tab, summary_tab, chatbot_tab = st.tabs([
    "Tabular",
    "Image Models",
    "Text Models",
    "Audio Demo",
    "Video Demo",
    "Image Detection",
    "Video Detection",
    "Agent",
    "RAG",
    "Multimodal",
    "GitHub Summary",
    "Chatbot",
])

with tabular_tab:
    train_view, test_view, compare_view = st.tabs(["Train", "Test", "Compare"])

    with train_view:
        st.subheader("Train")
        quick_defaults = quick_preset_state(quick_action)
        if quick_action in ["Quick Classification", "Quick Regression"]:
            st.info("Quick preset applied. Data source set to Preset Dataset.")
            st.session_state["train_source_default"] = "Preset Dataset"
            st.session_state["train_preset_default"] = quick_defaults.get("preset")

        with st.expander("Step 1: Data", expanded=True):
            train_df, preset_target = pick_data_source("train")

        if train_df is not None:
            if preset_target and preset_target in train_df.columns:
                default_target = train_df.columns.get_loc(preset_target)
            else:
                default_target = len(train_df.columns) - 1

            with st.expander("Step 2: Model", expanded=True):
                target_col = st.selectbox(
                    "Target column",
                    options=train_df.columns.tolist(),
                    index=default_target,
                )

                show_data_profile(train_df, target_col)
                show_dataset_summary(train_df, target_col)

                task_type_options = ["classification", "regression"]
                task_default = task_type_options.index(quick_defaults.get("task_type")) if quick_defaults.get("task_type") in task_type_options else 0
                task_type = st.selectbox("Task type", options=task_type_options, index=task_default)

                if task_type == "regression" and not pd.api.types.is_numeric_dtype(train_df[target_col]):
                    st.error("회귀 타겟은 숫자형이어야 합니다. 분류로 변경하거나 타겟 컬럼을 수정하세요.")
                if task_type == "classification" and pd.api.types.is_numeric_dtype(train_df[target_col]):
                    if train_df[target_col].nunique() > 20:
                        st.warning("클래스 수가 많아 보입니다. 회귀 문제인지 확인하세요.")

                model_options = ["auto", "dnn", "rf", "svm", "knn", "lr", "gbm", "xgboost"]
                model_default = model_options.index(quick_defaults.get("model_type")) if quick_defaults.get("model_type") in model_options else 0
                model_type = st.selectbox("Model type", options=model_options, index=model_default)
                seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1, key="tab_seed")

                st.subheader("Model Parameters")
                params: dict[str, Any] = {}
                if model_type == "auto":
                    rec_model, rec_params = recommend_model(train_df, target_col, task_type)
                    st.info(f"추천 모델: {rec_model}")
                    st.json(rec_params)
                else:
                    params = model_param_controls(model_type)

            with st.expander("Step 3: Run & Results", expanded=True):
                if st.button("학습 실행", type="primary"):
                    tmp_train_path = Path("/tmp/easy_dl_train.csv")
                    train_df.to_csv(tmp_train_path, index=False)

                    with st.spinner("학습 및 아티팩트 저장 중..."):
                        result = train_and_save(
                            data_path=tmp_train_path,
                            config_path=Path("Easy_Deep_Learning/config/model_config.yaml"),
                            target_column=target_col,
                            task_type=task_type,
                            model_type=model_type,
                            seed=int(seed),
                            model_params=params,
                        )

                    st.success(f"완료: run_id={result.run_id}")
                    cols = st.columns(max(1, len(result.metrics)))
                    for i, (k, v) in enumerate(result.metrics.items()):
                        cols[i % len(cols)].metric(label=k, value=f"{v:.4f}")
                    st.code(str(result.run_path.resolve()))
                    show_run_artifacts(result.run_path)

                st.subheader("Cross Validation")
                cv_folds = st.number_input("CV folds", min_value=3, max_value=10, value=5, step=1, key="tab_cv_folds")
                if st.button("Run Cross-Validation", type="secondary", key="tab_cv_run"):
                    from Easy_Deep_Learning.core.workflows import cross_validate_and_report, save_cv_report

                    tmp_train_path = Path("/tmp/easy_dl_train.csv")
                    train_df.to_csv(tmp_train_path, index=False)
                    with st.spinner("Cross-validation running..."):
                        cv_result = cross_validate_and_report(
                            data_path=tmp_train_path,
                            target_column=target_col,
                            task_type=task_type,
                            model_type=model_type,
                            seed=int(seed),
                            folds=int(cv_folds),
                            model_params=params,
                        )
                        cv_run_path = save_cv_report(cv_result)
                    st.subheader("CV Mean Metrics")
                    st.json(cv_result.get("mean_metrics", {}))
                    st.subheader("CV Metrics per Fold")
                    st.dataframe(pd.DataFrame(cv_result.get("metrics", [])), use_container_width=True)
                    st.code(str(cv_run_path.resolve()))

                st.subheader("AutoML Leaderboard")
                max_models = st.number_input("Max models", min_value=2, max_value=10, value=6, step=1, key="tab_automl_max")
                if st.button("Leaderboard 실행", type="secondary"):
                    tmp_train_path = Path("/tmp/easy_dl_train.csv")
                    train_df.to_csv(tmp_train_path, index=False)

                    with st.spinner("여러 모델을 학습하고 리더보드를 생성 중..."):
                        lb_result = run_leaderboard(
                            data_path=tmp_train_path,
                            config_path=Path("Easy_Deep_Learning/config/model_config.yaml"),
                            target_column=target_col,
                            task_type=task_type,
                            seed=int(seed),
                            max_models=int(max_models),
                        )

                    st.success(f"리더보드 완료: run_id={lb_result['run_id']}")
                    st.session_state["last_leaderboard"] = lb_result
                    lb_df = pd.DataFrame(lb_result["leaderboard"])
                    st.dataframe(lb_df, use_container_width=True)
                    if not lb_df.empty:
                        csv_bytes = lb_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download leaderboard.csv",
                            data=csv_bytes,
                            file_name="leaderboard.csv",
                            mime="text/csv",
                        )
                    best_run = lb_result.get("best_run")
                    if best_run:
                        st.info(f"Best run: {best_run['run_id']} ({best_run['model_type']})")

                last_lb = st.session_state.get("last_leaderboard")
                if last_lb and last_lb.get("best_run"):
                    best_run = last_lb["best_run"]
                    if st.button("Best run 재학습", type="primary", key="retrain_best"):
                        import yaml

                        best_run_id = best_run["run_id"]
                        snapshot_path = Path("runs") / best_run_id / "config_snapshot.yaml"
                        params_path = Path("runs") / best_run_id / "model_params.json"
                        if not snapshot_path.exists() or not params_path.exists():
                            st.error("Best run 정보가 충분하지 않습니다.")
                        else:
                            snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
                            params = json.loads(params_path.read_text(encoding="utf-8"))

                            tmp_train_path = Path("/tmp/easy_dl_train.csv")
                            train_df.to_csv(tmp_train_path, index=False)
                            with st.spinner("Best run 설정으로 재학습 중..."):
                                result = train_and_save(
                                    data_path=tmp_train_path,
                                    config_path=Path("Easy_Deep_Learning/config/model_config.yaml"),
                                    target_column=snapshot["input"]["target_column"],
                                    task_type=snapshot["input"]["task_type"],
                                    model_type=best_run["model_type"],
                                    seed=int(seed),
                                    model_params=params,
                                )
                            st.success(f"재학습 완료: run_id={result.run_id}")
                            st.code(str(result.run_path.resolve()))

                st.subheader("Auto Tuning")
                tune_model = st.selectbox(
                    "Tuning model type",
                    options=["rf", "gbm", "xgboost", "svm", "knn", "lr"],
                    index=0,
                    key="tab_tune_model",
                )
                max_trials = st.number_input("Max trials", min_value=3, max_value=30, value=10, step=1, key="tab_tune_trials")
                if st.button("Auto Tuning 실행", type="secondary", key="tab_tune_run"):
                    tmp_train_path = Path("/tmp/easy_dl_train.csv")
                    train_df.to_csv(tmp_train_path, index=False)

                    with st.spinner("하이퍼파라미터 튜닝 + 학습 중..."):
                        result = auto_tune_and_train(
                            data_path=tmp_train_path,
                            config_path=Path("Easy_Deep_Learning/config/model_config.yaml"),
                            target_column=target_col,
                            task_type=task_type,
                            model_type=tune_model,
                            seed=int(seed),
                            max_trials=int(max_trials),
                        )
                    st.success(f"튜닝 완료: run_id={result.run_id}")
                    st.code(str(result.run_path.resolve()))
                    show_run_artifacts(result.run_path)

    with test_view:
        st.subheader("Test Saved Model")
        runs_dir = Path("runs")
        run_ids = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()], reverse=True) if runs_dir.exists() else []

        with st.expander("Step 1: Select Run", expanded=True):
            run_options = run_ids if run_ids else [""]
            default_index = run_options.index(selected_sidebar_run) if selected_sidebar_run in run_options else 0
            selected_run = st.selectbox("Run ID 선택", options=run_options, index=default_index)
            if selected_run:
                show_run_artifacts(Path("runs") / selected_run)

        with st.expander("Step 2: Data", expanded=True):
            test_df, _ = pick_data_source("test")
            target_override = st.text_input("Target column override (선택)", value="")

        if test_df is not None:
            show_data_profile(test_df, target_override or None)
            show_dataset_summary(test_df, target_override or None)

            if st.button("테스트 실행", type="primary"):
                if not selected_run:
                    st.error("먼저 run_id를 선택하세요.")
                else:
                    tmp_test_path = Path("/tmp/easy_dl_test.csv")
                    test_df.to_csv(tmp_test_path, index=False)

                    with st.spinner("저장된 모델로 평가 중..."):
                        result = test_from_run(
                            run_id=selected_run,
                            test_data_path=tmp_test_path,
                            target_column=target_override.strip() or None,
                            save_artifacts=True,
                        )

                    st.success("평가 완료")
                    metrics = result.get("metrics", {})
                    cols = st.columns(max(1, len(metrics)))
                    for i, (k, v) in enumerate(metrics.items()):
                        cols[i % len(cols)].metric(label=k, value=f"{v:.4f}")

                    st.subheader("Prediction Preview")
                    preview = result.get("prediction_preview", {})
                    st.dataframe(pd.DataFrame(preview), use_container_width=True)

                    plot_confusion_and_roc(result)

                    st.subheader("Result JSON")
                    st.json(result)
                    show_run_artifacts(Path("runs") / selected_run)

    with compare_view:
        st.subheader("Compare Runs")
        runs_dir = Path("runs")
        run_ids = sorted([p.name for p in runs_dir.iterdir() if p.is_dir()], reverse=True) if runs_dir.exists() else []

        compare_ids = st.multiselect("Run ID 선택", options=run_ids, key="compare_runs")
        if compare_ids:
            rows = []
            for rid in compare_ids:
                metrics_path = Path("runs") / rid / "metrics.json"
                info_path = Path("runs") / rid / "model_info.json"
                if metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                else:
                    metrics = {}
                model_type = ""
                task_type = ""
                if info_path.exists():
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                    model_type = info.get("model_type", "")
                    task_type = info.get("task_type", "")
                row = {"run_id": rid, "model_type": model_type, "task_type": task_type}
                row.update(metrics)
                rows.append(row)
            df_rows = pd.DataFrame(rows)
            if not df_rows.empty:
                def _score_row(r: pd.Series) -> float:
                    if r.get("task_type") == "classification":
                        if "f1_weighted" in r:
                            return float(r.get("f1_weighted", 0.0))
                        return float(r.get("accuracy", 0.0))
                    return float(r.get("r2", -1e9))

                df_rows["score"] = df_rows.apply(_score_row, axis=1)
                best_idx = df_rows["score"].idxmax()

                def _highlight_best(row: pd.Series) -> list[str]:
                    if row.name == best_idx:
                        return ["background-color: #dcfce7"] * len(row)
                    return ["" for _ in row]

                st.dataframe(df_rows.style.apply(_highlight_best, axis=1), use_container_width=True)
            else:
                st.dataframe(df_rows, use_container_width=True)

            if st.button("Generate Compare Report", type="secondary"):
                from Easy_Deep_Learning.core.compare import generate_compare_report

                result = generate_compare_report(compare_ids)
                st.success("Comparison report generated.")
                st.code(result.get("report_dir", ""))
                report_path = Path(result.get("report_dir", "")) / "compare_report.html"
                if report_path.exists():
                    with report_path.open("rb") as f:
                        st.download_button("Download compare_report.html", f, file_name="compare_report.html", mime="text/html")

with image_tab:
    st.subheader("Image Models (CNN)")
    data_dir = st.text_input("Dataset cache dir", value="/tmp/easy_dl", key="img_cache")

    if not torch_available():
        st.info("Torch/torchvision이 설치되어 있지 않아 이미지 모델을 사용할 수 없습니다.")
    else:
        if quick_action == "Quick Image Demo":
            st.info("Quick Image Demo preset applied.")
        image_defaults = quick_preset_state(quick_action)
        with st.expander("Step 1: Dataset & Model", expanded=True):
            dataset_options = ["MNIST", "FashionMNIST", "CIFAR10", "SVHN", "EMNIST"]
            dataset_default = dataset_options.index(image_defaults.get("image_dataset")) if image_defaults.get("image_dataset") in dataset_options else 0
            dataset = st.selectbox(
                "Dataset",
                options=dataset_options,
                index=dataset_default,
                key="img_dataset",
            )
            arch_options = ["cnn", "resnet18"]
            arch_default = arch_options.index(image_defaults.get("image_arch")) if image_defaults.get("image_arch") in arch_options else 0
            model_arch = st.selectbox("Model architecture", options=arch_options, index=arch_default, key="img_arch")

        with st.expander("Step 2: Training Params", expanded=True):
            epochs = st.number_input("Epochs", min_value=1, max_value=50, value=5, step=1, key="img_epochs")
            lr = st.number_input("Learning rate", min_value=1e-4, max_value=1e-1, value=1e-3, format="%.5f", key="img_lr")
            batch_size = st.number_input("Batch size", min_value=16, max_value=512, value=64, step=16, key="img_batch")
            seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1, key="img_seed")

        with st.expander("Step 3: Train & Results", expanded=True):
            if st.button("Train CNN", type="primary"):
                from Easy_Deep_Learning.core.torch_workflows import train_cnn_image

                with st.spinner("Training CNN..."):
                    result = train_cnn_image(
                        dataset_name=dataset,
                        epochs=int(epochs),
                        lr=float(lr),
                        batch_size=int(batch_size),
                        seed=int(seed),
                        data_dir=Path(data_dir),
                        model_arch=model_arch,
                    )
                st.success(f"완료: run_id={result.run_id}")
                st.metric("accuracy", f"{result.metrics['accuracy']:.4f}")
                st.code(str(result.run_path.resolve()))

            st.subheader("Dataset Preview")
            if st.button("Load Preview", key="img_preview"):
                import torchvision
                from torchvision import transforms

                if dataset == "MNIST":
                    ds_cls = torchvision.datasets.MNIST
                elif dataset == "FashionMNIST":
                    ds_cls = torchvision.datasets.FashionMNIST
                elif dataset == "SVHN":
                    ds_cls = torchvision.datasets.SVHN
                elif dataset == "EMNIST":
                    ds_cls = torchvision.datasets.EMNIST
                else:
                    ds_cls = torchvision.datasets.CIFAR10

                if dataset == "SVHN":
                    preview_ds = ds_cls(root=str(data_dir), split="train", download=True, transform=transforms.ToTensor())
                elif dataset == "EMNIST":
                    preview_ds = ds_cls(root=str(data_dir), split="balanced", train=True, download=True, transform=transforms.ToTensor())
                else:
                    preview_ds = ds_cls(root=str(data_dir), train=True, download=True, transform=transforms.ToTensor())
                images = []
                labels = []
                for i in range(min(12, len(preview_ds))):
                    img, lbl = preview_ds[i]
                    img_np = img.detach().cpu().numpy()
                    if img_np.shape[0] in (1, 3):
                        img_np = img_np.transpose(1, 2, 0)
                    images.append(img_np)
                    labels.append(str(lbl))

                st.image(images, caption=labels, width=120)

            st.subheader("Test Saved CNN")
            cnn_runs = [rid for rid in run_ids if rid.endswith("_cnn")]
            selected_cnn = st.selectbox("CNN run_id", options=cnn_runs, key="cnn_run")
            if st.button("Test CNN", type="secondary") and selected_cnn:
                from Easy_Deep_Learning.core.torch_workflows import test_cnn_image

                with st.spinner("Testing CNN..."):
                    result = test_cnn_image(selected_cnn)
                st.metric("accuracy", f"{result['accuracy']:.4f}")

            st.subheader("Custom Image Prediction")
            uploaded_imgs = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="img_upload")
            pred_run = st.selectbox("Run ID for prediction", options=cnn_runs, key="img_pred_run")
            show_cam = st.checkbox("Show Grad-CAM", value=False, key="img_cam")
            if uploaded_imgs and pred_run:
                if show_cam:
                    from Easy_Deep_Learning.core.torch_workflows import predict_cnn_images_with_cam
                    import matplotlib.cm as cm

                    img_bytes = [u.getvalue() for u in uploaded_imgs]
                    with st.spinner("Running predictions + Grad-CAM..."):
                        preds = predict_cnn_images_with_cam(pred_run, img_bytes)
                    for i, pred in enumerate(preds):
                        heat = cm.magma(pred["cam"])
                        st.image(uploaded_imgs[i], caption=f"{pred['label']} ({pred['prob']:.3f})")
                        st.image(heat, caption="Grad-CAM", width=180)
                else:
                    from Easy_Deep_Learning.core.torch_workflows import predict_cnn_images

                    img_bytes = [u.getvalue() for u in uploaded_imgs]
                    with st.spinner("Running predictions..."):
                        preds = predict_cnn_images(pred_run, img_bytes)
                    for i, pred in enumerate(preds):
                        st.image(uploaded_imgs[i], caption=f"{pred['label']} ({pred['prob']:.3f})")

with text_tab:
    st.subheader("Text Models (RNN)")
    data_dir = st.text_input("Dataset cache dir", value="/tmp/easy_dl", key="txt_cache")

    text_defaults = quick_preset_state(quick_action)
    if quick_action == "Quick Text Demo":
        st.info("Quick Text Demo preset applied.")
    text_options = ["AG_NEWS_SAMPLE", "SST2_SAMPLE", "TREC_SAMPLE", "Upload CSV"]
    text_default = text_options.index(text_defaults.get("text_dataset")) if text_defaults.get("text_dataset") in text_options else 0

    with st.expander("Step 1: Data", expanded=True):
        dataset_choice = st.selectbox(
            "Dataset",
            options=text_options,
            index=text_default,
            key="txt_dataset",
        )
        if dataset_choice == "Upload CSV":
            uploaded_text = st.file_uploader("Upload text CSV", type=["csv"], key="text_upload")
            text_df = pd.read_csv(uploaded_text) if uploaded_text else None
        elif dataset_choice == "SST2_SAMPLE":
            text_df = pd.read_csv(Path("Easy_Deep_Learning/data/text_sample_sst2.csv"))
        elif dataset_choice == "TREC_SAMPLE":
            text_df = pd.read_csv(Path("Easy_Deep_Learning/data/text_sample_trec.csv"))
        else:
            text_df = pd.read_csv(Path("Easy_Deep_Learning/data/text_sample.csv"))

        text_col = st.text_input("Text column", value="text", key="txt_col")
        label_col = st.text_input("Label column", value="label", key="lbl_col")
        max_vocab = st.number_input("Max vocab", min_value=100, max_value=50000, value=5000, step=100, key="txt_vocab")
        max_len = st.number_input("Max length", min_value=10, max_value=400, value=100, step=10, key="txt_len")

        st.subheader("Text Preview")
        if text_df is not None:
            preview_cols = [c for c in [text_col, label_col] if c in text_df.columns]
            st.dataframe(text_df[preview_cols].head(20), use_container_width=True)
        else:
            st.info("텍스트 데이터가 없습니다.")

    with st.expander("Step 2: Preprocessing & Model", expanded=True):
        st.subheader("Preprocessing")
        stopwords = st.checkbox("Remove stopwords", value=False, key="txt_stop")
        ngram = st.number_input("n-gram (1-3)", min_value=1, max_value=3, value=1, step=1, key="txt_ngram")
        bpe = st.checkbox("Use BPE", value=False, key="txt_bpe")
        bpe_vocab_size = st.number_input("BPE vocab size", min_value=50, max_value=2000, value=200, step=50, key="txt_bpe_vocab")
        text_arch_options = ["gru", "lstm", "textcnn", "transformer"]
        text_arch_default = text_arch_options.index(text_defaults.get("text_arch")) if text_defaults.get("text_arch") in text_arch_options else 0
        model_arch = st.selectbox("Model architecture", options=text_arch_options, index=text_arch_default, key="txt_arch")

        epochs = st.number_input("Epochs", min_value=1, max_value=20, value=3, step=1, key="txt_epochs")
        lr = st.number_input("Learning rate", min_value=1e-4, max_value=1e-1, value=1e-3, format="%.5f", key="txt_lr")
        batch_size = st.number_input("Batch size", min_value=16, max_value=512, value=64, step=16, key="txt_batch")
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1, key="txt_seed")

    with st.expander("Step 3: Train & Test", expanded=True):
        if st.button("Train RNN", type="primary"):
            from Easy_Deep_Learning.core.torch_workflows import train_rnn_text

            if text_df is None:
                st.error("텍스트 CSV를 선택하세요.")
            else:
                tmp_text_path = Path("/tmp/easy_dl_text.csv")
                text_df.to_csv(tmp_text_path, index=False)

                with st.spinner("Training RNN..."):
                    result = train_rnn_text(
                        dataset_name=(
                            dataset_choice if dataset_choice != "Upload CSV" else "CUSTOM"
                        ),
                        epochs=int(epochs),
                        lr=float(lr),
                        batch_size=int(batch_size),
                        seed=int(seed),
                        data_dir=Path(data_dir),
                        data_path=tmp_text_path,
                        text_column=text_col,
                        label_column=label_col,
                        max_vocab=int(max_vocab),
                        max_len=int(max_len),
                        stopwords=bool(stopwords),
                        ngram=int(ngram),
                        bpe=bool(bpe),
                        bpe_vocab_size=int(bpe_vocab_size),
                        model_arch=model_arch,
                    )
                st.success(f"완료: run_id={result.run_id}")
                st.metric("test_accuracy", f"{result.metrics['test_accuracy']:.4f}")
                st.code(str(result.run_path.resolve()))

        st.subheader("Test Saved RNN")
        rnn_runs = [rid for rid in run_ids if rid.endswith("_rnn")]
        selected_rnn = st.selectbox("RNN run_id", options=rnn_runs, key="rnn_run")
        if st.button("Test RNN", type="secondary") and selected_rnn:
            from Easy_Deep_Learning.core.torch_workflows import test_rnn_text

            with st.spinner("Testing RNN..."):
                result = test_rnn_text(selected_rnn, data_path=None)
            st.metric("test_accuracy", f"{result['test_accuracy']:.4f}")

with audio_tab:
    st.subheader("Audio Demo (WAV)")
    st.caption("Built-in sine wave or WAV upload. Feature extraction + ASR + demo classifier.")
    from Easy_Deep_Learning.core.media_demo import generate_sine_wave, load_wav_bytes, write_wav_bytes, audio_features, build_audio_dataset
    from Easy_Deep_Learning.core.asr import compute_wer, compute_cer, transcribe_openai
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier

    samples_dir = Path("Easy_Deep_Learning/data/audio_samples")
    built_in_files = []
    if samples_dir.exists():
        built_in_files = sorted([p.name for p in samples_dir.glob("*.wav")])
    built_in = st.selectbox(
        "Built-in sample",
        options=["Sine 440Hz", "Sine 880Hz"] + built_in_files,
        key="audio_builtin",
    )
    uploaded = st.file_uploader("Upload WAV", type=["wav"], key="audio_upload")
    recorded = None
    webrtc_signal = None
    webrtc_wav = st.session_state.get("webrtc_wav")
    if hasattr(st, "audio_input"):
        recorded = st.audio_input("Record audio (optional)", key="audio_record")
    else:
        st.info("이 Streamlit 버전은 audio_input을 지원하지 않습니다. 웹 녹음 컴포넌트를 사용합니다.")
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode
            import av

            class AudioProcessor:
                def __init__(self) -> None:
                    self.frames: list[np.ndarray] = []
                    self.is_recording = False

                def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
                    pcm = frame.to_ndarray()
                    self.frames.append(pcm)
                    self.is_recording = True
                    return frame

            ctx = webrtc_streamer(
                key="audio_webrtc",
                mode=WebRtcMode.SENDONLY,
                audio_receiver_size=256,
                media_stream_constraints={"audio": True, "video": False},
                async_processing=True,
                audio_processor_factory=AudioProcessor,
            )
            if ctx.state.playing:
                st.success("녹음 중...")
            else:
                st.info("녹음을 시작하려면 Start를 누르세요.")
            if ctx and ctx.audio_processor and ctx.audio_processor.frames:
                pcm = np.concatenate(ctx.audio_processor.frames, axis=1).flatten()
                pcm = pcm.astype(np.float32)
                pcm /= np.max(np.abs(pcm)) + 1e-9
                webrtc_signal = pcm

            col_rec_1, col_rec_2 = st.columns(2)
            with col_rec_1:
                if st.button("Save Recording", key="save_webrtc"):
                    if webrtc_signal is not None:
                        webrtc_wav = write_wav_bytes(webrtc_signal, sr=16000)
                        st.session_state["webrtc_wav"] = webrtc_wav
                        st.success("녹음이 저장되었습니다.")
                    else:
                        st.warning("저장할 녹음이 없습니다.")
            with col_rec_2:
                if st.button("Clear Recording", key="clear_webrtc"):
                    st.session_state.pop("webrtc_wav", None)
                    webrtc_wav = None
        except Exception as exc:
            st.info(f"웹 녹음 컴포넌트를 사용할 수 없습니다: {exc}")
    sr = 16000
    if uploaded:
        signal, sr = load_wav_bytes(uploaded.read())
    elif recorded:
        signal, sr = load_wav_bytes(recorded.getvalue())
    elif webrtc_signal is not None:
        signal = webrtc_signal
        sr = 16000
    elif built_in in built_in_files:
        data = (samples_dir / built_in).read_bytes()
        signal, sr = load_wav_bytes(data)
    else:
        freq = 440.0 if built_in == "Sine 440Hz" else 880.0
        signal = generate_sine_wave(freq=freq, duration=1.0, sr=sr)

    st.subheader("Waveform")
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(signal[: min(len(signal), sr)])
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)
    st.subheader("Audio Features")
    feats = audio_features(signal, sr)
    st.json(feats)

    if webrtc_wav is not None:
        st.subheader("Recorded Audio")
        st.audio(webrtc_wav, format="audio/wav")
        st.download_button("Download recording", data=webrtc_wav, file_name="recording.wav", mime="audio/wav")

    st.subheader("ASR (Speech → Text)")
    ref_text = st.text_input("Reference text (optional)", value="", key="asr_ref")
    if st.button("Transcribe Audio", type="secondary"):
        try:
            audio_bytes = uploaded.read() if uploaded else (recorded.getvalue() if recorded else (webrtc_wav if webrtc_wav is not None else None))
            if audio_bytes is None:
                st.error("WAV 파일을 업로드하거나 녹음하세요.")
            else:
                text = transcribe_openai(audio_bytes)
                st.session_state["asr_text"] = text
                st.success("Transcription completed.")
        except Exception as exc:
            st.error(f"Transcription failed: {exc}")

    if "asr_text" in st.session_state:
        st.subheader("Transcription")
        st.write(st.session_state["asr_text"])
        if ref_text.strip():
            wer = compute_wer(ref_text, st.session_state["asr_text"])
            cer = compute_cer(ref_text, st.session_state["asr_text"])
            st.metric("WER", f"{wer:.4f}")
            st.metric("CER", f"{cer:.4f}")

    st.subheader("Audio Classification Demo")
    if st.button("Train Audio Demo Classifier", type="primary"):
        X, y = build_audio_dataset()
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X, y)
        st.session_state["audio_clf"] = clf
        st.success("Audio demo model trained.")

    if "audio_clf" in st.session_state:
        X_infer = np.array([[feats["rms"], feats["zcr"], feats["spectral_centroid"]]], dtype=np.float32)
        pred = st.session_state["audio_clf"].predict(X_infer)[0]
        label = "Low freq" if int(pred) == 0 else "High freq"
        st.metric("Audio Demo Prediction", label)

with video_tab:
    st.subheader("Video Demo (Frame Sequence)")
    st.caption("Built-in synthetic frames or multiple image upload as frames. Feature extraction + demo classifier.")
    from Easy_Deep_Learning.core.media_demo import generate_synthetic_video, video_features, build_video_dataset
    from PIL import Image
    from sklearn.ensemble import RandomForestClassifier

    use_builtin = st.checkbox("Use built-in synthetic video", value=True, key="video_builtin")
    frames = []
    if use_builtin:
        frames = generate_synthetic_video()
    else:
        uploaded_imgs = st.file_uploader("Upload frames (multiple images)", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="video_upload")
        if uploaded_imgs:
            for f in uploaded_imgs:
                img = Image.open(f).convert("RGB")
                frames.append(np.array(img))

    if frames:
        st.subheader("Frame Preview")
        st.image(frames[:6], caption=[f"frame {i}" for i in range(min(6, len(frames)))], width=120)
        st.subheader("Video Features")
        v_feats = video_features(frames)
        st.json(v_feats)
        st.subheader("Video Classification Demo")
        if st.button("Train Video Demo Classifier", type="primary"):
            X, y = build_video_dataset()
            clf = RandomForestClassifier(n_estimators=200, random_state=42)
            clf.fit(X, y)
            st.session_state["video_clf"] = clf
            st.success("Video demo model trained.")

        if "video_clf" in st.session_state:
            X_infer = np.array([[v_feats["mean_intensity"], v_feats["motion_energy"], v_feats["num_frames"]]], dtype=np.float32)
            pred = st.session_state["video_clf"].predict(X_infer)[0]
            label = "Low motion" if int(pred) == 0 else "High motion"
            st.metric("Video Demo Prediction", label)
    else:
        st.info("프레임을 업로드하세요.")

with image_det_tab:
    st.subheader("Image Detection")
    st.caption("대표 데이터셋 샘플 또는 이미지 업로드로 객체 탐지 데모.")
    from Easy_Deep_Learning.core.detection import detect_image_pil
    from PIL import Image

    det_data_dir = st.text_input("Detection dataset cache dir", value="/tmp/easy_dl", key="det_cache")
    det_source = st.selectbox("Data source", options=["Upload Image", "VOC (download)", "COCO (download)"], key="det_source")
    image = None
    if det_source == "Upload Image":
        up = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], key="det_upload")
        if up:
            image = Image.open(up).convert("RGB")
    elif det_source == "VOC (download)":
        try:
            import torchvision
            ds = torchvision.datasets.VOCDetection(root=det_data_dir, year="2007", image_set="val", download=True)
            image, _ = ds[0]
        except Exception as exc:
            st.error(f"VOC load failed: {exc}")
    else:
        try:
            import torchvision
            ds = torchvision.datasets.CocoDetection(root=det_data_dir, annFile=str(Path(det_data_dir)/'annotations/instances_val2017.json'))
            image, _ = ds[0]
        except Exception as exc:
            st.error(f"COCO load failed: {exc}")

    model_choice = st.selectbox("Model", options=["YOLOv8n (ultralytics)", "Faster R-CNN (torchvision)"], key="det_model")
    conf = st.slider("Confidence", min_value=0.05, max_value=0.9, value=0.25, step=0.05, key="det_conf")
    yolo_weights = st.text_input("YOLO weights (optional)", value="yolov8n.pt", key="det_weights")

    if image is not None:
        st.image(image, caption="Input", use_container_width=True)
        if st.button("Run Detection", type="primary", key="det_run"):
            try:
                if model_choice.startswith("YOLO"):
                    out_img, dets = detect_image_pil(image, model_type="yolo", conf=conf, model_name=yolo_weights)
                else:
                    out_img, dets = detect_image_pil(image, model_type="fasterrcnn", conf=conf)
                st.subheader("Detections")
                st.image(out_img, use_container_width=True)
                st.json(dets)
            except Exception as exc:
                st.error(f"Detection failed: {exc}")
    else:
        st.info("이미지를 준비하세요.")

with video_det_tab:
    st.subheader("Video Detection")
    st.caption("MP4 업로드 후 프레임 단위 객체 탐지 데모.")
    from Easy_Deep_Learning.core.detection import detect_video_bytes
    vid = st.file_uploader("Upload video (mp4)", type=["mp4", "mov", "avi"], key="det_video")
    model_choice = st.selectbox("Model", options=["YOLOv8n (ultralytics)", "Faster R-CNN (torchvision)"], key="det_video_model")
    conf = st.slider("Confidence", min_value=0.05, max_value=0.9, value=0.25, step=0.05, key="det_video_conf")
    frame_stride = st.number_input("Frame stride", min_value=1, max_value=30, value=10, step=1, key="det_stride")
    max_frames = st.number_input("Max frames", min_value=1, max_value=60, value=20, step=1, key="det_max_frames")
    yolo_weights = st.text_input("YOLO weights (optional)", value="yolov8n.pt", key="det_video_weights")

    if vid:
        if st.button("Run Video Detection", type="primary", key="det_video_run"):
            try:
                frames, dets = detect_video_bytes(
                    video_bytes=vid.read(),
                    model_type="yolo" if model_choice.startswith("YOLO") else "fasterrcnn",
                    conf=conf,
                    model_name=yolo_weights,
                    frame_stride=int(frame_stride),
                    max_frames=int(max_frames),
                )
                st.subheader("Detected Frames")
                if frames:
                    st.image(frames, width=180)
                st.subheader("Detections (sample)")
                st.json(dets[:3])
            except Exception as exc:
                st.error(f"Video detection failed: {exc}")
    else:
        st.info("비디오를 업로드하세요.")

with agent_tab:
    st.subheader("Tool-Using Agent")
    agent_df, agent_target = pick_data_source("agent")

    if agent_df is not None:
        if agent_target and agent_target in agent_df.columns:
            agent_default_target = agent_df.columns.get_loc(agent_target)
        else:
            agent_default_target = len(agent_df.columns) - 1

        agent_target_col = st.selectbox(
            "Target column",
            options=agent_df.columns.tolist(),
            index=agent_default_target,
        )
        agent_task_type = st.selectbox("Task type", options=["classification", "regression"], index=0, key="agent_task")

        show_dataset_summary(agent_df, agent_target_col)

        if st.button("Run Agent", type="primary"):
            from Easy_Deep_Learning.agents.tool_agent import AgentInput, ToolUsingAgent, make_default_tools

            tmp_agent_path = Path("/tmp/easy_dl_agent.csv")
            agent_df.to_csv(tmp_agent_path, index=False)

            with st.spinner("Running tool-using agent..."):
                agent = ToolUsingAgent()
                result = agent.run(
                    AgentInput(
                        dataset_path=tmp_agent_path,
                        target_column=agent_target_col,
                        task_type=agent_task_type,
                    ),
                    tools=make_default_tools(),
                )

            st.subheader("Tool Calls")
            st.json([call.__dict__ for call in result.tool_calls])

            st.subheader("Tool Results")
            st.json([res.__dict__ for res in result.tool_results])

            st.subheader("Summary")
            st.write(result.final_summary)

with rag_tab:
    st.subheader("RAG + Auto Evaluation")
    docs_input = st.text_area("Documents (one per line)", height=180, key="rag_docs")
    query_input = st.text_input("Query", key="rag_query")
    top_k = st.number_input("Top-K", min_value=1, max_value=10, value=3, step=1, key="rag_topk")
    chunk_size = st.number_input("Chunk size", min_value=100, max_value=2000, value=400, step=50, key="rag_chunk")
    overlap = st.number_input("Overlap", min_value=0, max_value=500, value=80, step=10, key="rag_overlap")

    if st.button("Run RAG", type="primary"):
        from Easy_Deep_Learning.core.rag import run_rag

        docs = [line.strip() for line in docs_input.splitlines() if line.strip()]
        if not docs:
            st.error("문서가 비어 있습니다.")
        elif not query_input.strip():
            st.error("질문을 입력하세요.")
        else:
            result = run_rag(
                query=query_input,
                docs=docs,
                top_k=int(top_k),
                chunk_size=int(chunk_size),
                overlap=int(overlap),
            )
            st.subheader("Answer")
            st.write(result.answer)
            st.subheader("Contexts")
            for i, ctx in enumerate(result.contexts):
                st.write(f"[{i+1}] score={result.scores[i]:.3f}")
                st.write(ctx)
            st.subheader("Auto Evaluation")
            st.json(result.eval)

with mm_tab:
    st.subheader("Multimodal Search (Lite)")
    st.caption("이미지/텍스트를 함께 업로드하여 간단한 유사도 검색 데모를 수행합니다.")

    mm_texts = st.text_area("Texts (one per line)", height=120, key="mm_texts")
    mm_images = st.file_uploader(
        "Images (same count as texts)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="mm_images",
    )
    query_text = st.text_input("Text query", key="mm_query_text")
    query_image = st.file_uploader("Image query", type=["png", "jpg", "jpeg"], key="mm_query_image")
    top_k = st.number_input("Top-K", min_value=1, max_value=10, value=3, step=1, key="mm_topk")

    if st.button("Build Index", type="primary"):
        if not mm_texts.strip():
            st.error("텍스트를 입력하세요.")
        elif not mm_images:
            st.error("이미지를 업로드하세요.")
        else:
            from Easy_Deep_Learning.core.multimodal import MMItem, build_index
            from PIL import Image

            texts = [line.strip() for line in mm_texts.splitlines() if line.strip()]
            if len(texts) != len(mm_images):
                st.error("텍스트 라인 수와 이미지 수가 같아야 합니다.")
            else:
                items = []
                for i, (t, img_file) in enumerate(zip(texts, mm_images)):
                    img = Image.open(img_file).convert("RGB")
                    items.append(MMItem(id=f"item_{i}", text=t, image=img))

                st.session_state["mm_index"] = build_index(items)
                st.success("Index built.")

    if "mm_index" in st.session_state:
        index = st.session_state["mm_index"]
        from Easy_Deep_Learning.core.multimodal import search_by_text, search_by_image
        from PIL import Image

        col1, col2 = st.columns(2)
        with col1:
            if query_text.strip():
                st.subheader("Text → Image/Text Results")
                results = search_by_text(index, query_text, top_k=int(top_k))
                st.json(results)
        with col2:
            if query_image is not None:
                st.subheader("Image → Image/Text Results")
                img = Image.open(query_image).convert("RGB")
                results = search_by_image(index, img, top_k=int(top_k))
                st.json(results)

with summary_tab:
    st.subheader("GitHub README Summary")
    st.caption("리포트형 요약 탭입니다. GitHub 링크 또는 README 텍스트를 분석해 구조/기능/사용법을 정리합니다.")

    github_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/owner/repo", key="chatbot_url")
    readme_text = st.text_area("Or paste README content", height=180, key="chatbot_readme")
    if st.button("Summarize README", type="primary"):
        from Easy_Deep_Learning.core.chatbot import summarize_github_readme, summarize_readme_text

        try:
            if github_url.strip():
                result = summarize_github_readme(github_url.strip())
            elif readme_text.strip():
                result = summarize_readme_text(readme_text.strip(), source="manual")
            else:
                st.error("GitHub URL 또는 README 텍스트를 입력하세요.")
                result = None
        except Exception as exc:
            st.error(f"요약 실패: {exc}")
            result = None

        if result:
            st.subheader(result.title)
            st.write(result.summary)

            if result.features:
                st.subheader("Features")
                st.write(result.features)
            if result.setup:
                st.subheader("Setup")
                st.write(result.setup)
            if result.usage:
                st.subheader("Usage")
                st.write(result.usage)
            if result.commands:
                st.subheader("Commands")
                for block in result.commands:
                    st.code(block)
            if result.notes:
                st.subheader("Notes")
                st.write(result.notes)

    if st.button("Analyze Repo", type="secondary"):
        from Easy_Deep_Learning.core.chatbot import summarize_github_repo

        if not github_url.strip():
            st.error("GitHub URL을 입력하세요.")
        else:
            try:
                info = summarize_github_repo(github_url.strip())
                st.subheader("Repository Overview")
                st.write(info.get("summary", ""))
                st.subheader("Key Files")
                st.write(info.get("repo_info", {}).get("key_files", []))
                st.subheader("Tech Stack")
                st.write(info.get("repo_info", {}).get("tech_stack", []))
                st.subheader("Suggested Commands")
                st.write(info.get("repo_info", {}).get("commands", []))
            except Exception as exc:
                st.error(f"분석 실패: {exc}")

with chatbot_tab:
    st.subheader("Chatbot")
    st.caption("질문형 챗봇입니다. GitHub 링크를 포함한 질문에 답변합니다.")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("메시지를 입력하세요")
    if user_input:
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        from Easy_Deep_Learning.core.chatbot import chat_response

        with st.chat_message("assistant"):
            reply = chat_response(user_input)
            st.write(reply)
        st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
