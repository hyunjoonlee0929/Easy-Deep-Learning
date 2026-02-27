"""Streamlit dashboard for OmniInsight."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from OmniInsight.adapters.general_adapter import GeneralAdapter, GeneralAdapterConfig
from OmniInsight.core.logging_utils import configure_logging

configure_logging("INFO")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="OmniInsight", layout="wide")
st.title("OmniInsight Dashboard")
st.caption("Upload a CSV and run automated preprocessing, model training, SHAP interpretation, and reporting.")

with st.sidebar:
    st.header("Run Settings")
    task_type = st.selectbox("Task type", options=["classification", "regression"], index=0)
    model_type = st.selectbox("Model type", options=["xgboost", "dnn"], index=0)
    top_k_features = st.slider("Top-k SHAP features", min_value=3, max_value=30, value=10)
    hidden_layers_text = st.text_input("DNN hidden layers", value="128,64,32")
    dnn_dropout = st.slider("DNN dropout", min_value=0.0, max_value=0.8, value=0.2, step=0.05)
    dnn_learning_rate = st.number_input("DNN learning rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")
    dnn_max_epochs = st.number_input("DNN max epochs", min_value=10, max_value=1000, value=200, step=10)
    dnn_patience = st.number_input("DNN early stopping patience", min_value=3, max_value=200, value=20, step=1)
    dnn_batch_size = st.number_input("DNN batch size", min_value=4, max_value=1024, value=32, step=4)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
use_example = st.checkbox("Use built-in example dataset", value=uploaded is None)

if uploaded is None and not use_example:
    st.info("Upload a CSV file or enable the built-in example dataset.")
    st.stop()

if use_example and uploaded is None:
    data_path = Path("OmniInsight/data/example_dataset.csv")
    df = pd.read_csv(data_path)
    st.caption(f"Loaded example dataset: `{data_path}`")
else:
    df = pd.read_csv(uploaded)

st.subheader("Data Preview")
st.dataframe(df.head(20), use_container_width=True)

if len(df.columns) < 2:
    st.error("Dataset must contain at least one feature column and one target column.")
    st.stop()

target_column = st.selectbox("Target column", options=df.columns.tolist(), index=len(df.columns) - 1)

if st.button("Run Analysis", type="primary"):
    try:
        hidden_layers = [int(x.strip()) for x in hidden_layers_text.split(",") if x.strip()]
        if not hidden_layers:
            raise ValueError("At least one hidden layer size is required.")

        cfg = GeneralAdapterConfig(
            target_column=target_column,
            task_type=task_type,
            model_type=model_type,
            top_k_features=top_k_features,
            dnn_hidden_layers=hidden_layers,
            dnn_dropout=float(dnn_dropout),
            dnn_learning_rate=float(dnn_learning_rate),
            dnn_max_epochs=int(dnn_max_epochs),
            dnn_patience=int(dnn_patience),
            dnn_batch_size=int(dnn_batch_size),
        )

        adapter = GeneralAdapter()
        with st.spinner("Running full OmniInsight pipeline..."):
            logger.info("Starting dashboard analysis run")
            report = adapter.run(df=df, config=cfg)

        st.success("Analysis completed")

        report_body = report.get("report", {})
        model_payload = report_body.get("model", {})
        metrics = model_payload.get("metrics", {})
        top_features = report_body.get("top_features", [])

        st.subheader("Model Metrics")
        st.json(metrics)

        st.subheader("Top Features")
        st.write(top_features)

        st.subheader("Executive Summary")
        st.write(report.get("executive_summary", "No summary available."))

        st.subheader("Full Report JSON")
        st.json(report)
    except Exception as exc:
        logger.exception("Dashboard run failed")
        st.error(f"Analysis failed: {exc}")
