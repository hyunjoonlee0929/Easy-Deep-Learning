"""General adapter for standard tabular ML workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from OmniInsight.agents.interpretation_agent import FeatureInterpretationAgent
from OmniInsight.agents.report_agent import ExecutiveReportAgent
from OmniInsight.agents.summary_agent import DataSummaryAgent
from OmniInsight.core.data_validator import DataValidator
from OmniInsight.core.model_engine import ModelResult
from OmniInsight.core.preprocessing import AutoPreprocessor
from OmniInsight.core.trainer import Trainer, TrainingConfig
from OmniInsight.interpretation.shap_engine import ShapEngine

from .base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


@dataclass
class GeneralAdapterConfig:
    """Runtime configuration for GeneralAdapter."""

    target_column: str
    task_type: str
    model_type: str = "xgboost"
    top_k_features: int = 10
    dnn_hidden_layers: list[int] | None = None
    dnn_dropout: float = 0.2
    dnn_learning_rate: float = 1e-3
    dnn_max_epochs: int = 200
    dnn_patience: int = 20
    dnn_batch_size: int = 32


class GeneralAdapter(BaseAdapter):
    """Standard tabular adapter implementing end-to-end OmniInsight flow."""

    def __init__(self) -> None:
        self.validator = DataValidator()
        self.preprocessor = AutoPreprocessor()
        self.trainer = Trainer()
        self.shap_engine = ShapEngine()
        self.summary_agent = DataSummaryAgent()
        self.interpretation_agent = FeatureInterpretationAgent()
        self.report_agent = ExecutiveReportAgent()

    def preprocess(self, df: pd.DataFrame, target_column: str) -> dict[str, Any]:
        """Run validation and preprocessing for a general tabular dataset."""
        logger.info("Validating dataset with %d rows and %d columns", len(df), len(df.columns))
        validation = self.validator.validate(df=df, target_column=target_column)
        processed = self.preprocessor.fit_transform(df=df, target_column=target_column)
        return {
            "validation": validation,
            "processed": processed,
            "raw_df": df,
            "target_column": target_column,
        }

    def train(self, processed: dict[str, Any]) -> ModelResult:
        """Train configured model from processed payload."""
        pdata = processed["processed"]
        cfg: GeneralAdapterConfig = processed["config"]

        train_cfg = TrainingConfig(
            task_type=cfg.task_type,
            model_type=cfg.model_type,
            dnn_hidden_layers=cfg.dnn_hidden_layers or [128, 64, 32],
            dnn_dropout=cfg.dnn_dropout,
            dnn_learning_rate=cfg.dnn_learning_rate,
            dnn_max_epochs=cfg.dnn_max_epochs,
            dnn_patience=cfg.dnn_patience,
            dnn_batch_size=cfg.dnn_batch_size,
        )

        logger.info("Training model_type=%s for task_type=%s", cfg.model_type, cfg.task_type)
        return self.trainer.train(
            X_train=pdata.X_train,
            y_train=pdata.y_train,
            X_test=pdata.X_test,
            y_test=pdata.y_test,
            cfg=train_cfg,
        )

    def interpret(self, processed: dict[str, Any], model_result: ModelResult) -> dict[str, Any]:
        """Compute SHAP-based feature importance and convert to agent output."""
        pdata = processed["processed"]
        cfg: GeneralAdapterConfig = processed["config"]

        logger.info("Running SHAP interpretation for %s", cfg.model_type)
        shap_payload = self.shap_engine.explain(
            model=model_result.model,
            X_reference=pdata.X_train,
            X_target=pdata.X_test,
            feature_names=pdata.feature_names,
            task_type=cfg.task_type,
            model_type=cfg.model_type,
            top_k=cfg.top_k_features,
        )
        return self.interpretation_agent.run(shap_payload)

    def generate_report(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Generate final executive JSON report from modular agent outputs."""
        return self.report_agent.run(payload)

    def run(self, df: pd.DataFrame, config: GeneralAdapterConfig) -> dict[str, Any]:
        """Execute full tabular workflow and return a structured report."""
        pre = self.preprocess(df=df, target_column=config.target_column)
        pre["config"] = config

        model_result = self.train(pre)
        summary = self.summary_agent.run(pre["validation"].to_dict())
        interpretation = self.interpret(pre, model_result)

        payload = {
            "summary": summary,
            "model": {
                "model_name": model_result.model_name,
                "task_type": model_result.task_type,
                "metrics": model_result.metrics,
                "label_classes": model_result.label_classes,
            },
            "interpretation": interpretation,
        }
        report = self.generate_report(payload)
        logger.info("Pipeline completed")
        return report
