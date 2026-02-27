"""Entrypoint for running the full OmniInsight pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from OmniInsight.adapters.bio_adapter import BioAdapter, BioAdapterConfig
from OmniInsight.adapters.general_adapter import GeneralAdapter, GeneralAdapterConfig
from OmniInsight.core.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run OmniInsight end-to-end pipeline")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("OmniInsight/data/example_dataset.csv"),
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--protein-data",
        type=Path,
        default=None,
        help="Optional protein CSV for bio adapter mode",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("OmniInsight/config/model_config.yaml"),
        help="Path to model configuration YAML",
    )
    parser.add_argument(
        "--task-type",
        choices=["classification", "regression"],
        default=None,
        help="Override task type",
    )
    parser.add_argument(
        "--model-type",
        choices=["xgboost", "dnn"],
        default=None,
        help="Override model type",
    )
    parser.add_argument("--target-column", type=str, default=None, help="Override target column")
    parser.add_argument("--top-k", type=int, default=None, help="Number of top SHAP features")
    parser.add_argument("--use-bio-adapter", action="store_true", help="Use BioAdapter for multi-omics mode")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for JSON report")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def _build_general_config(config_raw: dict[str, Any], args: argparse.Namespace) -> GeneralAdapterConfig:
    default_cfg = config_raw["default"]
    dnn_cfg = config_raw.get("dnn", {})

    return GeneralAdapterConfig(
        target_column=args.target_column or default_cfg["target_column"],
        task_type=args.task_type or default_cfg["task_type"],
        model_type=args.model_type or default_cfg["model_type"],
        top_k_features=args.top_k or default_cfg.get("top_k_features", 10),
        dnn_hidden_layers=dnn_cfg.get("hidden_layers", [128, 64, 32]),
        dnn_dropout=float(dnn_cfg.get("dropout", 0.2)),
        dnn_learning_rate=float(dnn_cfg.get("learning_rate", 1e-3)),
        dnn_max_epochs=int(dnn_cfg.get("max_epochs", 200)),
        dnn_patience=int(dnn_cfg.get("patience", 20)),
        dnn_batch_size=int(dnn_cfg.get("batch_size", 32)),
    )


def main() -> None:
    """Run full pipeline and print structured JSON report."""
    args = parse_args()
    configure_logging(args.log_level)

    logger.info("Loading config from %s", args.config)
    cfg = load_config(args.config)

    logger.info("Reading data from %s", args.data)
    df = pd.read_csv(args.data)

    general_cfg = _build_general_config(cfg, args)

    if args.use_bio_adapter:
        protein_df = pd.read_csv(args.protein_data) if args.protein_data else None
        adapter = BioAdapter()
        adapter_cfg = BioAdapterConfig(**general_cfg.__dict__, protein_df=protein_df)
        logger.info("Running BioAdapter")
        report = adapter.run(df=df, config=adapter_cfg)
    else:
        adapter = GeneralAdapter()
        logger.info("Running GeneralAdapter")
        report = adapter.run(df=df, config=general_cfg)

    output = json.dumps(report, indent=2)
    print(output)

    if args.output is not None:
        args.output.write_text(output, encoding="utf-8")
        logger.info("Saved report to %s", args.output)


if __name__ == "__main__":
    main()
