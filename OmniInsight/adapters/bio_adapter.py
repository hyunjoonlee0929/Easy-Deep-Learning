"""Biology-focused adapter with multi-modal tabular support."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .general_adapter import GeneralAdapter, GeneralAdapterConfig

logger = logging.getLogger(__name__)


@dataclass
class BioAdapterConfig(GeneralAdapterConfig):
    """Extended config for omics workflows."""

    protein_df: pd.DataFrame | None = None


class BioAdapter(GeneralAdapter):
    """Adapter extending the general flow for multi-omics datasets."""

    def _combine_modalities(self, gene_df: pd.DataFrame, protein_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Combine gene and protein tables by row index for a unified feature table."""
        gene = gene_df.reset_index(drop=True).copy()
        protein = protein_df.reset_index(drop=True).copy()

        if len(gene) != len(protein):
            raise ValueError("Gene and protein tables must have the same number of rows for alignment.")

        protein = protein.add_prefix("protein_")
        if f"protein_{target_column}" in protein.columns:
            protein = protein.drop(columns=[f"protein_{target_column}"])

        combined = pd.concat([gene, protein], axis=1)
        return combined

    def pathway_mapping(self, top_features: list[str]) -> dict[str, Any]:
        """Map top features to simple pathway categories using deterministic rules."""
        pathway_rules = {
            "glucose": "Glucose Metabolism",
            "bmi": "Metabolic Syndrome",
            "age": "Aging and Risk",
            "protein": "Protein Signaling",
            "gene": "Gene Regulation",
        }

        mapped = []
        for feature in top_features:
            f_lower = feature.lower()
            pathway = "General Biomarker Pathway"
            for key, value in pathway_rules.items():
                if key in f_lower:
                    pathway = value
                    break
            mapped.append({"feature": feature, "pathway": pathway})

        return {
            "status": "mapped",
            "mapped_features": mapped,
        }

    def run(self, df: pd.DataFrame, config: BioAdapterConfig) -> dict[str, Any]:
        """Run full workflow with optional multi-modal input integration."""
        working_df = df
        if config.protein_df is not None:
            logger.info("Combining gene and protein modalities")
            working_df = self._combine_modalities(df, config.protein_df, config.target_column)

        report = super().run(df=working_df, config=config)
        top_features = report.get("report", {}).get("top_features", [])
        report["pathway_mapping"] = self.pathway_mapping(top_features)

        report["multi_modal"] = {
            "enabled": config.protein_df is not None,
            "gene_features": len(df.columns) - 1,
            "protein_features": 0 if config.protein_df is None else len(config.protein_df.columns),
        }
        return report
