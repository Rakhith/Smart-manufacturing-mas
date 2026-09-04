"""Semantic mapping engine for Phase 2.

Maps raw and derived features to canonical semantic representations,
applies safe unit conversions, isolates target labels, and produces
dataset-partitioned Canonical Machine State tables.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml
import numpy as np
import pandas as pd

from phase2.ontology import (
    FeatureSemanticMetadata,
    SemanticModality,
    PhysicalQuantity,
    MeasurementRole,
    FeatureTransformation,
    SemanticConfidence,
    convert_unit,
    load_ontology,
)


class SemanticNormalizer:
    def __init__(self, mappings_dir: Optional[Path] = None, ontology_path: Optional[Path] = None):
        self.base_dir = Path(__file__).parent
        self.mappings_dir = mappings_dir or (self.base_dir / "dataset_mappings")
        self.ontology = load_ontology(ontology_path)
        self.dataset_configs: Dict[str, Dict[str, Any]] = self._load_dataset_configs()

    def _load_dataset_configs(self) -> Dict[str, Dict[str, Any]]:
        configs = {}
        for file in self.mappings_dir.glob("*.yaml"):
            dataset_id = file.stem
            with open(file, "r", encoding="utf-8") as stream:
                configs[dataset_id] = yaml.safe_load(stream)
        return configs

    def parse_derived_feature(self, feature_name: str) -> Tuple[str, str, Optional[str], str]:
        """Decompose feature name into (base_feature, transformation, parameter, temporal_role)."""
        # Built-in identifiers
        if feature_name in {"prepared_observation_id", "dataset_id"}:
            return feature_name, "raw", None, "identifier"

        if "__" not in feature_name:
            return feature_name, "raw", None, "instantaneous_measurement"

        parts = feature_name.split("__")
        base = parts[0]
        tail = parts[1:]

        # Compound case like channel_1__rms__rolling_mean_10
        if len(tail) == 2 and tail[0] == "rms" and tail[1].startswith("rolling_mean_"):
            window = tail[1].replace("rolling_mean_", "")
            return base, "rolling_mean", f"rms_window_{window}", "causal_rolling_statistic"
        elif len(tail) == 2 and tail[0] == "rms" and tail[1] == "delta_previous":
            return base, "delta_1", "rms_lag_1", "causal_difference_statistic"

        action = tail[0]

        # Waveform / spectral features
        if action in {
            "mean", "rms", "variance", "std", "skewness", "kurtosis",
            "crest_factor", "dominant_frequency_hz", "spectral_energy", "spectral_entropy"
        }:
            if action.startswith("spectral") or action == "dominant_frequency_hz":
                return base, f"fft_{action}" if not action.startswith("fft_") else action, None, "frequency_domain_statistic"
            return base, action, None, "waveform_statistic"

        # Cycle summaries (UCI Hydraulic)
        if action in {"first", "last"}:
            return base, action, None, "cycle_boundary_statistic"
        if action == "slope_per_sample":
            return base, "cycle_slope", "per_sample", "cycle_summary_statistic"

        # Windowed summaries (MetroPT-3)
        if action == "delta_window":
            return base, "window_delta", None, "window_summary_statistic"

        # Causal rolling features
        if action.startswith("rolling_mean_"):
            window = action.replace("rolling_mean_", "")
            return base, "rolling_mean", f"window_{window}", "causal_rolling_statistic"
        if action.startswith("rolling_std_"):
            window = action.replace("rolling_std_", "")
            return base, "rolling_std", f"window_{window}", "causal_rolling_statistic"
        if action.startswith("slope_"):
            window = action.replace("slope_", "")
            return base, "slope", f"window_{window}", "causal_trend_statistic"
        if action == "delta_1":
            return base, "delta_1", "lag_1", "causal_difference_statistic"

        # Fallback for generic window / cycle aggregations (e.g. min, max, std, mean)
        if action in {"min", "max", "mean", "std"}:
            return base, action, None, "window_summary_statistic"

        return base, "derived", "__".join(tail), "derived_statistic"

    def map_feature(self, dataset_id: str, feature_name: str) -> FeatureSemanticMetadata:
        """Map a single feature to FeatureSemanticMetadata according to ontology and config."""
        config = self.dataset_configs.get(dataset_id, {})
        explicit_features = config.get("features", {})
        asset_context = config.get("asset_context", "industrial_asset")
        machine_archetype = config.get("machine_archetype", "unknown")
        provenance = config.get("provenance", "dataset_metadata")

        # Global identifiers
        if feature_name in {"prepared_observation_id", "dataset_id"}:
            return FeatureSemanticMetadata(
                dataset_id=dataset_id,
                original_feature_name=feature_name,
                base_feature_name=feature_name,
                is_target=False,
                semantic_modality=SemanticModality.METADATA_IDENTIFIER.value,
                physical_quantity=PhysicalQuantity.UNKNOWN.value,
                measurement_role=MeasurementRole.IDENTIFIER.value,
                original_unit="id",
                canonical_unit="id",
                transformation=FeatureTransformation.RAW.value,
                temporal_role="identifier",
                asset_context=asset_context,
                machine_archetype=machine_archetype,
                provenance=provenance,
                semantic_confidence=SemanticConfidence.HIGH.value,
                mapping_source="system_pipeline",
            )

        base_name, transformation, param, temporal_role = self.parse_derived_feature(feature_name)

        # 1. Check explicit feature configuration
        feature_spec = explicit_features.get(base_name)

        # 2. Check pattern configurations
        if feature_spec is None:
            if "default_sensor_pattern" in config and base_name.startswith(config["default_sensor_pattern"].get("prefix", "")):
                feature_spec = config["default_sensor_pattern"]
            elif "default_feature_pattern" in config and base_name.startswith(config["default_feature_pattern"].get("prefix", "")):
                feature_spec = config["default_feature_pattern"]
            elif "xmeas_pattern" in config and base_name.startswith(config["xmeas_pattern"].get("prefix", "")):
                feature_spec = config["xmeas_pattern"]
            elif "xmv_pattern" in config and base_name.startswith(config["xmv_pattern"].get("prefix", "")):
                feature_spec = config["xmv_pattern"]
            elif "channel_pattern" in config and base_name.startswith(config["channel_pattern"].get("prefix", "")):
                feature_spec = config["channel_pattern"]

        # 3. Default fallback if unmapped
        if feature_spec is None:
            return FeatureSemanticMetadata(
                dataset_id=dataset_id,
                original_feature_name=feature_name,
                base_feature_name=base_name,
                is_target=False,
                semantic_modality=SemanticModality.UNKNOWN.value,
                physical_quantity=PhysicalQuantity.UNKNOWN.value,
                measurement_role=MeasurementRole.PROCESS_PARAMETER.value,
                original_unit="UNKNOWN",
                canonical_unit="UNKNOWN",
                transformation=transformation,
                temporal_role=temporal_role,
                transformation_parameter=param,
                asset_context=asset_context,
                machine_archetype=machine_archetype,
                provenance=provenance,
                semantic_confidence=SemanticConfidence.LOW.value,
                mapping_source="unmapped_fallback",
            )

        is_target = bool(feature_spec.get("is_target", False))
        modality = feature_spec.get("modality", SemanticModality.UNKNOWN.value)
        quantity = feature_spec.get("physical_quantity", PhysicalQuantity.UNKNOWN.value)
        role = feature_spec.get("role", MeasurementRole.PROCESS_PARAMETER.value)
        original_unit = feature_spec.get("unit", "UNKNOWN")
        canonical_unit = feature_spec.get("canonical_unit", original_unit)
        conversion_rule = feature_spec.get("conversion_rule")
        confidence = feature_spec.get("confidence", SemanticConfidence.LOW.value)

        # If it's a target label, force outcome_label modality
        if is_target:
            modality = SemanticModality.OUTCOME_LABEL.value
            temporal_role = "target_outcome"

        return FeatureSemanticMetadata(
            dataset_id=dataset_id,
            original_feature_name=feature_name,
            base_feature_name=base_name,
            is_target=is_target,
            semantic_modality=modality,
            physical_quantity=quantity,
            measurement_role=role,
            original_unit=original_unit,
            canonical_unit=canonical_unit,
            transformation=transformation,
            temporal_role=temporal_role,
            transformation_parameter=param,
            unit_conversion_rule=conversion_rule,
            asset_context=asset_context,
            machine_archetype=machine_archetype,
            provenance=provenance,
            semantic_confidence=confidence,
            mapping_source="dataset_yaml_mapping",
        )

    def generate_feature_catalogue(self, dataset_columns: Dict[str, List[str]]) -> pd.DataFrame:
        """Generate a complete, inspectable DataFrame catalogue of all features across datasets."""
        rows = []
        for dataset_id, columns in dataset_columns.items():
            for col in columns:
                meta = self.map_feature(dataset_id, col)
                rows.append(meta.to_dict())
        return pd.DataFrame(rows)

    def normalize_dataset_observations(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        feature_catalogue: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Produce a canonical machine state table for the dataset:
        1. Preserves raw observation values.
        2. Applies verified unit conversions to generate canonical value columns where appropriate.
        3. Identifies and isolates target columns.
        4. Calculates semantic summary metrics (modality presence, confidence score).
        """
        output_df = df.copy()

        # Map all columns
        col_mappings: Dict[str, FeatureSemanticMetadata] = {
            col: self.map_feature(dataset_id, col) for col in df.columns
        }

        # Apply safe unit conversions for columns with non-identity rules
        converted_cols = []
        for col, meta in col_mappings.items():
            if meta.unit_conversion_rule and meta.unit_conversion_rule != "x":
                rule = meta.unit_conversion_rule
                try:
                    vals = pd.to_numeric(output_df[col], errors="coerce")
                    # Safe linear transformation
                    canonical_vals = vals.apply(
                        lambda v: convert_unit(v, meta.original_unit, meta.canonical_unit, rule) if pd.notna(v) else np.nan
                    )
                    canon_col_name = f"{col}__canonical_{meta.canonical_unit}"
                    output_df[canon_col_name] = canonical_vals
                    converted_cols.append(canon_col_name)
                except Exception:
                    pass

        # Check target columns
        target_columns = [col for col, meta in col_mappings.items() if meta.is_target]
        state_columns = [
            col for col, meta in col_mappings.items()
            if not meta.is_target and meta.semantic_modality != SemanticModality.METADATA_IDENTIFIER.value
        ]

        # Summarize modalities
        modalities_present = sorted(list({
            meta.semantic_modality for meta in col_mappings.values()
            if not meta.is_target and meta.semantic_modality != SemanticModality.METADATA_IDENTIFIER.value
        }))

        stats = {
            "dataset_id": dataset_id,
            "total_records": len(df),
            "total_columns": int(df.shape[1]),
            "state_columns_count": len(state_columns),
            "target_columns_count": len(target_columns),
            "target_columns": target_columns,
            "converted_unit_columns": converted_cols,
            "modalities_present": modalities_present,
            "confidence_distribution": {
                tier: sum(1 for m in col_mappings.values() if m.semantic_confidence == tier)
                for tier in ("high", "medium", "low", "unknown")
            }
        }

        return output_df, stats
