"""External Dataset Adapter for Industrial Maintenance Synthetic Work-Order Corpus."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from phase5.action_mapper import ActionMapper


class IndustrialSyntheticAdapter:
    """Ingests enterprise SAP/Maximo work orders for natural language action extraction."""

    def __init__(self, data_path: Path = Path("data/external/industrial_synthetic/maintenance.parquet")):
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.action_mapper = ActionMapper()

    def load(self, sample_limit: Optional[int] = 50000) -> None:
        """Loads work order parquet table (optionally sampled for memory efficiency)."""
        df = pd.read_parquet(self.data_path)
        if sample_limit and len(df) > sample_limit:
            df = df.sample(n=sample_limit, random_state=42).reset_index(drop=True)
        self.df = df

    def parse_natural_language_action(self, description: str) -> Dict[str, Any]:
        """Rules-based semantic parser extracting action, component, and canonical mapping."""
        text = str(description).lower()

        # Component extraction
        component = "general_assembly"
        if "bearing" in text:
            component = "bearing"
        elif "seal" in text or "gasket" in text or "packing" in text:
            component = "seal"
        elif "valve" in text:
            component = "valve"
        elif "pump" in text or "impeller" in text:
            component = "pump"
        elif "motor" in text or "winding" in text or "coil" in text:
            component = "motor"
        elif "coupling" in text or "belt" in text or "shaft" in text:
            component = "coupling_drive"
        elif "filter" in text or "radiator" in text or "heat exchanger" in text:
            component = "cooling_filter"

        # Action extraction
        action_verb = "inspect"
        canonical_aid = "ACT_INSP_SUBSYSTEM"
        confidence = 0.70

        if any(w in text for w in ["replace", "renew", "install new", "swap"]):
            action_verb = "replace"
            if component == "bearing":
                canonical_aid = "ACT_REPL_BEARING"
                confidence = 0.95
            elif component in {"seal", "valve"}:
                canonical_aid = "ACT_REPL_SEAL_VALVE"
                confidence = 0.95
            else:
                canonical_aid = "ACT_REPL_OVERHAUL"
                confidence = 0.80

        elif any(w in text for w in ["lubricate", "grease", "top up oil", "lube"]):
            action_verb = "lubricate"
            canonical_aid = "ACT_CORR_LUBRICATE"
            confidence = 0.95

        elif any(w in text for w in ["align", "torque", "tighten", "fasten"]):
            action_verb = "align_tighten"
            canonical_aid = "ACT_CORR_FASTEN_ALIGN"
            confidence = 0.90

        elif any(w in text for w in ["calibrate", "zero", "tune", "trim"]):
            action_verb = "calibrate"
            canonical_aid = "ACT_CORR_CALIBRATE"
            confidence = 0.90

        elif any(w in text for w in ["clean", "flush", "purge", "wash"]):
            action_verb = "clean_purge"
            canonical_aid = "ACT_CORR_CLEAN_PURGE"
            confidence = 0.90

        elif any(w in text for w in ["overhaul", "rebuild", "refurbish"]):
            action_verb = "overhaul"
            canonical_aid = "ACT_REPL_OVERHAUL"
            confidence = 0.95

        elif any(w in text for w in ["vibration", "spectrum", "fft"]):
            action_verb = "vibration_diagnostics"
            canonical_aid = "ACT_INSP_VIBRATION"
            confidence = 0.90

        elif any(w in text for w in ["inspect", "check", "examine", "audit", "survey"]):
            action_verb = "inspect"
            canonical_aid = "ACT_INSP_SUBSYSTEM"
            confidence = 0.85

        return {
            "original_text": description,
            "extracted_component": component,
            "extracted_action_verb": action_verb,
            "canonical_action_id": canonical_aid,
            "confidence": confidence,
            "mapping_provenance": "NLP_Rule_Extractor_v1",
        }

    def get_action_distribution(self, n_sample: int = 1000) -> Dict[str, Any]:
        """Samples work order descriptions and extracts canonical action distribution."""
        if self.df is None:
            self.load()

        sample_subset = self.df.sample(n=min(n_sample, len(self.df)), random_state=42)
        parsed = []
        for _, row in sample_subset.iterrows():
            desc = row.get("WorkOrderDescription", "")
            p = self.parse_natural_language_action(desc)
            p["work_order"] = row.get("WorkOrder")
            p["order_type"] = row.get("OrderType")
            p["planned_work"] = row.get("PlannedWork")
            p["actual_work"] = row.get("ActualWork")
            parsed.append(p)

        df_parsed = pd.DataFrame(parsed)
        counts = df_parsed["canonical_action_id"].value_counts().to_dict()
        comp_counts = df_parsed["extracted_component"].value_counts().to_dict()

        return {
            "total_sampled": len(parsed),
            "canonical_action_distribution": counts,
            "component_distribution": comp_counts,
            "sample_records": parsed[:10],
        }
