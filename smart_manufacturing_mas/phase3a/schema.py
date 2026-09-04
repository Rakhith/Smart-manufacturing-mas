"""Typed schema definitions for Phase 3A Decision States."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class StateSeverity(str, Enum):
    HEALTHY = "HEALTHY"
    WATCH = "WATCH"
    DEGRADING = "DEGRADING"
    CRITICAL = "CRITICAL"
    UNCLASSIFIED = "UNCLASSIFIED"


@dataclass
class AssetContext:
    machine_archetype: str
    asset_id: str
    operational_domain: str
    temporal_type: str  # 'time_series' | 'static_tabular'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalContext:
    sequence_step: Optional[int] = None
    cycle: Optional[int] = None
    timestamp: Optional[str] = None
    trajectory_phase: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionState:
    decision_state_id: str
    dataset_id: str
    asset_context: Dict[str, Any]
    temporal_context: Dict[str, Any]
    decision_severity: str
    decision_relevance_rationale: str
    input_machine_state: Dict[str, Dict[str, Any]]
    trend_summary: Dict[str, Any]
    ground_truth_context: Dict[str, Any]
    provenance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_flat_record(self) -> Dict[str, Any]:
        """Flatten state for Parquet tabular storage while keeping targets strictly quarantined."""
        flat = {
            "decision_state_id": self.decision_state_id,
            "dataset_id": self.dataset_id,
            "machine_archetype": self.asset_context.get("machine_archetype", "unknown"),
            "asset_id": self.asset_context.get("asset_id", "unknown"),
            "operational_domain": self.asset_context.get("operational_domain", "unknown"),
            "temporal_type": self.asset_context.get("temporal_type", "unknown"),
            "sequence_step": self.temporal_context.get("sequence_step"),
            "cycle": self.temporal_context.get("cycle"),
            "timestamp": self.temporal_context.get("timestamp"),
            "trajectory_phase": self.temporal_context.get("trajectory_phase"),
            "decision_severity": self.decision_severity,
            "decision_relevance_rationale": self.decision_relevance_rationale,
        }

        # Flatten input state modalities with prefix
        for modality, features in self.input_machine_state.items():
            for feat_name, feat_val in features.items():
                flat[f"input__{modality}__{feat_name}"] = feat_val

        # Flatten trend summary
        for k, v in self.trend_summary.items():
            flat[f"trend__{k}"] = str(v) if isinstance(v, (list, dict)) else v

        # Flatten ground truth context with explicit target prefix
        for k, v in self.ground_truth_context.items():
            flat[f"target_ground_truth__{k}"] = v

        # Flatten provenance
        flat["provenance_source_ids"] = str(self.provenance.get("source_observation_ids", []))
        flat["provenance_table"] = str(self.provenance.get("phase2_table", ""))

        return flat
