"""Abstract Base Interface for External Dataset Adapters in Phase 5."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


class ExternalDatasetAdapter(ABC):
    """Common interface exposing standardized methods across all external datasets."""

    def __init__(self, data_dir: Path, dataset_id: str):
        self.data_dir = Path(data_dir)
        self.dataset_id = dataset_id
        self.is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """Loads dataset tables from disk."""
        pass

    @abstractmethod
    def validate_schema(self) -> Dict[str, Any]:
        """Validates existence and integrity of required tables and columns."""
        pass

    @abstractmethod
    def get_asset_ids(self) -> List[str]:
        """Returns list of unique asset/machine identifiers."""
        pass

    @abstractmethod
    def get_temporal_sequence(self, asset_id: str) -> pd.DataFrame:
        """Returns ordered time series of telemetry for a specific asset."""
        pass

    @abstractmethod
    def get_ground_truth_events(self, asset_id: str) -> Dict[str, Any]:
        """Returns dictionary of historical work orders and failure events for an asset."""
        pass

    @abstractmethod
    def build_decision_state_at(
        self,
        asset_id: str,
        timestamp: str,
        history_window_size: int = 10,
    ) -> Dict[str, Any]:
        """Constructs a canonical DecisionState dictionary at time t using ONLY data <= t."""
        pass

    @abstractmethod
    def map_action(self, raw_action: str) -> Tuple[Optional[str], float, str]:
        """Maps raw external action label to (canonical_action_id, confidence, notes)."""
        pass
