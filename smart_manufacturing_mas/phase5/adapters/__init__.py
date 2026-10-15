"""Phase 5 External Dataset Adapters Package."""

from phase5.adapters.base_adapter import ExternalDatasetAdapter
from phase5.adapters.oil040_adapter import Oil040Adapter
from phase5.adapters.factory_sensors_adapter import FactorySensorsAdapter
from phase5.adapters.industrial_synthetic_adapter import IndustrialSyntheticAdapter

__all__ = [
    "ExternalDatasetAdapter",
    "Oil040Adapter",
    "FactorySensorsAdapter",
    "IndustrialSyntheticAdapter",
]
