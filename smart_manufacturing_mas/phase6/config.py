"""Configuration classes and parameters for Phase 6 Low-Resource Cross-Domain Adaptation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DomainConfig:
    domain_id: str
    domain_name: str
    equipment_type: str
    supervision_type: str  # 'historical_work_order' | 'silver_preference' | 'synthetic_failure'
    is_synthetic: bool
    source_exclusion_type: str  # 'external_unseen' | 'leave_one_domain_out'
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Default Domain Metadata
SUPPORTED_DOMAINS: Dict[str, DomainConfig] = {
    "factory_sensors": DomainConfig(
        domain_id="factory_sensors",
        domain_name="Factory Sensors & Failure Labels",
        equipment_type="centrifugal_slurry_pump",
        supervision_type="historical_work_order",
        is_synthetic=True,
        source_exclusion_type="external_unseen",
        description="6 centrifugal slurry pumps, 52,560 hourly rows, 63 ground-truth work orders with parts cost and labor.",
    ),
    "oil040": DomainConfig(
        domain_id="oil040",
        domain_name="OIL-040 Rotating Machinery Benchmark",
        equipment_type="rotating_turbomachinery",
        supervision_type="historical_work_order",
        is_synthetic=True,
        source_exclusion_type="external_unseen",
        description="80 industrial rotating equipment assets, 9,600 telemetry records, 80 work orders with downtime.",
    ),
    "cmapss_lodo": DomainConfig(
        domain_id="cmapss_lodo",
        domain_name="NASA C-MAPSS Turbofan (Leave-One-Domain-Out)",
        equipment_type="turbofan_aircraft_engine",
        supervision_type="silver_preference",
        is_synthetic=True,
        source_exclusion_type="leave_one_domain_out",
        description="NASA turbofan engine run-to-failure degradation; strict LODO exclusion from source model.",
    ),
}


DEFAULT_BUDGETS: List[float] = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00]
DEFAULT_SEEDS: List[int] = [42, 101, 2024]


@dataclass
class AdaptationExperimentConfig:
    """Experiment configuration controlling data budgets, methods, domains, and leakage isolation."""
    target_domains: List[str] = field(default_factory=lambda: ["factory_sensors", "oil040", "cmapss_lodo"])
    budget_fractions: List[float] = field(default_factory=lambda: list(DEFAULT_BUDGETS))
    random_seeds: List[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))
    train_ratio: float = 0.40
    val_ratio: float = 0.30
    test_ratio: float = 0.30
    reference_alpha: float = 0.75
    cbr_k: int = 5
    bootstrap_resamples: int = 1000
    confidence_level: float = 0.95
    model_dir: Path = Path("outputs/phase4/models")
    corpus_path: Path = Path("outputs/phase3a/selected_decision_states/decision_states_corpus.json")
    ontology_path: Path = Path("outputs/phase3b/action_ontology/action_ontology.json")
    training_table_path: Path = Path("outputs/phase4/training/phase4_training_table.parquet")
    output_dir: Path = Path("outputs/phase6")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["model_dir"] = str(self.model_dir)
        d["corpus_path"] = str(self.corpus_path)
        d["ontology_path"] = str(self.ontology_path)
        d["training_table_path"] = str(self.training_table_path)
        d["output_dir"] = str(self.output_dir)
        d["domains_metadata"] = {k: SUPPORTED_DOMAINS[k].to_dict() for k in self.target_domains if k in SUPPORTED_DOMAINS}
        return d

    def save(self, file_path: Path) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: Path) -> AdaptationExperimentConfig:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.pop("domains_metadata", None)
        data["model_dir"] = Path(data.get("model_dir", "outputs/phase4/models"))
        data["corpus_path"] = Path(data.get("corpus_path", "outputs/phase3a/selected_decision_states/decision_states_corpus.json"))
        data["ontology_path"] = Path(data.get("ontology_path", "outputs/phase3b/action_ontology/action_ontology.json"))
        data["training_table_path"] = Path(data.get("training_table_path", "outputs/phase4/training/phase4_training_table.parquet"))
        data["output_dir"] = Path(data.get("output_dir", "outputs/phase6"))
        return cls(**data)
