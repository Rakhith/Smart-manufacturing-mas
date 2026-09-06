"""Action ontology definitions, schemas, and loader for Phase 3B."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml


class ActionCategory(str, Enum):
    MONITORING_OBSERVATION = "monitoring_observation"
    INSPECTION_DIAGNOSIS = "inspection_diagnosis"
    CORRECTIVE_MAINTENANCE = "corrective_maintenance"
    REPLACEMENT_OVERHAUL = "replacement_overhaul"
    OPERATIONAL_MITIGATION = "operational_mitigation"


class InterventionRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DowntimeCost(str, Enum):
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ActionPreconditions:
    min_severity: str = "HEALTHY"
    max_severity: str = "CRITICAL"

    def is_severity_admissible(self, severity: str) -> bool:
        severity_rank = {
            "HEALTHY": 0,
            "WATCH": 1,
            "DEGRADING": 2,
            "CRITICAL": 3,
            "UNCLASSIFIED": 1,
        }
        target_rank = severity_rank.get(severity.upper(), 1)
        min_rank = severity_rank.get(self.min_severity.upper(), 0)
        max_rank = severity_rank.get(self.max_severity.upper(), 3)
        return min_rank <= target_rank <= max_rank

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MaintenanceAction:
    action_id: str
    action_name: str
    category: str
    description: str
    applicable_modalities: List[str]
    applicable_archetypes: List[str]
    preconditions: ActionPreconditions
    intervention_risk: str
    operational_downtime_cost: str

    def applies_to_archetype(self, archetype: str) -> bool:
        if "all" in self.applicable_archetypes:
            return True
        return archetype.lower() in [a.lower() for a in self.applicable_archetypes]

    def applies_to_modality(self, modality: str) -> bool:
        if "all" in self.applicable_modalities:
            return True
        return modality.lower() in [m.lower() for m in self.applicable_modalities]

    def is_admissible_for_state(self, archetype: str, severity: str) -> bool:
        if not self.applies_to_archetype(archetype):
            return False
        return self.preconditions.is_severity_admissible(severity)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["preconditions"] = self.preconditions.to_dict()
        return d


class ActionOntology:
    """Manages the controlled maintenance action catalog."""

    def __init__(self, actions: Dict[str, MaintenanceAction]):
        self._actions: Dict[str, MaintenanceAction] = actions

    @classmethod
    def load_from_yaml(cls, path: Optional[Path] = None) -> ActionOntology:
        if path is None:
            path = Path(__file__).parent / "action_ontology.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Ontology YAML not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        raw_actions = raw_data.get("actions", {})
        action_map: Dict[str, MaintenanceAction] = {}
        for action_id, info in raw_actions.items():
            precond_raw = info.get("preconditions", {})
            precond = ActionPreconditions(
                min_severity=precond_raw.get("min_severity", "HEALTHY"),
                max_severity=precond_raw.get("max_severity", "CRITICAL"),
            )
            action_map[action_id] = MaintenanceAction(
                action_id=action_id,
                action_name=info["action_name"],
                category=info["category"],
                description=info["description"],
                applicable_modalities=info.get("applicable_modalities", ["all"]),
                applicable_archetypes=info.get("applicable_archetypes", ["all"]),
                preconditions=precond,
                intervention_risk=info.get("intervention_risk", "low"),
                operational_downtime_cost=info.get("operational_downtime_cost", "low"),
            )
        return cls(actions=action_map)

    def get_action(self, action_id: str) -> Optional[MaintenanceAction]:
        return self._actions.get(action_id)

    def list_actions(self) -> List[MaintenanceAction]:
        return list(self._actions.values())

    def filter_actions(
        self,
        archetype: Optional[str] = None,
        severity: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[MaintenanceAction]:
        results: List[MaintenanceAction] = []
        for action in self._actions.values():
            if archetype and not action.applies_to_archetype(archetype):
                continue
            if severity and not action.preconditions.is_severity_admissible(severity):
                continue
            if category and action.category != category:
                continue
            results.append(action)
        return results

    def export_json(self, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {aid: act.to_dict() for aid, act in self._actions.items()}
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
