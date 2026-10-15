"""Episodic Feedback Store for tracking closed-loop state transitions and outcomes."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from phase5.closed_loop.types import ExecutionMode, MaintenanceOutcome, TransitionRecord


class FeedbackStore:
    """Maintains sequential execution history, outcomes, and rolling metrics."""

    def __init__(self):
        self.records: List[TransitionRecord] = []
        self.asset_histories: Dict[str, List[TransitionRecord]] = defaultdict(list)

    def add_record(self, record: TransitionRecord) -> None:
        """Appends a new transition record to the episodic store."""
        self.records.append(record)
        self.asset_histories[record.asset_id].append(record)

    def get_history(self, asset_id: str) -> List[TransitionRecord]:
        return self.asset_histories.get(asset_id, [])

    def get_recent_trajectory(self, asset_id: str, k: int = 5) -> List[TransitionRecord]:
        hist = self.get_history(asset_id)
        return hist[-k:] if len(hist) >= k else hist

    def compute_asset_metrics(self, asset_id: str) -> Dict[str, Any]:
        """Calculates closed-loop operational metrics for a specific asset."""
        hist = self.get_history(asset_id)
        if not hist:
            return {"asset_id": asset_id, "total_decisions": 0}

        total_decisions = len(hist)
        interventions = 0
        inspections = 0
        repairs = 0
        replacements = 0
        shutdowns = 0
        total_downtime = 0.0
        total_parts_cost = 0.0
        total_labor_hours = 0.0
        prevented_failures = 0
        missed_failures = 0
        ground_truth_matches = 0

        for r in hist:
            aid = r.executed_action_id
            out = r.outcome
            if "MON" not in aid:
                interventions += 1
            if "INSP" in aid:
                inspections += 1
            elif "CORR" in aid:
                repairs += 1
            elif "REPL" in aid:
                replacements += 1
            elif "OP_CONTROLLED_SHUTDOWN" in aid or "OP_EMERGENCY_STOP" in aid:
                shutdowns += 1

            total_downtime += out.downtime_hours
            total_parts_cost += out.parts_cost_usd
            total_labor_hours += out.labor_hours
            if out.prevented_failure:
                prevented_failures += 1
            if out.missed_failure:
                missed_failures += 1
            if r.ground_truth_matched:
                ground_truth_matches += 1

        agreement_rate = (ground_truth_matches / total_decisions) if total_decisions > 0 else 0.0

        return {
            "asset_id": asset_id,
            "total_decisions": total_decisions,
            "interventions": interventions,
            "inspections": inspections,
            "repairs": repairs,
            "replacements": replacements,
            "shutdowns": shutdowns,
            "total_downtime_hours": round(total_downtime, 2),
            "total_parts_cost_usd": round(total_parts_cost, 2),
            "total_labor_hours": round(total_labor_hours, 2),
            "prevented_failures": prevented_failures,
            "missed_failures": missed_failures,
            "ground_truth_matches": ground_truth_matches,
            "agreement_rate": round(agreement_rate, 4),
        }

    def compute_fleet_metrics(self) -> Dict[str, Any]:
        """Aggregates metrics across all tracked assets."""
        if not self.records:
            return {"total_decisions": 0, "total_assets": 0}

        per_asset = [self.compute_asset_metrics(aid) for aid in sorted(self.asset_histories.keys())]

        total_decisions = len(self.records)
        total_interventions = sum(a["interventions"] for a in per_asset)
        total_inspections = sum(a["inspections"] for a in per_asset)
        total_repairs = sum(a["repairs"] for a in per_asset)
        total_replacements = sum(a["replacements"] for a in per_asset)
        total_shutdowns = sum(a["shutdowns"] for a in per_asset)
        total_downtime = sum(a["total_downtime_hours"] for a in per_asset)
        total_parts_cost = sum(a["total_parts_cost_usd"] for a in per_asset)
        total_labor_hours = sum(a["total_labor_hours"] for a in per_asset)
        total_prevented = sum(a["prevented_failures"] for a in per_asset)
        total_missed = sum(a["missed_failures"] for a in per_asset)
        total_matches = sum(a["ground_truth_matches"] for a in per_asset)

        return {
            "total_assets": len(self.asset_histories),
            "total_decisions": total_decisions,
            "total_interventions": total_interventions,
            "total_inspections": total_inspections,
            "total_repairs": total_repairs,
            "total_replacements": total_replacements,
            "total_shutdowns": total_shutdowns,
            "total_downtime_hours": round(total_downtime, 2),
            "total_parts_cost_usd": round(total_parts_cost, 2),
            "total_labor_hours": round(total_labor_hours, 2),
            "prevented_failures": total_prevented,
            "missed_failures": total_missed,
            "ground_truth_matches": total_matches,
            "overall_agreement_rate": round(total_matches / total_decisions, 4) if total_decisions > 0 else 0.0,
            "per_asset_breakdown": per_asset,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Converts all episodic transition records to a flat DataFrame."""
        rows = []
        for r in self.records:
            d = r.to_dict()
            out = d.pop("outcome", {})
            for ok, ov in out.items():
                d[f"outcome__{ok}"] = ov
            rows.append(d)
        return pd.DataFrame(rows)

    def export(self, filepath: Path) -> None:
        """Exports transitions to CSV / JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.suffix == ".json":
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in self.records], f, indent=2)
        else:
            df = self.to_dataframe()
            df.to_csv(filepath, index=False)
