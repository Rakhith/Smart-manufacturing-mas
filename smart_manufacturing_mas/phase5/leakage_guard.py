"""Temporal Leakage Guard and Audit Engine for Phase 5.

Guarantees strict causal ordering:
    latest_input_timestamp <= prediction_timestamp < future_event_timestamp
Prevents any target, work-order, or future telemetry leakage into decision states.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


class TemporalLeakageGuard:
    """Audits temporal sequences and verifies causal boundaries."""

    def __init__(self):
        self.audit_records: List[Dict[str, Any]] = []
        self.violations: List[Dict[str, Any]] = []

    def audit_prediction_point(
        self,
        asset_id: str,
        prediction_timestamp: str,
        latest_input_timestamp: str,
        future_event_timestamp: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> bool:
        """Audits a single prediction timepoint against causal invariants."""
        pred_dt = pd.to_datetime(prediction_timestamp)
        input_dt = pd.to_datetime(latest_input_timestamp)

        # Invariant 1: latest_input_timestamp <= prediction_timestamp
        delta_input = (pred_dt - input_dt).total_seconds()
        input_valid = delta_input >= 0

        # Invariant 2: prediction_timestamp < future_event_timestamp (if future event exists)
        event_valid = True
        delta_event = None
        if future_event_timestamp is not None:
            event_dt = pd.to_datetime(future_event_timestamp)
            delta_event = (event_dt - pred_dt).total_seconds()
            event_valid = delta_event > 0

        passed = input_valid and event_valid

        record = {
            "asset_id": asset_id,
            "prediction_timestamp": str(prediction_timestamp),
            "latest_input_timestamp": str(latest_input_timestamp),
            "future_event_timestamp": str(future_event_timestamp) if future_event_timestamp else None,
            "event_type": event_type,
            "delta_input_sec": delta_input,
            "delta_future_event_sec": delta_event,
            "input_causal_pass": input_valid,
            "future_event_uncontaminated_pass": event_valid,
            "overall_pass": passed,
        }

        self.audit_records.append(record)
        if not passed:
            self.violations.append(record)

        return passed

    def get_summary(self) -> Dict[str, Any]:
        """Returns summary of temporal audit results."""
        total = len(self.audit_records)
        v_count = len(self.violations)
        return {
            "total_prediction_points_audited": total,
            "total_violations": v_count,
            "temporal_integrity_rate": round((total - v_count) / total * 100, 2) if total > 0 else 100.0,
            "status": "PASS" if v_count == 0 else "FAIL_LEAKAGE_DETECTED",
            "violations_sample": self.violations[:5],
        }

    def save_audit_report(self, output_path: Path) -> None:
        """Saves audit report to JSON and CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary = self.get_summary()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": summary,
                "records": self.audit_records,
            }, f, indent=2)

        csv_path = output_path.with_suffix(".csv")
        pd.DataFrame(self.audit_records).to_csv(csv_path, index=False)
