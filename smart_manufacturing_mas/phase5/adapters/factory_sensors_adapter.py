"""External Dataset Adapter for Factory Sensors & Failure Labels Benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from phase5.action_mapper import ActionMapper
from phase5.adapters.base_adapter import ExternalDatasetAdapter


STAGE_SEVERITY_MAP = {
    "healthy": "HEALTHY",
    "degrading": "DEGRADING",
    "critical": "CRITICAL",
    "stopped": "CRITICAL",
}


class FactorySensorsAdapter(ExternalDatasetAdapter):
    """Ingests and normalizes the Factory Sensors Centrifugal Slurry Pump benchmark."""

    def __init__(self, data_dir: Path = Path("data/external/factory_sensors")):
        super().__init__(data_dir=data_dir, dataset_id="factory_sensors")
        self.action_mapper = ActionMapper()
        self.assets_df: Optional[pd.DataFrame] = None
        self.readings_df: Optional[pd.DataFrame] = None
        self.work_orders_df: Optional[pd.DataFrame] = None
        self.failure_events_df: Optional[pd.DataFrame] = None

    def load(self) -> None:
        """Loads and sorts tables."""
        d = self.data_dir
        self.assets_df = pd.read_csv(d / "assets.csv")
        self.work_orders_df = pd.read_csv(d / "work_orders.csv")
        self.failure_events_df = pd.read_csv(d / "failure_events.csv")
        readings = pd.read_csv(d / "sensor_readings.csv")

        # Datetime parsing
        readings["recorded_at"] = pd.to_datetime(readings["recorded_at"])
        readings = readings.sort_values(by=["asset_id", "recorded_at"]).reset_index(drop=True)
        self.readings_df = readings

        self.work_orders_df["opened_at"] = pd.to_datetime(self.work_orders_df["opened_at"])
        self.work_orders_df["closed_at"] = pd.to_datetime(self.work_orders_df["closed_at"])
        self.failure_events_df["failed_at"] = pd.to_datetime(self.failure_events_df["failed_at"])
        self.failure_events_df["restored_at"] = pd.to_datetime(self.failure_events_df["restored_at"])

        # Link each work order to its nearest failure event to extract failure_mode
        canon_actions = []
        for _, wo in self.work_orders_df.iterrows():
            aid = wo["asset_id"]
            wo_time = wo["opened_at"]
            # Find closest failure on this asset
            asset_fails = self.failure_events_df[self.failure_events_df["asset_id"] == aid]
            matched_mode = None
            if not asset_fails.empty:
                # Find failure within +/- 48 hours of work order
                diffs = (asset_fails["failed_at"] - wo_time).abs()
                min_idx = diffs.idxmin()
                if diffs.loc[min_idx].total_seconds() <= 48 * 3600:
                    matched_mode = asset_fails.loc[min_idx, "failure_mode"]

            if matched_mode:
                c_act, _, _ = self.action_mapper.map_external_action("factory_sensors", matched_mode)
            else:
                order_type = str(wo.get("order_type", "preventive")).lower()
                c_act, _, _ = self.action_mapper.map_external_action("factory_sensors", order_type)

            canon_actions.append(c_act or "ACT_INSP_SUBSYSTEM")

        self.work_orders_df["canonical_action_id"] = canon_actions
        self.is_loaded = True

    def validate_schema(self) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load()
        return {
            "dataset_id": self.dataset_id,
            "pump_assets_count": len(self.assets_df) if self.assets_df is not None else 0,
            "sensor_readings_rows": len(self.readings_df) if self.readings_df is not None else 0,
            "work_orders_count": len(self.work_orders_df) if self.work_orders_df is not None else 0,
            "failure_events_count": len(self.failure_events_df) if self.failure_events_df is not None else 0,
            "columns_present": list(self.readings_df.columns) if self.readings_df is not None else [],
            "status": "VALID",
        }

    def get_asset_ids(self) -> List[str]:
        if not self.is_loaded:
            self.load()
        return [str(aid) for aid in sorted(self.assets_df["asset_id"].unique().tolist())]

    def get_temporal_sequence(self, asset_id: str) -> pd.DataFrame:
        if not self.is_loaded:
            self.load()
        aid_int = int(asset_id)
        sub = self.readings_df[self.readings_df["asset_id"] == aid_int].copy()
        return sub.sort_values("recorded_at").reset_index(drop=True)

    def get_ground_truth_events(self, asset_id: str) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load()
        aid_int = int(asset_id)
        wos = self.work_orders_df[self.work_orders_df["asset_id"] == aid_int].to_dict(orient="records")
        fails = self.failure_events_df[self.failure_events_df["asset_id"] == aid_int].to_dict(orient="records")
        return {"work_orders": wos, "failure_events": fails}

    def map_action(self, raw_action: str) -> Tuple[Optional[str], float, str]:
        return self.action_mapper.map_external_action("factory_sensors", raw_action)

    def build_decision_state_at(
        self,
        asset_id: str,
        timestamp: str,
        history_window_size: int = 12,
    ) -> Dict[str, Any]:
        """Constructs a canonical DecisionState dictionary at time t using ONLY telemetry <= t."""
        if not self.is_loaded:
            self.load()

        dt = pd.to_datetime(timestamp)
        aid_int = int(asset_id)
        seq = self.readings_df[(self.readings_df["asset_id"] == aid_int) & (self.readings_df["recorded_at"] <= dt)]
        if seq.empty:
            raise ValueError(f"No sensor readings found for asset {asset_id} at or before {timestamp}")

        cur = seq.iloc[-1]
        raw_stage = str(cur["degradation_stage"]).lower()
        severity = STAGE_SEVERITY_MAP.get(raw_stage, "WATCH")

        # Compute synthetic health index for consistency with Phase 4 feature pipeline
        rul_h = float(cur["rul_hours"])
        hi = min(100.0, max(5.0, (rul_h / 500.0) * 100.0))

        # Trailing history for trend extraction
        trail = seq.iloc[-history_window_size:] if len(seq) >= history_window_size else seq
        trends: Dict[str, float] = {}
        if len(trail) > 1:
            trends["vibration_slope"] = float(trail["vibration_mm_s"].iloc[-1] - trail["vibration_mm_s"].iloc[0])
            trends["temp_slope"] = float(trail["bearing_temp_c"].iloc[-1] - trail["bearing_temp_c"].iloc[0])
            trends["pressure_slope"] = float(trail["discharge_pressure_bar"].iloc[-1] - trail["discharge_pressure_bar"].iloc[0])

        state_dict = {
            "decision_state_id": f"DS_FS_PUMP_{asset_id}_{cur['reading_id']}",
            "dataset_id": "factory_sensors_slurry_pump",
            "asset_context": {
                "machine_archetype": "industrial_machine",
                "asset_id": str(asset_id),
                "operational_domain": "slurry_pumping_process",
                "temporal_type": "time_series",
            },
            "temporal_context": {
                "sequence_step": int(len(seq)),
                "cycle": int(len(seq)),
                "timestamp": str(cur["recorded_at"]),
                "trajectory_phase": raw_stage,
            },
            "decision_severity": severity,
            "decision_relevance_rationale": f"Centrifugal pump hourly telemetry. Stage: {raw_stage}, RUL: {rul_h:.1f} hours.",
            "input_machine_state": {
                "mechanical": {
                    "vibration_rms_mm_s": float(cur["vibration_mm_s"]),
                },
                "thermal": {
                    "canonical_temp_degC": float(cur["bearing_temp_c"]),
                },
                "electrical": {
                    "electric_current_A": float(cur["motor_current_a"]),
                },
                "fluid": {
                    "canonical_pressure_bar": float(cur["discharge_pressure_bar"]),
                },
                "kinematic": {
                    "canonical_speed_rpm": float(cur["rotor_speed_rpm"]),
                },
                "health_degradation": {
                    "health_index": hi,
                    "rul_hours": rul_h,
                },
            },
            "trend_summary": trends,
            "ground_truth_context": {
                "reading_id": cur["reading_id"],
                "failure_next_24h": int(cur.get("failure_next_24h", 0)),
                "next_failure_mode": str(cur.get("next_failure_mode", "")),
            },
            "provenance": {
                "source_table": "data/external/factory_sensors/sensor_readings.csv",
                "asset_id": str(asset_id),
            },
        }

        return state_dict
