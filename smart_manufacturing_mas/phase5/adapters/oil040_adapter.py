"""External Dataset Adapter for OIL-040 Rotating Equipment Benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from phase5.action_mapper import ActionMapper
from phase5.adapters.base_adapter import ExternalDatasetAdapter


class Oil040Adapter(ExternalDatasetAdapter):
    """Ingests and normalizes the multi-table OIL-040 benchmark."""

    def __init__(self, data_dir: Path = Path("data/external/oil040")):
        super().__init__(data_dir=data_dir, dataset_id="oil040")
        self.action_mapper = ActionMapper()
        self.equipment_master: Optional[pd.DataFrame] = None
        self.telemetry_df: Optional[pd.DataFrame] = None
        self.work_orders_df: Optional[pd.DataFrame] = None
        self.failure_events_df: Optional[pd.DataFrame] = None

    def load(self) -> None:
        """Loads and joins relational tables."""
        d = self.data_dir
        self.equipment_master = pd.read_csv(d / "equipment_master.csv")
        self.work_orders_df = pd.read_csv(d / "maintenance_workorders.csv")
        if (d / "failure_events.csv").exists():
            self.failure_events_df = pd.read_csv(d / "failure_events.csv")
        else:
            self.failure_events_df = pd.DataFrame()

        # Join time series tables on record_id, equipment_id, timestamp
        vib = pd.read_csv(d / "vibration_timeseries.csv")
        temp = pd.read_csv(d / "temperature_telemetry.csv")
        press = pd.read_csv(d / "pressure_telemetry.csv")
        lube = pd.read_csv(d / "lubrication_analysis.csv")
        acoust = pd.read_csv(d / "acoustic_signals.csv")
        health = pd.read_csv(d / "health_scores.csv")
        rul = pd.read_csv(d / "remaining_useful_life.csv")

        # Merge on composite primary key
        join_keys = ["record_id", "equipment_id", "timestamp"]
        merged = vib.merge(temp, on=join_keys, how="inner")
        merged = merged.merge(press, on=join_keys, how="inner")
        merged = merged.merge(lube, on=join_keys, how="inner")
        merged = merged.merge(acoust, on=join_keys, how="inner")
        merged = merged.merge(health, on=join_keys, how="inner")
        merged = merged.merge(rul, on=join_keys, how="inner")

        # Sort chronologically
        merged["timestamp"] = pd.to_datetime(merged["timestamp"])
        merged = merged.sort_values(by=["equipment_id", "timestamp"]).reset_index(drop=True)
        self.telemetry_df = merged

        # Pre-map work order canonical actions
        if not self.work_orders_df.empty:
            canon_actions = []
            for _, row in self.work_orders_df.iterrows():
                mtype = row.get("maintenance_type", "")
                aid, conf, _ = self.action_mapper.map_external_action("oil040", mtype)
                canon_actions.append(aid)
            self.work_orders_df["canonical_action_id"] = canon_actions

        self.is_loaded = True

    def validate_schema(self) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load()
        return {
            "dataset_id": self.dataset_id,
            "equipment_count": len(self.equipment_master) if self.equipment_master is not None else 0,
            "telemetry_rows": len(self.telemetry_df) if self.telemetry_df is not None else 0,
            "work_order_count": len(self.work_orders_df) if self.work_orders_df is not None else 0,
            "failure_event_count": len(self.failure_events_df) if self.failure_events_df is not None else 0,
            "columns_present": list(self.telemetry_df.columns) if self.telemetry_df is not None else [],
            "status": "VALID",
        }

    def get_asset_ids(self) -> List[str]:
        if not self.is_loaded:
            self.load()
        return sorted(self.equipment_master["equipment_id"].unique().tolist())

    def get_temporal_sequence(self, asset_id: str) -> pd.DataFrame:
        if not self.is_loaded:
            self.load()
        sub = self.telemetry_df[self.telemetry_df["equipment_id"] == asset_id].copy()
        return sub.sort_values("timestamp").reset_index(drop=True)

    def get_ground_truth_events(self, asset_id: str) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load()
        wos = self.work_orders_df[self.work_orders_df["equipment_id"] == asset_id].to_dict(orient="records")
        fails = []
        if self.failure_events_df is not None and not self.failure_events_df.empty:
            fails = self.failure_events_df[self.failure_events_df["equipment_id"] == asset_id].to_dict(orient="records")
        return {"work_orders": wos, "failure_events": fails}

    def map_action(self, raw_action: str) -> Tuple[Optional[str], float, str]:
        return self.action_mapper.map_external_action("oil040", raw_action)

    def build_decision_state_at(
        self,
        asset_id: str,
        timestamp: str,
        history_window_size: int = 10,
    ) -> Dict[str, Any]:
        """Constructs a canonical DecisionState dictionary at time t using ONLY telemetry <= t."""
        if not self.is_loaded:
            self.load()

        dt = pd.to_datetime(timestamp)
        seq = self.telemetry_df[(self.telemetry_df["equipment_id"] == asset_id) & (self.telemetry_df["timestamp"] <= dt)]
        if seq.empty:
            raise ValueError(f"No telemetry found for asset {asset_id} at or before {timestamp}")

        cur = seq.iloc[-1]
        hi = float(cur["health_index"])

        # Determine severity
        if hi >= 80.0:
            severity = "HEALTHY"
        elif hi >= 65.0:
            severity = "WATCH"
        elif hi >= 40.0:
            severity = "DEGRADING"
        else:
            severity = "CRITICAL"

        # Trailing history for trend extraction
        trail = seq.iloc[-history_window_size:] if len(seq) >= history_window_size else seq

        # Trend summary
        trends: Dict[str, float] = {}
        if len(trail) > 1:
            vib_trend = float(trail["vibration_rms_mm_s"].iloc[-1] - trail["vibration_rms_mm_s"].iloc[0])
            temp_trend = float(trail["temperature_c"].iloc[-1] - trail["temperature_c"].iloc[0])
            hi_trend = float(trail["health_index"].iloc[-1] - trail["health_index"].iloc[0])
            trends["vibration_slope"] = vib_trend
            trends["temp_slope"] = temp_trend
            trends["health_index_trend"] = hi_trend

        # Machine archetype mapping
        eq_info = self.equipment_master[self.equipment_master["equipment_id"] == asset_id]
        raw_type = eq_info.iloc[0]["asset_type"] if not eq_info.empty else "compressor"
        arch = "air_compressor" if "compressor" in raw_type else "industrial_machine"

        # Safe unit conversions
        press_bar = float(cur["pressure_psi"]) * 0.0689476

        state_dict = {
            "decision_state_id": f"DS_OIL040_{asset_id}_{cur['record_id']}",
            "dataset_id": "oil040_benchmark",
            "asset_context": {
                "machine_archetype": arch,
                "asset_id": asset_id,
                "operational_domain": "oil_and_gas_rotating",
                "temporal_type": "time_series",
            },
            "temporal_context": {
                "sequence_step": int(len(seq)),
                "cycle": int(len(seq)),
                "timestamp": str(cur["timestamp"]),
                "trajectory_phase": severity.lower(),
            },
            "decision_severity": severity,
            "decision_relevance_rationale": f"OIL-040 telemetry snapshot. Health index: {hi:.2f}, RUL: {cur['rul_days']} days.",
            "input_machine_state": {
                "mechanical": {
                    "vibration_rms_mm_s": float(cur["vibration_rms_mm_s"]),
                    "crest_factor": float(cur["crest_factor"]),
                    "kurtosis": float(cur["kurtosis"]),
                    "axis_x_mm_s": float(cur["axis_x_mm_s"]),
                    "axis_y_mm_s": float(cur["axis_y_mm_s"]),
                    "axis_z_mm_s": float(cur["axis_z_mm_s"]),
                },
                "thermal": {
                    "canonical_temp_degC": float(cur["temperature_c"]),
                    "thermal_gradient_c": float(cur["thermal_gradient_c"]),
                },
                "fluid": {
                    "canonical_pressure_bar": press_bar,
                    "pressure_psi": float(cur["pressure_psi"]),
                    "pressure_delta_psi": float(cur["pressure_delta_psi"]),
                },
                "kinematic": {
                    "canonical_speed_rpm": float(cur["rpm"]),
                },
                "acoustic": {
                    "acoustic_db": float(cur["acoustic_db"]),
                    "ultrasonic_energy": float(cur["ultrasonic_energy"]),
                },
                "lubrication": {
                    "viscosity_index": float(cur["viscosity_index"]),
                    "water_ppm": float(cur["water_ppm"]),
                    "contamination_level": float(cur["contamination_level"]),
                },
                "health_degradation": {
                    "health_index": hi,
                    "degradation_score": float(cur["degradation_score"]),
                },
            },
            "trend_summary": trends,
            "ground_truth_context": {
                "source_record_id": cur["record_id"],
                "target_failure_30d": int(cur.get("target_failure_30d", 0)),
                "rul_days": float(cur["rul_days"]),
            },
            "provenance": {
                "source_table": "data/external/oil040/vibration_timeseries.csv",
                "equipment_id": asset_id,
            },
        }

        return state_dict
