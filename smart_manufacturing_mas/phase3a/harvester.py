"""Dataset-specific decision candidate harvester for Phase 3A.

Extracts decision-relevant candidate states from Phase 2 canonical Parquet tables
based on domain-grounded physical degradation, failure boundaries, and operational shifts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from phase3a.schema import DecisionState, StateSeverity


def safe_sample(df_sub: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Safely sample up to n rows without raising ValueError on small or empty subsets."""
    if df_sub.empty or n <= 0:
        return df_sub.iloc[0:0]
    return df_sub.sample(n=min(n, len(df_sub)), random_state=seed)


class CandidateHarvester:
    def __init__(self, phase2_dir: Path, catalogue_path: Optional[Path] = None):
        self.phase2_dir = phase2_dir
        self.cms_dir = phase2_dir / "canonical_machine_states"
        if catalogue_path is None:
            catalogue_path = phase2_dir / "semantic_feature_catalogue" / "semantic_feature_catalogue.parquet"
        self.catalogue_df = pd.read_parquet(catalogue_path)

        # Pre-build lookup: (dataset_id, column_name) -> (modality, is_target, canonical_unit)
        self.feature_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for _, row in self.catalogue_df.iterrows():
            self.feature_meta[(row["dataset_id"], row["original_feature_name"])] = {
                "modality": row["semantic_modality"],
                "is_target": bool(row["is_target"]),
                "canonical_unit": row["canonical_unit"],
                "transformation": row["transformation"],
            }

    def _build_decision_state(
        self,
        dataset_id: str,
        state_id: str,
        row: pd.Series,
        severity: StateSeverity,
        rationale: str,
        asset_context: Dict[str, Any],
        temporal_context: Dict[str, Any],
        phase2_filename: str,
    ) -> DecisionState:
        """Constructs a DecisionState with strict input vs. ground-truth target isolation."""
        input_state: Dict[str, Dict[str, Any]] = {
            "thermal": {},
            "mechanical": {},
            "kinematic": {},
            "fluid": {},
            "electrical": {},
            "acoustic": {},
            "process_operating": {},
            "health_degradation": {},
            "unmapped": {},
        }
        ground_truth: Dict[str, Any] = {}
        trend_summary: Dict[str, Any] = {}

        obs_id = str(row.get("prepared_observation_id", state_id))

        for col, val in row.items():
            if pd.isna(val):
                continue

            val_native = val.item() if hasattr(val, "item") else val

            meta = self.feature_meta.get((dataset_id, str(col)))
            if meta is None:
                # Handle newly added canonical unit columns (e.g. *__canonical_degC)
                if "__canonical_" in str(col):
                    base_col = str(col).split("__canonical_")[0]
                    base_meta = self.feature_meta.get((dataset_id, base_col))
                    if base_meta and not base_meta["is_target"]:
                        mod = base_meta["modality"]
                        target_mod = mod if mod in input_state else "unmapped"
                        input_state[target_mod][str(col)] = val_native
                continue

            if meta["is_target"]:
                ground_truth[str(col)] = val_native
            else:
                modality = meta["modality"]
                if modality not in {"metadata_identifier", "outcome_label"}:
                    target_mod = modality if modality in input_state else "unmapped"
                    input_state[target_mod][str(col)] = val_native

                # Check for trend/slope/delta features
                trans = meta.get("transformation", "")
                if trans in {"slope", "delta_1", "window_delta", "cycle_slope"}:
                    trend_summary[str(col)] = val_native

        return DecisionState(
            decision_state_id=state_id,
            dataset_id=dataset_id,
            asset_context=asset_context,
            temporal_context=temporal_context,
            decision_severity=severity.value,
            decision_relevance_rationale=rationale,
            input_machine_state=input_state,
            trend_summary=trend_summary,
            ground_truth_context=ground_truth,
            provenance={
                "source_observation_ids": [obs_id],
                "phase2_table": f"outputs/phase2/canonical_machine_states/{phase2_filename}",
            },
        )

    # -------------------------------------------------------------------------
    # Dataset-Specific Harvesters
    # -------------------------------------------------------------------------

    def harvest_ai4i(self) -> List[DecisionState]:
        df = pd.read_parquet(self.cms_dir / "ai4i_2020.parquet")
        states = []
        asset = {"machine_archetype": "cnc_mill", "asset_id": "ai4i_cnc_fleet", "operational_domain": "cnc_machining", "temporal_type": "static_tabular"}

        # 1. Critical Failure Rows
        failures = df[df["Machine failure"] == 1]
        for idx, row in failures.iterrows():
            modes = [m for m in ["TWF", "HDF", "PWF", "OSF", "RNF"] if row.get(m) == 1]
            rationale = f"Observed machine failure mode(s): {', '.join(modes) if modes else 'unspecified'}"
            s = self._build_decision_state("ai4i_2020", f"DS_AI4I_{idx}", row, StateSeverity.CRITICAL, rationale, asset, {"sequence_step": int(idx)}, "ai4i_2020.parquet")
            states.append(s)

        # 2. Degrading Rows (High Tool Wear > 185 min, No failure)
        high_wear = safe_sample(df[(df["Machine failure"] == 0) & (df["Tool wear [min]"] > 185)], 100)
        for idx, row in high_wear.iterrows():
            rationale = f"High accumulated tool wear ({row['Tool wear [min]']} min) approaching failure threshold"
            s = self._build_decision_state("ai4i_2020", f"DS_AI4I_{idx}", row, StateSeverity.DEGRADING, rationale, asset, {"sequence_step": int(idx)}, "ai4i_2020.parquet")
            states.append(s)

        # 3. Watch Rows (Moderate Tool Wear 100-160 min)
        med_wear = safe_sample(df[(df["Machine failure"] == 0) & (df["Tool wear [min]"] >= 100) & (df["Tool wear [min]"] <= 160)], 80)
        for idx, row in med_wear.iterrows():
            rationale = f"Moderate tool wear accumulation ({row['Tool wear [min]']} min); maintenance planning window"
            s = self._build_decision_state("ai4i_2020", f"DS_AI4I_{idx}", row, StateSeverity.WATCH, rationale, asset, {"sequence_step": int(idx)}, "ai4i_2020.parquet")
            states.append(s)

        # 4. Healthy Baseline (Low Tool Wear < 40 min)
        low_wear = safe_sample(df[(df["Machine failure"] == 0) & (df["Tool wear [min]"] < 40)], 80)
        for idx, row in low_wear.iterrows():
            rationale = f"Nominal operating point with fresh tooling ({row['Tool wear [min]']} min wear)"
            s = self._build_decision_state("ai4i_2020", f"DS_AI4I_{idx}", row, StateSeverity.HEALTHY, rationale, asset, {"sequence_step": int(idx)}, "ai4i_2020.parquet")
            states.append(s)

        return states

    def harvest_cmapss(self) -> List[DecisionState]:
        df = pd.read_parquet(self.cms_dir / "cmapss.parquet")
        states = []

        # Group by engine unit across all subsets (FD001 to FD004)
        for subset in df["subset"].unique():
            sub_df = df[df["subset"] == subset]
            sample_units = sub_df["unit_id"].unique()[:20]

            for u in sample_units:
                u_df = sub_df[sub_df["unit_id"] == u].sort_values("cycle")
                max_c = u_df["cycle"].max()
                asset = {"machine_archetype": "turbofan_engine", "asset_id": f"{subset}_unit_{u}", "operational_domain": "aerospace", "temporal_type": "time_series"}

                # Baseline: early cycle
                base_rows = u_df[u_df["cycle"] <= min(12, max_c)]
                if not base_rows.empty:
                    row = base_rows.iloc[-1]
                    s = self._build_decision_state("cmapss", f"DS_CMAPSS_{subset}_U{u}_C{row['cycle']}", row, StateSeverity.HEALTHY, f"Early operating cycle {row['cycle']} with RUL > 125 cycles", asset, {"cycle": int(row["cycle"]), "trajectory_phase": "healthy_baseline"}, "cmapss.parquet")
                    states.append(s)

                # Watch: RUL around 80-115
                watch_rows = u_df[(u_df["rul_cycles_label"] <= 115) & (u_df["rul_cycles_label"] >= 80)]
                if not watch_rows.empty:
                    row = watch_rows.iloc[len(watch_rows) // 2]
                    s = self._build_decision_state("cmapss", f"DS_CMAPSS_{subset}_U{u}_C{row['cycle']}", row, StateSeverity.WATCH, f"Degradation onset: RUL = {row['rul_cycles_label']} cycles", asset, {"cycle": int(row["cycle"]), "trajectory_phase": "degradation_onset"}, "cmapss.parquet")
                    states.append(s)

                # Degrading: RUL around 30-70
                deg_rows = u_df[(u_df["rul_cycles_label"] <= 70) & (u_df["rul_cycles_label"] >= 30)]
                if not deg_rows.empty:
                    row = deg_rows.iloc[len(deg_rows) // 2]
                    s = self._build_decision_state("cmapss", f"DS_CMAPSS_{subset}_U{u}_C{row['cycle']}", row, StateSeverity.DEGRADING, f"Progressive HPC/fan degradation: RUL = {row['rul_cycles_label']} cycles", asset, {"cycle": int(row["cycle"]), "trajectory_phase": "active_degradation"}, "cmapss.parquet")
                    states.append(s)

                # Critical: Final cycles
                crit_rows = u_df[u_df["rul_cycles_label"] <= 25]
                for offset in [1, 5, 12, 20]:
                    target_crit = crit_rows[crit_rows["rul_cycles_label"] == offset]
                    if not target_crit.empty:
                        row = target_crit.iloc[0]
                        s = self._build_decision_state("cmapss", f"DS_CMAPSS_{subset}_U{u}_C{row['cycle']}", row, StateSeverity.CRITICAL, f"Imminent failure boundary: RUL = {row['rul_cycles_label']} cycles", asset, {"cycle": int(row["cycle"]), "trajectory_phase": "failure_adjacent"}, "cmapss.parquet")
                        states.append(s)

        return states

    def harvest_smart_maintenance_static(self) -> List[DecisionState]:
        df = pd.read_parquet(self.cms_dir / "smart_maintenance_static.parquet")
        states = []
        asset = {"machine_archetype": "industrial_machine", "asset_id": "smart_maint_fleet", "operational_domain": "factory_maintenance", "temporal_type": "static_tabular"}

        # Handles both integer (3, 2, 1) and string ("High", "Medium", "Low") encodings
        priority_tiers = [
            ([3, "3", "High"], StateSeverity.CRITICAL, "High maintenance priority; high failure probability / severe degradation"),
            ([2, "2", "Medium"], StateSeverity.WATCH, "Medium maintenance priority; scheduled inspection recommended"),
            ([1, "1", "Low"], StateSeverity.HEALTHY, "Low maintenance priority; asset operating within nominal envelope"),
        ]

        for keys, sev, rationale_prefix in priority_tiers:
            subset = df[df["Maintenance_Priority"].isin(keys)]
            sampled = safe_sample(subset, 65)
            for idx, row in sampled.iterrows():
                m_id = row.get("Machine_ID", idx)
                rationale = f"{rationale_prefix} (Failure_Prob={row.get('Failure_Prob', 'N/A')})"
                s = self._build_decision_state("smart_maintenance_static", f"DS_SMSTATIC_{m_id}_{idx}", row, sev, rationale, asset, {"sequence_step": int(idx)}, "smart_maintenance_static.parquet")
                states.append(s)

        return states

    def harvest_smart_maintenance_timeseries(self) -> List[DecisionState]:
        df = pd.read_parquet(self.cms_dir / "smart_maintenance_timeseries.parquet")
        states = []
        asset = {"machine_archetype": "industrial_machine", "asset_id": "smart_maint_ts", "operational_domain": "factory_maintenance", "temporal_type": "time_series"}

        # 1. Critical: maintenance required flag triggered
        anomalies = df[df["maintenance_required"] == 1]
        sampled_anom = safe_sample(anomalies, 80)
        for idx, row in sampled_anom.iterrows():
            rationale = f"Maintenance required flag triggered with failure type: {row.get('failure_type', 'unspecified')}"
            s = self._build_decision_state("smart_maintenance_timeseries", f"DS_SMTS_CRIT_{idx}", row, StateSeverity.CRITICAL, rationale, asset, {"sequence_step": int(idx)}, "smart_maintenance_timeseries.parquet")
            states.append(s)

        # 2. Degrading / Watch: declining remaining life (RUL between 50 and 160)
        watch_subset = df[(df["maintenance_required"] == 0) & (df["predicted_remaining_life"] >= 50) & (df["predicted_remaining_life"] <= 160)]
        sampled_watch = safe_sample(watch_subset, 70)
        for idx, row in sampled_watch.iterrows():
            rationale = f"Declining remaining life ({row.get('predicted_remaining_life', 'N/A')} hours); maintenance planning required"
            s = self._build_decision_state("smart_maintenance_timeseries", f"DS_SMTS_WATCH_{idx}", row, StateSeverity.WATCH, rationale, asset, {"sequence_step": int(idx)}, "smart_maintenance_timeseries.parquet")
            states.append(s)

        # 3. Healthy Baseline: High remaining life (>350) and no anomalies
        healthy_subset = df[(df["maintenance_required"] == 0) & (df["anomaly_flag"] == 0) & (df["predicted_remaining_life"] > 350)]
        sampled_healthy = safe_sample(healthy_subset, 70)
        for idx, row in sampled_healthy.iterrows():
            rationale = "Nominal telemetry with zero anomaly flags and high remaining life (>350 hours)"
            s = self._build_decision_state("smart_maintenance_timeseries", f"DS_SMTS_HEALTHY_{idx}", row, StateSeverity.HEALTHY, rationale, asset, {"sequence_step": int(idx)}, "smart_maintenance_timeseries.parquet")
            states.append(s)

        return states

    def harvest_iiot_6g(self) -> List[DecisionState]:
        df = pd.read_parquet(self.cms_dir / "iiot_6g.parquet")
        states = []
        asset = {"machine_archetype": "iiot_machine", "asset_id": "iiot_6g_fleet", "operational_domain": "smart_factory_iiot", "temporal_type": "time_series"}

        # Critical: Low efficiency
        low_eff = safe_sample(df[df["Efficiency_Status"] == "Low"], 70)
        for idx, row in low_eff.iterrows():
            rationale = f"Low operational efficiency status with defect rate {row.get('Quality_Control_Defect_Rate_%', 'N/A')}%"
            s = self._build_decision_state("iiot_6g", f"DS_IIOT_CRIT_{idx}", row, StateSeverity.CRITICAL, rationale, asset, {"sequence_step": int(idx)}, "iiot_6g.parquet")
            states.append(s)

        # Watch: Medium efficiency
        med_eff = safe_sample(df[df["Efficiency_Status"] == "Medium"], 65)
        for idx, row in med_eff.iterrows():
            rationale = f"Medium operational efficiency status with latency {row.get('Network_Latency_ms', 'N/A')} ms"
            s = self._build_decision_state("iiot_6g", f"DS_IIOT_WATCH_{idx}", row, StateSeverity.WATCH, rationale, asset, {"sequence_step": int(idx)}, "iiot_6g.parquet")
            states.append(s)

        # Healthy: High efficiency
        high_eff = safe_sample(df[df["Efficiency_Status"] == "High"], 65)
        for idx, row in high_eff.iterrows():
            rationale = "High operational efficiency with low defect rate and stable latency"
            s = self._build_decision_state("iiot_6g", f"DS_IIOT_HEALTHY_{idx}", row, StateSeverity.HEALTHY, rationale, asset, {"sequence_step": int(idx)}, "iiot_6g.parquet")
            states.append(s)

        return states

    def harvest_metal_etch(self) -> List[DecisionState]:
        df = pd.read_parquet(self.cms_dir / "metal_etch.parquet")
        states = []
        asset = {"machine_archetype": "etch_tool", "asset_id": "metal_etch_chamber", "operational_domain": "semiconductor", "temporal_type": "static_tabular"}

        # Target = 1: Defect / Failure
        target_1 = safe_sample(df[df["Target"] == 1], 85)
        for idx, row in target_1.iterrows():
            s = self._build_decision_state("metal_etch", f"DS_ETCH_CRIT_{idx}", row, StateSeverity.CRITICAL, "Documented wafer etch endpoint defect / chamber fault (Target=1)", asset, {"sequence_step": int(idx)}, "metal_etch.parquet")
            states.append(s)

        # Target = 0: Nominal
        target_0 = safe_sample(df[df["Target"] == 0], 80)
        for idx, row in target_0.iterrows():
            s = self._build_decision_state("metal_etch", f"DS_ETCH_HEALTHY_{idx}", row, StateSeverity.HEALTHY, "Nominal wafer etch process parameters (Target=0)", asset, {"sequence_step": int(idx)}, "metal_etch.parquet")
            states.append(s)

        return states

    def harvest_metropt3(self) -> List[DecisionState]:
        df = pd.read_parquet(self.cms_dir / "metropt3.parquet")
        states = []
        asset = {"machine_archetype": "air_compressor", "asset_id": "metropt3_apu", "operational_domain": "railway_pneumatics", "temporal_type": "time_series"}

        oil_temp_z = (df["Oil_temperature__mean"] - df["Oil_temperature__mean"].mean()) / df["Oil_temperature__mean"].std()
        h1_z = (df["H1__mean"] - df["H1__mean"].mean()) / df["H1__mean"].std()
        comp_active = df["COMP__mean"] > 0.5

        # Critical: thermal stress under compressor duty
        thermal_stress = safe_sample(df[comp_active & (oil_temp_z > 1.8)], 70)
        for idx, row in thermal_stress.iterrows():
            s = self._build_decision_state("metropt3", f"DS_METRO_CRIT_{idx}", row, StateSeverity.CRITICAL, f"Compressor thermal stress: Oil temperature elevated ({row['Oil_temperature__mean']:.1f} C, z={oil_temp_z.loc[idx]:.1f})", asset, {"sequence_step": int(idx)}, "metropt3.parquet")
            states.append(s)

        # Watch: High differential pressure across cyclonic separator / dryer
        pressure_drop = safe_sample(df[comp_active & (h1_z > 1.4)], 70)
        for idx, row in pressure_drop.iterrows():
            s = self._build_decision_state("metropt3", f"DS_METRO_WATCH_{idx}", row, StateSeverity.WATCH, f"High pneumatic pressure drop across filter/separator: H1 = {row['H1__mean']:.2f} bar", asset, {"sequence_step": int(idx)}, "metropt3.parquet")
            states.append(s)

        # Healthy: Normal active compressor operation
        nominal_active = safe_sample(df[comp_active & (oil_temp_z.abs() < 0.7) & (h1_z.abs() < 0.7)], 70)
        for idx, row in nominal_active.iterrows():
            s = self._build_decision_state("metropt3", f"DS_METRO_HEALTHY_{idx}", row, StateSeverity.HEALTHY, f"Nominal air production duty cycle at {row['TP2__mean']:.1f} bar compressor pressure", asset, {"sequence_step": int(idx)}, "metropt3.parquet")
            states.append(s)

        return states

    def harvest_uci_hydraulic(self) -> List[DecisionState]:
        df = pd.read_parquet(self.cms_dir / "uci_hydraulic.parquet")
        states = []
        asset = {"machine_archetype": "hydraulic_test_rig", "asset_id": "uci_hydraulic_rig", "operational_domain": "industrial_hydraulics", "temporal_type": "multirate_cycle"}

        # Critical states: Cooler close to total failure (3%) or severe pump leakage (2) or accumulator low (90 bar)
        crit_mask = (df["cooler_condition_pct_label"] == 3) | (df["pump_leakage_label"] == 2) | (df["accumulator_pressure_bar_label"] == 90)
        crit_df = safe_sample(df[crit_mask], 75)
        for idx, row in crit_df.iterrows():
            reasons = []
            if row["cooler_condition_pct_label"] == 3:
                reasons.append("cooler efficiency 3% (near failure)")
            if row["pump_leakage_label"] == 2:
                reasons.append("severe internal pump leakage")
            if row["accumulator_pressure_bar_label"] == 90:
                reasons.append("severe accumulator gas pre-charge loss (90 bar)")
            s = self._build_decision_state("uci_hydraulic", f"DS_HYDR_CRIT_{idx}", row, StateSeverity.CRITICAL, f"Severe hydraulic sub-system degradation: {'; '.join(reasons)}", asset, {"cycle": int(idx)}, "uci_hydraulic.parquet")
            states.append(s)

        # Degrading states: Cooler at 20% or weak pump leakage (1)
        deg_mask = (~crit_mask) & ((df["cooler_condition_pct_label"] == 20) | (df["pump_leakage_label"] == 1) | (df["accumulator_pressure_bar_label"] == 100))
        deg_df = safe_sample(df[deg_mask], 70)
        for idx, row in deg_df.iterrows():
            reasons = []
            if row["cooler_condition_pct_label"] == 20:
                reasons.append("reduced cooler efficiency (20%)")
            if row["pump_leakage_label"] == 1:
                reasons.append("weak pump leakage")
            s = self._build_decision_state("uci_hydraulic", f"DS_HYDR_DEG_{idx}", row, StateSeverity.DEGRADING, f"Progressive hydraulic component wear: {'; '.join(reasons)}", asset, {"cycle": int(idx)}, "uci_hydraulic.parquet")
            states.append(s)

        # Watch: Valve lag condition
        watch_mask = (~crit_mask) & (~deg_mask) & (df["valve_condition_pct_label"] < 100)
        watch_df = safe_sample(df[watch_mask], 50)
        for idx, row in watch_df.iterrows():
            s = self._build_decision_state("uci_hydraulic", f"DS_HYDR_WATCH_{idx}", row, StateSeverity.WATCH, f"Directional valve switching lag (condition={row['valve_condition_pct_label']}%)", asset, {"cycle": int(idx)}, "uci_hydraulic.parquet")
            states.append(s)

        # Healthy states: All components at 100% / 0 leakage / 115 bar
        healthy_mask = (df["cooler_condition_pct_label"] == 100) & (df["valve_condition_pct_label"] == 100) & (df["pump_leakage_label"] == 0) & (df["accumulator_pressure_bar_label"] == 115)
        healthy_df = safe_sample(df[healthy_mask], 60)
        for idx, row in healthy_df.iterrows():
            s = self._build_decision_state("uci_hydraulic", f"DS_HYDR_HEALTHY_{idx}", row, StateSeverity.HEALTHY, "Nominal hydraulic test rig baseline (100% cooling, 0 pump leakage, 115 bar pre-charge)", asset, {"cycle": int(idx)}, "uci_hydraulic.parquet")
            states.append(s)

        return states

    def harvest_tennessee_eastman(self) -> List[DecisionState]:
        """Harvests representative process disturbances from Tennessee Eastman while streaming row groups."""
        parquet_path = self.cms_dir / "tennessee_eastman.parquet"
        pf = pq.ParquetFile(parquet_path)
        states = []
        asset = {"machine_archetype": "chemical_plant", "asset_id": "tennessee_eastman_simulator", "operational_domain": "chemical_process", "temporal_type": "time_series"}

        rg_table = pf.read_row_group(0)
        df = rg_table.to_pandas()

        for fault_no in df["faultNumber"].unique()[:21]:
            fault_df = df[df["faultNumber"] == fault_no]
            if fault_df.empty:
                continue

            if fault_no == 0:
                sampled = safe_sample(fault_df, 40)
                for idx, row in sampled.iterrows():
                    s = self._build_decision_state("tennessee_eastman", f"DS_TEP_F0_{idx}", row, StateSeverity.HEALTHY, "Fault 0: Normal chemical plant steady-state operation", asset, {"sequence_step": int(row.get("sample", idx))}, "tennessee_eastman.parquet")
                    states.append(s)
            else:
                sample_col = "sample" if "sample" in fault_df.columns else fault_df.index
                early_onset = fault_df[fault_df[sample_col] <= 200]
                settled_fault = fault_df[fault_df[sample_col] > 250]

                sampled_early = safe_sample(early_onset, 4)
                for idx, row in sampled_early.iterrows():
                    s = self._build_decision_state("tennessee_eastman", f"DS_TEP_F{fault_no}_EARLY_{row.get('sample', idx)}", row, StateSeverity.WATCH, f"Fault {fault_no}: Initial transient disturbance onset", asset, {"sequence_step": int(row.get('sample', idx))}, "tennessee_eastman.parquet")
                    states.append(s)

                sampled_settled = safe_sample(settled_fault, 4)
                for idx, row in sampled_settled.iterrows():
                    s = self._build_decision_state("tennessee_eastman", f"DS_TEP_F{fault_no}_CRIT_{row.get('sample', idx)}", row, StateSeverity.CRITICAL, f"Fault {fault_no}: Established chemical process disturbance state", asset, {"sequence_step": int(row.get('sample', idx))}, "tennessee_eastman.parquet")
                    states.append(s)

        return states

    def harvest_nasa_ims(self) -> List[DecisionState]:
        df = pd.read_parquet(self.cms_dir / "nasa_ims.parquet")
        states = []

        for test_set in df["test_set"].unique():
            t_df = df[df["test_set"] == test_set].sort_values("snapshot_sequence")
            n_snaps = len(t_df)
            asset = {"machine_archetype": "bearing_test_rig", "asset_id": f"ims_{test_set}", "operational_domain": "rotating_machinery", "temporal_type": "high_frequency_vibration"}

            # 1. Early healthy snapshots (first 15%)
            healthy_snaps = safe_sample(t_df.iloc[: int(n_snaps * 0.20)], 45)
            for idx, row in healthy_snaps.iterrows():
                s = self._build_decision_state("nasa_ims", f"DS_IMS_{test_set}_HEALTHY_{row['snapshot_sequence']}", row, StateSeverity.HEALTHY, f"Baseline vibration signature at snapshot {row['snapshot_sequence']}", asset, {"sequence_step": int(row["snapshot_sequence"]), "trajectory_phase": "healthy_bearing"}, "nasa_ims.parquet")
                states.append(s)

            # 2. Mid watch snapshots (kurtosis rising)
            mid_snaps = safe_sample(t_df.iloc[int(n_snaps * 0.45): int(n_snaps * 0.85)], 45)
            for idx, row in mid_snaps.iterrows():
                s = self._build_decision_state("nasa_ims", f"DS_IMS_{test_set}_WATCH_{row['snapshot_sequence']}", row, StateSeverity.WATCH, f"Incipient bearing spalling/kurtosis shift at snapshot {row['snapshot_sequence']}", asset, {"sequence_step": int(row["snapshot_sequence"]), "trajectory_phase": "incipient_spalling"}, "nasa_ims.parquet")
                states.append(s)

            # 3. Critical late snapshots
            late_snaps = safe_sample(t_df.iloc[int(n_snaps * 0.90):], 45)
            for idx, row in late_snaps.iterrows():
                s = self._build_decision_state("nasa_ims", f"DS_IMS_{test_set}_CRIT_{row['snapshot_sequence']}", row, StateSeverity.CRITICAL, f"Severe bearing fatigue failure signature at snapshot {row['snapshot_sequence']}", asset, {"sequence_step": int(row["snapshot_sequence"]), "trajectory_phase": "failure_adjacent"}, "nasa_ims.parquet")
                states.append(s)

        return states

    def harvest_nasa_milling(self) -> List[DecisionState]:
        df = pd.read_parquet(self.cms_dir / "nasa_milling.parquet")
        states = []
        asset = {"machine_archetype": "cnc_mill", "asset_id": "bridgeport_cnc_mill", "operational_domain": "cnc_machining", "temporal_type": "signal_snapshot"}

        n_rows = len(df)
        # Early cuts: healthy
        early = safe_sample(df.iloc[: int(n_rows * 0.30)], 50)
        for idx, row in early.iterrows():
            s = self._build_decision_state("nasa_milling", f"DS_MILL_HEALTHY_{idx}", row, StateSeverity.HEALTHY, "Early cutting insert pass with baseline acoustic/cutting force energy", asset, {"sequence_step": int(idx), "trajectory_phase": "sharp_insert"}, "nasa_milling.parquet")
            states.append(s)

        # Mid cuts: watch
        mid = safe_sample(df.iloc[int(n_rows * 0.35): int(n_rows * 0.75)], 50)
        for idx, row in mid.iterrows():
            s = self._build_decision_state("nasa_milling", f"DS_MILL_WATCH_{idx}", row, StateSeverity.WATCH, "Moderate cutting insert flank wear progression", asset, {"sequence_step": int(idx), "trajectory_phase": "progressive_wear"}, "nasa_milling.parquet")
            states.append(s)

        # Late cuts: critical
        late = safe_sample(df.iloc[int(n_rows * 0.80):], 50)
        for idx, row in late.iterrows():
            s = self._build_decision_state("nasa_milling", f"DS_MILL_CRIT_{idx}", row, StateSeverity.CRITICAL, "Critical cutting insert flank wear nearing tool retirement threshold", asset, {"sequence_step": int(idx), "trajectory_phase": "severe_tool_wear"}, "nasa_milling.parquet")
            states.append(s)

        return states

    def harvest_all(self) -> Dict[str, List[DecisionState]]:
        """Harvest candidate decision states across all 11 active datasets."""
        harvest_funcs = {
            "ai4i_2020": self.harvest_ai4i,
            "cmapss": self.harvest_cmapss,
            "smart_maintenance_static": self.harvest_smart_maintenance_static,
            "smart_maintenance_timeseries": self.harvest_smart_maintenance_timeseries,
            "iiot_6g": self.harvest_iiot_6g,
            "metal_etch": self.harvest_metal_etch,
            "metropt3": self.harvest_metropt3,
            "uci_hydraulic": self.harvest_uci_hydraulic,
            "tennessee_eastman": self.harvest_tennessee_eastman,
            "nasa_ims": self.harvest_nasa_ims,
            "nasa_milling": self.harvest_nasa_milling,
        }

        results: Dict[str, List[DecisionState]] = {}
        for d_id, func in harvest_funcs.items():
            logging.info("Harvesting candidate decision states for %s...", d_id)
            candidates = func()
            results[d_id] = candidates
            logging.info("Harvested %d candidate states for %s", len(candidates), d_id)

        return results
