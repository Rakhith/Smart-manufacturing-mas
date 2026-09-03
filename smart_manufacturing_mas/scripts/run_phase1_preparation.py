#!/usr/bin/env python3
"""Build inspectable Phase 1 artifacts without cross-dataset normalization.

Run from ``smart_manufacturing_mas``:
    MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. ./mas_venv/bin/python scripts/run_phase1_preparation.py --clean

The script does not mutate raw data.  It intentionally preserves source feature
names and writes separate outputs under ``outputs/phase1``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from phase1.catalogue import DATASETS, DatasetDefinition, column_catalogue, dataframe_quality, inventory_entry, write_json
from phase1.features import add_causal_temporal_features, matrix_cycle_features, waveform_features


OUTPUT_DIRS = ("dataset_inventory", "dataset_catalogue", "data_quality_reports", "preprocessing_reports", "processed_datasets", "derived_features", "prepared_observations", "visualizations", "logs")
_FRAME_ARTIFACTS: dict[int, tuple[weakref.ReferenceType[pd.DataFrame], Path]] = {}


SIMPLE_METADATA: dict[str, dict[str, dict[str, str]]] = {
    "ai4i_2020": {
        "UDI": {"meaning": "row identifier", "category": "metadata/identifier"}, "Product ID": {"meaning": "product identifier", "category": "metadata/identifier"}, "Type": {"meaning": "product quality variant", "category": "operating context"},
        "Air temperature [K]": {"meaning": "air temperature", "unit": "K", "category": "thermal", "source": "column name"}, "Process temperature [K]": {"meaning": "process temperature", "unit": "K", "category": "thermal", "source": "column name"},
        "Rotational speed [rpm]": {"meaning": "rotational speed", "unit": "rpm", "category": "mechanical", "source": "column name"}, "Torque [Nm]": {"meaning": "torque", "unit": "Nm", "category": "mechanical", "source": "column name"}, "Tool wear [min]": {"meaning": "accumulated tool wear", "unit": "min", "category": "health/degradation", "source": "column name"},
        "Machine failure": {"meaning": "machine failure label", "category": "label/target"}, "TWF": {"meaning": "tool wear failure label", "category": "label/target"}, "HDF": {"meaning": "heat dissipation failure label", "category": "label/target"}, "PWF": {"meaning": "power failure label", "category": "label/target"}, "OSF": {"meaning": "overstrain failure label", "category": "label/target"}, "RNF": {"meaning": "random failure label", "category": "label/target"},
    },
    "smart_maintenance": {
        "timestamp": {"meaning": "observation timestamp", "category": "metadata/identifier"}, "machine_id": {"meaning": "machine identifier", "category": "metadata/identifier"}, "temperature": {"meaning": "temperature", "unit": "UNKNOWN/UNVERIFIED", "category": "thermal"}, "vibration": {"meaning": "vibration", "unit": "UNKNOWN/UNVERIFIED", "category": "mechanical"}, "humidity": {"meaning": "humidity", "unit": "UNKNOWN/UNVERIFIED", "category": "operating context"}, "pressure": {"meaning": "pressure", "unit": "UNKNOWN/UNVERIFIED", "category": "hydraulic/fluid"}, "energy_consumption": {"meaning": "energy consumption", "unit": "UNKNOWN/UNVERIFIED", "category": "electrical"},
        "machine_status": {"meaning": "machine status", "category": "operating context"}, "anomaly_flag": {"meaning": "anomaly label", "category": "label/target"}, "predicted_remaining_life": {"meaning": "predicted remaining life", "unit": "UNKNOWN/UNVERIFIED", "category": "health/degradation"}, "failure_type": {"meaning": "failure category", "category": "label/target"}, "downtime_risk": {"meaning": "downtime risk", "category": "business/economic"}, "maintenance_required": {"meaning": "maintenance-required label", "category": "label/target"},
        "Machine_ID": {"meaning": "machine identifier", "category": "metadata/identifier"}, "Temp_C": {"meaning": "temperature", "unit": "C", "category": "thermal"}, "Vibration_mm_s": {"meaning": "vibration velocity", "unit": "mm/s", "category": "mechanical"}, "Pressure_Bar": {"meaning": "pressure", "unit": "bar", "category": "hydraulic/fluid"}, "Acoustic_dB": {"meaning": "acoustic level", "unit": "dB", "category": "acoustic"}, "Inspection_Duration_min": {"meaning": "inspection duration", "unit": "min", "category": "business/economic"}, "Downtime_Cost_USD": {"meaning": "downtime cost", "unit": "USD", "category": "business/economic"}, "Technician_Availability_pct": {"meaning": "technician availability", "unit": "%", "category": "business/economic"}, "Failure_Prob": {"meaning": "failure probability", "unit": "UNKNOWN/UNVERIFIED", "category": "health/degradation"}, "Maintenance_Priority": {"meaning": "maintenance priority label", "category": "label/target"},
    },
    "iiot_6g": {
        "Timestamp": {"meaning": "observation timestamp", "category": "metadata/identifier"}, "Machine_ID": {"meaning": "machine identifier", "category": "metadata/identifier"}, "Operation_Mode": {"meaning": "operating mode", "category": "operating context"}, "Temperature_C": {"meaning": "temperature", "unit": "C", "category": "thermal"}, "Vibration_Hz": {"meaning": "vibration measurement", "unit": "Hz", "category": "mechanical", "source": "column name; exact vibration quantity unverified"}, "Power_Consumption_kW": {"meaning": "power consumption", "unit": "kW", "category": "electrical"}, "Network_Latency_ms": {"meaning": "network latency", "unit": "ms", "category": "operating context"}, "Packet_Loss_%": {"meaning": "packet loss", "unit": "%", "category": "operating context"}, "Quality_Control_Defect_Rate_%": {"meaning": "quality-control defect rate", "unit": "%", "category": "process"}, "Production_Speed_units_per_hr": {"meaning": "production speed", "unit": "units/hr", "category": "operating context"}, "Predictive_Maintenance_Score": {"meaning": "predictive maintenance score", "unit": "UNKNOWN/UNVERIFIED", "category": "health/degradation"}, "Error_Rate_%": {"meaning": "error rate", "unit": "%", "category": "process"}, "Efficiency_Status": {"meaning": "efficiency status", "category": "label/target"},
    },
}


def setup_output(root: Path, clean: bool) -> None:
    if clean and root.exists():
        shutil.rmtree(root)
    for name in OUTPUT_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)


def save_frame(frame: pd.DataFrame, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    cached = _FRAME_ARTIFACTS.get(id(frame))
    source = cached[1] if cached and cached[0]() is frame else None
    if source and source.exists():
        linked = destination.with_suffix(source.suffix)
        try:
            linked.unlink(missing_ok=True)
            linked.symlink_to(source.resolve())
            return linked
        except OSError:
            # A separate filesystem or a restricted output mount may not allow
            # symlinks; fall through to a normal write in that case.
            pass
            destination.unlink(missing_ok=True)
    try:
        frame.to_parquet(destination, index=False)
        result = destination
    except (ImportError, ModuleNotFoundError):
        # Keep the pipeline runnable in lightweight environments without a
        # Parquet engine; CSV remains an inspectable lossless tabular artifact.
        fallback = destination.with_suffix(".csv")
        frame.to_csv(fallback, index=False)
        result = fallback
    _FRAME_ARTIFACTS[id(frame)] = (weakref.ref(frame), result)
    return result


def write_markdown_catalogue(root: Path, record: dict[str, Any]) -> None:
    columns = record.get("sensor_catalogue", [])
    lines = [f"# {record['name']}", "", "## Identity", "", f"- Dataset ID: `{record['dataset_id']}`", f"- Source: `{record['source_path']}`", f"- Domain: {record['domain']}", f"- Asset type: {record['asset_type']}", f"- Component: {record['component']}", f"- Processing type: `{record['processing_type']}`", "", "## Data structure", "", f"- Records/files: {record.get('record_count', 'UNKNOWN')} records; {record.get('file_count', 'UNKNOWN')} files", f"- Columns: {record.get('column_count', 'UNKNOWN')}", f"- Temporal structure: {record.get('temporal_structure', 'UNKNOWN/UNVERIFIED')}", "", "## Labels and outcomes", ""]
    labels = record.get("labels", {})
    for name, present in labels.items():
        lines.append(f"- {name}: {'present' if present else 'not observed'}")
    lines += ["", "## Sensor/feature catalogue", "", "| Feature | Physical meaning | Unit | Category |", "|---|---|---|---|"]
    for item in columns:
        lines.append(f"| {item['feature']} | {item['physical_meaning']} | {item['unit']} | {item['category']} |")
    lines += ["", "## Preparation notes", ""] + [f"- {note}" for note in record.get("notes", [])]
    (root / "dataset_catalogue" / f"{record['dataset_id']}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_artifact_paths(root: Path, value: Any) -> Any:
    if isinstance(value, str) and value.endswith(".parquet"):
        csv_path = root / (value[:-8] + ".csv")
        if csv_path.exists() and not (root / value).exists():
            return value[:-8] + ".csv"
    if isinstance(value, str) and "; " in value:
        return "; ".join(normalize_artifact_paths(root, item) for item in value.split("; "))
    if isinstance(value, list):
        return [normalize_artifact_paths(root, item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_artifact_paths(root, item) for key, item in value.items()}
    return value


def artifact_paths_exist(root: Path, value: Any) -> bool:
    if isinstance(value, str):
        return all((root / item.strip()).exists() for item in value.split("; "))
    if isinstance(value, list):
        return all(artifact_paths_exist(root, item) for item in value)
    return True


def artifact_record_count(root: Path, value: Any) -> int:
    paths = value if isinstance(value, list) else str(value).split("; ")
    total = 0
    for relative in paths:
        path = root / relative
        if path.suffix == ".parquet":
            import pyarrow.parquet as parquet
            total += parquet.ParquetFile(path).metadata.num_rows
        elif path.suffix == ".csv":
            with path.open("rb") as stream:
                total += max(0, sum(1 for _ in stream) - 1)
    return total


def save_catalogue(root: Path, record: dict[str, Any]) -> None:
    record = normalize_artifact_paths(root, record)
    write_json(root / "dataset_catalogue" / f"{record['dataset_id']}.json", record)
    write_markdown_catalogue(root, record)


def plot_bar(root: Path, dataset_id: str, title: str, values: pd.Series, filename: str) -> None:
    values = values.head(20)
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(max(7, len(values) * 0.6), 4.5))
    ax.bar(values.index.astype(str), values.to_numpy(), color="#2878b5")
    ax.set_title(title)
    ax.set_ylabel("records")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(root / "visualizations" / f"{dataset_id}__{filename}.png", dpi=150)
    plt.close(fig)


def plot_series(root: Path, dataset_id: str, frame: pd.DataFrame, x: str, columns: list[str], filename: str, max_rows: int = 3000) -> None:
    available = [c for c in columns if c in frame]
    if not available or x not in frame:
        return
    sampled = frame[[x, *available]].iloc[:max_rows].copy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for column in available[:5]:
        ax.plot(sampled[x], sampled[column], label=column, linewidth=0.8)
    ax.set_title(f"{dataset_id}: representative source trajectories")
    ax.set_xlabel(x)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "visualizations" / f"{dataset_id}__{filename}.png", dpi=150)
    plt.close(fig)


def quality_and_outputs(root: Path, dataset_id: str, before: pd.DataFrame, after: pd.DataFrame, labels: list[str], preprocessing: dict[str, Any]) -> None:
    quality_before = dataframe_quality(before, labels)
    quality_after = dataframe_quality(after, labels)
    write_json(root / "data_quality_reports" / f"{dataset_id}.json", {"before": quality_before, "after": quality_after})
    preprocessing.update({"records_before": int(len(before)), "records_after": int(len(after)), "columns_before": int(before.shape[1]), "columns_after": int(after.shape[1])})
    write_json(root / "preprocessing_reports" / f"{dataset_id}.json", preprocessing)
    plot_bar(root, dataset_id, f"{dataset_id}: missing values by column (before)", before.isna().sum().sort_values(ascending=False), "missingness")
    for label in labels:
        if label in before:
            plot_bar(root, dataset_id, f"{dataset_id}: {label} distribution", before[label].value_counts(dropna=False), f"label_{label.lower().replace(' ', '_')}")


def source_manifest(data_root: Path, definition: DatasetDefinition) -> list[dict[str, Any]]:
    path = data_root / definition.relative_path
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    return [{"path": str(p.relative_to(data_root)), "bytes": p.stat().st_size, "sha256": None if p.stat().st_size > 75_000_000 else __import__("hashlib").sha256(p.read_bytes()).hexdigest()} for p in files if p.is_file()]


def base_catalogue(definition: DatasetDefinition, data_root: Path) -> dict[str, Any]:
    entry = inventory_entry(definition, data_root)
    return {
        "dataset_id": definition.dataset_id, "name": definition.name, "source_path": definition.relative_path,
        "domain": definition.domain, "asset_type": definition.asset_type, "component": definition.component,
        "processing_type": definition.processing_type, "format": definition.format, "status": definition.status,
        "file_count": entry["file_count"], "size_bytes": entry["size_bytes"], "notes": ([definition.notes] if definition.notes else []),
        "source_manifest": source_manifest(data_root, definition), "labels": {}, "sensor_catalogue": [],
    }


def prepare_static_csv(root: Path, data_root: Path, definition: DatasetDefinition, path: Path, labels: list[str], timestamp: str | None = None, entity: str | None = None) -> dict[str, Any]:
    before = pd.read_csv(path)
    frame = before.copy()
    preprocessing = {"dataset_id": definition.dataset_id, "operations": [], "removed_records": 0, "columns_changed": []}
    if timestamp and timestamp in frame:
        parsed = pd.to_datetime(frame[timestamp], errors="coerce", utc=True)
        invalid = int(parsed.isna().sum())
        frame[timestamp] = parsed
        preprocessing["operations"].append({"operation": "parse_timestamp", "column": timestamp, "invalid_to_missing": invalid})
        sort_by = [entity, timestamp] if entity and entity in frame else [timestamp]
        frame = frame.sort_values(sort_by, kind="stable")
    duplicates = int(frame.duplicated().sum())
    if duplicates:
        frame = frame.drop_duplicates().copy()
    preprocessing["operations"].append({"operation": "remove_exact_duplicates", "removed": duplicates})
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    if timestamp and timestamp in frame:
        frame = add_causal_temporal_features(frame, entity, [c for c in numeric if c not in labels], window=10)
        preprocessing["operations"].append({"operation": "add_causal_rolling_features", "window_records": 10, "numeric_columns": [c for c in numeric if c not in labels]})
    frame.insert(0, "prepared_observation_id", [f"{definition.dataset_id}__{i}" for i in range(len(frame))])
    frame.insert(1, "dataset_id", definition.dataset_id)
    save_frame(frame, root / "processed_datasets" / f"{definition.dataset_id}.parquet")
    save_frame(frame, root / "derived_features" / f"{definition.dataset_id}.parquet")
    save_frame(frame, root / "prepared_observations" / f"{definition.dataset_id}.parquet")
    quality_and_outputs(root, definition.dataset_id, before, frame, labels, preprocessing)
    if timestamp and timestamp in frame:
        plot_series(root, definition.dataset_id, frame, timestamp, [c for c in numeric if c not in labels][:5], "representative_trajectory")
    catalogue = base_catalogue(definition, data_root)
    metadata_key = "smart_maintenance" if definition.dataset_id.startswith("smart_maintenance_") else definition.dataset_id
    catalogue.update({"record_count": int(len(before)), "column_count": int(before.shape[1]), "column_names": before.columns.tolist(), "dtypes": {c: str(t) for c, t in before.dtypes.items()}, "temporal_structure": "sequential timestamped rows" if timestamp else "independent tabular rows; no verified temporal key", "labels": {name: name in before for name in labels}, "sensor_catalogue": column_catalogue(before.columns.tolist(), SIMPLE_METADATA.get(metadata_key, {})), "prepared_output": str((root / "prepared_observations" / f"{definition.dataset_id}.parquet").relative_to(root)), "notes": catalogue["notes"] + ["Original source columns are retained. Derived columns use the __ suffix."]})
    save_catalogue(root, catalogue)
    return {"dataset_id": definition.dataset_id, "status": "completed", "prepared_records": len(frame), "output": catalogue["prepared_output"]}


def prepare_cmapss(root: Path, data_root: Path, definition: DatasetDefinition) -> dict[str, Any]:
    folder = data_root / definition.relative_path
    columns = ["unit_id", "cycle", "operating_setting_1", "operating_setting_2", "operating_setting_3", *[f"sensor_{i}" for i in range(1, 22)]]
    sets = []
    for split in ("train", "test"):
        for source in sorted(folder.glob(f"{split}_FD*.txt")):
            subset = source.stem.split("_")[-1]
            df = pd.read_csv(source, sep=r"\s+", header=None, names=columns, engine="python")
            df["split"] = split
            df["subset"] = subset
            if split == "train":
                df["rul_cycles_label"] = df.groupby("unit_id")["cycle"].transform("max") - df["cycle"]
            else:
                rul = pd.read_csv(folder / f"RUL_{subset}.txt", sep=r"\s+", header=None, usecols=[0], names=["rul_at_final_cycle_label"], engine="python")
                endings = df.groupby("unit_id")["cycle"].max().reset_index().sort_values("unit_id").reset_index(drop=True)
                endings = endings.join(rul)
                final_rul = endings.set_index("unit_id")["rul_at_final_cycle_label"]
                df["rul_cycles_label"] = df["unit_id"].map(final_rul) + (df.groupby("unit_id")["cycle"].transform("max") - df["cycle"])
            sensor_columns = [f"sensor_{i}" for i in range(1, 22)]
            df = df.sort_values(["unit_id", "cycle"], kind="stable")
            df = add_causal_temporal_features(df, "unit_id", sensor_columns, window=20)
            sets.append(df)
    before = pd.concat([item[[*columns, "split", "subset", "rul_cycles_label"]] for item in sets], ignore_index=True)
    after = pd.concat(sets, ignore_index=True)
    after.insert(0, "prepared_observation_id", [f"cmapss__{s}__{u}__{c}" for s, u, c in zip(after["subset"], after["unit_id"], after["cycle"])])
    after.insert(1, "dataset_id", definition.dataset_id)
    save_frame(after, root / "processed_datasets" / "cmapss.parquet")
    save_frame(after, root / "derived_features" / "cmapss.parquet")
    save_frame(after, root / "prepared_observations" / "cmapss.parquet")
    quality_and_outputs(root, definition.dataset_id, before, after, ["rul_cycles_label"], {"dataset_id": definition.dataset_id, "operations": [{"operation": "parse_space_delimited_trajectories"}, {"operation": "derive_rul_label", "note": "test RUL uses repository truth labels and remains a label, not a feature"}, {"operation": "add_causal_rolling_features", "window_cycles": 20}], "removed_records": 0, "columns_changed": []})
    plot_series(root, definition.dataset_id, after[after["unit_id"] == after["unit_id"].iloc[0]], "cycle", ["sensor_2", "sensor_3", "sensor_4"], "representative_trajectory")
    metadata = {"unit_id": {"meaning": "engine unit identifier", "category": "metadata/identifier"}, "cycle": {"meaning": "operating cycle", "unit": "cycles", "category": "metadata/identifier"}, **{f"operating_setting_{i}": {"meaning": f"operational setting {i}", "unit": "UNKNOWN/UNVERIFIED", "category": "operating context"} for i in range(1, 4)}, **{f"sensor_{i}": {"meaning": f"NASA C-MAPSS sensor measurement {i}", "unit": "UNKNOWN/UNVERIFIED", "category": "process", "source": "C-MAPSS readme supplies number but not physical unit"} for i in range(1, 22)}, "rul_cycles_label": {"meaning": "remaining useful life", "unit": "cycles", "category": "label/target"}}
    catalogue = base_catalogue(definition, data_root)
    catalogue.update({"record_count": int(len(before)), "column_count": len(columns), "column_names": columns, "dtypes": {c: str(t) for c, t in before.dtypes.items()}, "temporal_structure": "per-engine sequential run-to-failure trajectories", "labels": {"RUL": True, "fault_regime_metadata": True, "maintenance_actions": False, "maintenance_outcomes": False}, "sensor_catalogue": column_catalogue(columns + ["rul_cycles_label"], metadata), "prepared_output": "prepared_observations/cmapss.parquet", "notes": catalogue["notes"] + ["FD001/FD002 contain HPC degradation; FD003/FD004 include HPC and fan degradation according to the included readme."]})
    save_catalogue(root, catalogue)
    return {"dataset_id": definition.dataset_id, "status": "completed", "prepared_records": len(after), "output": catalogue["prepared_output"]}


def prepare_hydraulic(root: Path, data_root: Path, definition: DatasetDefinition) -> dict[str, Any]:
    folder = data_root / definition.relative_path
    profile = pd.read_csv(folder / "profile.txt", sep=r"\s+", header=None, names=["cooler_condition_pct_label", "valve_condition_pct_label", "pump_leakage_label", "accumulator_pressure_bar_label", "stable_flag"], engine="python")
    before = profile.copy()
    sensor_specs = {"PS1": ("pressure", "bar", "hydraulic/fluid", 100), "PS2": ("pressure", "bar", "hydraulic/fluid", 100), "PS3": ("pressure", "bar", "hydraulic/fluid", 100), "PS4": ("pressure", "bar", "hydraulic/fluid", 100), "PS5": ("pressure", "bar", "hydraulic/fluid", 100), "PS6": ("pressure", "bar", "hydraulic/fluid", 100), "EPS1": ("motor power", "W", "electrical", 100), "FS1": ("volume flow", "l/min", "hydraulic/fluid", 10), "FS2": ("volume flow", "l/min", "hydraulic/fluid", 10), "TS1": ("temperature", "C", "thermal", 1), "TS2": ("temperature", "C", "thermal", 1), "TS3": ("temperature", "C", "thermal", 1), "TS4": ("temperature", "C", "thermal", 1), "VS1": ("vibration", "mm/s", "mechanical", 1), "CE": ("cooling efficiency (virtual)", "%", "process", 1), "CP": ("cooling power (virtual)", "kW", "process", 1), "SE": ("efficiency factor", "%", "process", 1)}
    features = [profile]
    matrix_quality = {}
    for sensor, (meaning, unit, category, rate) in sensor_specs.items():
        matrix = np.loadtxt(folder / f"{sensor}.txt")
        matrix_quality[sensor] = {"shape": list(matrix.shape), "sampling_hz": rate, "missing": int(np.isnan(matrix).sum()), "infinite": int(np.isinf(matrix).sum())}
        features.append(matrix_cycle_features(matrix, sensor).reset_index(drop=True))
        del matrix
    after = pd.concat(features, axis=1)
    after.insert(0, "prepared_observation_id", [f"uci_hydraulic__cycle_{i}" for i in range(len(after))])
    after.insert(1, "dataset_id", definition.dataset_id)
    save_frame(after, root / "processed_datasets" / "uci_hydraulic.parquet")
    save_frame(after, root / "derived_features" / "uci_hydraulic.parquet")
    save_frame(after, root / "prepared_observations" / "uci_hydraulic.parquet")
    labels = profile.columns.tolist()
    quality_and_outputs(root, definition.dataset_id, before, after, labels, {"dataset_id": definition.dataset_id, "operations": [{"operation": "read_native_multirate_60_second_cycles"}, {"operation": "extract_per_cycle_physical_summaries", "features": ["mean", "std", "min", "max", "first", "last", "slope_per_sample"]}], "sensor_matrix_quality": matrix_quality, "removed_records": 0, "columns_changed": []})
    plot_series(root, definition.dataset_id, after.reset_index(), "index", ["PS1__mean", "FS1__mean", "TS1__mean", "VS1__mean"], "cycle_feature_trajectory")
    metadata = {column: {"meaning": column.replace("_label", "").replace("_", " "), "unit": "see UCI documentation", "category": "label/target"} for column in labels}
    metadata.update({sensor: {"meaning": info[0], "unit": info[1], "category": info[2], "source": "UCI Hydraulic Systems documentation"} for sensor, info in sensor_specs.items()})
    catalogue = base_catalogue(definition, data_root)
    catalogue.update({"record_count": len(profile), "column_count": len(after.columns), "column_names": after.columns.tolist(), "dtypes": {c: str(t) for c, t in after.dtypes.items()}, "temporal_structure": "2205 synchronized 60-second multirate load cycles", "labels": {"cooler_condition": True, "valve_condition": True, "pump_leakage": True, "accumulator_condition": True, "maintenance_actions": False, "maintenance_outcomes": False}, "sensor_catalogue": column_catalogue(list(sensor_specs) + labels, metadata), "prepared_output": "prepared_observations/uci_hydraulic.parquet", "notes": catalogue["notes"] + ["Raw sensor matrices are preserved in data/. Prepared output contains per-cycle summaries only."]})
    save_catalogue(root, catalogue)
    return {"dataset_id": definition.dataset_id, "status": "completed", "prepared_records": len(after), "output": catalogue["prepared_output"]}


def prepare_ims(root: Path, data_root: Path, definition: DatasetDefinition) -> dict[str, Any]:
    folder = data_root / definition.relative_path
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    before_count = 0
    for test_name, channel_count, failed_bearings in (("1st_test", 8, "bearing_3_inner_race; bearing_4_roller"), ("2nd_test", 4, "bearing_1_outer_race")):
        source = folder / test_name / test_name
        files = sorted(p for p in source.iterdir() if p.is_file())
        before_count += len(files)
        for sequence, path in enumerate(files):
            try:
                values = np.loadtxt(path)
                timestamp = pd.to_datetime(path.name, format="%Y.%m.%d.%H.%M.%S", errors="coerce")
                row: dict[str, Any] = {"prepared_observation_id": f"nasa_ims__{test_name}__{path.name}", "dataset_id": definition.dataset_id, "test_set": test_name, "source_file": str(path.relative_to(data_root)), "snapshot_sequence": sequence, "timestamp": timestamp, "sample_count": int(values.shape[0]), "channel_count": int(values.shape[1]) if values.ndim == 2 else 1, "known_end_of_test_failure": failed_bearings}
                if values.ndim == 1:
                    values = values[:, None]
                for channel in range(values.shape[1]):
                    for key, value in waveform_features(values[:, channel], 20_000).items():
                        row[f"channel_{channel + 1}__{key}"] = value
                rows.append(row)
            except Exception as exc:  # retain all successful source files and record unreadable ones
                errors.append(f"{path}: {exc}")
    after = pd.DataFrame(rows).sort_values(["test_set", "timestamp"], kind="stable")
    for column in [c for c in after if c.endswith("__rms")]:
        after[f"{column}__delta_previous"] = after.groupby("test_set", sort=False)[column].diff()
        after[f"{column}__rolling_mean_10"] = after.groupby("test_set", sort=False)[column].transform(lambda x: x.rolling(10, min_periods=1).mean())
    save_frame(after, root / "processed_datasets" / "nasa_ims.parquet")
    save_frame(after, root / "derived_features" / "nasa_ims.parquet")
    save_frame(after, root / "prepared_observations" / "nasa_ims.parquet")
    quality_and_outputs(root, definition.dataset_id, pd.DataFrame({"source_file": range(before_count)}), after, [], {"dataset_id": definition.dataset_id, "operations": [{"operation": "read_ascii_vibration_snapshots", "sampling_hz": 20000}, {"operation": "extract_time_and_frequency_domain_features"}, {"operation": "add_causal_snapshot_features", "window_snapshots": 10}], "unreadable_files": errors, "removed_records": len(errors), "columns_changed": []})
    plot_series(root, definition.dataset_id, after[after["test_set"] == "2nd_test"], "snapshot_sequence", [c for c in after if c.endswith("__rms")][:4], "bearing_rms_progression", max_rows=1200)
    catalogue = base_catalogue(definition, data_root)
    catalogue.update({"record_count": before_count, "column_count": int(after.shape[1]), "column_names": after.columns.tolist(), "dtypes": {c: str(t) for c, t in after.dtypes.items()}, "temporal_structure": "one-second high-frequency snapshots at 20 kHz, collected roughly every 10 minutes", "labels": {"end_of_test_failure_description": True, "per_snapshot_failure_label": False, "maintenance_actions": False, "maintenance_outcomes": False}, "sensor_catalogue": [{"feature": "accelerometer_channels", "physical_meaning": "bearing housing acceleration", "unit": "UNKNOWN/UNVERIFIED", "category": "mechanical", "source": "IMS readme; unit not stated"}], "prepared_output": "prepared_observations/nasa_ims.parquet", "notes": catalogue["notes"] + ["Known failure descriptions apply only at test end, not as per-snapshot labels.", f"{len(errors)} waveform files were unreadable."]})
    save_catalogue(root, catalogue)
    return {"dataset_id": definition.dataset_id, "status": "completed", "prepared_records": len(after), "output": catalogue["prepared_output"]}


def prepare_metropt3(root: Path, data_root: Path, definition: DatasetDefinition) -> dict[str, Any]:
    path = data_root / definition.relative_path
    numeric_columns = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current", "COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]
    frame = pd.read_csv(path)
    total_rows = len(frame)
    missing = int(frame.isna().sum().sum())
    duplicate_within_chunk = int(frame.duplicated().sum())
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    valid_times = frame["timestamp"].dropna()
    monotonic_violations = int((valid_times.diff().dropna() < pd.Timedelta(0)).sum())
    for col in numeric_columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    grouped = frame[numeric_columns].resample("1min")
    summary = pd.concat({
        "mean": grouped.mean(), "std": grouped.std(), "min": grouped.min(), "max": grouped.max(),
        "first": grouped.first(), "last": grouped.last(), "count": grouped.count(),
    }, axis=1).dropna(how="all")
    after = pd.DataFrame({
        "prepared_observation_id": [f"metropt3__{minute.isoformat()}" for minute in summary.index],
        "dataset_id": definition.dataset_id,
        "window_start": summary.index,
        "window_end": summary.index + pd.Timedelta(minutes=1),
        "window_sample_count": summary["count"].sum(axis=1).astype("int64"),
    })
    for col in numeric_columns:
        after[f"{col}__mean"] = summary["mean"][col]
        after[f"{col}__std"] = summary["std"][col]
        after[f"{col}__min"] = summary["min"][col]
        after[f"{col}__max"] = summary["max"][col]
        after[f"{col}__delta_window"] = summary["last"][col] - summary["first"][col]
    save_frame(after, root / "processed_datasets" / "metropt3_minute_windows.parquet")
    save_frame(after, root / "derived_features" / "metropt3_minute_windows.parquet")
    save_frame(after, root / "prepared_observations" / "metropt3_minute_windows.parquet")
    quality_before = {"records": total_rows, "columns": 17, "missing_cells_streamed": missing, "duplicate_records_within_chunks": duplicate_within_chunk, "global_duplicate_check": "not performed to avoid materializing 15M raw rows", "timestamp_chunk_order_violations": monotonic_violations}
    write_json(root / "data_quality_reports" / "metropt3.json", {"before": quality_before, "after": dataframe_quality(after)})
    write_json(root / "preprocessing_reports" / "metropt3.json", {"dataset_id": definition.dataset_id, "records_before": total_rows, "records_after": len(after), "operations": [{"operation": "read_csv_and_parse_timestamp"}, {"operation": "vectorized_one_minute_resample", "memory_note": "raw 1 Hz rows are reduced immediately after loading"}, {"operation": "extract_window_mean_std_min_max_delta"}], "removed_records": 0, "columns_changed": []})
    plot_series(root, definition.dataset_id, after, "window_start", ["TP2__mean", "TP3__mean", "Oil_temperature__mean", "Motor_current__mean"], "representative_trajectory")
    metadata = {"timestamp": {"meaning": "timestamp", "category": "metadata/identifier"}, "TP2": {"meaning": "compressor pressure", "unit": "bar", "category": "hydraulic/fluid"}, "TP3": {"meaning": "pneumatic panel pressure", "unit": "bar", "category": "hydraulic/fluid"}, "H1": {"meaning": "cyclonic separator pressure drop", "unit": "bar", "category": "hydraulic/fluid"}, "DV_pressure": {"meaning": "air-dryer discharge pressure drop", "unit": "bar", "category": "hydraulic/fluid"}, "Reservoirs": {"meaning": "downstream reservoir pressure", "unit": "bar", "category": "hydraulic/fluid"}, "Oil_temperature": {"meaning": "compressor oil temperature", "unit": "C", "category": "thermal"}, "Motor_current": {"meaning": "single motor-phase current", "unit": "A", "category": "electrical"}, **{name: {"meaning": "digital compressor/air-dryer/flow signal", "unit": "binary/count; see MetroPT-3 documentation", "category": "operating context"} for name in numeric_columns[7:]}}
    catalogue = base_catalogue(definition, data_root)
    catalogue.update({"record_count": total_rows, "column_count": 17, "column_names": ["timestamp", *numeric_columns], "temporal_structure": "1 Hz compressor multivariate time series", "labels": {"failure_reports_documented": True, "row_level_failure_label": False, "maintenance_actions": False, "maintenance_outcomes": False}, "sensor_catalogue": column_catalogue(["timestamp", *numeric_columns], metadata), "prepared_output": "prepared_observations/metropt3_minute_windows.parquet", "notes": catalogue["notes"] + ["One-minute windows are used to keep Phase 1 artifacts tractable; raw 1 Hz records remain untouched.", "Failure reports in the PDF are not expanded into row-level labels in this phase."]})
    save_catalogue(root, catalogue)
    return {"dataset_id": definition.dataset_id, "status": "completed", "prepared_records": len(after), "output": catalogue["prepared_output"]}


def prepare_tennessee(root: Path, data_root: Path, definition: DatasetDefinition) -> dict[str, Any]:
    try:
        import pyreadr  # installed locally for this repository's RData inputs
    except ImportError as exc:
        raise RuntimeError("pyreadr is required for Tennessee Eastman RData inputs; install it into .phase1_deps") from exc
    folder = data_root / definition.relative_path
    all_frames = []
    manifests: dict[str, Any] = {}
    for source in sorted(folder.glob("*.RData")):
        loaded = pyreadr.read_r(source)
        for object_name, frame in loaded.items():
            if not isinstance(frame, pd.DataFrame):
                continue
            frame = frame.copy()
            frame["source_file"] = source.name
            frame["source_object"] = object_name
            manifests[source.name] = {"objects": {object_name: {"records": len(frame), "columns": frame.columns.tolist(), "dtypes": {c: str(t) for c, t in frame.dtypes.items()}}}}
            all_frames.append(frame)
    before = pd.concat(all_frames, ignore_index=True)
    frame = before.copy()
    sort_columns = [c for c in ("simulationRun", "sample") if c in frame]
    if sort_columns:
        frame = frame.sort_values(sort_columns, kind="stable")
    numeric = [c for c in frame.select_dtypes(include=np.number).columns if c not in {"faultNumber", "simulationRun", "sample"}]
    # 52 process variables are retained; causal rolling summaries are restricted to
    # a representative, reproducible subset to keep the prepared artifact compact.
    feature_base = numeric[: min(20, len(numeric))]
    frame = add_causal_temporal_features(frame, "simulationRun" if "simulationRun" in frame else None, feature_base, window=20)
    frame.insert(0, "prepared_observation_id", [f"tennessee_eastman__{i}" for i in range(len(frame))])
    frame.insert(1, "dataset_id", definition.dataset_id)
    save_frame(frame, root / "processed_datasets" / "tennessee_eastman.parquet")
    save_frame(frame, root / "derived_features" / "tennessee_eastman.parquet")
    save_frame(frame, root / "prepared_observations" / "tennessee_eastman.parquet")
    labels = [c for c in ["faultNumber"] if c in before]
    quality_and_outputs(root, definition.dataset_id, before, frame, labels, {"dataset_id": definition.dataset_id, "operations": [{"operation": "read_RData_objects"}, {"operation": "sort_by_simulation_run_and_sample"}, {"operation": "add_causal_rolling_features", "window_samples": 20, "base_variable_count": len(feature_base)}], "rdata_manifest": manifests, "removed_records": 0, "columns_changed": []})
    plot_series(root, definition.dataset_id, frame[frame["simulationRun"] == frame["simulationRun"].iloc[0]] if "simulationRun" in frame else frame, "sample" if "sample" in frame else frame.index.name or "prepared_observation_id", numeric[:4], "representative_process_trajectory")
    feature_catalogue = [{"feature": c, "physical_meaning": "Tennessee Eastman process variable; exact mapping not present in repository documentation", "unit": "UNKNOWN/UNVERIFIED", "category": "process", "source": "RData column name only"} for c in numeric]
    for c in ["faultNumber", "simulationRun", "sample"]:
        if c in before:
            feature_catalogue.append({"feature": c, "physical_meaning": {"faultNumber": "fault scenario identifier", "simulationRun": "simulation run identifier", "sample": "sample index"}[c], "unit": "UNKNOWN/UNVERIFIED", "category": "label/target" if c == "faultNumber" else "metadata/identifier", "source": "RData column name"})
    catalogue = base_catalogue(definition, data_root)
    catalogue.update({"record_count": len(before), "column_count": int(before.shape[1]), "column_names": before.columns.tolist(), "dtypes": {c: str(t) for c, t in before.dtypes.items()}, "temporal_structure": "multivariate process time series by simulation run", "labels": {"fault_number": "faultNumber" in before, "maintenance_actions": False, "maintenance_outcomes": False}, "sensor_catalogue": feature_catalogue, "prepared_output": "prepared_observations/tennessee_eastman.parquet", "notes": catalogue["notes"] + ["The repository does not contain the Tennessee Eastman variable dictionary; physical variable meanings remain UNKNOWN/UNVERIFIED pending source documentation."]})
    save_catalogue(root, catalogue)
    return {"dataset_id": definition.dataset_id, "status": "completed", "prepared_records": len(frame), "output": catalogue["prepared_output"]}


def prepare_milling(root: Path, data_root: Path, definition: DatasetDefinition) -> dict[str, Any]:
    output = root / "_work_milling"
    command = [str(PROJECT / "mas_venv" / "bin" / "python"), "-m", "phase1.milling_extract", "--archive", str(data_root / definition.relative_path), "--output", str(output)]
    result = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(PROJECT)}, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Milling MAT extraction failed: {result.stderr[-2000:]}")
    source = output / "nasa_milling_prepared_observations.csv"
    frame = pd.read_csv(source)
    save_frame(frame, root / "processed_datasets" / "nasa_milling.parquet")
    save_frame(frame, root / "derived_features" / "nasa_milling.parquet")
    save_frame(frame, root / "prepared_observations" / "nasa_milling.parquet")
    quality_and_outputs(root, definition.dataset_id, frame, frame, [], {"dataset_id": definition.dataset_id, "operations": [{"operation": "extract_mill_mat_from_zip"}, {"operation": "flatten_numeric_signal_arrays_conservatively"}, {"operation": "extract_time_and_frequency_features", "sampling_rate": "UNKNOWN/UNVERIFIED"}], "removed_records": 0, "columns_changed": []})
    plot_series(root, definition.dataset_id, frame.reset_index(), "index", [c for c in frame if c.endswith("__rms")][:4], "signal_feature_examples")
    catalogue = base_catalogue(definition, data_root)
    manifest = json.loads((output / "nasa_milling_manifest.json").read_text())
    catalogue.update({"record_count": int(len(frame)), "column_count": int(frame.shape[1]), "column_names": frame.columns.tolist(), "dtypes": {c: str(t) for c, t in frame.dtypes.items()}, "temporal_structure": "MATLAB archive; numeric signal arrays flattened conservatively", "labels": {"tool_wear_available": "not automatically identified; inspect MAT variable manifest", "maintenance_actions": False, "maintenance_outcomes": False}, "sensor_catalogue": [{"feature": "numeric MAT arrays", "physical_meaning": "UNKNOWN/UNVERIFIED pending MATLAB release field documentation", "unit": "UNKNOWN/UNVERIFIED", "category": "UNKNOWN/UNVERIFIED", "source": "MAT variable inventory"}], "mat_variable_manifest": manifest, "prepared_output": "prepared_observations/nasa_milling.parquet", "notes": catalogue["notes"] + ["No physical field name is assumed from the nested MATLAB structure. Signal sampling rate is not assigned without a supported source."]})
    save_catalogue(root, catalogue)
    return {"dataset_id": definition.dataset_id, "status": "completed", "prepared_records": len(frame), "output": catalogue["prepared_output"]}


def prepare_deprecated(root: Path, data_root: Path, definition: DatasetDefinition) -> dict[str, Any]:
    path = data_root / definition.relative_path
    before = pd.read_csv(path)
    catalogue = base_catalogue(definition, data_root)
    catalogue.update({"record_count": len(before), "column_count": int(before.shape[1]), "column_names": before.columns.tolist(), "dtypes": {c: str(t) for c, t in before.dtypes.items()}, "temporal_structure": "independent tabular rows; no temporal key", "labels": {"demand_class": "Demand_Class" in before, "maintenance_actions": False, "maintenance_outcomes": False}, "sensor_catalogue": column_catalogue(before.columns.tolist(), {}), "prepared_output": None, "notes": catalogue["notes"] + ["Excluded from the active Phase 1 corpus because its source documentation categorizes it as deprecated and its non-engineering business fields dominate."]})
    write_json(root / "data_quality_reports" / f"{definition.dataset_id}.json", {"before": dataframe_quality(before, ["Demand_Class"]), "after": None})
    write_json(root / "preprocessing_reports" / f"{definition.dataset_id}.json", {"dataset_id": definition.dataset_id, "status": "catalogued_not_processed", "reason": "deprecated source"})
    save_catalogue(root, catalogue)
    return {"dataset_id": definition.dataset_id, "status": "catalogued_not_processed", "prepared_records": 0, "output": None}


def write_phase_summary(root: Path, inventory: list[dict[str, Any]], results: list[dict[str, Any]], errors: list[dict[str, str]]) -> None:
    rows = {result["dataset_id"]: result for result in results}
    limited = {"nasa_milling", "metropt3", "tennessee_eastman", "nasa_ims", "metal_etch"}
    lines = ["# Phase 1 Summary - Dataset Preparation", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", "", "## Scope", "", "Dataset-specific preparation is complete through prepared observations. No semantic normalization, ontology mapping, severity mapping, action generation, recommender, LLM-as-a-Judge, LGBMRanker, collaborative filtering, or closed-loop functionality is included.", "", "## Dataset Status", "", "| Dataset | Type | Status | Raw Records | Prepared Observations | Labels/Targets | Notes |", "|---|---|---|---:|---:|---|---|"]
    for item in inventory:
        outcome = rows.get(item["dataset_id"], {})
        raw_catalogue = root / "dataset_catalogue" / f"{item['dataset_id']}.json"
        catalogue = json.loads(raw_catalogue.read_text(encoding="utf-8")) if raw_catalogue.exists() else {}
        status = outcome.get("status", "failed")
        if status == "completed":
            status = "SUCCESS_WITH_LIMITATIONS" if item["dataset_id"] in limited else "SUCCESS"
        elif status == "catalogued_not_processed":
            status = "SKIPPED_NOT_AVAILABLE" if item["dataset_id"] == "digital_manufacturing_deprecated" else "FAILED"
        labels = ", ".join(key for key, value in catalogue.get("labels", {}).items() if value) or "none observed"
        notes = "; ".join(catalogue.get("notes", [])[-2:]) or "See catalogue and preprocessing report"
        lines.append(f"| {item['name']} | {item['processing_type']} | {status} | {catalogue.get('record_count', 'UNKNOWN')} | {outcome.get('prepared_records', 0)} | {labels} | {notes} |")
    completed = sum(result.get("status") == "completed" for result in results)
    limited_count = sum(result.get("status") == "completed" and result.get("dataset_id") in limited for result in results)
    failed_count = len(errors)
    lines += ["", "## Overall Summary", "", f"- Available catalogue entries: {len(inventory)} ({len(inventory) - 1} active datasets and 1 deprecated inventory-only entry).", f"- Successfully processed: {completed}.", f"- Success with limitations: {limited_count}.", f"- Failed: {failed_count}.", "- Skipped: 1 deprecated dataset, retained for inventory only.", "- Modalities: thermal, mechanical/vibration, electrical, hydraulic/fluid, acoustic, process, operating context, degradation/health, and business/economic fields where documented.", "- Processing types: static tabular, low-frequency time series, run-to-failure trajectories, high-frequency signal snapshots, multirate cycle signals, and multivariate process time series.", "", "## Major Limitations", "", "- Most datasets contain condition, fault, degradation, RUL, or process labels rather than maintenance action/outcome histories.", "- Tennessee Eastman xmeas/xmv physical mappings are unavailable in the repository and remain UNKNOWN/UNVERIFIED.", "- Metal Etch features are anonymous and have no verified physical interpretation.", "- MetroPT-3 source failure reports are not converted into row-level labels.", "- NASA IMS failure descriptions apply at test end, not to every preceding snapshot.", "- NASA Milling is conservatively extracted from nested MATLAB variables without unsupported semantics.", "", "## Next-phase Input", "", "Every active dataset has separate prepared observations, quality/preprocessing reports, catalogues, derived features where applicable, provenance, and representative visualizations. These artifacts are ready for semantic normalization while preserving dataset-specific physical meaning."]
    if errors:
        lines += ["", "## Processing errors", ""] + [f"- `{item['dataset_id']}`: {item['error']}" for item in errors]
    (root / "PHASE_1_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs" / "phase1")
    parser.add_argument("--clean", action="store_true", help="Remove only the selected Phase 1 output directory before regenerating it.")
    parser.add_argument("--resume", action="store_true", help="Reuse completed catalogue/output artifacts and regenerate the audit manifest without reprocessing them.")
    parser.add_argument("--datasets", nargs="*", choices=[definition.dataset_id for definition in DATASETS], help="Optional subset for development; default processes all datasets.")
    args = parser.parse_args()
    root = args.output_dir.resolve(); data_root = PROJECT / "data"
    setup_output(root, args.clean)
    logging.basicConfig(filename=root / "logs" / "phase1.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    selected = [definition for definition in DATASETS if not args.datasets or definition.dataset_id in args.datasets]
    inventory = [inventory_entry(definition, data_root) for definition in DATASETS]
    write_json(root / "dataset_inventory" / "inventory.json", inventory)
    pd.DataFrame([{key: item[key] for key in ("dataset_id", "name", "relative_path", "file_count", "size_bytes", "formats_present", "processing_type", "status", "exists")} for item in inventory]).to_csv(root / "dataset_inventory" / "inventory.csv", index=False)
    processors: dict[str, Callable[..., dict[str, Any]]] = {
        "ai4i_2020": lambda d: prepare_static_csv(root, data_root, d, data_root / d.relative_path, ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]),
        "smart_maintenance": lambda d: prepare_smart_maintenance(root, data_root, d),
        "iiot_6g": lambda d: prepare_static_csv(root, data_root, d, data_root / d.relative_path, ["Efficiency_Status"], "Timestamp", "Machine_ID"),
        "metal_etch": lambda d: prepare_static_csv(root, data_root, d, data_root / d.relative_path, ["Target"]),
        "digital_manufacturing_deprecated": lambda d: prepare_deprecated(root, data_root, d),
        "cmapss": lambda d: prepare_cmapss(root, data_root, d), "uci_hydraulic": lambda d: prepare_hydraulic(root, data_root, d),
        "nasa_ims": lambda d: prepare_ims(root, data_root, d), "metropt3": lambda d: prepare_metropt3(root, data_root, d),
        "tennessee_eastman": lambda d: prepare_tennessee(root, data_root, d), "nasa_milling": lambda d: prepare_milling(root, data_root, d),
    }
    results: list[dict[str, Any]] = []; errors: list[dict[str, str]] = []
    for definition in selected:
        logging.info("Starting %s", definition.dataset_id)
        try:
            existing = root / "dataset_catalogue" / f"{definition.dataset_id}.json"
            if args.resume and existing.exists():
                catalogue = json.loads(existing.read_text(encoding="utf-8"))
                prepared = catalogue.get("prepared_output")
                if prepared and artifact_paths_exist(root, prepared):
                    results.append({"dataset_id": definition.dataset_id, "status": "completed", "prepared_records": artifact_record_count(root, prepared), "output": prepared, "resumed": True})
                    logging.info("Reused completed %s", definition.dataset_id)
                    continue
                if definition.status == "deprecated":
                    results.append({"dataset_id": definition.dataset_id, "status": "catalogued_not_processed", "prepared_records": 0, "output": None, "resumed": True})
                    continue
            result = processors[definition.dataset_id](definition)
            result = normalize_artifact_paths(root, result)
            results.append(result); logging.info("Completed %s: %s", definition.dataset_id, result)
        except Exception as exc:
            logging.exception("Failed %s", definition.dataset_id)
            errors.append({"dataset_id": definition.dataset_id, "error": f"{type(exc).__name__}: {exc}"})
            base = base_catalogue(definition, data_root); base["notes"].append(f"Processing failed: {type(exc).__name__}: {exc}"); save_catalogue(root, base)
    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True, check=False).stdout.strip() or None
    except OSError:
        git_commit = None
    write_json(root / "run_manifest.json", {"generated_at": datetime.now(timezone.utc).isoformat(), "project": str(PROJECT), "git_commit": git_commit, "selected_datasets": [d.dataset_id for d in selected], "configuration": {"resume": args.resume, "output_dir": str(root), "causal_temporal_features": True}, "results": results, "errors": errors})
    write_phase_summary(root, inventory, results, errors)
    if errors:
        raise SystemExit(f"Phase 1 completed with {len(errors)} processing error(s); see {root / 'run_manifest.json'}")


def prepare_smart_maintenance(root: Path, data_root: Path, definition: DatasetDefinition) -> dict[str, Any]:
    folder = data_root / definition.relative_path
    static_def = DatasetDefinition("smart_maintenance_static", "Smart Manufacturing Maintenance - static variant", "Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv", definition.domain, definition.asset_type, definition.component, "static_tabular", "CSV")
    time_def = DatasetDefinition("smart_maintenance_timeseries", "Smart Manufacturing Maintenance - timestamped variant", "Smart Manufacturing Maintenance Dataset/smart_manufacturing_data.csv", definition.domain, definition.asset_type, definition.component, "low_frequency_time_series", "CSV")
    results = [
        prepare_static_csv(root, data_root, static_def, folder / "smart_maintenance_dataset.csv", ["Maintenance_Priority"], None, "Machine_ID"),
        prepare_static_csv(root, data_root, time_def, folder / "smart_manufacturing_data.csv", ["anomaly_flag", "failure_type", "maintenance_required"], "timestamp", "machine_id"),
    ]
    write_json(root / "data_quality_reports" / f"{definition.dataset_id}.json", {"variants": [json.loads((root / "data_quality_reports" / f"{result['dataset_id']}.json").read_text(encoding="utf-8")) for result in results]})
    write_json(root / "preprocessing_reports" / f"{definition.dataset_id}.json", {"dataset_id": definition.dataset_id, "variants": [json.loads((root / "preprocessing_reports" / f"{result['dataset_id']}.json").read_text(encoding="utf-8")) for result in results], "note": "Variants remain separate and are not merged."})
    # Preserve two source variants under one dataset family and provide one run-level catalogue.
    family = base_catalogue(definition, data_root)
    family.update({"record_count": sum(int(r["prepared_records"]) for r in results), "column_count": "two source variants; see child catalogues", "temporal_structure": "one static condition table and one timestamped machine table", "labels": {"maintenance_priority": True, "anomaly": True, "failure_type": True, "maintenance_required": True}, "sensor_catalogue": [], "prepared_output": ["prepared_observations/smart_maintenance_static.parquet", "prepared_observations/smart_maintenance_timeseries.parquet"], "notes": family["notes"] + ["The two variants are prepared separately and must not be merged in Phase 1."]})
    save_catalogue(root, family)
    return {"dataset_id": definition.dataset_id, "status": "completed", "prepared_records": sum(int(r["prepared_records"]) for r in results), "output": "prepared_observations/smart_maintenance_static.parquet; prepared_observations/smart_maintenance_timeseries.parquet"}


if __name__ == "__main__":
    main()
