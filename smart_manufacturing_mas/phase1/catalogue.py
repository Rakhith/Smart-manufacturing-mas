"""Dataset definitions and reusable profiling helpers for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetDefinition:
    dataset_id: str
    name: str
    relative_path: str
    domain: str
    asset_type: str
    component: str
    processing_type: str
    format: str
    status: str = "active"
    notes: str = ""


DATASETS = (
    DatasetDefinition("ai4i_2020", "AI4I 2020", "ai4i2020.csv", "CNC machining", "CNC machine", "tooling", "static_tabular", "CSV"),
    DatasetDefinition("cmapss", "NASA C-MAPSS", "CMAPSSData", "aerospace", "turbofan engine", "HPC/fan", "run_to_failure", "space-delimited text"),
    DatasetDefinition("smart_maintenance", "Smart Manufacturing Maintenance", "Smart Manufacturing Maintenance Dataset", "factory maintenance", "machine fleet", "unknown", "mixed_tabular_time_series", "CSV"),
    DatasetDefinition("iiot_6g", "Intelligent Manufacturing 6G", "Intelligent Manufacturing Dataset/manufacturing_6G_dataset.csv", "smart factory / IIoT", "machine fleet", "unknown", "low_frequency_time_series", "CSV"),
    DatasetDefinition("nasa_milling", "NASA Milling Tool Wear", "NASA Milling Tool Wear/3. Milling/mill.zip", "CNC metal cutting", "milling machine", "cutting tool", "run_to_failure_signal", "ZIP/MATLAB"),
    DatasetDefinition("metropt3", "UCI MetroPT-3", "UCI MetroPT-3/MetroPT3(AirCompressor).csv", "railway pneumatics", "air production unit", "compressor", "low_frequency_time_series", "CSV"),
    DatasetDefinition("uci_hydraulic", "UCI Hydraulic Systems", "UCI Hydraulic Systems", "industrial hydraulics", "hydraulic test rig", "cooler/valve/pump/accumulator", "multirate_cycle_signal", "tab-delimited matrices"),
    DatasetDefinition("tennessee_eastman", "Tennessee Eastman Process", "Tennessee Eastman Process", "chemical process", "chemical plant simulator", "process units", "multivariate_process_time_series", "RData"),
    DatasetDefinition("nasa_ims", "NASA IMS Bearings", "NASA IMS Bearings", "rotating machinery", "bearing test rig", "rolling bearings", "high_frequency_signal", "ASCII waveform files"),
    DatasetDefinition("metal_etch", "Metal Etch", "metal_etch_data.csv", "unverified", "unknown", "unknown", "static_tabular", "CSV", notes="Anonymous features; physical meanings and units are unverified."),
    DatasetDefinition("digital_manufacturing_deprecated", "Digital Manufacturing (deprecated)", "_deprecated/digital_manufacturing_dataset.csv", "mixed manufacturing/business", "unknown", "unknown", "static_tabular", "CSV", status="deprecated", notes="Retained for inventory only; excluded from active PreparedObservation corpus."),
)


def file_hash(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def dataframe_quality(df: pd.DataFrame, label_columns: list[str] | None = None) -> dict[str, Any]:
    label_columns = label_columns or []
    numeric = df.select_dtypes(include=np.number)
    missing = df.isna().sum()
    profile = []
    for column in df.columns:
        series = df[column]
        item: dict[str, Any] = {
            "name": str(column), "dtype": str(series.dtype), "missing": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean() * 100), 6),
            "unique": int(series.nunique(dropna=True)), "constant": bool(series.nunique(dropna=True) <= 1),
        }
        if pd.api.types.is_numeric_dtype(series):
            finite = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
            item.update({"min": finite.min(), "max": finite.max(), "mean": finite.mean(), "std": finite.std()})
        profile.append(item)
    labels: dict[str, Any] = {}
    for column in label_columns:
        if column in df.columns:
            labels[column] = {str(key): int(value) for key, value in df[column].value_counts(dropna=False).head(50).items()}
    return {
        "records": int(len(df)), "columns": int(df.shape[1]), "duplicate_records": int(df.duplicated().sum()),
        "missing_cells": int(missing.sum()), "missing_by_column": {str(k): int(v) for k, v in missing[missing > 0].items()},
        "constant_columns": [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1],
        "infinite_numeric_values": int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()) if not numeric.empty else 0,
        "label_distribution": labels, "columns_profile": profile,
    }


def column_catalogue(columns: list[str], metadata: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    result = []
    for column in columns:
        info = metadata.get(column, {})
        result.append({
            "feature": column,
            "physical_meaning": info.get("meaning", "UNKNOWN/UNVERIFIED"),
            "unit": info.get("unit", "UNKNOWN/UNVERIFIED"),
            "category": info.get("category", "UNKNOWN/UNVERIFIED"),
            "source": info.get("source", "dataset column name only"),
        })
    return result


def inventory_entry(definition: DatasetDefinition, data_root: Path) -> dict[str, Any]:
    path = data_root / definition.relative_path
    files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else ([path] if path.exists() else [])
    # Timestamped IMS waveform filenames contain dots but are extensionless data
    # files; report the configured format rather than misleading pseudo-suffixes.
    formats = [definition.format]
    return {
        **asdict(definition), "path": str(path), "exists": path.exists(), "file_count": len(files),
        "size_bytes": int(sum(p.stat().st_size for p in files)), "formats_present": formats,
        "files": [{"path": str(p.relative_to(data_root)), "bytes": p.stat().st_size} for p in files[:100]],
        "files_truncated": len(files) > 100,
    }
