import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


DEFAULT_PRETRAINED_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "pretrained_models"


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_pretrained_dir(path: Optional[str] = None) -> Path:
    if path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return Path(__file__).resolve().parent.parent / candidate
    return DEFAULT_PRETRAINED_DIR


def load_registry(path: Optional[str] = None) -> Dict[str, Any]:
    base = get_pretrained_dir(path)
    registry_path = base / "registry.json"
    if not registry_path.exists():
        return {"regression": [], "classification": []}
    with registry_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("regression", [])
    data.setdefault("classification", [])
    return data


def _bundle_entry_score(entry: Dict[str, Any], problem_type: str) -> float:
    metrics = entry.get("metrics", {}) or {}
    if problem_type == "regression":
        return _to_float(metrics.get("r2", metrics.get("cv_r2", -1e9)), -1e9)
    return _to_float(metrics.get("accuracy", metrics.get("cv_accuracy", -1e9)), -1e9)


def select_bundle_metadata(
    problem_type: str,
    target_column: Optional[str] = None,
    preferred_model: Optional[str] = None,
    path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    registry = load_registry(path)
    entries: List[Dict[str, Any]] = list(registry.get(problem_type, []))
    if not entries:
        base = get_pretrained_dir(path)
        discovered: List[Dict[str, Any]] = []
        prefix = f"{problem_type}_"
        for file in sorted(base.glob(f"{prefix}*.joblib")):
            stem = file.stem
            model_name = stem[len(prefix):] if stem.startswith(prefix) else stem
            discovered_target = None
            if "__" in stem:
                # Expected fallback pattern: <problem_type>__<target>__<model>
                parts = stem.split("__")
                if len(parts) >= 3:
                    discovered_target = parts[1]
            discovered.append(
                {
                    "model_name": model_name,
                    "bundle_file": file.name,
                    "target_column": discovered_target,
                    "metrics": {},
                }
            )

        entries = discovered
        if not entries:
            return None

    if target_column:
        requested = str(target_column).strip().lower()
        target_matches = [
            e for e in entries
            if str(e.get("target_column", "")).strip().lower() == requested
        ]
        if not target_matches:
            # Do not silently fall back to an unrelated target when caller requested one.
            return None
        entries = target_matches

    if preferred_model:
        for entry in entries:
            if str(entry.get("model_name", "")).lower() == preferred_model.lower():
                return entry

    return sorted(entries, key=lambda e: _bundle_entry_score(e, problem_type), reverse=True)[0]


def load_bundle(bundle_file: str, path: Optional[str] = None) -> Dict[str, Any]:
    base = get_pretrained_dir(path)
    bundle_path = base / bundle_file
    if not bundle_path.exists():
        raise FileNotFoundError(f"Pretrained bundle not found: {bundle_path}")
    return joblib.load(bundle_path)


def align_features(input_df: pd.DataFrame, expected_columns: List[str]) -> pd.DataFrame:
    aligned = input_df.copy()
    for col in expected_columns:
        if col not in aligned.columns:
            aligned[col] = np.nan
    return aligned[expected_columns]


def predict_with_bundle(
    bundle: Dict[str, Any],
    data: pd.DataFrame,
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    expected_features = list(bundle.get("feature_columns", []))
    pipeline = bundle["pipeline"]
    bundle_target = bundle.get("target_column")

    drop_cols = []
    if target_column:
        drop_cols.append(target_column)
    if bundle_target and bundle_target not in drop_cols:
        drop_cols.append(bundle_target)

    X_source = data.drop(columns=drop_cols, errors="ignore") if drop_cols else data.copy()
    X = align_features(X_source, expected_features)
    predictions = pipeline.predict(X)

    result: Dict[str, Any] = {
        "model": bundle.get("model_name", "pretrained_model"),
        "predictions": predictions,
        "X_test": X,
        "feature_names": expected_features,
        "from_pretrained": True,
    }

    problem_type = bundle.get("problem_type")
    metrics = bundle.get("metrics", {}) or {}
    result["pretrained_metrics"] = metrics
    result["bundle_target_column"] = bundle_target

    if problem_type == "regression":
        baseline = bundle.get("train_prediction_stats", {}) or {}
        if baseline:
            result["regression_baseline"] = {
                "mean": _to_float(baseline.get("mean"), 0.0),
                "std": max(_to_float(baseline.get("std"), 1.0), 1e-12),
            }

    eval_target = None
    if bundle_target and bundle_target in data.columns:
        eval_target = bundle_target
    elif target_column and target_column in data.columns:
        eval_target = target_column

    if eval_target:
        y_true = data[eval_target]
        result["y_test"] = y_true
        result["evaluation_target_column"] = eval_target
        if problem_type == "regression":
            try:
                from sklearn.metrics import mean_squared_error, r2_score

                result["mse"] = float(mean_squared_error(y_true, predictions))
                result["r2"] = float(r2_score(y_true, predictions))
            except Exception:
                pass
        elif problem_type == "classification":
            try:
                from sklearn.metrics import accuracy_score, classification_report

                result["accuracy"] = float(accuracy_score(y_true, predictions))
                result["classification_report"] = classification_report(y_true, predictions)
            except Exception:
                pass

    if target_column and bundle_target and target_column != bundle_target:
        result["target_mismatch_warning"] = (
            f"Run target '{target_column}' differs from pretrained target '{bundle_target}'. "
            f"Evaluation used '{eval_target or bundle_target}'."
        )

    return result
