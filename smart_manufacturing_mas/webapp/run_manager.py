from __future__ import annotations

import json
import logging
import hashlib
import re
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agents.rules_first_planner import RulesFirstPlannerAgent
from utils.pretrained_model_store import load_bundle, predict_with_bundle, select_bundle_metadata
from utils.synthetic_quality_analyzer import SyntheticQualityAnalyzer
from utils.prediction_analyzer import PredictionAnalyzer


ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
WEB_UPLOAD_DIR = ARTIFACTS_DIR / "web_uploads"
WEB_SYNTHETIC_DIR = ARTIFACTS_DIR / "web_synthetic"
WEB_RUN_DIR = ARTIFACTS_DIR / "web_runs"
PRETRAINED_DIR = ARTIFACTS_DIR / "pretrained_models"

WEB_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
WEB_SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
WEB_RUN_DIR.mkdir(parents=True, exist_ok=True)


def _build_cloud_llm_model() -> Optional[Any]:
    """Create a Gemini model for Reflexion summary when GEMINI_API_KEY is set."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        env_file = ROOT_DIR / ".env"
        if env_file.exists():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    key, value = stripped.split("=", 1)
                    if key.strip() == "GEMINI_API_KEY":
                        api_key = value.strip().strip('"').strip("'")
                        if api_key:
                            break
            except Exception:
                api_key = None
    if not api_key:
        logging.debug("No GEMINI_API_KEY found; Cloud LLM unavailable")
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        logging.info(f"Gemini API configured successfully")
        
        # Try models in order of preference (latest to fallback)
        model_candidates = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro"
        ]
        
        for model_name in model_candidates:
            try:
                model_instance = genai.GenerativeModel(model_name)
                logging.info(f"Successfully initialized Cloud LLM with model: {model_name}")
                return model_instance
            except Exception as model_error:
                logging.debug(f"Model {model_name} initialization failed: {model_error}")
                continue
        
        # If all models failed to initialize
        logging.warning(f"All Cloud LLM models failed to initialize; using plain-text summary fallback")
        return None
        
    except Exception as exc:
        logging.warning(f"Cloud LLM initialization failed: {exc}; using plain-text summary fallback")
        return None


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, pd.Series):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [{str(k): _json_safe(v) for k, v in row.items()} for row in value.to_dict(orient="records")]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if not isinstance(value, (str, bytes, dict, list, tuple, set, Path)):
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
    return value


def _preview_dataframe(df: Optional[pd.DataFrame], max_rows: int = 5, max_cols: int = 10) -> Dict[str, Any]:
    if df is None:
        return {"shape": [0, 0], "columns": [], "rows": []}

    preview = df.iloc[:max_rows, :max_cols].copy()
    preview = preview.replace({np.nan: None})
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [str(col) for col in preview.columns.tolist()],
        "truncated_columns": max(0, int(df.shape[1] - preview.shape[1])),
        "truncated_rows": max(0, int(df.shape[0] - preview.shape[0])),
        "rows": [{str(k): _json_safe(v) for k, v in row.items()} for row in preview.to_dict(orient="records")],
    }


def _top_missing_values(df: pd.DataFrame, limit: int = 8) -> Dict[str, int]:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0].head(limit)
    return {str(k): int(v) for k, v in missing.to_dict().items()}


def _top_feature_importance(results: Dict[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
    importances = results.get("feature_importances")
    feature_names = results.get("feature_names", [])
    if importances is None or feature_names is None:
        return []

    try:
        pairs = [
            {"feature": str(name), "importance": float(score)}
            for name, score in zip(feature_names, importances)
        ]
    except Exception:
        return []
    pairs.sort(key=lambda item: item["importance"], reverse=True)
    return pairs[:limit]


def _stage_overview(stages: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0}
    for stage in stages:
        status = stage.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def _slugify_text(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    return text.strip("_") or "dataset"


def _synthetic_config_signature(config: Dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


def _metrics_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "accuracy",
        "cv_accuracy",
        "cv_std",
        "r2",
        "mse",
        "cv_r2",
        "from_pretrained",
        "from_cache",
        "n_anomalies",
        "anomaly_rate",
    ]
    summary = {key: _json_safe(results.get(key)) for key in keys if key in results}
    summary["model_name"] = results.get("model")
    return summary


class _SyntheticDataGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def _analyze_numeric(self, series: pd.Series) -> Dict[str, Any]:
        clean = series.dropna()
        return {
            "mean": float(clean.mean()),
            "std": float(clean.std() or 1.0),
            "min": float(clean.min()),
            "max": float(clean.max()),
            "range": float((clean.max() - clean.min()) or 1.0),
            "dtype": str(series.dtype),
        }

    def _analyze_categorical(self, series: pd.Series) -> Dict[str, Any]:
        counts = series.value_counts(normalize=True, dropna=True)
        return {
            "categories": counts.index.tolist(),
            "probabilities": counts.values.tolist(),
        }

    def _generate_numeric(self, stats: Dict[str, Any], signal_value: Optional[float] = None) -> Any:
        noise_scale = max(stats["std"] * 0.35, stats["range"] * 0.03, 1e-6)
        if signal_value is None:
            base_value = stats["mean"]
        else:
            base_value = stats["mean"] + (float(signal_value) - stats["mean"]) * 0.65

        value = base_value + np.random.normal(0.0, noise_scale)
        value = np.clip(value, stats["min"], stats["max"])
        if "int" in stats["dtype"]:
            return int(round(float(value)))
        return float(value)

    def _generate_categorical(self, stats: Dict[str, Any]) -> Any:
        probabilities = np.asarray(stats["probabilities"], dtype=float)
        if probabilities.size == 0:
            return None
        probabilities = np.power(probabilities, 0.9)
        probabilities = probabilities / probabilities.sum()
        return np.random.choice(stats["categories"], p=probabilities)

    def generate(
        self,
        df: pd.DataFrame,
        n_rows: int,
        target_column: Optional[str] = None,
    ) -> pd.DataFrame:
        if target_column and target_column in df.columns:
            sampled = df.sample(n=n_rows, replace=True, random_state=self.seed).reset_index(drop=True).copy()
            numeric_cols = [col for col in sampled.columns if col != target_column and pd.api.types.is_numeric_dtype(sampled[col])]
            categorical_cols = [col for col in sampled.columns if col != target_column and col not in numeric_cols]

            for col in numeric_cols:
                clean = df[col].dropna()
                if clean.empty:
                    continue
                scale = float(clean.std() or 1.0) * 0.05
                noise = np.random.normal(0.0, scale, size=n_rows)
                sampled[col] = np.clip(sampled[col].astype(float) + noise, float(clean.min()), float(clean.max()))
                if pd.api.types.is_integer_dtype(df[col]):
                    sampled[col] = sampled[col].round().astype(int)

            for col in categorical_cols:
                if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    sampled[col] = sampled[col].fillna(df[col].mode(dropna=True).iloc[0] if not df[col].mode(dropna=True).empty else sampled[col])

            return sampled

        exclude = [col for col in df.columns if col.lower().endswith("_id") or col.lower() in {"id", "timestamp"}]
        feature_cols = [col for col in df.columns if col not in exclude and col != target_column]

        column_stats: Dict[str, Dict[str, Any]] = {}
        numeric_cols: List[str] = []
        categorical_cols: List[str] = []

        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
                column_stats[col] = self._analyze_numeric(df[col])
            else:
                categorical_cols.append(col)
                column_stats[col] = self._analyze_categorical(df[col])

        if target_column and target_column in df.columns:
            sampled = df.sample(n=n_rows, replace=True, random_state=self.seed).reset_index(drop=True).copy()

            for col in numeric_cols:
                sampled[col] = [
                    self._generate_numeric(column_stats[col], signal_value=value)
                    for value in sampled[col].astype(float).fillna(column_stats[col]["mean"]).tolist()
                ]

            for col in categorical_cols:
                sampled[col] = [self._generate_categorical(column_stats[col]) for _ in range(n_rows)]

            synthetic_df = sampled
        else:
            rows: List[Dict[str, Any]] = []
            for _ in range(n_rows):
                row: Dict[str, Any] = {}
                for col in numeric_cols:
                    row[col] = self._generate_numeric(column_stats[col])
                for col in categorical_cols:
                    row[col] = self._generate_categorical(column_stats[col])
                rows.append(row)

            synthetic_df = pd.DataFrame(rows)
        if target_column and target_column in df.columns:
            if pd.api.types.is_numeric_dtype(df[target_column]):
                stats = self._analyze_numeric(df[target_column])
                synthetic_df[target_column] = [self._generate_numeric(stats) for _ in range(n_rows)]
            else:
                stats = self._analyze_categorical(df[target_column])
                synthetic_df[target_column] = [self._generate_categorical(stats) for _ in range(n_rows)]
        return synthetic_df


@dataclass
class RunConfig:
    dataset_path: str
    dataset_label: str
    feature_columns: Optional[List[str]]
    target_column: Optional[str]
    problem_type: Optional[str]
    use_pca: bool
    use_cache: bool
    train_mode: str
    preferred_model: Optional[str]


class _RunLogHandler(logging.Handler):
    def __init__(self, manager: "PipelineRunManager", run_id: str):
        super().__init__()
        self.manager = manager
        self.run_id = run_id
        self.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.manager.append_log(self.run_id, self.format(record))
        except Exception:
            pass


class PipelineRunManager:
    def __init__(self) -> None:
        self._runs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_run(self, config: RunConfig) -> str:
        run_id = uuid.uuid4().hex[:10]
        run = {
            "id": run_id,
            "status": "queued",
            "created_at": _iso_now(),
            "updated_at": _iso_now(),
            "config": {
                "dataset_path": config.dataset_path,
                "dataset_label": config.dataset_label,
                "feature_columns": config.feature_columns or [],
                "target_column": config.target_column,
                "problem_type": config.problem_type,
                "use_pca": config.use_pca,
                "use_cache": config.use_cache,
                "train_mode": config.train_mode,
                "preferred_model": config.preferred_model,
            },
            "logs": [],
            "stages": [],
            "artifacts": [],
            "result": None,
            "error": None,
        }
        with self._lock:
            self._runs[run_id] = run

        thread = threading.Thread(target=self._execute_run, args=(run_id, config), daemon=True)
        thread.start()
        return run_id

    def list_runs(self) -> List[Dict[str, Any]]:
        with self._lock:
            runs = list(self._runs.values())
        return sorted(runs, key=lambda item: item["created_at"], reverse=True)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            return json.loads(json.dumps(run, default=_json_safe))

    def append_log(self, run_id: str, message: str) -> None:
        with self._lock:
            run = self._runs.get(run_id)
            if not run:
                return
            run["logs"].append({"timestamp": _iso_now(), "message": message})
            run["logs"] = run["logs"][-300:]
            run["updated_at"] = _iso_now()

    def _set_run_status(self, run_id: str, status: str, error: Optional[str] = None) -> None:
        with self._lock:
            run = self._runs[run_id]
            run["status"] = status
            run["updated_at"] = _iso_now()
            if error:
                run["error"] = error

    def _set_stage(
        self,
        run_id: str,
        key: str,
        title: str,
        status: str,
        input_summary: Optional[Dict[str, Any]] = None,
        output_summary: Optional[Dict[str, Any]] = None,
        preview: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        notes: Optional[List[str]] = None,
        duration_seconds: Optional[float] = None,
    ) -> None:
        with self._lock:
            run = self._runs[run_id]
            stage = next((item for item in run["stages"] if item["key"] == key), None)
            if stage is None:
                stage = {"key": key, "title": title, "started_at": _iso_now()}
                run["stages"].append(stage)
            stage.update(
                {
                    "title": title,
                    "status": status,
                    "input_summary": input_summary or stage.get("input_summary") or {},
                    "output_summary": output_summary or stage.get("output_summary") or {},
                    "preview": preview or stage.get("preview") or {},
                    "metrics": metrics or stage.get("metrics") or {},
                    "notes": notes or stage.get("notes") or [],
                    "duration_seconds": duration_seconds,
                }
            )
            if status in {"completed", "failed"}:
                stage["finished_at"] = _iso_now()
            run["updated_at"] = _iso_now()

    def _add_artifact(self, run_id: str, label: str, path: Path, kind: str) -> None:
        rel = path.relative_to(ROOT_DIR)
        with self._lock:
            self._runs[run_id]["artifacts"].append(
                {"label": label, "path": str(rel), "kind": kind}
            )
            self._runs[run_id]["updated_at"] = _iso_now()

    def _set_result(self, run_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            self._runs[run_id]["result"] = _json_safe(result)
            self._runs[run_id]["updated_at"] = _iso_now()

    def _persist_run_outputs(
        self,
        run_id: str,
        result_payload: Dict[str, Any],
        recommendations: pd.DataFrame,
    ) -> Dict[str, Any]:
        summary_path = WEB_RUN_DIR / f"{run_id}_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(result_payload), f, indent=2)
        self._add_artifact(run_id, "Run summary", summary_path, "json")

        persisted: Dict[str, Any] = {"summary_json": str(summary_path.relative_to(ROOT_DIR))}

        if not recommendations.empty:
            csv_path = WEB_RUN_DIR / f"{run_id}_recommendations.csv"
            json_path = WEB_RUN_DIR / f"{run_id}_recommendations.json"
            recommendations.to_csv(csv_path, index=False)
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(_json_safe(recommendations), f, indent=2)
            self._add_artifact(run_id, "Recommendations CSV", csv_path, "csv")
            self._add_artifact(run_id, "Recommendations JSON", json_path, "json")
            persisted["recommendations_csv"] = str(csv_path.relative_to(ROOT_DIR))
            persisted["recommendations_json"] = str(json_path.relative_to(ROOT_DIR))
        return persisted

    def _execute_run(self, run_id: str, config: RunConfig) -> None:
        self._set_run_status(run_id, "running")
        log_handler = _RunLogHandler(self, run_id)
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)

        cloud_llm_model = _build_cloud_llm_model()
        planner = RulesFirstPlannerAgent(
            dataset_path=config.dataset_path,
            feature_columns=config.feature_columns,
            target_column=config.target_column,
            problem_type=config.problem_type,
            llm_model=cloud_llm_model,
            use_pca=config.use_pca,
            use_cache=config.use_cache,
            auto_hitl=True,
            inference_only=config.train_mode != "live",
            pretrained_dir=str(PRETRAINED_DIR),
            preferred_model=config.preferred_model,
        )

        try:
            overall_started = time.time()
            self.append_log(run_id, f"Starting run for dataset: {config.dataset_label}")

            stage_started = time.time()
            self._set_stage(
                run_id,
                key="resolve",
                title="Resolve Problem Type",
                status="running",
                input_summary={
                    "dataset": config.dataset_label,
                    "requested_problem_type": config.problem_type or "auto",
                    "requested_target": config.target_column or "auto",
                    "requested_features": len(config.feature_columns or []),
                    "train_mode": config.train_mode,
                },
            )
            if not planner._step0_resolve_problem_type():
                raise RuntimeError("Step 0 failed while resolving the problem type.")
            self._set_stage(
                run_id,
                key="resolve",
                title="Resolve Problem Type",
                status="completed",
                output_summary={
                    "problem_type": planner.problem_type,
                    "target_column": planner.target_column,
                    "feature_count": len(planner.feature_columns or []),
                },
                duration_seconds=round(time.time() - stage_started, 3),
            )

            stage_started = time.time()
            self._set_stage(
                run_id,
                key="load",
                title="Load Dataset",
                status="running",
                input_summary={"dataset_path": config.dataset_path},
            )
            if not planner._step1_load_data():
                raise RuntimeError("Step 1 failed while loading the dataset.")
            raw_df = planner.raw_data if planner.raw_data is not None else pd.DataFrame()
            self._set_stage(
                run_id,
                key="load",
                title="Load Dataset",
                status="completed",
                output_summary={
                    "rows": int(raw_df.shape[0]),
                    "columns": int(raw_df.shape[1]),
                    "missing_values": _top_missing_values(raw_df),
                },
                preview=_preview_dataframe(raw_df),
                duration_seconds=round(time.time() - stage_started, 3),
            )

            stage_started = time.time()
            self._set_stage(
                run_id,
                key="preprocess",
                title="Preprocess Data",
                status="running",
                input_summary={
                    "input_shape": [int(raw_df.shape[0]), int(raw_df.shape[1])],
                    "use_pca": config.use_pca,
                },
            )
            if not planner._step2_preprocess():
                raise RuntimeError("Step 2 failed during preprocessing.")
            processed_df = planner.preprocessed_data if planner.preprocessed_data is not None else pd.DataFrame()
            self._set_stage(
                run_id,
                key="preprocess",
                title="Preprocess Data",
                status="completed",
                output_summary={
                    "processed_rows": int(processed_df.shape[0]),
                    "processed_columns": int(processed_df.shape[1]),
                },
                preview=_preview_dataframe(processed_df),
                duration_seconds=round(time.time() - stage_started, 3),
            )

            stage_started = time.time()
            self._set_stage(
                run_id,
                key="analyze",
                title="Model Analysis",
                status="running",
                input_summary={
                    "problem_type": planner.problem_type,
                    "target_column": planner.target_column,
                    "train_mode": config.train_mode,
                },
            )
            if not planner._step3_analyse():
                raise RuntimeError("Step 3 failed during model analysis.")
            analysis = planner.analysis_results or {}
            self._set_stage(
                run_id,
                key="analyze",
                title="Model Analysis",
                status="completed",
                output_summary={
                    "selected_model": analysis.get("model"),
                    "prediction_count": len(analysis.get("predictions", [])) if analysis.get("predictions") is not None else 0,
                },
                metrics=_metrics_summary(analysis),
                preview={
                    "predictions_preview": _json_safe(list(np.asarray(analysis.get("predictions", []))[:10])),
                    "top_feature_importance": _top_feature_importance(analysis),
                },
                duration_seconds=round(time.time() - stage_started, 3),
            )

            stage_started = time.time()
            self._set_stage(
                run_id,
                key="optimize",
                title="Generate Recommendations",
                status="running",
                input_summary={"model": analysis.get("model")},
            )
            if not planner._step4_optimise():
                raise RuntimeError("Step 4 failed while generating recommendations.")
            recommendations = planner.recommendations if isinstance(planner.recommendations, pd.DataFrame) else pd.DataFrame()
            self._set_stage(
                run_id,
                key="optimize",
                title="Generate Recommendations",
                status="completed",
                output_summary={"recommendation_count": int(len(recommendations))},
                preview=_preview_dataframe(recommendations),
                duration_seconds=round(time.time() - stage_started, 3),
            )

            stage_started = time.time()
            self._set_stage(
                run_id,
                key="summary",
                title="Workflow Summary",
                status="running",
            )
            summary_text = planner._step5_reflexion_summary()
            self._set_stage(
                run_id,
                key="summary",
                title="Workflow Summary",
                status="completed",
                output_summary={"summary_length": len(summary_text)},
                notes=[summary_text],
                duration_seconds=round(time.time() - stage_started, 3),
            )

            final_result = {
                "status": "success",
                "problem_type": planner.problem_type,
                "target_column": planner.target_column,
                "feature_columns": planner.feature_columns,
                "analysis_metrics": _metrics_summary(analysis),
                "summary": summary_text,
                "recommendation_count": int(len(recommendations)),
                "recommendations_preview": _preview_dataframe(recommendations, max_rows=8, max_cols=8),
                "total_duration_seconds": round(time.time() - overall_started, 3),
            }

            final_result["stage_overview"] = _stage_overview(self._runs[run_id]["stages"])
            final_result["saved_outputs"] = self._persist_run_outputs(run_id, final_result, recommendations)

            self._set_result(run_id, final_result)
            self._set_run_status(run_id, "completed")
            self.append_log(run_id, "Run completed successfully.")
        except Exception as exc:
            logging.exception("Pipeline run failed.")
            self._set_run_status(run_id, "failed", error=str(exc))
            self.append_log(run_id, f"Run failed: {exc}")
        finally:
            root_logger.removeHandler(log_handler)

    def _run_synthetic_generation(
        self,
        run_id: str,
        raw_df: pd.DataFrame,
        target_column: Optional[str],
        problem_type: Optional[str],
        preferred_model: Optional[str],
        n_rows: int,
    ) -> Dict[str, Any]:
        stage_started = time.time()
        self._set_stage(
            run_id,
            key="synthetic",
            title="Generate Synthetic Data",
            status="running",
            input_summary={
                "input_rows": int(raw_df.shape[0]),
                "requested_rows": int(n_rows),
                "target_column": target_column,
            },
        )

        generator = _SyntheticDataGenerator(seed=42)
        synthetic_df = generator.generate(raw_df, n_rows=n_rows, target_column=target_column)
        csv_path = WEB_SYNTHETIC_DIR / f"{run_id}_synthetic.csv"
        synthetic_df.to_csv(csv_path, index=False)
        self._add_artifact(run_id, "Synthetic dataset", csv_path, "csv")

        output_summary: Dict[str, Any] = {"synthetic_rows": int(synthetic_df.shape[0])}
        preview = _preview_dataframe(synthetic_df)
        artifact_summary: Dict[str, Any] = {
            "csv_path": str(csv_path.relative_to(ROOT_DIR)),
        }

        # Analyze data quality (original vs synthetic comparison)
        try:
            quality_analyzer = SyntheticQualityAnalyzer(raw_df, synthetic_df)
            quality_summary = quality_analyzer.get_summary_for_display()
            artifact_summary["data_quality"] = quality_summary
            preview["data_quality"] = quality_summary
            output_summary["quality_score"] = quality_summary.get("quality_score")
            output_summary["quality_level"] = quality_summary.get("quality_level")
        except Exception as qa_exc:
            self.append_log(run_id, f"Data quality analysis warning: {qa_exc}")

        if problem_type in {"classification", "regression"} and target_column:
            meta = select_bundle_metadata(
                problem_type=problem_type,
                target_column=target_column,
                preferred_model=preferred_model,
                path=str(PRETRAINED_DIR),
            )
            if meta and meta.get("bundle_file"):
                try:
                    bundle = load_bundle(meta["bundle_file"], path=str(PRETRAINED_DIR))
                    prediction_results = predict_with_bundle(bundle, synthetic_df, target_column=target_column)
                    
                    # Analyze predictions
                    predictions = prediction_results.get("predictions", [])
                    pred_analyzer = PredictionAnalyzer(
                        predictions=predictions,
                        problem_type=problem_type,
                    )
                    pred_summary = pred_analyzer.get_summary()
                    
                    # Save enhanced inference results
                    inference_output = {
                        "predictions": _json_safe(predictions),
                        "prediction_analysis": pred_summary,
                    }
                    json_path = WEB_SYNTHETIC_DIR / f"{run_id}_synthetic_inference.json"
                    with json_path.open("w", encoding="utf-8") as f:
                        json.dump(_json_safe(inference_output), f, indent=2, default=_json_safe)
                    self._add_artifact(run_id, "Synthetic inference summary", json_path, "json")
                    
                    output_summary["bundle_used"] = meta["bundle_file"]
                    output_summary["prediction_count"] = len(predictions)
                    output_summary["predictions_analysis"] = pred_summary
                    
                    preview["prediction_preview"] = _json_safe(list(np.asarray(predictions)[:10]))
                    preview["predictions_analysis"] = pred_summary
                    artifact_summary["inference_json"] = str(json_path.relative_to(ROOT_DIR))
                    artifact_summary["predictions_analysis"] = pred_summary
                    
                except Exception as pred_exc:
                    self.append_log(run_id, f"Prediction analysis warning: {pred_exc}")

        self._set_stage(
            run_id,
            key="synthetic",
            title="Generate Synthetic Data",
            status="completed",
            output_summary=output_summary,
            preview=preview,
            notes=[json.dumps(artifact_summary, indent=2)],
            duration_seconds=round(time.time() - stage_started, 3),
        )
        return artifact_summary

    def generate_synthetic_data(
        self,
        dataset_path: str,
        n_rows: int,
        target_column: Optional[str],
        seed: int,
        problem_type: Optional[str] = None,
        preferred_model: Optional[str] = None,
    ) -> str:
        """Generate synthetic data from a real dataset and return synthetic_id."""
        config = {
            "generator_version": "v3_awgn_blended",
            "source_dataset": str(Path(dataset_path).resolve()),
            "n_rows": int(n_rows),
            "target_column": target_column or "",
            "seed": int(seed),
            "problem_type": problem_type or "",
            "preferred_model": preferred_model or "",
        }
        signature = _synthetic_config_signature(config)
        source_slug = _slugify_text(Path(dataset_path).stem)
        target_slug = _slugify_text(target_column) if target_column else "all_columns"
        synthetic_name = f"{source_slug}__rows{n_rows}__target-{target_slug}__seed{seed}__sig-{signature}"
        synthetic_id = f"syn_{signature}"

        with self._lock:
            if not hasattr(self, "_synthetic_datasets"):
                self._synthetic_datasets = {}
            existing = self._synthetic_datasets.get(synthetic_id)
            if existing:
                existing["updated_at"] = _iso_now()
                return synthetic_id

            existing_file = WEB_SYNTHETIC_DIR / f"{synthetic_name}.csv"
            if existing_file.exists():
                try:
                    existing_df = pd.read_csv(existing_file)
                    existing_preview = _preview_dataframe(existing_df, max_rows=5)
                except Exception:
                    existing_df = None
                    existing_preview = None
                synthetic_record = {
                    "id": synthetic_id,
                    "name": synthetic_name,
                    "status": "ready_for_inference",
                    "created_at": _iso_now(),
                    "updated_at": _iso_now(),
                    "config": config,
                    "signature": signature,
                    "generation_result": {
                        "csv_path": str(existing_file.relative_to(ROOT_DIR)),
                        "file_name": existing_file.name,
                        "n_rows": int(n_rows),
                        "n_columns": int(existing_df.shape[1]) if existing_df is not None else None,
                        "columns": existing_df.columns.tolist() if existing_df is not None else [],
                        "preview": existing_preview,
                    },
                    "inference_result": None,
                    "error": None,
                }
                self._synthetic_datasets[synthetic_id] = synthetic_record
                return synthetic_id

        synthetic_record = {
            "id": synthetic_id,
            "name": synthetic_name,
            "status": "queued",
            "created_at": _iso_now(),
            "updated_at": _iso_now(),
            "config": config,
            "signature": signature,
            "generation_result": None,
            "inference_result": None,
            "error": None,
        }

        self._synthetic_datasets[synthetic_id] = synthetic_record

        thread = threading.Thread(
            target=self._execute_synthetic_generation,
            args=(synthetic_id, dataset_path, n_rows, target_column, seed, synthetic_name, signature),
            daemon=True,
        )
        thread.start()
        return synthetic_id

    def _execute_synthetic_generation(
        self,
        synthetic_id: str,
        dataset_path: str,
        n_rows: int,
        target_column: Optional[str],
        seed: int,
        synthetic_name: str,
        signature: str,
    ) -> None:
        """Execute synthetic data generation in background."""
        try:
            with self._lock:
                self._synthetic_datasets[synthetic_id]["status"] = "running"
                self._synthetic_datasets[synthetic_id]["updated_at"] = _iso_now()

            # Load real data
            df = pd.read_csv(dataset_path)

            # Generate synthetic data
            generator = _SyntheticDataGenerator(seed=seed)
            synthetic_df = generator.generate(df, n_rows=n_rows, target_column=target_column)

            quality_summary = None
            try:
                quality_analyzer = SyntheticQualityAnalyzer(df, synthetic_df)
                quality_summary = quality_analyzer.get_summary_for_display()
                logging.info(f"Synthetic dataset quality analysis completed successfully")
            except Exception as quality_exc:
                logging.warning(f"Synthetic quality analysis failed: {quality_exc}")

            # Save to disk
            csv_path = WEB_SYNTHETIC_DIR / f"{synthetic_name}.csv"
            synthetic_df.to_csv(csv_path, index=False)

            # Prepare result
            with self._lock:
                self._synthetic_datasets[synthetic_id]["generation_result"] = {
                    "csv_path": str(csv_path.relative_to(ROOT_DIR)),
                    "file_name": csv_path.name,
                    "n_rows": int(synthetic_df.shape[0]),
                    "n_columns": int(synthetic_df.shape[1]),
                    "columns": synthetic_df.columns.tolist(),
                    "preview": _preview_dataframe(synthetic_df, max_rows=5),
                    "data_quality": quality_summary,
                }
                self._synthetic_datasets[synthetic_id]["signature"] = signature
                self._synthetic_datasets[synthetic_id]["status"] = "ready_for_inference"
                self._synthetic_datasets[synthetic_id]["updated_at"] = _iso_now()
        except Exception as e:
            with self._lock:
                self._synthetic_datasets[synthetic_id]["status"] = "failed"
                self._synthetic_datasets[synthetic_id]["error"] = str(e)
                self._synthetic_datasets[synthetic_id]["updated_at"] = _iso_now()

    def run_inference_on_synthetic(
        self,
        synthetic_id: str,
        problem_type: Optional[str],
        target_column: Optional[str],
        preferred_model: Optional[str],
    ) -> str:
        """Run inference on synthetic data and return inference_id (same as synthetic_id)."""
        with self._lock:
            record = getattr(self, "_synthetic_datasets", {}).get(synthetic_id)
            if record:
                config = record.get("config", {})
                problem_type = problem_type or config.get("problem_type") or None
                target_column = target_column or config.get("target_column") or None
                preferred_model = preferred_model or config.get("preferred_model") or None

        inference_id = synthetic_id
        thread = threading.Thread(
            target=self._execute_synthetic_inference,
            args=(synthetic_id, problem_type, target_column, preferred_model),
            daemon=True,
        )
        thread.start()
        return inference_id

    def _execute_synthetic_inference(
        self,
        synthetic_id: str,
        problem_type: Optional[str],
        target_column: Optional[str],
        preferred_model: Optional[str],
    ) -> None:
        """Execute inference on synthetic data in background."""
        try:
            with self._lock:
                if not hasattr(self, "_synthetic_datasets"):
                    return
                record = self._synthetic_datasets.get(synthetic_id)
                if not record or not record.get("generation_result"):
                    return
                record["status"] = "running_inference"
                record["updated_at"] = _iso_now()

            # Load synthetic data
            csv_path = record["generation_result"]["csv_path"]
            synthetic_df = pd.read_csv(ROOT_DIR / csv_path)

            # Get metadata for pretrained model
            meta = select_bundle_metadata(
                problem_type=problem_type,
                target_column=target_column,
                preferred_model=preferred_model,
                path=str(PRETRAINED_DIR),
            )

            if not meta or not meta.get("bundle_file"):
                raise ValueError(f"No pretrained model found for {problem_type}, {target_column}")

            # Load bundle and run inference
            bundle = load_bundle(meta["bundle_file"], path=str(PRETRAINED_DIR))
            prediction_bundle = predict_with_bundle(bundle, synthetic_df, target_column=target_column)
            predictions = prediction_bundle.get("predictions")
            if predictions is None:
                raise ValueError("Pretrained bundle did not return predictions.")

            # Analyze results
            actual_target = (
                synthetic_df.get(target_column)
                if target_column and target_column in synthetic_df.columns
                else prediction_bundle.get("y_test")
            )
            analyzer = PredictionAnalyzer(
                predictions=predictions,
                actual_values=actual_target,
                problem_type=problem_type or "regression",
            )
            analysis_summary = analyzer.get_summary()

            inference_result = {
                "model_name": meta.get("model_name"),
                "model_type": problem_type,
                "target_column": target_column,
                "bundle_file": meta.get("bundle_file"),
                "bundle_target_column": prediction_bundle.get("bundle_target_column"),
                "prediction_warning": prediction_bundle.get("target_mismatch_warning"),
                "n_predictions": len(predictions),
                "predictions_preview": predictions[:10].tolist() if hasattr(predictions, "tolist") else list(predictions[:10]),
                "analysis": analysis_summary.get("analysis", {}),
                "recommendations": analysis_summary.get("recommendations", []),
            }

            with self._lock:
                record["inference_result"] = inference_result
                record["status"] = "complete"
                record["updated_at"] = _iso_now()

        except Exception as e:
            with self._lock:
                record = self._synthetic_datasets.get(synthetic_id)
                if record:
                    record["status"] = "inference_failed"
                    record["error"] = str(e)
                    record["updated_at"] = _iso_now()

    def list_synthetic_datasets(self) -> List[Dict[str, Any]]:
        """List all generated synthetic datasets."""
        if not hasattr(self, "_synthetic_datasets"):
            return []
        with self._lock:
            datasets = list(self._synthetic_datasets.values())
        return sorted(datasets, key=lambda x: x["created_at"], reverse=True)

    def get_synthetic_dataset(self, synthetic_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a synthetic dataset."""
        if not hasattr(self, "_synthetic_datasets"):
            return None
        with self._lock:
            dataset = self._synthetic_datasets.get(synthetic_id)
            if dataset:
                return json.loads(json.dumps(dataset, default=_json_safe))
        return None

    def get_synthetic_inference_result(self, synthetic_id: str) -> Optional[Dict[str, Any]]:
        """Get inference results for a synthetic dataset."""
        dataset = self.get_synthetic_dataset(synthetic_id)
        if not dataset:
            return None
        return {
            "status": dataset.get("status"),
            "generation_result": dataset.get("generation_result"),
            "inference_result": dataset.get("inference_result"),
            "error": dataset.get("error"),
        }


run_manager = PipelineRunManager()
