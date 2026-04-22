from __future__ import annotations

import json
import logging
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
            "dtype": str(series.dtype),
        }

    def _analyze_categorical(self, series: pd.Series) -> Dict[str, Any]:
        counts = series.value_counts(normalize=True, dropna=True)
        return {
            "categories": counts.index.tolist(),
            "probabilities": counts.values.tolist(),
        }

    def _generate_numeric(self, stats: Dict[str, Any]) -> Any:
        value = np.random.normal(stats["mean"], stats["std"])
        value = np.clip(value, stats["min"], stats["max"])
        if "int" in stats["dtype"]:
            return int(round(float(value)))
        return float(value)

    def _generate_categorical(self, stats: Dict[str, Any]) -> Any:
        return np.random.choice(stats["categories"], p=stats["probabilities"])

    def generate(
        self,
        df: pd.DataFrame,
        n_rows: int,
        target_column: Optional[str] = None,
    ) -> pd.DataFrame:
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
    generate_synthetic: bool
    synthetic_rows: int
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
                "generate_synthetic": config.generate_synthetic,
                "synthetic_rows": config.synthetic_rows,
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

        planner = RulesFirstPlannerAgent(
            dataset_path=config.dataset_path,
            feature_columns=config.feature_columns,
            target_column=config.target_column,
            problem_type=config.problem_type,
            llm_model=None,
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

            if config.generate_synthetic and planner.raw_data is not None:
                try:
                    synthetic_artifacts = self._run_synthetic_generation(
                        run_id=run_id,
                        raw_df=planner.raw_data.copy(),
                        target_column=planner.target_column,
                        problem_type=planner.problem_type,
                        preferred_model=config.preferred_model,
                        n_rows=config.synthetic_rows,
                    )
                    final_result["synthetic_generation"] = synthetic_artifacts
                except Exception as synthetic_exc:
                    self._set_stage(
                        run_id,
                        key="synthetic",
                        title="Generate Synthetic Data",
                        status="failed",
                        notes=[f"Synthetic generation failed: {synthetic_exc}"],
                    )
                    final_result["synthetic_generation"] = {"error": str(synthetic_exc)}
                    self.append_log(run_id, f"Synthetic generation warning: {synthetic_exc}")

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


run_manager = PipelineRunManager()
