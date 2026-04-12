"""
agents/dynamic_analysis_agent.py
----------------------------------
Dynamic Analysis Agent — model selection, training, and adaptive retry.

Improvements over baseline:
  ✦ Model-cache integration via ModelCache.
    Pass model_cache=ModelCache() + dataset_path + feature_columns to enable.
    Same config → instant load; different config → fresh train.

Architecture note (Three-Tier Intelligence Hierarchy):
  - Model family selection : Rule-Based ToolDecider (TIER 3), NOT SLM.
  - Anomaly params          : Local SLM (TIER 2) — called from LLMPlannerAgent.
  - Adaptive Intelligence   : Rule-based retry across all model families.

SLM Reduction:
  - SLM 3a (Model Selection): ELIMINATED — ToolDecider rule table.
  - SLM 3b (Anomaly Params) : RETAINED — called from LLMPlannerAgent.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC, SVR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.tool_decider import ToolDecider, create_data_summary, get_tool_decider
from utils.column_utils import is_identifier_column
from utils.pretrained_model_store import select_bundle_metadata, load_bundle, predict_with_bundle

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')


class DynamicAnalysisAgent:
    """
    Selects and trains an ML model appropriate for the given task.

    Args:
        data            : Preprocessed DataFrame (target column included for supervised tasks).
        target_column   : Target column name (None for anomaly detection).
        task            : 'classification' | 'regression' | 'anomaly_detection'.
        params          : Optional hyperparameter overrides.
        tool_decider    : Rule-based model selector (defaults to rule_based).
        model_cache     : Optional ModelCache instance for persistence.
        dataset_path    : Required when model_cache is provided — used for the cache key.
        feature_columns : Required when model_cache is provided.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None,
        task: str = "classification",
        params: Optional[Dict[str, Any]] = None,
        tool_decider: Optional[ToolDecider] = None,
        model_cache: Optional[Any] = None,
        dataset_path: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        inference_only: bool = False,
        pretrained_dir: Optional[str] = None,
        preferred_model: Optional[str] = None,
    ):
        self.data = data
        self.target_column = target_column
        self.task = task
        self.params = params or {}
        self.tool_decider = tool_decider or get_tool_decider("rule_based")

        self.model_cache = model_cache
        self.dataset_path = dataset_path or ""
        self.feature_columns = feature_columns or (
            [c for c in data.columns if c != target_column] if target_column else list(data.columns)
        )
        self.inference_only = inference_only
        self.pretrained_dir = pretrained_dir
        self.preferred_model = preferred_model

        self.model = None
        self.model_name: Optional[str] = None
        self.results: Dict[str, Any] = {}
        self.tried_models: List[str] = []
        self.best_performance: float = -float("inf")
        self.best_model: Optional[str] = None
        self.best_results: Optional[Dict[str, Any]] = None

        logging.info(
            f"DynamicAnalysisAgent init — task={task}, "
            f"cache={'enabled' if model_cache else 'disabled'}, "
            f"inference_only={self.inference_only}, params={self.params}"
        )

    # ── Tool selection ────────────────────────────────────────────────────────

    def choose_tool(self) -> str:
        """Select the initial model family via the ToolDecider rule table."""
        if self.task == "anomaly_detection":
            self.model_name = "IsolationForest"
            logging.info("Anomaly detection task — selected IsolationForest.")
        else:
            data_summary = create_data_summary(self.data)
            if self.task == "classification":
                candidates = ["LogisticRegression", "RandomForestClassifier", "SVC"]
            elif self.task == "regression":
                candidates = [
                    "LinearRegression",
                    "Ridge",
                    "Lasso",
                    "HistGradientBoostingRegressor",
                    "RandomForestRegressor",
                    "SVR",
                ]
            else:
                candidates = ["RandomForestClassifier", "LogisticRegression", "SVC"]

            decision = self.tool_decider.decide_model_family(self.task, data_summary, candidates)
            default = "LinearRegression" if self.task == "regression" else "RandomForestClassifier"
            self.model_name = decision.get("model", default)
            logging.info(f"ToolDecider selected '{self.model_name}', reason: {decision.get('reason', 'N/A')}")

        return self.model_name

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self, force_retry: bool = False) -> Optional[Dict[str, Any]]:
        """
        Train (or load from cache) and return results.
        force_retry=True triggers Adaptive Intelligence (sweeps all model families).
        """
        if self.inference_only and self.task in ("classification", "regression"):
            return self._run_pretrained_inference()

        # ── Cache check ───────────────────────────────────────────────────
        if self.model_cache is not None:
            entry = self.model_cache.load(
                self.dataset_path, self.feature_columns, self.target_column, self.task
            )
            if entry:
                logging.info("[ModelCache] Cache HIT — loading cached model, skipping training.")
                self.model = entry.get("model")
                metadata = entry.get("metadata", {})
                self.model_name = metadata.get("model_name", "cached_model")
                results = dict(metadata)
                results["from_cache"] = True
                results["model"] = self.model
                return results

        if self.task == "anomaly_detection":
            return self._run_anomaly_detection()

        if self.target_column is None:
            logging.error("Target column required for supervised learning tasks.")
            return None

        results = self._try_multiple_models() if force_retry else self._dispatch_single_model()

        # ── Save to cache ─────────────────────────────────────────────────
        if results and self.model_cache is not None and self.model is not None:
            metadata = {
                k: v for k, v in results.items()
                if k not in ("model", "X_test", "y_test", "predictions", "train_predictions")
            }
            metadata["model_name"] = self.model_name
            self.model_cache.save(
                model=self.model,
                dataset_path=self.dataset_path,
                feature_columns=self.feature_columns,
                target_column=self.target_column,
                problem_type=self.task,
                metadata=metadata,
            )
            logging.info(f"[ModelCache] Saved trained '{self.model_name}'.")

        return results

    def _run_pretrained_inference(self) -> Optional[Dict[str, Any]]:
        """Load a pre-trained bundle for the current task and run inference on provided data."""
        meta = select_bundle_metadata(
            problem_type=self.task,
            target_column=self.target_column,
            preferred_model=self.preferred_model,
            path=self.pretrained_dir,
        )
        if not meta:
            logging.error(
                f"No pretrained bundles found for task '{self.task}'. "
                "Train and save bundles first via the offline training notebook."
            )
            return None

        bundle_file = meta.get("bundle_file")
        if not bundle_file:
            logging.error("Pretrained registry entry is missing 'bundle_file'.")
            return None

        try:
            bundle = load_bundle(bundle_file, path=self.pretrained_dir)
            results = predict_with_bundle(bundle, self.data, target_column=self.target_column)
            results["from_cache"] = False
            results["from_pretrained"] = True
            self.model_name = results.get("model")
            self.model = bundle.get("pipeline")
            logging.info(
                f"Loaded pretrained model '{self.model_name}' from '{bundle_file}' for inference."
            )
            if results.get("target_mismatch_warning"):
                logging.warning(results["target_mismatch_warning"])
            if "r2" in results:
                logging.info(f"Pretrained inference R²={results['r2']:.4f}, MSE={results.get('mse', float('nan')):.4f}")
            if "accuracy" in results:
                logging.info(f"Pretrained inference accuracy={results['accuracy']:.4f}")
            return results
        except Exception as exc:
            logging.error(f"Failed to run pretrained inference: {exc}", exc_info=True)
            return None

    # ── Adaptive Intelligence ─────────────────────────────────────────────────

    def _try_multiple_models(self) -> Optional[Dict[str, Any]]:
        """Sweep all model families and return the best-performing results."""
        logging.info("🧠 ADAPTIVE INTELLIGENCE: Trying multiple models for better performance...")

        n_rows = len(self.data)

        if self.task == "classification":
            candidates = [
                ("RandomForestClassifier", self._run_random_forest),
                ("LogisticRegression",     self._run_logistic_regression),
                ("SVC",                    self._run_svc),
            ]
        elif self.task == "regression":
            candidates = [
                ("LinearRegression",       self._run_linear_regression),
                ("Ridge",                  self._run_ridge),
                ("Lasso",                  self._run_lasso),
                ("HistGradientBoostingRegressor", self._run_hist_gradient_boosting_regressor),
                ("RandomForestRegressor",  self._run_random_forest_regressor),
                ("SVR",                    self._run_svr),
            ]
            if n_rows > 30_000:
                candidates = [c for c in candidates if c[0] != "SVR"]
                logging.info("Large dataset detected — skipping SVR during adaptive retry.")
        else:
            return None

        best_perf = -float("inf")
        best_name = None
        best_results = None

        for name, func in candidates:
            if name in self.tried_models:
                logging.info(f"⏭️  Skipping {name} (already tried).")
                continue
            try:
                logging.info(f"🔄  Trying {name}...")
                res = func()
                if res:
                    perf = res.get("accuracy", 0) if self.task == "classification" else res.get("r2", -float("inf"))
                    logging.info(f"   {name} performance: {perf:.4f}")
                    if perf > best_perf:
                        best_perf = perf
                        best_name = name
                        best_results = res
                        self.best_model = name
                        self.best_results = res
                        self.best_performance = perf
                self.tried_models.append(name)
            except Exception as exc:
                logging.warning(f"   {name} failed: {exc}")
                self.tried_models.append(name)

        if best_results:
            logging.info(f"🏆 Best model: {best_name} (performance: {best_perf:.4f})")
            return best_results

        logging.error("❌ All models failed or produced invalid results.")
        return None

    # ── Single-model dispatch ─────────────────────────────────────────────────

    def _dispatch_single_model(self) -> Optional[Dict[str, Any]]:
        tool = self.choose_tool()
        dispatch = {
            "RandomForestClassifier": self._run_random_forest,
            "LogisticRegression":     self._run_logistic_regression,
            "SVC":                    self._run_svc,
            "RandomForestRegressor":  self._run_random_forest_regressor,
            "HistGradientBoostingRegressor": self._run_hist_gradient_boosting_regressor,
            "LinearRegression":       self._run_linear_regression,
            "Ridge":                  self._run_ridge,
            "Lasso":                  self._run_lasso,
            "SVR":                    self._run_svr,
        }
        func = dispatch.get(tool)
        if func:
            return func()
        return self._run_linear_regression() if self.task == "regression" else self._run_random_forest()

    # ── Feature/target helpers ────────────────────────────────────────────────

    def _get_X_y(self):
        """Return (X, y) with target and ID columns removed from X."""
        X = self.data.drop(columns=[self.target_column])
        id_cols = [c for c in X.columns if is_identifier_column(c)]
        if id_cols:
            logging.info(f"Dropping ID columns from model training: {id_cols}")
            X = X.drop(columns=id_cols)
        y = self.data[self.target_column]
        return X, y

    # ── Classification models ─────────────────────────────────────────────────

    def _run_random_forest(self) -> Dict[str, Any]:
        X, y = self._get_X_y()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        n = min(100, max(10, len(X_train) // 10))
        self.model = RandomForestClassifier(
            n_estimators=n, 
            max_depth=15,  # NEW: Add regularization to prevent overfitting
            min_samples_leaf=5,  # NEW: Require minimum samples per leaf
            min_samples_split=10,  # NEW: Require minimum samples to split
            random_state=42
        )
        self.model_name = "RandomForestClassifier"
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # NEW: Add cross-validation for more reliable performance estimate
        try:
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
            logging.info(
                f"RandomForestClassifier — test_accuracy={acc:.4f}, "
                f"cv_accuracy={cv_scores.mean():.4f}±{cv_scores.std():.4f}"
            )
        except Exception as e:
            logging.warning(f"Cross-validation failed: {e}")
            cv_scores = None
        
        results = {
            "model": "RandomForestClassifier", "accuracy": acc,
            "classification_report": classification_report(y_test, preds),
            "predictions": preds, "X_test": X_test, "y_test": y_test,
            "feature_importances": self.model.feature_importances_,
            "feature_names": X.columns.tolist(),
        }
        
        # Add cross-validation results if available
        if cv_scores is not None:
            results["cv_accuracy"] = cv_scores.mean()
            results["cv_std"] = cv_scores.std()
            results["cv_scores"] = cv_scores
        
        return results

    def _run_logistic_regression(self) -> Dict[str, Any]:
        X, y = self._get_X_y()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model_name = "LogisticRegression"
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logging.info(f"LogisticRegression — accuracy={acc:.4f}")
        return {
            "model": "LogisticRegression", "accuracy": acc,
            "classification_report": classification_report(y_test, preds),
            "predictions": preds, "X_test": X_test, "y_test": y_test,
            "feature_names": X.columns.tolist(),
        }

    def _run_svc(self) -> Dict[str, Any]:
        X, y = self._get_X_y()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.model = SVC(kernel=self.params.get("kernel", "rbf"), C=self.params.get("C", 1.0), random_state=42)
        self.model_name = "SVC"
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logging.info(f"SVC — accuracy={acc:.4f}")
        return {
            "model": "SVC", "accuracy": acc,
            "classification_report": classification_report(y_test, preds),
            "predictions": preds, "X_test": X_test, "y_test": y_test,
            "feature_names": X.columns.tolist(),
        }

    # ── Regression models ─────────────────────────────────────────────────────

    def _run_random_forest_regressor(self) -> Dict[str, Any]:
        X, y = self._get_X_y()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        data_summary = create_data_summary(self.data)
        hp = self.tool_decider.decide_hyperparameters("RandomForestRegressor", "regression", data_summary)
        self.model = RandomForestRegressor(
            n_estimators=hp.get("n_estimators", 100),
            max_depth=min(hp.get("max_depth", 15), 15),  # IMPROVED: Cap depth to prevent overfitting
            min_samples_leaf=max(hp.get("min_samples_leaf", 5), 5),  # IMPROVED: Enforce minimum leaf size
            min_samples_split=max(hp.get("min_samples_split", 10), 10),  # IMPROVED: Enforce minimum split size
            random_state=hp.get("random_state", 42),
            n_jobs=-1,
        )
        self.model_name = "RandomForestRegressor"
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mse, r2 = mean_squared_error(y_test, preds), r2_score(y_test, preds)
        
        # NEW: Add cross-validation for more reliable performance estimate
        try:
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
            logging.info(
                f"RandomForestRegressor — test_R²={r2:.4f}, "
                f"cv_R²={cv_scores.mean():.4f}±{cv_scores.std():.4f}"
            )
        except Exception as e:
            logging.warning(f"Cross-validation failed: {e}")
            cv_scores = None
        
        results = {
            "model": "RandomForestRegressor", "mse": mse, "r2": r2,
            "predictions": preds, "train_predictions": self.model.predict(X_train),
            "X_test": X_test, "y_test": y_test,
            "feature_importances": self.model.feature_importances_,
            "feature_names": X.columns.tolist(),
        }
        
        # Add cross-validation results if available
        if cv_scores is not None:
            results["cv_r2"] = cv_scores.mean()
            results["cv_std"] = cv_scores.std()
            results["cv_scores"] = cv_scores
        
        return results

    def _run_hist_gradient_boosting_regressor(self) -> Dict[str, Any]:
        X, y = self._get_X_y()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = HistGradientBoostingRegressor(random_state=42)
        self.model_name = "HistGradientBoostingRegressor"
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mse, r2 = mean_squared_error(y_test, preds), r2_score(y_test, preds)
        logging.info(f"HistGradientBoostingRegressor — R²={r2:.4f}, MSE={mse:.4f}")
        return {
            "model": "HistGradientBoostingRegressor", "mse": mse, "r2": r2,
            "predictions": preds, "train_predictions": self.model.predict(X_train),
            "X_test": X_test, "y_test": y_test, "feature_names": X.columns.tolist(),
        }

    def _run_linear_regression(self) -> Dict[str, Any]:
        X, y = self._get_X_y()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = LinearRegression()
        self.model_name = "LinearRegression"
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mse, r2 = mean_squared_error(y_test, preds), r2_score(y_test, preds)
        
        # NEW: Add cross-validation for more reliable performance estimate
        try:
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
            logging.info(
                f"LinearRegression — test_R²={r2:.4f}, "
                f"cv_R²={cv_scores.mean():.4f}±{cv_scores.std():.4f}"
            )
        except Exception as e:
            logging.warning(f"Cross-validation failed: {e}")
            cv_scores = None
        
        results = {
            "model": "LinearRegression", "mse": mse, "r2": r2,
            "predictions": preds, "train_predictions": self.model.predict(X_train),
            "X_test": X_test, "y_test": y_test, "feature_names": X.columns.tolist(),
        }
        
        # Add cross-validation results if available
        if cv_scores is not None:
            results["cv_r2"] = cv_scores.mean()
            results["cv_std"] = cv_scores.std()
            results["cv_scores"] = cv_scores
        
        return results

    def _run_ridge(self) -> Dict[str, Any]:
        X, y = self._get_X_y()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = Ridge(alpha=self.params.get("alpha", 1.0))
        self.model_name = "Ridge"
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mse, r2 = mean_squared_error(y_test, preds), r2_score(y_test, preds)
        
        # NEW: Add cross-validation for more reliable performance estimate
        try:
            cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='r2')
            logging.info(
                f"Ridge — test_R²={r2:.4f}, "
                f"cv_R²={cv_scores.mean():.4f}±{cv_scores.std():.4f}"
            )
        except Exception as e:
            logging.warning(f"Cross-validation failed: {e}")
            cv_scores = None
        
        results = {
            "model": "Ridge", "mse": mse, "r2": r2,
            "predictions": preds, "train_predictions": self.model.predict(X_train),
            "X_test": X_test, "y_test": y_test, "feature_names": X.columns.tolist(),
        }
        
        # Add cross-validation results if available
        if cv_scores is not None:
            results["cv_r2"] = cv_scores.mean()
            results["cv_std"] = cv_scores.std()
            results["cv_scores"] = cv_scores
        
        return results

    def _run_lasso(self) -> Dict[str, Any]:
        X, y = self._get_X_y()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = Lasso(alpha=self.params.get("alpha", 1.0))
        self.model_name = "Lasso"
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mse, r2 = mean_squared_error(y_test, preds), r2_score(y_test, preds)
        logging.info(f"Lasso — R²={r2:.4f}, MSE={mse:.4f}")
        return {
            "model": "Lasso", "mse": mse, "r2": r2,
            "predictions": preds, "train_predictions": self.model.predict(X_train),
            "X_test": X_test, "y_test": y_test, "feature_names": X.columns.tolist(),
        }

    def _run_svr(self) -> Dict[str, Any]:
        X, y = self._get_X_y()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = SVR(kernel=self.params.get("kernel", "rbf"), C=self.params.get("C", 1.0))
        self.model_name = "SVR"
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mse, r2 = mean_squared_error(y_test, preds), r2_score(y_test, preds)
        logging.info(f"SVR — R²={r2:.4f}, MSE={mse:.4f}")
        return {
            "model": "SVR", "mse": mse, "r2": r2,
            "predictions": preds, "train_predictions": self.model.predict(X_train),
            "X_test": X_test, "y_test": y_test, "feature_names": X.columns.tolist(),
        }

    # ── Anomaly detection ─────────────────────────────────────────────────────

    def _run_anomaly_detection(self) -> Dict[str, Any]:
        """Run IsolationForest with params from self.params (suggested by SLM 3b)."""
        contamination = self.params.get("contamination", "auto")
        n_estimators = self.params.get("n_estimators", 200)
        random_state = self.params.get("random_state", 42)
        logging.info(f"IsolationForest — contamination={contamination}, n_estimators={n_estimators}")

        X = self.data.copy()
        id_cols = [c for c in X.columns if is_identifier_column(c)]
        if id_cols:
            logging.info(f"Excluding ID columns from anomaly model: {id_cols}")
            X = X.drop(columns=id_cols)

        self.model = IsolationForest(
            contamination=contamination, n_estimators=n_estimators, random_state=random_state
        )
        self.model_name = "IsolationForest"
        labels = self.model.fit_predict(X)
        scores = self.model.score_samples(X)

        n_anomalies = int((labels == -1).sum())
        logging.info(f"IsolationForest — {n_anomalies}/{len(X)} anomalies ({n_anomalies / len(X):.2%})")

        results_df = self.data.copy()
        # Canonical anomaly columns used by downstream OptimizationAgent.
        results_df['Is_Anomaly'] = (labels == -1)
        results_df['Anomaly_Score'] = scores

        # Backward-compatible aliases for older reporting paths.
        results_df['anomaly_label'] = labels
        results_df['anomaly_score'] = scores

        # Per-feature z-scores
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            z_scores = (X[numeric_cols] - X[numeric_cols].mean()) / X[numeric_cols].std(ddof=0).replace(0, np.nan)
            # Canonical naming: <feature>_zscore
            results_df[[f"{c}_zscore" for c in numeric_cols]] = z_scores
            # Backward-compatible naming: z_<feature>
            results_df[[f"z_{c}" for c in numeric_cols]] = z_scores

        return {
            "model": "IsolationForest",
            "n_anomalies": n_anomalies,
            "anomaly_rate": float(n_anomalies / len(X)),
            "anomaly_labels": labels,
            "anomaly_scores": scores,
            "results_df": results_df,
            "feature_names": X.columns.tolist(),
        }
