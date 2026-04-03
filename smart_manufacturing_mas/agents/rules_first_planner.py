"""
agents/rules_first_planner.py
------------------------------
Implements the "Proposed Next Architecture" from the presentation.

  Current flow  : User → LLM decides each step → calls agents one by one.
  Proposed flow : Rules run first → structured results → LLM interprets once.

┌─────────────────────────────────────────────────────────────────────────┐
│  RULES-FIRST PIPELINE                                                   │
│                                                                         │
│  Step 0  Auto-detect or confirm problem type & target column.           │
│           HITL confirmation when detection confidence < 75%.            │
│                                                                         │
│  Step 1  DataLoaderAgent   — load & inspect CSV.                        │
│  Step 2  PreprocessingAgent — clean, scale, encode (+ optional PCA).   │
│  Step 3  DynamicAnalysisAgent — train model (+ optional cache).        │
│  Step 4  OptimizationAgent  — priority scores + recommendations.        │
│                                                                         │
│  Step 5  Cloud LLM (Gemini) interprets ALL collected structured         │
│           results and generates a narrative summary via the             │
│           Reflexion loop (draft → self-critique → revised draft).       │
│                                                                         │
│  HITL review gate after recommendations; all decisions logged.          │
└─────────────────────────────────────────────────────────────────────────┘

Key advantages over the original LLM-orchestrated approach:
  • Zero hallucination risk in pipeline orchestration (deterministic rules).
  • LLM latency paid only ONCE (step 5), not at every tool call.
  • Model caching drastically reduces repeat-run latency.
  • Auto-detect removes the need for the user to specify ML terminology.
  • Finish-sentinel bug is irrelevant — there is no sentinel needed.
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.data_loader_agent import DataLoaderAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.dynamic_analysis_agent import DynamicAnalysisAgent
from agents.optimization_agent import OptimizationAgent
from utils.auto_detect import (
    auto_detect_problem_type,
    log_detection_result,
    needs_hitl_confirmation,
    suggest_target_column,
)
from utils.model_cache import ModelCache
from utils.hitl_interface import HitlInterface, get_hitl_interface

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] - %(message)s')


class RulesFirstPlannerAgent:
    """
    Deterministic pipeline orchestrator that offloads *interpretation* (not orchestration)
    to the Cloud LLM.

    Args:
        dataset_path           : Path to the CSV dataset.
        feature_columns        : Feature column list. Pass None → auto-selected.
        target_column          : Target column name. Pass None → auto-suggest + auto-detect.
        problem_type           : 'classification'|'regression'|'anomaly_detection'.
                                 Pass None → auto-detection from data.
        llm_model              : Fitted Gemini GenerativeModel for Reflexion summary.
                                 Pass None → plain-text fallback summary.
        hitl_interface         : HITL interface (defaults to CLI).
        use_pca                : Enable PCA in preprocessing (default: False).
        pca_variance_threshold : Variance to retain when use_pca=True (default: 0.95).
        use_cache              : Enable model cache (default: False).
        cache_dir              : Cache directory (default: ./model_cache).
        anomaly_params         : Dict {'contamination': float|'auto', 'n_estimators': int}.
        auto_hitl              : Auto-approve all HITL gates (non-interactive / CI mode).
    """

    PERFORMANCE_THRESHOLDS = {
        "classification": 0.60,
        "regression":     0.10,
    }

    def __init__(
        self,
        dataset_path: str,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        problem_type: Optional[str] = None,
        llm_model: Optional[Any] = None,
        hitl_interface: Optional[HitlInterface] = None,
        use_pca: bool = False,
        pca_variance_threshold: float = 0.95,
        use_cache: bool = False,
        cache_dir: Optional[str] = None,
        anomaly_params: Optional[Dict[str, Any]] = None,
        auto_hitl: bool = False,
    ):
        self.dataset_path = dataset_path
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.problem_type = problem_type
        self.llm_model = llm_model
        self.hitl_interface = hitl_interface or get_hitl_interface("cli")
        self.use_pca = use_pca
        self.pca_variance_threshold = pca_variance_threshold
        self.anomaly_params = anomaly_params or {}
        self.auto_hitl = auto_hitl or bool(os.environ.get("HITL_AUTO"))

        self.model_cache: Optional[ModelCache] = (
            ModelCache(cache_dir=cache_dir) if use_cache else None
        )

        # Workflow state
        self.raw_data: Optional[pd.DataFrame] = None
        self.preprocessed_data: Optional[pd.DataFrame] = None
        self.analysis_results: Optional[Dict[str, Any]] = None
        self.recommendations = None
        self._timings: Dict[str, float] = {}

        logging.info(
            f"RulesFirstPlannerAgent initialised — "
            f"use_pca={use_pca}, use_cache={use_cache}, auto_hitl={self.auto_hitl}"
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def run_workflow(self) -> Dict[str, Any]:
        """
        Execute the full rules-first pipeline.

        Returns dict with keys:
          problem_type, target_column, feature_columns,
          analysis_results, recommendations, reflexion_summary, timings, status.
        """
        logging.info("=" * 70)
        logging.info("  RULES-FIRST PIPELINE — starting")
        logging.info("=" * 70)
        t0_total = time.time()

        if not self._step0_resolve_problem_type():
            return self._failed("step0_problem_type")
        if not self._step1_load_data():
            return self._failed("step1_load_data")
        if not self._step2_preprocess():
            return self._failed("step2_preprocess")
        if not self._step3_analyse():
            return self._failed("step3_analyse")
        if not self._step4_optimise():
            return self._failed("step4_optimise")

        reflexion_summary = self._step5_reflexion_summary()

        self._timings["total"] = round(time.time() - t0_total, 2)
        logging.info(f"  RULES-FIRST PIPELINE — finished in {self._timings['total']}s")
        logging.info("=" * 70)

        return {
            "problem_type":      self.problem_type,
            "target_column":     self.target_column,
            "feature_columns":   self.feature_columns,
            "analysis_results":  self.analysis_results,
            "recommendations":   self.recommendations,
            "reflexion_summary": reflexion_summary,
            "timings":           self._timings,
            "status":            "success",
        }

    # ── Steps ─────────────────────────────────────────────────────────────────

    def _step0_resolve_problem_type(self) -> bool:
        """Auto-detect or confirm the problem type and target column."""
        t0 = time.time()
        logging.info("── Step 0: Resolving problem type & target column")

        try:
            sample = pd.read_csv(self.dataset_path, nrows=200)
        except Exception as exc:
            logging.error(f"Cannot read dataset for auto-detection: {exc}")
            return False

        # Suggest target column if not provided
        if self.target_column is None and self.problem_type != "anomaly_detection":
            self.target_column = suggest_target_column(sample)
            logging.info(f"Auto-suggested target column: '{self.target_column}'")

        # Auto-detect problem type if not provided
        if self.problem_type is None:
            prob_type, confidence, reasoning = auto_detect_problem_type(sample, self.target_column)
            log_detection_result(prob_type, confidence, reasoning)

            if needs_hitl_confirmation(confidence) and not self.auto_hitl:
                self.problem_type = self._hitl_confirm_problem_type(prob_type, confidence, reasoning)
            else:
                self.problem_type = prob_type
                logging.info(
                    f"Auto-detected: '{self.problem_type}' (confidence={confidence:.0%}) — no HITL needed."
                )
        else:
            logging.info(f"Using provided problem type: '{self.problem_type}'")

        if self.problem_type == "anomaly_detection":
            self.target_column = None

        self._timings["step0_resolve"] = round(time.time() - t0, 3)
        return True

    def _step1_load_data(self) -> bool:
        """Load dataset and select feature + target columns."""
        t0 = time.time()
        logging.info("── Step 1: Data loading & inspection")

        agent = DataLoaderAgent(self.dataset_path)
        full_df = agent.load_data()
        if full_df is None:
            return False
        agent.inspect_data()

        # Auto-select feature columns if not specified
        if self.feature_columns is None:
            exclude = {self.target_column} if self.target_column else set()
            id_cols = {
                c for c in full_df.columns
                if any(kw in c.upper() for kw in ("_ID", "ID_")) or c.upper() == "ID"
            }
            self.feature_columns = [
                c for c in full_df.columns if c not in exclude and c not in id_cols
            ]
            logging.info(
                f"Auto-selected {len(self.feature_columns)} feature columns: "
                f"{self.feature_columns[:6]}{'...' if len(self.feature_columns) > 6 else ''}"
            )

        cols = list(self.feature_columns)
        if self.target_column and self.target_column in full_df.columns:
            cols.append(self.target_column)
        # Also keep ID columns for Machine_ID pass-through fix
        id_passthrough = [
            c for c in full_df.columns
            if ('ID' in c.upper() or c.lower() in ('machine', 'asset', 'unit'))
            and c not in cols
        ]
        cols = cols + id_passthrough

        missing = [c for c in cols if c not in full_df.columns]
        if missing:
            logging.error(f"Columns not found in dataset: {missing}")
            return False

        self.raw_data = full_df[[c for c in cols if c in full_df.columns]]
        logging.info(f"Dataset shape after column selection: {self.raw_data.shape}")
        self._timings["step1_load"] = round(time.time() - t0, 3)
        return True

    def _step2_preprocess(self) -> bool:
        """Clean, scale, encode, and optionally apply PCA."""
        t0 = time.time()
        logging.info("── Step 2: Preprocessing")

        if self.raw_data is None:
            logging.error("raw_data is None.")
            return False

        protected = list(self.feature_columns or [])
        # For anomaly detection, include ID columns as protected (pass-through).
        # For other problem types, ID columns should be dropped (not protected).
        if self.problem_type == "anomaly_detection":
            id_passthrough = [
                c for c in self.raw_data.columns
                if ('ID' in c.upper() or c.lower() in ('machine', 'asset', 'unit'))
                and c not in protected
                and c != self.target_column
            ]
            protected = protected + id_passthrough

        agent = PreprocessingAgent(
            data=self.raw_data,
            target_column=self.target_column,
            problem_type=self.problem_type,
            protected_columns=protected,
            use_pca=self.use_pca,
            pca_variance_threshold=self.pca_variance_threshold,
        )
        processed_features = agent.preprocess()

        if processed_features is None:
            logging.error("Preprocessing failed.")
            return False

        if self.problem_type != "anomaly_detection" and self.target_column is not None:
            target = self.raw_data[[self.target_column]]
            self.preprocessed_data = pd.concat([processed_features, target], axis=1)
        else:
            self.preprocessed_data = processed_features

        logging.info(f"Preprocessed shape: {self.preprocessed_data.shape}")
        self._timings["step2_preprocess"] = round(time.time() - t0, 3)
        return True

    def _step3_analyse(self) -> bool:
        """Train model (checking cache first), activate Adaptive Intelligence if needed."""
        t0 = time.time()
        logging.info("── Step 3: Model training & analysis")

        if self.preprocessed_data is None:
            logging.error("preprocessed_data is None.")
            return False

        params: Dict[str, Any] = {}
        if self.problem_type == "anomaly_detection":
            params = {
                "contamination": self.anomaly_params.get("contamination", "auto"),
                "n_estimators":  self.anomaly_params.get("n_estimators", 200),
            }

        agent = DynamicAnalysisAgent(
            data=self.preprocessed_data,
            target_column=self.target_column,
            task=self.problem_type,
            params=params,
            model_cache=self.model_cache,
            dataset_path=self.dataset_path,
            feature_columns=self.feature_columns or [],
        )
        results = agent.run(force_retry=False)

        # Adaptive Intelligence gate for poor initial performance
        if results and self.problem_type in self.PERFORMANCE_THRESHOLDS:
            threshold = self.PERFORMANCE_THRESHOLDS[self.problem_type]
            perf_key = "accuracy" if self.problem_type == "classification" else "r2"
            perf_val = results.get(perf_key, threshold)
            if perf_val < threshold and not results.get("from_cache"):
                logging.warning(
                    f"Initial {perf_key}={perf_val:.4f} below threshold {threshold} — "
                    "activating Adaptive Intelligence."
                )
                retry = True
                if not self.auto_hitl:
                    retry = self._hitl_ask_retry(perf_key, perf_val, threshold)
                if retry:
                    results = agent.run(force_retry=True)

        if results is None:
            logging.error("Analysis failed — no results returned.")
            return False

        self.analysis_results = results
        model_name = results.get("model", "unknown")
        from_cache = " [from cache]" if results.get("from_cache") else ""
        logging.info(f"Analysis complete — model='{model_name}'{from_cache}")
        if "accuracy" in results:
            logging.info(f"  accuracy={results['accuracy']:.4f}")
        if "r2" in results:
            logging.info(f"  R²={results['r2']:.4f}, MSE={results.get('mse', '?'):.6f}")
        if "n_anomalies" in results:
            logging.info(f"  anomalies={results['n_anomalies']} ({results.get('anomaly_rate', 0):.2%})")

        self._timings["step3_analyse"] = round(time.time() - t0, 3)
        return True

    def _step4_optimise(self) -> bool:
        """Generate ranked prescriptive recommendations with HITL review."""
        t0 = time.time()
        logging.info("── Step 4: Optimisation & recommendations")

        if self.analysis_results is None:
            logging.error("analysis_results is None.")
            return False

        if self.problem_type == "anomaly_detection":
            payload = {
                "results_df":     self.analysis_results["results_df"],
                "anomaly_labels": self.analysis_results["anomaly_labels"],
            }
        else:
            X_test = self.analysis_results.get("X_test")
            if X_test is None:
                logging.error("No X_test in analysis results.")
                return False
            original_ctx = self.raw_data.loc[X_test.index] if self.raw_data is not None else X_test
            fi = self.analysis_results.get("feature_importances")
            if fi is not None:
                import pandas as pd
                feature_names = self.analysis_results.get("feature_names", [])
                fi_df = pd.DataFrame({"feature": feature_names, "importance": fi}).sort_values(
                    "importance", ascending=False
                )
            else:
                fi_df = None
            payload = {
                "test_data":          original_ctx,
                "test_predictions":   self.analysis_results["predictions"],
                "train_predictions":  self.analysis_results.get("train_predictions"),
                "feature_importances": fi_df,
            }
            for k in ("accuracy", "r2", "mse"):
                if k in self.analysis_results:
                    payload[k] = self.analysis_results[k]

        try:
            opt_agent = OptimizationAgent(payload)
            self.recommendations = opt_agent.generate_recommendations()
            summary_report = opt_agent.generate_summary_report(self.recommendations)
            logging.info(f"\n{summary_report}")
        except Exception as exc:
            logging.error(f"Optimisation failed: {exc}", exc_info=True)
            return False

        # HITL review gate
        if not self.auto_hitl:
            self._hitl_review_recommendations()

        self._timings["step4_optimise"] = round(time.time() - t0, 3)
        return True

    def _step5_reflexion_summary(self) -> str:
        """
        Cloud LLM Reflexion loop:  draft → self-critique → revised summary.
        Falls back to plain-text when no LLM model is provided.
        """
        t0 = time.time()
        logging.info("── Step 5: Reflexion summary (Cloud LLM)")

        if self.llm_model is None:
            summary = self._plain_text_summary()
            logging.info("[Reflexion] No LLM available — plain-text summary used.")
            self._timings["step5_reflexion"] = round(time.time() - t0, 3)
            return summary

        context = self._build_reflexion_context()

        try:
            # ── Draft ─────────────────────────────────────────────────────
            draft_prompt = (
                "You are summarising the results of an automated smart-manufacturing "
                "prescriptive maintenance workflow for a non-technical operator.\n\n"
                "Workflow context (JSON):\n"
                f"{json.dumps(context, indent=2, default=str)}\n\n"
                "Write a concise narrative summary (≤ 200 words) covering:\n"
                "1. What dataset was analysed and what problem was solved.\n"
                "2. Key preprocessing decisions.\n"
                "3. Model selected and its performance metrics.\n"
                "4. Top 3 maintenance recommendations with priority and reason.\n"
                "5. Any reliability warnings the operator should know about.\n"
                "Be specific — reference actual numbers from the context."
            )
            draft = self.llm_model.generate_content(draft_prompt).text.strip()
            logging.info(f"[Reflexion] Draft generated ({len(draft)} chars).")

            # ── Critique ──────────────────────────────────────────────────
            critique_prompt = (
                "Review this workflow summary against the JSON context below.\n"
                "Identify any inaccuracies, missing critical numbers, or unclear statements (≤ 60 words).\n\n"
                f"Summary:\n{draft}\n\nContext:\n{json.dumps(context, indent=2, default=str)}"
            )
            critique = self.llm_model.generate_content(critique_prompt).text.strip()
            logging.info(f"[Reflexion] Critique: {critique[:120]}…")

            # ── Revised draft ─────────────────────────────────────────────
            revise_prompt = (
                "Revise the summary below by applying the critique. "
                "Keep the result to ≤ 200 words. Output only the revised summary.\n\n"
                f"Original summary:\n{draft}\n\nCritique:\n{critique}"
            )
            final_summary = self.llm_model.generate_content(revise_prompt).text.strip()
            logging.info(f"[Reflexion] Final summary ready ({len(final_summary)} chars).")

        except Exception as exc:
            logging.warning(f"[Reflexion] LLM call failed ({exc}); using plain-text fallback.")
            final_summary = self._plain_text_summary()

        self._timings["step5_reflexion"] = round(time.time() - t0, 3)
        logging.info("\n" + "=" * 60)
        logging.info("INTELLIGENT WORKFLOW SUMMARY")
        logging.info("=" * 60)
        logging.info(final_summary)
        logging.info("=" * 60)
        return final_summary

    # ── HITL helpers ─────────────────────────────────────────────────────────

    def _hitl_confirm_problem_type(self, detected: str, confidence: float, reasoning: Dict) -> str:
        options = ["classification", "regression", "anomaly_detection"]
        msg = (
            f"Auto-detected problem type: '{detected}' (confidence={confidence:.0%}).\n"
            f"Reason: {reasoning['reason']}\nConfirm or override:"
        )
        chosen = self.hitl_interface.prompt_with_audit(
            msg, options=options, context={"step": "confirm_problem_type", "detected": detected}
        )
        logging.info(f"[HITL] Operator confirmed problem type: '{chosen}'")
        return chosen

    def _hitl_ask_retry(self, metric: str, value: float, threshold: float) -> bool:
        ans = self.hitl_interface.prompt_with_audit(
            f"Model {metric}={value:.4f} is below threshold {threshold}. "
            "Activate Adaptive Intelligence (try all model families)?",
            options=["yes", "no"],
            context={"step": "adaptive_intelligence_gate", "metric": metric, "value": value},
        )
        return ans == "yes"

    def _hitl_review_recommendations(self):
        if self.recommendations is None or (
            hasattr(self.recommendations, "empty") and self.recommendations.empty
        ):
            return
        n = len(self.recommendations)
        ans = self.hitl_interface.prompt_with_audit(
            f"Review the {n} prescriptive recommendations above. Approve to proceed.",
            options=["approve", "reject"],
            context={"step": "recommendation_review", "n_recommendations": n},
        )
        logging.info(f"[HITL] Recommendation review: {ans}")

    # ── Utility helpers ───────────────────────────────────────────────────────

    def _build_reflexion_context(self) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {
            "dataset":       os.path.basename(self.dataset_path),
            "problem_type":  self.problem_type,
            "target_column": self.target_column,
            "timings_s":     self._timings,
        }
        if self.analysis_results:
            ctx["model"] = self.analysis_results.get("model")
            for k in ("accuracy", "r2", "mse", "n_anomalies", "anomaly_rate"):
                if k in self.analysis_results:
                    val = self.analysis_results[k]
                    ctx[k] = round(float(val), 4) if isinstance(val, (float, np.floating)) else val
        if self.recommendations is not None:
            try:
                recs = (
                    self.recommendations.to_dict("records")
                    if hasattr(self.recommendations, "to_dict")
                    else self.recommendations
                )
                ctx["top_3_recommendations"] = recs[:3] if recs else []
            except Exception:
                ctx["top_3_recommendations"] = str(self.recommendations)[:500]
        return ctx

    def _plain_text_summary(self) -> str:
        lines = [
            f"Dataset       : {os.path.basename(self.dataset_path)}",
            f"Problem type  : {self.problem_type}",
            f"Target column : {self.target_column or 'N/A (unsupervised)'}",
        ]
        if self.analysis_results:
            model = self.analysis_results.get("model", "unknown")
            cached = "[cached] " if self.analysis_results.get("from_cache") else ""
            lines.append(f"Model trained : {cached}{model}")
            if "accuracy" in self.analysis_results:
                lines.append(f"Accuracy      : {self.analysis_results['accuracy']:.4f}")
            if "r2" in self.analysis_results:
                lines.append(f"R²            : {self.analysis_results['r2']:.4f}")
            if "n_anomalies" in self.analysis_results:
                lines.append(
                    f"Anomalies     : {self.analysis_results['n_anomalies']} "
                    f"({self.analysis_results.get('anomaly_rate', 0):.2%})"
                )
        lines.append(f"Total runtime : {self._timings.get('total', '?')}s")
        return "\n".join(lines)

    def _failed(self, step: str) -> Dict[str, Any]:
        return {
            "status":            f"failed_at_{step}",
            "problem_type":      self.problem_type,
            "target_column":     self.target_column,
            "analysis_results":  None,
            "recommendations":   None,
            "reflexion_summary": "",
            "timings":           self._timings,
        }
