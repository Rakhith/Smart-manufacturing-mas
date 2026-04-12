"""
main_llm.py
-----------
Smart Manufacturing Multi-Agent System — main entry point.

Two orchestration modes selectable at runtime via --mode:

  --mode llm          (default) Original LLM-orchestrated workflow.
                                Cloud LLM decides which agent to call at each step.

  --mode rules-first            NEW — Proposed Next Architecture.
                                All 4 pipeline steps run deterministically first.
                                Cloud LLM called once at the end for Reflexion summary.

New flags (all optional, all backward-compatible):
  --auto-detect         Auto-detect problem type from dataset statistics.
  --target              Target column name (used with --auto or --rules-first).
  --features            Feature column names (space-separated).
  --problem-type        Override: 'classification' | 'regression' | 'anomaly_detection'.
  --use-pca             Enable optional PCA in preprocessing.
  --pca-threshold       Variance threshold for PCA (default: 0.95).
  --use-cache           Enable model persistence via hash-keyed ModelCache.
  --cache-dir           Cache directory (default: ./model_cache).
  --invalidate-cache    Delete cache for current config, then exit.

Architecture (Three-Tier Intelligence Hierarchy):
  TIER 1  Cloud LLM (Gemini 2.5-Flash)  — orchestration OR Reflexion summary
  TIER 2  Local SLM (Qwen3:4B / Ollama) — anomaly params ONLY (SLM 3b retained)
  TIER 3  Rule-Based ToolDecider         — preprocessing + model selection

SLM Reduction (4 → 1):
  SLM 1 (Perception)   : ELIMINATED — pandas + HITL
  SLM 2 (Preprocessing): ELIMINATED — ToolDecider
  SLM 3a (Model select): ELIMINATED — ToolDecider
  SLM 3b (Anomaly params): RETAINED  ← only remaining SLM call
  SLM 4 (Summary)      : ELIMINATED — Cloud LLM Reflexion loop
"""

import argparse
import json
import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from utils.llm_output_logger import log_llm_output

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Main] - %(message)s",
)


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Smart Manufacturing MAS — LLM-powered workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode
    p.add_argument(
        "--mode", choices=["llm", "rules-first"], default="llm",
        help="'llm' = LLM decides each step (original). 'rules-first' = deterministic pipeline, LLM interprets at end.",
    )

    # LLM backends
    p.add_argument("--planner-llm", default="gemini", help="Planner LLM: 'gemini' | 'ollama' | 'llamacpp' | 'mock'.")
    p.add_argument("--planner-model", default=None, help="Model tag / path for local planner LLM.")
    p.add_argument("--decision-llm", default=None, help="SLM backend for anomaly params: 'ollama' | 'llamacpp' | 'mock' | None.")
    p.add_argument("--decision-model", default=None, help="Model tag / path for decision SLM (e.g. 'qwen3:4b').")

    # Dataset
    p.add_argument("--dataset", default=None, help="Path to dataset (.csv or .npz).")
    p.add_argument("--auto", action="store_true", help="Non-interactive — auto-approve all HITL gates.")
    p.add_argument("--batch", action="store_true", help="Process all datasets under ./data/ (implies --auto).")

    # Auto-detect (NEW)
    p.add_argument(
        "--auto-detect", action="store_true",
        help="NEW: auto-detect problem type from target column statistics.",
    )
    p.add_argument("--target", default=None, help="Target column name.")
    p.add_argument("--features", nargs="*", default=None, help="Feature column names (space-separated).")
    p.add_argument(
        "--problem-type", choices=["classification", "regression", "anomaly_detection"], default=None,
        help="Problem type override (skips auto-detection).",
    )

    # PCA (NEW)
    p.add_argument(
        "--use-pca", action="store_true",
        help="Enable PCA after preprocessing. WARNING: loses named feature interpretability.",
    )
    p.add_argument("--pca-threshold", type=float, default=0.95, help="Variance to retain when --use-pca is set.")

    # Model cache (NEW)
    p.add_argument(
        "--use-cache", action="store_true",
        help="Cache trained models keyed by hash(dataset+features+target+task).",
    )
    p.add_argument("--cache-dir", default=None, help="Model cache directory (default: ./model_cache).")
    p.add_argument(
        "--invalidate-cache", action="store_true",
        help="Delete cache for the current config, then exit.",
    )

    # Pretrained inference (NEW)
    p.add_argument(
        "--inference-only", action="store_true",
        help="Use pre-trained model bundles for supervised tasks instead of live training.",
    )
    p.add_argument(
        "--train-live", action="store_true",
        help="Force live training for supervised tasks (overrides pretrained inference default).",
    )
    p.add_argument(
        "--pretrained-dir", default="artifacts/pretrained_models",
        help="Directory containing pretrained model bundles and registry.json.",
    )
    p.add_argument(
        "--preferred-model", default=None,
        help="Optional model name to force during pretrained inference (e.g., 'Ridge').",
    )

    # Interface
    p.add_argument("--interface", default="cli", help="HITL interface: 'cli' or 'web'.")

    return p


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_llm_agent(backend: str | None, model: str | None):
    """Return a LocalLLMAgent or None (→ use Gemini directly)."""
    if backend in ("ollama", "mock", "llamacpp", "transformers"):
        from agents.local_llm_agent import LocalLLMAgent
        return LocalLLMAgent(backend=backend, model_name=model)
    return None


def _get_gemini_model():
    """Initialise and return a Gemini GenerativeModel, or None if key missing."""
    import google.generativeai as genai
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.warning("GEMINI_API_KEY not set — Cloud LLM features unavailable.")
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def _get_anomaly_params(args, csv_path: str, feature_cols, hitl_interface) -> dict:
    """
    Use the retained SLM 3b to suggest IsolationForest hyperparameters.
    Falls back to sklearn defaults if the SLM is unavailable.
    HITL gate always shown before params are used.
    """
    if args.decision_llm is None:
        return {}

    decision_agent = _make_llm_agent(args.decision_llm, args.decision_model)
    if decision_agent is None:
        return {}

    try:
        from agents.data_loader_agent import DataLoaderAgent
        sample = DataLoaderAgent.load_dataframe(csv_path, nrows=1000)
        if feature_cols:
            sample = sample[[c for c in feature_cols if c in sample.columns]]
        desc = sample.describe().to_string()
    except Exception:
        desc = "dataset summary unavailable"

    prompt = (
        "You are configuring an IsolationForest for anomaly detection.\n"
        f"Dataset profile:\n{desc}\n\n"
        "Suggest hyperparameters as JSON: "
        "{\"contamination\": <float 0.001-0.2 or 'auto'>, \"n_estimators\": <int 100-400>, \"reason\": \"...\"}"
    )

    import numpy as np

    def _extract_json_from_text(text: str):
        if not text or not isinstance(text, str):
            return None
        decoder = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        return None

    def _build_reason(reason_value, contamination_value, n_est_value, source: str) -> str:
        candidate = str(reason_value).strip() if reason_value is not None else ""
        if candidate:
            return f"{candidate} (by {source})"
        return (
            f"Fallback reason: contamination={contamination_value}, n_estimators={n_est_value} "
            f"derived from dataset summary and bounded safety defaults (by {source})"
        )

    try:
        resp = decision_agent.generate(prompt)
        raw_output = resp.get("raw")
        parsed = resp.get("parsed")
        if not isinstance(parsed, dict) or not parsed:
            parsed = _extract_json_from_text(raw_output or "") or {}
        raw_cont = parsed.get("contamination", "auto")
        try:
            contamination = (
                float(np.clip(float(raw_cont), 0.001, 0.2))
                if isinstance(raw_cont, (int, float))
                   or (isinstance(raw_cont, str) and raw_cont.strip().lower() != "auto")
                else "auto"
            )
        except (ValueError, TypeError):
            contamination = "auto"
        n_est = max(50, int(float(parsed.get("n_estimators", 200))))
        reason = _build_reason(parsed.get("reason"), contamination, n_est, "decision LLM")
        params = {"contamination": contamination, "n_estimators": n_est, "reason": reason}

        log_llm_output(
            {
                "stage": "anomaly_param_suggestion",
                "source": "main_llm._get_anomaly_params",
                "backend": args.decision_llm,
                "model": args.decision_model,
                "prompt": prompt,
                "raw_output": raw_output,
                "parsed_output": parsed,
                "effective_params": params,
                "parse_status": "parsed" if parsed else "fallback",
            }
        )
    except Exception as exc:
        logging.warning(f"SLM anomaly param suggestion failed ({exc}); using defaults.")
        params = {
            "contamination": "auto",
            "n_estimators": 200,
            "reason": "Fallback reason: decision LLM unavailable or unparsable output, using sklearn-safe defaults.",
        }
        log_llm_output(
            {
                "stage": "anomaly_param_suggestion",
                "source": "main_llm._get_anomaly_params",
                "backend": args.decision_llm,
                "model": args.decision_model,
                "prompt": prompt,
                "raw_output": None,
                "parsed_output": None,
                "effective_params": params,
                "parse_status": "exception",
                "error": str(exc),
            }
        )

    # HITL gate for SLM 3b
    if not (args.auto or args.batch):
        logging.info(f"SLM suggested anomaly params: {params}")
        ans = hitl_interface.prompt_with_audit(
            f"Approve anomaly params? contamination={params['contamination']}, "
            f"n_estimators={params['n_estimators']}",
            options=["approve", "modify"],
            context={"step": "anomaly_params_gate"},
        )
        if ans == "modify":
            cont = input("Enter contamination (number or 'auto'): ").strip()
            ne = input("Enter n_estimators (int): ").strip()
            try:
                params["contamination"] = cont if cont == "auto" else float(cont)
                params["n_estimators"] = int(ne)
            except ValueError:
                logging.warning("Invalid input — keeping SLM suggestion.")

    return params


# ── Rules-first mode (NEW) ────────────────────────────────────────────────────

def _run_rules_first(args, hitl_interface):
    """Execute the new Rules-First pipeline."""
    from agents.rules_first_planner import RulesFirstPlannerAgent
    from utils.model_cache import ModelCache

    csv_path = args.dataset
    if csv_path is None:
        logging.error("--dataset is required for --mode rules-first.")
        sys.exit(1)

    llm_model = _get_gemini_model() if args.planner_llm == "gemini" else None

    # Handle --invalidate-cache
    if args.invalidate_cache:
        from utils.model_cache import ModelCache
        cache = ModelCache(cache_dir=args.cache_dir)
        target = args.target
        features = args.features or []
        problem = args.problem_type or "classification"
        removed = cache.invalidate(csv_path, features, target, problem)
        logging.info(f"Cache invalidated: {removed}")
        return

    anomaly_params = _get_anomaly_params(args, csv_path, args.features, hitl_interface)

    planner = RulesFirstPlannerAgent(
        dataset_path=csv_path,
        feature_columns=args.features,
        target_column=args.target,
        problem_type=args.problem_type,          # None → auto-detect
        llm_model=llm_model,
        hitl_interface=hitl_interface,
        use_pca=args.use_pca,
        pca_variance_threshold=args.pca_threshold,
        use_cache=args.use_cache,
        cache_dir=args.cache_dir,
        anomaly_params=anomaly_params,
        auto_hitl=args.auto or args.batch,
        inference_only=(not args.train_live) or args.inference_only,
        pretrained_dir=args.pretrained_dir,
        preferred_model=args.preferred_model,
    )

    result = planner.run_workflow()
    logging.info(f"Rules-first workflow status: {result['status']}")
    if result.get("timings"):
        logging.info(f"Timings: {result['timings']}")


# ── LLM-orchestrated mode (original) ─────────────────────────────────────────

def _run_llm_mode(args, hitl_interface):
    """Execute the original LLM-orchestrated workflow (LLMPlannerAgent)."""
    from agents.llm_planner_agent import LLMPlannerAgent
    from utils.schema_discovery import discover_dataset_schema

    if args.auto or args.batch:
        os.environ["HITL_AUTO"] = "1"

    def _auto_select(csv_path: str):
        """Select target/features/problem automatically from schema discovery."""
        from agents.data_loader_agent import DataLoaderAgent
        df = DataLoaderAgent.load_dataframe(csv_path)

        # --auto-detect uses the new auto_detect utility
        if args.auto_detect:
            from utils.auto_detect import auto_detect_problem_type, suggest_target_column, log_detection_result
            target = args.target or suggest_target_column(df)
            ptype, conf, reasoning = auto_detect_problem_type(df, target)
            log_detection_result(ptype, conf, reasoning)
            features = args.features or [c for c in df.columns if c != target]
            return features, target, ptype

        # Original schema_discovery path
        schema = discover_dataset_schema(df)
        suggested = schema.get("suggested_targets", []) or []
        if suggested:
            best = sorted(suggested, key=lambda x: x.get("score", 0), reverse=True)[0]
            target = best.get("column")
            problem = best.get("suggested_task", "classification")
        else:
            last_col = df.columns[-1]
            target = last_col
            problem = (
                "classification"
                if str(df[last_col].dtype) in ["object", "category"] or df[last_col].nunique() <= 20
                else "regression"
            )
        cols_info = schema.get("columns", {})
        features = [
            c for c in df.columns
            if c != target and cols_info.get(c, {}).get("role") not in ["identifier", "timestamp"]
        ]
        return features, target, problem

    def _launch(csv_path, features, target, problem):
        goal = (
            f"Load the selected dataset, preprocess it, analyse it to solve a "
            f"{problem} problem, and generate a prescriptive action plan."
        )
        llm_agent = _make_llm_agent(args.planner_llm, args.planner_model)
        decision_llm_agent = _make_llm_agent(args.decision_llm, args.decision_model)
        try:
            planner = LLMPlannerAgent(
                dataset_path=csv_path,
                feature_columns=features,
                target_column=target,
                problem_type=problem,
                llm_agent=llm_agent,
                decision_llm_agent=decision_llm_agent,
                hitl_interface=hitl_interface,
            )
            planner.run_workflow_with_llm(goal)
        except Exception as exc:
            logging.error(f"LLM workflow error: {exc}", exc_info=True)

    def run_single(csv_path: str):
        if args.auto or args.auto_detect:
            features, target, problem = _auto_select(csv_path)
            _launch(csv_path, features, target, problem)
        else:
            dataset_path, features, target, problem = LLMPlannerAgent.interactive_setup(hitl_interface)
            _launch(dataset_path, features, target, problem)

    if args.batch:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        all_csvs = [
            os.path.join(root, f)
            for root, _, files in os.walk(data_dir)
            for f in files if f.lower().endswith((".csv", ".npz"))
        ]
        logging.info(f"Batch mode: {len(all_csvs)} datasets found.")
        for csv in all_csvs:
            logging.info(f"── Processing: {csv}")
            run_single(csv)
    elif args.dataset:
        run_single(args.dataset)
    else:
        run_single(None)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.auto or args.batch:
        os.environ["HITL_AUTO"] = "1"

    from utils.hitl_interface import get_hitl_interface
    hitl_interface = get_hitl_interface(args.interface)

    logging.info(f"Mode: {args.mode}")
    logging.info(
        f"Flags: auto_detect={args.auto_detect}, use_pca={args.use_pca}, use_cache={args.use_cache}"
    )

    if args.mode == "rules-first":
        _run_rules_first(args, hitl_interface)
    else:
        _run_llm_mode(args, hitl_interface)

    logging.info("MAS application finished.")


if __name__ == "__main__":
    main()
