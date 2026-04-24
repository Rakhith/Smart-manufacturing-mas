# Repo Summary

## 1. Repository Purpose
Smart Manufacturing MAS is a multi-agent ML system for prescriptive maintenance workflows. It supports:
- Dataset loading and schema inspection
- Deterministic preprocessing and model analysis
- Optimization/recommendation generation
- LLM-generated workflow summaries
- Local web execution and artifact management

It is designed to run in two orchestration modes:
- `llm` mode: cloud/local planner decides steps
- `rules-first` mode: deterministic pipeline first, LLM interpretation at the end

## 2. High-Level Workflow

### A) CLI Workflow (`main_llm.py`)
1. Parse runtime flags (dataset path, mode, target/problem settings, cache/PCA, LLM backends).
2. Build planner/decision model interfaces (Gemini and/or local SLM as configured).
3. Resolve task setup:
- target/problem selection (auto-detect or provided)
- anomaly parameter suggestion (optional SLM path)
4. Execute pipeline:
- `llm` mode -> planner agents orchestrate each step
- `rules-first` mode -> deterministic steps run first, then summary is generated
5. Emit outputs:
- model metrics/results
- recommendations
- optional logs/cache artifacts

### B) Rules-First Internal Flow (`agents/rules_first_planner.py`)
1. Step 0: Problem/target resolution with auto-detection and HITL fallback.
2. Step 1: Data loading and inspection.
3. Step 2: Preprocessing (cleaning, encoding/scaling, optional PCA).
4. Step 3: Model analysis/training or inference selection.
5. Step 4: Optimization and recommendation generation.
6. Step 5: Reflexion-based summary generation (cloud first, local fallback, plain-text fallback).

### C) Local Web App Flow (`webapp/app.py` + `webapp/run_manager.py`)
1. User uploads/selects dataset in UI.
2. API creates `RunConfig` and enqueues a run.
3. `run_manager` executes workflow in background thread.
4. Stage-by-stage progress is stored under web run artifacts.
5. Frontend polls run status and renders:
- data profile
- model outputs
- recommendations
- workflow summary
6. Optional synthetic dataset generation and re-run supported.

## 3. Directory Map and Storage Semantics

### Workspace Root
- `logs/`: external/runtime logs at workspace level.
- `mas_venv/`: Python virtual environment (dependencies, scripts, site-packages).
- `smart_manufacturing_mas/`: main project source tree.

### `smart_manufacturing_mas/`
- `.env`: local secret/config values (for example `GEMINI_API_KEY`).
- `.env.example`: template for required environment variables.
- `.gitignore`: git exclusions.
- `main_llm.py`: CLI entry point and mode orchestration.
- `README.md`: primary usage and setup guide.
- `QUICKSTART.md`: fast setup and run commands.
- `SYNTHETIC_DATA_GUIDE.md`: synthetic-data workflow instructions.
- `Repo Summary.md`: this repository-wide architecture and file summary.
- `requirements.txt`: Python dependency lock list for runtime.

### `agents/`
- `__init__.py`: package initializer.
- `data_loader_agent.py`: dataset loading, basic schema/inspection utilities.
- `preprocessing_agent.py`: preprocessing pipeline (cleaning, transforms, optional PCA).
- `dynamic_analysis_agent.py`: model analysis/training execution logic.
- `optimization_agent.py`: recommendation scoring and optimization outputs.
- `planner_agent.py`: original planner orchestration utilities.
- `llm_planner_agent.py`: LLM-assisted planning and JSON parser logic.
- `rules_first_planner.py`: deterministic orchestrator for the recommended architecture.
- `local_llm_agent.py`: local backend wrapper (`ollama`, `llamacpp`, `transformers`, `mock`).

### `utils/`
- `__init__.py`: package initializer.
- `auto_detect.py`: automatic target/problem-type detection helpers.
- `column_utils.py`: column-role helpers (identifier detection, etc.).
- `hitl_interface.py`: human-in-the-loop interaction abstraction.
- `intelligent_feature_analysis.py`: feature signal/quality analysis.
- `intelligent_summarization.py`: reflexion summary generation and sanitization.
- `llm_output_logger.py`: structured logging for LLM decisions/outputs.
- `model_cache.py`: hash-based model caching and retrieval.
- `prediction_analyzer.py`: prediction quality diagnostics for regression/classification.
- `pretrained_model_store.py`: registry/bundle selection and inference utilities.
- `reporting.py`: report/summary assembly utilities.
- `schema_discovery.py`: schema profiling and dataset metadata extraction.
- `synthetic_quality_analyzer.py`: synthetic vs source-data quality comparison.
- `tool_decider.py`: deterministic tool/model decision rules.

### `webapp/`
- `__init__.py`: package initializer.
- `app.py`: FastAPI app (dataset APIs, run APIs, synthetic generation APIs).
- `run_manager.py`: background run state machine, artifacts, synthetic generation orchestration.
- `static/app.js`: frontend behavior and API polling.
- `static/app.css`: UI styling.
- `templates/index.html`: web interface template.

### `scripts/`
- `run_local_app.py`: starts uvicorn local server for web app.
- `generate_synthetic_data_and_infer.py`: synthetic data generation and inference workflow script.
- `test_complete_flow.py`: end-to-end functional validation script.

### `training/`
- `offline_model_training.ipynb`: notebook for exporting pretrained bundles.
- `synthetic_data_inference_analysis.ipynb`: notebook for synthetic inference analysis.
- `train_and_export_pretrained.py`: script for training/export workflows.
- `artifacts/`: training-produced outputs (model binaries/metadata if generated there).

### `data/`
- `smart_manufacturing_data.csv`: dataset variant for experiments.
- `smart_manufacturing_dataset.csv`: dataset variant for supervised tasks.
- `digital_manufacturing_dataset.csv`: additional dataset variant.
- `Intelligent Manufacturing Dataset/`: source dataset folder.
- `Smart Manufacturing Maintenance Dataset/`: source maintenance dataset folder.
- `superconductivty+data/`: auxiliary dataset folder.

### `artifacts/`
- `pretrained_models/`: model bundle files and `registry.json` for inference-only mode.
- `web_uploads/`: user-uploaded datasets from web app.
- `web_synthetic/`: generated synthetic datasets from web UI/API.
- `web_runs/`: per-run state/output snapshots for local app pipeline runs.

### Runtime/Cache Directories
- `logs/` (inside project): run logs and audit traces.
- `model_cache/`: cached model artifacts keyed by dataset/features/problem config.

## 4. Component-Level Behavior

### LLM/SLM Responsibilities
- Cloud LLM (Gemini): primary planner and/or final summary generation.
- Local SLM (`qwen3:4b` via Ollama): retained for specific local decision paths and fallback summary path.
- Deterministic rule system: used heavily to reduce hallucination risk in orchestration.

### Summary Generation Stack
- Primary: cloud reflexion loop (`draft -> critique -> revise`).
- Fallback: local reflexion through local model adapter.
- Final fallback: rule-based plain-text summary.
- Sanitization removes reasoning tags and keeps only user-visible summary content.

### HITL (Human In The Loop)
- Used when confidence or configuration ambiguity is high.
- Can be auto-approved via CLI flags for unattended runs.

## 5. What Is Stored Where
- Input datasets: `data/` and uploaded files in `artifacts/web_uploads/`.
- Synthetic datasets: `artifacts/web_synthetic/`.
- Pretrained bundles/registry: `artifacts/pretrained_models/`.
- Live run state and outputs for web sessions: `artifacts/web_runs/`.
- Cache of model fits: `model_cache/`.
- Logs/audit traces: `logs/` (project-level and workspace-level usage).

## 6. Notes on Current Repository State
This summary reflects the current cleaned repository after removing redundant docs/checkpoint artifacts. The retained documentation files are:
- `README.md`
- `QUICKSTART.md`
- `SYNTHETIC_DATA_GUIDE.md`
- `Repo Summary.md`
