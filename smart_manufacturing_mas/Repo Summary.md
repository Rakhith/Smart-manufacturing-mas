# Repo Summary

## 1. Repository Purpose
Smart Manufacturing MAS is a multi-agent cyber-physical platform engineered for predictive analytics and prescriptive maintenance workflows in smart manufacturing environments. It supports:
- Dataset loading, schema inspection, and data quality profiling
- Deterministic preprocessing and reproducible feature transformations
- Dynamic multi-model analysis (Classification, Regression, Isolation Forest Anomaly Detection)
- Pretrained model bundle inference (`artifacts/pretrained_models`) and hash-keyed caching
- Autonomous Prescriptive Action Ranking (`LGBMRanker` + Collaborative Filtering)
- Self-explaining interpretability via local `TreeSHAP` and Case-Based historical incident retrieval
- Autonomous execution state machine and closed-loop post-intervention feedback logging
- Cloud LLM / Local SLM Reflexion workflow summaries
- Interactive local Web Dashboard (`webapp/`) with asynchronous background state management and synthetic data generation

---

## 2. High-Level Workflows

### A) Web Application Flow (`webapp/app.py` + `webapp/run_manager.py`)
1. User selects a built-in dataset or uploads a custom CSV in the web UI.
2. Web UI configures task parameters (problem type, pretrained inference vs. live training, caching, PCA).
3. API dispatches a `RunConfig` to the background thread runner (`RunManager`).
4. Pipeline executes stage-by-stage:
   - `Step 1: DataLoaderAgent` (Schema & Quality Profiling)
   - `Step 2: PreprocessingAgent` (Transformations & Scaling)
   - `Step 3: DynamicAnalysisAgent` (Model Performance & Diagnostics)
   * `Step 4: OptimizationAgent` (Prescriptive Action Ranking & Explainability)
   - `Step 5: Reflexion Summary` (Intelligent Narrative Synthesis)
5. Frontend polls `/api/run/{run_id}` for state updates and live log feeds.
6. Frontend renders interactive charts, recommendation tables, and download links for execution artifacts under `artifacts/web_runs/{run_id}/`.
7. Users can generate synthetic datasets via `/api/generate-synthetic` and immediately trigger inference.

### B) Rules-First CLI Workflow (`agents/rules_first_planner.py`)
1. **Step 0**: Statistical auto-detection of problem type and target column (with automated or HITL fallback).
2. **Step 1**: Dataset loading, quality profiling, and column subset selection (`DataLoaderAgent`).
3. **Step 2**: Pipeline preprocessing, categorical encoding, scaling, and optional PCA (`PreprocessingAgent`).
4. **Step 3**: Pretrained bundle inference or multi-family model training with caching and Adaptive Intelligence retry (`DynamicAnalysisAgent`).
5. **Step 4**: Prescriptive action ranking and explainability generation (`OptimizationAgent`).
6. **Step 5**: Reflexion-based workflow summary loop (Cloud LLM $\rightarrow$ Local SLM $\rightarrow$ Plain text fallback).

### C) Autonomous Closed-Loop Architecture (Semester 7 Pivot)
```
[Telemetry Ingestion] ──► [Dynamic Analysis] ──► [Learned Recommender (LGBMRanker + CF)]
        ▲                                                    │
        │                                                    ▼
[State & Outcome Feedback] ◄── [Automated Action Execution] ◄── [Explainability Layer (SHAP + CBR)]
```
- **Learned Recommender**: Replaces static if-else keyword heuristics with `LGBMRanker` (LambdaMART) per asset query group + multi-sensor archetype cosine similarity.
- **Explainability Layer**: Local `TreeSHAP` attributions + Case-Based historical incident retrieval provide auditable rationales.
- **Autonomous Execution State Machine**: Dispatches actions within validated safety boundaries and tracks post-intervention sensor recovery ($\Delta \text{Recovery}$ over horizon $H$) into the feedback store.

---

## 3. Directory Map and Storage Semantics

### Workspace Root
- `logs/`: Project runtime logs, audit traces, and workflow summaries.
- `mas_venv/`: Python virtual environment.
- `smart_manufacturing_mas/`: Main source tree.

### `smart_manufacturing_mas/`
- `.env`: Local environment variables (`GEMINI_API_KEY`, `LOG_LEVEL`).
- `.env.example`: Configuration template for environment variables.
- `.gitignore`: Git exclusions.
- `main_llm.py`: Unified CLI entry point.
- `README.md`: Primary platform guide and architecture overview.
- `QUICKSTART.md`: Fast 5-minute setup and CLI/GUI cheat sheet.
- `SYNTHETIC_DATA_GUIDE.md`: Synthetic telemetry generation and validation manual.
- `Repo Summary.md`: Comprehensive codebase architecture and file guide.
- `requirements.txt`: Python dependency lock manifest.

### `agents/`
- `data_loader_agent.py`: Ingestion, schema discovery, sampling, ID column detection.
- `preprocessing_agent.py`: Pipeline-based cleaning, encoding, scaling, and optional PCA.
- `dynamic_analysis_agent.py`: Supervised model training, pretrained inference, adaptive retry.
- `optimization_agent.py`: Prescriptive maintenance recommendation and ranking engine.
- `rules_first_planner.py`: Deterministic orchestrator for rules-first workflow.
- `llm_planner_agent.py`: LLM-orchestrated planner and interactive setup engine.
- `local_llm_agent.py`: Local SLM backend wrapper (Ollama, LlamaCpp, HuggingFace).
- `planner_agent.py`: Rule-based emergency fallback orchestrator.

### `utils/`
- `auto_detect.py`: Statistical problem-type and target column auto-detection.
- `column_utils.py`: Identifier column identification and column-role utilities.
- `hitl_interface.py`: CLI and Web Human-in-the-Loop abstractions.
- `intelligent_feature_analysis.py`: Mutual information and feature signal profiling.
- `intelligent_summarization.py`: Reflexion summary generation and sanitization.
- `llm_output_logger.py`: Structured JSON logger for LLM decisions and traces.
- `model_cache.py`: Hash-keyed model persistence and retrieval engine.
- `prediction_analyzer.py`: Diagnostics for regression and classification predictions.
- `pretrained_model_store.py`: Bundle registry loader and inference helpers.
- `reporting.py`: Workflow report assembler and JSON snapshot exporter.
- `schema_discovery.py`: Dataset profiling and column role inspection.
- `synthetic_quality_analyzer.py`: Statistical fidelity comparison for synthetic data.
- `tool_decider.py`: Deterministic preprocessing and model family selection rules.

### `webapp/`
- `app.py`: FastAPI web server with dataset, execution, artifact, and synthetic data REST APIs.
- `run_manager.py`: Background thread execution worker, state machine, and artifact storage manager.
- `static/app.js`: Frontend state management, REST API polling, and UI rendering logic.
- `static/app.css`: UI styling and responsive layouts.
- `templates/index.html`: Web interface HTML template.

### `scripts/`
- `run_local_app.py`: Launches the local FastAPI/Uvicorn web application server.
- `generate_synthetic_data_and_infer.py`: CLI synthetic telemetry generator and evaluation script.
- `test_complete_flow.py`: End-to-end functional validation test suite.

### `training/`
- `train_and_export_pretrained.py`: Script for training and exporting supervised `.joblib` bundles.
- `offline_model_training.ipynb`: Jupyter notebook for model exploration and bundle export.
- `synthetic_data_inference_analysis.ipynb`: Analysis notebook for evaluating synthetic generalization.

### `data/`
- `smart_manufacturing_data.csv`: Source dataset for manufacturing operations.
- `smart_manufacturing_dataset.csv`: Supervised dataset for failure probability & maintenance tasks.
- `digital_manufacturing_dataset.csv`: Auxiliary manufacturing telemetry dataset.
- `Intelligent Manufacturing Dataset/`: Source dataset folder (6G manufacturing).
- `Smart Manufacturing Maintenance Dataset/`: Source maintenance dataset folder.
- `superconductivty+data/`: Auxiliary benchmarking dataset.

### `artifacts/`
- `pretrained_models/`: Pretrained model bundles (`.joblib`) and `registry.json`.
- `web_uploads/`: Datasets uploaded by operators via the Web GUI.
- `web_synthetic/`: Synthetic datasets generated via the Web GUI or API.
- `web_runs/`: Per-run execution state snapshots, dataset previews, and output logs.

### `model_cache/`
- Auto-generated directory storing serialized `.joblib` model fits keyed by configuration hash.

---

## 4. Intelligence Hierarchy & SLM Reduction

- **Tier 1 (Cloud LLM - Gemini 2.5 Flash)**: Strategic planning and Reflexion narrative summary generation.
- **Tier 2 (Local SLM - Qwen3:4B via Ollama/LlamaCpp)**: Tactical anomaly detection parameter optimization at the edge.
- **Tier 3 (Deterministic ToolDecider)**: Preprocessing strategy and model family selection.

**SLM Reduction Rationale**:
- SLM 1 (Perception): Replaced by deterministic pandas dtype discovery + schema analysis.
- SLM 2 (Preprocessing): Replaced by deterministic `ToolDecider` rules.
- SLM 3a (Model Selection): Replaced by `ToolDecider` decision tables.
- SLM 3b (Anomaly Parameters): **Retained** for multi-signal contamination trade-offs.
- SLM 4 (Summarization): Handled by Cloud LLM Reflexion loop with local SLM / plain-text fallbacks.
