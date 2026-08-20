# Smart Manufacturing Multi-Agent System (MAS)

> **Autonomous Closed-Loop Cyber-Physical Prescriptive Maintenance Platform**  
> *Hybrid Agentic AI, Multi-Agent Architecture, Learned Ranking, Self-Explaining Interpretability, and Closed-Loop Feedback Execution.*

---

## 📌 Executive Summary

Smart Manufacturing MAS is an enterprise multi-agent machine learning platform engineered for predictive and prescriptive maintenance in smart industrial environments. The system transitions industrial operations from static, human-gated advisory reports to an **end-to-end autonomous, closed-loop decision and execution cycle**.

```
[Telemetry Ingestion] ──► [Dynamic Analysis] ──► [Learned Recommender (LGBMRanker + CF)]
        ▲                                                    │
        │                                                    ▼
[State & Outcome Feedback] ◄── [Automated Action Execution] ◄── [Explainability Layer (SHAP + CBR)]
```

---

## 🚀 Key Features

*   🖥️ **Local Web Dashboard**: Interactive FastAPI + Vanilla JS web interface with real-time pipeline tracking, synthetic data generation, and artifact downloads.
*   🤖 **Autonomous Closed-Loop Pipeline**: Replaces static, human-blocking review gates with confidence-bounded autonomous action dispatch and closed-loop state tracking.
*   🎯 **Learned Prescriptive Recommender**: `LGBMRanker` (LambdaMART) with multi-sensor archetype collaborative filtering (cosine similarity) to programmatically rank optimal maintenance interventions.
*   🔍 **Self-Explaining Interpretability Layer**: Local `TreeSHAP` quantitative feature attribution combined with Case-Based Reasoning (historical peer incident precedents) for verifiable audit trails.
*   ⚙️ **Dynamic Multi-Task Analysis**: Automated support for Classification, Regression, and Isolation Forest Anomaly Detection with offline pretrained model bundles and fallback live training.
*   🧠 **3-Tier Intelligence Hierarchy**:
    *   **Tier 1 (Cloud LLM - Gemini 2.5 Flash)**: Strategic orchestration & Reflexion summary loops (Draft $\rightarrow$ Critique $\rightarrow$ Revise).
    *   **Tier 2 (Local SLM - Qwen3:4B via Ollama/LlamaCpp)**: Tactical parameter configuration for anomaly detection at the edge.
    *   **Tier 3 (Deterministic ToolDecider)**: Zero-latency, zero-hallucination data preprocessing and model family selection.
*   ⚡ **Model Persistence & Caching**: Hash-keyed model caching (`ModelCache`) for instant subsequent inference.

---

## 🏛️ Architecture & Semester 7 Pivot

### Evolution: Last Semester (Sem 6) vs. This Semester (Sem 7)

| Dimension | Last Semester (Sem 6) | Proposed This Semester (Sem 7 Pivot) |
| :--- | :--- | :--- |
| **System Loop** | **Open-loop** (terminates at static CSV/JSON report) | **Closed-loop** (Prediction $\rightarrow$ Ranking $\rightarrow$ Simulated Execution $\rightarrow$ Feedback) |
| **Operator Involvement** | **Heavy HITL gates** at every pipeline stage | **Autonomous execution** with automated safety fallbacks |
| **Action Generation** | Static heuristic rules & keyword matching (`_action_plan_from_score`) | **Learned Recommender** (`LGBMRanker` + Collaborative Filtering) |
| **Cross-Asset Signal** | None (assets analyzed in isolation) | **Cosine similarity** over multi-sensor archetypes |
| **Action Justification** | Generic canned template strings | **Dual-pillar explainability** (Local `TreeSHAP` + Case-Based Reasoning) |
| **Evaluation Metrics** | Standard ML metrics ($R^2$, Accuracy, MSE) | **RecSys & Economic metrics** ($\text{NDCG@k}$, $\text{Precision@k}$, Net Cost-Saved) |

---

## 💻 Web Frontend & GUI Dashboard

The platform includes a modern, lightweight web application for uploading datasets, generating synthetic streams, monitoring stage-by-stage multi-agent execution, and visualizing prescriptive actions.

### Launching the Web Application

```bash
# 1. Activate your virtual environment
# Windows:
mas_venv\Scripts\activate
# macOS/Linux:
source mas_venv/bin/activate

# 2. Start the local server
python scripts/run_local_app.py
```

Alternatively, launch directly with Uvicorn:
```bash
uvicorn webapp.app:app --host 127.0.0.1 --port 8000 --reload
```

Then open your browser at:
```text
http://127.0.0.1:8000
```

### Web UI Features
1. **Dataset Selection & Upload**: Choose from built-in manufacturing datasets or drag-and-drop custom industrial telemetry CSVs.
2. **Execution Modes**: Toggle between Pretrained Inference (`artifacts/pretrained_models`) and Live Training.
3. **Synthetic Generator**: Generate realistic synthetic sensor datasets on the fly with custom row counts and seeds.
4. **Live Stage Progress**: Watch the pipeline transition dynamically across:
   * `Step 1: DataLoaderAgent` (Schema & Quality Profiling)
   * `Step 2: PreprocessingAgent` (Transformations & Scaling)
   * `Step 3: DynamicAnalysisAgent` (Model Performance & Diagnostics)
   * `Step 4: OptimizationAgent` (Prescriptive Action Ranking & Explainability)
   * `Step 5: Reflexion Summary` (Intelligent Narrative Synthesis)
5. **Artifact Downloader**: Inspect and download run configurations, recommendation tables, and summary reports directly from the browser.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
*   **Python**: Version 3.8 to 3.11
*   **Google Gemini API Key**: (Optional, for Cloud LLM Reflexion summaries)
*   **Ollama / LlamaCpp**: (Optional, for local edge SLM inference)

### 2. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd smart_manufacturing_mas

# Create virtual environment
python -m venv mas_venv

# Activate virtual environment
# Windows:
mas_venv\Scripts\activate
# Linux/macOS:
source mas_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Secrets (Optional)

Create a `.env` file from the provided template:
```bash
cp .env.example .env
```
Edit `.env` to include your Google Gemini API key:
```env
GEMINI_API_KEY="your-gemini-api-key-here"
```

### 4. Local Edge SLM Setup (Optional)

If utilizing local SLM capabilities for anomaly parameter generation:
```bash
# Install and run Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve

# Pull Qwen3 edge model
ollama pull qwen3:4b
```

---

## 🎯 Command Line Interface (CLI) Guide

The platform can be run directly from the terminal in either `rules-first` (recommended) or `llm` orchestration mode.

### Quick Commands

```bash
# 1. Autonomous Rules-First Mode with Auto-Detection & Caching (Recommended)
python main_llm.py --mode rules-first --dataset "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv" --auto-detect --use-cache

# 2. Supervised Pretrained Inference Mode (Instant execution without retraining)
python main_llm.py --mode rules-first --dataset "data/smart_manufacturing_dataset.csv" --problem-type regression --inference-only

# 3. Supervised Live Model Training
python main_llm.py --mode rules-first --dataset "data/smart_manufacturing_dataset.csv" --problem-type classification --train-live

# 4. Anomaly Detection with Local SLM Edge Guidance
python main_llm.py --mode rules-first --dataset "data/Intelligent Manufacturing Dataset/manufacturing_6G_dataset.csv" --problem-type anomaly_detection --decision-llm ollama --decision-model qwen3:4b

# 5. Non-Interactive Unattended Batch Run (CI / Headless Mode)
python main_llm.py --mode rules-first --batch --auto
```

---

## 📖 Complete CLI Flags Reference

| CLI Argument | Type / Default | Description |
| :--- | :--- | :--- |
| `--mode` | `rules-first` \| `llm` (default: `llm`) | Orchestration paradigm. `rules-first` executes deterministic steps first with Reflexion interpretation at the end. |
| `--dataset` | `str` | Path to dataset (`.csv` or `.npz`). |
| `--auto-detect` | `flag` | Automatically infers problem type and target column from dataset statistics. |
| `--target` | `str` | Explicit target column name (e.g. `Maintenance_Priority`, `Failure_Prob`). |
| `--features` | `str...` | Space-separated list of feature columns to utilize. |
| `--problem-type` | `classification` \| `regression` \| `anomaly_detection` | Explicit problem type override. |
| `--inference-only` | `flag` | Uses pre-trained model bundles (`artifacts/pretrained_models`) instead of live training. |
| `--train-live` | `flag` | Forces live training for supervised tasks (overrides default pretrained inference). |
| `--pretrained-dir` | `str` (default: `artifacts/pretrained_models`) | Directory containing pretrained model bundles and `registry.json`. |
| `--preferred-model`| `str` | Force a specific pretrained model architecture (e.g. `RandomForestClassifier`, `Ridge`). |
| `--use-cache` | `flag` | Enables hash-keyed model persistence under `model_cache/`. |
| `--cache-dir` | `str` (default: `./model_cache`) | Model cache storage location. |
| `--invalidate-cache`| `flag` | Deletes cache matching current configuration and exits. |
| `--use-pca` | `flag` | Applies Principal Component Analysis dimensionality reduction during preprocessing. |
| `--pca-threshold` | `float` (default: `0.95`) | Variance ratio to retain when `--use-pca` is enabled. |
| `--planner-llm` | `gemini` \| `ollama` \| `llamacpp` \| `mock` | Primary LLM backend for planning/Reflexion summaries. |
| `--decision-llm` | `ollama` \| `llamacpp` \| `mock` \| `None` | Secondary SLM backend for anomaly parameter suggestions. |
| `--decision-model` | `str` (e.g. `qwen3:4b`) | Model tag or GGUF path for tactical SLM. |
| `--auto` | `flag` | Autonomous mode: automatically approves all HITL confirmation gates. |
| `--batch` | `flag` | Processes all CSV files discovered in `./data/`. |
| `--interface` | `cli` \| `web` (default: `cli`) | User interaction interface backend. |

---

## 📂 Repository Directory Structure

```text
smart_manufacturing_mas/
├── README.md                          # Primary platform documentation
├── QUICKSTART.md                      # Fast 5-minute setup and command cheat sheet
├── SYNTHETIC_DATA_GUIDE.md            # Synthetic data generation and evaluation guide
├── Repo Summary.md                    # Technical codebase and architecture overview
├── requirements.txt                   # Locked Python dependency manifest
├── main_llm.py                        # Unified CLI entry point
│
├── agents/                            # Multi-Agent subsystem
│   ├── data_loader_agent.py           # Ingestion, schema discovery, sampling
│   ├── preprocessing_agent.py         # Pipeline imputation, scaling, OHE, PCA
│   ├── dynamic_analysis_agent.py      # Multi-family model training and pretrained inference
│   ├── optimization_agent.py          # Prescriptive recommendation and ranking engine
│   ├── rules_first_planner.py         # Deterministic pipeline orchestrator
│   ├── llm_planner_agent.py           # LLM-orchestrated planner
│   ├── local_llm_agent.py             # Local SLM backend wrapper (Ollama/LlamaCpp)
│   └── planner_agent.py               # Rule-based fallback orchestrator
│
├── utils/                             # Core utilities and intelligence algorithms
│   ├── auto_detect.py                 # Statistical problem-type auto-detection
│   ├── column_utils.py                # Identifier and feature column helpers
│   ├── hitl_interface.py              # CLI & Web Human-in-the-Loop abstractions
│   ├── intelligent_feature_analysis.py# Mutual info, correlation, and feature profiling
│   ├── intelligent_summarization.py   # Reflexion loop narrative summarizer
│   ├── model_cache.py                 # Hash-keyed model caching engine
│   ├── prediction_analyzer.py         # Regression and classification diagnostics
│   ├── pretrained_model_store.py      # Bundle loader and registry manager
│   ├── reporting.py                   # Report assembler and export utilities
│   ├── schema_discovery.py            # Dataset profiling and type inspection
│   ├── synthetic_quality_analyzer.py  # Statistical fidelity evaluator for synthetic data
│   └── tool_decider.py                # Deterministic preprocessing & model selector
│
├── webapp/                            # Local Web GUI Dashboard
│   ├── app.py                         # FastAPI backend and REST API endpoints
│   ├── run_manager.py                 # Background execution state machine & artifact manager
│   ├── static/                        # Frontend assets (app.js, app.css)
│   └── templates/                     # HTML templates (index.html)
│
├── scripts/                           # Automation scripts
│   ├── run_local_app.py               # Local web app launcher
│   ├── generate_synthetic_data_and_infer.py # CLI synthetic generator and inference script
│   └── test_complete_flow.py          # End-to-end integration test suite
│
├── training/                          # Pretrained modeling & analysis assets
│   ├── train_and_export_pretrained.py # Pretrained bundle training and export script
│   ├── offline_model_training.ipynb   # Model training & registry export notebook
│   └── synthetic_data_inference_analysis.ipynb # Synthetic generalization analysis
│
├── data/                              # Source manufacturing datasets
│   ├── Smart Manufacturing Maintenance Dataset/
│   ├── Intelligent Manufacturing Dataset/
│   └── digital_manufacturing_dataset.csv
│
├── artifacts/                         # Generated outputs and persistence stores
│   ├── pretrained_models/             # Exported .joblib bundles & registry.json
│   ├── web_uploads/                   # Datasets uploaded via Web GUI
│   ├── web_synthetic/                 # Datasets generated via Web GUI
│   └── web_runs/                      # Per-run execution state snapshots & outputs
│
├── logs/                              # Audit logs and run reports
└── documentation/                     # Deep-dive architecture and user guides
    ├── architecture_and_workflow.md   # Architectural design and Sem 7 pivot specs
    ├── adaptive_intelligence_system.md# Adaptive retry and continuous learning docs
    └── usage_guide.md                 # Detailed walkthrough and troubleshooting
```

---

## 🔬 Evaluation Metrics

The platform evaluates performance across dual criteria:

1. **Diagnostic & Predictive ML Performance**:
   * **Classification**: Balanced Accuracy, Precision, Recall, Macro-F1, ROC-AUC.
   * **Regression**: Coefficient of Determination ($R^2$), Mean Squared Error ($\text{MSE}$), Mean Absolute Error ($\text{MAE}$).
   * **Anomaly Detection**: Anomaly Contamination Rate, Mean Z-Score Outlier Distance.

2. **Prescriptive & Economic Metrics (Sem 7 Pivot)**:
   * **Ranking Utility**: Normalized Discounted Cumulative Gain ($\text{NDCG@k}$), Mean Reciprocal Rank ($\text{MRR}$), $\text{Precision@k}$.
   * **Industrial Economics**: $\text{Net Cost Saved} = \text{Avoided Downtime Cost} - \text{Intervention Cost}$.
   * **Physical Recovery**: Post-Intervention Sensor Delta Recovery ($\Delta \text{Recovery}$ over horizon $H$) and Mean Time Between Failures ($\text{MTBF}$).

---

## 🤝 Contributing & License

Contributions, issues, and feature requests are welcome. This project is developed as an Advanced Multi-Agent System for Smart Manufacturing.
