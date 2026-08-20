# Detailed Usage & Operations Guide

## 1. Environment Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd smart_manufacturing_mas

# 2. Create Python virtual environment
python -m venv mas_venv

# 3. Activate virtual environment
# Windows:
mas_venv\Scripts\activate
# macOS/Linux:
source mas_venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure API keys (optional, for Cloud LLM)
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key
```

---

## 2. Web Frontend Dashboard Guide

The system includes a local web GUI that visualizes pipeline execution stage-by-stage.

### Starting the Web Server

```bash
# Using the launcher script
python scripts/run_local_app.py

# Or directly with Uvicorn
uvicorn webapp.app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at: **`http://127.0.0.1:8000`**

### Web GUI Capabilities
1. **Dataset Selection / Upload**: Pick from built-in datasets in `data/` or upload custom industrial telemetry files.
2. **Execution Configuration**: Select between Pretrained Model Inference (`artifacts/pretrained_models`) and Live Training. Configure Model Caching and PCA.
3. **Synthetic Generator**: Generate artificial telemetry data with custom row counts and seeds directly in the UI.
4. **Stage-by-Stage Tracking**: Real-time progress bar across Data Ingestion, Preprocessing, Dynamic Analysis, Prescriptive Optimization, and Reflexion Summary.
5. **Artifact Downloader**: View and download full JSON run state snapshots and recommendation tables.

### REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/datasets` | `GET` | Lists all available datasets (built-in, uploaded, and synthetic). |
| `/api/upload` | `POST` | Uploads a new CSV dataset into `artifacts/web_uploads/`. |
| `/api/run` | `POST` | Enqueues a new background pipeline run with `RunConfig`. |
| `/api/run/{run_id}` | `GET` | Polls the live execution state, stage progress, logs, and metrics. |
| `/api/generate-synthetic` | `POST` | Generates a new synthetic dataset based on source distribution stats. |
| `/api/artifacts/{run_id}/{file}` | `GET` | Streams generated artifact files (JSON, CSV, summaries). |

---

## 3. Command Line Interface (CLI) Guide

### A. Autonomous Rules-First Workflow (Recommended)
Executes deterministic preprocessing and model analysis first, followed by Cloud LLM Reflexion summary synthesis:

```bash
# Auto-detect problem type and target column with caching enabled
python main_llm.py --mode rules-first --dataset "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv" --auto-detect --use-cache

# Explicit classification override
python main_llm.py --mode rules-first --dataset "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv" --target Maintenance_Priority --problem-type classification --use-cache
```

### B. Pretrained Inference vs. Live Training
```bash
# 1. Inference-only using pretrained bundles (fastest, no training overhead)
python main_llm.py --mode rules-first --dataset "data/smart_manufacturing_dataset.csv" --problem-type regression --inference-only

# 2. Force a specific pretrained architecture
python main_llm.py --mode rules-first --dataset "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv" --problem-type classification --preferred-model RandomForestClassifier

# 3. Force live training (retrains model on input dataset)
python main_llm.py --mode rules-first --dataset "data/smart_manufacturing_dataset.csv" --problem-type regression --train-live
```

### C. Edge Anomaly Detection with Local SLM
```bash
# Start Ollama service in a separate terminal:
# ollama serve && ollama pull qwen3:4b

python main_llm.py --mode rules-first \
  --dataset "data/Intelligent Manufacturing Dataset/manufacturing_6G_dataset.csv" \
  --problem-type anomaly_detection \
  --decision-llm ollama \
  --decision-model qwen3:4b
```

### D. PCA Dimensionality Reduction
```bash
# Enable PCA with 90% variance retention
python main_llm.py --mode rules-first \
  --dataset "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv" \
  --auto-detect \
  --use-pca \
  --pca-threshold 0.90 \
  --use-cache
```

### E. Headless Automated Batch Execution (CI / Scripting)
```bash
# Process all datasets in data/ non-interactively
python main_llm.py --mode rules-first --batch --auto
```

---

## 4. Model Cache Management

The `ModelCache` subsystem uses hash-based persistence keyed by `SHA-256(dataset_basename + feature_columns + target_column + problem_type)`.

```bash
# Inspect cache entries and statistics via Python CLI
python -c "from utils.model_cache import ModelCache; c = ModelCache(); print(c.stats()); print(list(c.list_entries().keys()))"

# Invalidate cache for a specific configuration
python main_llm.py --mode rules-first \
  --dataset "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv" \
  --target Maintenance_Priority \
  --invalidate-cache

# Clear all cached model bundles
python -c "from utils.model_cache import ModelCache; ModelCache().clear()"
```

---

## 5. Output Directory Semantics

*   `artifacts/web_runs/{run_id}/`: Contains per-run state snapshots (`state.json`), pipeline summaries (`summary.txt`), and recommendations (`recommendations.csv`).
*   `artifacts/pretrained_models/`: Contains pre-trained `.joblib` model bundles and `registry.json`.
*   `artifacts/web_uploads/`: Stores user-uploaded CSV datasets.
*   `artifacts/web_synthetic/`: Stores synthetically generated datasets.
*   `model_cache/`: Stores cached `.joblib` model fits.
*   `logs/`: Stores execution traces and `hitl_audit.json`.

---

## 6. Troubleshooting

*   **`GEMINI_API_KEY not set`**: Copy `.env.example` to `.env` and configure your API key. If omitted, the system falls back to plain-text summaries without crashing.
*   **`Ollama connection failed`**: Run `ollama serve` in a background terminal and check `ollama list` for model availability.
*   **Port collision on Web App**: Run Uvicorn on an alternate port: `python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8080`.
*   **Low model performance warning**: For datasets with weak linear signals, the system will alert the operator and activate Adaptive Intelligence to evaluate all alternative model families.
