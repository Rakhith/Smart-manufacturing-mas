# Quick Start Guide 🚀

Get up and running with the Smart Manufacturing Multi-Agent System (MAS) in 5 minutes!

---

## ⚡ 1-Minute Web Dashboard Launcher

For an interactive GUI with real-time stage tracking, synthetic data generation, and artifact downloads:

```bash
# 1. Activate virtual environment
# Windows:
mas_venv\Scripts\activate
# macOS/Linux:
source mas_venv/bin/activate

# 2. Launch local web application
python scripts/run_local_app.py
```

Then navigate to: **`http://127.0.0.1:8000`** in your browser.

---

## 🎯 5-Minute Terminal Setup

### Step 1: Environment Setup (2 minutes)

```bash
# Clone the repository
git clone <repository-url>
cd smart_manufacturing_mas

# Create virtual environment
python -m venv mas_venv

# Activate virtual environment
# Windows:
mas_venv\Scripts\activate
# macOS/Linux:
source mas_venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure LLM Backend (1 minute)

**Option A: Google Gemini (Recommended for Reflexion Summaries)**
```bash
# Copy template and add your API key
cp .env.example .env
# Edit .env and set: GEMINI_API_KEY="your-gemini-api-key"
```

**Option B: Local Ollama (Offline / Factory Edge Mode)**
```bash
# Start Ollama service and pull edge model
ollama serve
ollama pull qwen3:4b
```

**Option C: No API Key / Mock Mode**
You can run full rules-first pipelines and pretrained inference without any API key (plain-text summary fallback is automatically used).

---

### Step 3: Run Your First Analysis (2 minutes)

#### 1. Autonomous Rules-First Pipeline (Recommended)
Automatically detects whether the task is classification, regression, or anomaly detection, runs deterministic ML preprocessing and training, and generates prescriptive actions:

```bash
python main_llm.py --mode rules-first --dataset "data/Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv" --auto-detect --use-cache
```

#### 2. Supervised Pretrained Inference
Instantly scores datasets using pre-trained model bundles without re-training:

```bash
python main_llm.py --mode rules-first --dataset "data/smart_manufacturing_dataset.csv" --problem-type regression --inference-only
```

#### 3. Edge Anomaly Detection with Local SLM
Uses local Qwen3:4B via Ollama to intelligently configure Isolation Forest contamination parameters:

```bash
python main_llm.py --mode rules-first --dataset "data/Intelligent Manufacturing Dataset/manufacturing_6G_dataset.csv" --problem-type anomaly_detection --decision-llm ollama --decision-model qwen3:4b
```

#### 4. Headless Automated Batch Run (CI / Scripting)
Processes all datasets under `data/` non-interactively:

```bash
python main_llm.py --mode rules-first --batch --auto
```

---

## 🏛️ Architecture Overview & Semester 7 Pivot

The platform implements a **Three-Tier Intelligence Hierarchy**:
1. **Tier 1 (Cloud LLM - Gemini 2.5 Flash)**: Strategic orchestration and Reflexion summary loop (Draft $\rightarrow$ Critique $\rightarrow$ Revise).
2. **Tier 2 (Local SLM - Qwen3:4B via Ollama)**: Tactical parameter suggestions for anomaly detection.
3. **Tier 3 (Rule-Based ToolDecider)**: Deterministic preprocessing and model family selection (zero hallucination, zero latency).

### Semester 7 Autonomous Closed-Loop Pivot:
*   **Learned Recommender**: `LGBMRanker` (LambdaMART) + Multi-Sensor Archetype Collaborative Filtering (cosine similarity) replaces static if-else action strings.
*   **Self-Explaining Interpretability Layer**: Local `TreeSHAP` attributions + Case-Based historical incident retrieval explain *why* actions are prioritized.
*   **Autonomous Execution State Machine**: Replaces blocking manual HITL review gates with autonomous action dispatch, post-intervention recovery tracking ($\Delta \text{Recovery}$), and continuous cost-saving feedback.

---

## 📁 Output Files & Artifacts

After execution, all run artifacts and audit traces are saved in:

*   **`artifacts/web_runs/`**: Per-run execution state snapshots, dataset previews, model diagnostics, recommendations, and Reflexion summaries from the Web UI.
*   **`artifacts/pretrained_models/`**: Serialized model bundles and `registry.json`.
*   **`artifacts/web_synthetic/`**: Synthetic datasets generated from the UI or API.
*   **`logs/`**: Structured execution logs, including `hitl_audit.json` and workflow report snapshots.
*   **`model_cache/`**: Hash-keyed model cache files (`.joblib` / `.pkl`) for instant cache-hit execution.

---

## 🐛 Troubleshooting

*   **Virtual environment not activated**:
    Ensure your prompt displays `(mas_venv)`. Run `mas_venv\Scripts\activate` (Windows) or `source mas_venv/bin/activate` (Linux/macOS).
*   **Gemini API Key missing**:
    Verify `.env` exists in the project root with `GEMINI_API_KEY=your_key`. If omitted, the system falls back gracefully to local SLM or plain-text summaries.
*   **Port 8000 already in use**:
    Run Uvicorn on a different port: `uvicorn webapp.app:app --host 127.0.0.1 --port 8080`.
*   **Ollama connection refused**:
    Start the Ollama daemon with `ollama serve` in a separate terminal and ensure `ollama list` shows `qwen3:4b`.

---

## 📚 Next Steps & Documentation

*   [Detailed Usage Guide](documentation/usage_guide.md)
*   [Architecture and Workflow](documentation/architecture_and_workflow.md)
*   [Adaptive Intelligence System](documentation/adaptive_intelligence_system.md)
*   [Synthetic Data Generation Guide](SYNTHETIC_DATA_GUIDE.md)
