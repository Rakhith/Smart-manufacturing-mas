# Smart Manufacturing Multi-Agent System (MAS)

Hybrid Agentic AI and Multi-Agent System for Prescriptive Maintenance in Smart Manufacturing.

An intelligent Multi-Agent System for predictive maintenance and optimization in smart manufacturing. Uses LLM-powered orchestration to automatically load, preprocess, analyze, and generate prescriptive recommendations from manufacturing data.

## 🚀 Features

- **LLM-Powered Orchestration**: Intelligent workflow planning using Google Gemini or local LLMs (Ollama/LlamaCpp)
- **Rules-First Pipeline**: Deterministic preprocessing + Reflexion-based LLM interpretation
- **Adaptive Intelligence**: Automatic model selection and performance optimization
- **Multi-Model Analysis**: Supports classification, regression, and anomaly detection
- **Prescriptive Recommendations**: Actionable maintenance suggestions with priority ranking
- **Human-in-the-Loop**: Interactive approval workflow for critical decisions
- **Model Persistence**: Hash-keyed caching of trained models
- **Comprehensive Logging**: Detailed audit trails and performance metrics

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **API Key**: Google Gemini API key (optional if using local LLMs)
- **Ollama** or **LlamaCpp**: Optional, for local LLM support

## Architecture

**Three-Tier Intelligence Hierarchy:**

| Tier | Component | Role |
|------|-----------|------|
| TIER 1 | Cloud LLM (Gemini 2.5-Flash) | Strategic orchestration OR Reflexion summary |
| TIER 2 | Local SLM (Qwen3:4B / Ollama) | Anomaly params ONLY (SLM 3b retained) |
| TIER 3 | Rule-Based ToolDecider | Preprocessing + model selection |

**SLM Reduction (4 → 1):**
- SLM 1 (Perception): **ELIMINATED** — pandas dtypes + HITL
- SLM 2 (Preprocessing): **ELIMINATED** — ToolDecider if-else rules
- SLM 3a (Model Selection): **ELIMINATED** — ToolDecider rule table
- SLM 3b (Anomaly Params): ✅ **RETAINED** — only remaining SLM call
- SLM 4 (Summary): **ELIMINATED** — Cloud LLM Reflexion loop

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd smart_manufacturing_mas
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv mas_venv

# Activate virtual environment
# On macOS/Linux:
source mas_venv/bin/activate
# On Windows:
mas_venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables (Optional)

If using Google Gemini, create a `.env` file:

```bash
cp .env.example .env
# Edit .env and add GEMINI_API_KEY=your_key
```

### 5. Install Ollama or LlamaCpp (Optional - for Local LLMs)

**Ollama:**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (example with Qwen3)
ollama pull qwen3:4b
```

**LlamaCpp:**
```bash
# Download pre-quantized models from huggingface.co
# Place .gguf files in a models/ directory
# Reference with --decision-model /path/to/model.Q4_K_M.gguf
```

## 🎯 Quick Start

```bash
# 1. Clone and set up
git clone <repo>
cd smart_manufacturing_mas
python -m venv mas_venv && source mas_venv/bin/activate
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and add GEMINI_API_KEY=your_key

# 3. Place your CSV datasets in data/
# e.g. data/Smart_Manufacturing_Maintenance_Dataset/smart_maintenance.csv

# 4. Run (interactive, original LLM mode)
python main_llm.py

# 5. Run (rules-first mode, auto-detect problem type)
python main_llm.py --mode rules-first --dataset data/.../your_file.csv --auto-detect
```

## Orchestration Modes

### `--mode llm` (default — original)
Cloud LLM decides which agent to call at each step. Full chain-of-thought reasoning.

```bash
python main_llm.py --dataset data/.../smmd.csv --planner-llm gemini
```

### `--mode rules-first` (NEW — Proposed Next Architecture)
Rules run first, LLM interprets results once at the end via Reflexion loop.

```bash
python main_llm.py --mode rules-first --dataset data/.../smmd.csv --auto-detect
```

## All CLI Flags

| Flag | Description |
|------|-------------|
| `--mode` | `llm` (default) or `rules-first` |
| `--auto-detect` | Auto-detect problem type from data statistics |
| `--target COL` | Target column name |
| `--features C1 C2` | Feature column names |
| `--problem-type` | Override: `classification` / `regression` / `anomaly_detection` |
| `--use-pca` | Enable PCA after preprocessing ⚠️ loses feature interpretability |
| `--pca-threshold` | Variance to retain (default: 0.95) |
| `--use-cache` | Cache trained models by feature hash |
| `--cache-dir DIR` | Cache directory (default: `./model_cache`) |
| `--invalidate-cache` | Delete cache for current config, then exit |
| `--planner-llm` | `gemini` / `ollama` / `llamacpp` / `mock` |
| `--decision-llm` | SLM for anomaly params: `ollama` / `llamacpp` / `mock` / None |
| `--decision-model` | Model tag/path for decision SLM |
| `--auto` | Non-interactive: auto-approve all HITL gates |
| `--batch` | Process all CSVs under `./data/` |
| `--interface` | `cli` (default) or `web` |

## Usage Examples

```bash
# Classification, auto-detect, with cache
python main_llm.py --mode rules-first \
  --dataset data/Smart_Manufacturing_Maintenance_Dataset/smmd.csv \
  --auto-detect --use-cache

# Anomaly detection with local SLM for params
python main_llm.py --mode rules-first \
  --dataset data/Intelligent_Manufacturing_Dataset/6gmr.csv \
  --problem-type anomaly_detection \
  --decision-llm ollama --decision-model qwen3:4b

# PCA + cache (high-dimensional sensor data, interpretability not required)
python main_llm.py --mode rules-first \
  --dataset data/.../sensors.csv \
  --auto-detect --use-pca --pca-threshold 0.90 --use-cache

# Use LlamaCpp SLM (CPU-only edge node)
python main_llm.py --mode rules-first \
  --dataset data/.../smmd.csv \
  --problem-type anomaly_detection \
  --decision-llm llamacpp \
  --decision-model /models/qwen3-4b.Q4_K_M.gguf

# Non-interactive batch processing
python main_llm.py --mode rules-first --batch --auto

# Clear the cache for a specific config
python main_llm.py --mode rules-first \
  --dataset data/.../smmd.csv \
  --target Maintenance_Priority --invalidate-cache
```

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Make sure virtual environment is activated
source mas_venv/bin/activate

# Verify installation
pip list | grep scikit-learn
```

**2. API Key Issues**
```bash
# Check .env file exists
cat .env

# Verify API key is valid
python -c "import google.generativeai as genai; genai.configure(api_key='your_key')"
```

**3. Ollama Connection Issues**
```bash
# Start Ollama service
ollama serve

# Test model availability
ollama list
```

**4. Dataset Issues**
```bash
# Verify dataset path
ls -la data/

# Check CSV format
head -5 data/Smart_Manufacturing_Maintenance_Dataset/smart_maintenance_dataset.csv
```

### Debug Mode

Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
python main_llm.py --mode rules-first --auto-detect --auto
```

## Project Structure

```
smart_manufacturing_mas/
├── main_llm.py                        # Entry point
├── requirements.txt
├── .env.example
├── agents/
│   ├── data_loader_agent.py           # Step 1: load & inspect CSV
│   ├── preprocessing_agent.py         # Step 2: clean, scale, encode (+ optional PCA)
│   ├── dynamic_analysis_agent.py      # Step 3: model selection, training (+ cache)
│   ├── optimization_agent.py          # Step 4: priority scoring + recommendations
│   ├── llm_planner_agent.py           # Original LLM-orchestrated workflow
│   ├── rules_first_planner.py         # NEW: deterministic pipeline + Reflexion
│   ├── local_llm_agent.py             # Ollama / LlamaCpp / HuggingFace adapter
│   └── planner_agent.py               # Rule-based emergency fallback
├── utils/
│   ├── auto_detect.py                 # NEW: auto-detect problem type
│   ├── model_cache.py                 # NEW: hash-keyed model persistence
│   ├── tool_decider.py                # Rule-based model/preprocessing selector
│   ├── schema_discovery.py            # Column role detection
│   ├── intelligent_feature_analysis.py # MI + correlation + feature importance
│   ├── intelligent_summarization.py   # Reflexion loop summarizer
│   ├── reporting.py                   # Workflow report generator
│   └── hitl_interface.py              # CLI / Web HITL interface
├── data/
│   ├── Smart_Manufacturing_Maintenance_Dataset/
│   └── Intelligent_Manufacturing_Dataset/
├── model_cache/                       # Auto-created; stores .pkl model files
├── logs/                              # Workflow reports + HITL audit
└── documentation/
    ├── architecture_and_workflow.md
    ├── adaptive_intelligence_system.md
    └── usage_guide.md
```

## Key Improvements Over Baseline

| Improvement | File | Flag |
|-------------|------|------|
| Rules-First Pipeline | `agents/rules_first_planner.py` | `--mode rules-first` |
| Auto-Detect Problem Type | `utils/auto_detect.py` | `--auto-detect` |
| Model Persistence Cache | `utils/model_cache.py` | `--use-cache` |
| LlamaCpp Backend | `agents/local_llm_agent.py` | `--decision-llm llamacpp` |
| Optional PCA | `agents/preprocessing_agent.py` | `--use-pca` |
| Finish-sentinel fix | `agents/llm_planner_agent.py` | (always active) |
| Machine_ID pass-through fix | `agents/llm_planner_agent.py` | (always active) |

## 📖 Learn More

- [Detailed Usage Guide](documentation/usage_guide.md)
- [Architecture and Workflow](documentation/architecture_and_workflow.md)
- [Adaptive Intelligence System](documentation/adaptive_intelligence_system.md)

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:
1. Create a feature branch
2. Add tests for new functionality
3. Update documentation
4. Submit a pull request

---

**Ready to get started?** Run `python main_llm.py` to begin your first analysis!
