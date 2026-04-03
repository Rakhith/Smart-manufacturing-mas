# Usage Guide

## Setup

```bash
git clone <repo>
cd smart_manufacturing_mas

# Create virtual environment
python3 -m venv mas_venv
source mas_venv/bin/activate        # macOS / Linux
# mas_venv\Scripts\activate         # Windows

pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Open .env and set: GEMINI_API_KEY=your_key_here
```

## Adding Your Datasets

Place CSV files in `data/`. The system will auto-discover them in `--batch` mode.
```
data/
  Smart_Manufacturing_Maintenance_Dataset/
    smart_maintenance.csv
  Intelligent_Manufacturing_Dataset/
    6g_manufacturing.csv
```

## Running the System

### Interactive (recommended first run)
```bash
python main_llm.py
# Follow the prompts to select dataset, problem type, columns
```

### Fully automated (CI / scripting)
```bash
python main_llm.py \
  --mode rules-first \
  --dataset data/Smart_Manufacturing_Maintenance_Dataset/smmd.csv \
  --auto-detect \
  --auto           # auto-approve all HITL gates
```

### With model caching (repeat runs)
```bash
# First run: trains and caches the model
python main_llm.py --mode rules-first \
  --dataset data/.../smmd.csv --auto-detect --use-cache

# Second run on identical config: loads instantly from cache
python main_llm.py --mode rules-first \
  --dataset data/.../smmd.csv --auto-detect --use-cache
```

### With anomaly detection SLM (Ollama)
```bash
# Start Ollama first: ollama serve && ollama pull qwen3:4b

python main_llm.py --mode rules-first \
  --dataset data/.../6gmr.csv \
  --problem-type anomaly_detection \
  --decision-llm ollama --decision-model qwen3:4b
```

### With LlamaCpp (CPU-only edge node)
```bash
# Download a GGUF model first
# e.g. wget https://huggingface.co/.../qwen3-4b.Q4_K_M.gguf -O /models/qwen3.gguf

python main_llm.py --mode rules-first \
  --dataset data/.../smmd.csv \
  --problem-type anomaly_detection \
  --decision-llm llamacpp \
  --decision-model /models/qwen3.gguf
```

### With PCA (high-dimensional data, interpretability not required)
```bash
python main_llm.py --mode rules-first \
  --dataset data/.../high_dim_sensors.csv \
  --auto-detect --use-pca --pca-threshold 0.90
```

## Model Cache Management

```bash
# Show cache stats
python3 -c "
from utils.model_cache import ModelCache
c = ModelCache()
print(c.stats())
print(list(c.list_entries().keys()))
"

# Invalidate a specific config
python main_llm.py --mode rules-first \
  --dataset data/.../smmd.csv \
  --target Maintenance_Priority \
  --problem-type classification \
  --invalidate-cache

# Clear all cached models
python3 -c "from utils.model_cache import ModelCache; ModelCache().clear()"
```

## Auto-Detect Logic

| Target column dtype | Unique values | Detected type | Confidence |
|---------------------|---------------|---------------|------------|
| object / category   | any           | classification | 97% |
| bool                | 2             | classification | 99% |
| int                 | ≤ 20          | classification | 82–95% |
| int                 | > 20          | regression     | 75% |
| float               | ≤ 5           | classification | 70% |
| float               | > 5           | regression     | 78–92% |
| (none specified)    | —             | anomaly_detection | 95% |

When confidence < 75% and running interactively, a HITL confirmation is shown.

## Output Files

All outputs are saved to `logs/`:
- `workflow_report_TIMESTAMP.json` — full workflow log
- `detailed_results_TIMESTAMP.json` — model results and recommendations
- `publication_snapshot_TIMESTAMP.json` — formatted for research use
- `hitl_audit.json` — all human-in-the-loop interactions

## Troubleshooting

**`GEMINI_API_KEY not set`**
→ Copy `.env.example` to `.env` and add your key.

**`Ollama generation failed`**
→ Run `ollama serve` in a separate terminal, then `ollama pull qwen3:4b`.

**`llama-cpp-python not installed`**
→ Run `pip install llama-cpp-python` (builds from source, may take a few minutes).

**Model cache conflicts**
→ Run with `--invalidate-cache` to clear the specific entry, or delete `model_cache/` manually.
