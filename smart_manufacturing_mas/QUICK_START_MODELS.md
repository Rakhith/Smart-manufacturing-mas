# Quick Reference - Pre-trained Models & Caching

## Artifact Locations
- **Pre-trained Models**: `artifacts/pretrained_models/` (DEFAULT)
- **Cache**: `model_cache/` 
- Both are NOW CLEARED and ready for fresh models

## Three Model Strategies

### 1️⃣ Pre-trained Inference (Recommended for stable inference)
```bash
python main_llm.py --mode rules-first \
  --dataset "data/your_dataset.csv" \
  --inference-only
```
- Uses models trained offline
- Fast (no training)
- Consistent across runs

### 2️⃣ Live Training (Recommended for exploration)
```bash
python main_llm.py --mode rules-first \
  --dataset "data/your_dataset.csv" \
  --train-live
```
- Trains fresh model each time
- Slower but adaptive
- Can combine with `--use-cache`

### 3️⃣ Caching (Recommended for repeated configs)
```bash
python main_llm.py --mode rules-first \
  --dataset "data/your_dataset.csv" \
  --use-cache
```
- Caches based on: dataset + features + target + task
- Same config = instant load
- Different config = fresh train

## Incompatible Flags (Will Error)
```bash
# ❌ DON'T DO THIS - they conflict
--use-cache --inference-only
--use-cache --train-live --inference-only  
--train-live --inference-only
--invalidate-cache --inference-only
```

## Target Detection Logic
```
Priority: bundle_target > target_column > auto-detect
```
- Bundle's target is preferred
- Falls back to passed `--target`
- Finally auto-detects if neither

## Setting Up Pre-trained Models
1. Open `training/offline_model_training.ipynb`
2. Run all cells
3. Models saved to `artifacts/pretrained_models/`
4. Registry created automatically
5. Use `--inference-only` to load them

## Test Everything Works
```bash
python scripts/test_complete_flow.py
```
Expected output: `TESTS PASSED - System is ready`

## Web UI
```bash
python scripts/run_local_app.py
# Opens http://127.0.0.1:8000
```

## Check What's in Artifacts
```bash
python -c "
import json
from pathlib import Path

# Pre-trained
with open('artifacts/pretrained_models/registry.json') as f:
    pretrained = json.load(f)
print(f'Pre-trained: {sum(len(pretrained.get(k, [])) for k in pretrained)} models')

# Cache
with open('model_cache/registry.json') as f:
    cache = json.load(f)
print(f'Cache: {len(cache)} entries')
"
```

---

## The Fix You Needed
**Problem**: Pre-trained models not working, target mismatch causing bad R²

**Solution**:
1. ✅ Cleared bad artifacts
2. ✅ Added flag validation (prevent conflicting options)  
3. ✅ Fixed web UI text wrapping
4. ✅ Improved error messages
5. ✅ Documented the complete flow

**Result**: System is now clean and ready for proper offline model training + inference

---

## Next Action
1. Run `python scripts/test_complete_flow.py` - Verify everything ✓
2. Open `training/offline_model_training.ipynb` - Train models
3. Run `python main_llm.py --mode rules-first --dataset "..." --inference-only` - Use models
