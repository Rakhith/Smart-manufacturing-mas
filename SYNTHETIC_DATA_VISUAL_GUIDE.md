# 🎨 Synthetic Data Generation - Visual Summary

## Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                         WEB INTERFACE                              │
│  User: Checks "Generate synthetic data" + Sets rows (e.g., 300)   │
└──────────────────────────┬─────────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  SYNTHETIC GENERATION ENGINE          │
        │  (run_manager.py)                     │
        │  ┌──────────────────────────────────┐ │
        │  │ Analyzes original dataset:       │ │
        │  │ • Mean, std, min, max            │ │
        │  │ • Distribution parameters        │ │
        │  │ • Category proportions            │ │
        │  └──────────────────────────────────┘ │
        │              │                         │
        │              ▼                         │
        │  ┌──────────────────────────────────┐ │
        │  │ Generates N new rows by:         │ │
        │  │ • Numeric: Sample from dist.    │ │
        │  │ • Categorical: Sample from %    │ │
        │  │ • Exclude ID columns             │ │
        │  └──────────────────────────────────┘ │
        └──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │   SAVE SYNTHETIC CSV                  │
        │   artifacts/web_synthetic/            │
        │   {run_id}_synthetic.csv             │
        │   (300 rows × 10 columns)            │
        └──────────────────────────────────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
    ┌─────────────────────┐  ┌─────────────────────┐
    │ QUALITY ANALYZER    │  │  PREDICTION ENGINE  │
    │ (synthetic_quality_ │  │ (prediction_analyzer│
    │  analyzer.py)       │  │  .py)               │
    │                     │  │                     │
    │ Compares:           │  │ Uses pretrained     │
    │ • Distributions     │  │ model to predict:   │
    │ • Statistics        │  │ • 300 predictions   │
    │ • KS/Chi-square     │  │ • Classification or │
    │   tests             │  │   regression values │
    │                     │  │ • Store predictions │
    │ Output:             │  │                     │
    │ Quality Score       │  │ Output:             │
    │ (0-100)             │  │ predictions[]       │
    │ + metrics           │  │ + metrics           │
    └─────────────────────┘  └─────────────────────┘
                │                     │
                └──────────┬──────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │   SAVE INFERENCE JSON                 │
        │   artifacts/web_synthetic/            │
        │   {run_id}_synthetic_inference.json  │
        │                                      │
        │ Contains:                            │
        │ • predictions array (300 values)    │
        │ • quality analysis                  │
        │ • prediction analysis               │
        │ • recommendations                   │
        └──────────────────────────────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │   RENDER WEB DASHBOARD (app.js)      │
        │                                      │
        │   CARD 1: Data Quality               │
        │   ├─ Quality Score: 85/100 ✓         │
        │   ├─ Mean Diff: 2.3%                 │
        │   ├─ Distribution Similarity: 92%    │
        │   └─ Status: EXCELLENT               │
        │                                      │
        │   CARD 2: Predictions                │
        │   ├─ Total Predictions: 300          │
        │   ├─ Accuracy: 87%                   │
        │   ├─ Class A: 40% (120 samples)     │
        │   ├─ Class B: 60% (180 samples)     │
        │   └─ Recommendations: 3              │
        │                                      │
        │   DOWNLOAD:                          │
        │   • CSV of synthetic data            │
        │   • JSON with analysis               │
        └──────────────────────────────────────┘
                           │
                           ▼
                   USER SEES RESULTS
```

---

## Data Flow Within Each Step

### Step 1: Synthetic Generation

```
Original Data (1,430 rows)
│
├─ Numeric Column "Temperature" (45°C ± 5°C)
│  │
│  ├─ Learn: mean=45, std=5, min=35, max=55
│  │
│  └─ Generate 300 values from N(45, 5²)
│     Result: [44.2, 45.8, 46.1, ..., 44.9]
│
├─ Categorical Column "Maintenance_Priority"
│  │
│  ├─ Learn: Critical=40%, Minor=30%, Planned=30%
│  │
│  └─ Sample 300 values maintaining % distribution
│     Result: [Critical, Minor, Planned, ..., Critical]
│
└─ Output: DataFrame(300 rows, 10 columns)
   → Saved to CSV
```

### Step 2: Quality Analysis

```
Original Dataset         Synthetic Dataset
│                        │
├─ Temp mean: 45.0 ◄────┼─► Temp mean: 44.8
├─ Temp std: 5.0  ◄────┼─► Temp std: 5.1
├─ Priority:           │
│  ├─ Critical: 40% ◄────┼─► Critical: 39%
│  ├─ Minor: 30%    ◄────┼─► Minor: 31%
│  └─ Planned: 30%  ◄────┼─► Planned: 30%
│
└─ Compare Similarity
   │
   ├─ KS Test: 0.98 (very similar)
   ├─ Chi-square: p=0.82 (not significantly different)
   └─ Quality Score: 90/100 ✓
```

### Step 3: Prediction & Analysis

```
Synthetic Dataset (300 rows)
│
├─ Load pretrained model:
│  RandomForestClassifier_Maintenance_Priority.joblib
│
├─ Extract features (exclude target)
│  → 299 columns × 300 rows
│
├─ Run: predictions = model.predict(features)
│  → [1, 0, 2, 1, 0, ..., 1]  (300 predictions)
│
└─ Analyze:
   ├─ Class 0: 120 (40%)
   ├─ Class 1: 180 (60%)
   ├─ Matches original ratio
   └─ Prediction distribution healthy ✓
```

---

## File Storage & Access

### Where Files Are Stored

```
/artifacts/
├── web_synthetic/                          ← Main storage
│   ├── 2026-04-23T10-35-22-abc123_synthetic.csv
│   ├── 2026-04-23T10-35-22-abc123_synthetic_inference.json
│   ├── 2026-04-23T10-45-15-def456_synthetic.csv
│   ├── 2026-04-23T10-45-15-def456_synthetic_inference.json
│   └── ... (more runs)
│
├── pretrained_models/                      ← Models used
│   ├── classification__Maintenance_Priority__RandomForestClassifier.joblib
│   ├── regression__Failure_Prob__RandomForestRegressor.joblib
│   └── registry.json
│
└── web_runs/                               ← Run history
    ├── run_2026-04-23T10-35-22-abc123.json
    └── ... (complete run info)
```

### Access from Code

```python
# In run_manager.py
from pathlib import Path

WEB_SYNTHETIC_DIR = ARTIFACTS_DIR / "web_synthetic"
csv_path = WEB_SYNTHETIC_DIR / f"{run_id}_synthetic.csv"
json_path = WEB_SYNTHETIC_DIR / f"{run_id}_synthetic_inference.json"

# Save files
synthetic_df.to_csv(csv_path, index=False)
json_path.write_text(json.dumps(results))

# Load files (later)
results_df = pd.read_csv(csv_path)
with open(json_path) as f:
    analysis = json.load(f)
```

---

## Quality Score Interpretation

```
Quality Score Scale:
┌─────────────────────────────────────────────────────┐
│                                                     │
│ 85-100 ▓▓▓▓▓ EXCELLENT ✓                           │
│ │      → Use without hesitation                    │
│ │      → Suitable for augmentation                 │
│ │      → Reliable for validation                   │
│                                                     │
│ 70-84  ▓▓▓▓░ GOOD ✓                                │
│ │      → Generally safe to use                     │
│ │      → Review metrics first                      │
│ │      → OK for augmentation                       │
│                                                     │
│ 50-69  ▓▓▓░░ FAIR ⚠️                               │
│ │      → Use with caution                          │
│ │      → Regenerate if possible                    │
│ │      → Not ideal for critical work               │
│                                                     │
│ 0-49   ▓░░░░ POOR ❌                               │
│        → Do not use                                │
│        → Regenerate with different settings        │
│        → Check original data quality first         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Testing Flow

```
Generate Synthetic
       │
       ▼
Quality >= 70?
       │
    ┌──┴──┐
    │     │
   YES    NO
    │     │
    ▼     ▼
Continue  ⚠️  Warn User
    │     │
    ▼     │
Make      │
Predictions
    │
    ▼
Accuracy >= 70%?
       │
    ┌──┴──┐
    │     │
   YES    NO
    │     │
    ▼     ▼
✓ Valid  ⚠️  Check
Data     Data Quality
    │     
    ▼     
Show Results
in Dashboard
```

---

## Example Output JSON

```json
{
  "predictions": [1, 0, 2, 1, 0, ..., 1],
  "prediction_analysis": {
    "problem_type": "classification",
    "total_predictions": 300,
    "unique_classes": 3,
    "class_distribution": {
      "0": 120,
      "1": 180,
      "2": 0
    },
    "class_percentages": {
      "0": 40.0,
      "1": 60.0,
      "2": 0.0
    },
    "most_common_class": "1",
    "least_common_class": "2",
    "accuracy": 0.87,
    "precision": 0.86,
    "recall": 0.87,
    "f1_score": 0.865,
    "recommendations": [
      {
        "severity": "INFO",
        "category": "Quality",
        "message": "Quality score 90/100 - Excellent"
      },
      {
        "severity": "WARNING",
        "category": "Balance",
        "message": "Class 2 has 0% representation"
      }
    ]
  }
}
```

---

## Performance Metrics

### Typical Quality Scores by Dataset
```
Well-distributed, large dataset (>1000 rows)
    → Quality: 85-95 ✓ EXCELLENT

Normal dataset (500-1000 rows)
    → Quality: 70-85 ✓ GOOD

Small dataset (<500 rows)
    → Quality: 50-70 ⚠️ FAIR

Highly skewed dataset
    → Quality: 40-60 ⚠️ FAIR/POOR
```

### Prediction Accuracy Correlation
```
Quality Score | Expected Prediction Accuracy
    90-100    → 85-95%  (High confidence)
    70-89     → 75-90%  (Moderate confidence)
    50-69     → 60-80%  (Use with caution)
    0-49      → <60%    (Not recommended)
```

---

## Summary

**3-Part Process:**
1. **Generate** - Create synthetic rows from learned distributions
2. **Analyze** - Compare with original (quality score 0-100)
3. **Validate** - Make predictions and check performance

**Where Stored:**
- CSV: `artifacts/web_synthetic/{run_id}_synthetic.csv`
- JSON: `artifacts/web_synthetic/{run_id}_synthetic_inference.json`

**How Tested:**
- Quality metrics (KS test, Chi-square)
- Distribution similarity
- Prediction accuracy
- Class balance analysis

**User Gets:**
- Quality score (0-100) with interpretation
- Prediction results
- Recommendations
- Downloadable artifacts
