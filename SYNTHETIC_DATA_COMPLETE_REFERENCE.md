# 📚 Synthetic Data Generation - Complete Reference

## Overview Summary

**Question:** "How exactly does the synthetic generation work? If selected, how is it getting created and where is it stored and how is testing done?"

**Short Answer:**
```
When user clicks "Generate synthetic data":
1. CREATE:   Analyzes original data patterns → Generates N new rows
2. STORE:    Saves CSV to artifacts/web_synthetic/{run_id}_synthetic.csv
3. TEST:     Compares vs original → Quality score (0-100)
4. PREDICT:  Runs pretrained model → Gets predictions
5. ANALYZE:  Shows metrics + recommendations in dashboard
```

---

## 3-Part Summary

### Part 1: CREATION - How Synthetic Data is Generated

**File:** `webapp/run_manager.py` (Class: `_SyntheticDataGenerator`)

**Process:**
```
Input: Original dataset (e.g., 1,430 rows × 10 columns)
       Number of rows to generate (e.g., 300)

For each column:
  IF numeric column (Temperature, Pressure, etc.):
    ├─ Calculate: mean, std deviation
    ├─ Learn distribution: Normal(mean, std²)
    └─ Generate: Sample 300 values from distribution
    
  IF categorical column (Maintenance_Priority, etc.):
    ├─ Calculate: Proportion of each category (40%, 30%, 30%)
    ├─ Learn distribution: Multinomial(p=[0.4, 0.3, 0.3])
    └─ Generate: Sample 300 values maintaining proportions

Output: New synthetic dataset (300 rows × 10 columns)
```

**Example:**
```
Original Temperature: [43, 44, 45, 46, 47] → mean=45, std=1.58
Generated Temperature: [44.2, 45.8, 46.1, 44.9, 45.3, ...] × 300

Original Priority: [Critical(40%), Minor(30%), Planned(30%)]
Generated Priority: [Critical, Minor, Critical, Planned, ...] × 300
```

### Part 2: STORAGE - Where Files Are Saved

**Location:** `artifacts/web_synthetic/`

**Two files created per run:**

```
1. CSV FILE
   Name: {run_id}_synthetic.csv
   Size: ~50 KB (for 300 rows)
   Contains: Raw synthetic dataset
   Example path: artifacts/web_synthetic/2026-04-23T10-35-22-abc123_synthetic.csv
   
   Content:
   Temperature,Pressure,Vibration,Maintenance_Priority,Machine_ID,Failure_Prob
   45.2,12.3,0.8,Critical,M001,0.85
   48.1,11.9,0.7,Minor,M002,0.25

2. JSON FILE
   Name: {run_id}_synthetic_inference.json
   Size: ~120 KB (for 300 predictions)
   Contains: Predictions + Analysis + Recommendations
   Example path: artifacts/web_synthetic/2026-04-23T10-35-22-abc123_synthetic_inference.json
   
   Content:
   {
     "predictions": [1, 0, 2, 1, ...],
     "prediction_analysis": {
       "total_predictions": 300,
       "accuracy": 0.87,
       "class_distribution": {...},
       "recommendations": [...]
     }
   }
```

### Part 3: TESTING - How Quality is Validated

**File:** `utils/synthetic_quality_analyzer.py` & `utils/prediction_analyzer.py`

**Testing Approach:**

```
TEST 1: QUALITY ANALYSIS
├─ Compare Numeric Columns
│  ├─ Mean difference (%)
│  ├─ Std deviation difference (%)
│  ├─ Kolmogorov-Smirnov test (0=identical, 1=different)
│  └─ Range coverage
│
├─ Compare Categorical Columns
│  ├─ Proportion difference per category (%)
│  ├─ Chi-square test (p>0.05 = similar)
│  └─ Category diversity
│
├─ Calculate Overall Quality Score
│  40% numeric similarity + 40% categorical + 20% health
│
└─ Result: Quality Score (0-100)
   • 85-100: EXCELLENT ✓
   • 70-84:  GOOD ✓
   • 50-69:  FAIR ⚠️
   • 0-49:   POOR ❌

TEST 2: PREDICTION VALIDATION
├─ Load pretrained model (e.g., RandomForestClassifier)
├─ Make predictions on 300 synthetic rows
├─ Analyze prediction distribution
├─ Check class balance matches original
└─ Generate recommendations based on metrics

TEST 3: COMPARISON TESTS
├─ Statistical Tests:
│  • KS Test: Is distribution similar?
│  • Chi-square: Are proportions consistent?
│
├─ Similarity Metrics:
│  • Mean diff < 5%: PASS ✓
│  • Std diff < 5%: PASS ✓
│
└─ Decision:
   All tests pass → Quality ≥ 70 → Safe to use
   Some tests fail → Quality < 70 → Regenerate
```

---

## Technical Details - How Each Component Works

### Component 1: Synthetic Data Generator

**File:** `webapp/run_manager.py` → `_SyntheticDataGenerator` class

**Key Methods:**
```python
def generate(self, raw_df, n_rows, target_column):
    """Generate N synthetic rows"""
    
def analyze_numeric_column(series):
    """Extract mean, std, min, max"""
    
def analyze_categorical_column(series):
    """Extract category proportions"""
    
def generate_numeric_value(stats):
    """Sample from learned normal distribution"""
    
def generate_categorical_value(stats):
    """Sample from learned categorical distribution"""
```

**Algorithm:**
```python
def generate(self, raw_df, n_rows, target_column):
    synthetic_rows = []
    
    for _ in range(n_rows):  # Create N new rows
        row = {}
        
        for col in raw_df.columns:
            if col in ['Machine_ID']:  # Skip ID columns
                continue
                
            if is_numeric(col):
                stats = self.analyze_numeric_column(raw_df[col])
                row[col] = self.generate_numeric_value(stats)
                
            elif is_categorical(col):
                stats = self.analyze_categorical_column(raw_df[col])
                row[col] = self.generate_categorical_value(stats)
        
        synthetic_rows.append(row)
    
    return pd.DataFrame(synthetic_rows)
```

---

### Component 2: Quality Analyzer

**File:** `utils/synthetic_quality_analyzer.py` → `SyntheticQualityAnalyzer` class

**What it compares:**

```python
class SyntheticQualityAnalyzer:
    def compare_numeric_distributions(self):
        # For each numeric column:
        # 1. Calculate original stats (mean, median, std, min, max, quartiles)
        # 2. Calculate synthetic stats (same)
        # 3. Compare:
        #    - Mean difference %
        #    - Std difference %
        #    - KS statistic (distribution shape similarity)
        #    - Range coverage
        return comparison_dict
    
    def compare_categorical_distributions(self):
        # For each categorical column:
        # 1. Calculate original proportions
        # 2. Calculate synthetic proportions
        # 3. Compare:
        #    - Proportion difference per category %
        #    - Chi-square test (are distributions same?)
        #    - Category presence
        return comparison_dict
    
    def get_quality_score(self):
        # Combine all metrics into 0-100 score
        # Weight: 40% numeric + 40% categorical + 20% health
        return score (0-100)
```

**Example Output:**
```python
quality_report = {
    "quality_score": 90,
    "quality_level": "EXCELLENT",
    "numeric_analysis": {
        "Temperature": {
            "mean_diff_pct": 0.44,
            "std_diff_pct": 2.15,
            "ks_statistic": 0.05,
            "similarity_score": 98
        },
        ...
    },
    "categorical_analysis": {
        "Maintenance_Priority": {
            "Critical": {"original": 40.0, "synthetic": 39.0, "diff": 1.0},
            "Minor": {"original": 30.0, "synthetic": 31.0, "diff": 1.0},
            ...
        },
        ...
    },
    "recommendations": [
        "Synthetic data quality is excellent (90/100)",
        "All distributions match original within 5%",
        "Safe to use for augmentation"
    ]
}
```

---

### Component 3: Prediction Analyzer

**File:** `utils/prediction_analyzer.py` → `PredictionAnalyzer` class

**What it analyzes:**

```python
class PredictionAnalyzer:
    def analyze_classification_predictions(self):
        # Count: How many predictions per class?
        # Proportion: What % is each class?
        # Balance: Are classes balanced like original?
        return analysis_dict
    
    def analyze_regression_predictions(self):
        # Stats: Mean, std, min, max of predictions
        # Distribution: Shape of prediction distribution
        # Outliers: Are there extreme predictions?
        return analysis_dict
    
    def generate_recommendations(self):
        # Based on metrics, generate 3-5 recommendations
        # With severity levels: INFO, WARNING, ERROR
        return recommendations_list
```

**Example Output (Classification):**
```python
analysis = {
    "total_predictions": 300,
    "unique_classes": 3,
    "class_distribution": {
        "Critical": 120,
        "Minor": 90,
        "Planned": 90
    },
    "class_percentages": {
        "Critical": 40.0,
        "Minor": 30.0,
        "Planned": 30.0
    },
    "most_common_class": "Critical",
    "least_common_class": "Minor",
    "recommendations": [
        {
            "severity": "INFO",
            "category": "Quality",
            "message": "Synthetic data quality is good (90/100)"
        },
        {
            "severity": "INFO",
            "category": "Balance",
            "message": "Class distribution well maintained: matches original"
        }
    ]
}
```

---

## Complete Data Flow Diagram

```
┌──────────────────────────┐
│   USER INPUT             │
│ Generate 300 synthetic   │
│ rows from original data  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────────────┐
│   SYNTHETIC GENERATION           │
│ run_manager.py                   │
├──────────────────────────────────┤
│ 1. Load original data            │
│ 2. Analyze distributions:        │
│    • Numeric: mean, std          │
│    • Categorical: proportions    │
│ 3. Generate 300 new rows by:     │
│    • Sampling from distributions │
│    • Maintaining patterns        │
│ 4. Save → CSV file               │
└────────────┬─────────────────────┘
             │
             ▼
    ┌─────────────────────┐
    │ artifacts/          │
    │ web_synthetic/      │
    │ {run_id}            │
    │ _synthetic.csv      │
    │ (300 rows × 10 cols)│
    └────────┬────────────┘
             │
    ┌────────┴───────────────────┐
    │                            │
    ▼                            ▼
┌─────────────────────┐  ┌─────────────────────┐
│  QUALITY ANALYZER   │  │ PREDICTION ENGINE   │
│ synthetic_quality_  │  │ Load pretrained     │
│ analyzer.py         │  │ model               │
├─────────────────────┤  │                     │
│ Compare:            │  │ Predict 300 values: │
│ • Distributions     │  │ • Load model        │
│ • Statistics        │  │ • Run predictions   │
│ • KS test           │  │ • Store results     │
│ • Chi-square test   │  │                     │
│                     │  │ Output:             │
│ Output:             │  │ predictions[] + metrics
│ Quality: 0-100      │  │                     │
│ + recommendations   │  │                     │
└────────┬────────────┘  └────────┬────────────┘
         │                        │
         └────────────┬───────────┘
                      │
                      ▼
         ┌──────────────────────────┐
         │  Save Inference Results  │
         ├──────────────────────────┤
         │ {run_id}                 │
         │ _synthetic_inference.json│
         │                          │
         │ Contains:                │
         │ • predictions array      │
         │ • quality analysis       │
         │ • prediction analysis    │
         │ • recommendations        │
         └────────────┬─────────────┘
                      │
                      ▼
         ┌──────────────────────────┐
         │  Return to Web Dashboard │
         ├──────────────────────────┤
         │ Display 2 cards:         │
         │ 1. Data Quality          │
         │    Score: 90/100 ✓       │
         │    Metrics               │
         │                          │
         │ 2. Predictions           │
         │    Accuracy: 87%         │
         │    Class dist: ...       │
         │    Recommendations       │
         │                          │
         │ Download buttons         │
         │ ✓ CSV                    │
         │ ✓ JSON                   │
         └──────────────────────────┘
```

---

## Key Metrics Reference

### Quality Score Components
```
Quality Score = W1×numeric + W2×categorical + W3×health

W1 = 40% (Numeric distribution similarity)
  Components:
  • Mean similarity (KS test) - 50% weight
  • Std deviation similarity - 30% weight
  • Range coverage - 20% weight
  
W2 = 40% (Categorical distribution similarity)
  Components:
  • Chi-square test p-value - 50% weight
  • Proportion differences - 30% weight
  • Category presence - 20% weight
  
W3 = 20% (Data health)
  Components:
  • Column coverage - 50% weight
  • Value diversity - 50% weight
```

### Interpretation
```
85-100: EXCELLENT  → Distributions nearly identical
                   → Safe for all use cases
                   → Suitable for augmentation

70-84:  GOOD       → Distributions very similar
                   → Generally safe to use
                   → Minor differences acceptable

50-69:  FAIR       → Distributions somewhat similar
                   → Use with caution
                   → Review metrics before use

0-49:   POOR       → Distributions significantly different
                   → Not recommended
                   → Regenerate with different approach
```

---

## Practical Usage Checklist

### ✅ When to Use Synthetic Data
- Quality score ≥ 70
- Key distributions maintained
- Class balance acceptable
- Model accuracy on synthetic ≥ 70%

### ⚠️ When to Regenerate
- Quality score < 70
- Mean difference > 10%
- Class imbalance > 20%
- Model accuracy drops significantly

### ❌ When NOT to Use
- Quality score < 50
- Original data < 100 rows
- Severe class imbalance in original
- Too many missing values

---

## Files Reference

| File | Purpose | Key Classes |
|------|---------|------------|
| `run_manager.py` | Orchestrates entire pipeline | `PipelineRunManager`, `_SyntheticDataGenerator` |
| `synthetic_quality_analyzer.py` | Compares data distributions | `SyntheticQualityAnalyzer` |
| `prediction_analyzer.py` | Analyzes predictions | `PredictionAnalyzer` |
| `pretrained_model_store.py` | Loads pretrained models | `select_bundle_metadata`, `load_bundle` |
| `app.js` | Displays results in UI | Dashboard rendering functions |

---

## FAQ

**Q: How is randomness controlled?**
```python
# Seed is fixed for reproducibility
generator = _SyntheticDataGenerator(seed=42)
# Same input → Same synthetic data always
```

**Q: Can I regenerate and get different data?**
```python
# Yes, but seed 42 gives same results
# To get different: Pass different seed
# (Requires code modification)
```

**Q: What if quality is low?**
```
Options:
1. Original data might be too small (< 100 rows)
2. Original data has extreme outliers
3. Original has severe class imbalance
→ Solution: Clean original data first
```

**Q: How long does generation take?**
```
Typical times:
• 300 rows: 2-5 seconds
• 1000 rows: 3-8 seconds
• 10000 rows: 10-30 seconds
Depends on: Dataset size + complexity
```

**Q: Can I combine synthetic + original?**
```python
combined = pd.concat([original_df, synthetic_df])
# Now you have 1730 rows for training
```

---

## Summary

```
CREATION:   Original data patterns → Statistical distributions
            → Sampled new values → Synthetic dataset

STORAGE:    Synthetic CSV file
            + Inference JSON file
            → artifacts/web_synthetic/

TESTING:    Quality analyzer (0-100 score)
            + Prediction analyzer (accuracy metrics)
            + Statistical tests (KS, Chi-square)
            → Dashboard display + recommendations
```

**The entire process is automated when user checks the "Generate synthetic data" checkbox!**
