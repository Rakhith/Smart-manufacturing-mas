# 🚀 Synthetic Data Generation - Hands-On Guide

## Quick Reference

### How to Generate
```
1. Open: http://127.0.0.1:8001
2. Load dataset (dropdown or upload CSV)
3. Configure:
   - Problem Type: classification or regression
   - Target Column: select from dropdown
4. Check: ✅ "Generate synthetic data"
5. Set: Synthetic rows = 300 (or desired amount)
6. Click: "Launch Run"
7. Wait: Pipeline executes (2-5 minutes typically)
8. View: Results in dashboard
9. Download: CSV + JSON artifacts
```

---

## What Gets Created - File by File

### File 1: Synthetic Data CSV
**Location:** `artifacts/web_synthetic/{run_id}_synthetic.csv`

**What it contains:**
- New rows generated from learned distributions
- Same columns as original dataset
- Can be used for further analysis or training

**Example:**
```csv
Temperature,Pressure,Vibration,Maintenance_Priority,Machine_ID,Failure_Prob
45.2,12.3,0.8,Critical,M001,0.85
48.1,11.9,0.7,Minor,M002,0.25
44.8,12.1,0.82,Critical,M003,0.92
46.5,12.5,0.75,Planned,M004,0.45
```

**Size:**
- Rows: As many as you requested (e.g., 300)
- Columns: Same as original (e.g., 10 columns)
- File size: Usually 10-50 KB for 300 rows

---

### File 2: Inference Results JSON
**Location:** `artifacts/web_synthetic/{run_id}_synthetic_inference.json`

**What it contains:**
- Predictions made by pretrained model on each synthetic row
- Quality analysis metrics
- Performance metrics
- Recommendations

**Structure:**
```json
{
  "predictions": [
    "Critical", "Minor", "Planned", "Critical", ...
  ],
  "prediction_analysis": {
    "problem_type": "classification",
    "total_predictions": 300,
    "analysis": {
      "accuracy": 0.87,
      "class_distribution": {
        "Critical": 120,
        "Minor": 100,
        "Planned": 80
      }
    },
    "recommendations": [
      {
        "severity": "INFO",
        "category": "Quality",
        "message": "Synthetic data quality is excellent"
      }
    ]
  }
}
```

**File size:** Usually 50-200 KB

---

## How Data is Generated - Technical Details

### Numeric Columns
```
Original: Temperature = [43, 44, 45, 46, 47]
           mean = 45.0, std = 1.58

Generated: Sample 300 values from Normal(45.0, 1.58²)
Result: [44.2, 45.8, 46.1, 44.9, 45.3, ...]
```

**Algorithm:**
1. Calculate mean and std from original
2. For each synthetic row, sample: `value = mean + random() × std`
3. Repeat N times

---

### Categorical Columns
```
Original: Maintenance_Priority = ["Critical", "Minor", "Planned", "Critical", ...]
          Distribution: Critical=40%, Minor=30%, Planned=30%

Generated: Sample 300 values maintaining 40/30/30% distribution
Result: ["Critical", "Minor", "Critical", "Planned", ...]
```

**Algorithm:**
1. Count each category proportion in original
2. For each synthetic row, randomly sample a category:
   - 40% chance of "Critical"
   - 30% chance of "Minor"
   - 30% chance of "Planned"
3. Repeat N times

---

## How Quality is Measured

### For Numeric Columns

**1. Mean Similarity**
```
Original mean: 45.0°C
Synthetic mean: 44.8°C
Difference: 0.2°C
% Difference: 0.2/45.0 × 100 = 0.44%
Score: 99/100 ✓
```

**2. Distribution Shape (KS Test)**
```
Kolmogorov-Smirnov statistic: 0.05 (range 0-1)
- 0 = identical distributions
- 1 = completely different

0.05 → Very similar ✓
```

---

### For Categorical Columns

**1. Proportion Difference**
```
Original: Critical = 40%
Synthetic: Critical = 39%
Difference: 1%
Score: 99/100 ✓
```

**2. Chi-Square Test**
```
Compares observed vs expected frequencies
p-value > 0.05 → distributions are similar ✓
p-value < 0.05 → distributions are different ⚠️
```

---

### Overall Quality Score

```
Quality = 0.4 × numeric_score + 0.4 × categorical_score + 0.2 × health_score

Where:
- numeric_score: Average similarity of all numeric columns
- categorical_score: Average similarity of all categorical columns
- health_score: Data completeness and integrity
```

**Interpretation:**
```
85-100: EXCELLENT ✓    → Use without hesitation
70-84:  GOOD ✓         → Generally safe to use
50-69:  FAIR ⚠️        → Use with caution
0-49:   POOR ❌        → Do not use
```

---

## How Predictions Work

### Step 1: Load Pretrained Model
```
Bundle file: classification__Maintenance_Priority__RandomForestClassifier.joblib

Contains:
├─ Trained model
├─ Feature columns list
├─ Target column name
└─ Preprocessing parameters
```

### Step 2: Prepare Synthetic Data
```
Raw Synthetic Data (300 rows × 10 columns)
         │
         ├─ Remove target column
         │
         ├─ Select only feature columns
         │
         └─ Align to model's expected features
            
Result: Feature matrix (300 × 9)
```

### Step 3: Make Predictions
```
predictions = model.predict(feature_matrix)

Returns: ["Critical", "Minor", "Critical", ...] × 300
```

### Step 4: Analyze Predictions
```
Analysis includes:
├─ Count per class
├─ Distribution percentages
├─ Most/least common prediction
├─ Class balance assessment
└─ Recommendations
```

---

## How Testing is Done

### Test 1: Quality Score Test
```python
if quality_score >= 70:
    print("✓ PASS - Synthetic data is reliable")
else:
    print("⚠ FAIL - Consider regenerating")
```

### Test 2: Distribution Similarity
```python
# For numeric columns
ks_statistic < 0.1 → PASS ✓
ks_statistic >= 0.1 → FAIL ⚠️

# For categorical columns
chi2_pvalue > 0.05 → PASS ✓
chi2_pvalue <= 0.05 → FAIL ⚠️
```

### Test 3: Class Balance
```python
original_prop = original_df.groupby('target').size() / len(original_df)
synthetic_prop = synthetic_df.groupby('target').size() / len(synthetic_df)

diff = abs(original_prop - synthetic_prop)
if all(diff < 0.05):  # Within 5%
    print("✓ PASS - Class balance maintained")
else:
    print("⚠ WARNING - Class imbalance detected")
```

### Test 4: Model Performance
```python
# If we had actual labels (we don't for synthetic)
# But we show what model WOULD predict
predictions_accuracy = calculate_accuracy(predictions, true_labels)

if predictions_accuracy >= 0.70:
    print("✓ PASS - Model works well on synthetic data")
else:
    print("⚠ WARNING - Model accuracy dropped")
```

---

## Example: Complete Run

### Scenario: Generate 300 rows from Manufacturing Dataset

**Input:**
```
Original Dataset: smart_maintenance_dataset.csv
- Rows: 1,430
- Columns: 10 (Temperature, Pressure, Vibration, etc.)
- Target: Maintenance_Priority (Critical, Minor, Planned)
- Problem Type: Classification

User Request: Generate 300 synthetic rows
```

**Process:**

1. **Analysis Phase (2 seconds)**
   ```
   Analyzing original data...
   • Temperature: mean=45.2°C, std=5.1
   • Pressure: mean=12.0 bar, std=0.8
   • Maintenance_Priority: Critical=40%, Minor=30%, Planned=30%
   ```

2. **Generation Phase (1 second)**
   ```
   Generating 300 synthetic rows...
   • Sampling 300 temperature values from N(45.2, 5.1²)
   • Sampling 300 pressure values from N(12.0, 0.8²)
   • Sampling 300 priorities: 40% Critical, 30% Minor, 30% Planned
   ```

3. **Quality Analysis Phase (2 seconds)**
   ```
   Comparing synthetic vs original:
   • Temperature similarity: 95% ✓
   • Pressure similarity: 92% ✓
   • Priority distribution: 98% ✓
   • Overall quality: 95/100 ✓ EXCELLENT
   ```

4. **Prediction Phase (1 second)**
   ```
   Loading: RandomForestClassifier_Maintenance_Priority
   Making 300 predictions...
   • Critical: 119 (40%) ✓
   • Minor: 90 (30%) ✓
   • Planned: 91 (30%) ✓
   ```

5. **Results Generated**
   ```
   CSV saved to: artifacts/web_synthetic/abc123_synthetic.csv (50 KB)
   JSON saved to: artifacts/web_synthetic/abc123_synthetic_inference.json (120 KB)
   ```

**Output (Dashboard):**
```
┌─────────────────────────────────────┐
│ SYNTHETIC DATA ANALYSIS RESULTS     │
├─────────────────────────────────────┤
│                                     │
│ Quality Score: 95/100 ✓ EXCELLENT   │
│                                     │
│ Metrics:                            │
│ • Temperature similarity: 95%       │
│ • Pressure similarity: 92%          │
│ • Distribution preserved: 98%       │
│                                     │
│ Predictions:                        │
│ • Model: RandomForestClassifier    │
│ • Predictions: 300                 │
│ • Critical: 40%                    │
│ • Minor: 30%                       │
│ • Planned: 30%                     │
│                                     │
│ Recommendations:                    │
│ ✓ Quality score excellent          │
│ ✓ Safe to use for augmentation     │
│ ✓ Class distribution well-balanced │
│                                     │
│ [Download CSV] [Download JSON]      │
└─────────────────────────────────────┘
```

---

## Storage & File Management

### Directory Structure
```
artifacts/
├── web_synthetic/
│   ├── 2026-04-23T10-35-22-abc123_synthetic.csv
│   ├── 2026-04-23T10-35-22-abc123_synthetic_inference.json
│   ├── 2026-04-23T10-45-15-def456_synthetic.csv
│   ├── 2026-04-23T10-45-15-def456_synthetic_inference.json
│   └── ... (20+ GB of historical runs possible)
```

### Cleanup (if needed)
```bash
# Remove old synthetic data (keep last 10)
cd /path/to/artifacts/web_synthetic
ls -t | tail -n +21 | xargs rm -f

# Or remove all
rm -rf artifacts/web_synthetic/*
```

---

## Common Use Cases & Workflows

### Use Case 1: Validate Before Production Deployment
```
Workflow:
1. Generate 500 synthetic rows
2. Check quality score (should be >80)
3. Check model predictions (should match training accuracy)
4. If both pass → Ready for production
5. If fail → Improve original data quality first
```

### Use Case 2: Augment Training Data
```
Workflow:
1. Generate 1000 synthetic rows
2. Check quality score (should be >75)
3. Combine with original: pd.concat([original, synthetic])
4. Retrain model on combined dataset
5. Evaluate on test set
```

### Use Case 3: Test Model Generalization
```
Workflow:
1. Generate 300 synthetic rows (as validation set)
2. Check model accuracy on synthetic data
3. Compare with accuracy on original test set
   • Similar? → Model generalizes well ✓
   • Drops? → Model may be overfitting ⚠️
```

### Use Case 4: Identify Data Quality Issues
```
Workflow:
1. Generate synthetic data
2. If quality score <70:
   • Check original data for outliers
   • Look for class imbalance
   • Verify data types
3. Clean original data
4. Regenerate synthetic data
```

---

## Troubleshooting

### Problem: Quality Score Too Low (<50)

**Diagnostics:**
```python
# Check original data size
if len(original_df) < 100:
    print("❌ Original dataset too small - quality will be poor")
    
# Check for extreme values
if (original_df > original_df.quantile(0.95)).sum().sum() > 0:
    print("⚠️ Many outliers detected - might affect quality")
    
# Check class imbalance
class_dist = original_df['target'].value_counts(normalize=True)
if class_dist.max() > 0.9:
    print("⚠️ Severe class imbalance - quality will be affected")
```

**Solutions:**
1. Increase original dataset size to 500+ rows
2. Remove outliers: `df = df[(df > df.quantile(0.01)) & (df < df.quantile(0.99))]`
3. Handle class imbalance with stratification
4. Regenerate synthetic data

---

### Problem: Predictions Don't Make Sense

**Diagnostics:**
```python
# Check feature alignment
print(f"Expected features: {model.feature_names_}")
print(f"Got features: {synthetic_df.columns.tolist()}")

# Check data types
print(synthetic_df.dtypes)
print(model.input_types)
```

**Solutions:**
1. Verify target_column is correct
2. Check feature_columns match model expectations
3. Ensure no missing values in synthetic data
4. Try with different preferred_model

---

## API Reference (for developers)

### Generate Synthetic Data (Python)
```python
from utils.synthetic_quality_analyzer import SyntheticQualityAnalyzer
from utils.prediction_analyzer import PredictionAnalyzer

# Quality Analysis
analyzer = SyntheticQualityAnalyzer(original_df, synthetic_df)
quality_report = analyzer.get_summary_for_display()

# Prediction Analysis
pred_analyzer = PredictionAnalyzer(
    predictions=predictions,
    problem_type="classification"
)
pred_summary = pred_analyzer.get_summary()
```

### Access Results (via API endpoints)
```python
# GET /api/run/{run_id}
# Returns: Complete run info including synthetic generation

# Download artifacts
# GET /api/run/{run_id}/artifacts/{artifact_id}
```

---

## Key Takeaways

✅ **Synthetic data generation creates new rows by:**
1. Learning statistical properties of original data
2. Sampling new values from learned distributions
3. Maintaining original data patterns

✅ **Quality is measured by:**
1. Distribution similarity (0-100 score)
2. Statistical metric comparison
3. Chi-square and KS tests

✅ **Testing involves:**
1. Quality score validation
2. Distribution comparison
3. Model prediction accuracy
4. Class balance analysis

✅ **Files stored in:**
- `artifacts/web_synthetic/{run_id}_synthetic.csv`
- `artifacts/web_synthetic/{run_id}_synthetic_inference.json`

✅ **When to use synthetic data:**
- Quality score ≥ 70: Generally safe
- Quality score ≥ 85: Excellent for use
- Quality score < 50: Don't use, regenerate
