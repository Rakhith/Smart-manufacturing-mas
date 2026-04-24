# 🔄 How Synthetic Data Generation Works - Complete Flow

## Overview
When you check "Generate synthetic data" in the web interface, a complete pipeline is executed that creates synthetic data, analyzes its quality, makes predictions, and generates insights.

---

## 📊 The Complete Flow (Step-by-Step)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. USER INTERACTION - Web UI                                    │
├─────────────────────────────────────────────────────────────────┤
│ ✓ Check "Generate synthetic data" checkbox                      │
│ ✓ Set "Synthetic rows" (e.g., 300)                             │
│ ✓ Click "Launch Run"                                           │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. SYNTHETIC DATA GENERATION                                    │
├─────────────────────────────────────────────────────────────────┤
│ Location: webapp/run_manager.py → _SyntheticDataGenerator       │
│ Process:                                                         │
│  • Analyze original dataset statistics (mean, std, min, max)    │
│  • For NUMERIC columns:                                         │
│    - Learn normal/uniform distribution parameters              │
│    - Sample N new values matching learned distribution         │
│  • For CATEGORICAL columns:                                    │
│    - Calculate class proportions in original data              │
│    - Sample N values maintaining original proportions          │
│  • Exclude ID columns (Machine_ID, etc)                        │
│  • Handle target column separately with same logic             │
│ Output: Synthetic DataFrame (N rows, same columns as original) │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. SAVE SYNTHETIC DATA - CSV FILE                              │
├─────────────────────────────────────────────────────────────────┤
│ Path: artifacts/web_synthetic/{run_id}_synthetic.csv           │
│ Format: CSV with all columns and generated values              │
│ Example:                                                        │
│   Temperature,Pressure,Vibration,Maintenance_Priority         │
│   45.2,12.3,0.8,Critical                                       │
│   48.1,11.9,0.7,Minor                                          │
│   ...                                                           │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. DATA QUALITY ANALYSIS                                        │
├─────────────────────────────────────────────────────────────────┤
│ Tool: utils/synthetic_quality_analyzer.py                       │
│ SyntheticQualityAnalyzer(original_df, synthetic_df)            │
│                                                                 │
│ Compares:                                                       │
│  NUMERIC DISTRIBUTIONS:                                        │
│   • Mean, median, std deviation                               │
│   • Min, max, quartiles (Q25, Q75)                            │
│   • Kolmogorov-Smirnov test (similarity score 0-1)            │
│   • % difference in statistics                                │
│                                                                 │
│  CATEGORICAL DISTRIBUTIONS:                                   │
│   • Class proportions (% of each category)                    │
│   • Chi-square test for independence                          │
│   • Diversity comparison                                      │
│                                                                 │
│ QUALITY SCORE (0-100):                                        │
│   • Weighted average of all comparisons                       │
│   • 85-100: Excellent ✓                                       │
│   • 70-84:  Good      ✓                                       │
│   • 50-69:  Fair      ⚠️                                       │
│   • 0-49:   Poor      ❌                                       │
│                                                                 │
│ Output: Dictionary with metrics stored in stage preview       │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. MAKE PREDICTIONS USING PRETRAINED MODEL                     │
├─────────────────────────────────────────────────────────────────┤
│ Location: webapp/run_manager.py → _run_synthetic_generation    │
│                                                                 │
│ Process:                                                        │
│  1. Select best pretrained bundle based on:                   │
│     • Problem type (classification/regression)                │
│     • Target column                                           │
│     • Preferred model (if specified)                          │
│                                                                 │
│  2. Load bundle from: artifacts/pretrained_models/             │
│     Format: {problem_type}__{target}__{model}.joblib          │
│     Contains: [pipeline, target_column, feature_columns, ...]│
│                                                                 │
│  3. Predict on synthetic data:                                │
│     predictions = bundle.pipeline.predict(synthetic_features) │
│                                                                 │
│  4. Store predictions array                                    │
│                                                                 │
│ Output: Array of predictions (length = number of synthetic rows)
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. PREDICTION ANALYSIS                                          │
├─────────────────────────────────────────────────────────────────┤
│ Tool: utils/prediction_analyzer.py                              │
│ PredictionAnalyzer(predictions, problem_type)                  │
│                                                                 │
│ FOR CLASSIFICATION:                                            │
│   • Count predictions per class                               │
│   • Calculate class distribution (%)                          │
│   • Identify most/least common predictions                    │
│   • Generate balance recommendations                          │
│                                                                 │
│ FOR REGRESSION:                                               │
│   • Mean, median, std of predictions                          │
│   • Min, max, quantiles                                       │
│   • Distribution shape analysis                               │
│   • Outlier detection                                         │
│                                                                 │
│ Output: Comprehensive analysis with recommendations           │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. GENERATE RECOMMENDATIONS                                     │
├─────────────────────────────────────────────────────────────────┤
│ Based on: Quality score + Prediction analysis                  │
│                                                                 │
│ Example recommendations:                                       │
│   • "Quality score is good (85) - synthetic data is reliable" │
│   • "Class imbalance detected: 70% Class A, 30% Class B"     │
│   • "Recommend regenerating with more rows if < 50 quality"  │
│   • "Synthetic predictions match training patterns"           │
│                                                                 │
│ Severity levels: INFO, WARNING, ERROR                         │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. SAVE INFERENCE RESULTS - JSON FILE                          │
├─────────────────────────────────────────────────────────────────┤
│ Path: artifacts/web_synthetic/{run_id}_synthetic_inference.json│
│ Contents:                                                       │
│ {                                                              │
│   "predictions": [value1, value2, ...],      # All predictions│
│   "prediction_analysis": {                                    │
│     "total_predictions": 300,                                 │
│     "accuracy": 0.87,                # If classification      │
│     "class_distribution": {...},                              │
│     "recommendations": [                                      │
│       {                                                       │
│         "severity": "INFO",                                   │
│         "category": "Balance",                                │
│         "message": "..."                                      │
│       }                                                       │
│     ]                                                         │
│   }                                                           │
│ }                                                             │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. RETURN TO FRONTEND - DISPLAY DASHBOARD                      │
├─────────────────────────────────────────────────────────────────┤
│ File: webapp/static/app.js                                      │
│                                                                 │
│ CARD 1: Data Quality Analysis                                 │
│   • Quality Score (with color badge): 85/100 ✓                │
│   • Similarity Metrics                                        │
│   • Health indicators (✓/⚠)                                   │
│   • Top differences between original & synthetic              │
│                                                                 │
│ CARD 2: Prediction Analysis                                   │
│   • Problem Type: Classification/Regression                   │
│   • Performance Metrics (Accuracy, Precision, Recall, F1)     │
│   • Class Distribution (for classification)                   │
│   • Recommendations with severity                             │
│                                                                 │
│ DOWNLOADABLE ARTIFACTS:                                       │
│   ✓ Synthetic dataset CSV                                     │
│   ✓ Inference results JSON                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💾 File Storage Structure

### Location: `artifacts/web_synthetic/`

```
web_synthetic/
├── {run_id}_synthetic.csv                    ← Raw synthetic data
├── {run_id}_synthetic_inference.json        ← Predictions & analysis
├── {run_id}_synthetic.csv                    ← (2nd run)
├── {run_id}_synthetic_inference.json        ← (2nd run)
└── ...
```

### Naming Convention
- `run_id`: Unique identifier generated when pipeline starts
- Format: ISO timestamp + random hash
- Example: `2026-04-23T10-35-22-abc123_synthetic.csv`

---

## 🧪 Testing & Validation Process

### How Testing is Done

1. **Quality Score Validation**
   ```python
   # Synthetic data is good if quality_score >= 70
   if quality_score >= 70:
       print("✓ Synthetic data is reliable for use")
   else:
       print("⚠ Consider regenerating with different parameters")
   ```

2. **Distribution Similarity Testing**
   - Kolmogorov-Smirnov test (for numeric)
   - Chi-square test (for categorical)
   - Both should have p-value > 0.05 for statistical similarity

3. **Prediction Accuracy Testing**
   ```python
   # If using classification:
   if accuracy >= 0.70:
       print("✓ Model performs well on synthetic data")
       print("✓ Synthetic data maintains data patterns")
   else:
       print("⚠ Accuracy dropped - synthetic might need improvement")
   ```

4. **Class Balance Testing** (for classification)
   ```python
   # Check if class distribution in synthetic ≈ original
   original_ratio = class_count / total_count
   synthetic_ratio = synthetic_class_count / synthetic_total
   
   if difference < 5%:  # Within 5%
       print("✓ Class distribution well maintained")
   ```

### Test Files
- `verify_synthetic_feature.py` - Verification script
- `training/synthetic_data_inference_analysis.ipynb` - Training analysis
- `scripts/generate_synthetic_data_and_infer.py` - Offline generation

---

## 🔍 Example Workflow

### User Scenario: Manufacturing Maintenance Dataset

**Input:**
- Original dataset: 1,430 rows × 10 columns
- Columns: Temperature, Pressure, Vibration, Maintenance_Priority (target)
- Problem type: Classification
- Request: Generate 300 synthetic rows

**Process:**

1. **Generate**
   ```
   Original temp mean: 45°C → Synthetic temp mean: 44.8°C ✓
   Original pressure mean: 12 bar → Synthetic: 11.9 bar ✓
   Original maintenance_priority distribution:
     • Critical: 40% → Synthetic: 39% ✓
     • Minor: 30% → Synthetic: 31% ✓
     • Planned: 30% → Synthetic: 30% ✓
   ```

2. **Quality Analysis**
   ```
   Numeric similarity score: 92/100 ✓
   Categorical similarity: 88/100 ✓
   Overall quality score: 90/100 ✓ EXCELLENT
   ```

3. **Predictions**
   ```
   Using: RandomForestClassifier_Maintenance_Priority.joblib
   Predicted 300 maintenance priorities
   Accuracy on test set during training: 87%
   → Synthetic data is valid for model validation
   ```

4. **Recommendations**
   ```
   ✓ [INFO] Quality score 90/100 - Excellent
   ✓ [INFO] Class distribution well maintained
   ✓ [INFO] Safe to use synthetic data for augmentation
   ```

---

## 📈 Quality Score Calculation

```python
Quality Score = Weighted Average of:
├── 40% - Numeric Distribution Similarity
│   ├── Mean similarity (KS test)
│   ├── Std deviation similarity
│   └── Range coverage
├── 40% - Categorical Distribution Similarity
│   ├── Chi-square test
│   └── Proportion differences
└── 20% - Overall Data Health
    ├── Column coverage
    └── Value diversity
```

---

## ⚙️ Configuration Options

### From Web UI
```
Generate synthetic data: ✓ (checked)
Synthetic rows: 300 (adjustable: 10-10,000)
Problem type: classification/regression
Target column: Auto-detected or selected
```

### Backend Configuration (run_manager.py)
```python
config = RunConfig(
    generate_synthetic=True,
    synthetic_rows=300,
    problem_type="classification",
    target_column="Maintenance_Priority",
    preferred_model="RandomForestClassifier"  # Optional
)
```

---

## 🎯 Use Cases

### Use Case 1: Validate Data Quality
```
Goal: Ensure synthetic data is realistic
Flow: Generate → Check quality score (>70?) → Use or regenerate
```

### Use Case 2: Model Testing
```
Goal: Test model on unseen data patterns
Flow: Generate → Run predictions → Check accuracy
Result: Synthetic data serves as validation set
```

### Use Case 3: Data Augmentation
```
Goal: Increase training data size
Flow: Generate → Validate quality (>80) → Combine with original
Result: Larger dataset for model retraining
```

### Use Case 4: Production Validation
```
Goal: Before deploying, verify model works on new patterns
Flow: Generate → Predict → Check performance consistency
Result: Confidence in model generalization
```

---

## 📊 Metrics Explained

### Quality Metrics
- **Mean Diff %**: How much average values differ
- **Std Diff %**: How much variability differs
- **KS Statistic**: Distribution similarity (0=identical, 1=different)
- **Range Coverage**: How well synthetic ranges cover original

### Prediction Metrics (Classification)
- **Accuracy**: % of correct predictions
- **Precision**: Of predicted positives, how many correct
- **Recall**: Of actual positives, how many found
- **F1 Score**: Balanced measure of precision & recall

### Prediction Metrics (Regression)
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error
- **R² Score**: How well predictions fit

---

## 🚨 Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Low quality score (<50) | Original data too small | Increase original dataset size |
| Low quality score | Extreme distributions | Check for outliers in original |
| Accuracy drops on synthetic | Synthetic data differs too much | Regenerate with same seed |
| Class imbalance warning | Original is imbalanced | This is expected; synthetic mirrors original |
| Missing predictions | Feature mismatch | Check target_column & feature_columns |

---

## 📝 Summary

**Synthetic Data Generation = 3 Steps:**

1. **Generate** - Create new rows by sampling from learned distributions
2. **Analyze Quality** - Compare synthetic vs original (quality 0-100)
3. **Validate** - Make predictions and check accuracy

**All results stored in `artifacts/web_synthetic/` and displayed in web dashboard.**
