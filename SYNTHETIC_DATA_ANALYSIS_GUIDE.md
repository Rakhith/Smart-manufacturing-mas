# Synthetic Data Generation with Quality Analysis & Predictions

## Overview

This feature enhances the synthetic data generation pipeline with three key capabilities:

1. **Data Quality Analysis** - Compare synthetic vs original dataset distributions
2. **Prediction Analysis** - Make predictions on synthetic data and analyze model performance
3. **Smart Recommendations** - Generate actionable insights and recommendations

---

## Architecture

### New Modules

#### 1. `utils/synthetic_quality_analyzer.py`
**Purpose**: Analyzes data quality and compares original vs synthetic datasets

**Key Classes**:
- `SyntheticQualityAnalyzer`: Main analyzer class

**Key Methods**:

| Method | Purpose | Returns |
|--------|---------|---------|
| `compare_numeric_distributions()` | Compare numeric columns using KS-test, Wasserstein distance | Dict with stats, similarity scores |
| `compare_categorical_distributions()` | Compare categorical columns using Jensen-Shannon divergence | Dict with distribution metrics |
| `compare_missing_values()` | Analyze missing value patterns | Dict with missing value comparison |
| `detect_outliers_comparison()` | Compare outlier patterns using IQR method | Dict with outlier statistics |
| `calculate_overall_quality_score()` | Generate 0-100 quality score | Dict with overall metrics |
| `get_summary_for_display()` | Get concise summary for frontend | Dict optimized for UI display |

**Quality Scoring Logic**:
- Numeric similarity: Based on average KS-statistic across numeric columns
- Categorical similarity: Based on average Jensen-Shannon divergence
- Overall score: Weighted average (numeric weight > categorical for manufacturing data)
- Quality Levels:
  - **85+**: Excellent - Highly representative
  - **70-84**: Good - Captures main characteristics
  - **50-69**: Fair - Acceptable but some differences
  - **<50**: Poor - Differs significantly

#### 2. `utils/prediction_analyzer.py`
**Purpose**: Analyzes predictions and generates recommendations

**Key Classes**:
- `PredictionAnalyzer`: Analyzes model predictions

**Key Methods**:

| Method | Purpose | Returns |
|--------|---------|---------|
| `analyze_classification_predictions()` | Classification metrics (accuracy, precision, recall, F1) | Dict with metrics |
| `analyze_regression_predictions()` | Regression metrics (RMSE, MAE, R², residuals) | Dict with metrics |
| `generate_recommendations()` | AI-driven recommendations based on analysis | List of recommendation dicts |
| `get_summary()` | Summary with both analysis and recommendations | Dict for frontend |

**Recommendation Categories**:
- **Classification**:
  - Class imbalance detection
  - Minority class warnings
  - Model accuracy assessment
- **Regression**:
  - Model fit quality (R² score)
  - Prediction error analysis
  - Residual statistics

### Modified Module

#### `webapp/run_manager.py`
**Changes**:
- Imports: Added `SyntheticQualityAnalyzer`, `PredictionAnalyzer`
- Method: Enhanced `_run_synthetic_generation()` to:
  1. Generate synthetic data (unchanged)
  2. **NEW**: Analyze data quality
  3. **NEW**: Make predictions on synthetic data
  4. **NEW**: Analyze predictions and generate recommendations
  5. Store all results in artifacts and preview data

**Data Flow**:
```
Raw Dataset
    ↓
Synthetic Data Generator
    ↓
[Quality Analysis] ← Compare with original
    ↓
Pretrained Model
    ↓
[Prediction Analysis] ← Generate insights
    ↓
Results stored in:
- CSV file
- JSON with predictions + analysis
- Stage preview with dashboards
```

---

## Frontend Integration

### New JavaScript Functions

#### `renderQualityScoreBadge(score)`
Displays quality score with visual indicator

**Score Ranges**:
- 85+: Green (#4caf50) - Excellent
- 70-84: Light Green (#8bc34a) - Good
- 50-69: Orange (#ff9800) - Fair
- <50: Red (#f44336) - Poor

#### `renderDataQualitySection(qualityData)`
Shows:
- Quality score badge
- Recommendation text
- Similarity metrics (numeric, categorical)
- Health indicators (missing values, outliers)
- Top column differences
- Dataset size comparison

#### `renderPredictionAnalysisSection(predictionData)`
Shows:
- Problem type badge (CLASSIFICATION/REGRESSION)
- Performance metrics (context-dependent)
- Class distribution (for classification)
- Recommendations with color-coded severity:
  - 🟢 Green: Success (≥85% accuracy)
  - 🔵 Blue: Info (general insights)
  - 🟠 Orange: Warning (moderate issues)
  - 🔴 Red: Error (critical issues)

#### `renderSyntheticDashboard(stage)`
Main orchestrator that combines:
1. Data Quality Analysis section
2. Prediction Analysis section

#### Enhanced `renderStages(run)`
Now appends synthetic dashboard to the pipeline stages view when synthetic stage completes

---

## Usage Example

### Frontend Flow

1. User checks "Generate synthetic data" in the form
2. Runs the pipeline
3. System generates synthetic data
4. **NEW**: Analyzes quality vs original
5. **NEW**: Makes predictions with pretrained model
6. **NEW**: Generates recommendations
7. User sees comprehensive dashboard with:
   - Quality score (0-100)
   - Data comparison metrics
   - Prediction analysis
   - Actionable recommendations

### Python API Usage

```python
from utils.synthetic_quality_analyzer import SyntheticQualityAnalyzer
from utils.prediction_analyzer import PredictionAnalyzer

# Quality Analysis
quality_analyzer = SyntheticQualityAnalyzer(original_df, synthetic_df)
quality_report = quality_analyzer.get_summary_for_display()
print(f"Quality Score: {quality_report['quality_score']}")
print(f"Recommendation: {quality_report['recommendation']}")

# Prediction Analysis
pred_analyzer = PredictionAnalyzer(
    predictions=model_predictions,
    problem_type="classification",
)
pred_summary = pred_analyzer.get_summary()
print(f"Accuracy: {pred_summary['analysis']['accuracy']}")
for rec in pred_summary['recommendations']:
    print(f"- [{rec['severity']}] {rec['category']}: {rec['message']}")
```

---

## Data Structures

### Quality Summary (for display)

```json
{
  "quality_score": 82.5,
  "quality_level": "Good",
  "recommendation": "Synthetic data captures main characteristics...",
  "summary_metrics": {
    "numeric_similarity": 0.78,
    "categorical_similarity": 0.85,
    "missing_values_healthy": true,
    "outlier_pattern_healthy": true,
    "total_original_rows": 500,
    "total_synthetic_rows": 300
  },
  "top_column_differences": [
    {"column": "temperature", "mean_difference_pct": 12.5},
    {"column": "pressure", "mean_difference_pct": 8.3}
  ],
  "health_indicators": {
    "missing_values": "✓ Healthy",
    "outlier_patterns": "✓ Healthy",
    "numeric_distribution": "✓ Good",
    "categorical_distribution": "✓ Good"
  }
}
```

### Prediction Summary (for display)

```json
{
  "problem_type": "classification",
  "total_predictions": 300,
  "analysis": {
    "accuracy": 0.92,
    "precision": 0.89,
    "recall": 0.91,
    "f1_score": 0.90,
    "class_distribution": {
      "0": 150,
      "1": 75,
      "2": 75
    }
  },
  "recommendations": [
    {
      "severity": "success",
      "category": "Good Performance",
      "message": "Model achieves 92% accuracy on synthetic data..."
    }
  ]
}
```

---

## Quality Metrics Explained

### Numeric Distribution Comparison

**Kolmogorov-Smirnov (KS) Test**:
- Range: 0 to 1
- 0 = identical distributions
- 1 = completely different
- Similarity Score = 1 - KS_statistic

**Wasserstein Distance**:
- Measures optimal transport distance between distributions
- Lower is better
- Scale-dependent (meaningful in data units)

### Categorical Distribution Comparison

**Jensen-Shannon Divergence**:
- Range: 0 to 1
- 0 = identical distributions
- 1 = completely different
- Symmetric unlike KL divergence
- Similarity Score = 1 - JS_divergence

### Outlier Detection

**IQR Method**:
- Q1 = 25th percentile
- Q3 = 75th percentile
- IQR = Q3 - Q1
- Outlier bounds: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- Compares outlier percentage in original vs synthetic

---

## Files Modified/Created

### Created
- `/smart_manufacturing_mas/utils/synthetic_quality_analyzer.py` (463 lines)
- `/smart_manufacturing_mas/utils/prediction_analyzer.py` (250 lines)

### Modified
- `/smart_manufacturing_mas/webapp/run_manager.py` (imports + _run_synthetic_generation method)
- `/smart_manufacturing_mas/webapp/static/app.js` (added rendering functions + enhanced renderStages)

---

## Testing Checklist

- [x] Python syntax validation
- [ ] Import dependencies available (scipy, numpy, pandas, sklearn)
- [ ] Test with classification problem type
- [ ] Test with regression problem type
- [ ] Verify quality score calculation
- [ ] Verify prediction analysis
- [ ] Test frontend dashboard rendering
- [ ] Verify artifacts are saved correctly
- [ ] Test with edge cases (very small datasets, single class, etc.)

---

## Future Enhancements

1. **Visualization Enhancements**:
   - Distribution plots (histograms, KDE)
   - Correlation heatmaps
   - ROC curves for classification

2. **Advanced Quality Metrics**:
   - Mutual information between columns
   - Copula-based dependency measures
   - Synthetic data validity checks

3. **Recommendation Engine**:
   - ML-based anomaly detection in synthetic data
   - Data augmentation suggestions
   - Model retraining recommendations

4. **Export Options**:
   - Generate HTML report
   - Export analysis to PDF
   - Integration with MLOps tools

---

## Dependencies

- `scipy.stats` - Statistical tests (KS, Wasserstein, Jensen-Shannon)
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `sklearn.metrics` - Classification/regression metrics

All dependencies already in `requirements.txt`
