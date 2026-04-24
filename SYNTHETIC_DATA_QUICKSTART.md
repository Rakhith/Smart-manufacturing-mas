# 🚀 Quick Start: Synthetic Data Enhancement

## Access the Application
**URL**: http://127.0.0.1:8001

## Generate Synthetic Data (3 Steps)

### Step 1: Load Dataset
- Select a dataset from the dropdown OR upload a CSV file
- Preview appears automatically

### Step 2: Configure Run
- **Problem Type**: Select "classification" or "regression"
- **Target Column**: Choose your target variable
- **Feature Columns**: (optional) Specify features
- **Use Cache**: Keep enabled for faster runs

### Step 3: Enable Synthetic Generation
- ✅ Check: "Generate synthetic data"
- Set "Synthetic rows": 300 (adjust as needed, 10-10,000)
- Click **"Launch Run"**

## View Results

### In the Pipeline Stages Section:

#### 1️⃣ Data Quality Analysis Card
Shows:
- **Quality Score** (0-100) with color badge
  - 🟢 85-100: Excellent
  - 🟢 70-84: Good
  - 🟠 50-69: Fair
  - 🔴 0-49: Poor
- **Similarity Metrics**: How close synthetic data matches original
- **Health Indicators**: ✓ or ⚠ for each aspect
- **Top Differences**: Which columns differ most

#### 2️⃣ Prediction Analysis Card
Shows:
- **Problem Type**: Classification or Regression
- **Performance Metrics**: 
  - Classification: Accuracy, Precision, Recall, F1
  - Regression: RMSE, MAE, R²
- **Class Distribution** (classification only)
- **Recommendations**: Auto-generated insights with severity levels

## 📊 Understanding the Metrics

### Quality Score Meaning
- **Excellent (85-100)**: Synthetic data is very similar to original
- **Good (70-84)**: Synthetic data captures key patterns well
- **Fair (50-69)**: Synthetic data has acceptable similarity
- **Poor (<50)**: Synthetic data differs significantly

### Distribution Similarity
- Numeric Similarity: How close are numeric columns?
- Categorical Similarity: How close are category distributions?
- Values: 0.0 (completely different) to 1.0 (identical)

### Model Accuracy Levels
- ✓ **>85%**: Excellent performance
- ⚠️ **70-85%**: Good performance
- ❌ **<70%**: Poor performance - investigate

## 💾 Download Results

In the **Artifacts** section, download:
1. **Synthetic dataset** (CSV) - The generated data
2. **Synthetic inference** (JSON) - Model predictions and analysis

## 🔍 Common Recommendations

### Good Signs ✓
- Quality score > 80
- All health indicators showing ✓
- Model accuracy > 85%
- Balanced class distribution (for classification)

### Warning Signs ⚠️
- Quality score 50-70: Data differences exist
- Class imbalance: Some classes over 80%
- Accuracy 70-85%: Room for improvement

### Critical Issues ❌
- Quality score < 50: Major data differences
- Accuracy < 70%: Model needs retraining
- Missing values mismatch: Check data quality

## 🎯 Example Workflow

```
1. Upload "Smart Manufacturing Maintenance Dataset"
   └─> See dataset preview with 500 rows, 10 columns

2. Configure
   └─> Problem Type: classification
   └─> Target: equipment_failure (Yes/No)
   └─> Synthetic Rows: 300

3. Enable "Generate synthetic data" and run

4. Wait for completion (~30-60 seconds)

5. Review Results
   └─> Quality Score: 82 (Good) ✓
   └─> Similarity: 0.78 (numeric), 0.85 (categorical)
   └─> Model Accuracy: 92% ✓
   └─> All health checks passing ✓

6. Download synthetic CSV and analysis JSON
```

## 📱 Dashboard Layout

```
┌─────────────────────────────────────────────────────┐
│  Dataset Preview                                    │
│  [Dataset loaded: 500 rows, 10 columns]            │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Pipeline Stages                                    │
│  ┌───────┬────────┬──────────┬──────────┐          │
│  │Resolve│ Load   │Preprocess│ Analyze  │  ...     │
│  └───────┴────────┴──────────┴──────────┘          │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │ 🔵 Data Quality Analysis                     │  │
│  │ Quality: 82 | Health: All ✓                  │  │
│  │ [Similarity metrics and differences...]      │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │ 🔵 Prediction Analysis (CLASSIFICATION)      │  │
│  │ Accuracy: 92% | Precision: 91%              │  │
│  │ [Performance metrics and recommendations...]  │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  Artifacts (Download)                              │
│  📄 Synthetic dataset (CSV)                        │
│  📄 Synthetic inference (JSON)                     │
└─────────────────────────────────────────────────────┘
```

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8001 busy | Use `APP_PORT=8002 python scripts/run_local_app.py` |
| Module not found | Ensure virtualenv: `source mas_venv/bin/activate` |
| No synthetic data | Check "Generate synthetic data" checkbox |
| Slow generation | Reduce "Synthetic rows" or check system resources |

## 📚 Learn More
- See `SYNTHETIC_DATA_ENHANCEMENT.md` for technical details
- Check `webapp/static/app.js` for frontend code
- Review `utils/synthetic_quality_analyzer.py` for metrics

---

**Ready to generate synthetic data? Visit http://127.0.0.1:8001 now!**
