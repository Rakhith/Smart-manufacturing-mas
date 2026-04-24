# 🧪 Quick Testing Guide - Synthetic Data Enhancement

## Application Running Successfully ✅

```
Uvicorn running on http://127.0.0.1:8000
```

---

## Step-by-Step Testing

### Step 1: Open the Application
```
🔗 URL: http://127.0.0.1:8000
```

### Step 2: Select or Upload Dataset
- Choose an existing dataset from dropdown OR
- Upload a CSV file with:
  - At least 10 rows
  - Mix of numeric and categorical columns
  - A clear target column for classification/regression

**Recommended Test Files**:
```
smart_manufacturing_mas/data/Smart Manufacturing Maintenance Dataset/
smart_manufacturing_mas/data/metal_etch_data.csv
```

### Step 3: Configure the Run

Fill in the form fields:
```
├─ Dataset: [Selected]
├─ Target Column: [Column Name] ← Important!
├─ Problem Type: [classification/regression]
├─ Feature Columns: [Optional - leave empty for auto]
├─ Model: [pretrained]
├─ Cache: [checked]
└─ ✅ Generate Synthetic Data ← CHECK THIS!
   └─ Synthetic Rows: 300 (range: 10-10000)
```

### Step 4: Launch Run

Click **"Launch Run"** button

### Step 5: Monitor Progress

Watch the "Pipeline Stages" section:
```
1. Resolve    ◉ → Running → ✓ Done
2. Load       ◉ → Running → ✓ Done
3. Preprocess ◉ → Running → ✓ Done
4. Analyze    ◉ → Running → ✓ Done
5. Recommend  ◉ → Running → ✓ Done
6. Summarize  ◉ → Running → ✓ Done
7. Synthetic  ◉ → Running → ✓ Done ← NEW!
```

### Step 6: View Results

Scroll down to see new sections:

#### **Data Quality Analysis Card**
```
┌─────────────────────────────────────┐
│ Synthetic Data Quality Analysis     │
├─────────────────────────────────────┤
│ Quality Score: XX.X (Quality Level) │
│                                     │
│ Recommendation: [Message]           │
├─────────────────────────────────────┤
│ Similarity Metrics:                 │
│ • Numeric Similarity: 0.XX          │
│ • Categorical Similarity: 0.XX      │
├─────────────────────────────────────┤
│ Health Status:                      │
│ ✓/⚠ Missing Values: [Status]       │
│ ✓/⚠ Outlier Patterns: [Status]     │
│ ✓/⚠ Numeric Distribution: [Status] │
│ ✓/⚠ Categorical Distribution: [Sta]│
├─────────────────────────────────────┤
│ Dataset Info:                       │
│ • Original Rows: XXX                │
│ • Synthetic Rows: XXX               │
├─────────────────────────────────────┤
│ Top Column Differences:             │
│ • column_name: X.X% difference      │
│ • column_name: X.X% difference      │
│ • column_name: X.X% difference      │
└─────────────────────────────────────┘
```

#### **Prediction Analysis Card**
```
┌─────────────────────────────────────┐
│ Prediction Analysis                 │
│ Model Performance on Synthetic Data │
├─────────────────────────────────────┤
│ [CLASSIFICATION/REGRESSION]         │
├─────────────────────────────────────┤
│ Performance Metrics:                │
│ • Total Predictions: XXX            │
│ • Accuracy/R²: X.XXX                │
│ • Precision/RMSE: X.XXX             │
│ • Recall/MAE: X.XXX                 │
│ • F1 Score: X.XXX                   │
├─────────────────────────────────────┤
│ [Class Distribution / Regression]   │
│ • Class A: XXX (X%)                 │
│ • Class B: XXX (X%)                 │
├─────────────────────────────────────┤
│ Recommendations:                    │
│ ✅ [Green] Good Performance         │
│ ⚠️  [Orange] Check Distribution    │
│ ❌ [Red] Needs Attention            │
│ ℹ️  [Blue] Information              │
└─────────────────────────────────────┘
```

---

## 📊 Expected Quality Scores

### Classification Model Example
```
If Original Dataset:
  - 100 rows of maintenance logs
  - Binary target: Failed/OK

Synthetic Dataset (300 rows):
  - Quality Score: 75-85 (Good)
  - Numeric Similarity: 0.80-0.90
  - Categorical Similarity: 0.70-0.80
  
Recommendations:
  ✅ Good model performance (92% accuracy)
  ✓ Reasonable class balance (58/42 split)
```

### Regression Model Example
```
If Original Dataset:
  - 150 rows of sensor data
  - Target: Temperature prediction

Synthetic Dataset (500 rows):
  - Quality Score: 70-80 (Good)
  - Numeric Similarity: 0.85-0.92
  - Prediction R²: 0.88
  
Recommendations:
  ✅ Good R² score (explains 88% variance)
  ✓ RMSE acceptable (within 3% of data range)
```

---

## ⚠️ What to Look For

### ✅ Good Signs
```
✓ Quality score 70+
✓ Numeric/Categorical similarity > 0.7
✓ Health indicators mostly green (✓)
✓ Accuracy/R² > 0.85
✓ Minimal top column differences (< 10%)
```

### ⚠️ Warning Signs
```
⚠ Quality score 50-70
⚠ Similarity scores 0.5-0.7
⚠ Some health indicators orange (⚠)
⚠ Accuracy 0.7-0.85
⚠ Large column differences (> 20%)
```

### ❌ Issue Signs
```
❌ Quality score < 50
❌ Similarity scores < 0.5
❌ Health indicators red (❌)
❌ Accuracy < 0.7
❌ Extreme column differences (> 30%)
```

---

## 🔍 Debug Information

### Check Browser Console
```javascript
// Open DevTools (F12)
// Go to Console tab
// Should see no errors, only information logs
```

### Check Terminal Output
```bash
# Watch for error messages like:
- "Data quality analysis warning:"
- "Prediction analysis warning:"

# These are non-blocking warnings - app continues
```

### Check Artifacts Created
```bash
ls -la artifacts/web_synthetic/

# Should contain:
# - {run_id}_synthetic.csv (the generated data)
# - {run_id}_synthetic_inference.json (predictions + analysis)
```

---

## 🧪 Test Scenarios

### Test 1: Small Dataset (Classification)
```
Dataset: 50 rows, 5 features
Target: Binary classification
Synthetic: 100 rows
Expected: Fast execution, clear quality metrics
```

### Test 2: Medium Dataset (Regression)
```
Dataset: 200 rows, 10 features
Target: Continuous value prediction
Synthetic: 300 rows
Expected: More comprehensive analysis
```

### Test 3: Large Dataset (Manufacturing Data)
```
Dataset: 1000+ rows, 20+ features
Target: Complex manufacturing metric
Synthetic: 500+ rows
Expected: Rich quality analysis, detailed recommendations
```

---

## 📝 Troubleshooting

### Issue: "Synthetic dashboard not showing"
```
Solution: 
1. Check if "Generate Synthetic Data" was checked
2. Wait for run to complete (check Synthetic stage)
3. Refresh browser if needed
4. Check browser console for errors
```

### Issue: "Quality score is very low"
```
Solution:
1. Check original dataset size (needs 50+ rows)
2. Check feature variety (needs diverse data)
3. Try with different random seed (in code)
4. Verify target column selection
```

### Issue: "No recommendations showing"
```
Solution:
1. Ensure problem type is set correctly
2. Check if model predictions were made
3. Verify no errors in prediction stage
4. Check terminal for warnings
```

---

## 🎯 Success Criteria

✅ **Feature is working correctly when:**
- Data Quality Analysis card appears
- Quality score is calculated (0-100)
- Similarity metrics are populated
- Health indicators show status
- Prediction Analysis card appears
- Recommendations are displayed
- No errors in browser console
- No blocking errors in terminal

---

**Ready to test! Open http://127.0.0.1:8000 in your browser.**
