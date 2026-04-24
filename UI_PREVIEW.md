# UI/UX Preview: Synthetic Data Dashboard

## 🎨 How It Looks in the Web Interface

### Pipeline View
```
┌─────────────────────────────────────────────────────────────────┐
│ Run Status: COMPLETED                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage Progress:                                                │
│  ✓ Resolve  ✓ Load  ✓ Preprocess  ✓ Analyze  ✓ Recommend       │
│  ✓ Summary  ✓ Synthetic                                         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│ STAGE CARDS (Scrollable)                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ┌─ Resolve ────────────────────────────────────────────────┐   │
│ │ [Details...]                                             │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│ ┌─ Load ───────────────────────────────────────────────────┐   │
│ │ [Details...]                                             │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│ ... [Other stages] ...                                          │
│                                                                 │
│ ┌─ Generate Synthetic Data ────────────────────────────────┐   │
│ │                                                          │   │
│ │  Status: ✓ COMPLETED                                    │   │
│ │                                                          │   │
│ ├──────────────────────────────────────────────────────────┤   │
│ │ DATA QUALITY ANALYSIS                                    │   │
│ ├──────────────────────────────────────────────────────────┤   │
│ │                                                          │   │
│ │  ┌─────────────────────────────────────────────────┐    │   │
│ │  │ 82.5                                            │    │   │
│ │  │  GOOD                                           │    │   │
│ │  │  Quality Score                                  │    │   │
│ │  └─────────────────────────────────────────────────┘    │   │
│ │                                                          │   │
│ │  "Synthetic data captures main characteristics of       │   │
│ │   original distribution. Good for training models."     │   │
│ │                                                          │   │
│ │  ┌────────────┬────────────┬────────────┐               │   │
│ │  │ Similarity │   Health   │  Dataset   │               │   │
│ │  │  Metrics   │  Status    │    Info    │               │   │
│ │  ├────────────┼────────────┼────────────┤               │   │
│ │  │ Numeric:   │ Missing:   │ Original:  │               │   │
│ │  │ 0.78       │ ✓ Healthy  │ 500 rows   │               │   │
│ │  │            │            │            │               │   │
│ │  │ Categoric: │ Outliers:  │ Synthetic: │               │   │
│ │  │ 0.85       │ ✓ Healthy  │ 300 rows   │               │   │
│ │  └────────────┴────────────┴────────────┘               │   │
│ │                                                          │   │
│ │  Top Column Differences:                                │   │
│ │  • temperature: 12.5% difference                        │   │
│ │  • pressure:    8.3% difference                         │   │
│ │  • flow_rate:   5.2% difference                         │   │
│ │                                                          │   │
│ ├──────────────────────────────────────────────────────────┤   │
│ │ PREDICTION ANALYSIS                                      │   │
│ ├──────────────────────────────────────────────────────────┤   │
│ │                                                          │   │
│ │  [CLASSIFICATION]                                       │   │
│ │                                                          │   │
│ │  ┌────────────┬────────────┬────────────┐               │   │
│ │  │ Performance│  Classes   │   Metrics  │               │   │
│ │  │  Metrics   │Distribution│            │               │   │
│ │  ├────────────┼────────────┼────────────┤               │   │
│ │  │ Accuracy:  │ Class 0:   │ Total:     │               │   │
│ │  │ 92%        │ 150 (50%)  │ 300 preds  │               │   │
│ │  │            │            │            │               │   │
│ │  │ Precision: │ Class 1:   │ Unique:    │               │   │
│ │  │ 89%        │ 75 (25%)   │ 3 classes  │               │   │
│ │  │            │            │            │               │   │
│ │  │ Recall:    │ Class 2:   │            │               │   │
│ │  │ 91%        │ 75 (25%)   │            │               │   │
│ │  │            │            │            │               │   │
│ │  │ F1 Score:  │            │            │               │   │
│ │  │ 0.90       │            │            │               │   │
│ │  └────────────┴────────────┴────────────┘               │   │
│ │                                                          │   │
│ │  RECOMMENDATIONS                                         │   │
│ │                                                          │   │
│ │  ✓ Good Performance (SUCCESS)                           │   │
│ │    Model achieves 92% accuracy on synthetic data.       │   │
│ │    Good model generalization.                           │   │
│ │                                                          │   │
│ │  ℹ️  Sample Size (INFO)                                 │   │
│ │    300 predictions available. Results are reliable.    │   │
│ │                                                          │   │
│ └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

ARTIFACTS SECTION
┌─────────────────────────────────────────────────────────────────┐
│ Downloadable Files:                                             │
│ • Synthetic dataset                              [CSV]          │
│ • Synthetic inference summary                    [JSON]         │
│ • Recommendation preview                         [See above]    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎨 Color Scheme

### Quality Score Badge Colors
```
Score 85-100:   🟢 GREEN    (#4caf50)  → Excellent
Score 70-84:    🟢 LIGHT    (#8bc34a)  → Good
Score 50-69:    🟠 ORANGE   (#ff9800)  → Fair
Score <50:      🔴 RED      (#f44336)  → Poor
```

### Recommendation Severity Colors
```
SUCCESS  🟢 Green   (#e8f5e9 bg, #4caf50 border)
         Accuracy ≥ 85%, Good performance

WARNING  🟠 Orange  (#fff3e0 bg, #ff9800 border)
         Moderate issues, class imbalance, etc.

ERROR    🔴 Red     (#ffebee bg, #f44336 border)
         Accuracy < 70%, critical issues

INFO     🔵 Blue    (#e3f2fd bg, #2196F3 border)
         General insights, sample size info
```

### Health Indicator Status
```
✓ Healthy       → Green text
✓ Good          → Green text
⚠️ Attention needed → Orange text
⚠️ Check differences → Orange text
```

---

## 📱 Responsive Layout

### Desktop (Full Width)
```
┌─────────────────────────────────────────────────────────┐
│ QUALITY SCORE BADGE    │ SIMILARITY METRICS            │
│ (Left side)            │ (Right side)                  │
├─────────────────────────────────────────────────────────┤
│ HEALTH STATUS          │ DATASET INFO                  │
│ (2 columns)                                             │
├─────────────────────────────────────────────────────────┤
│ TOP COLUMN DIFFERENCES (Full width)                     │
├─────────────────────────────────────────────────────────┤
│ PREDICTION ANALYSIS (2-3 columns depending on type)     │
├─────────────────────────────────────────────────────────┤
│ RECOMMENDATIONS (Full width, stacked cards)             │
└─────────────────────────────────────────────────────────┘
```

### Mobile (Stacked)
```
┌──────────────────────┐
│ QUALITY SCORE BADGE  │
├──────────────────────┤
│ SIMILARITY METRICS   │
├──────────────────────┤
│ HEALTH STATUS        │
├──────────────────────┤
│ DATASET INFO         │
├──────────────────────┤
│ TOP DIFFERENCES      │
├──────────────────────┤
│ PREDICTION ANALYSIS  │
├──────────────────────┤
│ RECOMMENDATIONS      │
└──────────────────────┘
```

---

## 🎯 Interactive Elements

### Quality Score Badge
- **Hover**: Shows full recommendation text
- **Click**: Opens detailed analysis view (future feature)
- **Animation**: Smooth number count-up on first load

### Health Indicators
- **Hover**: Shows explanation of each indicator
- **Color**: Changes based on health status
- **Icon**: Shows visual indicator (✓ or ⚠️)

### Recommendations List
- **Hover**: Highlights recommendation card
- **Click**: Expands to show more details (future feature)
- **Color**: Immediate visual indication of severity

### Metric Cards
- **Hover**: Shows trend indicator (up/down arrows)
- **Click**: Links to detailed analysis (future feature)

---

## 📊 Data Displayed in Each Section

### Data Quality Card

**Input**:
```json
{
  "input_rows": 500,
  "requested_rows": 300,
  "target_column": "equipment_failure"
}
```

**Output**:
```json
{
  "quality_score": 82.5,
  "quality_level": "Good",
  "synthetic_rows": 300,
  "quality_metrics": {...}
}
```

### Prediction Card

**For Classification**:
```
Total Predictions: 300
Unique Classes: 3
Most Common: Class_0 (50%)
Least Common: Class_1 (25%)

Metrics:
- Accuracy: 92%
- Precision: 89%
- Recall: 91%
- F1-Score: 0.90
```

**For Regression**:
```
Total Predictions: 300
Mean: 102.5
Median: 100.3
Std Dev: 15.2

Metrics:
- RMSE: 5.2
- MAE: 4.1
- R² Score: 0.87
```

---

## 🔄 Data Flow Display

```
User Input
    ↓
    ├─ Dataset
    ├─ Target Column
    └─ Rows to Generate (300)
    
Pipeline Execution
    ↓
    ├─ Generate Synthetic Data ✓
    ├─ Quality Analysis ✓
    ├─ Make Predictions ✓
    └─ Analyze Results ✓

Results Display
    ↓
    ├─ Quality Score Card
    │   ├─ Score Badge
    │   ├─ Recommendation
    │   └─ Metrics
    │
    ├─ Data Comparison
    │   ├─ Similarity Scores
    │   ├─ Health Indicators
    │   └─ Top Differences
    │
    └─ Prediction Analysis
        ├─ Performance Metrics
        ├─ Class/Value Distribution
        └─ Recommendations

Download Options
    ↓
    ├─ Synthetic Data (CSV)
    └─ Analysis Results (JSON)
```

---

## 📝 Example Recommendation Displays

### Success Example
```
┌─────────────────────────────────────────┐
│ ✓ Good Performance (SUCCESS)            │
│                                         │
│ Model achieves 92% accuracy on          │
│ synthetic data. Good model              │
│ generalization.                         │
└─────────────────────────────────────────┘
```

### Warning Example
```
┌─────────────────────────────────────────┐
│ ⚠️  Class Imbalance (WARNING)            │
│                                         │
│ Highly skewed predictions: Class_A      │
│ represents 85% of predictions.          │
│ Consider class balancing strategies.    │
└─────────────────────────────────────────┘
```

### Error Example
```
┌─────────────────────────────────────────┐
│ ✗ Low Accuracy (ERROR)                  │
│                                         │
│ Model accuracy on synthetic data is     │
│ 62%. Consider retraining or data        │
│ validation.                             │
└─────────────────────────────────────────┘
```

### Info Example
```
┌─────────────────────────────────────────┐
│ ℹ️  Sample Size (INFO)                  │
│                                         │
│ Only 300 predictions available.         │
│ Results may be more reliable with       │
│ larger datasets.                        │
└─────────────────────────────────────────┘
```

---

## 🎨 CSS Classes Used

```css
.quality-badge {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  /* Color changes based on score */
}

.stage-card {
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.subcard {
  background: #f9f9f9;
  border: 1px solid #eee;
  border-radius: 4px;
  padding: 12px;
  margin: 8px;
}

.recommendation {
  padding: 12px;
  margin: 8px 0;
  border-left: 4px solid;
  border-radius: 4px;
  /* Color changes based on severity */
}

.status-pill {
  display: inline-block;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: bold;
  /* Color changes based on status */
}
```

---

## 🎬 Animation Flows

### On Load
1. Stage card appears with fade-in
2. Quality badge numbers count up (0 → score)
3. Health indicators slide in
4. Recommendation cards fade in one by one

### On Update
1. If status changed, update badge color
2. If metrics changed, show small animation
3. New recommendations fade in

### On Hover
1. Card shadow increases
2. Recommendation card highlights
3. Tooltip appears on metrics

---

## 📱 Mobile Optimizations

- Single column layout
- Larger touch targets (48x48px minimum)
- Simplified tables (show 3-4 most important metrics)
- Stack all cards vertically
- Fullscreen badge for better visibility
- Simplified recommendation display

---

## ♿ Accessibility Features

- Semantic HTML structure
- Proper heading hierarchy (h3, h4, h5)
- Color + text indicators (not color alone)
- Sufficient contrast ratios
- ARIA labels for icons
- Keyboard navigation support
- Screen reader friendly

---

**This preview shows how users will experience your synthetic data quality analysis feature! 🎨✨**
