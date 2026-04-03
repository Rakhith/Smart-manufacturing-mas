# Adaptive Intelligence System

## Overview

The Adaptive Intelligence module is triggered when initial model performance falls below a defined threshold. It systematically explores alternative model configurations and selects the best-performing option.

## Performance Thresholds

| Task | Metric | Threshold |
|------|--------|-----------|
| Classification | Accuracy | < 0.60 |
| Regression | R² | < 0.10 |
| Anomaly Detection | N/A | (no threshold — IsolationForest always used) |

## Trigger Flow

```
Initial model run (ToolDecider selection)
          ↓
    Performance check
          ↓ (below threshold)
    HITL gate: "retry with alternative models?"
          ↓ (user approves)
    _try_multiple_models()
       tries all model families sequentially
       records all outcomes
       selects best performance
          ↓
    Final results returned
```

## Model Families Tried

**Classification:**
1. RandomForestClassifier (`n_estimators = min(100, max(10, n_samples // 10))`)
2. LogisticRegression (`max_iter=1000`)
3. SVC (`kernel='rbf', C=1.0`)

**Regression:**
1. LinearRegression (default)
2. Ridge (`alpha=1.0`)
3. Lasso (`alpha=1.0`)
4. RandomForestRegressor (`n_estimators=100`)
5. SVR (`kernel='rbf', C=1.0`)

## Logging

All model runs, including failed attempts, are logged to ensure:
- Transparency: every decision is auditable
- Reproducibility: full trial history recorded
- Cumulative learning: failed patterns stored in workflow context

## Example Log Output

```
🧠 ADAPTIVE INTELLIGENCE: Trying multiple models for better performance...
🔄  Trying LinearRegression...
   LinearRegression performance: 0.0273
🔄  Trying Ridge...
   Ridge performance: 0.0272
🔄  Trying Lasso...
   Lasso performance: -0.0001
🔄  Trying RandomForestRegressor...
   RandomForestRegressor performance: -0.0374
🔄  Trying SVR...
   SVR performance: 0.0322
🏆 Best model: SVR (performance: 0.0322)
```

Note: Low R² values indicate that the target variable may have weak linear separability from the selected features, which is expected for some manufacturing datasets. The framework correctly identifies and reports this rather than silently proceeding.

## Model Cache Interaction

When `--use-cache` is enabled:
- If a cache HIT is found, Adaptive Intelligence is **skipped** (cached model is assumed to be already the best).
- Results include `"from_cache": True` flag so downstream logic knows not to re-trigger the performance gate.
- After Adaptive Intelligence completes, the winning model is saved to cache so the next run loads it instantly.
