# Adaptive Intelligence & Continuous Learning System

## 1. Overview

The **Adaptive Intelligence** subsystem ensures robustness against unpredictable industrial data distributions. When initial model performance falls below defined quality thresholds, the system automatically sweeps across all supported model families to find the optimal estimator architecture.

In the **Semester 7 Closed-Loop Architecture**, this concept extends beyond initial model selection to **Continuous Policy Adaptation**, where action rankings dynamically update based on observed post-intervention telemetry recovery and economic cost feedback.

---

## 2. Performance Thresholds & Trigger Logic

| Problem Type | Primary Metric | Trigger Threshold | Model Families Evaluated |
| :--- | :--- | :--- | :--- |
| **Classification** | Balanced Accuracy | $< 0.60$ | RandomForestClassifier, LogisticRegression, Support Vector Classifier (SVC) |
| **Regression** | $R^2$ Score | $< 0.10$ | LinearRegression, Ridge, Lasso, HistGradientBoosting, RandomForestRegressor, SVR |
| **Anomaly Detection** | N/A | Heuristic | IsolationForest (hyperparameter optimization via Local SLM) |

### Adaptive Intelligence Execution Flow

```
1. Initial Model Selected via ToolDecider (Tier 3) or Pretrained Bundle
                      ↓
2. Evaluation on Validation Set / In-sample Diagnostics
                      ↓
3. Performance Gate Check
   [Pass] ──► Proceed to Prescriptive Ranking
   [Fail] ──► Trigger Adaptive Intelligence Sweep:
              a. Sequentially train all alternative candidate model families
              b. Evaluate metrics ($R^2$, Accuracy, MSE)
              c. Select highest-performing model
              d. Update ModelCache with winner
                      ↓
4. Downstream Prescriptive Recommendation & Ranking
```

---

## 3. Pretrained Bundle & Model Cache Interaction

1. **Pretrained Inference Mode (`--inference-only`)**:
   - Checks `artifacts/pretrained_models/registry.json` for compatible bundles.
   - If compatible, instant inference runs with zero training latency.
   - If no compatible bundle exists, the system gracefully triggers a live training fallback with automatic caching.
2. **Model Caching (`--use-cache`)**:
   - Checks `model_cache/` for matching hash keys.
   - On cache **HIT**, Adaptive Intelligence is bypassed since the optimal model is already persisted.
   - On cache **MISS**, fresh training or adaptive sweeps run, and the winning model is automatically saved for future runs.

---

## 4. Continuous Closed-Loop Policy Adaptation (Semester 7 Pivot)

In addition to static batch training, the closed-loop system continually adapts the **Learned Recommender** (`LGBMRanker`) and **Collaborative Filtering** similarity matrices using field feedback:

```
[Telemetry Ingestion] ──► [Ranked Action Dispatch] ──► [Post-Intervention Tracking]
        ▲                                                          │
        │                                                          ▼
[Model Weight Updates] ◄── [Replay Buffer] ◄── [Cost & Recovery Feedback Store]
```

1. **Transition Storage**: Every autonomous execution records $(s_t, a_t, r_t, s_{t+1})$ where:
   - $s_t$: Machine telemetry and operational hours before action.
   - $a_t$: Dispatched maintenance action.
   - $r_t$: Economic reward ($\text{Avoided Downtime Cost} - \text{Intervention Cost}$).
   - $s_{t+1}$: Post-intervention telemetry state after horizon $H$.
2. **Continuous Learning**: Feedback tuples update the ranking utility weights and peer-similarity matrices, allowing the system to learn true non-linear recovery dynamics over time.

---

## 5. Evaluation Metrics Framework

| Category | Metric | Formula / Description | Target Goal |
| :--- | :--- | :--- | :--- |
| **Predictive ML** | Balanced Accuracy | $\frac{1}{2} (\text{TPR} + \text{TNR})$ | $> 0.85$ |
| | $R^2$ Score | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | $> 0.70$ |
| **Ranking Quality** | $\text{NDCG@k}$ | $\frac{\text{DCG@k}}{\text{IDCG@k}}$ where $\text{DCG@k} = \sum_{i=1}^k \frac{2^{rel_i} - 1}{\log_2(i + 1)}$ | $> 0.90$ |
| | $\text{Precision@k}$ | Proportion of top-$k$ recommended actions that successfully resolved degradation | $> 0.80$ |
| **Industrial Economics**| Net Cost Saved | $\text{Avoided Unplanned Downtime Cost} - \text{Action Cost}$ | Maximize (\$) |
| **Physical Recovery** | Delta Recovery ($\Delta R$) | $\|\mathbf{x}_{t} - \mathbf{x}_{\text{nominal}}\| - \|\mathbf{x}_{t+H} - \mathbf{x}_{\text{nominal}}\|$ | Positive (Nominal Restored) |
