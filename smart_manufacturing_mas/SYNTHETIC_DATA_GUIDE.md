# Synthetic Data Generation & Closed-Loop Simulation Guide

## 1. Purpose & Overview

The **Synthetic Data Generation & Evaluation** subsystem provides high-fidelity artificial industrial telemetry for:
1. **Unbiased Generalization Testing**: Evaluating pre-trained model bundles on completely unseen data distributions.
2. **Closed-Loop Cyber-Physical Simulation**: Simulating degradation trajectories, multi-sensor anomalies, and post-intervention recovery curves to validate the **Autonomous Execution State Machine** and **Learned Recommender** before physical deployment.

---

## 2. Generating Synthetic Telemetry

### Method A: Via Web GUI (Fastest)

1. Launch the web application:
   ```bash
   python scripts/run_local_app.py
   ```
2. Open **`http://127.0.0.1:8000`** in your browser.
3. In the **Synthetic Data Generator** card:
   - Select the source dataset archetype (e.g. `smart_maintenance_dataset.csv`).
   - Specify the desired row count (e.g., `500`, `2000`, `10000`).
   - Set an optional random seed.
   - Click **"Generate Synthetic Dataset"**.
4. The dataset is immediately saved to `artifacts/web_synthetic/` and becomes available in the dataset selector for instant inference runs.

### Method B: Via CLI Script

```bash
# Generate 500 synthetic rows using default seed (42)
python scripts/generate_synthetic_data_and_infer.py --n-rows 500

# Generate 2,000 rows with a custom seed and custom output directory
python scripts/generate_synthetic_data_and_infer.py --n-rows 2000 --seed 123 --output-dir artifacts/custom_synthetic
```

### Method C: Via REST API

```bash
curl -X POST http://127.0.0.1:8000/api/generate-synthetic \
  -H "Content-Type: application/json" \
  -d '{"source_dataset": "data/smart_manufacturing_dataset.csv", "n_rows": 1000, "seed": 42}'
```

---

## 3. How the Synthetic Generator Works

```
Source Dataset Statistics (μ, σ, min, max, quantiles, proportions)
                           ↓
Feature Distribution Modeling:
  • Numerical: Sampled from Gaussian / Uniform distributions bounded by [min, max]
  • Categorical: Sampled proportionally according to empirical class frequencies
  • Identifiers: Preserves realistic Machine_ID and Agent_ID patterns
                           ↓
Exported Synthetic Dataset (.csv)
                           ↓
Automated Pretrained Pipeline Inference
                           ↓
Metrics & Generalization Audit (.json)
```

---

## 4. Analyzing Generalization Quality

To inspect synthetic dataset distributions and verify that models generalize without overfitting:

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook training/synthetic_data_inference_analysis.ipynb
   ```
2. The notebook executes:
   - Statistical distribution comparison (Wasserstein distance, mean differences).
   - In-sample vs. out-of-sample prediction divergence.
   - Classification class balance fidelity checks.
   - Visualization of feature correlations and degradation trends.

---

## 5. Synthetic Telemetry in Closed-Loop Simulation (Sem 7 Pivot)

For the **Autonomous Closed-Loop Execution State Machine**, synthetic telemetry simulates physical machine dynamics over an observation window $t \dots t+H$:

```
Initial Machine State (s_t) ──► Action Dispatched (a_t) ──► Degradation / Recovery Function
                                                                      │
                                                                      ▼
Post-Intervention State (s_{t+H}) ◄── Simulated Sensor Dynamics (Δ Recovery)
```

- **Degradation Simulation**: Injects progressive wear signals (e.g. vibration drift $+0.15\,\text{mm/s}$ per 100 hours, temperature spikes).
- **Intervention Recovery**: Simulates physics-informed recovery dynamics (e.g. executing "Recalibrate PID & Lubricate" reduces vibration back to nominal within 4 hours).
- **Feedback Logging**: Feeds simulated $(s_t, a_t, r_t, s_{t+1})$ transitions to the feedback store to validate continuous learning algorithms.
