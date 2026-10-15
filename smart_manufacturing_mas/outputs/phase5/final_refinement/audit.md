# Phase 5 Final Refinement: Comprehensive Codebase & Scientific Audit

**Document**: `outputs/phase5/final_refinement/audit.md`  
**Date**: September 2026  
**Status**: Completed Prior to Final Refinement Experiments  
**Objective**: Rigorous audit of Phase 4 and Phase 5 codebase, artifacts, adapters, ontology, leakage guards, closed-loop simulation, and temporal splitting prior to executing the final experimental refinement pass for academic manuscript preparation.

---

## 1. Executive Context & The Core Scientific Problem

### Two Distinct Ground-Truth Targets
The foundational scientific distinction governing this project is:
- **Target A (Phase 4 Target)**: Offline LLM-generated silver maintenance preferences derived from prompt-based domain heuristics.
- **Target B (Phase 5 Target)**: Historical maintenance actions recorded by human plant technicians and recorded progressive failure events on external industrial equipment.

These two targets are **not identical**:
1. Human technician actions represent *observational plant practice* (frequently calendar-based or constrained by shift staffing), not provably optimal physical ground truth.
2. LLM silver preferences represent *heuristic condition-based alignment*, which heavily penalizes premature capital expenditure when assets are in non-critical states.
3. Therefore, Phase 5 final refinement must explicitly separate:
   - Preference-policy alignment (Phase 4)
   - Historical-action agreement (Zero-shot Phase 5)
   - Semantic action relevance (Secondary LLM judge)
   - Component & failure identification (Component-first & failure-oriented metrics)
   - Action-grounded adaptation (Chronological transfer learning)
   - Simulated closed-loop consequences (Counterfactual dynamics with sensitivity bounds)

---

## 2. Phase 4 Artifacts & Frozen Model State

### Cryptographic Immutability & Model Integrity
- **Artifact Path**: `outputs/phase4/models/lgbm_ranker.joblib`
- **Model Architecture**: LightGBM LambdaRank (`n_estimators=50, learning_rate=0.03, num_leaves=15, objective=lambdarank`) fused with Case-Based Reasoning (CBR, $k=5$).
- **Locked Weights SHA-256**: `bc4b48094f51e6e5...` (verified immutable by `FrozenPhase4Recommender.verify_frozen_invariants()`).
- **Feature Space**: 95 canonical features (32 continuous telemetry statistics, 43 categorical one-hot embeddings, 20 candidate action descriptors).
- **Frozen Hyperparameters**: Fusion weight $\alpha = 0.75$, CBR nearest neighbors $k = 5$.
- **CBR Case Base**: Indexed strictly on Phase 4 training states (`train_state_ids` in `outputs/phase4/splits/split_manifest.json`) with zero external state contamination.

---

## 3. External Dataset Adapters & Physical Topology

| Dataset | Path | Assets | Telemetry Volume | Temporal Resolution | Ground Truth Events |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Factory Sensors** | `data/external/factory_sensors/` | 6 Centrifugal Slurry Pumps | 52,560 hourly rows (1 full year) | 1 hour continuous | 63 Work Orders, 39 Failure Events (Bearing wear, Seal leak, Overheating, Imbalance, Electrical) |
| **OIL-040 Benchmark** | `data/external/oil040/` | 80 Heavy Rotating Assets | 9,600 synchronized rows (1 month) | 6 hours periodic | 80 Maintenance Work Orders, measured downtime hours |
| **Industrial Synthetic** | `data/external/industrial_synthetic/` | General Manufacturing | 1,000,000 SAP/Maximo work orders | Transactional | 1M CMMS text descriptions, parts costs, labor hours |

### Adapter Implementations:
- `phase5/adapters/factory_sensors_adapter.py`: Ingests hourly telemetry across 6 pumps spanning 2025-09-01 to 2026-08-31. Links work orders to nearest failure events (+/- 48h) and maps to canonical actions. DecisionState builder uses 12h history window strictly prior to timestamp $t$.
- `phase5/adapters/oil040_adapter.py`: Ingests 12 relational CSV tables across 80 assets. Converts temperature to Celsius, vibration to mm/s, pressure to bar. DecisionState builder enforces quarantine gating.
- `phase5/adapters/industrial_synthetic_adapter.py`: NLP ingestion of SAP/Maximo maintenance text.

---

## 4. Action Ontology & Current Mapping

### Canonical 20-Action Hierarchy
Structured into 5 operational categories:
1. **Monitoring & Observation (3 actions)**: `ACT_MON_CONTINUE`, `ACT_MON_ENHANCED`, `ACT_MON_PARAMETER_LOG`
2. **Inspection & Diagnosis (6 actions)**: `ACT_INSP_VIBRATION`, `ACT_INSP_LUBRICATION`, `ACT_INSP_THERMAL_ELECTRICAL`, `ACT_INSP_TOOL_WEAR`, `ACT_INSP_PRESSURE_HYDRAULIC`, `ACT_INSP_SUBSYSTEM`
3. **Corrective Maintenance (4 actions)**: `ACT_CORR_LUBRICATE`, `ACT_CORR_CLEAN_PURGE`, `ACT_CORR_CALIBRATE`, `ACT_CORR_FASTEN_ALIGN`
4. **Component Replacement & Overhaul (4 actions)**: `ACT_REPL_TOOL_INSERT`, `ACT_REPL_BEARING`, `ACT_REPL_SEAL_VALVE`, `ACT_REPL_OVERHAUL`
5. **Operational Mitigation (4 actions)**: `ACT_OP_ADJUST_PARAMS`, `ACT_OP_DERATE_LOAD`, `ACT_OP_CONTROLLED_SHUTDOWN`, `ACT_OP_EMERGENCY_STOP`

### Existing Mapping Logic (`phase5/action_mapper.py`):
- Deterministic mappings with confidence scores and rationale for OIL-040 and Factory Sensors.
- Current coverage: 100% of evaluated work orders were assigned a primary canonical action.
- **Identified Gap for Final Refinement**: The current mapping assigns a single target without explicitly auditing unmapped/ambiguous codes, secondary actions in multi-procedure work orders, or tracking an `OUT_OF_ONTOLOGY` class.

---

## 5. Current Evaluation & Temporal Splitting

### Current Evaluation Protocol (`phase5/evaluator.py`):
- Evaluates a **single snapshot** immediately preceding each work order ($t_{pred} < t_{event}$).
- Computes: Top-1 Match, Top-3 Recall, Category Match, NDCG@1, NDCG@3, NDCG@5, MRR, Median Lead Time.
- Baseline policies: `NoMaintenancePolicy`, `RandomRankingPolicy`, `SeverityHeuristicPolicy`.

### Baseline Zero-Shot Results (Current):
- **Factory Sensors (63 work orders, progressive failure)**:
  - Top-1 Exact Action Match: **25.40%**
  - Top-3 Action Recall: **34.92%**
  - Category Agreement: **34.92%**
  - NDCG@3: **0.3099** | MRR: **0.3487**
  - Severity Heuristic Baseline: Top-1 = **17.46%**, MRR = **0.2874**
  - Random Baseline: Top-1 = **3.17%**, MRR = **0.1728**
- **OIL-040 (80 work orders, scheduled overhauls)**:
  - Top-1 Exact Action Match: **0.00%**
  - MRR: **0.0659**
  - Phenomenon: At $t_{pred}$ (6h prior), 52.5% of assets were in `WATCH` condition (Health Index ~74.2%). Model prescribed `ACT_MON_ENHANCED` (Top-1 in 90%) and `ACT_INSP_VIBRATION` (Top-3) to avoid premature $1,200 bearing replacement, while historical plant logs recorded pre-scheduled overhauls.

### Temporal Splitting Status:
- Currently, external evaluation is conducted zero-shot over the entire dataset duration.
- **Required Modification for Final Refinement**: Implement a strict chronological 3-way partition on the 1-year Factory Sensors dataset:
  - **Early Period (Adaptation/Dev)**: First 40% of time (Sep 2025 - Jan 2026) -> ~25 work orders
  - **Middle Period (Validation)**: Next 30% of time (Feb 2026 - Apr 2026) -> ~19 work orders
  - **Late Period (Untouched Final Test)**: Final 30% of time (May 2026 - Aug 2026) -> ~19 work orders
  - Final Test must remain strictly untouched during all adaptation fitting, prior tuning, and threshold calibration!

---

## 6. Closed-Loop Simulation Dynamics & Assumptions

### Current Architecture (`phase5/closed_loop/`):
- **Mode A (Historical Replay)**: Evaluates recommendations against actual historical events without state distortion (`is_simulated = False`).
- **Mode B (Surrogate Simulation)**: State transitions governed by physics-informed bounds (`is_simulated = True`):
  $$\Delta HI_{t+1} = \Delta HI_{degradation} + \Delta HI_{action} + \epsilon_t$$
  - `ACT_REPL_BEARING`: $\Delta HI = +45.0$, reset severity to `HEALTHY`, downtime = 6.0h, cost = $2,500
  - `ACT_CORR_LUBRICATE`: $\Delta HI = +15.0$, reset severity to `WATCH`, downtime = 1.0h, cost = $300
  - `ACT_MON_CONTINUE`: $\Delta HI = -2.0$ (natural progressive degradation)

### Identified Gaps for Final Refinement:
- Lack of formal sensitivity analysis across parameter bounds (conservative vs. baseline vs. optimistic).
- Lack of granular event-level cost and downtime breakdowns alongside aggregate simulation metrics.

---

## 7. Current Temporal Leakage Controls

### Invariant Enforced:
$$t_{input\_latest} \le t_{prediction} < t_{future\_event}$$

- Handled by `TemporalLeakageGuard.audit_prediction_point()`.
- Verified across 143 audited prediction points with **0 violations (100% PASS rate)**.
- Quarantine barrier ensures no target labels, failure codes, or future sensor readings enter the feature pipeline.

---

## 8. Summary of What Already Works vs. What Needs Modification

### What Already Works (Must Remain Unchanged):
1. The frozen Phase 4 LightGBM LambdaRank + CBR recommender weights and invariants (`SHA256: bc4b48094f51e6e5...`, $\alpha=0.75$, $k=5$).
2. The core 20-action canonical ontology and category hierarchy.
3. The raw external dataset ingestion and physical unit normalization pipelines (`FactorySensorsAdapter`, `Oil040Adapter`).
4. The causal non-leakage quarantine barrier and `TemporalLeakageGuard`.
5. The decoupled Mode A (Historical Replay) and Mode B (Surrogate Simulation) engine.
6. The existing 13-test automated test suite (`tests/test_phase5_suite.py`).

### What Needs Modification / Addition for Final Refinement:
1. **Multi-Horizon Evaluation (Part 2)**: Replace single snapshot with evaluations at 48h, 24h, 12h, 6h, 3h, 1h where supported.
2. **Event-Window / Opportunity Evaluation (Part 3)**: Evaluate predictive opportunities over a [48h, event) window and collapse repeated predictions per episode.
3. **Hierarchical Action Evaluation (Part 4)**: Evaluate Family, Category, Exact Action, and Action+Component levels.
4. **Action Mapping Audit (Part 5)**: Explicitly audit ontology coverage, multi-action WOs (primary/secondary), and out-of-ontology tracking.
5. **Secondary LLM Semantic Relevance Evaluator (Part 6)**: Strict JSON output (`EXACT_MATCH`, `SEMANTICALLY_EQUIVALENT`, `PARTIALLY_RELEVANT`, `NOT_RELEVANT`, `INSUFFICIENT_INFORMATION`) with consistency validation.
6. **Graded Deterministic Action Relevance (Part 7)**: Ontology-derived hierarchical scoring scheme.
7. **Component-First Evaluation (Part 8)**: Separate affected component from exact procedure (Top-1, Top-3, Precision, Recall, F1).
8. **Failure-Oriented Evaluation (Part 9)**: Event-level anticipation, component localization, missed failures, false alerts.
9. **Selective Prediction & Abstention (Part 10)**: Coverage-performance curves (50%, 70%, 80%, 90%, 100%) with thresholds tuned on validation only.
10. **Confidence Calibration (Part 11)**: Reliability diagram, ECE, and Brier score on validation/test.
11. **Historical-Policy Baselines (Part 12)**: Add Global Majority and Component-Conditioned Majority baselines.
12. **Action-Grounded Adaptation (Part 13)**: Chronological split (Early adaptation, Middle validation, Late untouched test). Lightweight action-prior / residual adaptation comparing Frozen vs. Adapted on the SAME untouched test split.
13. **Data-Efficiency Analysis (Part 14)**: Sweep 5%, 10%, 25%, 50%, 100% of adaptation data.
14. **Comprehensive Evaluation of Adaptation (Part 15)**: Verify adaptation across all 7 dimensions (exact, hierarchical, semantic, component, failure, lead time, selective).
15. **Closed-Loop Sensitivity Analysis (Part 17)**: Run conservative, baseline, and optimistic scenarios.
16. **Event-Level Cost/Benefit Reporting (Part 18)**: Traceable per-event economic metrics.
17. **Statistical Reporting & Confidence Intervals (Part 19)**: Traceable numerators/denominators and bootstrap confidence intervals.
18. **Final Leakage Audit (Part 20)**: Comprehensive 14-point audit document.
19. **Final Consolidated Results Table (Part 21)**: `outputs/phase5/final_refinement/final_results.csv`.
20. **Paper-Ready Figures Suite (Part 22)**: 16 figures covering the complete scientific narrative.
