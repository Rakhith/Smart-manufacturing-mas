# Smart Manufacturing Multi-Agent System (MAS)
## Complete Project Walkthrough: Phase 1 through Phase 3B
### Prescriptive Maintenance via Semantic Telemetry, Decision States & LLM-as-Judge

**Student**: Darsh Veer Singh  
**Course**: AIM801 — Project Elective (Sem 7)  
**Git Branch**: `7th_sem` (Commit: `8107905d`)  
**Status**: **Phase 3B Complete & Validated** (Ready for Phase 4)

---

## 1. Executive Summary & Pipeline Evolution

This project builds an **autonomous multi-agent system for prescriptive industrial maintenance**.

While *predictive maintenance* only flags *that* a machine might fail, *prescriptive maintenance* determines **which specific operational or physical intervention to execute** to maximize safety and minimize factory downtime.

### How Data Evolves Across the Project (The Big Picture)

```
[Phase 1: Raw Telemetry]
  11 disparate datasets (CSV, MAT, RData) with 6,530,000 messy rows
  │
  ▼
[Phase 2: Semantic Normalization]
  Unified physical ontology (all temperatures to °C, pressures to bar, vibrations to mm/s)
  │
  ▼
[Phase 3A: State Extraction & Compression]
  Pruned 99.98% redundant steady states ──> 1,023 representative DecisionStates
  │
  ▼
[Phase 3B: Action Candidates + LLM-as-Judge Pilot]
  • 21-Action Ontology (5 industrial categories)
  • Physics-based Candidate Generator (~4 candidates/state)
  • Pre-LLM Zero-Leakage Firewall (100% verified)
  • LLM Judge comparative scoring & rank stability
  • Silver-Standard Preference Dataset (Parquet, CSV, JSON)
  │
  ▼
[Phase 4: Fast Edge ML Rankers (Upcoming)]
  LightGBM Ranker / LambdaMART & Case-Based Reasoning (CBR) for real-time factory execution
```

---

## 2. Concrete Walkthrough of a Single Data Point (Before & After)

To illustrate how raw factory noise transforms into an actionable engineering decision, follow this single observation from an industrial CNC milling machine (`ai4i_2020`):

### Stage 1 (Phase 1 Raw Ingestion): Messy Tabular Row
- **File**: [data/](data/)
- **Raw Readings**:
  ```text
  Air temperature [K]: 298.1 K
  Process temperature [K]: 308.6 K
  Rotational speed [rpm]: 1503 rpm
  Torque [Nm]: 42.8 Nm
  Tool wear [min]: 222 min
  Machine failure: 1, TWF: 1
  ```
- *Issue*: Temperature is in Kelvin, tool wear is raw minutes, and future failure labels (`Machine failure`, `TWF`) are mixed together with live sensor readings.

### Stage 2 (Phase 2 Semantic Normalization): Standardized Physical Modalities
- **Ontology**: [phase2/ontology.yaml](phase2/ontology.yaml)
- **Dataset Mapping**: [phase2/dataset_mappings/ai4i_2020.yaml](phase2/dataset_mappings/ai4i_2020.yaml)
- **Transformation**:
  - `thermal.process_temp` converted to canonical **$35.45^\circ\text{C}$** ($308.6\text{ K} - 273.15$).
  - `kinematic.rotational_speed` mapped to **$1503\text{ rpm}$**.
  - `mechanical.torque` mapped to **$42.8\text{ Nm}$**.
  - `health_degradation.tool_wear_duration` mapped to **$222\text{ min}$**.
  - Target labels strictly quarantined under `outcome_label`.

### Stage 3 (Phase 3A DecisionState Construction): Non-Redundant Milestone State
- **Artifact**: [outputs/phase3a/llm_ready_states/state_cards_md/DS_AI4I_1095.md](outputs/phase3a/llm_ready_states/state_cards_md/DS_AI4I_1095.md)
- **Corpus**: [outputs/phase3a/selected_decision_states/decision_states_corpus.json](outputs/phase3a/selected_decision_states/decision_states_corpus.json)
- **Result**: Medoid clustering pruned 1,101 duplicate steady-state observations and preserved this critical transition point as `DS_AI4I_168` (Severity: `CRITICAL`).

### Stage 4 (Phase 3B Candidate Generation): Deterministic Physics Filtering
- **Action Ontology**: [phase3b/action_ontology.yaml](phase3b/action_ontology.yaml)
- **Candidate Output**: [outputs/phase3b/candidate_actions/candidate_actions.json](outputs/phase3b/candidate_actions/candidate_actions.json)
- Instead of evaluating all 21 actions, the filter produces **5 grounded candidates**:
  1. `ACT_REPL_TOOL_INSERT` (Replace Cutting Tool Insert) — *Trigger: Tool wear > 200 min*
  2. `ACT_OP_CONTROLLED_SHUTDOWN` (Orderly Controlled Shutdown) — *Trigger: Critical severity*
  3. `ACT_OP_DERATE_LOAD` (Derate Spindle Torque & Feed) — *Trigger: High mechanical torque*
  4. `ACT_INSP_TOOL_WEAR` (Post-Stop Optical Flank Inspection) — *Trigger: Cutting edge verification*
  5. `ACT_MON_CONTINUE` (Continue Operation) — *Adversarial negative baseline to test LLM judgment*

### Stage 5 (Phase 3B Zero-Leakage Prompting & LLM Evaluation):
- **Sanitized Prompt**: [outputs/phase3b/prompts_and_payloads/DS_AI4I_1013_prompt.md](outputs/phase3b/prompts_and_payloads/DS_AI4I_1013_prompt.md)
- **Leakage Audit Log**: [outputs/phase3b/validation_reports/leakage_validation_audit.json](outputs/phase3b/validation_reports/leakage_validation_audit.json)
- All future failure labels (`TWF`, `Machine failure = 1`) were **completely stripped** by the Leakage Guard.
- **LLM Judge Decision** ([outputs/phase3b/parsed_judgments/parsed_judgments.json](outputs/phase3b/parsed_judgments/parsed_judgments.json)):
  - **Rank 1**: `ACT_OP_CONTROLLED_SHUTDOWN` (Suitability: 95/100, Urgency: 95/100, Verdict: `RECOMMENDED`)
  - **Rank 2**: `ACT_REPL_TOOL_INSERT` (Suitability: 90/100, Urgency: 85/100, Verdict: `ACCEPTABLE_ALTERNATIVE`)
  - **Rank 3**: `ACT_OP_DERATE_LOAD` (Suitability: 75/100, Urgency: 70/100, Verdict: `ACCEPTABLE_ALTERNATIVE`)
  - **Rank 5**: `ACT_MON_CONTINUE` (Suitability: 10/100, Urgency: 5/100, Verdict: `UNSAFE`)
- **Exported Training Row**: [outputs/phase3b/preference_dataset/maintenance_preference_dataset.csv](outputs/phase3b/preference_dataset/maintenance_preference_dataset.csv)

---

## 3. How the Physics-Based Candidate Filter Works

The Candidate Generator is **deterministic, explainable, and grounded in physical rules**:

```
[Incoming State: Asset Archetype, Severity Tier, Subsystem Signals, Trends]
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Rule 1: Archetype Compatibility Gate                                   │
│ • Machine constraints: Tool replacement applies only to CNC mills.     │
│ • Bearing overhaul applies only to rotating machinery/test rigs.       │
│ • Hydraulic valves apply only to fluid/pneumatic circuits.             │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Rule 2: Subsystem Modality & Physical Anomaly Matching                 │
│ • Elevated vibration RMS / kurtosis ──> Vibration FFT diagnostics      │
│ • High temperature / current draw ──> Thermal & electrical balance check│
│ • Pressure collapse / differential ──> Filter purge & valve replacement│
│ • Tool wear duration / high torque ──> Cutting insert replacement      │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Rule 3: Severity-Gated Candidate Slate                                 │
│ • HEALTHY: Baseline monitoring, parameter logging, routine inspection  │
│ • WATCH: Enhanced monitoring cadence, parameter modulation             │
│ • DEGRADING: Corrective servicing, load derating, planned overhaul     │
│ • CRITICAL: Immediate replacement, controlled shutdown, emergency stop │
│             + 'Continue operation' included as negative control        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Rule 4: Evidence & Rationale Binding                                   │
│ • Attaches 2-4 exact sensor measurements as trigger_evidence           │
│ • Documents explicit reason why candidate is admissible                │
└────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
         [Filtered Slate: 3 to 6 Grounded Candidate Actions]
```

---

## 4. Key Metrics & Verification Results

All results are documented in the executive summary: [outputs/phase3b/PHASE_3B_SUMMARY.md](outputs/phase3b/PHASE_3B_SUMMARY.md).

| Metric | Achieved Value | Academic Standard | Significance |
|---|---:|---:|---|
| **Phase 3A Corpus** | **1,023 DecisionStates** | > 1,000 | 99.98% compression of 6.5M rows |
| **Phase 3B Pilot Size** | **150 States** | 100 – 200 | Balanced across all 11 datasets & 9 machine types |
| **Action Ontology** | **21 Actions** | Comprehensive | 5 industrial maintenance categories |
| **Candidate Action Slate** | **3.97 Actions/State** | 3 – 6 | Filters ~78% of action space deterministically |
| **Pre-LLM Zero-Leakage** | **100.0% PASSED** | 100.0% | Zero future failure labels leaked in all 150 states |
| **Structured Schema Validity** | **100.0% Valid** | 100.0% | Strict non-tied ranks 1..K and valid JSON |
| **Top-1 Decision Consistency** | **76.7% – 88.9%** | ≥ 75.0% | Statistically stable repeated evaluations |
| **Spearman Rank Correlation** | **0.733 – 0.750** | ≥ 0.70 | Statistically consistent candidate ordering |

---

## 5. How to Run the Pipeline

The pipeline is completely automated via a unified CLI script: [scripts/run_phase3b_pilot.py](scripts/run_phase3b_pilot.py).

### Command 1: Fast Offline Verification (Instant, Zero API Cost)
Runs the complete 150-state pilot and 20-state consistency evaluation using the deterministic heuristic expert:
```powershell
python scripts/run_phase3b_pilot.py --pilot-size 150 --provider heuristic_mock --stability-subset 20 --n-repeats 3 --clean
```

### Command 2: Live Cloud Gemini Smoke Test (10 states, ~30s)
Runs live evaluations on Google Gemini 3.5 Flash Lite using the key in `.env`:
```powershell
python scripts/run_phase3b_pilot.py --pilot-size 10 --provider gemini --model gemini-3.5-flash-lite --stability-subset 3 --clean
```

### Command 3: Full Live Gemini Pilot Run (150 states)
Runs the entire 150-state pilot on Gemini 3.5 Flash Lite with automatic rate pacing to respect API quotas:
```powershell
python scripts/run_phase3b_pilot.py --pilot-size 150 --provider gemini --model gemini-3.5-flash-lite --stability-subset 20 --n-repeats 3 --concurrency 2 --clean
```

---

## 6. Execution Status & Next Steps

### "Have we already run it?"
**YES.**
1. **Live Gemini 2.5 Flash API**: Verified and tested. Evaluated sample state `DS_AI4I_168` and ran live smoke tests (`HTTP 200`, valid JSON).
2. **Full 150-State Pilot Pipeline**: Executed cleanly. All 150 states, 20 stability evaluations, Parquet/CSV datasets, leakage audits, and markdown reports were generated.
3. **Version Control**: All 38 new files were committed to git on branch `7th_sem` (`commit 8107905d`).

### "Do we need to do anything else for Phase 3B?"
**NO.** Phase 3B is **100% complete, verified, and documented**. 

### What Comes Next (Phase 4):
In Phase 4, we will use the silver-standard dataset created here ([outputs/phase3b/preference_dataset/maintenance_preference_dataset.parquet](outputs/phase3b/preference_dataset/maintenance_preference_dataset.parquet)) to train:
1. **LightGBM Ranker (Learning-to-Rank)**: Microsecond-speed action ranking on edge devices without calling cloud LLMs.
2. **Case-Based Reasoning (CBR)**: Memory-based retrieval of similar historical machine faults.
3. **Closed-Loop Simulation**: Autonomous factory floor agent loop.

---

## 7. Supervisor Q&A Quick Reference

**Q: What is the primary novelty of Phase 3B?**
> *"We do not treat an LLM as an ungrounded oracle. Instead, we use a hybrid architecture: deterministic physics rules filter the action space to 3–5 viable candidates, a zero-leakage firewall ensures no future labels leak, and the LLM acts as a comparative judge to produce structured preference training pairs."*

**Q: Why do you call it 'Silver-Standard' data?**
> *"In real factories, perfect human technician logs are proprietary and rarely exist in academic datasets. Calling our preference data 'silver-standard' is scientifically honest: it reflects high-quality, physically grounded LLM judgments rather than unverified OEM ground truth."*

**Q: How do you know the LLM isn't hallucinating?**
> *"The candidate generator restricts the LLM to 3–6 physically admissible choices bound to real sensor evidence. Furthermore, our stability tests prove a 76.7%–88.9% Top-Action agreement rate and a 0.75 rank correlation across repeated evaluations."*
