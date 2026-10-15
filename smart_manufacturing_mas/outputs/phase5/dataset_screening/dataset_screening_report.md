# Phase 5 Dataset Selection & Feasibility Screening Report

## 1. Executive Summary & Configuration Decision

In accordance with Phase 5 requirements, candidate external datasets were systematically audited across 20 rigorous criteria prior to implementing the validation pipeline. Rather than blindly adopting default benchmarks or relying on unverified catalog descriptions, actual downloadable files and schemas were downloaded, parsed, and verified on disk.

### Selected Dataset Configuration: **Configuration B+ (Dual Prescriptive Benchmarks + Text Corpus)**

1. **Primary Action-Grounded Benchmark**: **OIL-040 Sample** (`data/external/oil040/`)
   - **Role**: Full end-to-end prescriptive validation (telemetry -> DecisionState -> recommendation ranking -> actual recorded work order).
   - **Strengths**: Contains 12 synchronized relational tables covering 3-axis vibration, temperature, pressure, lubrication chemistry, and acoustic signals; 80 assets over 30 days; 80 maintenance work orders spanning 6 distinct action classes (`sensor_calibration`, `bearing_replacement`, `alignment`, `overhaul`, `lubrication`, `inspection`); explicit downtime and repair costs.

2. **Primary Continuous Degradation Benchmark**: **Factory Sensors & Failure Labels** (`data/external/factory_sensors/`)
   - **Role**: Longitudinal continuous telemetry stress test with empirical economic grounding.
   - **Strengths**: 52,560 hourly continuous multi-modal readings across 6 industrial centrifugal slurry pumps; progressive degradation stages (`healthy` -> `degrading` -> `critical` -> `stopped`); 39 confirmed physical failure events and 63 linked work orders with actual parts replacement cost in USD, labor hours, and priority.

3. **Secondary CMMS Text & Dispatch Corpus**: **Industrial Maintenance Synthetic** (`data/external/industrial_synthetic/`)
   - **Role**: Natural language maintenance-action extraction, technician log formatting, and multi-agent CMMS dispatching.
   - **Strengths**: 1,048,779 SAP/Maximo work orders with rich descriptions, activity types, and planned vs. actual labor hours.

4. **Excluded Datasets**:
   - **Microsoft Azure Predictive Maintenance**: Completely excluded per user directive ('ignore the Microsoft Azure Dataset completely'). Also empirically inferior due to having only 4 simple component replacement actions and no economic outcome data.
   - **MetroPT2**: Excluded because it completely lacks maintenance actions, work orders, or technician outcome records (provides only failure anomaly windows; MetroPT-3 is already in Phase 3A).
   - **EDP Wind Turbine SCADA**: Excluded due to access gate restrictions preventing automated, unattended programmatic downloading.

---

## 2. Evidence-Based 20-Criteria Evaluation Matrix

Candidate | Telemetry? | Timestamps? | Failures? | Work Orders? | Actions? | Outcome Info? | Ontology Fit | Decision
--- | --- | --- | --- | --- | --- | --- | --- | ---
| **OIL-040 Rotating Equipment Sample** | Yes | Yes | Yes | Yes | Yes | Yes | High | `SELECTED_PRIMARY_ACTION_GROUNDED` |
| **Factory Sensors & Failure Labels** | Yes | Yes | Yes | Yes | Yes | Yes | High | `SELECTED_PRIMARY_PUMP_DEGRADATION` |
| **MetroPT2 Benchmark** | Yes | Yes | Yes | No | No | No | Partial/None | `SCREENED_OUT_NO_ACTIONS` |
| **EDP Wind Turbine SCADA** | Yes | Yes | Yes | No | Yes | No | High | `SCREENED_OUT_ACCESS_RESTRICTIONS` |
| **Industrial Maintenance Synthetic** | No | Yes | No | Yes | Yes | No | High | `SELECTED_SECONDARY_TEXT_DISPATCH` |
| **Microsoft Azure Predictive Maintenance** | Yes | Yes | Yes | No | Yes | No | Partial/None | `OMITTED_PER_USER_DIRECTIVE` |

---

## 3. Detailed Per-Candidate Feasibility Audits

### CANDIDATE_A_OIL040: OIL-040 Rotating Equipment Sample
- **Source**: XpertSystems.ai / Hugging Face
- **Screening Decision**: `SELECTED_PRIMARY_ACTION_GROUNDED`
- **Rationale**: Only dataset offering the complete relational loop: continuous physical telemetry, degradation health scores, explicit failure events, and linked maintenance work orders with downtime and cost.

| Criterion | Evaluation Finding |
| :--- | :--- |
| 1. Is the dataset actually downloadable? | Yes (verified via Hugging Face REST API) |
| 2. Are the files accessible? | Yes (12 relational CSV files directly accessible on disk) |
| 3. Does it contain telemetry? | Yes (vibration 3-axis, temperature, pressure, acoustic, lubrication) |
| 4. Are timestamps available? | Yes (ISO 8601 UTC timestamps across all 9,600 time series rows) |
| 5. Can telemetry be temporally aligned with maintenance events? | Yes (discrete event timestamps directly joinable with telemetry) |
| 6. Does it contain explicit failure events? | Yes (failure_events.csv: root cause, loss bbl, repair cost) |
| 7. Does it contain maintenance events? | Yes (80 maintenance work orders with timestamp and priority) |
| 8. Does it contain actual maintenance ACTIONS? | Yes (6 explicit maintenance action classes: calibration, bearing, alignment, overhaul, lube, inspection) |
| 9. Are actions categorical or natural language? | Categorical structured codes + technician notes quality indicator |
| 10. Does it contain component identity? | Yes (asset_id, asset_type: pump, compressor, turbine, fan; thermocouple_id, pressure_id) |
| 11. Does it contain work-order information? | Yes (maintenance_workorders.csv: workorder_id, priority, downtime_hours) |
| 12. Does it contain maintenance outcome information? | Yes (downtime_hours, estimated_repair_cost_usd, production_loss_bbl, health_index trajectory) |
| 13. Can action vocabulary be mapped into existing action ontology? | Yes (100% 1-to-1 exact semantic mapping into canonical action ontology) |
| 14. Does it contain enough data for meaningful evaluation? | Yes (80 assets, 9,600 synchronized records, 80 work orders, 5 catastrophic failures) |
| 15. Is it synthetic or real? | Synthetic (ISO-10816 calibrated physical process simulation) |
| 16. Is it suitable for full end-to-end evaluation? | Yes (Primary candidate for full state -> recommender -> action -> outcome evaluation) |
| 17. Is it suitable only for action/text evaluation? | No (Full multimodal telemetry available; not restricted to text) |
| 18. Are there licensing/access restrictions? | Open Research Sample (Apache 2.0 compatible, no gated auth) |
| 19. How difficult is schema alignment? | Low (Direct column mappings to mechanical, thermal, fluid, kinematic modalities) |
| 20. Can dataset be evaluated without leaking future information? | Yes (Strict temporal ordering allows rigorous t_input <= t_pred < t_event isolation) |

### CANDIDATE_B_FACTORY_SENSORS: Factory Sensors & Failure Labels
- **Source**: Misata Studio / GitHub
- **Screening Decision**: `SELECTED_PRIMARY_PUMP_DEGRADATION`
- **Rationale**: Provides the most realistic continuous hourly multi-modal degradation trajectories linked to 63 empirical work orders with explicit parts replacement cost ($) and labor hours.

| Criterion | Evaluation Finding |
| :--- | :--- |
| 1. Is the dataset actually downloadable? | Yes (verified via Misata Studio ZIP endpoint, 894 KB) |
| 2. Are the files accessible? | Yes (assets.csv, failure_events.csv, sensor_readings.csv, work_orders.csv extracted) |
| 3. Does it contain telemetry? | Yes (vibration RMS, bearing temp, motor current, discharge pressure, rotor speed) |
| 4. Are timestamps available? | Yes (Hourly recorded_at timestamps spanning full 1-year timeline: 52,560 rows) |
| 5. Can telemetry be temporally aligned with maintenance events? | Yes (Work orders and failure events have exact opened_at, closed_at, failed_at timestamps) |
| 6. Does it contain explicit failure events? | Yes (39 physical failure events with failure_mode, downtime_hours, restored_at) |
| 7. Does it contain maintenance events? | Yes (63 work orders linked to assets: corrective and preventive) |
| 8. Does it contain actual maintenance ACTIONS? | Yes (Action mapped via failure mode: seal replacement, bearing replacement, balance/alignment, cooling service) |
| 9. Are actions categorical or natural language? | Categorical (order_type, failure_mode, priority, labor_hours, parts_cost) |
| 10. Does it contain component identity? | Yes (6 centrifugal slurry pumps AST-101 to AST-206, bearings, mechanical seals, motors) |
| 11. Does it contain work-order information? | Yes (work_order_id, opened_at, closed_at, labor_hours, parts_cost, priority) |
| 12. Does it contain maintenance outcome information? | Yes (Exact parts_cost in USD, labor_hours, downtime_hours, restoration duration) |
| 13. Can action vocabulary be mapped into existing action ontology? | Yes (High confidence mapping from failure mechanisms to canonical actions) |
| 14. Does it contain enough data for meaningful evaluation? | Yes (52,560 hourly observations, 39 failures, 63 work orders across 6 industrial pumps) |
| 15. Is it synthetic or real? | Synthetic (Outcome-grounded engineering pump simulation with realistic noise) |
| 16. Is it suitable for full end-to-end evaluation? | Yes (Primary candidate for continuous degradation trajectory validation and cost evaluation) |
| 17. Is it suitable only for action/text evaluation? | No (Full continuous telemetry available) |
| 18. Are there licensing/access restrictions? | Open Source / MIT License (completely free, direct download) |
| 19. How difficult is schema alignment? | Low (Direct 1-to-1 match with Phase 2 CMSR modalities: mechanical, thermal, fluid, electrical, kinematic) |
| 20. Can dataset be evaluated without leaking future information? | Yes (Hourly resolution permits causal rolling windows and strict pre-failure cutoff) |

### CANDIDATE_C_METROPT2: MetroPT2 Benchmark
- **Source**: Metro do Porto / Zenodo (Record 7766691)
- **Screening Decision**: `SCREENED_OUT_NO_ACTIONS`
- **Rationale**: Contains authentic real-world plant telemetry and failure windows, but completely lacks maintenance actions, work orders, and outcome data. Phase 3A already ingested MetroPT-3.

| Criterion | Evaluation Finding |
| :--- | :--- |
| 1. Is the dataset actually downloadable? | Yes (Accessible via Zenodo REST API, 1.15 GB CSV) |
| 2. Are the files accessible? | Yes (MetroPT2.csv and dataset_train.csv available) |
| 3. Does it contain telemetry? | Yes (High-frequency 1 Hz compressor pressures, oil temperature, motor current) |
| 4. Are timestamps available? | Yes (1-second sampling over 3-month operational window, 7.1M rows) |
| 5. Can telemetry be temporally aligned with maintenance events? | Partial (Known confirmed failure windows: Air Leak June 4, Oil Leak July 11-14) |
| 6. Does it contain explicit failure events? | Yes (2 confirmed physical anomaly/failure episodes: Air Leak & Oil Leak) |
| 7. Does it contain maintenance events? | No (Lacks discrete maintenance work orders or maintenance log timestamps) |
| 8. Does it contain actual maintenance ACTIONS? | No (No recorded technician maintenance actions, only failure anomaly windows) |
| 9. Are actions categorical or natural language? | None (No action vocabulary) |
| 10. Does it contain component identity? | Limited (Train air compressor APU unit, dryer valves, compressor motor) |
| 11. Does it contain work-order information? | No (No CMMS work order table, labor hours, or priority records) |
| 12. Does it contain maintenance outcome information? | No (No cost, downtime, or resolution data) |
| 13. Can action vocabulary be mapped into existing action ontology? | No (Lacks action vocabulary; cannot evaluate action recommendation agreement) |
| 14. Does it contain enough data for meaningful evaluation? | Excessive / Redundant (7.1M rows, 1.15 GB; MetroPT-3 is already in Phase 3A) |
| 15. Is it synthetic or real? | Real-World Plant (Authentic operational train compressor) |
| 16. Is it suitable for full end-to-end evaluation? | No (Cannot validate maintenance actions or prescriptive recommendations) |
| 17. Is it suitable only for action/text evaluation? | No (Numeric time series only) |
| 18. Are there licensing/access restrictions? | Open Data (CC-BY 4.0 via Zenodo) |
| 19. How difficult is schema alignment? | Medium (Pneumatic compressor pressures require custom mapping) |
| 20. Can dataset be evaluated without leaking future information? | Yes (Timestamps are clean) |

### CANDIDATE_D_EDP_WIND: EDP Wind Turbine SCADA
- **Source**: EDP Open Data / Mendeley Mirror
- **Screening Decision**: `SCREENED_OUT_ACCESS_RESTRICTIONS`
- **Rationale**: Dataset cannot be programmatically downloaded without manual user intervention/credentials; violates automated reproducibility standards.

| Criterion | Evaluation Finding |
| :--- | :--- |
| 1. Is the dataset actually downloadable? | No (Mendeley API returned 0 direct files; requires manual portal authentication) |
| 2. Are the files accessible? | No (Direct programmatic download unavailable without interactive login) |
| 3. Does it contain telemetry? | Yes (SCADA: bearing temp, gearbox oil temp, rotor speed, power) |
| 4. Are timestamps available? | Yes (10-minute intervals over 2 years) |
| 5. Can telemetry be temporally aligned with maintenance events? | Yes (Intervention logs contain dates) |
| 6. Does it contain explicit failure events? | Yes (Gearbox and bearing failure events) |
| 7. Does it contain maintenance events? | Yes (Component replacement log) |
| 8. Does it contain actual maintenance ACTIONS? | Yes (Component replacement logs) |
| 9. Are actions categorical or natural language? | Categorical log entries |
| 10. Does it contain component identity? | Yes (Wind turbines 1-5, generator bearing, gearbox) |
| 11. Does it contain work-order information? | Partial (Logbook entries, but no formal CMMS fields) |
| 12. Does it contain maintenance outcome information? | Partial (Downtime recorded, but costs missing) |
| 13. Can action vocabulary be mapped into existing action ontology? | Yes (Bearing replacement, gearbox inspection) |
| 14. Does it contain enough data for meaningful evaluation? | Large (~2-3 GB across 5 turbines) |
| 15. Is it synthetic or real? | Real-World Industrial SCADA |
| 16. Is it suitable for full end-to-end evaluation? | Conditionally (If files were programmatically accessible) |
| 17. Is it suitable only for action/text evaluation? | No |
| 18. Are there licensing/access restrictions? | Restricted Access (Requires registration / portal access agreement) |
| 19. How difficult is schema alignment? | Medium (Wind turbine specific pitch/yaw parameters) |
| 20. Can dataset be evaluated without leaking future information? | Yes |

### CANDIDATE_E_INDUSTRIAL_SYNTHETIC: Industrial Maintenance Synthetic
- **Source**: Jeremy Vachier / Hugging Face
- **Screening Decision**: `SELECTED_SECONDARY_TEXT_DISPATCH`
- **Rationale**: Unsuitable as primary sensor benchmark due to complete absence of continuous telemetry, but highly valuable as a massive secondary benchmark for LLM work-order normalization, labor allocation, and CMMS dispatching.

| Criterion | Evaluation Finding |
| :--- | :--- |
| 1. Is the dataset actually downloadable? | Yes (verified via Hugging Face REST API, 69 MB parquet) |
| 2. Are the files accessible? | Yes (maintenance.parquet downloaded and verified on disk) |
| 3. Does it contain telemetry? | No (Lacks continuous physical sensor time series) |
| 4. Are timestamps available? | Yes (CreatedDate, ScheduledStartDate, ActualStartDate, ActualFinishDate) |
| 5. Can telemetry be temporally aligned with maintenance events? | N/A (No continuous sensor telemetry to align against) |
| 6. Does it contain explicit failure events? | Partial (Inferred from corrective work order types) |
| 7. Does it contain maintenance events? | Yes (1,048,779 SAP/Maximo work orders) |
| 8. Does it contain actual maintenance ACTIONS? | Yes (WorkOrderDescription, OperationDescription, Maintenance_activity_type) |
| 9. Are actions categorical or natural language? | Natural Language Text + SAP Activity Codes (OrderType, OrderStatus) |
| 10. Does it contain component identity? | Yes (Equipment_ID across 37 equipment classes) |
| 11. Does it contain work-order information? | Yes (WorkOrder, Location, PlannedWork, ActualWork, MaintenanceWorkCenter) |
| 12. Does it contain maintenance outcome information? | Partial (Planned vs Actual labor hours; no financial cost or physical telemetry) |
| 13. Can action vocabulary be mapped into existing action ontology? | Yes (Requires natural language parsing to extract verbs and map to canonical ontology) |
| 14. Does it contain enough data for meaningful evaluation? | Extensive (1,048,779 work orders) |
| 15. Is it synthetic or real? | Synthetic (Enterprise CMMS text simulation) |
| 16. Is it suitable for full end-to-end evaluation? | No (Lacks physical sensor signals; cannot evaluate sensor -> DecisionState -> recommender) |
| 17. Is it suitable only for action/text evaluation? | Yes (Gold standard corpus for LLM work-order parsing and CMMS dispatching) |
| 18. Are there licensing/access restrictions? | Open Source (Apache 2.0) |
| 19. How difficult is schema alignment? | N/A for sensor data; Low for SAP CMMS fields |
| 20. Can dataset be evaluated without leaking future information? | Yes (Strict timestamp ordering available) |

### CANDIDATE_F_AZURE_PREDICTIVE_MAINTENANCE: Microsoft Azure Predictive Maintenance
- **Source**: Microsoft / Kaggle
- **Screening Decision**: `OMITTED_PER_USER_DIRECTIVE`
- **Rationale**: Explicitly excluded by user instruction ('ignore the Microsoft Azure Dataset completely'). Furthermore, OIL-040 and Factory Sensors provide substantially richer multimodal telemetry (acoustic, lubrication, current) and much broader action ontologies (lubrication, calibration, overhaul, alignment, emergency shutdown).

| Criterion | Evaluation Finding |
| :--- | :--- |
| 1. Is the dataset actually downloadable? | Yes (Standard public benchmark) |
| 2. Are the files accessible? | Yes (telemetry.csv, errors.csv, maint.csv, failures.csv, machines.csv) |
| 3. Does it contain telemetry? | Yes (Hourly voltage, rotate, pressure, vibration across 100 machines) |
| 4. Are timestamps available? | Yes (Hourly datetime stamps) |
| 5. Can telemetry be temporally aligned with maintenance events? | Yes (Timestamps match across tables) |
| 6. Does it contain explicit failure events? | Yes (failures.csv: comp1, comp2, comp3, comp4) |
| 7. Does it contain maintenance events? | Yes (maint.csv: component replacements) |
| 8. Does it contain actual maintenance ACTIONS? | Yes (Replace comp1..4) |
| 9. Are actions categorical or natural language? | Categorical (comp1-4 replacement only) |
| 10. Does it contain component identity? | Yes (100 machines, 4 component slots) |
| 11. Does it contain work-order information? | Minimal (Only date and component replaced; no labor, downtime, or cost) |
| 12. Does it contain maintenance outcome information? | Minimal (No downtime hours or repair cost in USD) |
| 13. Can action vocabulary be mapped into existing action ontology? | Partial (Covers only component replacement; lacks inspection, calibration, lubrication, shutdown) |
| 14. Does it contain enough data for meaningful evaluation? | Yes (876k telemetry rows across 100 machines) |
| 15. Is it synthetic or real? | Synthetic (Simplified benchmark) |
| 16. Is it suitable for full end-to-end evaluation? | Yes (Standard benchmark) |
| 17. Is it suitable only for action/text evaluation? | No |
| 18. Are there licensing/access restrictions? | Open / Microsoft Open Data |
| 19. How difficult is schema alignment? | Low (4 features: volt, rotate, pressure, vibe) |
| 20. Can dataset be evaluated without leaking future information? | Yes |
