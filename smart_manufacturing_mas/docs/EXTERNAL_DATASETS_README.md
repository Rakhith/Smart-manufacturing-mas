# Industrial Maintenance Datasets: OOD Testing & Multi-Agent Integration Guide

This guide evaluates external industrial datasets for **Out-of-Distribution (OOD) validation** of our **Phase 4 Prescriptive Maintenance Recommender** (LGBMRanker + Case-Based Reasoning) and future **Phase 5 Multi-Agent System (MAS)** coordination.

---

## 1. Master Ranking & Compatibility Matrix

Datasets are ranked by suitability for **Phase 4** (Telemetry $\rightarrow$ DecisionState $\rightarrow$ Action Ranking) and **Phase 5** (LLM Coordinator & Technician Agents $\rightarrow$ CMMS Work Order Dispatching).

| Rank | Dataset | Equipment Type | Real vs. Synth | Phase 4 Suitability | Phase 5 Suitability | Primary Role |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | [**OIL-040 Sample**](https://huggingface.co/datasets/xpertsystems/oil040-sample) | Oil & Gas Rotating Equipment | Synthetic (ISO-calibrated) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **Gold Standard Prescriptive Benchmark** (12 relational tables: 3-axis vibration, FFT, temps, pressures, RUL, work orders). |
| **2** | [**Factory Sensors & Failure Labels**](https://www.misata.studio/datasets/factory-sensors-failure-labels) | Centrifugal Slurry Pumps | Synthetic (Outcome-grounded) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **End-to-End Pump Benchmark** (Multimodal telemetry + hourly degradation + linked work orders with labor & parts cost). |
| **3** | [**MetroPT2 Benchmark**](https://zenodo.org/records/7766691) | Train Air Compressor (APU) | **Real-World Plant** (7.1M rows @ 1Hz) | ⭐⭐⭐⭐☆ | ⭐⭐⭐☆☆ | **Real Operational Noise Stress Test** (Authentic plant telemetry with confirmed Air Leak & Oil Leak failure windows). |
| **4** | [**EDP Wind Turbine SCADA**](https://www.edp.com/en/innovation/open-data) ([Mendeley Mirror](https://data.mendeley.com/datasets/s95jpdn54n/1)) | Onshore Wind Turbines | **Real-World SCADA** (2 years @ 10-min) | ⭐⭐⭐⭐☆ | ⭐⭐⭐⭐☆ | **Real Industrial SCADA Benchmark** (Genuine component failures: generator bearing, gearbox, hydraulic group + logs). |
| **5** | [**XJTU-SY Bearing Dataset**](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets) | Rolling Element Bearings | **Real Lab Accelerated** (15 run-to-failures) | ⭐⭐⭐⭐☆ | ⭐⭐☆☆☆ | **Run-to-Failure Degradation Baseline** (High-rate 25.6 kHz vibration trajectories through catastrophic failure). |
| **6** | [**AI4I 2020 Predictive Maintenance**](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) | CNC Milling Machine | Synthetic (UCI Benchmark) | ⭐⭐⭐☆☆ | ⭐⭐⭐☆☆ | **Lightweight Pipeline Smoke Test** (10,000 tabular points with 5 independent failure mechanisms). |
| **7** | [**Industrial Maintenance Synthetic**](https://huggingface.co/datasets/Jvachier/industrial-maintenance-synthetic) | 37 Industrial Equipment Types | Synthetic (CMMS Text) | ⭐⭐☆☆☆ | ⭐⭐⭐⭐⭐ | **MAS Dispatch & Text Grounding** (1M SAP/Maximo work orders; no sensor time-series, perfect for LLM agent coordination). |
| **8** | [**NASA Milling Tool Wear**](https://data.nasa.gov/docs/legacy/mill.zip) | High-Speed CNC Milling Machine | **Real Machining Run-to-Failure** | ⭐⭐⭐☆☆ | ⭐⭐☆☆☆ | **Subtractive Machining Generalization** (Flank wear, cutting forces, acoustic emission, current). |
| **9** | [**CWRU Bearing Data Center**](https://engineering.case.edu/bearingdatacenter) | Ball Bearings | **Real Test Rig** (Motor loads 0–3 hp) | ⭐⭐⭐☆☆ | ⭐☆☆☆☆ | **Vibration Diagnostics Baseline** (Canonical bearing vibration benchmark; lack of thermal/fluid telemetry or work orders). |

---

## 2. Deep Dive: The 4 Core Evaluated Datasets

### 1. [OIL-040 Synthetic Vibration & Sensor Dataset](https://huggingface.co/datasets/xpertsystems/oil040-sample)
* **Provider**: XpertSystems.ai (Hugging Face)
* **Equipment**: Rotating equipment in oil & gas (pumps, compressors, turbines, fans) across 80 assets over 30 days (preview of full 15,000-asset product).
* **Modality Coverage**:
  * Mechanical: 3-axis vibration time-series (`vibration_timeseries.csv`), 32-bin spectral FFT (`fft_spectra.csv`), acoustic emission (`acoustic_signals.csv`).
  * Thermal & Fluid: Process and bearing temperatures (`temperature_telemetry.csv`), inlet/discharge pressures (`pressure_telemetry.csv`).
  * Chemical/Lubrication: ISO 4406 oil cleanliness, viscosity, particulate count (`lubrication_analysis.csv`).
* **Prescriptive Action Grounding**:
  * `health_scores.csv`: Quantitative asset degradation trajectory.
  * `remaining_useful_life.csv`: Per-timestamp RUL prognosis.
  * `failure_events.csv`: Root causes, severity tier, repair cost, production loss.
  * `maintenance_workorders.csv`: 6 distinct maintenance action classes, priority, downtime, and technician notes.
* **Why it's #1**: It is one of the only datasets in existence that contains the complete relational schema from raw vibration/FFT to health index, failure events, and actual maintenance work orders with repair costs.

---

### 2. [Factory Sensors & Failure Labels](https://www.misata.studio/datasets/factory-sensors-failure-labels)
* **Provider**: Misata Studio
* **Equipment**: 6 industrial centrifugal slurry pumps (`AST-101` to `AST-206`) monitored hourly over 1 year (52,669 records).
* **Modality Coverage**:
  * Mechanical: Overall RMS vibration (`vibration_mm_s`), peak acceleration (`vibration_peak_accel_g`), high-frequency acceleration (`high_freq_accel_g`), crest factor, axial vibration.
  * Thermal: Bearing temperature (`bearing_temp_c`), motor surface temp, ambient temp.
  * Electrical & Kinematic: Motor current (`motor_current_a`), rotor speed (`rotor_speed_rpm`).
  * Fluid Dynamics: Discharge pressure (`discharge_pressure_bar`), flow rate (`flow_rate_m3_h`), seal flush pressure.
* **Prescriptive Action Grounding**:
  * Degradation stage (`normal` $\rightarrow$ `early` $\rightarrow$ `advanced` $\rightarrow$ `critical`), RUL in hours, and 24-hour failure probability.
  * Failure modes: *Bearing Outer Race Fluting BPFO*, *Impeller Cavitation*, *Mechanical Seal Failure*, *Motor Winding Breakdown*.
  * Linked `work_orders`: Labor hours, parts replacement cost in USD, downtime hours, priority, resolution notes.
* **Why it's #2**: Directly compatible with our CMSR normalization layer and DecisionState feature vector. Clean degradation trajectories make it ideal for verifying whether our `LGBMRanker` recommends the correct replacement vs. repair action.

---

### 3. [MetroPT2 Benchmark Dataset](https://zenodo.org/records/7766691)
* **Provider**: Metro do Porto & University of Porto (Zenodo / Nature Scientific Data 2022)
* **Equipment**: Train Air Compressor (Air Production Unit - APU) on Porto's urban railway.
* **Scale**: 7,116,940 instances logged at 1 Hz from April 28 to July 28, 2022.
* **Modality Coverage**:
  * Pressure: `TP2` (compressor output), `TP3` (pneumatic panel), `H1` (separator drop), `DV_pressure` (dryer valve), air reservoir pressure.
  * Electrical & Control: `Motor_current` (Amperes), compressor activation state (`COMP`), electrical dryer triggers (`DV_electric`).
  * Thermal & Fluid: `Oil_temp` (°C), air flowmeter impulses (`Caudal_impulses`), oil level, pressure switch states.
* **Failure Ground Truth**:
  * Confirmed Failure 1: **Air Leak** (2022-06-04).
  * Confirmed Failure 2: **Oil Leak** (2022-07-11 to 2022-07-14).
* **Why it's #3**: Real-world plant operational data with authentic environmental noise and unscripted degradation. It lacks work-order cost tables, but serves as the best test for whether our recommender raises urgent maintenance recommendations during pre-leak windows.

---

### 4. [Industrial Maintenance Synthetic Dataset](https://huggingface.co/datasets/Jvachier/industrial-maintenance-synthetic)
* **Provider**: Jeremy Vachier (Hugging Face)
* **Equipment**: 37 equipment classes across 145 asset instances.
* **Structure**:
  * `maintenance`: 1M maintenance work orders with `OrderType`, `WorkOrderDescription`, `OperationDescription`, `MaintenanceWorkCenter`, `PlannedWork`, `ActualWork`, `OrderStatus`.
  * `sensor_tags`: 1M standard industrial instrumentation tag strings (ISA-5.1 naming convention).
* **Important Caveat for Phase 4**: Does **NOT** contain continuous physical telemetry series (no continuous vibration or temperature readings over time).
* **Why it's #1 for Phase 5**: While unsuited for sensor-level feature extraction in Phase 4, it is the ideal text corpus for Phase 5 LLM Multi-Agent coordination: generating realistic technician logs, formatting CMMS work orders, and validating labor allocations across work centers.

---

## 3. Recommended Additional Industrial Benchmarks

### 5. [EDP Open Data Wind Turbine SCADA](https://www.edp.com/en/innovation/open-data) / [Mendeley Mirror](https://data.mendeley.com/datasets/s95jpdn54n/1)
* **Equipment**: 5 utility-scale onshore wind turbines in Portugal over 2 full years (10-minute intervals).
* **Telemetry**: Generator bearing temperatures, gearbox oil temperature, hydraulic pitch pressure, rotor RPM, electrical power output.
* **Ground Truth**: Real logbook of confirmed component replacements (generator bearing failures, hydraulic group leaks, gearbox repairs) with exact failure dates and downtime.
* **Value**: Real utility-scale renewable energy benchmark with genuine maintenance interventions.

---

### 6. [XJTU-SY Bearing Dataset](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets)
* **Provider**: Xi'an Jiaotong University & Changxing Sumyoung
* **Equipment**: 15 rolling element bearings run to complete catastrophic failure under 3 operational speed/load regimes.
* **Telemetry**: High-speed dual-axis vibration (horizontal and vertical) sampled at 25.6 kHz every minute.
* **Value**: Pristine run-to-failure degradation curves to test whether our degradation-stage feature engineering transfers outside the NASA IMS bearings ingested in Phase 1.

---

### 7. [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
* **Provider**: UCI Machine Learning Repository (Matzka, 2020)
* **Equipment**: 10,000 synthetic manufacturing/milling machine cycles.
* **Telemetry**: Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min].
* **Failure Modes**: Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), Overstrain Failure (OSF), Random Failure (RNF).
* **Value**: Extremely clean, lightweight tabular dataset. Perfect for fast unit testing and integration smoke tests.

---

### 8. [NASA Milling Tool Wear Dataset](https://data.nasa.gov/docs/legacy/mill.zip)
* **Provider**: UC Berkeley & NASA Prognostics Data Repository
* **Equipment**: CNC milling machine run under variable cutting speeds and depths of cut.
* **Telemetry**: Acoustic emission (table and spindle), 3-axis vibration (table and spindle), cutting force signals, AC motor current.
* **Ground Truth**: Direct optical flank wear ($VB$) measurements taken after each machining pass.
* **Value**: Broadens the system's scope from rotating shafts (bearings/pumps) to **subtractive manufacturing tooling**.

---

### 9. [Case Western Reserve University (CWRU) Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
* **Provider**: Case Western Reserve University
* **Equipment**: 2 hp Reliance electric motor driving a dynamometer with seeded electro-discharge machined faults.
* **Telemetry**: Drive-end and fan-end accelerometer recordings at 12 kHz and 48 kHz across 4 motor loads (0 to 3 hp).
* **Value**: The historical standard baseline for vibration fault classification (inner race, outer race, ball faults).

---

## 4. How to Test These Datasets Against Our Model

To run an OOD benchmark against our trained Phase 4 recommender:

```
External Telemetry (Vibration, Temp, Pressure, Speed)
                     ↓
Phase 2 CMSR Semantic Normalization (Units, Modality Types, Standardized Names)
                     ↓
Phase 3A DecisionState Extraction (Operating Regime, Health Index, Degradation Stage)
                     ↓
Phase 4 Feature Vectorization (Engineered Multi-Modal Telemetry Features)
                     ↓
Phase 4 LGBMRanker + CBR Hybrid Recommender
                     ↓
Ranked Prescriptive Actions (e.g., Replacement vs. Repair vs. Inspection)
                     ↓
Validation against Dataset Work Orders / Failure Events (Precision@1, NDCG@3)
```

1. **Step 1**: Ingest telemetry and map to our 10 CMSR modalities (Mechanical, Thermal, Fluid, Electrical, Kinematic, etc.).
2. **Step 2**: Sample timepoints near degradation inflection points and instantiate `DecisionState` objects.
3. **Step 3**: Score the candidate action catalog using our trained `LGBMRanker` and Case-Based Reasoning database.
4. **Step 4**: Compute **Precision@1** and **NDCG@3** against the actual maintenance actions performed in the dataset's work order logs.
