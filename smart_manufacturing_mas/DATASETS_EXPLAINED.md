# 📊 Comprehensive Guide to Testing Datasets: Features, Modalities & Selection Rationale

**Project**: Smart Manufacturing Multi-Agent System (MAS) for Autonomous Prescriptive Maintenance  
**Course**: AIM801 — Project Elective (Sem 7)  
**File Location**: [`DATASETS_EXPLAINED.md`](file:///Users/rakshith/Desktop/Smart-manufacturing-mas/smart_manufacturing_mas/DATASETS_EXPLAINED.md)

---

## 1. Executive Summary & Testing Suite Overview

To validate an **autonomous multi-agent system (MAS) for prescriptive maintenance**, using a single dataset (e.g., only CNC machine failure or only aero-engine degradation) is insufficient. Real-world smart factories deploy diverse machinery with vastly different physics: high-speed rotating bearings, heavy pneumatic compressors, multi-stage chemical reactors, high-pressure fluid hydraulics, and networked IIoT edge devices.

Our evaluation benchmark incorporates **10 active industrial datasets** (plus 1 documented deprecated entry) covering **9 distinct physical archetypes** and over **6.5 million raw telemetry observations**.

### Active Installed Dataset Registry

| # | Dataset ID | Industrial Archetype | Domain | Size | Sampling / Structure | Active Status |
|---|------------|----------------------|--------|------|----------------------|---------------|
| 1 | `ai4i_2020` | CNC Milling Machine | CNC Machining | 512 KB | Static Tabular (10k rows) | ✅ Active |
| 2 | `cmapss` | Turbofan Jet Engine | Aerospace | 43 MB | Run-to-Failure Trajectories (265k rows) | ✅ Active |
| 3 | `smart_maintenance` | Factory Motor Fleet | Factory Maintenance | 7.1 MB | Mixed Tabular & Time Series (101k rows) | ✅ Active |
| 4 | `iiot_6g` | Connected Smart Factory | IIoT / 6G Wireless | 19 MB | Low-Frequency Time Series (100k rows) | ✅ Active |
| 5 | `nasa_milling` | CNC Metal Cutting Tool | Manufacturing | 14 MB | Run-to-Failure Signals (1,002 runs) | ✅ Active |
| 6 | `metropt3` | Railway Air Compressor | Heavy Pneumatics | 208 MB | 1 Hz Multivariate Time Series (1.5M rows) | ✅ Active |
| 7 | `uci_hydraulic` | Hydraulic Test Rig | Fluid Power | 531 MB | Synchronized Multirate Cycles (2.2k cycles) | ✅ Active |
| 8 | `tennessee_eastman` | Chemical Process Plant | Chemical Engineering | 550 MB | Multivariate Simulation (5.7M rows) | ✅ Active |
| 9 | `nasa_ims` | High-Speed Bearing Rig | Rotary Drive | 2.8 GB | High-Frequency Vibration (20 kHz waveforms) | ✅ Active |
| 10 | `metal_etch` | Plasma Wafer Etcher | Semiconductor | 1.4 MB | Static Process Tabular (10.6k rows) | ✅ Active |
| 11 | `digital_manufacturing` | Supply Chain / ERP | ERP Business Sales | 2.0 MB | Deprecated (Non-engineering business fields) | ❌ Deprecated |

---

## 2. Strategic Rationale: Why These Specific Datasets Were Chosen

The selection of these datasets was driven by five core engineering principles essential for building a robust **prescriptive maintenance** (action-recommendation) pipeline rather than a simple *predictive maintenance* (binary alarm) pipeline:

### Rationale 1: Full Coverage of Industrial Physical Modalities
An industrial agent must reason over multiple physical phenomena. Our suite spans 8 core physical modalities:
- **Thermal**: Process temperatures, ambient air temperatures, oil temperatures, cooling power ($^\circ\text{C}$, $\text{K}$).
- **Mechanical & Kinematic**: Spindle torque ($\text{Nm}$), rotational speed ($\text{rpm}$), tool wear duration ($\text{min}$), axial forces ($\text{N}$).
- **Fluid & Hydraulics**: System pressures ($\text{bar}$), differential pressure drop ($\text{bar}$), volume flow rate ($\text{l/min}$).
- **Vibration & Kinematic Waves**: High-frequency acceleration ($g$), vibration velocity ($\text{mm/s}$), kurtosis, crest factor, spectral entropy.
- **Electrical**: Motor current ($\text{A}$), power draw ($\text{kW}$), total energy consumption ($\text{kWh}$).
- **Acoustic Emission**: Sound pressure levels ($\text{dB}$), ultrasonic strain waves ($\text{V}$).
- **IIoT & Cyber-Physical**: Network latency ($\text{ms}$), packet loss rate ($\%$), edge computing energy usage.
- **Chemical & Process**: Reactor liquid levels, mole fractions, feed flow rates, stripper pressures.

### Rationale 2: Diversity of Degradation Dynamics & Failure Mechanics
Different machines fail in different ways:
- **Continuous Wear & Friction**: Tool flank wear in `ai4i_2020` and `nasa_milling`.
- **Systemic Thermal/Compressor Aging**: High-Pressure Compressor (HPC) degradation in `cmapss`.
- **Spontaneous Multi-Component Breakdown**: Valve leakage, pump slip, cooler inefficiency, and accumulator pre-charge loss in `uci_hydraulic`.
- **Catastrophic Surface Fatigue & Pitting**: Outer-race and roller bearing spalling under 6,000 lbs radial load in `nasa_ims`.
- **Complex Multivariant Transients**: Unmeasured feed concentration shifts and cooling water valve trips in `tennessee_eastman`.

### Rationale 3: Temporal & Sampling Rate Spectrum
Industrial telemetry arrives at different time horizons. Our dataset suite exercises models across the complete sampling spectrum:
- **High-Frequency Waveforms (20 kHz)**: 1-second snapshots requiring FFT energy, spectral entropy, kurtosis, and crest-factor feature extraction (`nasa_ims`).
- **Synchronized Cycle-Based Multirate Signals (1 Hz – 100 Hz)**: 60-second operational cycles (`uci_hydraulic`).
- **Continuous Continuous Time-Series Telemetry (1 Hz)**: 1.5M continuous compressor records (`metropt3`).
- **Batch / Static Telemetry**: Instantaneous machine operational snapshots (`ai4i_2020`, `metal_etch`).

### Rationale 4: Action-Grounded Decision State Generation
For prescriptive maintenance, the system must evaluate **what corrective action to take**. The chosen datasets contain explicit sensor signatures that bind directly to physical maintenance actions:
- High Tool Wear $\rightarrow$ `ACT_REPL_TOOL_INSERT` (Replace Cutting Insert).
- Elevated Oil Temp / Low Cooling Efficiency $\rightarrow$ `ACT_CLEAN_HEAT_EXCHANGER` / `ACT_ADJ_COOLANT_FLOW`.
- Hydraulic Accumulator Pressure Collapse $\rightarrow$ `ACT_RECHARGE_ACCUMULATOR`.
- High Bearing Vibration Kurtosis $\rightarrow$ `ACT_GREASE_BEARING` or `ACT_REPL_BEARING_ASSEMBLY`.
- Air Compressor Line Pressure Drop $\rightarrow$ `ACT_PURGE_FILTER` / `ACT_INSP_PNEUMATIC_VALVE`.

### Rationale 5: Practical Bandwidth & Storage Tractability
Many public SCADA repositories exceed hundreds of gigabytes (e.g., 11+ GB wind turbine vibration archives), making edge deployment and rapid experimentation intractable. We enforced a strict constraint: **all benchmark datasets must be under ~3 GB**, enabling full reproducibility while preserving multi-gigabyte scientific rigor.

---

## 3. Detailed Feature Analysis of Each Dataset

---

### 1. AI4I 2020 Predictive Maintenance Dataset (`ai4i_2020`)
- **Domain**: CNC Milling Machine / Precision Tooling
- **Processing Type**: Static Tabular (`static_tabular`)
- **Scale**: 10,000 observations, 14 features (512 KB)
- **Description**: Reflects synthetic mill operational data generated to mirror real manufacturing conditions. Captures tool wear, torque, and process temperatures under three product quality grades (Low, Medium, High).

#### Feature & Sensor Catalogue

| Feature Name | Physical Meaning | Unit | Modality / Category |
|--------------|------------------|------|---------------------|
| `UDI` | Row identifier | None | Identifier |
| `Product ID` | Variant serial number (L/M/H) | String | Context |
| `Type` | Product quality class (L=50%, M=30%, H=20%) | Categorical | Operating Context |
| `Air temperature [K]` | Ambient air temperature | Kelvin ($\text{K}$) $\rightarrow$ Standardized to ${^\circ\text{C}}$ | Thermal |
| `Process temperature [K]` | Internal machining process temp | Kelvin ($\text{K}$) $\rightarrow$ Standardized to ${^\circ\text{C}}$ | Thermal |
| `Rotational speed [rpm]` | Spindle speed | $\text{rpm}$ | Kinematic |
| `Torque [Nm]` | Spindle torque load | $\text{Nm}$ | Mechanical |
| `Tool wear [min]` | Cumulative cutting tool operational time | $\text{min}$ | Health / Degradation |
| `Machine failure` | Overall binary failure indicator | $0/1$ | Outcome Label |
| `TWF` | Tool Wear Failure mode | $0/1$ | Outcome Label |
| `HDF` | Heat Dissipation Failure mode | $0/1$ | Outcome Label |
| `PWF` | Power Failure mode | $0/1$ | Outcome Label |
| `OSF` | Overstrain Failure mode | $0/1$ | Outcome Label |
| `RNF` | Random Failure mode | $0/1$ | Outcome Label |

- **Prescriptive Utility**: Excellent for evaluating trade-offs between tool replacement (`ACT_REPL_TOOL_INSERT`), feed derating (`ACT_OP_DERATE_LOAD`), thermal pauses, and controlled shutdown.

---

### 2. NASA C-MAPSS Turbofan Engine Dataset (`cmapss`)
- **Domain**: Aerospace Gas Turbine Engines (Commercial Modular Aero-Propulsion System Simulation)
- **Processing Type**: Run-to-Failure Trajectories (`run_to_failure`)
- **Scale**: 265,256 cycle observations across 4 sub-datasets (FD001–FD004), 26 columns (43 MB)
- **Description**: Simulated run-to-failure trajectories of turbofan engines under varying altitude, speed, and sea-level ambient conditions. Covers High-Pressure Compressor (HPC) and fan degradation.

#### Feature & Sensor Catalogue

| Feature Name | Physical Meaning | Unit | Modality / Category |
|--------------|------------------|------|---------------------|
| `unit_id` | Engine unit ID | Integer | Identifier |
| `cycle` | Completed flight operational cycle | Cycles | Temporal Identifier |
| `operating_setting_1` | Altitude / Mach number setting | Standardized | Operating Context |
| `operating_setting_2` | Throttle Resolver Angle (TRA) setting | Standardized | Operating Context |
| `operating_setting_3` | Sea-level ambient pressure setting | Standardized | Operating Context |
| `sensor_1` (`T2`) | Total temperature at fan inlet | $^\circ\text{R}$ | Thermal |
| `sensor_2` (`T24`) | Total temperature at LPC outlet | $^\circ\text{R}$ | Thermal |
| `sensor_3` (`T30`) | Total temperature at HPC outlet | $^\circ\text{R}$ | Thermal |
| `sensor_4` (`T50`) | Total temperature at LPT outlet | $^\circ\text{R}$ | Thermal |
| `sensor_5` (`P2`) | Pressure at fan inlet | $\text{psia}$ | Fluid Pressure |
| `sensor_6` (`P15`) | Total pressure in bypass duct | $\text{psia}$ | Fluid Pressure |
| `sensor_7` (`P30`) | Total pressure at HPC outlet | $\text{psia}$ | Fluid Pressure |
| `sensor_8` (`Nf`) | Physical fan speed | $\text{rpm}$ | Kinematic |
| `sensor_9` (`Nc`) | Physical core speed | $\text{rpm}$ | Kinematic |
| `sensor_10` (`epr`) | Engine pressure ratio ($P_{50}/P_2$) | Ratio | Fluid / Process |
| `sensor_11` (`Ps30`) | Static pressure at HPC outlet | $\text{psia}$ | Fluid Pressure |
| `sensor_12` (`phi`) | Ratio of fuel flow to $Ps_{30}$ | $\text{pps/psi}$ | Process / Fluid |
| `sensor_13` (`NRf`) | Corrected fan speed | $\text{rpm}$ | Kinematic |
| `sensor_14` (`NRc`) | Corrected core speed | $\text{rpm}$ | Kinematic |
| `sensor_15` (`BPR`) | Bypass Ratio | Ratio | Process |
| `sensor_16` (`farB`) | Burner fuel-air ratio | Ratio | Chemical / Process |
| `sensor_17` (`htBleed`) | Bleed Enthalpy | $\text{BTU/lb}$ | Thermal |
| `sensor_18` (`Nf_dmd`) | Demanded fan speed | $\text{rpm}$ | Control Signal |
| `sensor_19` (`PCNfR_dmd`) | Demanded corrected fan speed | $\text{rpm}$ | Control Signal |
| `sensor_20` (`W31`) | HPT coolant bleed flow | $\text{lbm/s}$ | Fluid Flow |
| `sensor_21` (`W32`) | LPT coolant bleed flow | $\text{lbm/s}$ | Fluid Flow |
| `rul_cycles_label` | Target Remaining Useful Life | Cycles | Outcome Label (Quarantined) |

- **Prescriptive Utility**: Benchmark for remaining-useful-life (RUL) modeling, derating flight load schedules (`ACT_OP_DERATE_LOAD`), and scheduling depot overhaul (`ACT_MAINT_OVERHAUL_MODULE`).

---

### 3. Smart Manufacturing Maintenance Dataset (`smart_maintenance`)
- **Domain**: Industrial Motor Fleet & Production Line Equipment
- **Processing Type**: Dual Variant (Static Tabular: 1,430 rows; Time Series: 100,000 rows) (7.1 MB)
- **Description**: Synthesizes multi-sensor physical telemetry combined with operational economic parameters (downtime cost, technician availability, inspection duration).

#### Feature & Sensor Catalogue (Combined Static & Timeseries)

| Feature Name | Physical Meaning | Unit | Modality / Category |
|--------------|------------------|------|---------------------|
| `timestamp` / `Machine_ID` | Time index or machine serial key | String | Identifier |
| `Temp_C` / `temperature` | Motor casing temperature | $^\circ\text{C}$ | Thermal |
| `Vibration_mm_s` / `vibration` | RMS vibration velocity | $\text{mm/s}$ | Mechanical / Vibration |
| `Pressure_Bar` / `pressure` | Supply line pressure | $\text{bar}$ | Fluid Pressure |
| `Acoustic_dB` | Ultrasonic noise emission | $\text{dB}$ | Acoustic |
| `humidity` | Ambient relative humidity | $\%$ | Operating Context |
| `energy_consumption` | Power consumption | $\text{kWh}$ | Electrical |
| `Inspection_Duration_min` | Duration of physical inspection | $\text{min}$ | Economic / Business |
| `Downtime_Cost_USD` | Downtime cost rate per hour | $\text{USD}$ | Economic / Business |
| `Technician_Availability_pct` | Available technician capacity | $\%$ | Economic / Business |
| `Failure_Prob` / `downtime_risk` | Modeled failure probability | $0.0 - 1.0$ | Health Degradation |
| `Maintenance_Priority` / `failure_type` | Target priority / Failure classification | Categorical | Outcome Label |

- **Prescriptive Utility**: Crucial for evaluating economic-aware action ranking, balancing cost of downtime vs immediate technician dispatch (`ACT_DISPATCH_TECHNICIAN`).

---

### 4. Intelligent Manufacturing 6G / IIoT Dataset (`iiot_6g`)
- **Domain**: Connected Smart Factory (Industry 4.0 / 6G Edge Networks)
- **Processing Type**: Low-Frequency Time Series (`low_frequency_time_series`)
- **Scale**: 100,000 observations, 13 features (19 MB)
- **Description**: Models cyber-physical manufacturing lines where physical machine health interacts with edge network latency, packet loss, and energy constraints.

#### Feature & Sensor Catalogue

| Feature Name | Physical Meaning | Unit | Modality / Category |
|--------------|------------------|------|---------------------|
| `Timestamp` | Time index | String | Identifier |
| `Machine_ID` | Machine node ID | String | Identifier |
| `Operation_Mode` | Current operating regime (High Speed, Eco, Heavy Duty) | Categorical | Operating Context |
| `Temperature_C` | Bearing / Component temperature | $^\circ\text{C}$ | Thermal |
| `Vibration_Hz` | Dominant vibration frequency | $\text{Hz}$ | Mechanical |
| `Power_Consumption_kW` | Instantaneous electrical power draw | $\text{kW}$ | Electrical |
| `Network_Latency_ms` | Edge control communication latency | $\text{ms}$ | Cyber-Physical Context |
| `Packet_Loss_%` | Wireless telemetry loss rate | $\%$ | Cyber-Physical Context |
| `Quality_Control_Defect_Rate_%` | Inline part defect rate | $\%$ | Quality / Process |
| `Production_Speed_units_per_hr` | Line throughput speed | $\text{units/hr}$ | Operating Context |
| `Predictive_Maintenance_Score` | Automated health score | $0.0 - 100.0$ | Health Degradation |
| `Error_Rate_%` | Edge controller error rate | $\%$ | Process |
| `Efficiency_Status` | Plant efficiency label | Categorical | Outcome Label |

- **Prescriptive Utility**: Evaluates edge-network prescriptive actions, such as throttling line throughput (`ACT_ADJ_THROUGHPUT`) or switching edge compute nodes during telemetry dropouts.

---

### 5. NASA Milling Tool Wear Dataset (`nasa_milling`)
- **Domain**: CNC Milling Cutter / High-Speed Machining
- **Processing Type**: Run-to-Failure Signal Trajectories (`run_to_failure_signal`)
- **Scale**: 1,002 machining runs across 16 different cut experiments (14 MB MATLAB archive)
- **Description**: Laboratory tool wear experiments capturing Acoustic Emission (AE) and spindle AC/DC current signals across repeated passes of a 6-insert milling cutter under varying depth of cut and feed rates.

#### Feature & Sensor Catalogue

| Feature Name | Physical Meaning | Unit | Modality / Category |
|--------------|------------------|------|---------------------|
| `case` | Experiment run index | Integer | Identifier |
| `run` | Cut pass number | Integer | Temporal Sequence |
| `VB` | Flank wear measurement | $\text{mm}$ | Health / Degradation (Target) |
| `DOC` | Depth of Cut setting | $\text{mm}$ | Operating Setting |
| `feed` | Feed rate per tooth | $\text{mm/rev}$ | Operating Setting |
| `material` | Workpiece material (Cast Iron / Steel) | Categorical | Operating Context |
| `smcAC` | Spindle Motor Current (AC component) | $\text{A}$ | Electrical |
| `smcDC` | Spindle Motor Current (DC component) | $\text{A}$ | Electrical |
| `tibAC` | Table Instrument Current (AC component) | $\text{A}$ | Electrical |
| `tibDC` | Table Instrument Current (DC component) | $\text{A}$ | Electrical |
| `ae_table` | Acoustic Emission signal from table sensor | $\text{V}$ | Acoustic Emission |
| `ae_spindle` | Acoustic Emission signal from spindle sensor | $\text{V}$ | Acoustic Emission |
| `vib_table` | Vibration acceleration from table sensor | $g$ | Mechanical / Vibration |
| `vib_spindle` | Vibration acceleration from spindle sensor | $g$ | Mechanical / Vibration |

- **Prescriptive Utility**: Provides true physical flank wear measurements ($\text{VB}$ in $\text{mm}$) to ground physical insert replacement triggers (`ACT_REPL_TOOL_INSERT`).

---

### 6. UCI MetroPT-3 Air Compressor Dataset (`metropt3`)
- **Domain**: Heavy Industrial Railway Air Production Unit (APU)
- **Processing Type**: 1 Hz Continuous Multivariate Time Series (`low_frequency_time_series`)
- **Scale**: 1,516,948 raw records, aggregated into 306,960 1-minute feature windows (208 MB)
- **Description**: Real-world telemetry from APU compressors mounted on APEX trains in Lisbon, Portugal. Captures pressure drops, oil temperatures, and motor current during real operational failures (e.g., air dryer leaks, intake filter clogging).

#### Feature & Sensor Catalogue

| Feature Name | Physical Meaning | Unit | Modality / Category |
|--------------|------------------|------|---------------------|
| `timestamp` | UTC Timestamp | Datetime | Identifier |
| `TP2` | Air pressure downstream of compressor | $\text{bar}$ | Fluid Pressure |
| `TP3` | Pneumatic panel line pressure | $\text{bar}$ | Fluid Pressure |
| `H1` | Cyclonic separator differential pressure | $\text{bar}$ | Fluid Pressure |
| `DV_pressure` | Air dryer valve pressure drop | $\text{bar}$ | Fluid Pressure |
| `Reservoirs` | Main reservoir pressure | $\text{bar}$ | Fluid Pressure |
| `Oil_temperature` | Compressor crankcase oil temperature | $^\circ\text{C}$ | Thermal |
| `Motor_current` | Motor electrical current | $\text{A}$ | Electrical |
| `COMP` | Digital compressor signal (ON/OFF) | $0/1$ | Control Context |
| `DV_eletric` | Air dryer valve electrical command | $0/1$ | Control Context |
| `Towers` | Air dryer tower switching state | $0/1$ | Control Context |
| `MPG` | Motor Protection Relay state | $0/1$ | Operating State |
| `LPS` | Low Pressure Switch signal | $0/1$ | Control Signal |
| `Pressure_switch` | Line pressure switch flag | $0/1$ | Control Signal |
| `Oil_level` | Crankcase oil level sensor | $0/1$ | Fluid Level |
| `Caudal_impulses` | Air flow pulse count | Count | Fluid Flow |

- **Prescriptive Utility**: Ideal for validating pneumatic system maintenance actions: cleaning dryer valves (`ACT_CLEAN_VALVE`), purging clogged filters (`ACT_PURGE_FILTER`), and topping up oil levels (`ACT_TOPUP_FLUID`).

---

### 7. UCI Hydraulic Systems Condition Monitoring Dataset (`uci_hydraulic`)
- **Domain**: Fluid Power & Industrial Hydraulics Test Rig
- **Processing Type**: Synchronized Multirate Cycle Signals (`multirate_cycle_signal`)
- **Scale**: 2,205 repeated 60-second load cycles, 21 physical raw sensor files (531 MB)
- **Description**: Telemetry from a hydraulic test rig equipped with multiple primary components undergoing progressive degraded states (Cooler efficiency, Valve switching degradation, Pump internal leakage, Accumulator pre-charge pressure loss).

#### Feature & Sensor Catalogue

| Sensor ID | Physical Measurement | Sampling Rate | Canonical Unit | Modality |
|-----------|----------------------|---------------|----------------|----------|
| `PS1` | Hydraulic Pressure at primary pump | 100 Hz | $\text{bar}$ | Fluid Pressure |
| `PS2` | Hydraulic Pressure upstream of main valve | 100 Hz | $\text{bar}$ | Fluid Pressure |
| `PS3` | Hydraulic Pressure downstream of main valve | 100 Hz | $\text{bar}$ | Fluid Pressure |
| `PS4` | Pressure at return line | 100 Hz | $\text{bar}$ | Fluid Pressure |
| `PS5` | Pressure at pressure reservoir | 100 Hz | $\text{bar}$ | Fluid Pressure |
| `PS6` | Pressure at accumulator line | 100 Hz | $\text{bar}$ | Fluid Pressure |
| `EPS1` | Electric motor power draw | 100 Hz | $\text{W} \rightarrow \text{kW}$ | Electrical |
| `FS1` | Volume flow rate (primary circuit) | 10 Hz | $\text{l/min}$ | Fluid Flow |
| `FS2` | Volume flow rate (secondary cooling circuit) | 10 Hz | $\text{l/min}$ | Fluid Flow |
| `TS1` | Temperature at oil reservoir | 1 Hz | $^\circ\text{C}$ | Thermal |
| `TS2` | Temperature at main valve | 1 Hz | $^\circ\text{C}$ | Thermal |
| `TS3` | Temperature at cooling circuit inlet | 1 Hz | $^\circ\text{C}$ | Thermal |
| `TS4` | Temperature at cooling circuit outlet | 1 Hz | $^\circ\text{C}$ | Thermal |
| `VS1` | Vibration velocity at pump casing | 1 Hz | $\text{mm/s}$ | Mechanical |
| `CE` | Virtual cooling efficiency | 1 Hz | $\%$ | Process / Thermal |
| `CP` | Virtual cooling power | 1 Hz | $\text{kW}$ | Process / Thermal |
| `SE` | Virtual system efficiency factor | 1 Hz | $\%$ | Process |
| `cooler_condition` | Cooler degradation state (3% / 20% / 100%) | Per-Cycle | Target Label | Quarantined Target |
| `valve_condition` | Valve switching degradation (100% to 73%) | Per-Cycle | Target Label | Quarantined Target |
| `pump_leakage` | Pump internal leakage level (None / Small / Severe) | Per-Cycle | Target Label | Quarantined Target |
| `accumulator_condition` | Accumulator nitrogen pressure (90 bar to 130 bar) | Per-Cycle | Target Label | Quarantined Target |

- **Prescriptive Utility**: Rich multi-target breakdown allows testing discrete component actions (`ACT_RECHARGE_ACCUMULATOR`, `ACT_REPL_HYD_PUMP`, `ACT_CLEAN_HEAT_EXCHANGER`, `ACT_REPL_CONTROL_VALVE`).

---

### 8. Tennessee Eastman Process (TEP) Dataset (`tennessee_eastman`)
- **Domain**: Chemical Manufacturing & Process Plant Simulation
- **Processing Type**: Multivariate Process Time Series (`multivariate_process_time_series`)
- **Scale**: 5,730,000 observations across 4 simulation files (550 MB)
- **Description**: Standard benchmark based on an actual chemical process containing 5 major unit operations: Reactor, Condenser, Compressor, Stripper, and Separator. Models 21 distinct process fault injection scenarios.

#### Feature & Sensor Catalogue Summary (52 Variables)

| Variable Range | Description | Typical Physical Quantities | Modality |
|----------------|-------------|-----------------------------|----------|
| `xmeas_1` – `xmeas_41` | Continuous Process Measurements (41 channels) | Reactor Pressure, Reactor Temp, Liquid Levels, Feed Flow Rates, Stripper Pressure, Compressor Work, Mole Fractions (A, B, C, D, E, F) | Chemical / Fluid / Thermal / Electrical |
| `xmv_1` – `xmv_11` | Manipulated Control Variables (11 valves) | D Feed Valve, E Feed Valve, A Feed Valve, Total Feed Valve, Compressor Recycle Valve, Purge Valve, Separator Liquid Outlet, Stripper Liquid Outlet, Reactor Cooling Water Valve, Condenser Cooling Water Valve, Agitator Speed | Control Actuators |
| `faultNumber` | Injected fault scenario ($0 = \text{Normal}, 1\text{--}21 = \text{Faults}$) | Discrete Fault ID | Outcome Label (Quarantined) |

- **Prescriptive Utility**: Tests high-level process control actions (`ACT_ADJ_COOLANT_FLOW`, `ACT_ADJ_FEED_RATE`, `ACT_PURGE_FILTER`, `ACT_OP_EMERGENCY_STOP`).

---

### 9. NASA IMS Bearings Dataset (`nasa_ims`)
- **Domain**: Rotary Drive / Rolling Element Bearings
- **Processing Type**: High-Frequency Waveform Snapshots (`high_frequency_signal`)
- **Scale**: 3,140 1-second 20 kHz vibration snapshots (2,048,000 points per snapshot) (2.8 GB)
- **Description**: Run-to-failure experiments on 4 bearings mounted on a shaft rotating at 2,000 rpm under 6,000 lbs radial load. Snapshots capture the transition from healthy baseline to micro-spalling, outer race defect, and inner race destruction.

#### Feature & Sensor Catalogue

| Feature / Extracted Metric | Physical Meaning | Canonical Unit | Modality |
|----------------------------|------------------|----------------|----------|
| `accel_ch1` – `accel_ch8` | High-speed accelerometer channels | $g$ (acceleration) | Vibration |
| `rms` *(derived)* | Root Mean Square energy | $g$ | Vibration Feature |
| `kurtosis` *(derived)* | Fourth standardized moment (impulsiveness indicator) | Dimensionless | Vibration Feature |
| `crest_factor` *(derived)* | Peak amplitude over RMS | Dimensionless | Vibration Feature |
| `spectral_entropy` *(derived)*| Shannon entropy of normalized FFT power spectrum | Dimensionless | Spectral Feature |
| `dominant_freq` *(derived)* | Frequency of maximum FFT power | $\text{Hz}$ | Spectral Feature |
| `end_of_test_failure` | End-of-test destruction report (Bearing 3 outer race defect) | Text Description | Outcome Label |

- **Prescriptive Utility**: Ground truth for mechanical bearing actions: relubrication (`ACT_GREASE_BEARING`), vibration monitoring cadence increase, and bearing replacement (`ACT_REPL_BEARING_ASSEMBLY`).

---

### 10. Semiconductor Metal Etch Dataset (`metal_etch`)
- **Domain**: Semiconductor Wafer Manufacturing / Plasma Etching
- **Processing Type**: Static Tabular (`static_tabular`)
- **Scale**: 10,663 observations, 22 columns (1.4 MB)
- **Description**: Features process variables collected during plasma etching of semiconductor wafers. Captures subtle chamber drift and process anomalies.

#### Feature & Sensor Catalogue

| Feature Name | Physical Meaning | Category |
|--------------|------------------|----------|
| `Feature_1` – `Feature_21` | Anonymized physical sensor telemetry (Chamber pressure, RF power, gas flows, temperature) | Process / Thermal / Electrical |
| `Target` | Binary wafer etch quality / chamber health flag | Outcome Label (Quarantined) |

- **Prescriptive Utility**: Validates model stability on anonymized feature sets and automated chamber recalibration recommendations (`ACT_RECALIBRATE_SENSOR`).

---

## 4. Summary Matrix of Dataset Coverage

To quickly assess how these datasets cover the full prescriptive maintenance problem, consult the summary matrix below:

| Dataset ID | Asset Archetype | Key Physical Modalities | Extracted Signals / Features | Prescriptive Actions Triggered |
|------------|-----------------|------------------------|-----------------------------|--------------------------------|
| `ai4i_2020` | CNC Milling Machine | Thermal, Kinematic, Mechanical | Temp ($^\circ\text{C}$), Torque ($\text{Nm}$), Wear ($\text{min}$) | `ACT_REPL_TOOL_INSERT`, `ACT_OP_DERATE_LOAD` |
| `cmapss` | Turbofan Jet Engine | Thermal, Fluid Pressure, Kinematic | Fan/Core Speeds, Pressures, Bleed Flows | `ACT_OP_DERATE_LOAD`, `ACT_MAINT_OVERHAUL_MODULE` |
| `smart_maintenance` | Factory Motor Fleet | Thermal, Vibration, Acoustic, Economic | Vibration, Temp, Downtime Cost ($/hr) | `ACT_DISPATCH_TECHNICIAN`, `ACT_OP_CONTROLLED_SHUTDOWN` |
| `iiot_6g` | Connected Smart Factory | Cyber-Physical, Electrical, Process | Latency ($\text{ms}$), Packet Loss ($\%$), Power ($\text{kW}$) | `ACT_ADJ_THROUGHPUT`, `ACT_RECALIBRATE_SENSOR` |
| `nasa_milling` | CNC Metal Cutting Tool | Acoustic Emission, Electrical, Mechanical | Flank Wear ($\text{VB}$ $\text{mm}$), AE ($\text{V}$), Spindle Current | `ACT_REPL_TOOL_INSERT`, `ACT_INSP_TOOL_WEAR` |
| `metropt3` | Railway Air Compressor | Fluid Pressure, Thermal, Electrical | Line Pressures ($\text{bar}$), Oil Temp, Motor Current | `ACT_CLEAN_VALVE`, `ACT_PURGE_FILTER`, `ACT_TOPUP_FLUID` |
| `uci_hydraulic` | Hydraulic Test Rig | Fluid Pressure, Flow Rate, Thermal, Power | Pressures ($\text{bar}$), Flow ($\text{l/min}$), Temp ($^\circ\text{C}$) | `ACT_RECHARGE_ACCUMULATOR`, `ACT_REPL_HYD_PUMP` |
| `tennessee_eastman` | Chemical Process Plant | Chemical, Fluid, Thermal, Control | 41 Process Variables, 11 Valve Positions | `ACT_ADJ_COOLANT_FLOW`, `ACT_OP_EMERGENCY_STOP` |
| `nasa_ims` | High-Speed Bearing Rig | High-Frequency Vibration | Acceleration ($g$), Kurtosis, Spectral Energy | `ACT_GREASE_BEARING`, `ACT_REPL_BEARING_ASSEMBLY` |
| `metal_etch` | Semiconductor Etcher | Process, Electrical, Thermal | 21 Process Telemetry Features | `ACT_RECALIBRATE_SENSOR`, `ACT_CLEAN_CHAMBER` |

---

## 5. Conclusion & Verification

By combining these **10 diverse industrial datasets**, our project ensures that the **LGBMRanker** and **LLM-as-a-Judge prescriptive maintenance agents** are evaluated across real-world factory complexities:
1. No single component or failure mode dominates training.
2. The physics-based candidate generator handles diverse units (Kelvin, $^\circ\text{C}$, bar, Pa, $g$, $\text{mm/s}$) via normalized ontology mapping.
3. Every proposed maintenance intervention is physically grounded in empirical sensor evidence.
