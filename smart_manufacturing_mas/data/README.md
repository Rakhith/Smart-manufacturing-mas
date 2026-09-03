# 📦 Dataset Registry — Smart Manufacturing Prescriptive Maintenance

This folder holds all datasets for the **LGBMRanker-based prescriptive maintenance model**.
Together these 9 active datasets cover thermal, kinematic, fluid hydraulics, electrical,
pneumatic, and chemical telemetry across 9 distinct industrial archetypes.

---

## ✅ Active Installed Datasets (9 Datasets — All < 3 GB)

| # | Folder / File | Industry | Size | Status |
|---|---------------|----------|------|--------|
| 1 | `ai4i2020.csv` | CNC Machining / Tooling | 512 KB | ✅ Ready |
| 2 | `CMAPSSData/` | Aerospace / Jet Engines | 43 MB | ✅ Ready |
| 3 | `Smart Manufacturing Maintenance Dataset/` | Factory Motor Fleet | 7.1 MB | ✅ Ready |
| 4 | `Intelligent Manufacturing Dataset/` | Connected Smart Factory (IIoT/6G) | 19 MB | ✅ Ready |
| 5 | `NASA Milling Tool Wear/` | CNC Metal Cutting & Acoustic Emission | 14 MB | ✅ Ready |
| 6 | `UCI MetroPT-3/` | Heavy Industrial Pneumatics | 208 MB | ✅ Ready |
| 7 | `UCI Hydraulic Systems/` | Fluid Power & Industrial Hydraulics | 531 MB | ✅ Ready |
| 8 | `Tennessee Eastman Process/` | Chemical & Process Plants | 550 MB | ✅ Ready |
| 9 | `NASA IMS Bearings/` | Rotary Drives / Run-to-Failure | 2.8 GB | ✅ Ready (1st & 2nd test sets) |

---

## 🗑️ Dropped / Deprecated

| Dataset / File | Reason |
|----------------|--------|
| `Wind Turbine SCADA` | Dropped (>11 GB bandwidth / size requirement) |
| `_deprecated/digital_manufacturing_dataset.csv` | Non-engineering ERP sales/supply-chain columns |

---

## 🗂️ Key Signals by Dataset

| Dataset | Key Signals |
|---------|------------|
| AI4I 2020 | Air/Process Temp, Rotational Speed, Torque, Tool Wear, TWF/HDF/PWF/OSF/RNF failure modes |
| C-MAPSS | 21 sensor channels (s1–s21), 3 operating settings, RUL cycles |
| Smart Maintenance | Temp, Vibration, Pressure, Acoustic dB, Downtime Cost, Failure Probability |
| IIoT / 6G | Network Latency, Packet Loss, Power kW, Production Speed, Defect Rate |
| NASA Milling | Flank Wear VB (mm), Depth of Cut, Feed Rate, Acoustic Emission, Spindle AC/DC Current |
| UCI MetroPT-3 | TP2/TP3 Reservoir & Line Pressure, H1, Motor Current, Oil Temp, Air Dryer Valves |
| UCI Hydraulics | PS1–6 Pressure, FS1–2 Volume Flow, EPS1 Motor Power, TS1–4 Temp, VS1 Vibration |
| Tennessee Eastman | 52 variables: Reactor Temp, Stripper Pressure, Liquid Levels, Mole Fractions |
| NASA IMS Bearings | 4–8 accelerometer vibration channels over run-to-destruction test series |

