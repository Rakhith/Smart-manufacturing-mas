# NASA C-MAPSS

## Identity

- Dataset ID: `cmapss`
- Source: `CMAPSSData`
- Domain: aerospace
- Asset type: turbofan engine
- Component: HPC/fan
- Processing type: `run_to_failure`

## Data structure

- Records/files: 265256 records; 14 files
- Columns: 26
- Temporal structure: per-engine sequential run-to-failure trajectories

## Labels and outcomes

- RUL: present
- fault_regime_metadata: present
- maintenance_actions: not observed
- maintenance_outcomes: not observed

## Sensor/feature catalogue

| Feature | Physical meaning | Unit | Category |
|---|---|---|---|
| unit_id | engine unit identifier | UNKNOWN/UNVERIFIED | metadata/identifier |
| cycle | operating cycle | cycles | metadata/identifier |
| operating_setting_1 | operational setting 1 | UNKNOWN/UNVERIFIED | operating context |
| operating_setting_2 | operational setting 2 | UNKNOWN/UNVERIFIED | operating context |
| operating_setting_3 | operational setting 3 | UNKNOWN/UNVERIFIED | operating context |
| sensor_1 | NASA C-MAPSS sensor measurement 1 | UNKNOWN/UNVERIFIED | process |
| sensor_2 | NASA C-MAPSS sensor measurement 2 | UNKNOWN/UNVERIFIED | process |
| sensor_3 | NASA C-MAPSS sensor measurement 3 | UNKNOWN/UNVERIFIED | process |
| sensor_4 | NASA C-MAPSS sensor measurement 4 | UNKNOWN/UNVERIFIED | process |
| sensor_5 | NASA C-MAPSS sensor measurement 5 | UNKNOWN/UNVERIFIED | process |
| sensor_6 | NASA C-MAPSS sensor measurement 6 | UNKNOWN/UNVERIFIED | process |
| sensor_7 | NASA C-MAPSS sensor measurement 7 | UNKNOWN/UNVERIFIED | process |
| sensor_8 | NASA C-MAPSS sensor measurement 8 | UNKNOWN/UNVERIFIED | process |
| sensor_9 | NASA C-MAPSS sensor measurement 9 | UNKNOWN/UNVERIFIED | process |
| sensor_10 | NASA C-MAPSS sensor measurement 10 | UNKNOWN/UNVERIFIED | process |
| sensor_11 | NASA C-MAPSS sensor measurement 11 | UNKNOWN/UNVERIFIED | process |
| sensor_12 | NASA C-MAPSS sensor measurement 12 | UNKNOWN/UNVERIFIED | process |
| sensor_13 | NASA C-MAPSS sensor measurement 13 | UNKNOWN/UNVERIFIED | process |
| sensor_14 | NASA C-MAPSS sensor measurement 14 | UNKNOWN/UNVERIFIED | process |
| sensor_15 | NASA C-MAPSS sensor measurement 15 | UNKNOWN/UNVERIFIED | process |
| sensor_16 | NASA C-MAPSS sensor measurement 16 | UNKNOWN/UNVERIFIED | process |
| sensor_17 | NASA C-MAPSS sensor measurement 17 | UNKNOWN/UNVERIFIED | process |
| sensor_18 | NASA C-MAPSS sensor measurement 18 | UNKNOWN/UNVERIFIED | process |
| sensor_19 | NASA C-MAPSS sensor measurement 19 | UNKNOWN/UNVERIFIED | process |
| sensor_20 | NASA C-MAPSS sensor measurement 20 | UNKNOWN/UNVERIFIED | process |
| sensor_21 | NASA C-MAPSS sensor measurement 21 | UNKNOWN/UNVERIFIED | process |
| rul_cycles_label | remaining useful life | cycles | label/target |

## Preparation notes

- FD001/FD002 contain HPC degradation; FD003/FD004 include HPC and fan degradation according to the included readme.
