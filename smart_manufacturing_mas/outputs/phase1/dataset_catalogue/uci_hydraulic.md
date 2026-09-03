# UCI Hydraulic Systems

## Identity

- Dataset ID: `uci_hydraulic`
- Source: `UCI Hydraulic Systems`
- Domain: industrial hydraulics
- Asset type: hydraulic test rig
- Component: cooler/valve/pump/accumulator
- Processing type: `multirate_cycle_signal`

## Data structure

- Records/files: 2205 records; 21 files
- Columns: 126
- Temporal structure: 2205 synchronized 60-second multirate load cycles

## Labels and outcomes

- cooler_condition: present
- valve_condition: present
- pump_leakage: present
- accumulator_condition: present
- maintenance_actions: not observed
- maintenance_outcomes: not observed

## Sensor/feature catalogue

| Feature | Physical meaning | Unit | Category |
|---|---|---|---|
| PS1 | pressure | bar | hydraulic/fluid |
| PS2 | pressure | bar | hydraulic/fluid |
| PS3 | pressure | bar | hydraulic/fluid |
| PS4 | pressure | bar | hydraulic/fluid |
| PS5 | pressure | bar | hydraulic/fluid |
| PS6 | pressure | bar | hydraulic/fluid |
| EPS1 | motor power | W | electrical |
| FS1 | volume flow | l/min | hydraulic/fluid |
| FS2 | volume flow | l/min | hydraulic/fluid |
| TS1 | temperature | C | thermal |
| TS2 | temperature | C | thermal |
| TS3 | temperature | C | thermal |
| TS4 | temperature | C | thermal |
| VS1 | vibration | mm/s | mechanical |
| CE | cooling efficiency (virtual) | % | process |
| CP | cooling power (virtual) | kW | process |
| SE | efficiency factor | % | process |
| cooler_condition_pct_label | cooler condition pct | see UCI documentation | label/target |
| valve_condition_pct_label | valve condition pct | see UCI documentation | label/target |
| pump_leakage_label | pump leakage | see UCI documentation | label/target |
| accumulator_pressure_bar_label | accumulator pressure bar | see UCI documentation | label/target |
| stable_flag | stable flag | see UCI documentation | label/target |

## Preparation notes

- Raw sensor matrices are preserved in data/. Prepared output contains per-cycle summaries only.
