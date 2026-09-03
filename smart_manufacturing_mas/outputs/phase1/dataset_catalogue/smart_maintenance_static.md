# Smart Manufacturing Maintenance - static variant

## Identity

- Dataset ID: `smart_maintenance_static`
- Source: `Smart Manufacturing Maintenance Dataset/smart_maintenance_dataset.csv`
- Domain: factory maintenance
- Asset type: machine fleet
- Component: unknown
- Processing type: `static_tabular`

## Data structure

- Records/files: 1430 records; 1 files
- Columns: 10
- Temporal structure: independent tabular rows; no verified temporal key

## Labels and outcomes

- Maintenance_Priority: present

## Sensor/feature catalogue

| Feature | Physical meaning | Unit | Category |
|---|---|---|---|
| Machine_ID | machine identifier | UNKNOWN/UNVERIFIED | metadata/identifier |
| Temp_C | temperature | C | thermal |
| Vibration_mm_s | vibration velocity | mm/s | mechanical |
| Pressure_Bar | pressure | bar | hydraulic/fluid |
| Acoustic_dB | acoustic level | dB | acoustic |
| Inspection_Duration_min | inspection duration | min | business/economic |
| Downtime_Cost_USD | downtime cost | USD | business/economic |
| Technician_Availability_pct | technician availability | % | business/economic |
| Failure_Prob | failure probability | UNKNOWN/UNVERIFIED | health/degradation |
| Maintenance_Priority | maintenance priority label | UNKNOWN/UNVERIFIED | label/target |

## Preparation notes

- Original source columns are retained. Derived columns use the __ suffix.
