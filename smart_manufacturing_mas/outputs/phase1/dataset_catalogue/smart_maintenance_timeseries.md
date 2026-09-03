# Smart Manufacturing Maintenance - timestamped variant

## Identity

- Dataset ID: `smart_maintenance_timeseries`
- Source: `Smart Manufacturing Maintenance Dataset/smart_manufacturing_data.csv`
- Domain: factory maintenance
- Asset type: machine fleet
- Component: unknown
- Processing type: `low_frequency_time_series`

## Data structure

- Records/files: 100000 records; 1 files
- Columns: 13
- Temporal structure: sequential timestamped rows

## Labels and outcomes

- anomaly_flag: present
- failure_type: present
- maintenance_required: present

## Sensor/feature catalogue

| Feature | Physical meaning | Unit | Category |
|---|---|---|---|
| timestamp | observation timestamp | UNKNOWN/UNVERIFIED | metadata/identifier |
| machine_id | machine identifier | UNKNOWN/UNVERIFIED | metadata/identifier |
| temperature | temperature | UNKNOWN/UNVERIFIED | thermal |
| vibration | vibration | UNKNOWN/UNVERIFIED | mechanical |
| humidity | humidity | UNKNOWN/UNVERIFIED | operating context |
| pressure | pressure | UNKNOWN/UNVERIFIED | hydraulic/fluid |
| energy_consumption | energy consumption | UNKNOWN/UNVERIFIED | electrical |
| machine_status | machine status | UNKNOWN/UNVERIFIED | operating context |
| anomaly_flag | anomaly label | UNKNOWN/UNVERIFIED | label/target |
| predicted_remaining_life | predicted remaining life | UNKNOWN/UNVERIFIED | health/degradation |
| failure_type | failure category | UNKNOWN/UNVERIFIED | label/target |
| downtime_risk | downtime risk | UNKNOWN/UNVERIFIED | business/economic |
| maintenance_required | maintenance-required label | UNKNOWN/UNVERIFIED | label/target |

## Preparation notes

- Original source columns are retained. Derived columns use the __ suffix.
