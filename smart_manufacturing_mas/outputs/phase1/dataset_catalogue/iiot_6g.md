# Intelligent Manufacturing 6G

## Identity

- Dataset ID: `iiot_6g`
- Source: `Intelligent Manufacturing Dataset/manufacturing_6G_dataset.csv`
- Domain: smart factory / IIoT
- Asset type: machine fleet
- Component: unknown
- Processing type: `low_frequency_time_series`

## Data structure

- Records/files: 100000 records; 1 files
- Columns: 13
- Temporal structure: sequential timestamped rows

## Labels and outcomes

- Efficiency_Status: present

## Sensor/feature catalogue

| Feature | Physical meaning | Unit | Category |
|---|---|---|---|
| Timestamp | observation timestamp | UNKNOWN/UNVERIFIED | metadata/identifier |
| Machine_ID | machine identifier | UNKNOWN/UNVERIFIED | metadata/identifier |
| Operation_Mode | operating mode | UNKNOWN/UNVERIFIED | operating context |
| Temperature_C | temperature | C | thermal |
| Vibration_Hz | vibration measurement | Hz | mechanical |
| Power_Consumption_kW | power consumption | kW | electrical |
| Network_Latency_ms | network latency | ms | operating context |
| Packet_Loss_% | packet loss | % | operating context |
| Quality_Control_Defect_Rate_% | quality-control defect rate | % | process |
| Production_Speed_units_per_hr | production speed | units/hr | operating context |
| Predictive_Maintenance_Score | predictive maintenance score | UNKNOWN/UNVERIFIED | health/degradation |
| Error_Rate_% | error rate | % | process |
| Efficiency_Status | efficiency status | UNKNOWN/UNVERIFIED | label/target |

## Preparation notes

- Original source columns are retained. Derived columns use the __ suffix.
