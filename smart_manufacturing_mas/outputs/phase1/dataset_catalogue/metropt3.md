# UCI MetroPT-3

## Identity

- Dataset ID: `metropt3`
- Source: `UCI MetroPT-3/MetroPT3(AirCompressor).csv`
- Domain: railway pneumatics
- Asset type: air production unit
- Component: compressor
- Processing type: `low_frequency_time_series`

## Data structure

- Records/files: 1516948 records; 1 files
- Columns: 17
- Temporal structure: 1 Hz compressor multivariate time series

## Labels and outcomes

- failure_reports_documented: present
- row_level_failure_label: not observed
- maintenance_actions: not observed
- maintenance_outcomes: not observed

## Sensor/feature catalogue

| Feature | Physical meaning | Unit | Category |
|---|---|---|---|
| timestamp | timestamp | UNKNOWN/UNVERIFIED | metadata/identifier |
| TP2 | compressor pressure | bar | hydraulic/fluid |
| TP3 | pneumatic panel pressure | bar | hydraulic/fluid |
| H1 | cyclonic separator pressure drop | bar | hydraulic/fluid |
| DV_pressure | air-dryer discharge pressure drop | bar | hydraulic/fluid |
| Reservoirs | downstream reservoir pressure | bar | hydraulic/fluid |
| Oil_temperature | compressor oil temperature | C | thermal |
| Motor_current | single motor-phase current | A | electrical |
| COMP | digital compressor/air-dryer/flow signal | binary/count; see MetroPT-3 documentation | operating context |
| DV_eletric | digital compressor/air-dryer/flow signal | binary/count; see MetroPT-3 documentation | operating context |
| Towers | digital compressor/air-dryer/flow signal | binary/count; see MetroPT-3 documentation | operating context |
| MPG | digital compressor/air-dryer/flow signal | binary/count; see MetroPT-3 documentation | operating context |
| LPS | digital compressor/air-dryer/flow signal | binary/count; see MetroPT-3 documentation | operating context |
| Pressure_switch | digital compressor/air-dryer/flow signal | binary/count; see MetroPT-3 documentation | operating context |
| Oil_level | digital compressor/air-dryer/flow signal | binary/count; see MetroPT-3 documentation | operating context |
| Caudal_impulses | digital compressor/air-dryer/flow signal | binary/count; see MetroPT-3 documentation | operating context |

## Preparation notes

- One-minute windows are used to keep Phase 1 artifacts tractable; raw 1 Hz records remain untouched.
- Failure reports in the PDF are not expanded into row-level labels in this phase.
