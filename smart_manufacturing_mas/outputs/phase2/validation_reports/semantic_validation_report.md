# Phase 2 Semantic Normalization Validation Report

Generated: 2026-09-03T17:13:36.485819+00:00

## Summary Metrics
- Total Features Catalogued: **730** across **11** active dataset streams.
- Target Outcome Columns Isolated: **20**.
- High Confidence Mappings: **310**.
- Medium Confidence Mappings: **152**.
- Low Confidence (Conservative Unknowns): **268**.

## Dataset-Level Validation

| Dataset ID | Total Records | Total Columns | State Features | Targets Isolated | Converted Units | Modalities Present |
|---|---:|---:|---:|---:|---:|---|
| `ai4i_2020` | 10,000 | 16 | 6 | 6 | 2 | health_degradation, kinematic, mechanical, process_operating... |
| `cmapss` | 265,256 | 115 | 108 | 1 | 0 | process_operating |
| `smart_maintenance_static` | 1,430 | 12 | 8 | 1 | 0 | acoustic, fluid, health_degradation, mechanical... |
| `smart_maintenance_timeseries` | 100,000 | 47 | 40 | 3 | 0 | electrical, fluid, health_degradation, mechanical... |
| `iiot_6g` | 100,000 | 51 | 46 | 1 | 0 | electrical, health_degradation, kinematic, mechanical... |
| `metal_etch` | 10,663 | 24 | 21 | 1 | 0 | unknown |
| `metropt3` | 306,960 | 80 | 75 | 0 | 0 | electrical, fluid, process_operating, thermal |
| `uci_hydraulic` | 2,205 | 126 | 119 | 5 | 7 | electrical, fluid, mechanical, process_operating... |
| `tennessee_eastman` | 5,730,000 | 139 | 132 | 1 | 0 | process_operating |
| `nasa_ims` | 3,140 | 105 | 96 | 1 | 0 | mechanical |
| `nasa_milling` | 1,002 | 15 | 10 | 0 | 0 | mechanical |

## Invariant Validation Assertions
- **Zero Semantic Hallucination**: Verified that 100% of anonymous features (Metal Etch, C-MAPSS, TEP) retain `unknown` physical quantity.
- **Strict Target Isolation**: Verified that 0 target labels exist within machine state representations.
- **Provenance Preservation**: All canonical machine state tables preserve original column names and unique observation identifiers.
- **Safe Unit Normalization**: Linear conversions applied strictly where documentary units were verified (Kelvin -> Celsius, Watt -> Kilowatt). Original values remain untouched.

