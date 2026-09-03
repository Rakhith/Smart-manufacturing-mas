# NASA Milling Tool Wear

## Identity

- Dataset ID: `nasa_milling`
- Source: `NASA Milling Tool Wear/3. Milling/mill.zip`
- Domain: CNC metal cutting
- Asset type: milling machine
- Component: cutting tool
- Processing type: `run_to_failure_signal`

## Data structure

- Records/files: 1002 records; 1 files
- Columns: 15
- Temporal structure: MATLAB archive; numeric signal arrays flattened conservatively

## Labels and outcomes

- tool_wear_available: present
- maintenance_actions: not observed
- maintenance_outcomes: not observed

## Sensor/feature catalogue

| Feature | Physical meaning | Unit | Category |
|---|---|---|---|
| numeric MAT arrays | UNKNOWN/UNVERIFIED pending MATLAB release field documentation | UNKNOWN/UNVERIFIED | UNKNOWN/UNVERIFIED |

## Preparation notes

- No physical field name is assumed from the nested MATLAB structure. Signal sampling rate is not assigned without a supported source.
- Prepared observations retain source_array as provenance.
