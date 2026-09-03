# NASA IMS Bearings

## Identity

- Dataset ID: `nasa_ims`
- Source: `NASA IMS Bearings`
- Domain: rotating machinery
- Asset type: bearing test rig
- Component: rolling bearings
- Processing type: `high_frequency_signal`

## Data structure

- Records/files: 3140 records; 3142 files
- Columns: 105
- Temporal structure: one-second high-frequency snapshots at 20 kHz, collected roughly every 10 minutes

## Labels and outcomes

- end_of_test_failure_description: present
- per_snapshot_failure_label: not observed
- maintenance_actions: not observed
- maintenance_outcomes: not observed

## Sensor/feature catalogue

| Feature | Physical meaning | Unit | Category |
|---|---|---|---|
| accelerometer_channels | bearing housing acceleration | UNKNOWN/UNVERIFIED | mechanical |

## Preparation notes

- Known failure descriptions apply only at test end, not as per-snapshot labels.
- 0 waveform files were unreadable.
