"""Column-name helpers shared across agents."""

import re


_IDENTIFIER_EXACT = {
    "id",
    "machine_id",
    "asset_id",
    "device_id",
    "sensor_id",
    "unit_id",
    "record_id",
    "event_id",
    "timestamp_id",
}


def is_identifier_column(column_name: str) -> bool:
    """
    Return True when a column name appears to be an identifier.

    Uses token boundaries to avoid false positives such as "Idle" (contains "id")
    while still matching names like "Machine_ID" or encoded names such as
    "cat__Machine_ID_M004".
    """
    if column_name is None:
        return False

    name = str(column_name).strip().lower()
    if not name:
        return False

    if name in _IDENTIFIER_EXACT:
        return True

    if name.endswith("_id") or name.startswith("id_"):
        return True

    tokens = [t for t in re.split(r"[^a-z0-9]+", name) if t]
    if "id" in tokens:
        return True

    return False
