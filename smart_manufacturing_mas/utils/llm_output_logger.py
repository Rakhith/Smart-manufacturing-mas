import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def log_llm_output(record: Dict[str, Any], file_name: str = "sllm_llm_outputs.json") -> None:
    """Append a structured SLM/LLM output record to logs/<file_name>."""
    try:
        root_dir = Path(__file__).resolve().parents[1]
        logs_dir = root_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        out_file = logs_dir / file_name

        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **record,
        }

        existing = []
        if out_file.exists():
            try:
                content = out_file.read_text(encoding="utf-8").strip()
                if content:
                    loaded = json.loads(content)
                    if isinstance(loaded, list):
                        existing = loaded
                    else:
                        existing = [loaded]
            except Exception:
                existing = []

        existing.append(payload)
        out_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception as exc:
        logging.warning(f"Could not write LLM output log: {exc}")
