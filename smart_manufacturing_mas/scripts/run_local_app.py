import os
from pathlib import Path
import subprocess
import sys


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    project_dir = Path(__file__).resolve().parents[1]
    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = os.environ.get("APP_PORT", "8000")
    reload_enabled = _env_flag("APP_RELOAD", default=False)
    cmd = [
        str(project_dir / "mas_venv" / "bin" / "python"),
        "-m",
        "uvicorn",
        "webapp.app:app",
        "--host",
        host,
        "--port",
        port,
    ]
    if reload_enabled:
        cmd.append("--reload")
    try:
        return subprocess.call(cmd, cwd=str(project_dir))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
