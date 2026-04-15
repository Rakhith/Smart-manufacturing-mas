import os
from pathlib import Path
import sys


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    import uvicorn
    
    project_dir = Path(__file__).resolve().parents[1]
    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = int(os.environ.get("APP_PORT", "8000"))
    reload_enabled = _env_flag("APP_RELOAD", default=False)
    
    # Add project directory to sys.path for module imports
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    
    # Change to project directory
    os.chdir(str(project_dir))
    
    # Run uvicorn directly in the current process
    uvicorn.run(
        "webapp.app:app",
        host=host,
        port=port,
        reload=reload_enabled,
    )
    return 0


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)
