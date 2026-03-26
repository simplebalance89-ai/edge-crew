import os
from pathlib import Path


BASE_DIR = Path(__file__).parent


def _resolve_dir(env_name: str, default_name: str) -> Path:
    raw = os.environ.get(env_name)
    if not raw:
        return BASE_DIR / default_name
    path = Path(raw)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


DATA_DIR = _resolve_dir("DATA_DIR", "data")
GRADES_DIR = _resolve_dir("GRADES_DIR", "grades")
PROFILES_DIR = BASE_DIR / "profiles"
