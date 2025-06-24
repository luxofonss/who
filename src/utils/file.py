import json
from pathlib import Path
from typing import Any, Union


PathLike = Union[str, Path]


def ensure_dir(path: PathLike):
    """Create the directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def read_json(path: PathLike, default: Any | None = None):
    p = Path(path)
    if not p.exists():
        return default
    try:
        with p.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except json.JSONDecodeError:
        return default


def write_json(path: PathLike, data: Any):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False) 