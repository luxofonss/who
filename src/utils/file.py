import json
import datetime
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

def _write_to_file(content: str, prefix: str, path: PathLike) -> str:
        try:
            prompts_dir = Path(path)
            prompts_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp}.txt"
            filepath = prompts_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Type: {prefix}\n")
                f.write("=" * 80 + "\n\n")
                f.write(content)
            return str(filepath)
        except Exception:
            return ""
