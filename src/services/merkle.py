from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

from utils.file import ensure_dir, read_json, write_json


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_merkle_tree(root: Path) -> Dict[str, str]:
    """Return mapping of relative file_path -> sha256 digest."""
    tree: Dict[str, str] = {}
    for p in root.rglob("*"):
        if p.is_file():
            tree[p.relative_to(root).as_posix()] = _file_hash(p)
    return tree


def diff_trees(old: Dict[str, str] | None, new: Dict[str, str]) -> List[str]:
    if old is None:
        return list(new.keys())
    changed = [path for path, digest in new.items() if old.get(path) != digest]
    removed = [path for path in old if path not in new]
    return sorted(set(changed + removed))


MERKLE_DIR = Path("storage/merkle")


def save_merkle(project_id: str, tree: Dict[str, str]):
    ensure_dir(MERKLE_DIR)
    write_json(MERKLE_DIR / f"{project_id}.json", tree)


def load_merkle(project_id: str) -> Dict[str, str] | None:
    return read_json(MERKLE_DIR / f"{project_id}.json", default=None) 