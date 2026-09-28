"""Configuration loading. Plain JSON so the package has no YAML dependency."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


def load_json(name_or_path: str) -> Dict[str, Any]:
    p = Path(name_or_path)
    if not p.exists():
        p = CONFIGS / name_or_path
    return json.loads(p.read_text(encoding="utf-8"))


def sim_config() -> Dict[str, Any]:
    return load_json("sim.json")


def config_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()[:12]
