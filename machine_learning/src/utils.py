from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    cfg_path = project_root() / config_path
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(config: Dict[str, Any], path_key: str) -> Path:
    rel_path = config["paths"][path_key]
    return project_root() / rel_path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    logs_dir = project_root() / "logs"
    ensure_dir(logs_dir)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_path = logs_dir / f"pipeline_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(file_path, encoding="utf-8"),
        ],
    )


def save_json(payload: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_model_registry(config: Dict[str, Any], record: Dict[str, Any]) -> None:
    registry_path = resolve_path(config, "model_registry")
    ensure_dir(registry_path.parent)

    if registry_path.exists():
        with registry_path.open("r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = {"models": []}

    registry.setdefault("models", []).append(record)

    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
