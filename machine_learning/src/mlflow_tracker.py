from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from .utils import project_root


def _get_mlflow_module():
    try:
        import mlflow  # type: ignore

        return mlflow
    except Exception:
        return None


def is_enabled(config: Dict) -> bool:
    mlflow_cfg = config.get("mlflow", {})
    return bool(mlflow_cfg.get("enabled", False))


def start_run(config: Dict, run_name: str):
    mlflow = _get_mlflow_module()
    if not is_enabled(config) or mlflow is None:
        return None

    tracking_dir = project_root() / "mlruns"
    tracking_uri = f"file://{tracking_dir.resolve()}"
    experiment_name = config.get("mlflow", {}).get("experiment_name", "default")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_params(config: Dict, params: Dict) -> None:
    mlflow = _get_mlflow_module()
    if not is_enabled(config) or mlflow is None:
        return

    flat = {str(k): str(v) for k, v in params.items()}
    mlflow.log_params(flat)


def log_metrics(config: Dict, metrics: Dict, step: Optional[int] = None) -> None:
    mlflow = _get_mlflow_module()
    if not is_enabled(config) or mlflow is None:
        return

    numeric = {}
    for k, v in metrics.items():
        try:
            numeric[str(k)] = float(v)
        except Exception:
            continue

    if not numeric:
        return

    if step is None:
        mlflow.log_metrics(numeric)
    else:
        mlflow.log_metrics(numeric, step=step)


def log_artifact(config: Dict, file_path: Path, artifact_path: Optional[str] = None) -> None:
    mlflow = _get_mlflow_module()
    if not is_enabled(config) or mlflow is None:
        return

    if not file_path.exists():
        logging.warning("MLflow artifact tidak ditemukan: %s", file_path)
        return

    if artifact_path:
        mlflow.log_artifact(str(file_path), artifact_path=artifact_path)
    else:
        mlflow.log_artifact(str(file_path))


def log_artifacts(config: Dict, directory: Path, artifact_path: Optional[str] = None) -> None:
    mlflow = _get_mlflow_module()
    if not is_enabled(config) or mlflow is None:
        return

    if not directory.exists():
        logging.warning("MLflow artifacts directory tidak ditemukan: %s", directory)
        return

    if artifact_path:
        mlflow.log_artifacts(str(directory), artifact_path=artifact_path)
    else:
        mlflow.log_artifacts(str(directory))
