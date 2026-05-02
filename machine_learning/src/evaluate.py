from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from .utils import ensure_dir, load_json, resolve_path, save_json


def compile_final_report(config: Dict) -> Dict[str, Any]:
    """Compile final_report.json dari semua artifact training yang tersedia."""
    reports_dir = resolve_path(config, "eda_dir").parent  # machine_learning/reports/
    ensure_dir(reports_dir)

    report: Dict[str, Any] = {
        "compiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "sentiment_baseline": None,
        "sentiment_transformer": None,
        "recommender": None,
        "model_registry": None,
    }

    # Sentiment baseline
    baseline_path = resolve_path(config, "sentiment_dir") / "metrics.json"
    if baseline_path.exists():
        report["sentiment_baseline"] = load_json(baseline_path)
        logging.info("Loaded sentiment baseline metrics: %s", baseline_path)
    else:
        logging.warning("Sentiment baseline metrics not found: %s", baseline_path)

    # Transformer sentiment
    transformer_path = resolve_path(config, "sentiment_transformer_dir") / "metrics.json"
    if transformer_path.exists():
        report["sentiment_transformer"] = load_json(transformer_path)
        logging.info("Loaded transformer metrics: %s", transformer_path)
    else:
        logging.warning("Transformer metrics not found: %s", transformer_path)

    # Recommender
    rec_path = resolve_path(config, "recommender_report_json")
    if rec_path.exists():
        report["recommender"] = load_json(rec_path)
        logging.info("Loaded recommender metrics: %s", rec_path)
    else:
        logging.warning("Recommender metrics not found: %s", rec_path)

    # Model registry
    registry_path = resolve_path(config, "model_registry")
    if registry_path.exists():
        report["model_registry"] = load_json(registry_path)

    final_report_path = reports_dir / "final_report.json"
    save_json(report, final_report_path)
    logging.info("Final report saved: %s", final_report_path)

    return report
