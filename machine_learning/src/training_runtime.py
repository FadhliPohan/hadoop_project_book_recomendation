from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .evaluate import compile_final_report
from .preprocessing import preprocess_and_split
from .train_recommender import train_recommenders
from .train_sentiment import train_baseline_sentiment
from .train_sentiment_transformer import train_transformer_sentiment
from .utils import ensure_dir, project_root, resolve_path, save_json

TRAINING_MODE_TO_SOURCE = {
    "without_worker": "local_dataset",
    "with_worker": "spark_hdfs",
}


def resolve_mode_preprocess_source(training_mode: str) -> str:
    if training_mode not in TRAINING_MODE_TO_SOURCE:
        raise ValueError(
            f"Unknown training mode: {training_mode}. "
            f"Expected one of: {', '.join(TRAINING_MODE_TO_SOURCE)}"
        )
    return TRAINING_MODE_TO_SOURCE[training_mode]


def get_master_ram_limit_gb(config: Dict[str, Any], ram_limit_gb: Optional[float] = None) -> float:
    if ram_limit_gb is not None:
        return float(ram_limit_gb)
    return float(config.get("training", {}).get("master_ram_limit_gb", 3.0))


def _peak_memory_mb() -> float:
    try:
        import resource
    except Exception:
        return 0.0

    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss = float(usage.ru_maxrss)

    # Linux ru_maxrss: KiB. Darwin/BSD: bytes.
    if sys.platform.startswith("darwin"):
        return max_rss / (1024.0 * 1024.0)
    return max_rss / 1024.0


def apply_master_ram_limit(config: Dict[str, Any], ram_limit_gb: Optional[float] = None) -> Optional[float]:
    limit_gb = get_master_ram_limit_gb(config, ram_limit_gb=ram_limit_gb)
    if limit_gb <= 0:
        logging.warning("RAM limit <= 0 terdeteksi, pembatas memori master dilewati.")
        return None

    try:
        import resource
    except Exception:
        logging.warning("Module resource tidak tersedia. Batas RAM master tidak dapat diterapkan.")
        return None

    limit_bytes = int(limit_gb * 1024 * 1024 * 1024)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    unlimited_values = {-1, resource.RLIM_INFINITY}
    hard_unlimited = hard in unlimited_values
    soft_unlimited = soft in unlimited_values

    effective_soft = limit_bytes if soft_unlimited else min(soft, limit_bytes)
    if not hard_unlimited:
        effective_soft = min(effective_soft, hard)

    if effective_soft <= 0:
        logging.warning("Gagal menerapkan batas RAM master, nilai batas tidak valid: %s", effective_soft)
        return None

    if effective_soft != soft:
        resource.setrlimit(resource.RLIMIT_AS, (effective_soft, hard))
        logging.info(
            "Batas RAM master diterapkan: %.2f GB (soft=%s, hard=%s)",
            limit_gb,
            effective_soft,
            hard,
        )
    else:
        logging.info(
            "Batas RAM master sudah <= %.2f GB sebelumnya (soft=%s, hard=%s).",
            limit_gb,
            soft,
            hard,
        )

    return limit_gb


def _run_worker_preprocess_submit(config: Dict[str, Any]) -> Dict[str, Any]:
    spark_cfg = config.get("spark", {})
    preprocess_cfg = config.get("spark_preprocess", {})
    repo_root = project_root().parent
    submit_script = repo_root / "scripts" / "spark_submit_training.sh"

    if not submit_script.exists():
        raise FileNotFoundError(f"Script spark submit tidak ditemukan: {submit_script}")

    cmd = [
        "bash",
        str(submit_script),
        "preprocess_spark",
        str(int(spark_cfg.get("num_executors", 2))),
        str(int(spark_cfg.get("executor_cores", 2))),
        str(spark_cfg.get("executor_memory", "2G")),
        str(spark_cfg.get("driver_memory", "2G")),
    ]

    env = os.environ.copy()
    env["SPARK_SAMPLE_FRACTION"] = str(float(preprocess_cfg.get("sample_fraction", 1.0)))
    env["SPARK_OUTPUT_PARTITIONS"] = str(int(preprocess_cfg.get("output_partitions", 0)))
    env["SPARK_LOG_ROW_COUNTS"] = "1" if bool(preprocess_cfg.get("log_row_counts", False)) else "0"
    env["SPARK_SHOW_LABEL_DISTRIBUTION"] = "1" if bool(
        preprocess_cfg.get("show_label_distribution", False)
    ) else "0"
    env["YARN_PREFLIGHT_TIMEOUT"] = str(int(spark_cfg.get("preflight_timeout_sec", 20)))

    timeout_sec = int(spark_cfg.get("submit_timeout_sec", 1800))
    started_at = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    duration_sec = time.perf_counter() - started_at

    payload = {
        "cmd": " ".join(cmd),
        "duration_sec": duration_sec,
        "rc": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }

    if proc.returncode != 0:
        raise RuntimeError(
            "Spark worker preprocessing gagal.\n"
            f"Command: {payload['cmd']}\n"
            f"Exit code: {proc.returncode}\n"
            f"Stderr (tail): {payload['stderr_tail']}"
        )

    logging.info("Spark worker preprocessing selesai dalam %.2f detik", duration_sec)
    return payload


def _summarize_sentiment_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    best_model = metrics.get("best_model")
    best_test_f1 = 0.0
    best_test_acc = 0.0
    metric_rows = metrics.get("metrics", {})
    if best_model and best_model in metric_rows:
        best_test = metric_rows[best_model].get("test", {})
        best_test_f1 = float(best_test.get("f1_weighted", 0.0))
        best_test_acc = float(best_test.get("accuracy", 0.0))

    return {
        "best_model": best_model,
        "best_validation_f1_weighted": float(metrics.get("best_validation_f1_weighted", 0.0)),
        "best_test_f1_weighted": best_test_f1,
        "best_test_accuracy": best_test_acc,
    }


def _capture_stage(
    stage_name: str,
    runner: Callable[[], Any],
    stage_timings: Dict[str, float],
    stage_peak_memories: Dict[str, float],
) -> Any:
    before_mem = _peak_memory_mb()
    started = time.perf_counter()
    result = runner()
    duration = time.perf_counter() - started
    after_mem = _peak_memory_mb()

    stage_timings[stage_name] = duration
    stage_peak_memories[stage_name] = max(before_mem, after_mem)
    logging.info(
        "Stage '%s' selesai dalam %.2f detik (peak memory ~ %.2f MB)",
        stage_name,
        duration,
        stage_peak_memories[stage_name],
    )
    return result


def run_training_pipeline(
    config: Dict[str, Any],
    training_mode: str,
    include_transformer: bool = False,
    run_worker_preprocess: bool = False,
    ram_limit_gb: Optional[float] = None,
) -> Dict[str, Any]:
    applied_limit = apply_master_ram_limit(config, ram_limit_gb=ram_limit_gb)
    source = resolve_mode_preprocess_source(training_mode)

    stage_timings: Dict[str, float] = {}
    stage_peak_memories: Dict[str, float] = {}
    run_started = time.perf_counter()
    run_started_at_utc = datetime.now(timezone.utc).isoformat()

    worker_preprocess_payload = None
    if training_mode == "with_worker" and run_worker_preprocess:
        worker_preprocess_payload = _capture_stage(
            "worker_preprocess_submit",
            lambda: _run_worker_preprocess_submit(config),
            stage_timings=stage_timings,
            stage_peak_memories=stage_peak_memories,
        )

    _capture_stage(
        "preprocess",
        lambda: preprocess_and_split(config, source=source),
        stage_timings=stage_timings,
        stage_peak_memories=stage_peak_memories,
    )
    sentiment_metrics = _capture_stage(
        "train_sentiment_baseline",
        lambda: train_baseline_sentiment(config),
        stage_timings=stage_timings,
        stage_peak_memories=stage_peak_memories,
    )
    transformer_metrics = None
    if include_transformer:
        transformer_metrics = _capture_stage(
            "train_sentiment_transformer",
            lambda: train_transformer_sentiment(config),
            stage_timings=stage_timings,
            stage_peak_memories=stage_peak_memories,
        )
    recommender_metrics = _capture_stage(
        "train_recommender",
        lambda: train_recommenders(config),
        stage_timings=stage_timings,
        stage_peak_memories=stage_peak_memories,
    )
    final_report = _capture_stage(
        "evaluate",
        lambda: compile_final_report(config),
        stage_timings=stage_timings,
        stage_peak_memories=stage_peak_memories,
    )

    total_duration = time.perf_counter() - run_started
    payload = {
        "status": "success",
        "training_mode": training_mode,
        "preprocess_source": source,
        "run_started_at_utc": run_started_at_utc,
        "run_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "include_transformer": bool(include_transformer),
        "worker_preprocess_submit_enabled": bool(run_worker_preprocess and training_mode == "with_worker"),
        "master_ram_limit_gb": applied_limit,
        "timings_sec": stage_timings,
        "peak_memory_mb": {
            "pipeline_peak": max(stage_peak_memories.values()) if stage_peak_memories else 0.0,
            "stages": stage_peak_memories,
        },
        "sentiment_baseline_summary": _summarize_sentiment_metrics(sentiment_metrics),
        "sentiment_transformer_summary": transformer_metrics or None,
        "recommender_summary": {
            "rmse": float(recommender_metrics.get("rmse", 0.0)),
            "mae": float(recommender_metrics.get("mae", 0.0)),
            "precision_at_5": float((recommender_metrics.get("precision_at_k") or {}).get("5", 0.0)),
            "recall_at_5": float((recommender_metrics.get("recall_at_k") or {}).get("5", 0.0)),
            "ndcg_at_10": float((recommender_metrics.get("ndcg_at_k") or {}).get("10", 0.0)),
        },
        "total_duration_sec": total_duration,
        "worker_preprocess_submit": worker_preprocess_payload,
        "artifacts": {
            "final_report": str((resolve_path(config, "eda_dir").parent / "final_report.json")),
            "sentiment_metrics": str(resolve_path(config, "sentiment_dir") / "metrics.json"),
            "recommender_metrics": str(resolve_path(config, "recommender_report_json")),
        },
        "final_report_snapshot": final_report,
    }

    experiments_dir = resolve_path(config, "training_experiments_dir")
    ensure_dir(experiments_dir)
    mode_report_path = experiments_dir / f"{training_mode}_latest_run.json"
    save_json(payload, mode_report_path)
    logging.info("Saved training pipeline run summary: %s", mode_report_path)

    return payload


def _build_comparison_summary(
    without_worker: Optional[Dict[str, Any]],
    with_worker: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not without_worker or not with_worker:
        return {}

    without_sent = without_worker.get("sentiment_baseline_summary", {})
    with_sent = with_worker.get("sentiment_baseline_summary", {})
    without_rec = without_worker.get("recommender_summary", {})
    with_rec = with_worker.get("recommender_summary", {})

    return {
        "total_duration_delta_sec_with_minus_without": float(with_worker.get("total_duration_sec", 0.0))
        - float(without_worker.get("total_duration_sec", 0.0)),
        "sentiment_test_f1_delta_with_minus_without": float(with_sent.get("best_test_f1_weighted", 0.0))
        - float(without_sent.get("best_test_f1_weighted", 0.0)),
        "sentiment_test_accuracy_delta_with_minus_without": float(with_sent.get("best_test_accuracy", 0.0))
        - float(without_sent.get("best_test_accuracy", 0.0)),
        "recommender_rmse_delta_with_minus_without": float(with_rec.get("rmse", 0.0))
        - float(without_rec.get("rmse", 0.0)),
        "recommender_mae_delta_with_minus_without": float(with_rec.get("mae", 0.0))
        - float(without_rec.get("mae", 0.0)),
        "recommender_ndcg10_delta_with_minus_without": float(with_rec.get("ndcg_at_10", 0.0))
        - float(without_rec.get("ndcg_at_10", 0.0)),
    }


def compare_training_modes(
    config: Dict[str, Any],
    include_transformer: bool = False,
    run_worker_preprocess: bool = False,
    ram_limit_gb: Optional[float] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "include_transformer": bool(include_transformer),
        "run_worker_preprocess": bool(run_worker_preprocess),
        "master_ram_limit_gb": get_master_ram_limit_gb(config, ram_limit_gb=ram_limit_gb),
        "runs": {},
        "errors": {},
        "comparison": {},
        "status": "pending",
    }

    for mode in ["without_worker", "with_worker"]:
        try:
            run_result = run_training_pipeline(
                config,
                training_mode=mode,
                include_transformer=include_transformer,
                run_worker_preprocess=run_worker_preprocess if mode == "with_worker" else False,
                ram_limit_gb=ram_limit_gb,
            )
            report["runs"][mode] = run_result
        except Exception as exc:
            report["errors"][mode] = {
                "error": str(exc),
                "traceback": traceback.format_exc(limit=10),
            }
            logging.exception("Compare training mode gagal pada mode=%s", mode)

    without_worker = report["runs"].get("without_worker")
    with_worker = report["runs"].get("with_worker")
    report["comparison"] = _build_comparison_summary(without_worker, with_worker)

    if without_worker and with_worker and not report["errors"]:
        report["status"] = "success"
    elif report["runs"] and report["errors"]:
        report["status"] = "partial_success"
    else:
        report["status"] = "failed"

    comparison_path = resolve_path(config, "training_comparison_json")
    save_json(report, comparison_path)
    logging.info("Saved training mode comparison report: %s", comparison_path)
    return report
