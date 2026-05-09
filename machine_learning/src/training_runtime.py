from __future__ import annotations

import logging
import os
import select
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
from .utils import ensure_dir, load_json, project_root, resolve_path, save_json

TRAINING_MODE_TO_SOURCE = {
    "without_worker": "local_dataset",
    "with_worker": "spark_hdfs",
}

DEFAULT_STAGE_DURATION_SEC = {
    "worker_preprocess_submit": 16 * 60.0,
    "preprocess": 18 * 60.0,
    "train_sentiment_baseline": 10 * 60.0,
    "train_sentiment_transformer": 22 * 60.0,
    "train_recommender": 9 * 60.0,
    "evaluate": 60.0,
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


def _format_duration(seconds: float) -> str:
    seconds = int(max(0, round(seconds)))
    mins, sec = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h {mins}m {sec}s"
    if mins > 0:
        return f"{mins}m {sec}s"
    return f"{sec}s"


def _load_stage_duration_hints(
    config: Dict[str, Any],
    training_mode: str,
    stage_names: list[str],
) -> Dict[str, float]:
    hints = {name: DEFAULT_STAGE_DURATION_SEC.get(name, 60.0) for name in stage_names}

    experiments_dir = resolve_path(config, "training_experiments_dir")
    latest_run_path = experiments_dir / f"{training_mode}_latest_run.json"
    previous = load_json(latest_run_path)
    previous_timings = previous.get("timings_sec", {}) if isinstance(previous, dict) else {}

    if isinstance(previous_timings, dict):
        for name in stage_names:
            value = previous_timings.get(name)
            if isinstance(value, (int, float)) and value > 0:
                hints[name] = float(value)
    return hints


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
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stdout_lines: list[str] = []
    timeout_hit = False
    last_heartbeat = started_at

    assert proc.stdout is not None
    while True:
        now = time.perf_counter()
        if timeout_sec > 0 and (now - started_at) > timeout_sec:
            timeout_hit = True
            proc.kill()
            break

        ready, _, _ = select.select([proc.stdout], [], [], 1.0)
        if ready:
            line = proc.stdout.readline()
            if line:
                stdout_lines.append(line)
                stripped = line.rstrip()
                if stripped:
                    logging.info("[worker_preprocess] %s", stripped)

        if now - last_heartbeat >= 30.0:
            logging.info(
                "Worker preprocess masih berjalan... elapsed %s",
                _format_duration(now - started_at),
            )
            last_heartbeat = now

        if proc.poll() is not None:
            break

    remaining_stdout = proc.stdout.read() or ""
    if remaining_stdout:
        stdout_lines.append(remaining_stdout)
        for line in remaining_stdout.splitlines():
            if line.strip():
                logging.info("[worker_preprocess] %s", line)

    return_code = proc.wait()
    duration_sec = time.perf_counter() - started_at
    stdout_text = "".join(stdout_lines)
    if timeout_hit:
        raise RuntimeError(
            "Spark worker preprocessing timeout.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Timeout: {timeout_sec} detik\n"
            f"Stdout (tail): {stdout_text[-4000:]}"
        )

    payload = {
        "cmd": " ".join(cmd),
        "duration_sec": duration_sec,
        "rc": int(return_code),
        "stdout_tail": stdout_text[-4000:],
        "stderr_tail": "",
    }

    if return_code != 0:
        raise RuntimeError(
            "Spark worker preprocessing gagal.\n"
            f"Command: {payload['cmd']}\n"
            f"Exit code: {return_code}\n"
            f"Stdout (tail): {payload['stdout_tail']}\n"
            f"Stderr (tail): {payload['stderr_tail']}"
        )

    logging.info("Spark worker preprocessing selesai dalam %.2f detik", duration_sec)
    return payload


def run_worker_preprocess_submit(config: Dict[str, Any]) -> Dict[str, Any]:
    """Public wrapper untuk submit preprocessing Spark sebelum stage lain membaca output HDFS."""
    return _run_worker_preprocess_submit(config)


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
    source = resolve_mode_preprocess_source(training_mode)

    stage_timings: Dict[str, float] = {}
    stage_peak_memories: Dict[str, float] = {}
    run_started = time.perf_counter()
    run_started_at_utc = datetime.now(timezone.utc).isoformat()

    stage_sequence: list[str] = []
    worker_stage_enabled = bool(training_mode == "with_worker" and run_worker_preprocess)
    if worker_stage_enabled:
        stage_sequence.append("worker_preprocess_submit")
    stage_sequence.extend(
        [
            "preprocess",
            "train_sentiment_baseline",
            *(
                ["train_sentiment_transformer"]
                if include_transformer
                else []
            ),
            "train_recommender",
            "evaluate",
        ]
    )
    duration_hints = _load_stage_duration_hints(config, training_mode, stage_sequence)
    total_expected = max(sum(duration_hints.values()), 1.0)
    completed_stages: list[str] = []

    def _estimate_eta(elapsed_sec: float) -> float:
        done_expected = sum(duration_hints.get(name, 0.0) for name in completed_stages)
        remaining_expected = sum(
            duration_hints.get(name, 0.0)
            for name in stage_sequence
            if name not in completed_stages
        )
        if done_expected <= 0:
            return remaining_expected
        scale = elapsed_sec / done_expected
        scale = max(0.5, min(scale, 4.0))
        return remaining_expected * scale

    def _log_progress(stage_name: str, when: str) -> None:
        elapsed_sec = time.perf_counter() - run_started
        done_expected = sum(duration_hints.get(name, 0.0) for name in completed_stages)
        progress_pct = min(100.0, (done_expected / total_expected) * 100.0)
        eta_sec = _estimate_eta(elapsed_sec)
        stage_index = stage_sequence.index(stage_name) + 1
        stage_total = len(stage_sequence)
        if when == "start":
            logging.info(
                "Progress %.1f%% | Stage %s/%s started: %s | ETA ~ %s",
                progress_pct,
                stage_index,
                stage_total,
                stage_name,
                _format_duration(eta_sec),
            )
        else:
            logging.info(
                "Progress %.1f%% | Stage %s/%s completed: %s | ETA ~ %s",
                progress_pct,
                stage_index,
                stage_total,
                stage_name,
                _format_duration(eta_sec),
            )

    def _run_stage(stage_name: str, runner: Callable[[], Any]) -> Any:
        _log_progress(stage_name, when="start")
        result = _capture_stage(
            stage_name,
            runner,
            stage_timings=stage_timings,
            stage_peak_memories=stage_peak_memories,
        )
        completed_stages.append(stage_name)
        _log_progress(stage_name, when="completed")
        return result

    worker_preprocess_payload = None
    if worker_stage_enabled:
        worker_preprocess_payload = _run_stage(
            "worker_preprocess_submit",
            lambda: _run_worker_preprocess_submit(config),
        )

    applied_limit = apply_master_ram_limit(config, ram_limit_gb=ram_limit_gb)

    _run_stage(
        "preprocess",
        lambda: preprocess_and_split(config, source=source),
    )
    sentiment_metrics = _run_stage(
        "train_sentiment_baseline",
        lambda: train_baseline_sentiment(config),
    )
    transformer_metrics = None
    if include_transformer:
        transformer_metrics = _run_stage(
            "train_sentiment_transformer",
            lambda: train_transformer_sentiment(config),
        )
    recommender_metrics = _run_stage(
        "train_recommender",
        lambda: train_recommenders(config),
    )
    final_report = _run_stage(
        "evaluate",
        lambda: compile_final_report(config),
    )
    logging.info("Progress 100.0%% | Pipeline training selesai.")

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
