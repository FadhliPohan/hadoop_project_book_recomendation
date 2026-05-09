from __future__ import annotations

import atexit
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Dict, Optional

import pandas as pd

from .utils import resolve_path


REVIEW_COLUMNS = [
    "Title",
    "User_id",
    "review/score",
    "review/text",
    "review/summary",
    "review/time",
]

BOOK_COLUMNS = [
    "Title",
    "description",
    "authors",
    "categories",
    "ratingsCount",
]

HDFS_PROGRESS_LOG_INTERVAL_SEC = 10.0
_HDFS_LOCAL_CACHE: Dict[str, Path] = {}
_HDFS_CACHE_TEMP_ROOTS: list[Path] = []


def _build_hdfs_uri(namenode_uri: str, hdfs_path: str) -> str:
    normalized_namenode = namenode_uri.rstrip("/")
    normalized_path = hdfs_path if hdfs_path.startswith("/") else f"/{hdfs_path}"
    return f"{normalized_namenode}{normalized_path}"


def _format_size(num_bytes: int) -> str:
    value = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _format_duration(seconds: float) -> str:
    seconds = int(max(0, round(seconds)))
    mins, sec = divmod(seconds, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h {mins}m {sec}s"
    if mins > 0:
        return f"{mins}m {sec}s"
    return f"{sec}s"


def _sum_local_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0

    total = 0
    for file_path in path.rglob("*"):
        try:
            if file_path.is_file():
                total += file_path.stat().st_size
        except OSError:
            continue
    return total


def _build_hdfs_subprocess_kwargs(capture_output: bool = True) -> Dict[str, object]:
    kwargs: Dict[str, object] = {"text": True}
    if capture_output:
        kwargs["capture_output"] = True

    try:
        import resource

        def _relax_rlimit_as() -> None:
            try:
                _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                unlimited_values = {-1, resource.RLIM_INFINITY}
                target_soft = resource.RLIM_INFINITY if hard in unlimited_values else hard
                resource.setrlimit(resource.RLIMIT_AS, (target_soft, hard))
            except Exception:
                return

        kwargs["preexec_fn"] = _relax_rlimit_as
    except Exception:
        pass

    return kwargs


def _run_hdfs_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """Run HDFS CLI command with a relaxed RLIMIT_AS on child process when possible."""
    return subprocess.run(cmd, **_build_hdfs_subprocess_kwargs(capture_output=True))


def _parse_hdfs_du_size_bytes(du_stdout: str) -> Optional[int]:
    line = (du_stdout or "").strip().splitlines()
    if not line:
        return None
    tokens = line[0].split()
    if not tokens:
        return None
    try:
        return int(tokens[0])
    except (TypeError, ValueError):
        return None


def _resolve_hdfs_content_size_bytes(hdfs_uri: str) -> Optional[int]:
    du = _run_hdfs_command(["hdfs", "dfs", "-du", "-s", hdfs_uri])
    if du.returncode != 0:
        return None
    return _parse_hdfs_du_size_bytes(du.stdout)


def _download_hdfs_dir_with_progress(
    parquet_uri: str,
    local_root: Path,
    expected_size_bytes: Optional[int],
) -> subprocess.CompletedProcess[str]:
    local_parquet_dir = local_root / PurePosixPath(parquet_uri).name
    cmd = ["hdfs", "dfs", "-get", "-f", parquet_uri, str(local_root)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_build_hdfs_subprocess_kwargs(capture_output=False),
    )

    started = time.perf_counter()
    last_log = 0.0
    while proc.poll() is None:
        now = time.perf_counter()
        if now - last_log >= HDFS_PROGRESS_LOG_INTERVAL_SEC:
            copied_bytes = _sum_local_size(local_parquet_dir)
            elapsed = max(now - started, 1e-6)
            speed_bps = copied_bytes / elapsed
            if expected_size_bytes and expected_size_bytes > 0:
                progress_pct = min(99.0, (copied_bytes / expected_size_bytes) * 100.0)
                remaining_bytes = max(expected_size_bytes - copied_bytes, 0)
                eta_sec = remaining_bytes / speed_bps if speed_bps > 0 else float("inf")
                logging.info(
                    "HDFS bridge progress: %.1f%% (%s / %s, %s/s, ETA ~ %s)",
                    progress_pct,
                    _format_size(copied_bytes),
                    _format_size(expected_size_bytes),
                    _format_size(int(speed_bps)),
                    _format_duration(eta_sec) if eta_sec != float("inf") else "unknown",
                )
            else:
                logging.info(
                    "HDFS bridge progress: copied %s (kecepatan ~ %s/s)",
                    _format_size(copied_bytes),
                    _format_size(int(speed_bps)),
                )
            last_log = now
        time.sleep(1.0)

    stdout, stderr = proc.communicate()
    finished = time.perf_counter()
    duration = finished - started
    final_size = _sum_local_size(local_parquet_dir)
    if proc.returncode == 0:
        logging.info(
            "HDFS bridge selesai dalam %s (local size %s)",
            _format_duration(duration),
            _format_size(final_size),
        )
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _cleanup_hdfs_cache() -> None:
    for path in _HDFS_CACHE_TEMP_ROOTS:
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup_hdfs_cache)


def _materialize_hdfs_parquet_dir(parquet_uri: str) -> Path:
    cached = _HDFS_LOCAL_CACHE.get(parquet_uri)
    if cached and cached.exists():
        logging.info("Reuse cached Spark HDFS bridge directory: %s", cached)
        return cached

    local_root = Path(tempfile.mkdtemp(prefix="spark_hdfs_bridge_", dir="/tmp"))
    _HDFS_CACHE_TEMP_ROOTS.append(local_root)

    expected_size_bytes = _resolve_hdfs_content_size_bytes(parquet_uri)
    if expected_size_bytes is not None:
        logging.info(
            "Mulai bridge HDFS -> local (estimasi size %s): %s",
            _format_size(expected_size_bytes),
            parquet_uri,
        )
    else:
        logging.info("Mulai bridge HDFS -> local: %s", parquet_uri)

    fetch = _download_hdfs_dir_with_progress(
        parquet_uri=parquet_uri,
        local_root=local_root,
        expected_size_bytes=expected_size_bytes,
    )
    if fetch.returncode != 0:
        stderr_detail = fetch.stderr.strip()
        stdout_detail = fetch.stdout.strip()
        debug_detail = stderr_detail or stdout_detail or "(kosong)"
        raise RuntimeError(
            "Gagal mengambil hasil preprocessing Spark dari HDFS.\n"
            f"Command: hdfs dfs -get -f {parquet_uri} {local_root}\n"
            f"Debug detail: {debug_detail}"
        )

    local_parquet_dir = local_root / PurePosixPath(parquet_uri).name
    if not local_parquet_dir.exists():
        raise FileNotFoundError(
            f"Direktori Parquet hasil download tidak ditemukan: {local_parquet_dir}"
        )

    _HDFS_LOCAL_CACHE[parquet_uri] = local_parquet_dir
    return local_parquet_dir


def _normalize_reviews_df(df: pd.DataFrame, config: Dict, source_name: str) -> pd.DataFrame:
    if "Title" in df.columns:
        df = df.rename(
            columns={
                "Title": "item_id",
                "User_id": "user_id",
                "review/score": "rating",
                "review/text": "review_text",
                "review/summary": "summary",
                "review/time": "review_time",
            }
        )

    for col, default in {"summary": "", "review_time": ""}.items():
        if col not in df.columns:
            df[col] = default

    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["review_text"] = df["review_text"].astype(str)
    df["summary"] = df["summary"].fillna("").astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    if "review_text_clean" in df.columns:
        df["review_text_clean"] = df["review_text_clean"].fillna("").astype(str)

    before = len(df)
    df = df.dropna(subset=["item_id", "user_id", "rating", "review_text"]).copy()
    df = df[df["review_text"].str.len() >= config["data"].get("min_review_text_length", 5)].copy()
    logging.info("%s: dropped %s invalid rows", source_name, before - len(df))

    return df


def load_reviews(config: Dict, sample_rows: Optional[int] = None) -> pd.DataFrame:
    csv_path = resolve_path(config, "reviews_csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {csv_path}")

    nrows = sample_rows if sample_rows is not None else config["data"].get("sample_rows")
    logging.info("Loading reviews from %s (nrows=%s)", csv_path, nrows)

    df = pd.read_csv(
        csv_path,
        usecols=REVIEW_COLUMNS,
        nrows=nrows,
        low_memory=False,
    )

    return _normalize_reviews_df(df, config, "Local dataset")


def load_reviews_from_hdfs_spark_output(config: Dict, sample_rows: Optional[int] = None) -> pd.DataFrame:
    try:
        import pyarrow.dataset as ds
    except Exception as exc:
        raise RuntimeError(
            "PyArrow tidak tersedia untuk membaca hasil distributed preprocessing. "
            "Gunakan source local dataset atau install pyarrow."
        ) from exc

    hadoop_cfg = config.get("hadoop", {})
    namenode_uri = hadoop_cfg.get("namenode_uri", "hdfs://fadhli:9000")
    output_hdfs_path = hadoop_cfg.get("output_hdfs_path", "/user/fadhli/output/amazon_books_ml")
    parquet_uri = _build_hdfs_uri(namenode_uri, f"{output_hdfs_path.rstrip('/')}/processed")

    logging.info("Loading distributed preprocessing output from %s", parquet_uri)

    exists = _run_hdfs_command(["hdfs", "dfs", "-test", "-e", parquet_uri])
    if exists.returncode != 0:
        stderr_detail = exists.stderr.strip()
        stdout_detail = exists.stdout.strip()
        debug_detail = stderr_detail or stdout_detail or "(kosong)"
        raise FileNotFoundError(
            "Output distributed preprocessing Spark belum ditemukan di HDFS.\n"
            f"Path: {parquet_uri}\n"
            "Jalankan Spark preprocessing terlebih dahulu:\n"
            "  bash scripts/spark_submit_training.sh preprocess_spark\n"
            "Atau saat training mode with_worker, tambahkan flag:\n"
            "  --run-worker-preprocess\n"
            f"Debug detail: {debug_detail}"
        )

    local_parquet_dir = _materialize_hdfs_parquet_dir(parquet_uri)
    dataset = ds.dataset(str(local_parquet_dir), format="parquet")
    expected_columns = [
        "item_id",
        "user_id",
        "rating",
        "review_text",
        "summary",
        "review_time",
        "sentiment_label",
        "sentiment_text",
        "review_text_clean",
    ]
    selected_columns = [col for col in expected_columns if col in dataset.schema.names]
    if not selected_columns:
        raise ValueError(
            "Output distributed preprocessing tidak memiliki kolom yang dikenali. "
            "Pastikan `preprocess_spark` sudah berjalan dengan sukses."
        )

    if sample_rows is not None:
        logging.info("Limiting Spark preprocessing output to %s rows for local training bridge", sample_rows)
        table = dataset.head(int(sample_rows), columns=selected_columns)
    else:
        table = dataset.to_table(columns=selected_columns)

    pdf = table.to_pandas()

    return _normalize_reviews_df(pdf, config, "Spark HDFS output")


def load_books(config: Dict, sample_rows: Optional[int] = None) -> pd.DataFrame:
    csv_path = resolve_path(config, "books_csv")
    if not csv_path.exists():
        logging.warning("Books metadata tidak ditemukan: %s", csv_path)
        return pd.DataFrame(columns=["item_id", "description", "authors", "categories", "ratingsCount"])

    nrows = sample_rows if sample_rows is not None else config["data"].get("sample_rows")
    logging.info("Loading books metadata from %s (nrows=%s)", csv_path, nrows)

    df = pd.read_csv(csv_path, usecols=BOOK_COLUMNS, nrows=nrows, low_memory=False)
    df = df.rename(columns={"Title": "item_id"})
    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str)

    return df
