#!/usr/bin/env python3
"""
spark_preprocess.py — Distributed preprocessing menggunakan PySpark + YARN.

Membaca dataset Books_rating.csv dari HDFS, melakukan cleaning & labeling,
lalu menyimpan hasil ke HDFS dalam format Parquet.

Cara menjalankan:
  ./scripts/spark_submit_training.sh preprocess_spark 2 2 2G 2G

Atau langsung:
  spark-submit --master yarn --deploy-mode client \\
    machine_learning/src/spark_preprocess.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

DEFAULT_NAMENODE_URI = "hdfs://fadhli:9000"
DEFAULT_DATASET_HDFS_PATH = "/user/fadhli/amazon_books"
DEFAULT_OUTPUT_HDFS_PATH = "/user/fadhli/output/amazon_books_ml"
DEFAULT_SAMPLE_FRACTION = 1.0
DEFAULT_LOG_ROW_COUNTS = False
DEFAULT_SHOW_LABEL_DISTRIBUTION = False
DEFAULT_OUTPUT_PARTITIONS = 0
DEFAULT_MAX_ROWS = 0


def _build_hdfs_uri(namenode_uri: str, hdfs_path: str) -> str:
    normalized_namenode = (namenode_uri or DEFAULT_NAMENODE_URI).rstrip("/")
    normalized_path = hdfs_path if hdfs_path.startswith("/") else f"/{hdfs_path}"
    return f"{normalized_namenode}{normalized_path}"


def _load_project_config() -> dict:
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    if not cfg_path.exists():
        return {}

    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_hdfs_io_paths(cfg: dict) -> tuple[str, str]:
    env_input = os.environ.get("HDFS_INPUT_URI")
    env_output = os.environ.get("HDFS_OUTPUT_URI")
    if env_input and env_output:
        return env_input, env_output

    hadoop_cfg = cfg.get("hadoop", {})
    namenode_uri = hadoop_cfg.get("namenode_uri", DEFAULT_NAMENODE_URI)
    dataset_hdfs_path = hadoop_cfg.get("dataset_hdfs_path", DEFAULT_DATASET_HDFS_PATH)
    output_hdfs_path = hadoop_cfg.get("output_hdfs_path", DEFAULT_OUTPUT_HDFS_PATH)

    hdfs_input = env_input or _build_hdfs_uri(namenode_uri, f"{dataset_hdfs_path.rstrip('/')}/Books_rating.csv")
    hdfs_output = env_output or _build_hdfs_uri(namenode_uri, f"{output_hdfs_path.rstrip('/')}/processed")
    return hdfs_input, hdfs_output


def _coerce_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_preprocess_options(cfg: dict) -> dict:
    spark_preprocess_cfg = cfg.get("spark_preprocess", {})

    sample_fraction = _coerce_float(
        os.environ.get("SPARK_SAMPLE_FRACTION", spark_preprocess_cfg.get("sample_fraction")),
        DEFAULT_SAMPLE_FRACTION,
    )
    if not 0 < sample_fraction <= 1.0:
        sample_fraction = DEFAULT_SAMPLE_FRACTION

    output_partitions = _coerce_int(
        os.environ.get("SPARK_OUTPUT_PARTITIONS", spark_preprocess_cfg.get("output_partitions")),
        DEFAULT_OUTPUT_PARTITIONS,
    )
    if output_partitions < 0:
        output_partitions = DEFAULT_OUTPUT_PARTITIONS

    max_rows = _coerce_int(
        os.environ.get("SPARK_MAX_ROWS", spark_preprocess_cfg.get("max_rows")),
        DEFAULT_MAX_ROWS,
    )
    if max_rows < 0:
        max_rows = DEFAULT_MAX_ROWS

    return {
        "sample_fraction": sample_fraction,
        "log_row_counts": _coerce_bool(
            os.environ.get("SPARK_LOG_ROW_COUNTS", spark_preprocess_cfg.get("log_row_counts")),
            DEFAULT_LOG_ROW_COUNTS,
        ),
        "show_label_distribution": _coerce_bool(
            os.environ.get("SPARK_SHOW_LABEL_DISTRIBUTION", spark_preprocess_cfg.get("show_label_distribution")),
            DEFAULT_SHOW_LABEL_DISTRIBUTION,
        ),
        "output_partitions": output_partitions,
        "max_rows": max_rows,
    }


def main() -> None:
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
    except ImportError:
        print("[ERROR] PySpark belum terpasang. Install: pip install pyspark", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Memulai Spark session...")
    spark = (
        SparkSession.builder
        .appName("AmazonBooks_Distributed_Preprocess")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    cfg = _load_project_config()
    hdfs_input, hdfs_output = _resolve_hdfs_io_paths(cfg)
    run_options = _resolve_preprocess_options(cfg)

    print(
        "[INFO] Opsi preprocess: "
        f"sample_fraction={run_options['sample_fraction']}, "
        f"log_row_counts={run_options['log_row_counts']}, "
        f"show_label_distribution={run_options['show_label_distribution']}, "
        f"output_partitions={run_options['output_partitions']}, "
        f"max_rows={run_options['max_rows']}"
    )

    print(f"[INFO] Membaca dataset dari HDFS: {hdfs_input}")
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .option("multiLine", True)
        .option("escape", '"')
        .csv(hdfs_input)
    )

    # Rename kolom sesuai konvensi project
    rename_map = {
        "Title": "item_id",
        "User_id": "user_id",
        "review/score": "rating",
        "review/text": "review_text",
        "review/summary": "summary",
        "review/time": "review_time",
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)

    # Pilih kolom yang dibutuhkan
    needed = [c for c in ["item_id", "user_id", "rating", "review_text", "summary", "review_time"] if c in df.columns]
    df = df.select(needed)

    # Drop null pada kolom kritis
    df = df.dropna(subset=["item_id", "user_id", "rating", "review_text"])

    # Cast rating ke float
    df = df.withColumn("rating", df["rating"].cast("float"))
    df = df.filter(F.col("rating").isNotNull())

    # Filter review_text minimal 5 karakter
    df = df.filter(F.length(F.col("review_text")) >= 5)

    # Labeling sentimen
    df = df.withColumn(
        "sentiment_label",
        F.when(F.col("rating") <= 2, 0)
         .when(F.col("rating") == 3, 1)
         .otherwise(2)
    )
    df = df.withColumn(
        "sentiment_text",
        F.when(F.col("sentiment_label") == 0, "Negative")
         .when(F.col("sentiment_label") == 1, "Neutral")
         .otherwise("Positive")
    )

    # Basic text cleaning (distributed, tanpa NLTK di worker)
    # NLTK tidak tersedia di worker tanpa distribusi resource tambahan
    df = df.withColumn("review_text_clean", F.lower(F.col("review_text")))
    df = df.withColumn("review_text_clean", F.regexp_replace("review_text_clean", r"https?://\S+|www\.\S+", " "))
    df = df.withColumn("review_text_clean", F.regexp_replace("review_text_clean", r"<[^>]+>", " "))
    df = df.withColumn("review_text_clean", F.regexp_replace("review_text_clean", r"[^a-z\s]", " "))
    df = df.withColumn("review_text_clean", F.regexp_replace("review_text_clean", r"\s+", " "))
    df = df.withColumn("review_text_clean", F.trim("review_text_clean"))

    if run_options["sample_fraction"] < 1.0:
        df = df.sample(fraction=run_options["sample_fraction"], seed=42)
        print(f"[INFO] Sampling aktif: {run_options['sample_fraction']:.3f}")

    if run_options["max_rows"] > 0:
        df = df.limit(int(run_options["max_rows"]))
        print(f"[INFO] Row cap aktif: max_rows={run_options['max_rows']}")

    should_materialize = run_options["log_row_counts"] or run_options["show_label_distribution"]
    if should_materialize:
        df = df.persist()

    if run_options["log_row_counts"]:
        total = df.count()
        print(f"[INFO] Total baris setelah filter & cleaning: {total}")

    if run_options["show_label_distribution"]:
        print("[INFO] Distribusi label sentimen:")
        df.groupBy("sentiment_text").count().orderBy("count", ascending=False).show()

    if run_options["output_partitions"] > 0:
        df = df.coalesce(run_options["output_partitions"])
        print(f"[INFO] Output akan ditulis dengan coalesce({run_options['output_partitions']})")

    # Simpan ke HDFS sebagai Parquet (efisien untuk dataset besar)
    print(f"[INFO] Menyimpan hasil ke HDFS: {hdfs_output}")
    df.write.mode("overwrite").parquet(hdfs_output)
    print("[INFO] Selesai. Data tersimpan di HDFS.")

    if should_materialize:
        df.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()
