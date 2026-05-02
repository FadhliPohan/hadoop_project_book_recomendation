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
SAMPLE_FRACTION = 1.0  # 1.0 = pakai semua data, 0.1 = 10% untuk test cepat


def _build_hdfs_uri(namenode_uri: str, hdfs_path: str) -> str:
    normalized_namenode = (namenode_uri or DEFAULT_NAMENODE_URI).rstrip("/")
    normalized_path = hdfs_path if hdfs_path.startswith("/") else f"/{hdfs_path}"
    return f"{normalized_namenode}{normalized_path}"


def _resolve_hdfs_io_paths() -> tuple[str, str]:
    env_input = os.environ.get("HDFS_INPUT_URI")
    env_output = os.environ.get("HDFS_OUTPUT_URI")
    if env_input and env_output:
        return env_input, env_output

    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    if not cfg_path.exists():
        default_input = _build_hdfs_uri(DEFAULT_NAMENODE_URI, f"{DEFAULT_DATASET_HDFS_PATH}/Books_rating.csv")
        default_output = _build_hdfs_uri(DEFAULT_NAMENODE_URI, f"{DEFAULT_OUTPUT_HDFS_PATH}/processed")
        return default_input, default_output

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    hadoop_cfg = cfg.get("hadoop", {})
    namenode_uri = hadoop_cfg.get("namenode_uri", DEFAULT_NAMENODE_URI)
    dataset_hdfs_path = hadoop_cfg.get("dataset_hdfs_path", DEFAULT_DATASET_HDFS_PATH)
    output_hdfs_path = hadoop_cfg.get("output_hdfs_path", DEFAULT_OUTPUT_HDFS_PATH)

    hdfs_input = env_input or _build_hdfs_uri(namenode_uri, f"{dataset_hdfs_path.rstrip('/')}/Books_rating.csv")
    hdfs_output = env_output or _build_hdfs_uri(namenode_uri, f"{output_hdfs_path.rstrip('/')}/processed")
    return hdfs_input, hdfs_output


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

    hdfs_input, hdfs_output = _resolve_hdfs_io_paths()

    print(f"[INFO] Membaca dataset dari HDFS: {hdfs_input}")
    df = spark.read.csv(
        hdfs_input,
        header=True,
        inferSchema=True,
        multiLine=True,
        escape='"',
    )

    print(f"[INFO] Total baris sebelum filter: {df.count()}")

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

    if SAMPLE_FRACTION < 1.0:
        df = df.sample(fraction=SAMPLE_FRACTION, seed=42)

    total = df.count()
    print(f"[INFO] Total baris setelah filter & cleaning: {total}")

    # Distribusi sentimen
    print("[INFO] Distribusi label sentimen:")
    df.groupBy("sentiment_text").count().orderBy("count", ascending=False).show()

    # Simpan ke HDFS sebagai Parquet (efisien untuk dataset besar)
    print(f"[INFO] Menyimpan hasil ke HDFS: {hdfs_output}")
    df.write.mode("overwrite").parquet(hdfs_output)
    print("[INFO] Selesai. Data tersimpan di HDFS.")

    spark.stop()


if __name__ == "__main__":
    main()
