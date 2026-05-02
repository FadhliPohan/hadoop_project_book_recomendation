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

import re
import sys
from pathlib import Path

# Config defaults (bisa di-override via env var)
HDFS_INPUT = "hdfs://fadhli:9000/data/amazon_books/Books_rating.csv"
HDFS_OUTPUT = "hdfs://fadhli:9000/output/amazon_books_ml/processed"
SAMPLE_FRACTION = 1.0  # 1.0 = pakai semua data, 0.1 = 10% untuk test cepat


def main() -> None:
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.types import IntegerType, StringType
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

    print(f"[INFO] Membaca dataset dari HDFS: {HDFS_INPUT}")
    df = spark.read.csv(
        HDFS_INPUT,
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
    print(f"[INFO] Menyimpan hasil ke HDFS: {HDFS_OUTPUT}")
    df.write.mode("overwrite").parquet(HDFS_OUTPUT)
    print("[INFO] Selesai. Data tersimpan di HDFS.")

    spark.stop()


if __name__ == "__main__":
    main()
