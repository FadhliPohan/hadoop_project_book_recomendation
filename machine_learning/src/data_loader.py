from __future__ import annotations

import logging
import subprocess
import tempfile
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


def _build_hdfs_uri(namenode_uri: str, hdfs_path: str) -> str:
    normalized_namenode = namenode_uri.rstrip("/")
    normalized_path = hdfs_path if hdfs_path.startswith("/") else f"/{hdfs_path}"
    return f"{normalized_namenode}{normalized_path}"


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

    with tempfile.TemporaryDirectory(prefix="spark_hdfs_bridge_", dir="/tmp") as temp_dir:
        local_root = Path(temp_dir)
        fetch = subprocess.run(
            ["hdfs", "dfs", "-get", "-f", parquet_uri, str(local_root)],
            capture_output=True,
            text=True,
        )
        if fetch.returncode != 0:
            raise RuntimeError(
                "Gagal mengambil hasil preprocessing Spark dari HDFS.\n"
                f"Command: hdfs dfs -get -f {parquet_uri} {local_root}\n"
                f"stderr: {fetch.stderr.strip()}"
            )

        local_parquet_dir = local_root / PurePosixPath(parquet_uri).name
        if not local_parquet_dir.exists():
            raise FileNotFoundError(
                f"Direktori Parquet hasil download tidak ditemukan: {local_parquet_dir}"
            )

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
