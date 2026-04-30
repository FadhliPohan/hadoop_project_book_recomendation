from __future__ import annotations

import logging
from pathlib import Path
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

    df["item_id"] = df["item_id"].astype(str).str.strip()
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["review_text"] = df["review_text"].astype(str)
    df["summary"] = df["summary"].fillna("").astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["item_id", "user_id", "rating", "review_text"]).copy()
    df = df[df["review_text"].str.len() >= config["data"].get("min_review_text_length", 5)].copy()
    logging.info("Dropped %s invalid rows", before - len(df))

    return df


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
