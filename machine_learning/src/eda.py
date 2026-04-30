from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from .data_loader import load_reviews
from .utils import ensure_dir, resolve_path


def _save_series(series: pd.Series, path: Path, index_name: str, value_name: str) -> None:
    out = series.rename(value_name).reset_index().rename(columns={"index": index_name})
    out.to_csv(path, index=False)


def run_eda(config: Dict) -> Dict:
    eda_dir = resolve_path(config, "eda_dir")
    ensure_dir(eda_dir)

    df = load_reviews(config)
    logging.info("EDA on %s rows", len(df))

    missing = df.isna().sum().sort_values(ascending=False)
    duplicates = int(df.duplicated(subset=["user_id", "item_id", "review_text", "rating"]).sum())

    rating_dist = df["rating"].value_counts().sort_index()
    df["review_length_words"] = df["review_text"].str.split().str.len()

    review_length_summary = df["review_length_words"].describe()
    reviews_per_user = df.groupby("user_id").size()
    reviews_per_item = df.groupby("item_id").size()

    basic_stats = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "unique_users": int(df["user_id"].nunique()),
        "unique_items": int(df["item_id"].nunique()),
        "duplicate_rows": duplicates,
        "avg_rating": float(df["rating"].mean()),
        "median_rating": float(df["rating"].median()),
    }

    pd.DataFrame(list(basic_stats.items()), columns=["metric", "value"]).to_csv(
        eda_dir / "basic_stats.csv", index=False
    )
    missing.rename("missing_count").to_csv(eda_dir / "missing_values.csv", header=True)
    _save_series(rating_dist, eda_dir / "rating_distribution.csv", "rating", "count")
    review_length_summary.rename("value").to_csv(eda_dir / "review_length_summary.csv", header=True)
    reviews_per_user.describe().rename("value").to_csv(
        eda_dir / "reviews_per_user_summary.csv", header=True
    )
    reviews_per_item.describe().rename("value").to_csv(
        eda_dir / "reviews_per_item_summary.csv", header=True
    )

    plt.figure(figsize=(8, 5))
    plt.bar(rating_dist.index.astype(str), rating_dist.values)
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(eda_dir / "rating_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["review_length_words"], bins=50)
    plt.title("Review Length Distribution (Words)")
    plt.xlabel("Words per Review")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(eda_dir / "review_length_distribution.png", dpi=150)
    plt.close()

    examples = []
    for label_name, selector in [
        ("Negative", df["rating"].isin([1, 2])),
        ("Neutral", df["rating"] == 3),
        ("Positive", df["rating"].isin([4, 5])),
    ]:
        sampled = df.loc[selector, ["rating", "review_text"]].head(3).copy()
        sampled["sentiment_bucket"] = label_name
        examples.append(sampled)

    if examples:
        pd.concat(examples, ignore_index=True).to_csv(eda_dir / "sentiment_examples.csv", index=False)

    insight_lines = [
        f"Total rows analysed: {basic_stats['rows']}",
        f"Unique users: {basic_stats['unique_users']}",
        f"Unique books/items: {basic_stats['unique_items']}",
        f"Duplicate reviews (user, item, text, rating): {duplicates}",
        f"Average rating: {basic_stats['avg_rating']:.4f}",
        f"Median rating: {basic_stats['median_rating']:.4f}",
        "Rating distribution cenderung tidak seimbang jika kelas 4-5 dominan.",
        "Panjang review bervariasi; perlu preprocessing agar fitur teks stabil.",
    ]
    (eda_dir / "eda_summary.txt").write_text("\n".join(insight_lines), encoding="utf-8")

    return basic_stats
