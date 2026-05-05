from __future__ import annotations

import html
import logging
import re
from datetime import datetime, timezone
from typing import Dict, Set, Tuple

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

from .data_loader import load_reviews, load_reviews_from_hdfs_spark_output
from .utils import ensure_dir, resolve_path, save_json

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<[^>]+>")
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")

SENTIMENT_ID = {"Negative": 0, "Neutral": 1, "Positive": 2}
SENTIMENT_TEXT = {0: "Negative", 1: "Neutral", 2: "Positive"}


def _ensure_nltk_assets() -> None:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def rating_to_sentiment_label(rating: float) -> int:
    if rating <= 2:
        return SENTIMENT_ID["Negative"]
    if rating == 3:
        return SENTIMENT_ID["Neutral"]
    return SENTIMENT_ID["Positive"]


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = html.unescape(text)
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = NON_ALPHA_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()
    return text


def remove_stopwords(text: str, stop_words: Set[str]) -> str:
    tokens = text.split()
    return " ".join(token for token in tokens if token not in stop_words)


def lemmatize_text(text: str, lemmatizer: WordNetLemmatizer) -> str:
    tokens = text.split()
    return " ".join(lemmatizer.lemmatize(token) for token in tokens)


def preprocess_text(text: str, stop_words: set[str], lemmatizer: WordNetLemmatizer) -> str:
    cleaned = clean_text(text)
    no_stop = remove_stopwords(cleaned, stop_words)
    lemma = lemmatize_text(no_stop, lemmatizer)
    return MULTI_SPACE_PATTERN.sub(" ", lemma).strip()


def _load_preprocess_source(config: Dict, source: str) -> pd.DataFrame:
    sample_rows = config["data"].get("sample_rows")
    if source == "spark_hdfs":
        return load_reviews_from_hdfs_spark_output(config, sample_rows=sample_rows)
    if source == "local_dataset":
        return load_reviews(config, sample_rows=sample_rows)
    raise ValueError(f"Unknown preprocess source: {source}")


def preprocess_and_split(
    config: Dict,
    source: str = "local_dataset",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ensure_nltk_assets()

    logging.info("Preprocess source selected: %s", source)
    df = _load_preprocess_source(config, source)
    df["sentiment_label"] = df["rating"].apply(rating_to_sentiment_label)
    df["sentiment_text"] = df["sentiment_label"].map(SENTIMENT_TEXT)

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    logging.info("Preprocessing review text")
    base_text_series = (
        df["review_text_clean"]
        if source == "spark_hdfs" and "review_text_clean" in df.columns
        else df["review_text"]
    )
    df["review_text_processed"] = base_text_series.fillna("").apply(
        lambda text: preprocess_text(text, stop_words, lemmatizer)
    )

    processed_path = resolve_path(config, "processed_reviews_csv")
    ensure_dir(processed_path.parent)
    df.to_csv(processed_path, index=False)
    logging.info("Saved processed dataset to %s", processed_path)

    test_size = config["preprocessing"]["test_size"]
    val_size = config["preprocessing"]["validation_size"]
    random_state = config["project"]["random_state"]

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["sentiment_label"],
        random_state=random_state,
    )

    adjusted_val_size = val_size / (1 - test_size)
    train, validation = train_test_split(
        train_val,
        test_size=adjusted_val_size,
        stratify=train_val["sentiment_label"],
        random_state=random_state,
    )

    train_path = resolve_path(config, "train_csv")
    val_path = resolve_path(config, "validation_csv")
    test_path = resolve_path(config, "test_csv")

    train.to_csv(train_path, index=False)
    validation.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)

    logging.info("Saved splits train=%s val=%s test=%s", len(train), len(validation), len(test))

    metadata_path = processed_path.parent / "preprocess_metadata.json"
    save_json(
        {
            "source": source,
            "rows_processed": int(len(df)),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        metadata_path,
    )
    logging.info("Saved preprocessing metadata to %s", metadata_path)

    return train, validation, test
