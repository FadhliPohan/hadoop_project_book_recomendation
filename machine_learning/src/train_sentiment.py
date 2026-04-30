from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .utils import append_model_registry, ensure_dir, resolve_path

LABEL_ORDER = [0, 1, 2]


def _load_splits(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(resolve_path(config, "train_csv"))
    val = pd.read_csv(resolve_path(config, "validation_csv"))
    test = pd.read_csv(resolve_path(config, "test_csv"))
    return train, val, test


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": report,
    }


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Neutral", "Positive"])
    disp.plot(cmap="Blues", values_format="d")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def train_baseline_sentiment(config: Dict) -> Dict:
    out_dir = resolve_path(config, "sentiment_dir")
    ensure_dir(out_dir)

    train_df, val_df, test_df = _load_splits(config)
    x_train = train_df["review_text_processed"].fillna("")
    y_train = train_df["sentiment_label"].astype(int)

    x_val = val_df["review_text_processed"].fillna("")
    y_val = val_df["sentiment_label"].astype(int)

    x_test = test_df["review_text_processed"].fillna("")
    y_test = test_df["sentiment_label"].astype(int)

    tfidf_max_features = config["sentiment"]["tfidf_max_features"]
    tfidf_min_df = config["sentiment"]["tfidf_min_df"]
    ngram_range = tuple(config["sentiment"]["ngram_range"])
    random_state = config["project"]["random_state"]

    models = {
        "tfidf_logreg_v1": LogisticRegression(max_iter=300, class_weight="balanced", random_state=random_state),
        "tfidf_nb_v1": MultinomialNB(),
        "tfidf_svm_v1": LinearSVC(class_weight="balanced", random_state=random_state),
    }

    all_metrics: Dict[str, Dict] = {}
    best_model_name = ""
    best_val_f1 = -1.0

    for model_name, estimator in models.items():
        logging.info("Training baseline model: %s", model_name)
        pipeline = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=tfidf_max_features,
                        min_df=tfidf_min_df,
                        ngram_range=ngram_range,
                    ),
                ),
                ("classifier", estimator),
            ]
        )

        pipeline.fit(x_train, y_train)

        val_pred = pipeline.predict(x_val)
        test_pred = pipeline.predict(x_test)

        val_metrics = _metrics(y_val.to_numpy(), val_pred)
        test_metrics = _metrics(y_test.to_numpy(), test_pred)

        model_path = out_dir / f"{model_name}.pkl"
        joblib.dump(pipeline, model_path)

        _plot_confusion(y_test.to_numpy(), test_pred, out_dir / f"{model_name}_confusion_matrix_test.png")
        _plot_confusion(y_val.to_numpy(), val_pred, out_dir / f"{model_name}_confusion_matrix_validation.png")

        report_txt_path = out_dir / f"{model_name}_classification_report.txt"
        report_txt_path.write_text(
            classification_report(y_test.to_numpy(), test_pred, digits=4),
            encoding="utf-8",
        )

        all_metrics[model_name] = {
            "validation": val_metrics,
            "test": test_metrics,
            "model_path": str(model_path),
        }

        if val_metrics["f1_weighted"] > best_val_f1:
            best_val_f1 = val_metrics["f1_weighted"]
            best_model_name = model_name

        append_model_registry(
            config,
            {
                "name": model_name,
                "task": "sentiment_baseline",
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": str(resolve_path(config, "processed_reviews_csv")),
                "metrics": test_metrics,
                "version": "v1",
                "path": str(model_path),
                "hyperparameters": {
                    "tfidf_max_features": tfidf_max_features,
                    "tfidf_min_df": tfidf_min_df,
                    "ngram_range": ngram_range,
                },
            },
        )

    summary = {
        "best_model": best_model_name,
        "best_validation_f1_weighted": best_val_f1,
        "metrics": all_metrics,
    }

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logging.info("Saved sentiment metrics: %s", metrics_path)
    return summary
