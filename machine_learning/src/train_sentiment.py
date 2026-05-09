from __future__ import annotations

from contextlib import nullcontext
import gc
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

from .mlflow_tracker import log_artifact, log_metrics, log_params, start_run
from .utils import append_model_registry, ensure_dir, resolve_path

LABEL_ORDER = [0, 1, 2]
SENTIMENT_SPLIT_COLUMNS = ["review_text_processed", "sentiment_label"]


def _resolve_dtype(dtype_name: str) -> np.dtype:
    lowered = str(dtype_name).strip().lower()
    if lowered == "float32":
        return np.float32
    if lowered == "float64":
        return np.float64
    raise ValueError(f"Unsupported TF-IDF dtype: {dtype_name}")


def _build_tfidf_profiles(config: Dict) -> list[Dict]:
    sentiment_cfg = config.get("sentiment", {})
    dtype = _resolve_dtype(sentiment_cfg.get("tfidf_dtype", "float32"))
    max_features = int(sentiment_cfg["tfidf_max_features"])
    min_df = int(sentiment_cfg["tfidf_min_df"])
    ngram_range = tuple(sentiment_cfg["ngram_range"])
    max_df = float(sentiment_cfg.get("tfidf_max_df", 0.95))
    lower_memory = bool(sentiment_cfg.get("enable_memory_fallback", True))

    profiles: list[Dict] = [
        {
            "name": "primary",
            "max_features": max_features,
            "min_df": min_df,
            "ngram_range": ngram_range,
            "dtype": dtype,
            "max_df": max_df,
        }
    ]

    if not lower_memory:
        return profiles

    fallback_candidates = [
        {
            "name": "fallback_1",
            "max_features": min(max_features, 8000),
            "min_df": max(min_df, 3),
            "ngram_range": (1, 1),
            "dtype": np.float32,
            "max_df": max_df,
        },
        {
            "name": "fallback_2",
            "max_features": min(max_features, 5000),
            "min_df": max(min_df, 5),
            "ngram_range": (1, 1),
            "dtype": np.float32,
            "max_df": max_df,
        },
    ]

    seen_keys = {
        (
            profiles[0]["max_features"],
            profiles[0]["min_df"],
            profiles[0]["ngram_range"],
            profiles[0]["dtype"],
            profiles[0]["max_df"],
        )
    }
    for candidate in fallback_candidates:
        key = (
            candidate["max_features"],
            candidate["min_df"],
            candidate["ngram_range"],
            candidate["dtype"],
            candidate["max_df"],
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        profiles.append(candidate)
    return profiles


def _read_split_csv(path: Path) -> pd.DataFrame:
    read_kwargs = {
        "usecols": SENTIMENT_SPLIT_COLUMNS,
        "dtype": {"sentiment_label": "int8"},
        "low_memory": True,
    }
    try:
        return pd.read_csv(path, **read_kwargs)
    except pd.errors.ParserError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        logging.warning("Parser C kehabisan memori saat baca %s. Fallback ke engine='python'.", path)
        return pd.read_csv(path, engine="python", **read_kwargs)
    except MemoryError:
        logging.warning("MemoryError saat baca %s. Fallback ke engine='python'.", path)
        return pd.read_csv(path, engine="python", **read_kwargs)


def _load_splits(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = _read_split_csv(resolve_path(config, "train_csv"))
    val = _read_split_csv(resolve_path(config, "validation_csv"))
    test = _read_split_csv(resolve_path(config, "test_csv"))
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

    tfidf_profiles = _build_tfidf_profiles(config)
    random_state = config["project"]["random_state"]

    models = {
        "tfidf_logreg_v1": LogisticRegression(max_iter=300, class_weight="balanced", random_state=random_state),
        "tfidf_nb_v1": MultinomialNB(),
        "tfidf_svm_v1": LinearSVC(class_weight="balanced", random_state=random_state, dual=False),
    }

    all_metrics: Dict[str, Dict] = {}
    best_model_name = ""
    best_val_f1 = -1.0

    for model_name, estimator in models.items():
        run_ctx = start_run(config, model_name) or nullcontext()
        with run_ctx:
            logging.info("Training baseline model: %s", model_name)
            pipeline = None
            active_profile = None
            last_error: Exception | None = None
            for profile in tfidf_profiles:
                logging.info(
                    "Trying TF-IDF profile '%s' for %s (max_features=%s, min_df=%s, ngram=%s, dtype=%s, max_df=%.3f)",
                    profile["name"],
                    model_name,
                    profile["max_features"],
                    profile["min_df"],
                    profile["ngram_range"],
                    np.dtype(profile["dtype"]).name,
                    profile["max_df"],
                )
                candidate = Pipeline(
                    steps=[
                        (
                            "tfidf",
                            TfidfVectorizer(
                                max_features=profile["max_features"],
                                min_df=profile["min_df"],
                                ngram_range=profile["ngram_range"],
                                dtype=profile["dtype"],
                                max_df=profile["max_df"],
                            ),
                        ),
                        ("classifier", estimator),
                    ]
                )
                try:
                    candidate.fit(x_train, y_train)
                    pipeline = candidate
                    active_profile = profile
                    break
                except Exception as err:
                    is_memory_error = isinstance(err, MemoryError) or (
                        isinstance(err, ValueError)
                        and (
                            "unable to allocate" in str(err).lower()
                            or "array is too big" in str(err).lower()
                        )
                    )
                    if not is_memory_error:
                        raise
                    last_error = err
                    del candidate
                    gc.collect()
                    logging.warning(
                        "Memory error while fitting %s with TF-IDF profile '%s'. Trying a lighter profile.",
                        model_name,
                        profile["name"],
                    )

            if pipeline is None or active_profile is None:
                if last_error is not None:
                    raise last_error
                raise RuntimeError(f"Training failed for {model_name}: no TF-IDF profile could be fitted.")

            val_pred = pipeline.predict(x_val)
            test_pred = pipeline.predict(x_test)

            val_metrics = _metrics(y_val.to_numpy(), val_pred)
            test_metrics = _metrics(y_test.to_numpy(), test_pred)

            model_path = out_dir / f"{model_name}.pkl"
            joblib.dump(pipeline, model_path)

            cm_test_path = out_dir / f"{model_name}_confusion_matrix_test.png"
            cm_val_path = out_dir / f"{model_name}_confusion_matrix_validation.png"
            _plot_confusion(y_test.to_numpy(), test_pred, cm_test_path)
            _plot_confusion(y_val.to_numpy(), val_pred, cm_val_path)

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
                        "tfidf_profile": active_profile["name"],
                        "tfidf_max_features": active_profile["max_features"],
                        "tfidf_min_df": active_profile["min_df"],
                        "ngram_range": active_profile["ngram_range"],
                        "tfidf_dtype": np.dtype(active_profile["dtype"]).name,
                        "tfidf_max_df": active_profile["max_df"],
                    },
                },
            )

            log_params(
                config,
                {
                    "model_name": model_name,
                    "tfidf_profile": active_profile["name"],
                    "tfidf_max_features": active_profile["max_features"],
                    "tfidf_min_df": active_profile["min_df"],
                    "ngram_range": active_profile["ngram_range"],
                    "tfidf_dtype": np.dtype(active_profile["dtype"]).name,
                    "tfidf_max_df": active_profile["max_df"],
                },
            )
            log_metrics(
                config,
                {
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision_weighted": val_metrics["precision_weighted"],
                    "val_recall_weighted": val_metrics["recall_weighted"],
                    "val_f1_weighted": val_metrics["f1_weighted"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_precision_weighted": test_metrics["precision_weighted"],
                    "test_recall_weighted": test_metrics["recall_weighted"],
                    "test_f1_weighted": test_metrics["f1_weighted"],
                },
            )
            log_artifact(config, model_path, artifact_path=f"sentiment_baseline/{model_name}")
            log_artifact(config, cm_test_path, artifact_path=f"sentiment_baseline/{model_name}")
            log_artifact(config, cm_val_path, artifact_path=f"sentiment_baseline/{model_name}")
            log_artifact(config, report_txt_path, artifact_path=f"sentiment_baseline/{model_name}")
            gc.collect()

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
