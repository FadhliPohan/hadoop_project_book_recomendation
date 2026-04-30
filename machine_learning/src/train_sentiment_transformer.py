from __future__ import annotations

from contextlib import nullcontext
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .mlflow_tracker import log_artifact, log_metrics, log_params, start_run
from .utils import append_model_registry, ensure_dir, resolve_path, save_json

LABEL_NAMES = ["Negative", "Neutral", "Positive"]
LABEL_ORDER = [0, 1, 2]


class TransformerDataset:
    """Simple torch dataset wrapper for tokenized text classification data."""

    def __init__(self, encodings: Dict[str, np.ndarray], labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict:
        import torch

        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


def _load_splits(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(resolve_path(config, "train_csv"))
    val = pd.read_csv(resolve_path(config, "validation_csv"))
    test = pd.read_csv(resolve_path(config, "test_csv"))
    return train, val, test


def _build_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=LABEL_ORDER,
            target_names=LABEL_NAMES,
            digits=4,
            output_dict=True,
            zero_division=0,
        ),
    }


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(cmap="Blues", values_format="d")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_transformer_sentiment(config: Dict) -> Dict:
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:
        raise RuntimeError(
            "Dependency transformer belum terpasang. Install dulu: pip install transformers torch"
        ) from exc

    train_df, val_df, test_df = _load_splits(config)

    transformer_cfg = config.get("transformer", {})
    model_name = transformer_cfg.get("model_name", "distilbert-base-uncased")
    max_length = int(transformer_cfg.get("max_length", 128))
    batch_size = int(transformer_cfg.get("batch_size", 16))
    learning_rate = float(transformer_cfg.get("learning_rate", 2e-5))
    epochs = int(transformer_cfg.get("epochs", 3))
    early_stop_patience = int(transformer_cfg.get("early_stopping_patience", 2))
    weight_decay = float(transformer_cfg.get("weight_decay", 0.01))

    out_dir = resolve_path(config, "sentiment_transformer_dir")
    checkpoint_dir = out_dir / "checkpoints"
    model_dir = out_dir / "model"
    tokenizer_dir = out_dir / "tokenizer"
    ensure_dir(checkpoint_dir)
    ensure_dir(model_dir)
    ensure_dir(tokenizer_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_texts(texts: pd.Series) -> Dict[str, np.ndarray]:
        return tokenizer(
            texts.fillna("").astype(str).tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_enc = tokenize_texts(train_df["review_text_processed"])
    val_enc = tokenize_texts(val_df["review_text_processed"])
    test_enc = tokenize_texts(test_df["review_text_processed"])

    train_dataset = TransformerDataset(train_enc, train_df["sentiment_label"].astype(int).to_numpy())
    val_dataset = TransformerDataset(val_enc, val_df["sentiment_label"].astype(int).to_numpy())
    test_labels = test_df["sentiment_label"].astype(int).to_numpy()
    test_dataset = TransformerDataset(test_enc, test_labels)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "precision_weighted": float(precision_score(labels, preds, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(labels, preds, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        }

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        save_total_limit=2,
        logging_strategy="epoch",
        report_to=[],
        seed=int(config["project"]["random_state"]),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stop_patience)],
    )

    run_name = "distilbert_sentiment_v1"
    run_ctx = start_run(config, run_name) or nullcontext()

    with run_ctx:
        logging.info("Training transformer model: %s", model_name)
        trainer.train()

        val_metrics_raw = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="validation")

        test_predictions = trainer.predict(test_dataset)
        test_pred_labels = np.argmax(test_predictions.predictions, axis=-1)

        validation_metrics = {
            "accuracy": float(val_metrics_raw.get("validation_accuracy", 0.0)),
            "precision_weighted": float(val_metrics_raw.get("validation_precision_weighted", 0.0)),
            "recall_weighted": float(val_metrics_raw.get("validation_recall_weighted", 0.0)),
            "f1_weighted": float(val_metrics_raw.get("validation_f1_weighted", 0.0)),
        }

        test_metrics = _build_metrics(test_labels, test_pred_labels)

        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(tokenizer_dir))

        confusion_png = out_dir / "confusion_matrix_test.png"
        _plot_confusion_matrix(test_labels, test_pred_labels, confusion_png)

        classification_txt = out_dir / "classification_report_test.txt"
        classification_txt.write_text(
            classification_report(
                test_labels,
                test_pred_labels,
                labels=LABEL_ORDER,
                target_names=LABEL_NAMES,
                digits=4,
                zero_division=0,
            ),
            encoding="utf-8",
        )

        training_args_json = out_dir / "training_args.json"
        save_json(training_args.to_dict(), training_args_json)

        metrics_payload = {
            "model_name": model_name,
            "validation": validation_metrics,
            "test": test_metrics,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        metrics_json = out_dir / "metrics.json"
        save_json(metrics_payload, metrics_json)

        append_model_registry(
            config,
            {
                "name": run_name,
                "task": "sentiment_transformer",
                "trained_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": str(resolve_path(config, "processed_reviews_csv")),
                "metrics": metrics_payload,
                "version": "v1",
                "path": str(model_dir),
                "hyperparameters": {
                    "model_name": model_name,
                    "max_length": max_length,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "epochs": epochs,
                    "weight_decay": weight_decay,
                    "early_stopping_patience": early_stop_patience,
                },
            },
        )

        log_params(
            config,
            {
                "model_name": model_name,
                "max_length": max_length,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "weight_decay": weight_decay,
                "early_stopping_patience": early_stop_patience,
            },
        )
        log_metrics(
            config,
            {
                "val_accuracy": validation_metrics["accuracy"],
                "val_precision_weighted": validation_metrics["precision_weighted"],
                "val_recall_weighted": validation_metrics["recall_weighted"],
                "val_f1_weighted": validation_metrics["f1_weighted"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision_weighted": test_metrics["precision_weighted"],
                "test_recall_weighted": test_metrics["recall_weighted"],
                "test_f1_weighted": test_metrics["f1_weighted"],
            },
        )
        log_artifact(config, metrics_json, artifact_path="sentiment_transformer")
        log_artifact(config, training_args_json, artifact_path="sentiment_transformer")
        log_artifact(config, confusion_png, artifact_path="sentiment_transformer")
        log_artifact(config, classification_txt, artifact_path="sentiment_transformer")

        return metrics_payload
