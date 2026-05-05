from __future__ import annotations

import argparse
import logging

from src.utils import load_config, setup_logging


TRAINING_STEPS = {
    "train_sentiment_baseline",
    "train_sentiment_transformer",
    "train_recommender",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Amazon Books ML Pipeline")
    parser.add_argument(
        "--step",
        required=True,
        choices=[
            "eda",
            "preprocess",
            "train_sentiment_baseline",
            "train_sentiment_transformer",
            "train_recommender",
            "evaluate",
            "all",
        ],
        help="Pipeline step to run",
    )
    parser.add_argument(
        "--allow-training",
        action="store_true",
        help="Wajib di-set jika ingin menjalankan step training.",
    )
    parser.add_argument(
        "--preprocess-source",
        choices=["local_dataset", "spark_hdfs"],
        default="local_dataset",
        help=(
            "Sumber data untuk step preprocess. "
            "`local_dataset` membaca CSV lokal mentah, "
            "`spark_hdfs` membaca hasil distributed preprocessing dari HDFS lalu membuat split lokal."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    setup_logging()

    logging.info("Running step: %s", args.step)
    logging.info("Preprocess source: %s", args.preprocess_source)
    if args.step in TRAINING_STEPS and not args.allow_training:
        raise ValueError(
            "Step training diblokir untuk mencegah training otomatis. "
            "Gunakan --allow-training jika memang ingin training."
        )

    if args.step == "eda":
        from src.eda import run_eda

        run_eda(config, source=args.preprocess_source)
        return

    if args.step == "preprocess":
        from src.preprocessing import preprocess_and_split

        preprocess_and_split(config, source=args.preprocess_source)
        return

    if args.step == "train_sentiment_baseline":
        from src.train_sentiment import train_baseline_sentiment

        train_baseline_sentiment(config)
        return

    if args.step == "train_sentiment_transformer":
        from src.train_sentiment_transformer import train_transformer_sentiment

        train_transformer_sentiment(config)
        return

    if args.step == "train_recommender":
        from src.train_recommender import train_recommenders

        train_recommenders(config)
        return

    if args.step == "evaluate":
        from src.evaluate import compile_final_report

        compile_final_report(config)
        return

    if args.step == "all":
        from src.eda import run_eda
        from src.evaluate import compile_final_report
        from src.preprocessing import preprocess_and_split

        # Default mode aman: no-training. Cocok untuk iterasi coding.
        run_eda(config, source=args.preprocess_source)
        preprocess_and_split(config, source=args.preprocess_source)
        if args.allow_training:
            from src.train_recommender import train_recommenders
            from src.train_sentiment import train_baseline_sentiment

            train_baseline_sentiment(config)
            train_recommenders(config)
        compile_final_report(config)
        logging.info("Pipeline all selesai")


if __name__ == "__main__":
    main()
