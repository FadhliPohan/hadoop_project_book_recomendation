from __future__ import annotations

import argparse
import logging

from src.utils import load_config, setup_logging
from src.training_runtime import (
    apply_master_ram_limit,
    compare_training_modes,
    resolve_mode_preprocess_source,
    run_worker_preprocess_submit,
    run_training_pipeline,
)


TRAINING_STEPS = {
    "train_sentiment_baseline",
    "train_sentiment_transformer",
    "train_recommender",
    "train_pipeline",
    "compare_training_modes",
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
            "train_pipeline",
            "compare_training_modes",
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
    parser.add_argument(
        "--training-mode",
        choices=["without_worker", "with_worker"],
        default="without_worker",
        help=(
            "Mode training pipeline. "
            "`without_worker`: preprocessing + training full di master. "
            "`with_worker`: preprocessing awal via Spark worker (HDFS), training tetap di master."
        ),
    )
    parser.add_argument(
        "--include-transformer",
        action="store_true",
        help="Jika di-set, training pipeline juga menjalankan step transformer.",
    )
    parser.add_argument(
        "--run-worker-preprocess",
        action="store_true",
        help=(
            "Khusus training mode `with_worker`: jalankan spark submit preprocess_spark terlebih dahulu "
            "sebelum bridge preprocess lokal."
        ),
    )
    parser.add_argument(
        "--ram-limit-gb",
        type=float,
        default=None,
        help="Override batas RAM master (GB) untuk proses training. Default baca dari config.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    setup_logging()

    logging.info("Running step: %s", args.step)
    logging.info("Preprocess source: %s", args.preprocess_source)
    logging.info("Training mode: %s", args.training_mode)
    if args.step in TRAINING_STEPS and not args.allow_training:
        raise ValueError(
            "Step training diblokir untuk mencegah training otomatis. "
            "Gunakan --allow-training jika memang ingin training."
        )

    training_steps_with_direct_local_train = {
        "train_sentiment_baseline",
        "train_sentiment_transformer",
        "train_recommender",
    }
    if args.step in training_steps_with_direct_local_train:
        apply_master_ram_limit(config, ram_limit_gb=args.ram_limit_gb)

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

    if args.step == "train_pipeline":
        run_training_pipeline(
            config,
            training_mode=args.training_mode,
            include_transformer=args.include_transformer,
            run_worker_preprocess=args.run_worker_preprocess,
            ram_limit_gb=args.ram_limit_gb,
        )
        return

    if args.step == "compare_training_modes":
        compare_training_modes(
            config,
            include_transformer=args.include_transformer,
            run_worker_preprocess=args.run_worker_preprocess,
            ram_limit_gb=args.ram_limit_gb,
        )
        return

    if args.step == "evaluate":
        from src.evaluate import compile_final_report

        compile_final_report(config)
        return

    if args.step == "all":
        from src.eda import run_eda

        source = args.preprocess_source
        run_worker_preprocess_for_pipeline = args.run_worker_preprocess
        if args.allow_training:
            logging.info("Progress ALL: 0%% | Mulai pipeline all (training enabled).")
        if args.allow_training:
            source = resolve_mode_preprocess_source(args.training_mode)
            if args.training_mode == "with_worker" and args.run_worker_preprocess:
                logging.info(
                    "Running Spark worker preprocess before EDA "
                    "because source=spark_hdfs and --run-worker-preprocess aktif."
                )
                run_worker_preprocess_submit(config)
                run_worker_preprocess_for_pipeline = False
                logging.info("Progress ALL: 25%% | Worker preprocess selesai.")

        # Default mode aman: no-training. Cocok untuk iterasi coding.
        if args.allow_training:
            logging.info("Progress ALL: 30%% | Mulai EDA.")
        run_eda(config, source=source)
        if args.allow_training:
            logging.info("Progress ALL: 45%% | EDA selesai. Lanjut training pipeline.")
        if args.allow_training:
            run_training_pipeline(
                config,
                training_mode=args.training_mode,
                include_transformer=args.include_transformer,
                run_worker_preprocess=run_worker_preprocess_for_pipeline,
                ram_limit_gb=args.ram_limit_gb,
            )
            logging.info("Progress ALL: 100%% | Pipeline all selesai.")
            logging.info("Pipeline all selesai dengan mode training=%s", args.training_mode)
            return

        from src.preprocessing import preprocess_and_split

        preprocess_and_split(config, source=source)
        from src.evaluate import compile_final_report

        compile_final_report(config)
        logging.info("Pipeline all selesai")


if __name__ == "__main__":
    main()
