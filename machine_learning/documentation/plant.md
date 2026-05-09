# Technical Design Machine Learning Module (As-Is)

Dokumen ini menjelaskan desain implementasi modul ML berdasarkan kode aktif di folder `machine_learning/`.

Tanggal pembaruan: 2026-05-09.

## 1. Tujuan Modul ML

1. Menyiapkan data review untuk analisis sentimen dan rekomendasi.
2. Melatih model sentiment baseline dan transformer.
3. Melatih model recommender hybrid.
4. Menyimpan artifact model, metrics, dan report komparasi eksperimen.

## 2. Entry Point dan Orkestrasi

- Entry CLI: `machine_learning/main.py`
- Orchestrator mode training: `src/training_runtime.py`

Step yang tersedia:
- `eda`
- `preprocess`
- `train_sentiment_baseline`
- `train_sentiment_transformer`
- `train_recommender`
- `train_pipeline`
- `compare_training_modes`
- `evaluate`
- `all`

Guard keamanan:
- Step training wajib `--allow-training`.

## 3. Sumber Data

### 3.1 Local source

- `machine_learning/dataset/Books_rating.csv`
- dimuat oleh `src/data_loader.py::load_reviews`.

### 3.2 Spark-HDFS source

- Hasil `spark_preprocess.py` pada HDFS path `.../output/.../processed`.
- dimuat oleh `src/data_loader.py::load_reviews_from_hdfs_spark_output`.
- menggunakan bridge HDFS -> local cache persisten.

## 4. Preprocessing

Modul: `src/preprocessing.py`

Tahapan:
1. Mapping rating -> sentiment label.
2. Cleaning text (lowercase, URL, HTML, non-alpha, whitespace).
3. Stopword removal.
4. Lemmatization.
5. Split stratified train/validation/test.

Output:
- `data/processed/processed_reviews.csv`
- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `data/processed/preprocess_metadata.json`

## 5. EDA

Modul: `src/eda.py`

Output utama:
- statistik dasar dataset,
- missing values,
- distribusi rating,
- distribusi panjang review,
- contoh review per bucket sentimen,
- insight ringkas.

Lokasi:
- `reports/eda/`

## 6. Sentiment Baseline

Modul: `src/train_sentiment.py`

Model:
- Logistic Regression
- Multinomial Naive Bayes
- LinearSVC

Fitur:
- TF-IDF dengan profil utama + fallback memory-safe.
- evaluasi validation/test (accuracy, precision, recall, F1, classification report).
- simpan confusion matrix dan model file.

Output:
- `models/sentiment/baseline/*.pkl`
- `models/sentiment/baseline/metrics.json`
- confusion matrix dan report txt per model.

## 7. Sentiment Transformer

Modul: `src/train_sentiment_transformer.py`

Implementasi:
- HuggingFace tokenizer + Trainer.
- early stopping callback.
- menyimpan model terbaik + tokenizer + training args + metrics.

Output:
- `models/sentiment/transformer/distilbert_v1/model/`
- `models/sentiment/transformer/distilbert_v1/tokenizer/`
- `models/sentiment/transformer/distilbert_v1/metrics.json`

## 8. Recommender Hybrid

Modul: `src/train_recommender.py`

Komponen:
1. Popularity-based scoring.
2. Collaborative filtering via SVD.
3. Content-based TF-IDF item text.
4. Hybrid scoring (`0.5 collab + 0.3 sentiment + 0.2 popularity`).

Evaluasi:
- RMSE
- MAE
- Precision@K
- Recall@K
- NDCG@K
- Coverage@K
- Diversity@K

Output:
- `models/recommender/*`
- `reports/recommender_metrics.json`

## 9. Runtime Training Pipeline

Modul: `src/training_runtime.py`

Fitur:
1. Menjalankan stage pipeline secara berurutan.
2. Logging progress dan ETA.
3. Capture durasi stage dan peak memory.
4. Enforcement RAM limit master (konfig default 3 GB).
5. Menyimpan run summary per mode.

Output:
- `reports/experiments/without_worker_latest_run.json`
- `reports/experiments/with_worker_latest_run.json`

## 10. Komparasi Mode Training

Method: `compare_training_modes` di `training_runtime.py`

Alur:
1. Jalankan mode `without_worker`.
2. Jalankan mode `with_worker`.
3. Hitung delta KPI (`with - without`).
4. Simpan error/warning per mode bila ada.

Output:
- `reports/training_mode_comparison.json`

## 11. Evaluasi Final dan Registry

### 11.1 Final report

Modul: `src/evaluate.py`

Kompilasi:
- sentiment baseline metrics,
- transformer metrics,
- recommender metrics,
- mode comparison report,
- model registry.

Output:
- `reports/final_report.json`

### 11.2 Model registry

Modul: `src/utils.py::append_model_registry`

Output:
- `models/model_registry.json`

## 12. Experiment Tracking

Modul: `src/mlflow_tracker.py`

Sifat:
- opsional,
- aman ketika MLflow disabled atau tidak terinstall.

Storage default:
- `machine_learning/mlruns/`

## 13. Konfigurasi Utama

File:
- `machine_learning/config.yaml`

Kelompok konfigurasi:
- `paths`
- `data`
- `preprocessing`
- `sentiment`
- `transformer`
- `recommender`
- `training`
- `hadoop`
- `spark`
- `spark_preprocess`

## 14. Kekuatan dan Batasan Implementasi

Kekuatan:
1. Pipeline modular dan reproducible.
2. Komparasi mode training terdokumentasi JSON.
3. Memory fallback untuk data besar sudah diimplementasikan.
4. Dashboard sudah bisa menjadi control panel operasional.

Batasan:
1. Training model belum distributed penuh di worker.
2. Throughput `with_worker` dipengaruhi performa bridge HDFS dan jaringan.
3. Perlu benchmark cluster stabil untuk angka eksperimen final.

## 15. Referensi Lanjutan

Untuk pemetaan fungsi tiap file secara rinci lihat:
- `documentation/structure.md`
