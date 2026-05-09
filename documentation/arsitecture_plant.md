# Architecture Plan (As-Is) Hadoop Amazon Books Reviews

Dokumen ini menjelaskan arsitektur aktual project berdasarkan implementasi kode saat ini, bukan rencana konseptual lama.

Tanggal pembaruan: 2026-05-09.

## 1. Tujuan Sistem

Sistem dibangun untuk:
- preprocessing dataset review buku Amazon,
- klasifikasi sentimen (baseline + transformer),
- rekomendasi buku hybrid,
- komparasi mode training (`without_worker` vs `with_worker`),
- kontrol pipeline melalui Streamlit.

## 2. Topologi Cluster

Komponen node:
- Master: `fadhli`
- Worker: `worker1`, `worker2`

Layanan:
- HDFS NameNode di master.
- HDFS DataNode di worker.
- YARN ResourceManager di master.
- YARN NodeManager di worker.
- Streamlit dashboard di master.

## 3. Arsitektur Eksekusi

### 3.1 Mode `without_worker`

Semua tahap berjalan di master:
1. Load CSV lokal.
2. Preprocess + split lokal.
3. Train sentiment baseline.
4. Optional train transformer.
5. Train recommender.
6. Compile final report.

### 3.2 Mode `with_worker`

Arsitektur hybrid:
1. Spark preprocess di worker (YARN executors) menulis output Parquet ke HDFS.
2. Master mengambil output Parquet melalui bridge HDFS->local (cache persisten).
3. Preprocess lanjutan lokal (lemmatization + split train/val/test).
4. Training model tetap di master.

Kesimpulan mode ini:
- distributed preprocessing: ya,
- distributed model training: belum.

## 4. Komponen Kode dan Peran

- `machine_learning/src/spark_preprocess.py`: preprocessing distributed.
- `machine_learning/src/data_loader.py`: bridge dan loader data local/HDFS.
- `machine_learning/src/preprocessing.py`: feature text lanjutan + split.
- `machine_learning/src/train_sentiment.py`: baseline sentiment.
- `machine_learning/src/train_sentiment_transformer.py`: fine-tuning transformer.
- `machine_learning/src/train_recommender.py`: recommender hybrid.
- `machine_learning/src/training_runtime.py`: orchestration training + compare mode.
- `streamlit/app.py`: UI operasional.
- `scripts/*.sh`: operasi cluster/network/reset/upload/submit.

## 5. Data Flow

### 5.1 Local data flow

```text
machine_learning/dataset/Books_rating.csv
  -> data_loader.load_reviews
  -> preprocessing.preprocess_and_split
  -> data/processed/*.csv
  -> training modules
  -> models/* + reports/*
```

### 5.2 Worker-assisted data flow

```text
HDFS /user/<user>/amazon_books/Books_rating.csv
  -> spark_preprocess.py (Spark on YARN)
  -> HDFS /user/<user>/output/amazon_books_ml/processed (Parquet)
  -> data_loader.load_reviews_from_hdfs_spark_output
  -> preprocess_and_split (local finalize)
  -> training modules (master)
```

## 6. Kontrol Operasional

### 6.1 CLI

Entry utama:
- `python3 machine_learning/main.py --step ...`

Step penting:
- `train_pipeline`
- `compare_training_modes`

### 6.2 Dashboard

`streamlit/app.py` menyediakan:
- Pipeline runner dengan live stdout/stderr.
- Cluster start/stop.
- Spark submit preprocess.
- Browse/upload/download HDFS.
- Reports komparasi mode detail.
- Sentiment dan recommender inference.

## 7. Manajemen Resource dan Stabilitas

Implementasi saat ini memiliki proteksi:

1. Batas RAM master (`training.master_ram_limit_gb`, default 3GB) via `RLIMIT_AS`.
2. Safety margin untuk mencegah limit diterapkan saat proses sudah dekat batas.
3. Fallback parser pada pembacaan CSV besar:
- sentiment split reader fallback ke `engine='python'`.
- recommender interactions loader membaca chunked + fallback engine.
4. Bridge HDFS dengan cache persisten agar run berikutnya tidak selalu download ulang.
5. Preflight YARN pada `spark_submit_training.sh` untuk mencegah job menggantung di ACCEPTED saat NodeManager 0.

## 8. Artifact Arsitektur

Output utama:
- Data processed: `machine_learning/data/processed/`
- Model: `machine_learning/models/`
- Report: `machine_learning/reports/`
- Eksperimen komparasi: `machine_learning/reports/experiments/`
- Log: `machine_learning/logs/`
- Tracking MLflow local: `machine_learning/mlruns/`

## 9. Risiko dan Batasan Saat Ini

1. Training sentiment/recommender belum distributed ke worker.
2. Performa mode `with_worker` bergantung kualitas jaringan VM dan throughput bridge HDFS.
3. Transformer training masih sensitif ke resource CPU/RAM/GPU environment.
4. Script network fix bersifat sistem-level, wajib dijalankan hati-hati (root).

## 10. Arah Pengembangan

Prioritas jika ingin scale-up:
1. Migrasi training model ke framework distributed (Spark MLlib atau strategi distributed lain).
2. Menyatukan preprocessing text lanjutan ke jalur distributed agar beban master berkurang.
3. Menambahkan benchmark otomatis per skenario cluster/network.
4. Menambahkan test automation untuk validasi artifact dan schema output antar step.
