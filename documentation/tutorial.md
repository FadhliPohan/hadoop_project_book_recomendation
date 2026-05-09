# Tutorial Operasional Project

Panduan ini menjelaskan cara menjalankan project sesuai implementasi kode saat ini.

Tanggal pembaruan: 2026-05-09.

## 1. Prasyarat

- Python 3.10+ disarankan.
- Virtual environment aktif.
- Dependency terpasang dari `machine_learning/requirements.txt`.
- Dataset berada di `machine_learning/dataset/`.

Jika memakai mode cluster:
- Hadoop + YARN aktif.
- Akses SSH ke worker tersedia.
- Path HDFS sesuai `machine_learning/config.yaml` dan `config/cluster.yaml`.

## 2. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r machine_learning/requirements.txt
```

## 3. Menjalankan dari CLI

### 3.1 Step non-training

```bash
python3 machine_learning/main.py --step eda
python3 machine_learning/main.py --step preprocess
python3 machine_learning/main.py --step evaluate
```

### 3.2 Step training (harus pakai `--allow-training`)

```bash
python3 machine_learning/main.py --step train_sentiment_baseline --allow-training
python3 machine_learning/main.py --step train_sentiment_transformer --allow-training
python3 machine_learning/main.py --step train_recommender --allow-training
```

### 3.3 Training pipeline mode tunggal

Tanpa worker:

```bash
python3 machine_learning/main.py --step train_pipeline --allow-training \
  --training-mode without_worker --ram-limit-gb 3
```

Dengan worker (preprocess Spark sudah tersedia di HDFS):

```bash
python3 machine_learning/main.py --step train_pipeline --allow-training \
  --training-mode with_worker --ram-limit-gb 3
```

Dengan worker + auto submit Spark preprocess:

```bash
python3 machine_learning/main.py --step train_pipeline --allow-training \
  --training-mode with_worker --run-worker-preprocess --ram-limit-gb 3
```

### 3.4 Compare dua mode training

```bash
python3 machine_learning/main.py --step compare_training_modes --allow-training \
  --ram-limit-gb 3
```

Output compare:
- `machine_learning/reports/training_mode_comparison.json`
- `machine_learning/reports/experiments/without_worker_latest_run.json`
- `machine_learning/reports/experiments/with_worker_latest_run.json`

Status report compare:
- `success`: dua mode sukses.
- `partial_success`: salah satu mode gagal.
- `failed`: keduanya gagal.

## 4. Menjalankan Semua Step Sekaligus

Mode aman (tanpa training):

```bash
python3 machine_learning/main.py --step all
```

Mode penuh:

```bash
python3 machine_learning/main.py --step all --allow-training \
  --training-mode without_worker --ram-limit-gb 3
```

## 5. Menjalankan Streamlit Dashboard

```bash
streamlit run streamlit/app.py
```

Tab penting:
- `Overview`: status artifact + ringkasan metrik.
- `Pipeline`: eksekusi step pipeline dengan output live.
- `Reports`: detail metrik baseline/transformer/recommender + komparasi mode.
- `Cluster`: operasi cluster, upload/download HDFS, spark submit.

## 6. Workflow Cluster (Opsional)

### 6.1 Start cluster dan upload dataset

```bash
bash scripts/start_cluster.sh
bash scripts/upload_to_hdfs.sh
```

### 6.2 Submit distributed preprocessing

```bash
bash scripts/spark_submit_training.sh preprocess_spark 2 2 2G 2G
```

Environment yang bisa di-set:
- `SPARK_SAMPLE_FRACTION`
- `SPARK_MAX_ROWS`
- `SPARK_OUTPUT_PARTITIONS`
- `SPARK_LOG_ROW_COUNTS`
- `SPARK_SHOW_LABEL_DISTRIBUTION`

### 6.3 Jalankan training pipeline `with_worker`

```bash
python3 machine_learning/main.py --step train_pipeline --allow-training \
  --training-mode with_worker --ram-limit-gb 3
```

## 7. Memahami Artifact Output

### 7.1 Data

- `machine_learning/data/processed/processed_reviews.csv`
- `machine_learning/data/processed/train.csv`
- `machine_learning/data/processed/validation.csv`
- `machine_learning/data/processed/test.csv`

### 7.2 Model

- Sentiment baseline: `machine_learning/models/sentiment/baseline/`
- Sentiment transformer: `machine_learning/models/sentiment/transformer/distilbert_v1/`
- Recommender: `machine_learning/models/recommender/`

### 7.3 Report

- Recommender metrics: `machine_learning/reports/recommender_metrics.json`
- Final report: `machine_learning/reports/final_report.json`
- Compare mode: `machine_learning/reports/training_mode_comparison.json`

## 8. Troubleshooting Umum

### 8.1 `Step training diblokir`

Penyebab:
- Anda menjalankan step training tanpa `--allow-training`.

Solusi:
- Tambahkan flag `--allow-training`.

### 8.2 Warning `Parser C kehabisan memori`

Penyebab:
- CSV besar, parser C pandas gagal alokasi.

Status implementasi:
- Kode sudah fallback ke `engine='python'` di pembacaan split sentimen.
- Loader recommender sudah membaca chunked + fallback engine.

### 8.3 Mode `with_worker` gagal karena output HDFS tidak ada

Penyebab:
- Spark preprocess belum dijalankan atau path output tidak sesuai.

Solusi:
1. Jalankan `spark_submit_training.sh preprocess_spark`.
2. Verifikasi path output via `hdfs dfs -ls -h`.
3. Cek kembali `hadoop.output_hdfs_path` di `machine_learning/config.yaml`.

### 8.4 Spark job lama di status ACCEPTED

Penyebab umum:
- NodeManager belum aktif.

Solusi:
1. Jalankan `bash scripts/start_cluster.sh`.
2. Verifikasi `yarn node -list`.
3. Perbaiki network mapping bila perlu dengan `fix_yarn_*` scripts.

### 8.5 Timeout dashboard saat pipeline panjang

Solusi:
- Di tab `Pipeline`, aktifkan opsi tanpa timeout.
- Atau naikkan nilai timeout sesuai kebutuhan.

## 9. Catatan Praktik Baik

1. Gunakan `without_worker` saat debug code cepat.
2. Gunakan `with_worker` saat ingin simulasi preprocessing distributed.
3. Jalankan `compare_training_modes` setelah artifact lama dibersihkan untuk hasil eksperimen yang bersih.
4. Simpan hasil compare JSON sebagai lampiran eksperimen akademik.
