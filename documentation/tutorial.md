# Tutorial: Amazon Books ML Dashboard

Panduan lengkap penggunaan project **Review Sentiment Analysis & Recommender System** berbasis dataset Amazon Books Reviews.

---

## Daftar Isi

1. [Gambaran Umum Sistem](#1-gambaran-umum-sistem)
2. [Prasyarat & Instalasi](#2-prasyarat--instalasi)
3. [Persiapan Dataset](#3-persiapan-dataset)
4. [Menjalankan Pipeline via CLI](#4-menjalankan-pipeline-via-cli)
5. [Menjalankan Dashboard Streamlit](#5-menjalankan-dashboard-streamlit)
6. [Panduan Tiap Tab Dashboard](#6-panduan-tiap-tab-dashboard)
   - [Tab Pipeline](#tab-pipeline)
   - [Tab Cluster](#tab-cluster)
   - [Tab Inference](#tab-inference)
   - [Tab Overview](#tab-overview)
7. [Mengelola Cluster Hadoop](#7-mengelola-cluster-hadoop)
8. [Tracking Eksperimen dengan MLflow](#8-tracking-eksperimen-dengan-mlflow)
9. [Konfigurasi Proyek](#9-konfigurasi-proyek)
10. [Struktur Folder](#10-struktur-folder)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Gambaran Umum Sistem

Project ini adalah pipeline machine learning end-to-end yang terdiri dari dua komponen utama:

| Komponen | Deskripsi |
|---|---|
| **Sentiment Analysis** | Mengklasifikasi review buku menjadi Positive, Neutral, atau Negative |
| **Recommender System** | Merekomendasikan buku kepada user berdasarkan rating, sentimen, dan behavior |

### Arsitektur Pipeline

```
Dataset CSV
    │
    ▼
[EDA] ──────────── reports/eda/
    │
    ▼
[Preprocess] ────── data/processed/
    │
    ▼
[Train Sentiment] ─ models/sentiment/
    │
    ▼
[Train Recommender] models/recommender/
    │
    ▼
[Evaluate] ──────── reports/
    │
    ▼
[Inference / Dashboard Streamlit]
```

### Label Sentimen

| Rating | Label | Kode |
|---|---|---|
| 1 – 2 bintang | Negative | 0 |
| 3 bintang | Neutral | 1 |
| 4 – 5 bintang | Positive | 2 |

---

## 2. Prasyarat & Instalasi

### Kebutuhan Sistem

- Python 3.9 atau lebih baru
- pip
- (Opsional) Hadoop terinstal jika ingin menggunakan fitur Cluster

### Langkah Instalasi

```bash
# 1. Clone / masuk ke direktori proyek
cd "Hadoop_Amazon_Books_Reviews"

# 2. Buat virtual environment
python3 -m venv .venv

# 3. Aktifkan virtual environment
source .venv/bin/activate

# 4. Install semua dependensi
pip install -r machine_learning/requirements.txt
```

### Verifikasi Instalasi

```bash
python3 -c "import pandas, sklearn, torch, transformers, streamlit; print('OK')"
```

---

## 3. Persiapan Dataset

Dataset yang digunakan adalah **Amazon Books Reviews** dari Kaggle.

**Link download:** https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data

Setelah download, letakkan file CSV di dalam folder berikut:

```
machine_learning/
└── dataset/
    ├── Books_rating.csv   ← file utama review
    └── books_data.csv     ← metadata buku (opsional)
```

> **Catatan:** File dataset di-ignore oleh git (`.gitignore`) sehingga tidak ikut ter-push ke repository. Pastikan Anda menyalin file secara manual ke folder `dataset/`.

---

## 4. Menjalankan Pipeline via CLI

Semua perintah dijalankan dari **root directory proyek** (bukan dari dalam `machine_learning/`).

Pastikan virtual environment sudah aktif:

```bash
source .venv/bin/activate
```

### Step-by-Step Pipeline

#### Step 1 — Exploratory Data Analysis (EDA)

```bash
python3 machine_learning/main.py --step eda
```

Output disimpan di: `machine_learning/reports/eda/`
- Grafik distribusi rating (PNG)
- Statistik dasar dataset (CSV)
- Ringkasan teks insight

#### Step 2 — Preprocessing

```bash
python3 machine_learning/main.py --step preprocess
```

Output disimpan di: `machine_learning/data/processed/`
- `processed_reviews.csv` — data setelah cleaning
- `train.csv` — 70% data untuk training
- `validation.csv` — 15% data untuk validasi
- `test.csv` — 15% data untuk pengujian akhir

#### Step 3 — Training Baseline Sentiment Model

> ⚠️ Memerlukan flag `--allow-training`

```bash
python3 machine_learning/main.py --step train_sentiment_baseline --allow-training
```

Model yang dilatih: TF-IDF + Logistic Regression, Naive Bayes, Linear SVM.

Output disimpan di: `machine_learning/models/sentiment/baseline/`

#### Step 4 — Training Transformer Sentiment Model

> ⚠️ Memerlukan flag `--allow-training` dan GPU/RAM yang cukup

```bash
python3 machine_learning/main.py --step train_sentiment_transformer --allow-training
```

Model yang digunakan: `distilbert-base-uncased`

Output disimpan di: `machine_learning/models/sentiment/transformer/distilbert_v1/`

#### Step 5 — Training Recommender System

> ⚠️ Memerlukan flag `--allow-training`

```bash
python3 machine_learning/main.py --step train_recommender --allow-training
```

Output disimpan di: `machine_learning/models/recommender/`

#### Step 6 — Training Pipeline per Mode (Rekomendasi untuk eksperimen)

Mode `without_worker` (tanpa worker):

```bash
python3 machine_learning/main.py --step train_pipeline --allow-training \
  --training-mode without_worker --ram-limit-gb 3
```

Mode `with_worker` (worker hanya preprocessing):

```bash
python3 machine_learning/main.py --step train_pipeline --allow-training \
  --training-mode with_worker --ram-limit-gb 3
```

Jika ingin otomatis submit Spark preprocessing saat mode `with_worker`:

```bash
python3 machine_learning/main.py --step train_pipeline --allow-training \
  --training-mode with_worker --run-worker-preprocess --ram-limit-gb 3
```

#### Step 7 — Komparasi Otomatis Dua Mode

```bash
python3 machine_learning/main.py --step compare_training_modes --allow-training \
  --ram-limit-gb 3
```

Output:
- `machine_learning/reports/training_mode_comparison.json`
- `machine_learning/reports/experiments/without_worker_latest_run.json`
- `machine_learning/reports/experiments/with_worker_latest_run.json`
- `machine_learning/reports/final_report.json`

#### Step 8 — Evaluasi (Kompilasi Report Final)

```bash
python3 machine_learning/main.py --step evaluate
```

---

### Menjalankan Semua Step Sekaligus

**Mode aman** (tanpa training):
```bash
python3 machine_learning/main.py --step all
```

**Mode penuh** (termasuk training):
```bash
python3 machine_learning/main.py --step all --allow-training \
  --training-mode without_worker --ram-limit-gb 3
```

---

### Menggunakan Script `submit_training.sh`

Alternatif cepat untuk menjalankan training:

```bash
# Format: ./scripts/submit_training.sh <step>
./scripts/submit_training.sh train_sentiment_baseline
./scripts/submit_training.sh train_recommender
```

Script ini secara otomatis menambahkan flag `--allow-training`.

---

## 5. Menjalankan Dashboard Streamlit

Dashboard adalah antarmuka grafis untuk mengontrol seluruh pipeline tanpa perlu mengetik perintah.

```bash
streamlit run streamlit/app.py
```

Jika perintah `streamlit` belum ada di PATH:

```bash
python3 -m streamlit run streamlit/app.py
```

Setelah berhasil, buka browser dan akses:

```
http://localhost:8501
```

---

## 6. Panduan Tiap Tab Dashboard

### Tab Pipeline

Gunakan tab ini untuk menjalankan setiap step pipeline secara grafis.

| Elemen | Fungsi |
|---|---|
| **Dropdown "Pilih step"** | Memilih step pipeline yang ingin dijalankan |
| **Checkbox "Izinkan training"** | Mengaktifkan `--allow-training` untuk step training |
| **Training Mode** | Memilih `without_worker` atau `with_worker` untuk step `train_pipeline` / `compare_training_modes` |
| **Batas RAM Master (GB)** | Override batas RAM proses training di master (default 3GB) |
| **Tombol "Jalankan Step"** | Mengeksekusi step yang dipilih |

**Cara pakai:**
1. Pilih step dari dropdown (contoh: `eda`)
2. Jika ingin menjalankan training, centang checkbox "Izinkan training"
3. Untuk eksperimen perbandingan, pilih:
   - `train_pipeline` untuk satu mode
   - `compare_training_modes` untuk dua mode sekaligus
4. Atur `Training Mode` dan pastikan `Batas RAM Master` = `3`
5. Klik **Jalankan Step**
6. Hasil stdout dan stderr akan muncul di bawah tombol

> ⚠️ **Peringatan:** Training diblokir secara default. Aktifkan checkbox hanya jika Anda memang bermaksud melatih model.

---

### Tab Cluster

Gunakan tab ini untuk mengelola Hadoop cluster dan upload dataset ke HDFS.

| Tombol | Fungsi |
|---|---|
| **Start Cluster** | Menjalankan `start-dfs.sh` dan `start-yarn.sh` |
| **Stop Cluster** | Menghentikan `stop-yarn.sh` dan `stop-dfs.sh` |
| **Fix Permission Worker1 & Worker2** | Membuat path HDFS user dan membuka permission agar upload tidak gagal |
| **Upload ke HDFS** | Mengupload dataset ke path HDFS yang ditentukan |

**Cara upload dataset ke HDFS:**
1. Isi field **HDFS target** dengan path tujuan (default: `/user/<username>/amazon_books`)
2. Klik tombol **Upload ke HDFS**

Jika muncul error `Permission denied ... inode="/"`, klik tombol **Fix Permission Worker1 & Worker2** terlebih dahulu.

> **Prasyarat:** Hadoop harus sudah terinstal dan cluster harus dalam keadaan aktif sebelum upload.

---

### Tab Inference

Gunakan tab ini untuk mencoba model yang sudah dilatih secara interaktif.

#### Sentiment Inference

1. Masukkan teks review pada kolom **Review text**
2. Klik **Prediksi Sentiment**
3. Hasil prediksi ditampilkan dalam format JSON:

```json
{
  "sentiment": "Positive",
  "confidence": 0.94,
  "probabilities": {
    "Negative": 0.02,
    "Neutral": 0.04,
    "Positive": 0.94
  }
}
```

#### Recommender Inference

1. Masukkan **User ID** dari dataset
2. Atur jumlah rekomendasi dengan slider **Top-N** (1–20)
3. Klik **Ambil Rekomendasi**
4. Hasil rekomendasi ditampilkan dalam bentuk tabel

> **Prasyarat:** Model sentiment dan artifact recommender harus sudah tersedia (jalankan step training terlebih dahulu).

---

### Tab Overview

Tab ini menampilkan status ketersediaan artifact penting proyek + ringkasan komparasi mode jika tersedia.

| Artifact | Path |
|---|---|
| Dataset reviews | `machine_learning/dataset/Books_rating.csv` |
| Processed train.csv | `machine_learning/data/processed/train.csv` |
| Sentiment metrics | `machine_learning/models/sentiment/baseline/metrics.json` |
| Recommender metrics | `machine_learning/reports/recommender_metrics.json` |
| Comparison report | `machine_learning/reports/training_mode_comparison.json` |
| Model registry | `machine_learning/models/model_registry.json` |

Jika `final_report.json` tersedia, isinya akan ditampilkan langsung di halaman ini.

---

## 7. Mengelola Cluster Hadoop

Selain melalui dashboard, cluster juga bisa dikelola langsung via terminal:

```bash
# Start cluster
bash scripts/start_cluster.sh

# Stop cluster
bash scripts/stop_cluster.sh
```

**Melihat status service Hadoop:**
```bash
jps
```

Output yang diharapkan saat cluster aktif:
```
NameNode
DataNode
ResourceManager
NodeManager
SecondaryNameNode
```

> **Catatan:** Jika muncul error timeout SSH ke `worker1` / `worker2`, pastikan koneksi SSH antar node sudah dikonfigurasi dengan benar (SSH passwordless).

---

## 8. Tracking Eksperimen dengan MLflow

Project ini menggunakan **MLflow** untuk mencatat setiap eksperimen training.

### Melihat UI MLflow

```bash
mlflow ui --backend-store-uri machine_learning/mlruns
```

Buka browser: `http://localhost:5000`

### Informasi yang Dicatat

Setiap training run mencatat:
- Nama model dan versi
- Hyperparameter yang digunakan
- Metrik: Accuracy, Precision, Recall, F1-score
- Confusion matrix (PNG)
- Artifact model dan tokenizer
- Waktu training

### Nama Eksperimen

Semua run dikelompokkan dalam satu eksperimen:

```
amazon_books_sentiment_recommender
```

---

## 9. Konfigurasi Proyek

File konfigurasi utama ada di `machine_learning/config.yaml`.

### Parameter Penting

```yaml
data:
  sample_rows: 120000        # Jumlah baris yang diambil dari dataset
  min_review_text_length: 5  # Panjang minimum teks review

preprocessing:
  test_size: 0.15            # 15% untuk test set
  validation_size: 0.15      # 15% untuk validation set

transformer:
  model_name: distilbert-base-uncased
  max_length: 128
  batch_size: 16
  learning_rate: 2.0e-5
  epochs: 3

recommender:
  min_user_interactions: 3   # Minimum interaksi user agar masuk model
  min_item_interactions: 3   # Minimum interaksi item
  latent_factors: 20
  top_k_values: [5, 10]      # K yang digunakan saat evaluasi

training:
  master_ram_limit_gb: 3     # Batas RAM master saat training
  auto_run_worker_preprocess: false

paths:
  training_comparison_json: reports/training_mode_comparison.json
  training_experiments_dir: reports/experiments
```

Ubah nilai-nilai di atas sesuai kebutuhan sebelum menjalankan pipeline.

---

## 10. Struktur Folder

```
Hadoop_Amazon_Books_Reviews/
├── .venv/                          ← Virtual environment (tidak di-push)
├── config/
│   └── paths.yaml
├── documentation/
│   └── tutorial.md                 ← File ini
├── machine_learning/
│   ├── config.yaml                 ← Konfigurasi utama pipeline
│   ├── requirements.txt            ← Daftar dependensi Python
│   ├── main.py                     ← Entry point CLI pipeline
│   ├── dataset/                    ← Letakkan file CSV di sini
│   ├── data/
│   │   └── processed/              ← Output preprocessing
│   ├── src/
│   │   ├── data_loader.py          ← Loading dataset
│   │   ├── eda.py                  ← Exploratory Data Analysis
│   │   ├── preprocessing.py        ← Cleaning & splitting data
│   │   ├── train_sentiment.py      ← Baseline sentiment model
│   │   ├── train_sentiment_transformer.py  ← DistilBERT model
│   │   ├── train_recommender.py    ← Recommender system
│   │   ├── training_runtime.py      ← Runner training mode + komparasi
│   │   ├── inference.py            ← Inference sentiment & rekomendasi
│   │   ├── evaluate.py             ← Kompilasi laporan akhir
│   │   ├── mlflow_tracker.py       ← Integrasi MLflow
│   │   └── utils.py                ← Fungsi utilitas
│   ├── models/                     ← Model hasil training (tidak di-push)
│   ├── reports/                    ← Laporan & metrik evaluasi
│   ├── logs/                       ← Log training
│   └── mlruns/                     ← Data MLflow tracking
├── scripts/
│   ├── start_cluster.sh            ← Menjalankan Hadoop cluster
│   ├── stop_cluster.sh             ← Menghentikan Hadoop cluster
│   ├── submit_training.sh          ← Shortcut menjalankan training
│   ├── spark_submit_training.sh    ← Submit distributed preprocess ke YARN
│   └── upload_to_hdfs.sh           ← Upload dataset ke HDFS
└── streamlit/
    └── app.py                      ← Dashboard web Streamlit
```

---

## 11. Troubleshooting

### ❌ `streamlit: command not found`

```bash
pip install streamlit
# atau jalankan dengan:
python3 -m streamlit run streamlit/app.py
```

---

### ❌ `Step training diblokir`

Anda lupa menambahkan flag `--allow-training`.

```bash
# Benar:
python3 machine_learning/main.py --step train_sentiment_baseline --allow-training

# Salah (akan error):
python3 machine_learning/main.py --step train_sentiment_baseline
```

---

### ❌ `FileNotFoundError: Books_rating.csv`

Dataset belum diletakkan di folder yang benar. Pastikan file ada di:

```
machine_learning/dataset/Books_rating.csv
```

---

### ❌ `Gagal menjalankan sentiment inference`

Model belum dilatih. Jalankan terlebih dahulu:

```bash
python3 machine_learning/main.py --step train_sentiment_baseline --allow-training
```

---

### ❌ `Tidak ada rekomendasi untuk user_id tersebut`

Kemungkinan penyebab:
1. Recommender belum dilatih → jalankan `--step train_recommender --allow-training`
2. User ID tidak ada dalam dataset training
3. User memiliki interaksi di bawah `min_user_interactions` (default: 3)

---

### ❌ SSH timeout ke worker node

Saat menjalankan `start_cluster.sh`, muncul error koneksi ke `worker1`/`worker2`:

1. Pastikan hostname `worker1` dan `worker2` terdaftar di `/etc/hosts`
2. Pastikan SSH passwordless sudah dikonfigurasi:
   ```bash
   ssh-copy-id fadhli@worker1
   ssh-copy-id fadhli@worker2
   ```
3. Pastikan mapping IP sesuai cluster aktif:
   - `worker1` -> `192.168.0.103`
   - `worker2` -> `192.168.0.105`

---

### ❌ Mode `with_worker` gagal karena output HDFS belum ada

Pastikan output Spark preprocessing sudah tersedia:

```bash
hdfs dfs -ls -h /user/fadhli/output/amazon_books_ml/processed
```

Jika belum ada, jalankan:

```bash
bash scripts/spark_submit_training.sh preprocess_spark
```

Lalu ulangi `train_pipeline --training-mode with_worker`.

---

### ❌ Training berhenti karena batas RAM 3GB

Ini normal jika dataset/konfigurasi terlalu berat. Opsi:
1. Turunkan `data.sample_rows` di `machine_learning/config.yaml`
2. Jalankan tanpa transformer (`--include-transformer` jangan diaktifkan)
3. Ulangi dengan RAM limit berbeda:
   ```bash
   python3 machine_learning/main.py --step train_pipeline --allow-training \
     --training-mode without_worker --ram-limit-gb 4
   ```

---

*Dokumentasi ini dibuat untuk proyek: **Amazon Books Sentiment Analysis & Recommender System***
