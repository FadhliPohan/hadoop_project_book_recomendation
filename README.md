# Hadoop Amazon Books Reviews

Project ini berisi pipeline machine learning untuk:
- Sentiment analysis review buku Amazon
- Recommender system berbasis rating, sentiment, dan konten review
- Orkestrasi pipeline yang aman dari training otomatis tidak sengaja
- Eksperimen komparasi training `without_worker` vs `with_worker` (worker hanya untuk preprocessing)

## Struktur Project

```text
Hadoop_Amazon_Books_Reviews/
├── machine_learning/
│   ├── config.yaml
│   ├── requirements.txt
│   ├── main.py
│   ├── dataset/
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── src/
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   ├── eda.py
│   │   ├── train_sentiment.py
│   │   ├── train_sentiment_transformer.py
│   │   ├── train_recommender.py
│   │   ├── inference.py
│   │   ├── evaluate.py
│   │   ├── mlflow_tracker.py
│   │   └── utils.py
│   ├── notebooks/
│   ├── models/
│   ├── reports/
│   ├── logs/
│   └── mlruns/
├── streamlit/
│   └── app.py
└── scripts/
    ├── start_cluster.sh
    ├── stop_cluster.sh
    ├── submit_training.sh
    └── upload_to_hdfs.sh
```

## Setup

Gunakan virtual environment lokal:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r machine_learning/requirements.txt
pip install streamlit
```

## Menjalankan Pipeline

Masuk ke root project, lalu jalankan:

```bash
python3 machine_learning/main.py --step eda
python3 machine_learning/main.py --step preprocess
python3 machine_learning/main.py --step evaluate
```

Training tidak akan berjalan kecuali Anda menambahkan `--allow-training`.

## Mode Training Baru

Pipeline sekarang mendukung dua mode:

- `without_worker`: preprocess + training dilakukan di master.
- `with_worker`: preprocessing awal dilakukan distributed via Spark/Hadoop worker, lalu training tetap di master.

Untuk menjaga fairness eksperimen, batas RAM proses training master default adalah **3GB** (bisa override dengan `--ram-limit-gb`).
Jika limit terlalu ketat terhadap footprint proses saat ini, pipeline akan otomatis melewati limit dan memberi warning agar proses tidak crash (dikontrol oleh `training.ram_limit_safety_margin_mb`).

Contoh training eksplisit:

```bash
python3 machine_learning/main.py --step train_sentiment_baseline --allow-training
python3 machine_learning/main.py --step train_sentiment_transformer --allow-training
python3 machine_learning/main.py --step train_recommender --allow-training
```

Training pipeline single-mode:

```bash
# Full train tanpa worker
python3 machine_learning/main.py --step train_pipeline --allow-training \
  --training-mode without_worker --ram-limit-gb 3

# Full train via worker (Spark preprocess sudah tersedia di HDFS)
python3 machine_learning/main.py --step train_pipeline --allow-training \
  --training-mode with_worker --ram-limit-gb 3

# Jika ingin pipeline otomatis submit Spark preprocess dulu
python3 machine_learning/main.py --step train_pipeline --allow-training \
  --training-mode with_worker --run-worker-preprocess --ram-limit-gb 3
```

## Optimasi Kecepatan `with_worker` (VirtualBox)

Untuk koneksi antar VM yang terbatas, aktifkan output Spark yang lebih kecil agar bridge HDFS ke master jauh lebih cepat:

- `machine_learning/config.yaml`:
  - `spark_preprocess.max_rows: 120000`
  - `spark_preprocess.sample_fraction: 1.0` (ubah < 1.0 jika ingin lebih kecil lagi)

Catatan:
- `max_rows=0` berarti simpan full output Spark ke HDFS.
- Bridge HDFS sekarang memakai cache persisten di `/tmp/spark_hdfs_bridge_cache`, jadi run berikutnya bisa reuse tanpa download ulang file besar jika output HDFS belum berubah.

Jika ingin reset cache bridge:

```bash
rm -rf /tmp/spark_hdfs_bridge_cache
```

Menjalankan eksperimen komparasi dua mode sekaligus:

```bash
python3 machine_learning/main.py --step compare_training_modes --allow-training \
  --ram-limit-gb 3
```

Output perbandingan:
- `machine_learning/reports/training_mode_comparison.json`
- `machine_learning/reports/experiments/without_worker_latest_run.json`
- `machine_learning/reports/experiments/with_worker_latest_run.json`

Jalankan semua step:

```bash
# Aman: hanya non-training
python3 machine_learning/main.py --step all

# Full termasuk training
python3 machine_learning/main.py --step all --allow-training \
  --training-mode without_worker --ram-limit-gb 3
```

## Menjalankan Dashboard Streamlit

```bash
streamlit run streamlit/app.py
```

Jika command `streamlit` belum ada di PATH:

```bash
python3 -m streamlit run streamlit/app.py
```

Fitur dashboard:
- Trigger step pipeline
- Pilihan mode training (`without_worker` / `with_worker`)
- Trigger komparasi dua mode training dan ringkasan delta waktu/metric
- Trigger start/stop cluster script
- Inference sentiment (jika model tersedia)
- Inference rekomendasi user (jika artifact rekomendasi tersedia)

## Dataset

Dataset acuan:
- Amazon Books Reviews (Kaggle)
- https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data

Catatan:
- File dataset CSV sengaja di-ignore dari git.

## Kebijakan File Besar

Agar repository tetap ringan, file artifact training tidak dipush:
- `machine_learning/data/processed/*`
- `machine_learning/models/*`
- `machine_learning/reports/*`
- `machine_learning/logs/*`
- `machine_learning/mlruns/*`

Folder tetap dipertahankan via `.gitkeep`.

## Status Pengembangan

Progress detail ada di:
- `machine_learning/documentation/report_progres.md`
