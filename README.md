# Hadoop Amazon Books Reviews

Project ini adalah aplikasi end-to-end untuk:
- analisis sentimen review buku Amazon,
- sistem rekomendasi buku hybrid,
- eksperimen komparasi pipeline `without_worker` vs `with_worker`,
- monitoring pipeline melalui dashboard Streamlit.

Implementasi saat ini memakai arsitektur **hybrid local + cluster**:
- `without_worker`: preprocessing + training model berjalan di master (lokal Python).
- `with_worker`: preprocessing awal berjalan distributed di Spark/YARN (worker), lalu hasilnya dijembatani ke master untuk training model lokal.

## Ringkasan Cepat

1. Aktifkan environment dan install dependency.
2. Siapkan dataset pada `machine_learning/dataset/`.
3. Jalankan pipeline dari CLI atau dari Streamlit.
4. Untuk komparasi mode training, jalankan `compare_training_modes`.

## Struktur Inti Repository

```text
Hadoop_Amazon_Books_Reviews/
├── machine_learning/
│   ├── main.py
│   ├── config.yaml
│   ├── src/
│   ├── dataset/
│   ├── data/
│   ├── models/
│   ├── reports/
│   ├── logs/
│   ├── mlruns/
│   └── documentation/
├── streamlit/
│   └── app.py
├── scripts/
├── config/
└── documentation/
```

Detail fungsi setiap file ada di [`documentation/structure.md`](documentation/structure.md).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r machine_learning/requirements.txt
```

## Dataset

Dataset utama:
- Amazon Books Reviews (Kaggle)
- https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data

Letakkan file CSV ke:
- `machine_learning/dataset/Books_rating.csv`
- `machine_learning/dataset/books_data.csv` (opsional metadata)

## Menjalankan Pipeline (CLI)

Semua perintah dijalankan dari root project.

### Non-training steps

```bash
python3 machine_learning/main.py --step eda
python3 machine_learning/main.py --step preprocess
python3 machine_learning/main.py --step evaluate
```

### Training steps (wajib `--allow-training`)

```bash
python3 machine_learning/main.py --step train_sentiment_baseline --allow-training
python3 machine_learning/main.py --step train_sentiment_transformer --allow-training
python3 machine_learning/main.py --step train_recommender --allow-training
```

### Training pipeline per mode

```bash
python3 machine_learning/main.py --step train_pipeline --allow-training \
  --training-mode without_worker --ram-limit-gb 3

python3 machine_learning/main.py --step train_pipeline --allow-training \
  --training-mode with_worker --ram-limit-gb 3
```

Jika ingin auto submit Spark preprocess saat mode `with_worker`:

```bash
python3 machine_learning/main.py --step train_pipeline --allow-training \
  --training-mode with_worker --run-worker-preprocess --ram-limit-gb 3
```

### Komparasi dua mode training

```bash
python3 machine_learning/main.py --step compare_training_modes --allow-training \
  --ram-limit-gb 3
```

Artifact komparasi:
- `machine_learning/reports/training_mode_comparison.json`
- `machine_learning/reports/experiments/without_worker_latest_run.json`
- `machine_learning/reports/experiments/with_worker_latest_run.json`

### Step `all`

```bash
# Aman: tanpa training
python3 machine_learning/main.py --step all

# Full: dengan training
python3 machine_learning/main.py --step all --allow-training \
  --training-mode without_worker --ram-limit-gb 3
```

## Menjalankan Spark Preprocess di Cluster

```bash
bash scripts/start_cluster.sh
bash scripts/upload_to_hdfs.sh
bash scripts/spark_submit_training.sh preprocess_spark 2 2 2G 2G
```

Catatan:
- `preprocess_spark` menulis output Parquet ke HDFS.
- Training model tetap berjalan lokal di master.

## Menjalankan Dashboard Streamlit

```bash
streamlit run streamlit/app.py
```

Tab utama dashboard:
- Overview
- EDA
- Pipeline
- Reports
- Inference
- Cluster

## File Output Utama

- Preprocess:
  - `machine_learning/data/processed/processed_reviews.csv`
  - `machine_learning/data/processed/train.csv`
  - `machine_learning/data/processed/validation.csv`
  - `machine_learning/data/processed/test.csv`
- Sentiment baseline:
  - `machine_learning/models/sentiment/baseline/`
- Sentiment transformer:
  - `machine_learning/models/sentiment/transformer/distilbert_v1/`
- Recommender:
  - `machine_learning/models/recommender/`
  - `machine_learning/reports/recommender_metrics.json`
- Final report:
  - `machine_learning/reports/final_report.json`

## Dokumentasi

- Arsitektur: [`documentation/arsitecture_plant.md`](documentation/arsitecture_plant.md)
- Tutorial operasional: [`documentation/tutorial.md`](documentation/tutorial.md)
- Struktur file dan fungsi per file: [`documentation/structure.md`](documentation/structure.md)
- Q&A teknis: [`documentation/QNA.md`](documentation/QNA.md)
- Progress ML: [`machine_learning/documentation/report_progres.md`](machine_learning/documentation/report_progres.md)

## Catatan Batasan Saat Ini

- Worker belum melatih model sentiment/recommender secara distributed.
- Worker dipakai untuk preprocessing Spark, bukan training model final.
- Jika cluster lambat, `spark_hdfs` bridge ke local bisa lambat; gunakan `sample_fraction`/`max_rows` untuk mengurangi ukuran output Spark.
