# Report Progres Pengembangan Machine Learning

Tanggal update: 30 April 2026

## Mode Kerja Saat Ini
- Fokus: **pengkodingan**
- Training otomatis: **dinonaktifkan**
- Artifact besar hasil training: **tidak dipush** (sudah di-ignore)

## Yang Sudah Selesai
- Struktur modular pipeline di `machine_learning/src`:
  - `data_loader.py`
  - `eda.py`
  - `preprocessing.py`
  - `train_sentiment.py` (baseline)
  - `train_recommender.py`
  - `train_sentiment_transformer.py` (implementasi ditambahkan)
  - `inference.py`
  - `evaluate.py`
  - `mlflow_tracker.py` (opsional, graceful jika mlflow belum terpasang)
- `main.py` mendukung langkah:
  - `eda`
  - `preprocess`
  - `train_sentiment_baseline`
  - `train_sentiment_transformer`
  - `train_recommender`
  - `evaluate`
  - `all`
- Proteksi anti training otomatis:
  - Semua step training butuh flag `--allow-training`.
  - `--step all` default menjalankan non-training.
- Script submit training diperbarui:
  - `scripts/submit_training.sh` wajib argumen step.
  - Otomatis memakai `.venv/bin/python` jika tersedia.
- Hygiene repository untuk artifact besar:
  - `.gitignore` diperluas untuk `data/processed`, `models`, `reports`, `logs`, `mlruns`.
  - File hasil training yang terlanjur ter-track sudah dihapus dari index git.
- Integrasi dashboard Streamlit sudah dibuat di `streamlit/app.py`:
  - Tab `Pipeline` untuk menjalankan step pipeline.
  - Tab `Cluster` untuk start/stop cluster dan upload ke HDFS.
  - Tab `Inference` untuk sentiment dan rekomendasi.
  - Tab `Status` untuk cek ketersediaan artifact.
- README utama sudah dilengkapi dengan setup, struktur, perintah pipeline, dan kebijakan file besar.

## Yang Belum Selesai
- Konten notebook EDA (`machine_learning/notebooks/`) belum dibuat detail.
- Integrasi visualisasi metrik lanjutan langsung di dashboard Streamlit.
- Validasi end-to-end step transformer pada environment yang sudah terpasang lengkap (`transformers` + `torch`).
- Evaluasi lanjutan recommender (tuning hyperparameter + quality improvement).

## Catatan Operasional
- Contoh menjalankan non-training:
  - `python3 machine_learning/main.py --step eda`
  - `python3 machine_learning/main.py --step preprocess`
  - `python3 machine_learning/main.py --step evaluate`
- Contoh menjalankan training (eksplisit):
  - `python3 machine_learning/main.py --step train_sentiment_baseline --allow-training`
  - `python3 machine_learning/main.py --step train_sentiment_transformer --allow-training`
  - `python3 machine_learning/main.py --step train_recommender --allow-training`
- Verifikasi script cluster:
  - `start_cluster.sh` dan `stop_cluster.sh` sudah diuji jalan.
  - Pada environment saat ini muncul timeout SSH ke `worker1` dan `worker2`, jadi perlu memastikan koneksi SSH antar node stabil sebelum operasi cluster penuh.

## Status
- Siap lanjut ke tahap notebook EDA dan pengayaan dashboard tanpa risiko training tidak sengaja.
