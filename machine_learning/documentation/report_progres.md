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
- `main.py` sudah mendukung langkah:
  - `eda`
  - `preprocess`
  - `train_sentiment_baseline`
  - `train_sentiment_transformer`
  - `train_recommender`
  - `evaluate`
  - `all`
- Proteksi anti training otomatis:
  - Semua step training butuh flag `--allow-training`.
  - `--step all` default menjalankan non-training saja.
- Script submit training diperbarui:
  - `scripts/submit_training.sh` kini wajib menerima argumen step.
- Hygiene repository untuk artifact besar:
  - `.gitignore` diperluas untuk `data/processed`, `models`, `reports`, `logs`, `mlruns`.
  - File hasil training yang sebelumnya ter-track sudah dihapus dari index git (tetap ada lokal bila dibutuhkan).

## Yang Belum Selesai
- Integrasi penuh dashboard Streamlit untuk trigger seluruh step pipeline.
- Notebook EDA (`notebooks/`) belum dibuat.
- `README.md` project utama masih perlu diisi detail lengkap pemakaian.
- Validasi end-to-end step transformer pada environment yang sudah terpasang `transformers` + `torch`.
- Evaluasi lanjutan recommender (tuning hyperparameter + perbaikan kualitas ranking).

## Catatan Operasional
- Contoh menjalankan non-training:
  - `python3 machine_learning/main.py --step eda`
  - `python3 machine_learning/main.py --step preprocess`
- Contoh menjalankan training (eksplisit):
  - `python3 machine_learning/main.py --step train_sentiment_baseline --allow-training`
  - `python3 machine_learning/main.py --step train_sentiment_transformer --allow-training`
  - `python3 machine_learning/main.py --step train_recommender --allow-training`

## Status
- Kondisi saat ini siap untuk lanjut coding tanpa risiko training tidak sengaja.
