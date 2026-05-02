# Apa yang Telah Dilakukan (what_you_do.md)

Tanggal: 2026-05-02

---

## 1. Code Review & Bug Fix — `machine_learning/src/`

### Bug Kritis yang Diperbaiki

| File | Masalah | Solusi |
|---|---|---|
| `src/evaluate.py` | **Tidak ada** — dipanggil `main.py --step evaluate` tapi file belum dibuat | File dibuat baru |
| `src/inference.py` | **Tidak ada** — diimport di `streamlit/app.py` → dashboard crash saat startup | File dibuat baru |
| `scripts/upload_to_hdfs.sh` | File **kosong**, belum ada implementasi | Diimplementasi penuh |
| `train_recommender.py` L163 | `groupby("item_id", as_index=False).apply()` — pandas ≥2.0 tidak lagi otomatis rename kolom | Ganti ke `.groupby().apply().reset_index().rename()` |
| `train_sentiment_transformer.py` L164 | `tokenizer=tokenizer` di `Trainer` deprecated di transformers ≥4.40 | Ditambahkan shim kompatibilitas menggunakan `inspect.signature` |
| `preprocessing.py` L50 | `set[str]` type hint tidak valid di Python < 3.9 | Diganti ke `Set[str]` dari `typing` |

### File Baru yang Dibuat

- **`machine_learning/src/evaluate.py`** — Mengkompilasi `final_report.json` dari semua metrik baseline, transformer, dan recommender yang tersedia.
- **`machine_learning/src/inference.py`** — Dua fungsi: `predict_sentiment(config, text)` menggunakan model baseline terbaik dari registry, `recommend_for_user(config, user_id, top_n)` membaca dari hybrid_recommendations.csv.
- **`machine_learning/src/spark_preprocess.py`** — Script PySpark distributed preprocessing yang membaca dataset dari HDFS dan menyimpan hasil Parquet kembali ke HDFS. Berjalan di YARN, worker node membantu proses komputasi.

---

## 2. Infrastruktur Hadoop & Spark

### File Baru

- **`config/cluster.yaml`** — Konfigurasi terpusat: hostname worker, SSH user, HDFS path, YARN port, Spark resource (num-executors, executor-memory, dll).
- **`scripts/upload_to_hdfs.sh`** — Upload semua CSV dari `machine_learning/dataset/` ke HDFS target path. Termasuk validasi apakah `hdfs` command tersedia.
- **`scripts/spark_submit_training.sh`** — Wrapper spark-submit ke YARN dengan parameter: step, num-executors, executor-cores, executor-memory, driver-memory.

### Konfigurasi Ditambahkan

- **`machine_learning/config.yaml`** — Ditambah section `hadoop` (namenode URI, HDFS paths, UI ports) dan `spark` (master, deploy-mode, resource defaults).

---

## 3. Streamlit Dashboard — Full Rebuild

`streamlit/app.py` di-rebuild total dengan **6 tab**:

| Tab | Fitur |
|---|---|
| 🏠 **Overview** | Status artifact (ada/tidak + ukuran), quick metrics dari final_report.json |
| 📊 **EDA** | Grafik distribusi rating & panjang review (PNG), tabel statistik, contoh review per sentimen |
| 🚀 **Pipeline** | Dropdown step, toggle allow-training, jalankan pipeline dengan output stdout/stderr |
| 📈 **Reports** | Tabel metrik semua baseline model, confusion matrix images, metrik recommender (RMSE, MAE, Precision@K, NDCG@K), model registry |
| 🔮 **Inference** | Sentiment inference (teks → label + confidence + probabilitas), recommender inference (user_id → top-N books) |
| 🖥️ **Cluster** | SSH status check ke worker1 & worker2, link ke HDFS UI & YARN UI, Start/Stop cluster, upload HDFS, Spark Submit launcher dengan konfigurasi resource, browse HDFS path |

---

## 4. Catatan Arsitektur

- Training ML (baseline, transformer, recommender) tetap berjalan di **master node** menggunakan scikit-learn.
- **Worker node** berperan aktif dalam `preprocess_spark` menggunakan PySpark + YARN:
  - worker membaca block HDFS secara paralel
  - cleaning dan labeling distributed di executor
  - hasil disimpan ke HDFS sebagai Parquet
- Untuk training distributed penuh, perlu migrasi ke **PySpark MLlib** (rencana pengembangan selanjutnya).

---

## 5. Status File

```
machine_learning/src/
  ✅ data_loader.py       — Tidak diubah (sudah benar)
  ✅ eda.py               — Tidak diubah (sudah benar)
  ✅ preprocessing.py     — Fix type hint Python 3.8
  ✅ train_sentiment.py   — Tidak diubah (sudah benar)
  ✅ train_recommender.py — Fix pandas ≥2.0 bug
  ✅ train_sentiment_transformer.py — Fix deprecated tokenizer param
  ✅ mlflow_tracker.py    — Tidak diubah (sudah benar)
  ✅ utils.py             — Tidak diubah (sudah benar)
  🆕 evaluate.py          — BARU
  🆕 inference.py         — BARU
  🆕 spark_preprocess.py  — BARU

scripts/
  ✅ start_cluster.sh     — Tidak diubah
  ✅ stop_cluster.sh      — Tidak diubah
  ✅ submit_training.sh   — Tidak diubah
  🔧 upload_to_hdfs.sh   — Diimplementasi (sebelumnya kosong)
  🆕 spark_submit_training.sh — BARU

config/
  🆕 cluster.yaml         — BARU

machine_learning/
  🔧 config.yaml          — Ditambah section hadoop & spark

streamlit/
  🔧 app.py               — Rebuild total (6 tab)

documentation/
  🔧 what_you_do.md       — File ini
```
