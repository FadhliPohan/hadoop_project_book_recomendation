# Apa yang Telah Dilakukan

Tanggal update: 2026-05-09

## 1. Pembaruan Arsitektur Pipeline

1. Menambahkan mode training eksplisit:
- `without_worker`
- `with_worker`

2. Menambahkan step pipeline baru:
- `train_pipeline`
- `compare_training_modes`

3. Menambahkan orchestrator runtime baru di `machine_learning/src/training_runtime.py` untuk:
- progress stage,
- timing per stage,
- peak memory per stage,
- report per mode,
- report komparasi dua mode.

## 2. Hardening Stabilitas Runtime

1. Proteksi memory master:
- `apply_master_ram_limit` (default 3GB dari config).

2. Fallback pembacaan data besar:
- `train_sentiment._read_split_csv` fallback parser C -> python engine.
- `train_recommender._load_processed_interactions` memakai pembacaan chunked + fallback engine.

3. Bridge HDFS yang lebih stabil:
- cache persisten,
- progress logging,
- warning saat throughput lambat.

## 3. Integrasi Jalur Worker

1. Spark preprocessing distributed ditangani `spark_preprocess.py`.
2. Wrapper submit `scripts/spark_submit_training.sh` ditingkatkan dengan:
- preflight YARN,
- warning hostname loopback,
- parameter env preprocess.
3. Runtime compare `with_worker` punya retry otomatis jika auto submit preprocess gagal.

## 4. Pembaruan Dashboard Streamlit

1. Tab `Pipeline` mendukung:
- mode training,
- include transformer,
- auto worker preprocess,
- RAM limit,
- live stdout/stderr,
- kontrol timeout.

2. Tab `Reports` menampilkan:
- KPI per mode,
- detail stage timing dan peak memory,
- tabel delta metrik `with_worker - without_worker`,
- warning/error komparasi.

3. Tab `Cluster` mendukung:
- start/stop cluster,
- spark submit,
- upload/download HDFS,
- reset training state,
- permission dan network helper.

## 5. Pembaruan Dokumentasi Menyeluruh

Semua dokumentasi sudah disinkronkan ulang ke kondisi kode terbaru:

- `README.md`
- `documentation/arsitecture_plant.md`
- `documentation/tutorial.md`
- `documentation/QNA.md`
- `documentation/structure.md`
- `documentation/history.md`
- `documentation/what_i_have.md`
- `documentation/what_you_do.md`
- `machine_learning/documentation/plant.md`
- `machine_learning/documentation/report_progres.md`
- `machine_learning/notebooks/README.md`

Poin terpenting:
- `documentation/structure.md` sekarang berisi pemetaan semua file kode dan fungsi per file.

## 6. Kondisi Saat Ini

1. Project siap dipakai untuk eksperimen komparasi training mode.
2. Arsitektur masih hybrid:
- worker untuk preprocessing,
- training model final tetap di master.
3. Fondasi dokumentasi kini konsisten dengan implementasi aktual.
