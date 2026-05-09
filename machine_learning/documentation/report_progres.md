# Report Progres Pengembangan Machine Learning

Tanggal update: 09 Mei 2026

## 1. Status Umum

- Mode pengerjaan: development + dokumentasi sinkronisasi total.
- Guard training: aktif (step training wajib `--allow-training`).
- Dokumentasi: sudah diperbarui menyeluruh mengikuti kondisi kode aktual.

## 2. Yang Sudah Selesai

1. Pipeline modular lengkap di `machine_learning/src`:
- data loading local/HDFS bridge,
- preprocessing + split,
- EDA,
- baseline sentiment,
- transformer sentiment,
- recommender hybrid,
- evaluator final report,
- inference,
- MLflow tracker.

2. Runtime orchestration komparasi mode:
- `train_pipeline` per mode,
- `compare_training_modes`,
- report per-run dan report komparasi.

3. Stabilitas memori diperkuat:
- fallback parser CSV pada modul sentiment/recommender,
- chunked reading untuk data interaction recommender,
- RAM limit master dengan safety margin.

4. Integrasi Spark preprocess:
- script submit dengan preflight YARN,
- opsi sampling/max_rows/output_partitions,
- bridge output HDFS ke local training flow.

5. Dashboard Streamlit terintegrasi:
- live command output,
- kontrol mode training,
- laporan komparasi detail,
- operasi cluster/HDFS,
- inference sentiment dan rekomendasi.

6. Dokumentasi diperbarui total:
- `README.md`
- `documentation/*`
- `machine_learning/documentation/*`
- `machine_learning/notebooks/README.md`

## 3. Yang Belum Selesai

1. Benchmark komparasi final pada cluster stabil (angka eksperimen final).
2. Otomasi test/smoke test untuk validasi artifact per step.
3. Optimasi lanjutan untuk mode `with_worker` pada jaringan VM lambat.
4. Rencana migrasi training distributed penuh jika dibutuhkan skala besar.

## 4. Catatan Operasional

Command inti:

```bash
python3 machine_learning/main.py --step eda
python3 machine_learning/main.py --step preprocess
python3 machine_learning/main.py --step train_pipeline --allow-training --training-mode without_worker --ram-limit-gb 3
python3 machine_learning/main.py --step train_pipeline --allow-training --training-mode with_worker --ram-limit-gb 3
python3 machine_learning/main.py --step compare_training_modes --allow-training --ram-limit-gb 3
python3 machine_learning/main.py --step evaluate
```

Artifact komparasi:
- `machine_learning/reports/training_mode_comparison.json`
- `machine_learning/reports/experiments/without_worker_latest_run.json`
- `machine_learning/reports/experiments/with_worker_latest_run.json`

## 5. Status Akhir Saat Ini

Project berada pada status:
- siap untuk eksekusi eksperimen komparasi,
- siap untuk penyusunan laporan akademik berbasis artifact JSON,
- siap untuk iterasi optimasi performa cluster.
