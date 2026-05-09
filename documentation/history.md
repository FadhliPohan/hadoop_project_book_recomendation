# History Pengerjaan

## 2026-05-09

### Sudah Dikerjakan

1. Analisa kode + dokumentasi proyek selesai:
- membaca seluruh alur `main.py`, modul `src/`, script Hadoop/Spark, dashboard Streamlit, dan dokumentasi.

2. Implementasi mode training baru selesai:
- menambah `training-mode`:
  - `without_worker`
  - `with_worker`
- menambah step CLI:
  - `train_pipeline`
  - `compare_training_modes`

3. Implementasi runtime training & komparasi selesai:
- file baru: `machine_learning/src/training_runtime.py`
- fitur:
  - enforce batas RAM master (default 3GB dari config)
  - jalankan training pipeline per mode
  - opsi auto submit Spark preprocess untuk mode `with_worker`
  - catat durasi per stage
  - catat peak memory master
  - simpan report per mode & report komparasi

4. Integrasi ke pipeline utama selesai:
- `machine_learning/main.py` diperbarui agar mendukung mode baru + opsi:
  - `--training-mode`
  - `--include-transformer`
  - `--run-worker-preprocess`
  - `--ram-limit-gb`

5. Integrasi report akhir selesai:
- `machine_learning/src/evaluate.py` kini juga membaca `training_mode_comparison.json` bila tersedia.

6. Integrasi dashboard Streamlit selesai:
- tab Pipeline:
  - pilihan step komparasi
  - pilihan training mode
  - input batas RAM master
  - toggle auto Spark preprocess
- tab Overview/Reports:
  - tampilkan artifact komparasi
  - ringkasan delta metric/waktu antar mode

7. Pembaruan konfigurasi selesai:
- `machine_learning/config.yaml` ditambah:
  - `training.master_ram_limit_gb: 3`
  - path report komparasi/eksperimen

8. Pembaruan dokumentasi selesai:
- `README.md`
- `documentation/tutorial.md`
- `documentation/what_you_do.md`
- `machine_learning/documentation/report_progres.md`

### Belum Dikerjakan / Lanjutan Berikutnya

1. Menjalankan benchmark komparasi end-to-end di cluster aktif:
- command target:
  - `python3 machine_learning/main.py --step compare_training_modes --allow-training --ram-limit-gb 3`
- prasyarat:
  - HDFS + YARN aktif
  - output Spark preprocess tersedia jika mode `with_worker` tanpa auto-submit

2. Mengumpulkan hasil final penelitian dari run aktual:
- bandingkan:
  - waktu total
  - waktu per stage
  - sentiment F1/accuracy
  - recommender RMSE/MAE/NDCG
- sumber utama:
  - `machine_learning/reports/training_mode_comparison.json`

3. Jika mode `with_worker` gagal:
- cek cluster:
  - `bash scripts/start_cluster.sh`
  - `hdfs dfs -ls -h /user/fadhli/output/amazon_books_ml`
- bila perlu generate output preprocess:
  - `bash scripts/spark_submit_training.sh preprocess_spark`

### Update Tambahan (Fix Timeout Streamlit Pipeline)

1. Timeout pipeline di dashboard diperbaiki:
- tab `Pipeline` sekarang punya opsi:
  - `Tanpa timeout (recommended untuk training panjang)`
  - `Timeout Pipeline (detik)` yang bisa diatur manual
- default timeout dibuat panjang (`10800` detik) saat timeout aktif.

2. Live terminal training di Streamlit diaktifkan:
- eksekusi di tab `Pipeline` sekarang pakai runner live (`_run_live`), bukan blocking `subprocess.run`.
- stdout/stderr proses training tampil real-time selama command berjalan.

3. Dampak:
- kasus `❌ Gagal karena timeout` saat training panjang dapat dihindari dengan menyalakan mode tanpa timeout.
- progres training bisa dipantau langsung dari dashboard tanpa buka terminal terpisah.
