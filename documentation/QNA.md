# Q&A Teknis Project

Dokumen ini merangkum pertanyaan yang paling sering muncul berdasarkan kondisi implementasi saat ini.

Tanggal pembaruan: 2026-05-09.

## 1. Apakah training `with_worker` sudah full distributed?

Belum. Yang distributed saat ini adalah tahap preprocessing Spark di worker. Training sentiment/recommender tetap berjalan lokal di master.

## 2. Bedanya `without_worker` dan `with_worker` apa?

- `without_worker`: source preprocess dari dataset lokal master.
- `with_worker`: source preprocess dari output Spark di HDFS, lalu dijembatani ke local untuk training.

## 3. Bagaimana cara menjalankan komparasi dua mode?

```bash
python3 machine_learning/main.py --step compare_training_modes --allow-training --ram-limit-gb 3
```

## 4. Di mana hasil komparasi disimpan?

- `machine_learning/reports/training_mode_comparison.json`
- `machine_learning/reports/experiments/without_worker_latest_run.json`
- `machine_learning/reports/experiments/with_worker_latest_run.json`

## 5. Jika `with_worker` gagal saat auto submit Spark preprocess, apa yang terjadi?

`compare_training_modes` akan mencoba retry `with_worker` tanpa auto submit (`run_worker_preprocess=False`). Jika retry sukses, report status bisa tetap `partial_success` atau `success` disertai warning mode.

## 6. Kenapa muncul warning `Parser C kehabisan memori`?

Karena parser C pandas gagal alokasi memori saat membaca CSV besar. Implementasi sekarang sudah fallback otomatis ke `engine='python'` pada reader terkait.

## 7. Kenapa compare mode dulu sering gagal di recommender?

Penyebab umum:
- pembacaan `processed_reviews.csv` terlalu berat,
- parser/memory issue,
- data terlalu besar untuk footprint proses.

Mitigasi yang sudah ada di kode:
- loader recommender membaca CSV secara chunked,
- fallback engine `c` -> `python`,
- pengisian default nilai kosong untuk kolom penting.

## 8. Apa fungsi `--ram-limit-gb`?

Mengatur batas RAM proses master melalui `RLIMIT_AS` di runtime training. Default di config adalah 3 GB.

## 9. Kenapa batas RAM kadang tidak diterapkan?

Jika proses sudah memakai memori virtual cukup besar ditambah safety margin, runtime sengaja melewati enforcement untuk mencegah crash instan.

## 10. Apakah bisa menjalankan pipeline tanpa risiko training otomatis?

Bisa. Step training diblokir default. Training hanya jalan jika Anda menambahkan `--allow-training`.

## 11. Spark submit di project ini dipakai untuk step apa?

Untuk `preprocess_spark` (distributed preprocessing). Step training model klasik tetap di master.

## 12. Apa output `preprocess_spark`?

Folder Parquet di HDFS:
- `hdfs://<namenode>/<output_hdfs_path>/processed`

## 13. Bagaimana cara memastikan output Spark ada?

```bash
hdfs dfs -ls -h /user/fadhli/output/amazon_books_ml
hdfs dfs -ls -h /user/fadhli/output/amazon_books_ml/processed
```

## 14. Bagaimana cara download output HDFS?

CLI:
```bash
hdfs dfs -get /user/fadhli/output/amazon_books_ml/processed ./hasil_dari_hdfs
```

Atau gunakan fitur `Download Dari HDFS` di tab `Cluster` pada dashboard.

## 15. Kenapa Spark job bisa lama di ACCEPTED?

Penyebab tipikal:
- NodeManager tidak aktif,
- ResourceManager tidak reachable,
- resolusi hostname salah (loopback).

Script `spark_submit_training.sh` sudah punya preflight `yarn node -list` untuk deteksi lebih cepat.

## 16. Apa saja KPI yang dibandingkan antar mode?

Ringkasan di report:
- total duration,
- sentiment test F1,
- sentiment test accuracy,
- recommender RMSE,
- recommender MAE,
- recommender NDCG@10,
- peak memory per stage dan total.

## 17. Bagaimana membaca delta komparasi?

`comparison` memakai format `with_worker - without_worker`.
- Nilai negatif durasi: `with_worker` lebih cepat.
- Nilai positif F1/accuracy: `with_worker` lebih baik metrik klasifikasi.
- RMSE/MAE lebih kecil biasanya lebih baik (jadi delta negatif berarti perbaikan).

## 18. Apa itu status `partial_success` pada report compare?

Artinya sebagian mode berhasil, sebagian mode gagal. Detail error ada pada field `errors`, warning ada pada field `warnings`.

## 19. Bagaimana menjalankan mode `with_worker` dari dashboard?

Di tab `Pipeline`:
1. Pilih step `train_pipeline` atau `compare_training_modes`.
2. Centang `Izinkan Training`.
3. Pilih `training_mode = with_worker`.
4. Atur `run_worker_preprocess` sesuai kebutuhan.
5. Jalankan.

## 20. Dokumen mana yang harus dibaca untuk memahami semua file kode?

Lihat dokumentasi lengkap file dan fungsi di:
- `documentation/structure.md`
