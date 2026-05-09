# History Pengerjaan

Dokumen ini mencatat perubahan penting lintas project.

## 2026-05-09

### A. Pembaruan Kode yang Sudah Ada

1. Runtime komparasi mode training sudah aktif:
- `train_pipeline`
- `compare_training_modes`
- report detail per mode + report delta komparasi.

2. Memory handling diperkuat:
- fallback parser pembacaan CSV besar pada training sentiment/recommender,
- pembatas RAM master berbasis `RLIMIT_AS` dengan safety margin.

3. Jalur `with_worker` distabilkan:
- preflight YARN sebelum Spark submit,
- retry compare mode bila auto worker preprocess gagal,
- bridge HDFS->local cache untuk membaca output Spark.

### B. Pembaruan Dokumentasi Menyeluruh (turn ini)

Semua file dokumentasi diperbarui agar sinkron dengan kondisi kode saat ini:

- `README.md`
- `documentation/arsitecture_plant.md`
- `documentation/tutorial.md`
- `documentation/QNA.md`
- `documentation/structure.md`
- `documentation/what_i_have.md`
- `documentation/what_you_do.md`
- `machine_learning/documentation/plant.md`
- `machine_learning/documentation/report_progres.md`
- `machine_learning/notebooks/README.md`

Poin kunci pembaruan:
1. Struktur kode aktual didokumentasikan ulang secara lengkap.
2. Semua file kode dan fungsi dipetakan pada `documentation/structure.md`.
3. Tutorial CLI/dashboard diperbarui sesuai flow terbaru.
4. Arsitektur `with_worker` vs `without_worker` dijelaskan sebagai kondisi as-is.
5. Q&A teknis diperbarui dengan kasus memori, fallback, compare status, dan interpretasi report.

### C. Fokus Lanjutan yang Direkomendasikan

1. Jalankan benchmark komparasi pada cluster yang stabil untuk mendapatkan angka final penelitian.
2. Tambahkan validasi otomatis (test/smoke test) agar regresi pada pipeline cepat terdeteksi.
3. Jika ingin scale-up, rancang migrasi training ke komputasi distributed penuh.
