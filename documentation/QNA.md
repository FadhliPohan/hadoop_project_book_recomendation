# Q&A Analisa Project Hadoop Amazon Books Reviews

Tanggal analisa: 2026-05-04

## Ringkasan Singkat

Project ini **bisa membantu proses training data dan training model**, tetapi dalam kondisi saat ini bantuan terbesarnya ada pada:

- pengelolaan dataset review buku Amazon
- preprocessing data
- training model sentiment baseline
- training model transformer
- training recommender system
- pelaporan hasil training lewat report, model registry, MLflow, dan dashboard Streamlit

Namun ada batasan penting:

- `worker node` saat ini **belum dipakai untuk full training model machine learning**
- `worker node` baru jelas dipakai pada proses **distributed preprocessing** lewat `PySpark` dan `YARN`
- sebagian besar artifact training dan report masih **disimpan di master/local project**, bukan di HDFS
- label sentiment dibuat otomatis dari rating, jadi ini bagus untuk bootstrapping dataset, tetapi bukan label manual yang benar-benar gold standard

---

## 1. Apa fungsi keseluruhan aplikasi ini?

Secara keseluruhan, aplikasi ini adalah gabungan antara:

- **Hadoop HDFS** untuk penyimpanan dataset besar
- **YARN** untuk pengelolaan resource cluster
- **Spark/PySpark** untuk preprocessing terdistribusi
- **pipeline machine learning lokal** untuk sentiment analysis dan recommender
- **Streamlit dashboard** untuk menjalankan pipeline, melihat artifact, dan melakukan inference

Jadi project ini bukan hanya model machine learning, tetapi sebuah **mini platform data + ML**.

---

## 2. Bagaimana alur kerja sistem ini?

Alur nyatanya saat ini seperti ini:

1. Dataset diletakkan di lokal project, terutama di `machine_learning/dataset/Books_rating.csv`.
2. Jika dibutuhkan untuk cluster, dataset bisa di-upload ke HDFS lewat `scripts/upload_to_hdfs.sh`.
3. Analisis awal dilakukan lewat step `eda`.
4. Preprocessing lokal dilakukan lewat step `preprocess`, yang menghasilkan file CSV processed dan split train/validation/test.
5. Training model dijalankan lewat `main.py` atau dashboard Streamlit.
6. Setelah training selesai, metrik, model, report, log, dan registry disimpan ke folder project.
7. Dashboard Streamlit membaca file hasil tersebut untuk ditampilkan kembali.

Kalau memakai jalur Spark:

1. Dataset dibaca dari HDFS.
2. Spark driver berjalan di master.
3. Executor berjalan di worker.
4. Hasil preprocessing Spark disimpan kembali ke HDFS dalam bentuk Parquet.

Poin pentingnya: **jalur Spark dan jalur training model lokal saat ini masih belum menyatu penuh**.

---

## 3. Apakah project ini bisa membantu untuk training datanya?

Jawabannya: **ya, bisa**, tetapi sifatnya **membantu dan mempercepat pipeline training**, belum menjadi sistem distributed training end-to-end.

### Yang sudah sangat membantu

- Struktur pipeline sudah rapi: `eda`, `preprocess`, `train_sentiment_baseline`, `train_sentiment_transformer`, `train_recommender`, `evaluate`.
- Ada proteksi `--allow-training`, sehingga training tidak jalan tanpa sengaja.
- Ada dashboard untuk trigger pipeline dan melihat hasil.
- Ada HDFS dan Spark untuk menangani data lebih besar.
- Ada `model_registry.json`, `final_report.json`, `logs`, dan `mlruns` untuk audit hasil.

### Yang perlu dipahami

- Sentiment label **tidak berasal dari label manual**, tetapi diturunkan dari rating:
  - rating `<= 2` menjadi `Negative`
  - rating `== 3` menjadi `Neutral`
  - rating `>= 4` menjadi `Positive`
- Artinya project ini cocok untuk **membangun dataset training secara otomatis** dari review + rating, tetapi kualitas label tetap bergantung pada konsistensi antara teks review dan rating user.
- Nilai `sample_rows` di `machine_learning/config.yaml` saat ini `120000`, jadi default training lokal belum otomatis memakai seluruh dataset besar.

### Kesimpulan untuk pertanyaan ini

Project ini **layak dipakai untuk membantu training data dan eksperimen model**, terutama untuk:

- pembersihan data
- pembentukan label awal
- training baseline
- evaluasi hasil
- tracking artifact

Tetapi jika targetnya adalah **training skala besar yang benar-benar memanfaatkan semua worker untuk model training**, project ini **belum sepenuhnya sampai ke sana**.

---

## 4. Kalau dilakukan training ulang, report akan tersimpan di mana?

Secara default, **mayoritas report tersimpan di master**, yaitu di filesystem lokal project ini.

### Lokasi output utama

| Jenis output | Lokasi | Tersimpan di mana | Perilaku saat training ulang |
|---|---|---|---|
| Processed CSV | `machine_learning/data/processed/processed_reviews.csv` | Master/local project | Ditimpa |
| Split train/val/test | `machine_learning/data/processed/` | Master/local project | Ditimpa |
| Baseline sentiment metrics | `machine_learning/models/sentiment/baseline/metrics.json` | Master/local project | Ditimpa |
| Baseline model `.pkl` | `machine_learning/models/sentiment/baseline/` | Master/local project | Ditimpa dengan nama model yang sama |
| Transformer metrics | `machine_learning/models/sentiment/transformer/distilbert_v1/metrics.json` | Master/local project | Ditimpa |
| Transformer model/tokenizer | `machine_learning/models/sentiment/transformer/distilbert_v1/` | Master/local project | Diupdate |
| Recommender metrics | `machine_learning/reports/recommender_metrics.json` | Master/local project | Ditimpa |
| Recommender artifact | `machine_learning/models/recommender/` | Master/local project | Ditimpa/diupdate |
| Final report | `machine_learning/reports/final_report.json` | Master/local project | Ditimpa saat step `evaluate` |
| Model registry | `machine_learning/models/model_registry.json` | Master/local project | **Ditambah append**, bukan reset penuh |
| Pipeline log | `machine_learning/logs/pipeline_*.log` | Master/local project | File baru setiap run |
| MLflow tracking | `machine_learning/mlruns/` | Master/local project | Run baru setiap training |
| Spark preprocessing output | `hdfs://fadhli:9000/user/fadhli/output/amazon_books_ml/processed` | HDFS | Ditimpa (`overwrite`) |

### Jawaban singkatnya

- **Report training model**: dominan tersimpan di **master**
- **Output preprocessing Spark**: tersimpan di **HDFS**
- **Worker node**: membantu komputasi Spark, tetapi **bukan lokasi utama penyimpanan report aplikasi**

---

## 5. Apakah saat ini training benar-benar berjalan di cluster?

**Belum sepenuhnya.**

### Yang sudah distributed

- `preprocess_spark`
- dataset dibaca dari HDFS
- eksekusi dibagi ke worker melalui Spark executor di YARN

### Yang masih berjalan di master

- `train_sentiment_baseline`
- `train_sentiment_transformer`
- `train_recommender`
- `evaluate`

Jadi, kalau Anda menekan tombol training biasa di Streamlit atau menjalankan `machine_learning/main.py`, maka proses utama training model masih berjalan di **master**.

Ini juga dipertegas oleh wrapper `scripts/spark_submit_training.sh`, karena selain `preprocess_spark`, step lain dijalankan lewat Python lokal di master.

---

## 6. Apakah hasil preprocessing Spark di HDFS langsung dipakai oleh training model?

**Belum otomatis.**

Ini adalah salah satu temuan terpenting dari analisa project ini.

Saat ini ada dua jalur:

- **jalur lokal**:
  - `preprocess`
  - menghasilkan `processed_reviews.csv`, `train.csv`, `validation.csv`, `test.csv`
  - file inilah yang dipakai training baseline, transformer, dan recommender

- **jalur Spark/HDFS**:
  - `preprocess_spark`
  - membaca CSV dari HDFS
  - menyimpan hasil ke HDFS dalam bentuk Parquet

Masalahnya, training lokal saat ini **belum membaca Parquet hasil Spark dari HDFS**. Jadi distributed preprocessing memang ada, tetapi belum otomatis menjadi input training lokal.

---

## 7. Jadi project ini kuatnya di bagian mana?

Project ini sudah kuat pada aspek:

- struktur pipeline yang modular
- pemisahan step yang jelas
- integrasi dashboard operasional
- tracking artifact hasil training
- penyimpanan dataset besar lewat HDFS
- preprocessing terdistribusi dengan Spark

Dengan kata lain, project ini sudah bagus sebagai:

- fondasi eksperimen machine learning
- fondasi dashboard monitoring hasil
- fondasi migrasi ke arsitektur ML yang lebih distributed

---

## 8. Apa kelemahan atau gap yang masih ada?

Beberapa gap penting:

- full training model belum distributed
- hasil `preprocess_spark` belum otomatis dipakai step training
- label sentiment masih rule-based dari rating, belum label manual
- default `sample_rows: 120000`, jadi dataset besar belum otomatis dipakai penuh
- ada dokumentasi lama yang menyebut `output/metrics.json`, tetapi kode aktif sekarang lebih banyak menyimpan report di folder `machine_learning/`

Jadi bila ada pertanyaan, "apakah report training ulang akan tersimpan di `output/metrics.json` atau di master?", maka jawaban yang paling tepat untuk **kode yang aktif sekarang** adalah:

- **bukan utama di `output/metrics.json`**
- **utama di folder `machine_learning/` pada master**

---

## 9. Jika ingin project ini benar-benar optimal untuk training skala besar, apa yang perlu ditingkatkan?

Saran pengembangan selanjutnya:

1. Ubah step training agar bisa membaca hasil Parquet dari HDFS.
2. Migrasikan training tertentu ke pendekatan distributed, misalnya Spark MLlib atau framework training terdistribusi lain.
3. Pisahkan artifact besar ke storage bersama, misalnya HDFS atau object storage, bukan hanya local master.
4. Tambahkan versioning dataset dan versioning model yang lebih formal.
5. Tambahkan evaluasi kualitas label sentiment, karena saat ini label dibentuk dari rating.

---

## 10. Kesimpulan akhir

Project ini **sudah bisa membantu proses training data dan training model**, terutama untuk kebutuhan eksperimen, preprocessing, evaluasi, dan visualisasi hasil.

Tetapi arsitektur saat ini lebih tepat disebut:

- **distributed data/preprocessing**
- **local model training on master**

Bukan:

- **distributed end-to-end model training**

Jadi jika dilakukan training ulang sekarang, maka:

- report utama tersimpan di **master/local project**
- log tersimpan di **master**
- MLflow run tersimpan di **master**
- hanya output preprocessing Spark yang tersimpan di **HDFS**

Jika tujuan akhirnya adalah memanfaatkan cluster untuk training penuh, maka project ini sudah menjadi fondasi yang bagus, tetapi masih perlu satu tahap integrasi lagi pada sisi training model.
