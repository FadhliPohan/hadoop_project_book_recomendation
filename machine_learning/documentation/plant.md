Kamu adalah seorang Senior Data Scientist dan Machine Learning Engineer yang ahli dalam NLP, Sentiment Analysis, dan Recommender System menggunakan Python.

Saya ingin membuat project machine learning dengan judul:

**Review Sentiment Analysis and Recommender System menggunakan dataset Amazon Books Reviews dari Kaggle**

Dataset:
https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data

Tujuan project:

1. Melakukan analisis sentimen terhadap review buku Amazon.
2. Membangun sistem rekomendasi buku berdasarkan rating, user behavior, dan hasil sentimen.
3. Membuat pipeline training yang rapi, modular, reproducible, dan setiap hasil training wajib disimpan.
4. Menggunakan Python sebagai bahasa utama.
5. Project harus cocok untuk kebutuhan akademik, portofolio, dan bisa dikembangkan ke production-level.
6. lokasi pengkodingan pada folder machine_learning

Buatkan saya project secara lengkap dengan tahapan berikut:

---

## 1. Project Planning dan Arsitektur

Jelaskan arsitektur lengkap project dari awal sampai akhir, mencakup:

* Data ingestion
* Data validation
* Exploratory Data Analysis
* Data preprocessing
* Feature engineering
* Sentiment analysis model
* Recommender system model
* Model evaluation
* Model saving
* Experiment tracking
* Final report

Buat struktur folder project yang rapi seperti:

project/
├── data/
│   ├── raw/
│   ├── processed/
├── notebooks/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train_sentiment.py
│   ├── train_recommender.py
│   ├── evaluate.py
│   ├── utils.py
├── models/
│   ├── sentiment/
│   ├── recommender/
├── reports/
├── logs/
├── mlruns/
├── config.yaml
├── requirements.txt
└── main.py

Jelaskan fungsi setiap folder dan file.

---

## 2. Data Understanding dan EDA

Lakukan analisis data secara lengkap.

Dataset kemungkinan memiliki kolom seperti:

* reviewerID
* asin
* reviewerName
* helpful
* reviewText
* overall
* summary
* unixReviewTime
* reviewTime

Tugas kamu:

* Load dataset menggunakan pandas.
* Tampilkan ukuran dataset.
* Tampilkan tipe data setiap kolom.
* Cek missing values.
* Cek data duplikat.
* Analisis distribusi rating.
* Analisis panjang review.
* Analisis jumlah user unik.
* Analisis jumlah produk/buku unik.
* Analisis jumlah review per user.
* Analisis jumlah review per buku.
* Visualisasikan distribusi rating.
* Visualisasikan distribusi panjang review.
* Tampilkan contoh review positif, netral, dan negatif.
* Berikan insight dari hasil EDA.

Simpan hasil EDA dalam bentuk:

* Grafik PNG
* Ringkasan teks
* File CSV hasil statistik dasar

---

## 3. Labeling Sentiment

Buat label sentimen berdasarkan kolom rating `overall`.

Aturan label:

* Rating 1 dan 2 = Negative
* Rating 3 = Neutral
* Rating 4 dan 5 = Positive

Buat kolom baru bernama:

`sentiment_label`

Mapping label:

* Negative = 0
* Neutral = 1
* Positive = 2

Tampilkan distribusi label sentimen dan cek apakah data imbalance.

Jika data imbalance, berikan solusi seperti:

* class weight
* undersampling
* oversampling
* stratified split

---

## 4. Text Preprocessing

Buat pipeline preprocessing teks untuk kolom `reviewText`.

Langkah preprocessing:

* Mengubah teks menjadi lowercase
* Menghapus URL
* Menghapus HTML tag
* Menghapus angka jika tidak diperlukan
* Menghapus tanda baca
* Menghapus special character
* Menghapus extra whitespace
* Menghapus stopwords bahasa Inggris
* Lemmatization
* Tokenization

Buat fungsi Python modular:

* clean_text()
* remove_stopwords()
* lemmatize_text()
* preprocess_text()

Simpan hasil preprocessing ke:

`data/processed/processed_reviews.csv`

Pastikan preprocessing tidak menyebabkan data leakage.

---

## 5. Data Splitting

Pisahkan data menjadi:

* Training set
* Validation set
* Test set

Gunakan stratified split berdasarkan `sentiment_label`.

Rasio:

* Train: 70%
* Validation: 15%
* Test: 15%

Simpan hasil split ke folder:

data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv

---

## 6. Baseline Sentiment Model

Bangun baseline model terlebih dahulu menggunakan:

* TF-IDF Vectorizer
* Logistic Regression
* Naive Bayes
* Linear SVM

Lakukan training pada data train.

Evaluasi pada validation dan test set menggunakan metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Classification report

Simpan semua model baseline ke folder:

models/sentiment/baseline/

Simpan juga:

* vectorizer
* model
* metrics JSON
* confusion matrix PNG
* classification report TXT

Gunakan joblib untuk menyimpan model.

---

## 7. Advanced Sentiment Model

Bangun model advanced menggunakan Transformer.

Gunakan salah satu model berikut:

* distilbert-base-uncased
* bert-base-uncased
* roberta-base

Prioritaskan `distilbert-base-uncased` karena lebih ringan.

Langkah yang harus dilakukan:

* Tokenisasi menggunakan HuggingFace tokenizer
* Membuat Dataset class
* Fine-tuning model untuk 3 kelas sentimen
* Menggunakan train, validation, dan test split
* Menggunakan early stopping
* Menggunakan learning rate scheduler
* Menyimpan checkpoint setiap epoch
* Menyimpan model terbaik berdasarkan validation F1-score

Training configuration:

* max_length = 128 atau 256
* batch_size = 16
* learning_rate = 2e-5
* epochs = 3 sampai 5
* optimizer = AdamW
* metric utama = weighted F1-score

Simpan model ke:

models/sentiment/transformer/

Simpan:

* tokenizer
* model
* training arguments
* metrics
* logs
* classification report
* confusion matrix
* best checkpoint

---

## 8. Experiment Tracking

Gunakan MLflow untuk tracking semua eksperimen.

Setiap training wajib mencatat:

* Nama model
* Hyperparameter
* Dataset version
* Train accuracy
* Validation accuracy
* Test accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Model artifact
* Vectorizer/tokenizer artifact
* Waktu training

Buat experiment MLflow dengan nama:

`amazon_books_sentiment_recommender`

Pastikan setiap run memiliki nama yang jelas, contoh:

* tfidf_logistic_regression_v1
* tfidf_svm_v1
* distilbert_sentiment_v1
* svd_recommender_v1
* hybrid_recommender_v1

---

## 9. Recommender System

Bangun recommender system berdasarkan dataset Amazon Books.

Gunakan kolom:

* reviewerID sebagai user_id
* asin sebagai item_id
* overall sebagai rating
* sentiment_label atau sentiment_score sebagai tambahan fitur

Bangun beberapa pendekatan:

### A. Popularity-Based Recommender

Rekomendasikan buku berdasarkan:

* rata-rata rating tertinggi
* jumlah review minimum
* skor popularitas

Simpan hasil rekomendasi ke CSV.

### B. Collaborative Filtering

Gunakan matrix factorization seperti:

* SVD menggunakan library Surprise
* Alternatif: implicit ALS jika cocok

Evaluasi menggunakan:

* RMSE
* MAE

Simpan model ke:

models/recommender/collaborative_filtering/

### C. Content-Based Filtering

Gunakan fitur teks:

* reviewText
* summary
* metadata buku jika tersedia

Gunakan TF-IDF dan cosine similarity.

Simpan similarity matrix atau pipeline model.

### D. Hybrid Recommender

Gabungkan:

* Collaborative filtering score
* Popularity score
* Sentiment score

Rumus contoh:

final_score = 0.5 * collaborative_score + 0.3 * sentiment_score + 0.2 * popularity_score

Berikan penjelasan kenapa hybrid recommender lebih kuat.

Simpan model hybrid ke:

models/recommender/hybrid/

---

## 10. Evaluasi Recommender System

Evaluasi recommender menggunakan:

* RMSE
* MAE
* Precision@K
* Recall@K
* NDCG@K
* Coverage
* Diversity jika memungkinkan

Gunakan K = 5 dan K = 10.

Buat fungsi:

* precision_at_k()
* recall_at_k()
* ndcg_at_k()

Simpan hasil evaluasi ke:

reports/recommender_metrics.json

---

## 11. Model Saving dan Versioning

Setiap hasil training wajib disimpan.

Gunakan format berikut:

models/
├── sentiment/
│   ├── baseline/
│   │   ├── tfidf_logreg_v1.pkl
│   │   ├── tfidf_svm_v1.pkl
│   │   ├── metrics.json
│   ├── transformer/
│   │   ├── distilbert_v1/
│   │   │   ├── model/
│   │   │   ├── tokenizer/
│   │   │   ├── metrics.json
├── recommender/
│   ├── popularity/
│   ├── collaborative_filtering/
│   ├── content_based/
│   ├── hybrid/

Setiap file model harus memiliki metadata:

* nama model
* tanggal training
* dataset yang digunakan
* hyperparameter
* metrics
* versi model
* path model

Buat file:

`model_registry.json`

untuk mencatat semua model yang sudah pernah dilatih.

---

## 12. Inference Pipeline

Buat script inference untuk sentiment analysis.

Input:

* teks review baru

Output:

* label sentimen
* confidence score
* probabilitas setiap kelas

Contoh:

Input:
"This book is very helpful and well written."

Output:
{
"sentiment": "Positive",
"confidence": 0.94,
"probabilities": {
"Negative": 0.02,
"Neutral": 0.04,
"Positive": 0.94
}
}

Buat juga inference recommender.

Input:

* user_id

Output:

* top-N book recommendation

---

## 13. Main Pipeline

Buat file `main.py` yang bisa menjalankan pipeline:

Pilihan command:

python main.py --step eda
python main.py --step preprocess
python main.py --step train_sentiment_baseline
python main.py --step train_sentiment_transformer
python main.py --step train_recommender
python main.py --step evaluate
python main.py --step all

Gunakan argparse.

---

## 14. Requirements

Buat file `requirements.txt` berisi library:

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* nltk
* spacy
* transformers
* torch
* datasets
* evaluate
* mlflow
* joblib
* surprise
* scipy
* tqdm
* pyyaml

---

## 15. Dokumentasi

Buat README.md yang menjelaskan:

* Deskripsi project
* Dataset
* Tujuan project
* Struktur folder
* Cara instalasi
* Cara menjalankan EDA
* Cara preprocessing
* Cara training sentiment model
* Cara training recommender system
* Cara melihat hasil MLflow
* Cara menjalankan inference
* Hasil evaluasi model
* Kesimpulan
* Pengembangan selanjutnya

---

## 16. Output Akhir yang Saya Inginkan

Saya ingin kamu menghasilkan:

1. Arsitektur project
2. Struktur folder
3. Semua script Python modular
4. Notebook EDA
5. Pipeline preprocessing
6. Training baseline sentiment model
7. Training transformer sentiment model
8. Training recommender system
9. Evaluasi lengkap
10. Penyimpanan model otomatis
11. MLflow tracking
12. Inference script
13. README.md
14. requirements.txt
15. Penjelasan setiap langkah secara detail

---

## 17. Standar Kualitas

Pastikan kode:

* Clean
* Modular
* Reusable
* Mudah dipahami
* Menggunakan function dan class jika perlu
* Memiliki komentar penting
* Menghindari data leakage
* Menyimpan semua output penting
* Bisa dijalankan ulang
* Cocok untuk dataset besar
* Menggunakan random_state agar reproducible

Gunakan:

random_state = 42

---

## 18. Catatan Penting

Jangan langsung menggunakan model kompleks tanpa baseline.

Urutan pengerjaan wajib:

1. EDA
2. Preprocessing
3. Baseline sentiment model
4. Advanced sentiment model
5. Recommender sederhana
6. Recommender collaborative filtering
7. Hybrid recommender
8. Evaluation
9. Model saving
10. Documentation

Berikan penjelasan kenapa setiap langkah dilakukan, bukan hanya kode.

Jika ada dataset terlalu besar, berikan opsi sampling untuk eksperimen awal.

Jika ada error karena memory atau GPU tidak tersedia, berikan alternatif training yang lebih ringan.

Mulai dari membuat struktur project, lalu lanjutkan ke implementasi satu per satu.
  