# Notebooks

Folder ini dipakai untuk eksplorasi interaktif yang melengkapi pipeline script.

## Status Saat Ini

Pipeline utama sudah berjalan dari modul script (`machine_learning/main.py` + `machine_learning/src/*`).
Notebook belum dijadikan entrypoint utama agar eksekusi tetap reproducible dari CLI/Streamlit.

## Rencana Notebook

1. `01_eda_exploration.ipynb`
- eksplorasi distribusi data,
- validasi insight dari `reports/eda/*`.

2. `02_sentiment_error_analysis.ipynb`
- analisis error model baseline/transformer,
- inspeksi confusion matrix dan contoh prediksi salah.

3. `03_recommender_analysis.ipynb`
- analisis coverage/diversity/relevansi rekomendasi,
- inspeksi scoring hybrid per user.

## Prinsip Penggunaan

1. Notebook hanya untuk eksplorasi dan visual analisis tambahan.
2. Output produksi tetap dihasilkan dari pipeline script.
3. Jika notebook menghasilkan insight permanen, ringkasannya dipindahkan ke dokumentasi utama.
