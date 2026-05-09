# Struktur Kode dan Fungsi Project (Kondisi Aktual)

Dokumen ini adalah peta teknis lengkap untuk seluruh kode aktif project, diperbarui sesuai kondisi repository saat ini.

Tanggal pembaruan: 2026-05-09.

## 1. Ringkasan Arsitektur Kode

Project berjalan dengan pola berikut:

1. `machine_learning/main.py` menjadi entrypoint CLI untuk semua step.
2. Modul `machine_learning/src/*` menjalankan data loading, preprocessing, training, evaluasi, inference, dan runtime orchestration.
3. `streamlit/app.py` menjadi antarmuka web untuk menjalankan pipeline, cluster ops, laporan, dan inference.
4. `scripts/*.sh` menangani operasi cluster/HDFS/YARN/Spark submit/reset.

Arsitektur training mode:
- `without_worker`: semua tahap preprocess + training berjalan di master.
- `with_worker`: preprocess awal dijalankan distributed via Spark/YARN ke HDFS, lalu hasil dijembatani ke master untuk training lokal.

## 2. Struktur Folder Kode yang Didokumentasikan

```text
machine_learning/
  main.py
  config.yaml
  src/
    __init__.py
    data_loader.py
    eda.py
    evaluate.py
    inference.py
    mlflow_tracker.py
    preprocessing.py
    spark_preprocess.py
    train_recommender.py
    train_sentiment.py
    train_sentiment_transformer.py
    training_runtime.py
    utils.py
streamlit/
  app.py
scripts/
  start_cluster.sh
  stop_cluster.sh
  submit_training.sh
  spark_submit_training.sh
  upload_to_hdfs.sh
  fix_hdfs_permissions.sh
  fix_yarn_master_network.sh
  fix_yarn_worker_network.sh
  reset_training_state.sh
config/
  cluster.yaml
  paths.yaml
```

## 3. Alur Eksekusi Utama

### 3.1 CLI Pipeline (`machine_learning/main.py`)

1. Parse argumen (`--step`, `--training-mode`, `--preprocess-source`, dst).
2. Load `config.yaml`.
3. Validasi guard `--allow-training`.
4. Dispatch ke modul sesuai step.
5. Untuk mode pipeline/compare, jalankan orchestrator `training_runtime.py`.

### 3.2 Training Mode Compare

Untuk `--step compare_training_modes`:
1. Jalankan `without_worker`.
2. Jalankan `with_worker`.
3. Jika `with_worker` gagal saat `run_worker_preprocess=True`, otomatis retry tanpa auto submit worker preprocess.
4. Simpan hasil komparasi ke `reports/training_mode_comparison.json`.

### 3.3 Path Data `with_worker`

1. Spark preprocess tulis Parquet ke HDFS (`.../output/.../processed`).
2. `data_loader.load_reviews_from_hdfs_spark_output` bridge HDFS ke local cache.
3. `preprocessing.preprocess_and_split` membentuk CSV lokal (`processed_reviews.csv`, `train/validation/test.csv`).
4. Training sentiment + recommender tetap membaca CSV lokal tersebut.

## 4. Dokumentasi Per File Kode

## 4.1 `machine_learning/main.py`

Peran: entrypoint CLI pipeline.

Fungsi:
- `parse_args` (line 25): mendefinisikan semua argumen CLI (step, source, mode, guard training, RAM limit).
- `main` (line 90): dispatcher utama step pipeline dan enforcement guard training.

Output/efek:
- Menjalankan step EDA, preprocess, training model, compare mode, evaluasi.
- Mengatur mode aman agar training tidak jalan tanpa `--allow-training`.

## 4.2 `machine_learning/src/__init__.py`

Peran: marker package Python (`src`).

Isi:
- Docstring package, tidak ada fungsi operasional.

## 4.3 `machine_learning/src/utils.py`

Peran: utility umum path, config, logging, JSON, registry model.

Fungsi:
- `project_root` (line 12): root folder `machine_learning/`.
- `load_config` (line 16): load YAML config.
- `resolve_path` (line 22): resolve key path relatif dari config menjadi path absolut berbasis project.
- `ensure_dir` (line 27): pastikan directory ada.
- `setup_logging` (line 31): setup logging ke console + file `logs/pipeline_*.log`.
- `save_json` (line 48): simpan JSON terformat.
- `load_json` (line 54): baca JSON (return `{}` jika file tidak ada).
- `append_model_registry` (line 61): append record model ke `models/model_registry.json`.

## 4.4 `machine_learning/src/data_loader.py`

Peran: load data dari local CSV dan hasil Spark di HDFS, plus bridge/cache HDFS lokal.

Fungsi internal HDFS utility:
- `_build_hdfs_uri` (49): gabung `namenode_uri` + path HDFS.
- `_format_size` (55): format byte human readable.
- `_format_duration` (64): format detik ke teks.
- `_sum_local_size` (75): hitung ukuran file/folder lokal rekursif.
- `_build_hdfs_subprocess_kwargs` (94): siapkan kwargs subprocess + relax RLIMIT child process.
- `_run_hdfs_command` (118): wrapper command HDFS CLI.
- `_parse_hdfs_du_size_bytes` (123): parse output `hdfs dfs -du -s`.
- `_resolve_hdfs_content_size_bytes` (136): baca ukuran konten HDFS.
- `_resolve_hdfs_modification_time` (143): baca mtime path HDFS.
- `_build_hdfs_signature` (154): signature cache dari size + mtime HDFS.
- `_build_hdfs_cache_key` (160): hash key cache per URI.
- `_read_json_file` (165): baca JSON helper.
- `_write_json_file` (175): tulis JSON helper.
- `_prepare_persistent_cache_paths` (183): siapkan lokasi cache persisten `/tmp/spark_hdfs_bridge_cache`.
- `_download_hdfs_dir_with_progress` (189): download folder HDFS dengan progress + warning slow bridge.
- `_cleanup_hdfs_cache` (254): cleanup cache temporary saat exit.
- `_materialize_hdfs_parquet_dir` (262): resolve cache reusable atau download ulang hasil HDFS.
- `_dataset_to_pandas_streaming` (339): convert PyArrow dataset ke pandas bertahap (batch kecil).
- `_normalize_reviews_df` (373): standarisasi kolom review + validasi minimum.

Fungsi public data source:
- `load_reviews` (407): load dataset review lokal CSV.
- `load_reviews_from_hdfs_spark_output` (425): load hasil Spark preprocess dari HDFS (Parquet) via bridge.
- `load_books` (493): load metadata buku lokal.

Catatan penting:
- Bridge HDFS memakai cache persisten untuk run berulang.
- Pembacaan Spark output dilakukan streaming agar lebih stabil pada batas RAM ketat.

## 4.5 `machine_learning/src/preprocessing.py`

Peran: preprocessing teks, sentiment labeling, dan split train/validation/test.

Fungsi:
- `_ensure_nltk_assets` (27): download stopwords/wordnet jika diperlukan.
- `rating_to_sentiment_label` (33): mapping rating -> label sentimen (0/1/2).
- `clean_text` (41): cleaning teks (lowercase, remove URL/HTML/non-alpha/spaces).
- `remove_stopwords` (51): hapus stopwords.
- `lemmatize_text` (56): lemmatization token.
- `preprocess_text` (61): pipeline full text preprocessing.
- `_load_preprocess_source` (68): pilih source `local_dataset` atau `spark_hdfs`.
- `preprocess_and_split` (77): preprocess end-to-end, simpan `processed_reviews.csv`, split train/val/test, simpan metadata preprocess.

Output utama:
- `data/processed/processed_reviews.csv`
- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `data/processed/preprocess_metadata.json`

## 4.6 `machine_learning/src/eda.py`

Peran: exploratory data analysis dan artifact EDA.

Fungsi:
- `_save_series` (14): helper simpan series ke CSV berformat.
- `run_eda` (19): jalankan EDA lengkap, statistik dasar, missing/duplicate, distribusi rating, distribusi panjang review, contoh review, insight text.

Output utama di `reports/eda/`:
- `basic_stats.csv`
- `missing_values.csv`
- `rating_distribution.csv`
- `review_length_summary.csv`
- `reviews_per_user_summary.csv`
- `reviews_per_item_summary.csv`
- `rating_distribution.png`
- `review_length_distribution.png`
- `sentiment_examples.csv`
- `eda_summary.txt`

## 4.7 `machine_learning/src/train_sentiment.py`

Peran: training baseline sentiment (TF-IDF + model klasik).

Fungsi:
- `_resolve_dtype` (37): validasi dtype TF-IDF.
- `_build_tfidf_profiles` (46): profil TF-IDF utama + fallback memory-safe.
- `_read_split_csv` (112): baca split CSV dengan fallback engine `python` saat parser/memory issue.
- `_load_splits` (130): load train/val/test split.
- `_metrics` (137): hitung accuracy/precision/recall/F1/report.
- `_plot_confusion` (148): simpan confusion matrix image.
- `train_baseline_sentiment` (157): train 3 model baseline (LogReg, NB, LinearSVC), evaluasi, simpan model/artifact, logging MLflow, update registry.

Output utama:
- `models/sentiment/baseline/*.pkl`
- `models/sentiment/baseline/*_confusion_matrix_*.png`
- `models/sentiment/baseline/*_classification_report.txt`
- `models/sentiment/baseline/metrics.json`

## 4.8 `machine_learning/src/train_sentiment_transformer.py`

Peran: fine-tuning model transformer untuk klasifikasi sentimen 3 kelas.

Class:
- `TransformerDataset` (29): wrapper dataset tokenized untuk Trainer.

Fungsi:
- `_load_splits` (47): load split CSV.
- `_build_metrics` (54): metrik klasifikasi.
- `_plot_confusion_matrix` (72): simpan confusion matrix test.
- `train_transformer_sentiment` (81): fine-tuning HF model, early stopping, save model/tokenizer/checkpoints/metrics, update registry, logging MLflow.

Catatan kompatibilitas:
- Menangani perbedaan argumen `Trainer` (`processing_class` vs `tokenizer`) antar versi transformers.

Output utama:
- `models/sentiment/transformer/distilbert_v1/model/`
- `models/sentiment/transformer/distilbert_v1/tokenizer/`
- `models/sentiment/transformer/distilbert_v1/checkpoints/`
- `models/sentiment/transformer/distilbert_v1/metrics.json`
- `models/sentiment/transformer/distilbert_v1/confusion_matrix_test.png`

## 4.9 `machine_learning/src/train_recommender.py`

Peran: training recommender (popularity + collaborative + content + hybrid) dan evaluasi ranking.

Class:
- `CollabModel` (25): container faktor SVD + indeks user/item + rentang rating.

Fungsi metrik ranking:
- `precision_at_k` (36)
- `recall_at_k` (46)
- `ndcg_at_k` (54)

Fungsi pembentuk model:
- `_iterative_filter` (71): filter interaksi minimum user/item secara iteratif.
- `_build_popularity_scores` (83): hitung skor popularitas item.
- `_build_collaborative_model` (102): bangun SVD collaborative filtering.
- `_predict_collab_raw` (143): prediksi rating raw user-item.
- `_normalize_rating` (154): normalisasi skor ke skala 0-1.
- `_build_content_model` (161): TF-IDF konten per item.
- `_content_user_profile_scores` (184): skor kemiripan profile user terhadap kandidat item.
- `_recommend_for_user` (215): scoring hybrid dan ranking kandidat per user.
- `_diversity_at_k` (245): metrik diversity rekomendasi.
- `_load_processed_interactions` (268): load interactions dengan strategi chunked + fallback engine untuk stabilitas memori.
- `train_recommenders` (321): orchestrate seluruh training/evaluasi/rekomendasi/artifact/registry/MLflow.

Output utama:
- `models/recommender/popularity/top_books_popularity.csv`
- `models/recommender/collaborative_filtering/factors.npz`
- `models/recommender/content_based/item_text_tfidf.npz`
- `models/recommender/hybrid/hybrid_recommendations.csv`
- `reports/recommender_metrics.json`

## 4.10 `machine_learning/src/mlflow_tracker.py`

Peran: wrapper MLflow opsional agar pipeline tetap jalan walau MLflow tidak aktif.

Fungsi:
- `_get_mlflow_module` (10): lazy import MLflow.
- `is_enabled` (19): cek flag `mlflow.enabled`.
- `start_run` (24): mulai run pada experiment lokal `mlruns/`.
- `log_params` (38): log parameter run.
- `log_metrics` (47): log metrik numerik.
- `log_artifact` (68): log satu artifact file.
- `log_artifacts` (83): log satu direktori artifact.

## 4.11 `machine_learning/src/inference.py`

Peran: inference sentiment dan inference rekomendasi dari artifact yang sudah dilatih.

Fungsi:
- `predict_sentiment` (9): load best baseline model dari metrics, prediksi sentimen + confidence + probabilitas.
- `recommend_for_user` (65): ambil top-N rekomendasi user dari file hybrid recommendation.

## 4.12 `machine_learning/src/evaluate.py`

Peran: kompilasi laporan akhir lintas komponen model.

Fungsi:
- `compile_final_report` (10): gabungkan metrics baseline, transformer, recommender, comparison mode, dan model registry ke `reports/final_report.json`.

## 4.13 `machine_learning/src/spark_preprocess.py`

Peran: distributed preprocessing di Spark/YARN.

Fungsi:
- `_build_hdfs_uri` (33): builder URI HDFS.
- `_load_project_config` (39): load `config.yaml`.
- `_resolve_hdfs_io_paths` (48): resolve input/output HDFS dari env/config.
- `_coerce_bool` (64): parser bool aman.
- `_coerce_float` (77): parser float aman.
- `_coerce_int` (84): parser int aman.
- `_resolve_preprocess_options` (91): gabung opsi preprocess dari env + config.
- `main` (130): Spark job: baca CSV HDFS, cleaning dasar, labeling sentimen, sampling/row cap optional, simpan Parquet ke HDFS.

Output utama:
- `hdfs://<namenode>/<output_hdfs_path>/processed`

## 4.14 `machine_learning/src/training_runtime.py`

Peran: orchestrator runtime training pipeline dan perbandingan mode.

Fungsi mode dan memory guard:
- `resolve_mode_preprocess_source` (36): mapping mode ke source preprocess.
- `get_master_ram_limit_gb` (45): resolve RAM limit.
- `_format_duration` (51): formatter durasi.
- `_load_stage_duration_hints` (62): ETA hint dari run sebelumnya.
- `_peak_memory_mb` (82): peak RSS proses.
- `_current_virtual_memory_bytes` (97): VmSize proses saat ini dari `/proc`.
- `_get_process_rlimit_as` (116): baca RLIMIT_AS.
- `_restore_process_rlimit_as` (128): restore RLIMIT_AS selesai run.
- `_build_relaxed_rlimit_preexec` (150): callback child process agar limit address space lebih longgar.
- `apply_master_ram_limit` (168): menerapkan batas RAM master aman (dengan safety margin).

Fungsi worker preprocess:
- `_run_worker_preprocess_submit` (226): jalankan `scripts/spark_submit_training.sh preprocess_spark` dengan monitoring output/timeout.
- `run_worker_preprocess_submit` (349): wrapper public.

Fungsi orchestration stage:
- `_summarize_sentiment_metrics` (354): ringkasan metrik baseline untuk report pipeline.
- `_capture_stage` (372): capture durasi + peak memory per stage.
- `run_training_pipeline` (395): jalankan preprocess -> sentiment -> optional transformer -> recommender -> evaluate dengan progress logging dan summary run.
- `_build_comparison_summary` (572): hitung delta metrik `with_worker - without_worker`.
- `compare_training_modes` (600): jalankan dua mode, retry fallback untuk with_worker, simpan report komparasi.

Output utama:
- `reports/experiments/without_worker_latest_run.json`
- `reports/experiments/with_worker_latest_run.json`
- `reports/training_mode_comparison.json`

## 4.15 `streamlit/app.py`

Peran: dashboard web untuk observasi, eksekusi pipeline, monitoring report, inference, dan operasi cluster.

Fungsi helper runtime:
- `_python_bin` (32): resolve python binary prefer `.venv/bin/python`.
- `_default_hdfs_dataset_path` (37): default HDFS dataset user aktif.
- `_default_hdfs_output_path` (41): default HDFS output user aktif.
- `_load_config` (45): load `machine_learning/config.yaml`.
- `_load_cluster_cfg` (54): load `config/cluster.yaml`.
- `_load_json` (76): safe JSON loader.
- `_load_recommender_user_ids` (85): load daftar user untuk inference from artifact.
- `_format_cmd` (98): pretty command formatter.
- `_run` (102): non-live command executor.
- `_enqueue_stream_output` (133): helper threading stream.
- `_run_live` (142): live subprocess runner dengan streaming stdout/stderr ke UI.
- `_show_result` (269): render hasil command ke UI.
- `_format_size` (289): formatter ukuran file.
- `_as_float` (298): safe cast float.
- `_get_nested` (305): akses nested dictionary aman.
- `_hdfs_basename` (314): basename path HDFS.
- `_prepare_hdfs_download` (322): siapkan payload download file/folder dari HDFS (folder di-zip).

Fungsi tab dashboard:
- `tab_overview` (413): status artifact + KPI ringkas + ringkasan komparasi.
- `tab_eda` (491): tampilkan hasil EDA.
- `tab_pipeline` (554): trigger pipeline steps dengan mode/source/timeout controls.
- `tab_reports` (701): tampilkan report baseline, transformer, recommender, komparasi mode detail, registry model.
- `tab_inference` (940): sentiment + recommender inference.
- `tab_cluster` (1000): operasi cluster, upload/download HDFS, spark submit, reset state, permission/network helper.
- `main` (1301): setup page dan mount semua tab.

## 4.16 `scripts/start_cluster.sh`

Peran: start cluster DFS + YARN dengan health check ringkas.

Fungsi shell:
- `run_if_exists` (6): jalankan command jika tersedia.
- `warn_loopback_hostname` (16): warning jika hostname resolve ke loopback.
- `count_expected_workers` (34): baca jumlah worker dari file Hadoop workers.
- `show_hdfs_health` (42): cek live datanodes.
- `show_yarn_health` (78): cek total NodeManager aktif dengan retry.

## 4.17 `scripts/stop_cluster.sh`

Peran: stop cluster YARN + DFS.

Fungsi shell:
- `run_if_exists` (4): jalankan stop command jika tersedia.

## 4.18 `scripts/spark_submit_training.sh`

Peran: wrapper submit Spark preprocess ke YARN + preflight validasi cluster.

Fungsi shell:
- `run_with_optional_timeout` (30): wrapper timeout command.
- `warn_loopback_hostname` (41): warning hostname loopback.
- `check_yarn_health` (63): preflight `yarn node -list` dengan retry.

Perilaku:
- Jika step `preprocess_spark` maka jalankan `spark-submit`.
- Jika step lain, fallback jalankan `machine_learning/main.py --allow-training` di master.

## 4.19 `scripts/submit_training.sh`

Peran: shortcut menjalankan `main.py --allow-training` untuk satu step.

Catatan:
- Tidak memiliki fungsi shell terpisah.
- Validasi argumen step wajib.

## 4.20 `scripts/upload_to_hdfs.sh`

Peran: upload semua file CSV dari `machine_learning/dataset/` ke HDFS target.

Catatan:
- Tidak memiliki fungsi shell terpisah.
- Menangani kasus path project mengandung spasi dengan `pushd` ke folder dataset.

## 4.21 `scripts/fix_hdfs_permissions.sh`

Peran: memperbaiki permission HDFS di worker agar user target bisa baca/tulis untuk lab/testing.

Fungsi shell (remote block):
- `run_hdfs` (79): jalankan command HDFS dengan optional `HADOOP_USER_NAME` (superuser override).

Catatan:
- Script membaca worker list dari `config/cluster.yaml`.
- Mode permission cukup permisif (`777`) untuk kemudahan environment eksperimen.

## 4.22 `scripts/fix_yarn_master_network.sh`

Peran: perbaiki konfigurasi hostname/IP dan YARN master binding.

Fungsi shell:
- `require_root` (14): wajib root.
- `backup_file` (22): backup file konfigurasi.
- `fix_hosts_file` (29): rewrite `/etc/hosts` block managed hosts.
- `fix_yarn_site` (77): update `yarn-site.xml` properti RM hostname/bind-host.
- `restart_yarn` (114): restart service YARN.
- `verify_cluster` (129): verifikasi hostname + `yarn node -list`.
- `main` (143): orkestrasi seluruh langkah.

## 4.23 `scripts/fix_yarn_worker_network.sh`

Peran: perbaiki `/etc/hosts` worker dan restart NodeManager.

Fungsi shell:
- `require_root` (12)
- `backup_file` (20)
- `fix_hosts_file` (27)
- `restart_nodemanager` (74)
- `verify_worker` (87)
- `main` (94)

## 4.24 `scripts/reset_training_state.sh`

Peran: reset artifact training lokal, dan opsional reset state Hadoop penuh (master + worker).

Fungsi shell:
- `usage` (12): bantuan argumen.
- `remove_dir_contents` (45): hapus isi folder aman.
- `reset_local_training_artifacts` (62): hapus artifact lokal training.
- `load_hadoop_reset_metadata` (93): baca metadata worker/dir dari config + XML Hadoop.
- `build_remote_cleanup_command` (220): bangun command cleanup remote.
- `cleanup_workers_hadoop_state` (238): cleanup state Hadoop di worker via SSH.
- `cleanup_master_hadoop_state` (258): cleanup state Hadoop master.
- `format_namenode` (276): format ulang NameNode.
- `reset_hadoop_cluster_state` (286): orkestrasi reset penuh cluster.

## 5. Dokumentasi Konfigurasi

## 5.1 `machine_learning/config.yaml`

Peran: konfigurasi pusat pipeline ML.

Key utama:
- `project`: metadata project dan random state.
- `paths`: semua lokasi input/output artifact.
- `data`: sample rows dan batas minimum text length.
- `preprocessing`: rasio split.
- `sentiment`: konfigurasi TF-IDF baseline.
- `transformer`: konfigurasi fine-tuning model HF.
- `recommender`: hiperparameter recommender.
- `mlflow`: enable/disable tracking.
- `training`: RAM limit master + default training options.
- `hadoop`: URI dan path HDFS.
- `spark`: resource dan mode submit.
- `spark_preprocess`: opsi sampling/row cap/logging preprocessing Spark.

## 5.2 `config/cluster.yaml`

Peran: konfigurasi cluster untuk dashboard/script ops.

Key utama:
- `cluster`: master, workers, ssh timeout.
- `hdfs`: namenode + path dataset/output + UI.
- `yarn`: UI host/port.
- `spark`: resource default submit + timeout preflight.

## 5.3 `config/paths.yaml`

Peran: metadata tambahan path local/HDFS/UI untuk kebutuhan operasional manual.

## 6. Mapping Artifact per Modul

- EDA: `src/eda.py` -> `reports/eda/*`
- Preprocess: `src/preprocessing.py` -> `data/processed/*`
- Baseline sentiment: `src/train_sentiment.py` -> `models/sentiment/baseline/*`
- Transformer: `src/train_sentiment_transformer.py` -> `models/sentiment/transformer/distilbert_v1/*`
- Recommender: `src/train_recommender.py` -> `models/recommender/*` + `reports/recommender_metrics.json`
- Runtime compare: `src/training_runtime.py` -> `reports/experiments/*` + `reports/training_mode_comparison.json`
- Final evaluator: `src/evaluate.py` -> `reports/final_report.json`

## 7. Kesimpulan Kondisi Kode Saat Ini

1. Semua file kode inti sudah modular dan memiliki boundary per tanggung jawab.
2. Mekanisme compare mode sudah punya fallback retry dan pelaporan error/warning per mode.
3. Stabilitas memory untuk pembacaan dataset besar sudah ditangani di baseline sentiment dan recommender loader dengan fallback strategi.
4. Dashboard Streamlit saat ini sudah menjadi control plane operasional untuk pipeline + cluster + report + inference.
