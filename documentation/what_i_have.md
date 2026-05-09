# Kondisi Infrastruktur yang Tersedia

Dokumen ini merangkum kondisi environment yang dimiliki untuk menjalankan project.

## 1. Akses Cluster

- SSH worker tersedia:
  - `fadhli@worker1`
  - `fadhli@worker2`
- Koneksi SSH kadang fluktuatif, jadi validasi koneksi sebelum run panjang sangat disarankan.

## 2. Komponen yang Sudah Terpasang

- Hadoop
- Spark
- Antar node sudah saling terhubung

## 3. Implikasi Operasional

1. Pipeline lokal (`without_worker`) bisa dijalankan walau worker sedang tidak stabil.
2. Pipeline `with_worker` membutuhkan koneksi cluster/HDFS/YARN yang sehat.
3. Untuk run Spark, cek dulu status node dan YARN agar job tidak berhenti di `ACCEPTED`.

## 4. Command Cek Cepat

```bash
bash scripts/start_cluster.sh
yarn node -list
hdfs dfs -ls -h /user/fadhli/amazon_books
```
