# Architecture Plan: Hadoop Cluster untuk Training Machine Learning dan Visualisasi Streamlit

## 1. Tujuan

Dokumen ini berisi rencana arsitektur untuk membangun cluster Hadoop yang dapat digunakan untuk:

- Menyimpan dataset besar menggunakan HDFS.
- Menjalankan training data menggunakan Python.
- Memanfaatkan worker node agar ikut membantu proses komputasi.
- Mengontrol proses training dari master node.
- Menampilkan visualisasi hasil training menggunakan Streamlit.

Dataset yang digunakan:

```text
Nama dataset : Amazon Books Reviews
Sumber       : Kaggle
URL          : https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews/data?select=Books_rating.csv
File utama   : Books_rating.csv
Ukuran       : ±3 GB
Bahasa ML    : Python
```

---

## 2. Spesifikasi Node

| Tipe | Nama Host | OS | Peran |
|---|---|---|---|
| Master | fadhli | Linux | NameNode, ResourceManager, Spark Driver, Streamlit |
| Worker | worker1 | Linux Server | DataNode, NodeManager, Spark Executor |
| Worker | worker2 | Linux Server | DataNode, NodeManager, Spark Executor |

---

## 3. Arsitektur Umum

```text
                         User / Admin
                              |
                              v
                    +------------------+
                    | Master: fadhli   |
                    | Linux            |
                    |------------------|
                    | NameNode         |
                    | ResourceManager  |
                    | Spark Driver     |
                    | Streamlit App    |
                    | History Server   |
                    +---------+--------+
                              |
              --------------------------------
              |                              |
              v                              v
+-------------------------+      +-------------------------+
| Worker: worker1         |      | Worker: worker2         |
| Linux Server            |      | Linux Server            |
|-------------------------|      |-------------------------|
| DataNode                |      | DataNode                |
| NodeManager             |      | NodeManager             |
| Spark Executor          |      | Spark Executor          |
+-------------------------+      +-------------------------+

Storage          : HDFS
Resource Manager : YARN
Training Engine  : Apache Spark / PySpark
Visualization    : Streamlit
```

---

## 4. Komponen Utama

### 4.1 Hadoop HDFS

HDFS digunakan sebagai sistem penyimpanan terdistribusi.

Peran HDFS:

- Menyimpan dataset `Books_rating.csv`.
- Membagi file menjadi beberapa block.
- Mendistribusikan block data ke worker node.
- Menyediakan akses data untuk Spark.

Service HDFS:

| Service | Lokasi |
|---|---|
| NameNode | master/fadhli |
| DataNode | worker1, worker2 |
| SecondaryNameNode | master/fadhli |

---

### 4.2 YARN

YARN digunakan sebagai resource manager cluster.

Peran YARN:

- Mengatur resource CPU dan RAM.
- Menjalankan aplikasi Spark.
- Menentukan worker mana yang menjalankan executor.
- Memantau status job training.

Service YARN:

| Service | Lokasi |
|---|---|
| ResourceManager | master/fadhli |
| NodeManager | worker1, worker2 |

---

### 4.3 Apache Spark

Spark digunakan sebagai engine utama untuk menjalankan proses training dan preprocessing data.

Peran Spark:

- Membaca dataset dari HDFS.
- Membagi proses komputasi ke worker.
- Menjalankan executor pada worker node.
- Mengirim hasil training ke HDFS atau local storage master.

Mode yang disarankan:

```text
Spark on YARN
```

Command umum:

```bash
spark-submit \
  --master yarn \
  --deploy-mode client \
  --num-executors 2 \
  --executor-cores 2 \
  --executor-memory 2G \
  --driver-memory 2G \
  train.py
```

---

### 4.4 Streamlit

Streamlit digunakan sebagai dashboard di master.

Peran Streamlit:

- Menjalankan tombol trigger training.
- Menampilkan status atau log training.
- Menampilkan hasil metric, grafik, atau output model.
- Membaca hasil dari file output training.

Lokasi:

```text
Master node: fadhli
```

Port default:

```text
8501
```

Akses:

```text
http://fadhli:8501
```

---

## 5. Alur Kerja Sistem

```text
1. Dataset di-download dari Kaggle ke master.
2. Dataset di-upload ke HDFS.
3. User membuka dashboard Streamlit di master.
4. User menekan tombol mulai training.
5. Streamlit menjalankan spark-submit.
6. Spark Driver berjalan di master.
7. YARN mengalokasikan resource ke worker1 dan worker2.
8. Spark Executor berjalan di worker.
9. Worker membaca block data dari HDFS.
10. Worker membantu proses preprocessing/training.
11. Hasil training disimpan ke HDFS atau local master.
12. Streamlit membaca hasil training.
13. User melihat visualisasi hasil training.
```

---

## 6. Arsitektur Data

```text
Local Master
/home/fadhli/MY PROJECT/Hadoop_Amazon_Books_Reviews/machine_learning/dataset/Books_rating.csv
        |
        | hdfs dfs -put
        v
HDFS
/user/fadhli/amazon_books/Books_rating.csv
        |
        | spark.read.csv()
        v
Spark DataFrame
        |
        | preprocessing / training
        v
Output
/user/fadhli/output/amazon_books_ml/processed
/home/fadhli/MY PROJECT/Hadoop_Amazon_Books_Reviews/output/metrics.json
/home/fadhli/MY PROJECT/Hadoop_Amazon_Books_Reviews/machine_learning/models/
```

---

## 7. Struktur Direktori Project

Struktur direktori yang disarankan di master:

```text
/home/fadhli/MY PROJECT/Hadoop_Amazon_Books_Reviews/
├── machine_learning/
│   ├── config.yaml
│   ├── dataset/
│   │   └── Books_rating.csv
│   └── src/
│       └── spark_preprocess.py
├── scripts/
│   ├── upload_to_hdfs.sh
│   ├── start_cluster.sh
│   ├── stop_cluster.sh
│   ├── submit_training.sh
│   └── spark_submit_training.sh
├── streamlit/
│   └── app.py
├── output/
│   └── metrics.json
├── config/
│   └── cluster.yaml
└── README.md
```

---

## 8. Kebutuhan Software

Install pada semua node:

| Software | Fungsi |
|---|---|
| Java JDK | Runtime Hadoop dan Spark |
| Apache Hadoop | HDFS dan YARN |
| Apache Spark | Engine distributed processing |
| Python 3 | Bahasa machine learning |
| PySpark | Integrasi Python dengan Spark |
| SSH | Komunikasi antar node |
| rsync/scp | Sinkronisasi file konfigurasi |

Install hanya di master:

| Software | Fungsi |
|---|---|
| Streamlit | Dashboard visualisasi |
| Kaggle CLI | Download dataset |
| Python library ML | Dependensi kode training |

---

## 9. Rekomendasi Hardware Minimum

| Node | CPU | RAM | Disk |
|---|---:|---:|---:|
| fadhli | 4 core | 8–16 GB | 40 GB+ |
| worker1 | 4 core | 8–16 GB | 40 GB (usable ~38 GB) |
| worker2 | 4 core | 8–16 GB | 40 GB (usable ~38 GB) |

Catatan:

- Dataset 3 GB masih relatif kecil untuk Hadoop.
- Hadoop tetap cocok untuk latihan arsitektur distributed.
- Untuk performa lebih baik, gunakan SSD.
- Pastikan jaringan antar node stabil.

---

## 10. Kebutuhan Jaringan

Semua node harus bisa saling berkomunikasi menggunakan hostname.

Contoh konfigurasi `/etc/hosts`:

```text
192.168.0.102 fadhli
192.168.0.103 worker1
192.168.0.105 worker2
```

Port penting:

| Service | Port |
|---|---:|
| HDFS NameNode UI | 9870 |
| YARN ResourceManager UI | 8088 |
| Streamlit | 8501 |
| HDFS RPC | 9000 |

---

## 11. Desain Resource Training

Untuk 2 worker, konfigurasi awal Spark yang disarankan:

```text
num-executors     : 2
executor-cores    : 2
executor-memory   : 2G
driver-memory     : 2G
deploy-mode       : client
master            : yarn
```

Contoh:

```bash
spark-submit \
  --master yarn \
  --deploy-mode client \
  --num-executors 2 \
  --executor-cores 2 \
  --executor-memory 2G \
  --driver-memory 2G \
  /home/fadhli/MY PROJECT/Hadoop_Amazon_Books_Reviews/machine_learning/src/spark_preprocess.py
```

---

## 12. Catatan Penting Tentang Machine Learning

Agar worker benar-benar membantu training, kode Python harus mendukung distributed computing.

### Opsi terbaik

```text
PySpark MLlib
```

Dengan PySpark MLlib:

- Data dibaca dari HDFS.
- Transformasi data berjalan secara distributed.
- Training model tertentu bisa berjalan di worker.
- Resource dikontrol oleh YARN.

### Jika kode masih memakai pandas/scikit-learn

Worker tidak otomatis membantu training penuh.

Kemungkinan pola yang bisa digunakan:

```text
Spark untuk preprocessing distributed
pandas/scikit-learn untuk training akhir di master
```

Atau kode perlu dimigrasikan ke PySpark.

---

## 13. Rencana Konfigurasi

Tahapan konfigurasi:

```text
1. Setup hostname dan IP antar node.
2. Setup user hadoop di semua node.
3. Setup passwordless SSH dari master ke worker.
4. Install Java di semua node.
5. Install Hadoop di semua node.
6. Konfigurasi core-site.xml.
7. Konfigurasi hdfs-site.xml.
8. Konfigurasi yarn-site.xml.
9. Konfigurasi mapred-site.xml.
10. Konfigurasi workers.
11. Format NameNode.
12. Start HDFS.
13. Start YARN.
14. Upload dataset ke HDFS.
15. Install Spark di semua node.
16. Test spark-submit di YARN.
17. Integrasi spark-submit dengan Streamlit.
18. Validasi worker ikut memproses job.
```

---

## 14. File Konfigurasi Utama

### 14.1 core-site.xml

```xml
<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://fadhli:9000</value>
    </property>
</configuration>
```

### 14.2 hdfs-site.xml

```xml
<configuration>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>file:///data/hadoop/hdfs/namenode</value>
    </property>

    <property>
        <name>dfs.datanode.data.dir</name>
        <value>file:///data/hadoop/hdfs/datanode</value>
    </property>

    <property>
        <name>dfs.replication</name>
        <value>2</value>
    </property>

    <property>
        <name>dfs.namenode.rpc-bind-host</name>
        <value>0.0.0.0</value>
    </property>

    <property>
        <name>dfs.namenode.http-bind-host</name>
        <value>0.0.0.0</value>
    </property>

    <property>
        <name>dfs.namenode.datanode.registration.ip-hostname-check</name>
        <value>false</value>
    </property>
</configuration>
```

### 14.3 yarn-site.xml

```xml
<configuration>
    <property>
        <name>yarn.resourcemanager.hostname</name>
        <value>fadhli</value>
    </property>

    <property>
        <name>yarn.nodemanager.aux-services</name>
        <value>mapreduce_shuffle</value>
    </property>

    <property>
        <name>yarn.nodemanager.resource.memory-mb</name>
        <value>8192</value>
    </property>

    <property>
        <name>yarn.scheduler.maximum-allocation-mb</name>
        <value>8192</value>
    </property>

    <property>
        <name>yarn.nodemanager.resource.cpu-vcores</name>
        <value>4</value>
    </property>
</configuration>
```

### 14.4 mapred-site.xml

```xml
<configuration>
    <property>
        <name>mapreduce.framework.name</name>
        <value>yarn</value>
    </property>
</configuration>
```

### 14.5 workers

```text
worker1
worker2
```

---

## 15. Validasi Keberhasilan

### 15.1 Cek HDFS

```bash
hdfs dfsadmin -report
```

Indikator berhasil:

```text
Live datanodes: 2
```

### 15.2 Cek YARN

```bash
yarn node -list
```

Indikator berhasil:

```text
Total Nodes: 2
```

### 15.3 Cek Web UI

```text
HDFS UI : http://fadhli:9870
YARN UI : http://fadhli:8088
```

### 15.4 Cek Dataset

```bash
hdfs dfs -ls -h /user/fadhli/amazon_books
```

### 15.5 Cek Spark Job

Jalankan:

```bash
spark-submit \
  --master yarn \
  --deploy-mode client \
  --num-executors 2 \
  --executor-cores 2 \
  --executor-memory 2G \
  train.py
```

Lalu cek YARN UI:

```text
http://fadhli:8088
```

---

## 16. Output yang Diharapkan

Setelah sistem selesai dibangun, output akhir yang diharapkan:

```text
1. Dataset tersimpan di HDFS.
2. worker1 dan worker2 aktif sebagai DataNode.
3. worker1 dan worker2 aktif sebagai NodeManager.
4. Spark job dapat dijalankan dari master.
5. Training dapat dikontrol dari master.
6. Worker membantu proses komputasi.
7. Hasil training tersimpan di output directory.
8. Streamlit dapat menampilkan hasil training.
```

---

## 17. Risiko dan Hal yang Perlu Diperhatikan

| Risiko | Penjelasan | Solusi |
|---|---|---|
| Worker tidak ikut training | Kode masih pandas/sklearn biasa | Gunakan PySpark atau distributed ML |
| SSH gagal | Passwordless SSH belum benar | Ulangi ssh-copy-id |
| DataNode tidak muncul | Hostname/IP salah | Cek `/etc/hosts` |
| Spark gagal submit | Hadoop config tidak terbaca | Set `HADOOP_CONF_DIR` |
| Memory error | Executor memory terlalu kecil | Naikkan executor-memory |
| Dataset lambat dibaca | Disk/jaringan lambat | Gunakan SSD dan jaringan stabil |

---

## 18. Kesimpulan Arsitektur

Arsitektur yang direkomendasikan adalah:

```text
Hadoop HDFS + YARN + Apache Spark + PySpark + Streamlit
```

Dengan pembagian:

```text
fadhli  : master controller
worker1 : compute dan storage worker
worker2 : compute dan storage worker
```

Arsitektur ini sesuai untuk kebutuhan:

- Distributed storage.
- Distributed training/preprocessing.
- Kontrol training dari master.
- Visualisasi hasil dengan Streamlit.
- Integrasi dengan kode Python yang sudah dimiliki.
