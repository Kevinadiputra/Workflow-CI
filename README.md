# 🚀 Model Training, Deployment CI & Serving/Monitoring - Heart Disease Classification

[![Model Training and Deployment CI](https://github.com/Kevinadiputra/Workflow-CI/actions/workflows/ci-training.yml/badge.svg)](https://github.com/Kevinadiputra/Workflow-CI/actions/workflows/ci-training.yml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.90+-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.0+-blue.svg?style=flat&logo=mlflow)](https://mlflow.org/)
[![Docker Hub](https://img.shields.io/badge/Docker_Hub-Serving_Image-blue.svg?style=flat&logo=docker)](https://hub.docker.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-orange.svg?style=flat&logo=prometheus)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboard-orange.svg?style=flat&logo=grafana)](https://grafana.com/)

Repositori ini merupakan pusat integrasi berkelanjutan (**Workflow-CI**) dan penyajian model (**Model Serving & Monitoring**) untuk tugas akhir Dicoding: **Membangun Sistem Machine Learning (MSML)**. Repositori ini melacak alur pelatihan model otomatis via MLflow, workflow integrasi berkelanjutan (GitHub Actions CI), instruksi kontainerisasi, model serving dengan FastAPI, dan visualisasi pemantauan real-time menggunakan Prometheus & Grafana.

---

## 🏋️ Pelatihan Model & MLflow Project

Proses pelatihan model dibungkus sebagai **MLflow Project** terstandarisasi untuk memastikan proses reproduksibilitas eksperimen.

* **Berkas Konfigurasi**:
  * [`MLProject/MLproject`](file:///MLProject/MLproject): Mendefinisikan parameter CLI (`n_estimators`, `max_depth`) dan perintah eksekusi pelatihan.
  * [`MLProject/conda.yaml`](file:///MLProject/conda.yaml): Berisi spesifikasi lingkungan terisolasi Conda dan dependensi pustaka Python.
* **Pelatihan Model & GridSearchCV**:
  * [`MLProject/modelling.py`](file:///MLProject/modelling.py): Melatih RandomForestClassifier dengan autologging parameter dan metrik performa secara otomatis.
  * Proses hyperparameter tuning GridSearchCV diintegrasikan secara remote ke **DagsHub MLflow Registry** untuk mencatat parameter terbaik, metrik performa (Accuracy, Precision, Recall, F1-Score), serta minimal 5 artefak visualisasi evaluasi model secara remote.

---

## ⛓️ Alur Integrasi Berkelanjutan (Workflow CI)

Alur CI pada GitHub Actions dikonfigurasi untuk mengeksekusi pipeline dari pelatihan hingga kontainerisasi secara otomatis saat terdapat push kode baru.

* **Lokasi Konfigurasi**: [`.github/workflows/ci-training.yml`](file:///.github/workflows/ci-training.yml)
* **Langkah-langkah Eksekusi CI**:
  1. **Checkout & Env Setup**: Melakukan checkout kode sumber dan mengonfigurasi Miniconda environment terisolasi.
  2. **MLflow Execution**: Menjalankan proyek MLflow secara otomatis menggunakan perintah `mlflow run`.
  3. **Artifact Logging**: Mengarsip riwayat runs MLflow, mengunggah sebagai build artifact GitHub (retensi 30 hari), dan menyimpan model latih ke direktori `saved_artifacts/` langsung di repositori ini.
  4. **Docker Login**: Melakukan otentikasi login ke akun Docker Hub pengguna.
  5. **Docker Build via MLflow**: Membangun Docker Image serving menggunakan fungsi resmi **`mlflow models build-docker`** (sesuai Kriteria 3 Advance).
  6. **Docker Push**: Mengunggah Docker Image hasil build ke repositori Docker Hub dengan tag `:latest`.

---

## 🔌 Serving & Pemantauan (FastAPI + Prometheus)

Penyajian model dikembangkan menggunakan **FastAPI** yang menggabungkan endpoint prediksi dan pemantauan sistem secara real-time.

* **Model Serving (`prometheus_exporter.py`)**:
  * **Endpoint `/predict` (POST)**: Menerima payload JSON 22 fitur klinis pasien. Script telah ditambahkan fitur **Dynamic Feature Reordering** yang mengurutkan kolom data request secara otomatis sesuai urutan fit pada model Random Forest guna menghindari *inconsistent feature names order* pada pandas/scikit-learn.
  * **Dynamic Model Accuracy**: Mengevaluasi model secara dinamis saat startup dengan menghitung `accuracy_score` asli dari dataset uji `test.csv` (menghapus semua hardcoded akurasi).
* **Prometheus Metrics (`/metrics` - GET)**:
  Menyediakan minimal **10 metrik sistem dan performa model** secara real-time:
  1. `prediction_count_total` (Counter): Total prediksi model.
  2. `prediction_success_total` (Counter): Jumlah prediksi sukses.
  3. `prediction_failed_total` (Counter): Jumlah prediksi gagal.
  4. `prediction_latency_seconds` (Histogram): Latensi waktu proses inferensi model.
  5. `request_count_total` (Counter): Total permintaan HTTP masuk ke API.
  6. `error_count_total` (Counter): Total error sistem yang terjadi.
  7. `cpu_usage_percent` (Gauge): Penggunaan CPU container.
  8. `memory_usage_percent` (Gauge): Penggunaan memori ram container.
  9. `disk_usage_percent` (Gauge): Kapasitas ruang penyimpanan disk container.
  10. `model_accuracy_ratio` (Gauge): Akurasi model riil berdasarkan data uji.
  11. `api_throughput_requests_per_second` (Gauge): Throughput API serving.
  12. `api_response_time_seconds` (Summary): Total response time keseluruhan API.

---

## 📈 Konfigurasi Pemantauan & Peringatan (Alerting)

### 1. Prometheus Scraping Target
Server Prometheus dikonfigurasi melalui [`prometheus/prometheus.yml`](file:///prometheus/prometheus.yml) untuk menarik data dari target model serving setiap 5 detik dengan status target **UP**.

### 2. Grafana Dashboard
Dashboard visualisasi eksternal dikonfigurasi melalui [`grafana_dashboard.json`](file:///grafana_dashboard.json) untuk menampilkan metrik server secara dinamis dan real-time setelah pengujian inferensi.

### 3. Aturan Peringatan Grafana (6 Alerting Rules)
Grafana dikonfigurasi dengan 6 Alerting Rules aktif untuk mendeteksi anomali pada sistem serving:
1. **High Latency Alert**: Respon waktu rata-rata API > 2 detik (`api_response_time_seconds_sum / api_response_time_seconds_count`).
2. **High CPU Alert**: Utilisasi CPU container melebihi 80% (`cpu_usage_percent`).
3. **High Error Rate Alert**: Tingkat persentase error request melebihi 5% (`(rate(error_count_total[1m]) / rate(request_count_total[1m])) * 100`).
4. **High Memory Alert**: Utilisasi RAM container melebihi 85% (`memory_usage_percent`).
5. **High Disk Usage Alert**: Kapasitas disk container hampir penuh melebihi 90% (`disk_usage_percent`).
6. **API Error Spike Alert**: Terjadi lonjakan error absolut > 5 kali dalam periode 1 menit (`increase(error_count_total[1m])`).

---

## 🛠️ Cara Menjalankan secara Lokal

1. **Jalankan Model Serving (FastAPI)**:
   ```bash
   conda run -n ml-1 uvicorn prometheus_exporter:app --host 127.0.0.1 --port 8000
   ```
   *Akses endpoint dokumentasi interaktif di [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).*

2. **Jalankan Simulasi Request Inferensi**:
   Buka terminal baru dan jalankan script client:
   ```bash
   conda run -n ml-1 python inference.py
   ```

3. **Jalankan Prometheus Server**:
   ```bash
   prometheus.exe --config.file=prometheus/prometheus.yml
   ```
   *Akses dasbor Prometheus di [http://localhost:9090](http://localhost:9090) untuk memantau status target.*

4. **Jalankan Grafana**:
   * Akses Grafana di [http://localhost:3000](http://localhost:3000) (admin / admin).
   * Tambahkan data source **Prometheus** ke `http://localhost:9090`.
   * Impor berkas **`grafana_dashboard.json`** untuk langsung melihat visualisasi metrik secara live.
