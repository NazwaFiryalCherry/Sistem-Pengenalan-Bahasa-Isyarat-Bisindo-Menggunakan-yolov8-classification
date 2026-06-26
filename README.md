# Sistem Pengenalan Bahasa Isyarat Indonesia (BISINDO) Menggunakan YOLOv8 Classification

## 📖 Deskripsi

Proyek ini merupakan sistem pengenalan **Bahasa Isyarat Indonesia (BISINDO)** menggunakan algoritma **YOLOv8 Classification** berbasis *Deep Learning*. Sistem dikembangkan untuk mengenali gesture BISINDO secara otomatis melalui citra yang diperoleh dari webcam.

Model **YOLOv8 Classification** digunakan untuk mengklasifikasikan citra gesture ke dalam beberapa kategori, yaitu **huruf, angka, dan kata** dalam Bahasa Isyarat Indonesia. Sebelum proses pelatihan, dataset melalui tahapan **preprocessing** dan **augmentasi data** untuk meningkatkan kualitas serta variasi data latih sehingga model mampu melakukan klasifikasi dengan lebih baik.

Proyek ini dikembangkan sebagai implementasi teknologi **Computer Vision** dan **Deep Learning** dalam membangun sistem pengenalan bahasa isyarat Indonesia yang dapat membantu proses komunikasi antara penyandang tunarungu dan masyarakat umum.

---

## ✨ Fitur

- Pengenalan gesture BISINDO menggunakan **YOLOv8 Classification**.
- Klasifikasi gesture huruf, angka, dan kata.
- Pengambilan dataset menggunakan webcam.
- Pelatihan model secara terpisah untuk setiap kategori dataset.
- Prediksi gesture menggunakan model hasil pelatihan.
- Mendukung klasifikasi citra secara **real-time** melalui webcam.

---

## 🛠️ Teknologi yang Digunakan

### Bahasa Pemrograman
- Python

### Deep Learning
- YOLOv8 Classification (Ultralytics)

### Computer Vision
- OpenCV

### Library
- Ultralytics
- OpenCV

---

## 📂 Struktur Proyek

```text
Sistem-Pengenalan-Bahasa-Isyarat-Bisindo-Menggunakan-yolov8-classification/
│
├── abjad/
│   ├── dataset/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   ├── train_abjad.py
│   ├── predict_abjad.py
│   └── yolov8n-cls.pt
│
├── angka/
│   ├── dataset/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   ├── train_angka.py
│   ├── predict_angka.py
│   └── yolov8n-cls.pt
│
├── kata/
│   ├── dataset/
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   ├── training_kata.py
│   ├── predict_kata.py
│   └── yolov8n-cls.pt
│
├── app.py
├── main.py
├── capture_data.py
├── capture_data_angka.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```
---

## 📊 Dataset

Dataset yang digunakan merupakan gabungan dari:

- Dataset yang dikumpulkan secara mandiri menggunakan webcam.
- Dataset dari Kaggle.
- Dataset dari Roboflow.

Seluruh dataset telah melalui proses **preprocessing** dan **augmentasi data** sebelum digunakan dalam proses pelatihan model **YOLOv8 Classification**.

Karena ukuran dataset cukup besar, dataset **tidak disertakan** dalam repository GitHub.

Dataset dapat diakses melalui Kaggle:

> **Tambahkan link Kaggle dataset milikmu di sini.**

---

## 📁 Kategori Dataset

### Huruf
- A–Z (**26 kelas**)

### Angka
- 0–9 (**10 kelas**)

### Kata BISINDO

Contoh kategori:

- Halo
- Terima Kasih
- Tolong
- Maaf
- Saya
- Kamu

> **Sesuaikan daftar kata di atas dengan kategori yang benar-benar digunakan pada datasetmu.**

---

## 🔄 Alur Kerja Sistem

1. Pengguna melakukan gesture BISINDO di depan webcam.
2. Webcam menangkap citra secara **real-time**.
3. Citra diproses menggunakan **OpenCV**.
4. Model **YOLOv8 Classification** melakukan klasifikasi gesture.
5. Sistem menampilkan hasil prediksi beserta nilai **confidence**.

---

## 🚀 Instalasi

### Clone Repository

```bash
git clone https://github.com/NazwaFiryalCherry/Sistem-Pengenalan-Bahasa-Isyarat-Bisindo-Menggunakan-yolov8-classification.git
```

Masuk ke folder project:

```bash
cd Sistem-Pengenalan-Bahasa-Isyarat-Bisindo-Menggunakan-yolov8-classification
```

### Install Dependency

```bash
pip install -r requirements.txt
```

---

## 🧠 Training Model

### Huruf

```bash
python abjad/train_abjad.py
```

### Angka

```bash
python angka/train_angka.py
```

### Kata

```bash
python kata/training_kata.py
```

---

## 🎯 Prediksi

### Huruf

```bash
python abjad/predict_abjad.py
```

### Angka

```bash
python angka/predict_angka.py
```

### Kata

```bash
python kata/predict_kata.py
```

---

## 📈 Hasil Sistem

Model **YOLOv8 Classification** mampu mengenali gesture **Bahasa Isyarat Indonesia (BISINDO)** berdasarkan citra yang diberikan.

Output sistem berupa:

- Label hasil klasifikasi.
- Nilai **confidence** dari hasil prediksi.
- Tampilan hasil klasifikasi secara **real-time** menggunakan webcam.

> **Tambahkan screenshot hasil prediksi atau tampilan aplikasi pada bagian ini agar repository lebih menarik.**

---

## 👩‍💻 Pengembang

**Nazwa Firyal Cherry**  
Mahasiswa Teknik Informatika  
Jurusan Teknologi Informasi dan Komputer  
Politeknik Negeri Lhokseumawe

---

## 📄 Lisensi

Project ini menggunakan **MIT License**.

Dataset yang digunakan tetap mengikuti lisensi dari masing-masing sumber, seperti **Kaggle**, **Roboflow**, maupun dataset yang dikumpulkan secara mandiri.
- NumPy
- PyTorch
