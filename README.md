# HIV Outcomes Lab

HIV Outcomes Lab adalah proyek kuliah Kecerdasan Buatan yang menggunakan Streamlit untuk mendemonstrasikan klasifikasi mortalitas lima tahun pada profil HIV sintetis. Dashboard mencakup simulasi prediksi, perbandingan model, dan ringkasan dataset. Seluruh evaluasi menggunakan data sintetis.

## Menjalankan aplikasi

Gunakan Python 3.13 atau 3.14. Dari folder repositori, buat lingkungan virtual dengan `python -m venv .venv`, lalu aktifkan sesuai sistem operasi.

| Sistem | Perintah aktivasi |
| --- | --- |
| Windows PowerShell | `\.venv\Scripts\Activate.ps1` |
| macOS atau Linux | `source .venv/bin/activate` |

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Buka [localhost:8501](http://localhost:8501). Pilih profil contoh pada halaman **Simulasi**, lalu tekan **Jalankan prediksi**. Aplikasi memuat model terlatih dari `artifacts/model.joblib`.

## Dataset

`archive.zip` berisi versi 1 **HIV/AIDS Clinical Global Dataset** dari Mobeen Fatimah. Dataset memuat 500.000 rekaman sintetis dan 33 kolom, dengan lisensi CC BY-NC-SA 4.0. [Sumber dataset](https://www.kaggle.com/datasets/mobeenfatimah/hivaids-clinical-global-dataset)

Target klasifikasi adalah `Mortality_5Yr_Outcome`. Kelas `1` menandai kematian dalam lima tahun pada data sintetis dan mencakup 4,83% rekaman. Kelas `0` menandai hasil lainnya. Model menerima empat fitur, yaitu jumlah CD4, viral load, koinfeksi tuberkulosis, dan mutasi resistansi obat. [Kode generator dataset](https://github.com/MobeenFatimaa/HIV-AIDS-Clinical-Global-dataset-Generator/blob/main/generate.py)

## Evaluasi model

Data dibagi secara terstratifikasi menjadi 350.000 rekaman latih, 75.000 rekaman validasi, dan 75.000 rekaman uji dengan seed `42`. Logistic Regression, Decision Tree, dan Random Forest dibandingkan dengan Dummy Classifier sebagai pembanding. Random Forest memperoleh Average Precision tertinggi pada data validasi. Ambang **16,83%** dipilih berdasarkan F1 pada data validasi.

| Metrik pada data uji | Hasil |
| --- | ---: |
| Average Precision | 0,304 |
| ROC AUC | 0,896 |
| Precision | 27,8% |
| Recall | 72,4% |
| F1 | 0,402 |

## Melatih ulang model

```sh
python train.py
```

Skrip membaca `archive.zip` dan menghasilkan `artifacts/model.joblib` serta `artifacts/metrics.json`. Gunakan `--data PATH` untuk memilih CSV atau ZIP yang berisi satu CSV dengan skema yang sama.

## Kredit visual

Ilustrasi pada dashboard mengambil inspirasi dari arsitektur dan patung Yunani serta dibuat dengan ImageGen.
