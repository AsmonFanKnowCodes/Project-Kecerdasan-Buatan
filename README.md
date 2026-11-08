# HIV Outcomes Lab

Tampilan Asklepios mengambil inspirasi dari arsitektur dan patung Yunani. Gambar dibuat dengan ImageGen. Arah visual `assets/parthenon.webp` adalah foto perjalanan tampak depan dengan fasad dan tangga utuh. `assets/asklepios.webp` adalah potret torso patung marmer dengan cahaya museum yang lembut, wajah menoleh, tongkat berular, dan latar transparan. `assets/sanctuary.webp` menampilkan deretan pilar dari dalam suaka penyembuhan dengan pepohonan zaitun dan cahaya sore. `assets/simulation-relief.webp` memakai marmer krem dengan relief Hygieia di kanan dan bidang batu yang rata di kiri.

Proyek kuliah Kecerdasan Buatan untuk mendemonstrasikan klasifikasi biner di Streamlit. Pengguna dapat mencoba profil fiktif, melihat skor model untuk label mortalitas lima tahun, dan membandingkan hasil evaluasi.

Eksperimen ini memakai data sintetis untuk pembelajaran. Skor mengikuti pola dalam generator dan belum divalidasi pada pasien nyata. Gunakan profil fiktif saat mencoba demo.

## Jalankan web

Butuh Python 3.13. Buka PowerShell di folder repo, lalu jalankan:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Buka [localhost:8501](http://localhost:8501). Dashboard langsung membuka **Simulasi**. Pilih salah satu profil contoh dan tekan **Jalankan prediksi**. Menu samping memuat **Evaluasi model**, **Tentang data**, dan **Beranda** dengan ilustrasi Parthenon serta Asklepios. Artefak model siap pakai ada di `artifacts/`.

## Latih ulang

```powershell
.\.venv\Scripts\python.exe train.py
```

Script membaca `archive.zip` yang sudah tersedia di repo dan menulis `artifacts/model.joblib` serta `artifacts/metrics.json`. Opsi `--data` menerima path ke ZIP berisi satu CSV atau ke CSV dengan skema dataset yang sama. Muat `model.joblib` hanya dari repo tepercaya atau hasil training sendiri.

## Dataset dan target

`archive.zip` identik dengan versi 1 dari dataset Kaggle oleh Mobeen Fatimah. Isinya 500.000 rekaman sintetis dan 33 kolom. Ukuran ZIP 17,67 MB dan CSV hasil ekstraksi 74,16 MB. Lisensi dataset tercantum sebagai CC BY-NC-SA 4.0. [Dataset Kaggle](https://www.kaggle.com/datasets/mobeenfatimah/hivaids-clinical-global-dataset)

Model mempelajari label `Mortality_5Yr_Outcome`. Kelas `1` menandakan kematian dalam lima tahun pada simulasi dataset dan kelas `0` menandakan label sebaliknya. Sekitar 4,83% rekaman berlabel `1`.

Empat input menggambarkan kondisi saat pengukuran, yaitu jumlah CD4 saat ini, viral load saat ini, koinfeksi TB, dan mutasi resistansi obat. Generator membentuk label dari CD4, koinfeksi TB, mutasi resistansi, status supresi yang diturunkan dari viral load, dan variasi acak. `Patient_ID`, `Viral_Suppression_Flag`, dan kolom target dikeluarkan dari input model. [Kode generator](https://github.com/MobeenFatimaa/HIV-AIDS-Clinical-Global-dataset-Generator/blob/main/generate.py)

## Metode dan hasil

- Split terstratifikasi dengan seed `42` menghasilkan 350.000 baris latih, 75.000 validasi, dan 75.000 uji.
- Logistic Regression memakai transformasi `log1p` untuk viral load dan standardisasi seluruh input. Decision Tree dan Random Forest memakai input asli. Dummy Classifier memakai prevalensi kelas latih sebagai skor tetap.
- Random Forest mendapat Average Precision validasi tertinggi (`0,296`). Ambang `0,1683` memaksimalkan F1 pada data validasi dan dipakai untuk menghasilkan kelas di data uji.
- Pada 75.000 rekaman uji, model mendapat Average Precision `0,304`, ROC AUC `0,896`, precision `27,8%`, recall `72,4%`, dan F1 `0,402`. Skor Average Precision pembanding tanpa fitur di validasi sekitar `0,048`.
- Model menandai 2.623 dari 3.621 rekaman berlabel `1`. Sebanyak 6.804 rekaman berlabel `0` juga ditandai sebagai kelas `1` pada ambang demo tersebut.

## Audit tampilan

UI-Max Scorecard: 93/100, berdasarkan audit desain dan pengujian Chrome.

| Kriteria | Nilai | Bukti |
| --- | --- | --- |
| MOTION-01 | 12/12 | Judul dan deskripsi masuk bertahap dengan offset 60 ms serta pergeseran vertikal. |
| MOTION-02 | 9/10 | Navigasi, tombol form, dan teks hero memiliki gerakan tersendiri. |
| LAYOUT-01 | 11/12 | Fasad dan tangga terlihat utuh pada hero satu layar yang langsung disusul panel patung. |
| DEPTH-01 | 9/12 | Potret patung transparan berada di depan latar suaka penyembuhan pada panel kedua. |
| INTERACTION-01 | 9/10 | Tautan dan tombol merespons hover serta tekan dengan pergeseran kecil. |
| A11Y-01 | 13/14 | Foto punya alt text, fokus keyboard terlihat, dan mode reduced motion mematikan animasi. |
| PERF-01 | 12/12 | Aset gambar memakai WebP dan animasi CSS berjalan tanpa listener scroll. |
| RESP-01 | 10/10 | Navigasi, simulasi, dan evaluasi lulus uji pada lebar 375, 768, dan 1440 px. |
| BRAND-01 | 8/8 | Foto Parthenon dan Asklepios memberi konteks visual untuk tema kesehatan. |
