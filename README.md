# Europe Hotel Satisfaction Score

## Domain Proyek

Proyek machine learning ini adalah Europe Hotel Satisfaction Score yaitu skor kepuasan hotel eropa.Perusahaan Jasa dengan ukuran dari kepuasan pelanggan. Penilaian kinerja perusahaan tersebut dengan menggunakan kuisoner[1]. Pada tiap perspektif tersebut ditentukan tujuan, target dan tolak ukur yang menjadi dasar pengukuran kinerja perusahaan.yang mana data skor ini didapat dari setiap pelanggan dengan mengisi *feedback* seperti kritik dan saran atau mengisi form khusus agar pemilik hotel mengetahui apa saja kekurangan pada hotel. data tersebut didapat setelah menginap di suatu hotel di eropa. dengan skor ini, kita bisa mengetahui hotel mana yang paling baik pelayanan dan fasilitas nya.

## Business Understanding

### Problem Statements

1. Bagaimana cara menganalisis data pelanggan yang baik?
2. Bagaimana cara membangun model yang dapat memprediksi hotel terbaik?
   ### Goals

- Menganalisis data pelanggan hotel yang ada eropa menggunakan model yang optimal
- Memproses model , yaitu untuk menentukan model yang dapat memproses data latih dengan baik

### Solution Statements

#### Pra-premprosesan Data :

1. Deskripsikan _variable_ yang ada pada dataset
2. Lakukan _dropping_ data yang tidak digunakan
3. Menangani _missing value_ pada data
4. Menangani _outlier_ pada data menggunakan _box plot_ dan _IQR method_
5. Normalisasi fitur _numeric_ pada dataset
   ### Pembangunan Model
   1. Random Forest
   2. Gradient Boosting Algorithm

### Data Understanding

#### Informasi Dataset

Dataset ini memiliki mempunyai total 103904 baris dan 17 kolom dataset ini tidak memiliki _missing value_ pada masing-masing kolom. Informasi detail dari dataset sebagai berikut:

* Nama Dataset = Europe Hotel Booking Satisfaction Score
* link Dataset = "https://www.kaggle.com/datasets/ishansingh88/europe-hotel-satisfaction-score?select=Europe+Hotel+Booking+Satisfaction+Score.csv"
* Sumber = kaggle.com
* Dataset Owner = ISHAN SINGH
* Kategori = _Hotels and Accommodations_
* Jenis dan ukuran berkas = CSV (9.65MB)

variabel variabel pada dataset dapat dilihat pada tabel1 :
|Nama Variabel |Keterangan|
|----------|:-------------:|
|Gender| laki-laki dan perempuan
|Age| Usia dari 7 tahun hingga 85 tahun
|purposeof_travel| tujuan perjalanan, seperti: penerbangan, akademik, pribadi, bisnis, pariwisata.
|Type of Travel| Jenis Perjalanan, seperti: Perjalanan kelompok, Perjalanan Pribadi.
|Type Of Booking| Jenis Pemesanan, seperti: Pemesanan rombongan, Perorangan/Pasangan.
|Hotel wifi service| Layanan wifi hotel dari peringkat 1 sampai 5.
|Departure/Arrival convenience| Kenyamanan Keberangkatan/Kedatangan dari peringkat 1 sampai 5.
|Ease of Online booking| Kemudahan pemesanan Online dari peringkat 1 sampai 5.
|Hotel location| Lokasi hotel dari peringkat 1 sampai 5.
|Food and drink| Makanan dan minuman dari peringkat 1 sampai 5.
|Stay comfort| Tetap nyaman dari peringkat 1 sampai 5.
|Common Room entertainment| Hiburan Common Room dari peringkat 1 sampai 5.
|Checkin/Checkout service| Layanan Checkin/Checkout dari peringkat 1 sampai 5.
|Other service| Layanan lain dari peringkat 1 sampai 5.
|Cleanliness| Kebersihan dari peringkat 1 sampai 5.
|satisfaction| kepuasan, ada beberapa pilihan seperti: puas, netral atau tidak puas.
tabel1. nama variabel pada dataset

#### Exploratary Data Analysist

##### Mengidentifikasi Missing Value dan Outlier

Pertama perlu dilakukan setelah memuat dataset yaitu mengetahui info dan deskripsi. Bisa dilakukan dengan perintah berikut secara berurutan dalam _cell_ yang berbeda:

df.info()
df.describe()

kemudian cari jumlah _missing value_ yang ada dalam data. dengan hasil pada tabel2 dan gambar1 dibawah ini
| | |
|----------|:-------------:|
|Gender| 0.0
|Age| 0.0
|purpose_of_travel| 0.0
|Type of Travel| 0.0
|Type Of Booking| 0.0
|Hotel wifi service| 0.0
|Departure/Arrival convenience| 0.0
|Ease of Online booking| 0.0
|Hotel location| 0.0
|Food and drink| 0.0
|Stay comfort| 0.0
|Common Room entertainment| 0.0
|Checkin/Checkout service| 0.0
|Other service| 0.0
|Cleanliness| 0.0
|satisfaction| 0.0
|dtype: float64|
tabel2. mengecek apakah ada _missing value_

![g8](https://user-images.githubusercontent.com/72668852/191710672-8aa0f3db-7f98-41ac-b44f-1d1822e92ea7.png)

gambar1. mengecek _missing value_ menggunakan _missingno_

Selanjutnya melihat _outlier_ dari data

- Kemudahan pemesanan online. dapa diliat pada gambar1

  ![g1](https://user-images.githubusercontent.com/72668852/191710967-d01f8d98-c6d0-4780-85bd-62a1f8985d9c.png)
  
  gambar2. Mendeteksi outliers untuk kemudahan pemesanan online

- Lokasi hotel

  ![g2](https://user-images.githubusercontent.com/72668852/191711083-1c7fd13c-bf12-42a5-b13a-b9c87e7a5de7.png)
  
  gambar3. Mendeteksi outliers untuk lokasi hotel

* Ruangan untuk hiburan bersama

  ![g3](https://user-images.githubusercontent.com/72668852/191711228-1396a313-689d-498f-922c-8886014a3501.png)
  
  gambar4. Mendeteksi outliers untuk ruangan hiburan

### Univariate Analysist

![g4](https://user-images.githubusercontent.com/72668852/191711834-05bdadb5-e319-45b6-95b2-42a6c099a331.png)

gambar5. Histogram numeric features

### Multivariate Analysist

![g5](https://user-images.githubusercontent.com/72668852/191711984-d1392a2e-f6c1-4c12-93b7-4fa8a636e08b.png)

gambar6. Mengevaluasi skor korelasinya

#### kesimpulan

- Terlihat pada grafik bahwa semua data cenderung distribusi nilainya membentuk seperti camel (naik dan turun)
- Volume memiliki korelasi yang sangat lemah dengan yang lain, artinya jumlah transaksi (volume) tidak ada hubungannya dengan fitur lain dan bisa di drop.

## Data Preparation

### Penerapan One-Hot-Encoding

_Encoding_ fitur kategori adalah teknik untuk mengubah fitur _categorical_ ke dalam bentuk vektor biner yang bernilai integer 0 dan 1. saya menggunakan 5 variabel dalam dataset yaitu: _Gender, purpose_of_travel, Type of Travel, Type Of Booking, satisfaction_. kemudian melakukan proses _encoding_ dengan fitur _get_dummies_. Dikarenakan data yang sangat banyak, mohon maaf tidak saya cantumkan disini. Hasil _output_ variabel kategori saya telah berubah menjadi variabel numerik. selanjutnya saya cek menggunakan fungsi pairplot, karena ketiga fitur ini memiliki informasi yang sama.
_get_dummies_ digunakan untuk mengubah variabel kategorikal menjadi variabel numerikal dengan melakukan proses One-Hot-Encode terhadap variabel kategorikal.
Berikut adalah hasil reduksi dimensi dapa dilihat pada gambar7 dibawah ini.

![g7](https://user-images.githubusercontent.com/72668852/191712121-046cf7a9-0a1d-4c2d-b42a-2f41669d7cb0.png)


gambar7. Hasil pairplot Reduksi Dimensi dengan PCA

selanjutnya,aplikasikan class PCA dari _library scikit learn_.
Paremeter yang saya masukkan ke dalam class adalah n_components dan random_state. Parameter n_components merupakan jumlah komponen atau dimensi, dalam kasus saya jumlahnya ada 3. dan mendapatkan hasil

array([0.4934, 0.3269, 0.1796])

selanjutnya saya membuat fitur bernama _dimension_. dengan beberapa perubahan yaitu:

- Menggunakan n_component = 1, karena kali ini, jumlah komponen kita hanya satu.
- Fit model dengan data masukan.
- Tambahkan fitur baru ke dataset dengan nama 'dimension' dan lakukan proses transformasi.
- Drop kolom.

### Penerapan Train-Test-Split

Dataset akan dibagai menjadi dua, yaitu sebagai _train_ data dan _test_. _Train_ data digunakan untuk _training_ model dan _test_ data digunakan sebagai validasi model. Proposi yang saya gunakan di sini adalah 90 berbanding 10, 90% sebagai _train_ data dan 10% sebagai _test_ data. dengan fungsi _train_test_split_ dari sklearn. Kemudian saya cek jumlah sampel pada masing-masing bagian.

- X_train: Untuk menampung data source yang akan dilatih.
- X_test: Untuk menampung data target yang akan dilatih.
- y_train: Untuk menampung data source yang akan digunakan untuk testing.
- y_test: Untuk menampung data target yang akan digunakan untuk testing.

### Penerapan Standarisasi

Saya menggunakan teknik _StandarScaler_ dari _library Scikitlearn_, yang mana hasil _output_ nya seperti di tabel3 berikut:
| |Age|
|----------|:-------------:|
|81999| -1.098512
|79433| -0.699700
|99954| -0.367357
|5154| 1.360828
|75327| -0.965575
tabel3. Penerapan _StandardScaler_ fitur standarisasi

- _StandardScaler_ melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.
- StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Hasilnya seperti tabel4 dibawah ini:

|       |    Age     |
| ----- | :--------: |
| count | 72810.0000 |
| mean  |   0.0000   |
| std   |   1.0000   |
| min   |  -2.1620   |
| 25%   |  -0.8326   |
| 50%   |   0.0315   |
| 75%   |   0.7626   |
| max   |   3.0225   |

tabel4. Proses standarisasi mengubah nilai rata-rata (mean) menjadi 0

Saya menerapkan fitur standarisasi pada data latih. Kemudian, pada tahap evaluasi, kita akan melakukan standarisasi pada data uji.

## Modeling

Pada proyek ini, Proses modeling dalam proyek ini menggunakan empat model, yaitu Adaptive Boosting, Random Forest. Kemudian, membandingkan performanya.

#### Random Forest

Algoritma ini disusun dari banyak algoritma pohon _(decision tree)_ yang pembagian data dan fiturnya dipilih secara acak. Untuk Parameter yang digunakan pada proyek kali, yaitu :

- ##### n_estimator
  yang merupakan jumlah _trees_ (pohon) di _forest_. Pada proyek ini penulis melakukan set nilai n_estimator sebesar 70 trees.
- ##### max_features
  persentase dari _features_ pada model yang kita pilih secara random setiap kita membuat cabang _decision tree_.
- ##### bootstrap

  untuk mempercepat proses pemodelan.

  #### Kelebihan :

- Algoritma Random Forest merupakan algoritma dengan pembelajaran paling akurat yang tersedia. Untuk banyak kumpulan data, algoritma ini menghasilkan pengklasifikasi yang sangat akurat.
- Berjalan secara efisien pada data besar[2].
- Dapat menangani ribuan variabel input tanpa penghapusan variabel.
- Memberikan perkiraan variabel apa yang penting dalam klasifikasi.
- Memiliki metode yang efektif untuk memperkirakan data yang hilang dan menjaga akurasi ketika sebagian besar data hilang.
  #### Kekurangan :
- Algoritma Random Forest overfiting untuk beberapa kumpulan data dengan tugas klasifikasi/regresi yang bising/noise.
- Untuk data yang menyertakan variabel kategorik dengan jumlah level yang berbeda, Random Forest menjadi bias dalam mendukung atribut dengan level yang lebih banyak. Oleh karena itu, skor kepentingan variabel dari Random Forest tidak dapat diandalkan untuk jenis data ini.

### Adaptive Boosting

Boosting adalah algoritma machine learning yang menggunakan teknik _ensembel learning_ dari _decision tree_ untuk memprediksi nilai. Boosting sangat mampu menangani pattern yang kompleks dan data ketika linear model tidak dapat menangani. Untuk parameter yang digunakan pada model ini, yaitu :

- ##### learning_rate
  _Hyperparameter training_ yang digunakan untuk menghitung nilai koreksi bobot padad waktu proses _training_. Umumnya nilai _learning rate_ berkisar antara 0 hingga 1. Pada proyek ini penulis menggunakan _learning_rate_ sebesar 0,01.
- ##### n_estimators
  Jumlah tahapan boosting yang akan dilakukan. Pada proyek ini penulis menggunakan n_estimators sebesar 1000.
  #### Kelebihan :
- Algoritma Boosting dapat mengurangi bias pada data.
- asil pemodelan yang lebih akurat.
- Algoritma ini sangat powerful dalam meningkatkan akurasi prediksi.
  #### Kekurangan :
- Waktu komputasi dan desain tinggi[3].
- Tingkat kesulitan yang tinggi dalam pemilihan model

### Kesimpulan

Dari kedua model diatas, menurut saya _random forest_ lah yang merupakan solusi terbaik untuk dataset proyek ini. karena lebih stabil dan bekerja baik pada _categorical_ dan _numerical_, selain itu, model ini lebih aman jika bertemu _noise_. Meski algoritma ini perlu waktu lebih lama untuk proses _training_.

### Evaluasi

Metrik yang saya gunakan pada prediksi ini adalah _MSE_ atau _Mean Squared Error_ yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. Sebelum menghitung nilai MSE dalam model lakukan proses scaling fitur numerik pada data uji. untuk menghindari kebocoran data.

#### Kesimpulan

- model _machine learning_ RF dan Boosting. yang mana model RF sudah memenuhi batas, artinya model sudah cukup bagus. Sedangkan _Boosting_ masih dibawah batas.

|          |  train   |     test |
| -------- | :------: | -------: |
| RF       | 0.000001 | 0.000059 |
| Boosting | 0.000092 | 0.000096 |

tabel5. hasil mse menggunakan 2 model

- model _random forest_ memiliki hasil _error_ MSE paling kecil yang artinya model ini yang terbaik dibanding _Boosting_

![g6](https://user-images.githubusercontent.com/72668852/191712308-5ab85f61-1db4-4788-9cc3-cc198b857771.png)


gambar7. Hasil diagram metrik

|       | y_true | prediksi_RF | prediksi_Boosting |
| ----- | ------ | ----------- | ----------------- |
| 30274 | 0      | 0           | 1                 |

tabel6. prediksi model

- untuk proyek kali ini model Boosting merupakan model yang berjalan dengan performa optimal sehingga dapat disimpulkan bahwa model dapat memprediksi hotel dengan scor terbaik. Sehingga kedepanya dapat membantu para wisatawan yang ingin menginap di hotel dengan fasilitas yang bagus untuk melakukan keputusan pemilihan.

## Referensi

[1] Muhammad Rusmono Jati Cgiash. (2006/07/07). ANALISIS KEPUASAN PELANGGAN ATAS KUALITASLAYANAN BERDASARKAN KONSEP SERVQUAL(Studi Kasus pada Hotel ARMI di Malang), from /http://eprints.umm.ac.id/6767/1/ANALISIS_KEPUASAN_PELANGGAN_ATAS_KUALITASLAYANAN_BERDASARKAN_KONSEP_SERVQUAL.pdf

[2] Chauhan, Ankit .(2021). Random Forest Classifier and its Hyperparameters. Medium. Retrieved September 15, 2022, from https://medium.com/analytics-vidhya/random-forest-classifier-and-its-hyperparameters-8467bec755f6

[3] Aliyev, V. (2020, October 7). Gradient boosting classification explained through python. Medium. Retrieved September 15, 2022, from https://towardsdatascience.com/gradient-boosting-classification-explained-through-python-60cc980eeb3d
[4] Khoiri. (2020). Cara Menghitung Mean Squared Error (MSE). Khoiri. Retrieved September 16, 2022, from https://www.khoiri.com/2020/12/pengertian-dan-cara-menghitung-mean-squared-error-mse.html
