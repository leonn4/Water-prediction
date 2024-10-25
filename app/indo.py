import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Menonaktifkan notasi ilmiah di pandas
pd.set_option('display.float_format', '{:.2f}'.format)

# Mengimpor data dari file CSV
df = pd.read_csv('data_air_bersih.csv')

# Mengubah nama kolom
df.columns = ['Kecamatan', 'Volume_Air_m3', 'Jumlah_Pelanggan', 'Jumlah_Penduduk']

# Menampilkan data yang diimpor
print("Data yang Diimpor dengan Nama Kolom yang Diperbarui:")
print(df)

# Menampilkan kolom yang tersedia dalam DataFrame
print("\nKolom yang Tersedia dalam DataFrame:")
print(df.columns)

# Mendefinisikan variabel independen (X) dan dependen (y)
X = df[['Jumlah_Pelanggan', 'Jumlah_Penduduk']]  # Variabel independen
y = df['Volume_Air_m3']  # Variabel dependen

# Menambahkan konstanta ke model
X = sm.add_constant(X)

# Membangun model regresi
model = sm.OLS(y, X).fit()

# Melihat ringkasan model
print("\nRingkasan Model Regresi:")
print(model.summary())

# Menghitung akurasi model
mse = mean_squared_error(y, model.predict(X))
r_squared = r2_score(y, model.predict(X))

# Menampilkan akurasi model
print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r_squared:.2f}")

# Membuat skenario di mana jumlah pelanggan sama dengan jumlah penduduk
df_skenario = df.copy()  # Menyalin DataFrame asli
df_skenario['Jumlah_Pelanggan'] = df_skenario['Jumlah_Penduduk']  # Mengatur pelanggan = penduduk

# Menyiapkan variabel X untuk prediksi berdasarkan skenario
X_skenario = df_skenario[['Jumlah_Pelanggan', 'Jumlah_Penduduk']]
X_skenario = sm.add_constant(X_skenario)  # Menambahkan konstanta

# Memprediksi berdasarkan skenario
prediksi_skenario = model.predict(X_skenario)

# Menambahkan hasil prediksi ke DataFrame
df_skenario['Volume_Air_Prediksi'] = prediksi_skenario

# Menampilkan hasil prediksi untuk skenario
print("\nPrediksi Volume Air jika Semua Penduduk Menjadi Pelanggan:")
print(df_skenario[['Kecamatan', 'Jumlah_Pelanggan', 'Jumlah_Penduduk', 'Volume_Air_Prediksi']])

# Menghitung jumlah penduduk yang belum menjadi pelanggan
df['Penduduk_Belum_Pelanggan'] = df['Jumlah_Penduduk'] - df['Jumlah_Pelanggan']

# Menampilkan data dengan jumlah penduduk yang belum menjadi pelanggan
print("\nData dengan Penduduk yang Belum Menjadi Pelanggan:")
print(df[['Kecamatan', 'Jumlah_Pelanggan', 'Jumlah_Penduduk', 'Penduduk_Belum_Pelanggan']])

# Visualisasi hasil prediksi dan penduduk yang belum menjadi pelanggan

# 1. Visualisasi prediksi volume air jika semua penduduk menjadi pelanggan
plt.figure(figsize=(10, 6))

# Grafik scatter untuk volume air prediksi berdasarkan jumlah pelanggan
plt.scatter(df_skenario['Jumlah_Pelanggan'], df_skenario['Volume_Air_Prediksi'], color='green', label='Prediksi Skenario', linewidth=2)

# Menambahkan garis yang menghubungkan titik-titik prediksi
plt.plot(df_skenario['Jumlah_Pelanggan'], df_skenario['Volume_Air_Prediksi'], color='green', linewidth=2)

# Mengatur nilai sumbu x dan y
plt.gca().ticklabel_format(style='plain', axis='x')  # Menonaktifkan notasi ilmiah pada sumbu X
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))  # Format sumbu Y sebagai angka reguler (tanpa desimal)

plt.title('Prediksi Volume Air Jika Semua Penduduk Menjadi Pelanggan')
plt.xlabel('Jumlah Pelanggan (Sama dengan Jumlah Penduduk)')
plt.ylabel('Volume Air (m³)')
plt.legend()
plt.show()

# 2. Visualisasi perbandingan penduduk yang belum menjadi pelanggan dan pelanggan yang ada
plt.figure(figsize=(10, 6))
plt.bar(df['Kecamatan'], df['Jumlah_Pelanggan'], color='blue', label='Pelanggan', alpha=0.7)
plt.bar(df['Kecamatan'], df['Penduduk_Belum_Pelanggan'], bottom=df['Jumlah_Pelanggan'], color='red', label='Penduduk Belum Pelanggan', alpha=0.7)
plt.title('Perbandingan Jumlah Pelanggan dan Penduduk yang Belum Menjadi Pelanggan per Kecamatan')
plt.xlabel('Kecamatan')
plt.ylabel('Jumlah Orang')
plt.xticks(rotation=45, ha='right')  # Memutar label kecamatan agar lebih mudah dibaca
plt.legend()
plt.show()
