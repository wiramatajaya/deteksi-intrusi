import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- 1. Memuat dan Mempersiapkan Data ---

# Catatan: Asumsi kolom fitur bernama 'x1' sampai 'x325' dan target bernama 'score'
# Sesuaikan nama kolom jika struktur dataset berbeda
data_path = "hf://datasets/shengqin/web-attacks/train.csv"
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print("Pastikan Anda sudah menginstal 'fsspec' dan 'huggingface_hub': pip install fsspec huggingface_hub")
    exit()

# Pisahkan fitur (X) dan target (y)
# Drop kolom ID jika ada, dan pastikan hanya fitur yang masuk ke X
X = df.drop(['id', 'score'], axis=1, errors='ignore') 
y = df['score'] 

# --- 2. Membagi Data ---
# 80% untuk training, 20% untuk testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data training shape: {X_train.shape}")
print(f"Data testing shape: {X_test.shape}")

# --- 3. Menginisialisasi & Melatih Model Random Forest ---
# Menggunakan 100 pohon keputusan
print("Melatih model Random Forest...")
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train) # Proses training
print("Training selesai.")

# --- 4. Mengevaluasi Model ---
y_pred = model_rf.predict(X_test) # Membuat prediksi pada data uji

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy (Akurasi) Model: {accuracy * 100:.2f}%")

# Laporan klasifikasi detail
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))
