import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn import preprocessing # Import modul preprocessing

# --- 1. Memuat dan Mempersiapkan Data ---
data_path = "hf://datasets/shengqin/web-attacks/train.csv"
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print("Pastikan Anda sudah menginstal 'fsspec' dan 'huggingface_hub': pip install fsspec huggingface_hub")
    exit()

# Pisahkan fitur (X) dan target (y)
X = df.drop(['ID', 'Label'], axis=1, errors='ignore') 
y = df['Label'] 

# --- Perbaikan Error: Konversi data non-numerik ke numerik ---
le = preprocessing.LabelEncoder()
for column_name in X.columns:
    if X[column_name].dtype == 'object':
        X[column_name] = le.fit_transform(X[column_name])
# Catatan: Kolom target 'y' mungkin juga perlu dikonversi jika berisi string.

# --- 2. Membagi Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data training shape: {X_train.shape}")
print(f"Data testing shape: {X_test.shape}")

# --- 3. Menginisialisasi & Melatih Model Random Forest ---
print("Melatih model Random Forest...")
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train) # Proses training sekarang seharusnya berhasil
print("Training selesai.")

# --- 4. Mengevaluasi Model ---
y_pred = model_rf.predict(X_test) 

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy (Akurasi) Model: {accuracy * 100:.2f}%")

print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))
