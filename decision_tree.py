import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier # Import algoritma Decision Tree
from sklearn.metrics import classification_report, accuracy_score
from sklearn import preprocessing
import joblib

# --- Asumsi Data: Memuat data log yang sudah diproses menjadi numerik ---
# Ganti 'web_logs_processed.csv' dengan path file Anda yang sudah siap
try:
    df = pd.read_csv('web_logs_processed.csv')
except FileNotFoundError:
    print("Error: File 'web_logs_processed.csv' tidak ditemukan.")
    print("Pastikan Anda memiliki dataset log yang sudah diolah menjadi format numerik.")
    exit()

# Pisahkan fitur (X) dan target (y)
X = df.drop(['Label'], axis=1) # Sesuaikan nama kolom target jika berbeda
y = df['Label']

# --- Pra-pemrosesan (Jika data belum sepenuhnya numerik, gunakan ini) ---
# Kode otomatisasi konversi objek/string ke numerik
le = preprocessing.LabelEncoder()
for column_name in X.columns:
    if X[column_name].dtype == 'object':
        X[column_name] = le.fit_transform(X[column_name])

# --- 1. Membagi Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Penggunaan 30% data uji sering digunakan dalam penelitian IDS

print(f"Data training shape: {X_train.shape}")
print(f"Data testing shape: {X_test.shape}")

# --- 2. Menginisialisasi & Melatih Model Decision Tree ---
print("Melatih model Decision Tree...")
# Menggunakan kriteria 'entropy' (Information Gain) untuk pemisahan node
model_dt = DecisionTreeClassifier(criterion='entropy', random_state=42) 
model_dt.fit(X_train, y_train) # Proses training
print("Training selesai.")

# --- 3. Mengevaluasi Model ---
y_pred = model_dt.predict(X_test) # Membuat prediksi pada data uji

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy (Akurasi) Model Decision Tree: {accuracy * 100:.2f}%")

# Laporan klasifikasi detail
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))

# --- 4. Menyimpan Model (Opsional) ---
joblib.dump(model_dt, 'ids_decision_tree_model.joblib')
print("\nModel disimpan sebagai 'ids_decision_tree_model.joblib'")
