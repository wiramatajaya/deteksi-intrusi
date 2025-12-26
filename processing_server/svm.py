import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# 1. Memuat Dataset dari Hugging Face
print("Sedang mengunduh dataset...")
dataset = load_dataset("shengqin/web-attacks", split="train")
df = pd.DataFrame(dataset)

# Menampilkan 5 data teratas untuk memastikan kolom
# Dataset ini biasanya memiliki kolom 'text' (payload) dan 'label' (0/1)
print(df.head())

# 2. Preprocessing Data
# Kita asumsikan kolom teks payload bernama 'text' dan target adalah 'label'
# Jika nama kolom berbeda (misal 'url' atau 'payload'), sesuaikan kodenya di sini.
X_raw = df['text_label'].astype(str) 
y = df['Label']

# Mengubah teks menjadi representasi numerik dengan TF-IDF (N-gram karakter)
# Menggunakan analyzer 'char' sangat efektif untuk mendeteksi pola SQLi/XSS
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000)
X = vectorizer.fit_transform(X_raw)

# 3. Split Data (Training & Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Inisialisasi dan Training Model SVM
# Menggunakan Linear SVC karena lebih cepat untuk dataset berukuran menengah-besar
print("Sedang melatih model SVM (ini mungkin memakan waktu beberapa menit)...")
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# 5. Evaluasi Model
y_pred = model.predict(X_test)
print("\n--- Laporan Klasifikasi ---")
print(f"Akurasi: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

# 6. Fungsi Prediksi Log/Payload Baru
def deteksi_payload(payload):
    transformed = vectorizer.transform([payload])
    pred = model.predict(transformed)
    prob = model.predict_proba(transformed)[0]
    
    hasil = "BERBAHAYA (Serangan)" if pred[0] == 1 else "AMAN (Normal)"
    return f"Hasil: {hasil} | Kepercayaan: {max(prob)*100:.2f}%"

# Contoh Pengujian
print("\n--- Uji Coba Real-time ---")
test_query = "SELECT * FROM users WHERE id='1' OR '1'='1'"
print(f"Payload: {test_query}")
print(deteksi_payload(test_query))
