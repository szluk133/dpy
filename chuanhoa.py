# Bước 2: Chuẩn hóa toàn cục các vector trong MongoDB
import pymongo
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- Cấu hình MongoDB ---
CONNECTION_STRING = "mongodb+srv://szluk133:sncuong2003@cluster0.dggxi.mongodb.net/"
DB_NAME = 'test3'
COLLECTION_NAME = 'audio_features'

client = pymongo.MongoClient(CONNECTION_STRING)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# --- Trích xuất toàn bộ vector đặc trưng ---
documents = list(collection.find({}, {"_id": 1, "raw_feature_vector": 1}))

if not documents:
    print("⚠️ Không tìm thấy vector nào trong MongoDB để chuẩn hóa.")
    exit()

vectors = np.array([doc["raw_feature_vector"] for doc in documents])

# --- Chuẩn hóa toàn cục (Z-score theo từng đặc trưng) ---
scaler = StandardScaler()
global_normalized_vectors = scaler.fit_transform(vectors)

# --- Cập nhật lại vector đã chuẩn hóa toàn cục vào MongoDB ---
for i, doc in enumerate(documents):
    collection.update_one(
        {"_id": doc["_id"]},
        {"$set": {"global_normalized_vector": global_normalized_vectors[i].tolist()}}
    )

print("✅ Hoàn tất chuẩn hóa toàn cục cho tất cả vector trong MongoDB.")

client.close()
