# create_scaler.py
import pymongo
import numpy as np
import pickle

# Kết nối MongoDB
client = pymongo.MongoClient("mongodb+srv://szluk133:sncuong2003@cluster0.dggxi.mongodb.net/")
db = client['test3']
collection = db['audio_features']

# Lấy tất cả raw_feature_vector từ MongoDB
docs = list(collection.find({}, {"raw_feature_vector": 1}))
vectors = np.array([doc["raw_feature_vector"] for doc in docs])

# Tính mean và std
mean = np.mean(vectors, axis=0)
std = np.std(vectors, axis=0)

# Lưu vào scaler.pkl
with open("scaler.pkl", "wb") as f:
    pickle.dump({"mean": mean, "std": std}, f)

print("✅ scaler.pkl đã được tạo thành công.")
