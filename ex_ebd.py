import os
import pymongo
import librosa
import numpy as np
import scipy.signal
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

CONNECTION_STRING = "mongodb+srv://szluk133:sncuong2003@cluster0.dggxi.mongodb.net/"
DB_NAME = 'test3'
COLLECTION_NAME = 'audio_features'

N_MFCC = 13
LPC_ORDER = 16
NUM_FORMANTS = 3

try:
    client = pymongo.MongoClient(CONNECTION_STRING)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    print("✅ Kết nối thành công tới MongoDB Atlas.")
except pymongo.errors.ConfigurationError as e:
    print(f"Lỗi cấu hình MongoDB: {e}")
    exit()
except pymongo.errors.ConnectionFailure as e:
    print(f"Không thể kết nối tới MongoDB: {e}")
    exit()

def estimate_formants(y, sr, order, num_formants_to_find):
    formant_freqs = [np.nan] * num_formants_to_find
    try:
        if len(y) < order + 1:
            return formant_freqs
        a = librosa.lpc(y, order=order)
        b_coeffs = np.array([1.0])
        a_coeffs = np.hstack([1.0, a])
        w_hz, h = scipy.signal.freqz(b_coeffs, a_coeffs, worN=4096, fs=sr)
        spectrum = np.abs(h)
        min_peak_height = np.mean(spectrum) * 0.5
        min_distance_hz = 300
        min_distance_samples = int(min_distance_hz / (sr / (2 * len(w_hz))))
        peaks_indices, _ = scipy.signal.find_peaks(spectrum, height=min_peak_height, distance=min_distance_samples)
        if len(peaks_indices) > 0:
            found_freqs = sorted(w_hz[peaks_indices])
            for i in range(min(len(found_freqs), num_formants_to_find)):
                formant_freqs[i] = found_freqs[i]
    except Exception:
        pass
    return formant_freqs

def extract_features(audio_path, y, sr):
    features = []
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        features.extend(np.mean(mfccs, axis=1))
    except:
        features.extend([np.nan] * N_MFCC)

    try:
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        mean_f0 = np.nanmean(f0)
        features.append(mean_f0 if not np.isnan(mean_f0) else 0.0)
    except:
        features.append(np.nan)

    try:
        formant_values = estimate_formants(y, sr, LPC_ORDER, NUM_FORMANTS)
        features.extend(formant_values)
    except:
        features.extend([np.nan] * NUM_FORMANTS)

    try:
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
    except:
        features.append(np.nan)

    try:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(cent))
    except:
        features.append(np.nan)

    try:
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
    except:
        features.append(np.nan)

    feature_vector = np.array(features, dtype=np.float64)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    return feature_vector.tolist()

def insert_features_to_db(file_path, duration, raw_feature_vector):
    document = {
        "file_path": file_path,
        "duration_seconds": duration,
        "raw_feature_vector": raw_feature_vector,
        "feature_names": [
            *[f"mfcc_{i+1}" for i in range(N_MFCC)],
            "pitch_f0",
            *[f"formant_{i+1}" for i in range(NUM_FORMANTS)],
            "energy_rms",
            "spectral_centroid",
            "zero_crossing_rate"
        ]
    }
    try:
        collection.insert_one(document)
    except Exception as e:
        print(f"Lỗi khi lưu vào MongoDB cho {os.path.basename(file_path)}: {e}")

def process_directory(folder_path):
    processed_count = 0
    error_count = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            print(f"Đang xử lý file: {filename}...")
            try:
                y, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                raw_feature_vector = extract_features(file_path, y, sr)
                if raw_feature_vector:
                    insert_features_to_db(file_path, duration, raw_feature_vector)
                    print(f"✅ Đã xử lý và lưu đặc trưng cho: {filename}")
                    processed_count += 1
                else:
                    print(f"⚠️ Không thể trích xuất đặc trưng cho: {filename}. Bỏ qua.")
                    error_count += 1
            except Exception as e:
                print(f" Lỗi không xác định khi xử lý file {filename}: {e}")
                error_count += 1
                continue
    print(f"\n--- Hoàn tất xử lý thư mục ---")
    print(f"Tổng số file đã xử lý thành công: {processed_count}")
    print(f"Tổng số file gặp lỗi: {error_count}")

if __name__ == "__main__":
    audio_folder = "E:/NAM_4_KY_2_2025/CSDL_ĐPT/filtered_audio"
    if not os.path.isdir(audio_folder):
        print(f"Lỗi: Thư mục '{audio_folder}' không tồn tại.")
    else:
        process_directory(audio_folder)
        print("🎯 Chương trình đã hoàn thành!")
    client.close()
    print("Đã đóng kết nối MongoDB.")
