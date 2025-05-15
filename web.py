import os
import shutil
import numpy as np
import pymongo
import librosa
import pickle
import scipy.signal
from flask import Flask, request, render_template, redirect, url_for, send_file, session
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CACHE_FOLDER'] = 'audio_cache'
app.secret_key = 'your_secret_key'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CACHE_FOLDER'], exist_ok=True)

connection_string = "mongodb+srv://szluk133:sncuong2003@cluster0.dggxi.mongodb.net/"
client = pymongo.MongoClient(connection_string)
db = client['test3']
collection = db['audio_features']

with open("scaler.pkl", "rb") as f:
    scaler_data = pickle.load(f)
    mean_vector = scaler_data['mean']
    std_vector = scaler_data['std']

N_MFCC = 13
NUM_FORMANTS = 3
LPC_ORDER = 16

def estimate_formants(y, sr, order, num_formants_to_find):
    formant_freqs = [0.0] * num_formants_to_find
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
    except:
        pass
    return formant_freqs

def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=None)
    features = []
    try:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        features.extend(np.mean(mfccs, axis=1))
    except:
        features.extend([0.0] * N_MFCC)
    try:
        f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        features.append(np.nanmean(f0) if not np.isnan(np.nanmean(f0)) else 0.0)
    except:
        features.append(0.0)
    try:
        formant_values = estimate_formants(y, sr, LPC_ORDER, NUM_FORMANTS)
        features.extend(formant_values)
    except:
        features.extend([0.0] * NUM_FORMANTS)
    try:
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
    except:
        features.append(0.0)
    try:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(cent))
    except:
        features.append(0.0)
    try:
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
    except:
        features.append(0.0)
    return np.array(features, dtype=np.float64)

def normalize_global(vec):
    std_safe = np.where(std_vector < 1e-8, 1, std_vector)
    return (vec - mean_vector) / std_safe

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_similar_audio(vector, top_k=3):
    all_docs = list(collection.find({}, {"file_path": 1, "duration_seconds": 1, "global_normalized_vector": 1}))
    similarities = []
    for doc in all_docs:
        db_vec = np.array(doc['global_normalized_vector'])
        sim = cosine_similarity(vector, db_vec)
        similarities.append({
            'file_path': doc['file_path'],
            'duration': doc['duration_seconds'],
            'similarity': sim
        })
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    return similarities[:top_k]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.endswith('.wav'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        input_cache_path = os.path.join(app.config['CACHE_FOLDER'], f"input_{filename}")
        shutil.copy2(filepath, input_cache_path)
        session['input_filename'] = f"input_{filename}"

        try:
            raw_vec = extract_features(filepath)
            norm_vec = normalize_global(raw_vec)
            similar_files = find_similar_audio(norm_vec)
            results = []
            for item in similar_files:
                file_path = item['file_path']
                file_name = os.path.basename(file_path)
                cache_path = os.path.join(app.config['CACHE_FOLDER'], file_name)
                try:
                    shutil.copy2(file_path, cache_path)
                except:
                    pass
                results.append({
                    'file_path': file_path,
                    'file_name': file_name,
                    'duration': item['duration'],
                    'similarity': item['similarity'],
                    'audio_url': f"/audio/{file_name}"
                })
            return render_template('index.html', 
                results=results, 
                input_filename=session.get('input_filename'),
                input_audio_url=f"/audio/{session.get('input_filename')}")
        except Exception as e:
            return f"Lỗi xử lý: {str(e)}"
    return redirect(request.url)

@app.route('/audio/<filename>')
def serve_audio(filename):
    cache_path = os.path.join(app.config['CACHE_FOLDER'], filename)
    if os.path.exists(cache_path):
        return send_file(cache_path, mimetype='audio/wav')
    return "File không tồn tại", 404

if __name__ == '__main__':
    app.run(debug=True)
