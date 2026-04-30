import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_PATH = "/home/gedex/Aiproject/Ai-Based-Lie-Detection-main"
TRUTH_PATH = os.path.join(BASE_PATH, "Dataset", "truth")
LIES_PATH = os.path.join(BASE_PATH, "Dataset", "lies")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "audio_model.pkl")

def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=3)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        
        if len(pitch_values) > 0:
            pitch_mean = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
            pitch_max = np.max(pitch_values)
            pitch_min = np.min(pitch_values)
        else:
            pitch_mean = pitch_std = pitch_max = pitch_min = 0
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        energy = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
        
        features = np.concatenate([
            [pitch_mean, pitch_std, pitch_max, pitch_min, energy, zcr],
            mfcc_means, mfcc_stds
        ])
        return features
    except Exception as e:
        print(f"Error: {e}")
        return None

print("Loading audio files...")
X, y = [], []

for filename in os.listdir(TRUTH_PATH):
    if filename.endswith(".wav"):
        features = extract_features(os.path.join(TRUTH_PATH, filename))
        if features is not None:
            X.append(features)
            y.append(0)
            print(f"  Truth: {filename}")

for filename in os.listdir(LIES_PATH):
    if filename.endswith(".wav"):
        features = extract_features(os.path.join(LIES_PATH, filename))
        if features is not None:
            X.append(features)
            y.append(1)
            print(f"  Lie: {filename}")

X = np.array(X)
y = np.array(y)
print(f"\nTotal: {len(X)} (Truth: {sum(y==0)}, Lie: {sum(y==1)})")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.2%}")

joblib.dump(model, MODEL_SAVE_PATH)
print(f"✅ Model saved to: {MODEL_SAVE_PATH}")
print("Done!")
