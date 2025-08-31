import os
import librosa
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Paths and parameters
WAV_DIR = "./sm_04_wav"
LAB_DIR = "./sm_04_lab"
SAMPLE_RATE = 16000
N_MFCC = 13

features = []
labels = []
i = 0
# Loop through .wav and .lab files
for file in os.listdir(WAV_DIR):
    if file.endswith(".wav"):
        base = os.path.splitext(file)[0]
        wav_path = os.path.join(WAV_DIR, base + ".wav")
        lab_path = os.path.join(LAB_DIR, base + ".lab")

        if not os.path.exists(lab_path):
            print(f"[!] Missing .lab for {base}")
            continue

        # Load audio
        y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
        i = i +1
        # Process .lab file
        with open(lab_path, "r") as f:
            for line in f:
                try:
                    start_raw, end_raw, phoneme = line.strip().split()
                    phoneme = phoneme.lower()

                    start_sec = int(start_raw) * 1e-7
                    end_sec = int(end_raw) * 1e-7
                    start_sample = int(start_sec * sr)
                    end_sample = int(end_sec * sr)

                    segment = y[start_sample:end_sample]

                    if len(segment) < 512:
                        continue
                    if np.max(np.abs(segment)) < 0.01:
                        continue

                    # MFCC extraction
                    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC, n_fft=512, hop_length=128)

                    if mfcc.shape[1] < 9:
                        continue

                    delta = librosa.feature.delta(mfcc)
                    delta2 = librosa.feature.delta(mfcc, order=2)
                    combined = np.vstack([mfcc, delta, delta2])
                    avg_vector = np.mean(combined, axis=1)

                    features.append(avg_vector)
                    labels.append(phoneme)

                except Exception as e:
                    print(f"[!] Error in {base}: {e}")
                    continue
print(i)

from sklearn.preprocessing import StandardScaler

# 3. Statistics before filtering
print(f"\nTotal phoneme samples: {len(features)}")
print(f"Number of unique phonemes: {len(set(labels))}")
print("Most common phonemes:")
for phon, count in Counter(labels).most_common(10):
    print(f"  {phon}: {count}")

# === POBOLJSANJA ===

long_to_short = {
    "a:": "a",
    "e:": "e",
    "i:": "i",
    "o:": "o",
    "u:": "u",
    "r:": "r"
}
labels_mapped = [long_to_short.get(l, l) for l in labels]

counts = Counter(labels_mapped)
valid_phonemes = {phon for phon, count in counts.items() if count >= 50}

filtered = [(f, l) for f, l in zip(features, labels_mapped) if l in valid_phonemes]
features_filtered, labels_filtered = zip(*filtered)

X = np.array(features_filtered)
y = np.array(labels_filtered)

print("features_filtered", len(features_filtered))
print("labels_filtered", len(labels_filtered))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("X_train", len(X_train))
print("X_test", len(X_test))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\n=== Classification report ===")
print(classification_report(y_test, y_pred, zero_division=0))

labels_sorted = sorted(set(y))
cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, xticklabels=labels_sorted, yticklabels=labels_sorted,
            annot=False, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Bayes Classifier")
plt.tight_layout()
plt.show()
