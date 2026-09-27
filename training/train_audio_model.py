"""Trains the audio model recommended in the paper (CRNN on the corrected log-mel front-end: 16 kHz, 3.0 s centre window)
on the notebook's 90-clip training split and saves it as models/crnn_corrected_emotion.keras."""
import os
import json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
DATA = os.path.join(ROOT, "data", "audio")                        # see data/README.md


def features(path, sr=16000, hop=375, T=128):
    """Corrected front-end (identical to the one used in the paper's cross-validation)."""
    a, _ = librosa.load(path, sr=sr)
    n = hop * (T - 1)
    a = a[(len(a) - n) // 2:][:n] if len(a) > n else np.pad(a, ((n - len(a)) // 2, n - len(a) - (n - len(a)) // 2))
    s = librosa.power_to_db(librosa.feature.melspectrogram(y=a, sr=sr, n_fft=1024, hop_length=hop, n_mels=128, fmax=8000), ref=np.max)
    return s[:, :T]


files, labels = [], []
for lab in os.listdir(DATA):                                  # same traversal as notebook 01
    for f in os.listdir(os.path.join(DATA, lab)):
        if f.endswith(".wav"):
            files.append(os.path.join(DATA, lab, f)); labels.append(lab)
classes = sorted(set(labels))                                 # ['angry', 'happy', 'sad'] as in classes.npy
y = np.array([classes.index(lab) for lab in labels])
X = np.stack([features(p) for p in files])[..., None]
tr, te = train_test_split(np.arange(len(files)), test_size=0.2, random_state=42)
if os.path.isdir(os.path.join(ROOT, "processed_data")):   # optional check: same split as the original notebooks
    assert (y[te] == np.load(os.path.join(ROOT, "processed_data", "y_test.npy"))).all()

tf.keras.utils.set_random_seed(42)
fit_i, val_i = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=42)
model = Sequential([layers.Input((128, 128, 1)),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
    layers.Reshape((-1, 64)), layers.GRU(64), layers.Dense(64, activation="relu"), layers.Dense(3, activation="softmax")])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
best = {"acc": -1, "w": None}


class KeepBest(tf.keras.callbacks.Callback):                  # best validation-accuracy epoch, as in the paper
    def on_epoch_end(self, epoch, logs=None):
        if logs["val_accuracy"] > best["acc"]:
            best["acc"], best["w"] = logs["val_accuracy"], self.model.get_weights()


model.fit(X[fit_i], y[fit_i], validation_data=(X[val_i], y[val_i]), epochs=75, batch_size=32, verbose=0, callbacks=[KeepBest()])
model.set_weights(best["w"])
correct = int((model.predict(X[te], verbose=0).argmax(1) == y[te]).sum())
print(f"hold-out accuracy (23 test clips): {correct}/{len(te)} = {100 * correct / len(te):.1f}%")
model.save(os.path.join(ROOT, "models", "crnn_corrected_emotion.keras"))
json.dump({"model": "models/crnn_corrected_emotion.keras", "classes": classes, "holdout_correct": correct, "holdout_total": len(te)},
          open(os.path.join(ROOT, "models", "crnn_corrected_emotion.json"), "w"), indent=1)
print("saved models/crnn_corrected_emotion.keras")
