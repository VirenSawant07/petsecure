"""Repeated stratified 5-fold CV for the audio models (architectures/hyper-parameters from notebooks 02-04).
Early stopping / best-epoch selection uses an inner 15% validation split of the TRAINING folds only.
feature sets:  orig  = notebook-01 pipeline (22.05 kHz, 128 mels, power_to_db(ref=max), np.resize to 128x128)
               fixed = common 16 kHz bandwidth, fmax 8 kHz, fixed 3.0 s centre window -> true 128x128 log-mel"""
import os, sys, json, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, librosa, tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
A = f"{ROOT}/proj/audio_emotion_recognition_project"
OUT = f"{ROOT}/results/audio"; os.makedirs(OUT, exist_ok=True)

def load_files():
    files, labels = [], []
    for lab in os.listdir(f"{A}/dataset"):                       # same traversal as notebook 01
        for f in os.listdir(f"{A}/dataset/{lab}"):
            if f.endswith(".wav"): files.append(f"{A}/dataset/{lab}/{f}"); labels.append(lab)
    classes = sorted(set(labels))
    return files, np.array([classes.index(l) for l in labels]), classes

def feat_orig(p):
    a, sr = librosa.load(p, sr=22050)
    s = librosa.power_to_db(librosa.feature.melspectrogram(y=a, sr=sr), ref=np.max)
    return np.resize(s, (128, 128))

def feat_fixed(p, sr=16000, hop=375, T=128):
    a, _ = librosa.load(p, sr=sr)
    n = hop * (T - 1)
    a = a[(len(a) - n) // 2:][:n] if len(a) > n else np.pad(a, ((n - len(a)) // 2, n - len(a) - (n - len(a)) // 2))
    s = librosa.power_to_db(librosa.feature.melspectrogram(y=a, sr=sr, n_fft=1024, hop_length=hop, n_mels=128, fmax=8000), ref=np.max)
    return s[:, :T]

def cnn():
    return Sequential([layers.Input((128, 128, 1)),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(), layers.Dense(128, activation="relu"), layers.Dropout(0.5),
        layers.Dense(3, activation="softmax")])

def crnn():
    return Sequential([layers.Input((128, 128, 1)),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Reshape((-1, 64)), layers.GRU(64), layers.Dense(64, activation="relu"),
        layers.Dense(3, activation="softmax")])

def transformer():
    i = layers.Input((128, 128, 1))
    x = layers.Reshape((1, 256))(layers.Dense(256, activation="relu")(layers.Flatten()(i)))
    x = layers.LayerNormalization()(layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x))
    return Model(i, layers.Dense(3, activation="softmax")(layers.Flatten()(x)))

CFG = {  # (builder, optimizer, batch, epochs, early-stop patience or None, reduce-LR)
    "CNN": (cnn, lambda: tf.keras.optimizers.AdamW(1e-4, weight_decay=1e-5), 16, 100, 15, True),
    "CRNN": (crnn, lambda: tf.keras.optimizers.Adam(1e-3), 32, 75, None, False),
    "Transformer": (transformer, lambda: tf.keras.optimizers.Adam(1e-3), 32, 75, None, False),
}

class BestWeights(tf.keras.callbacks.Callback):
    """ModelCheckpoint(save_best_only, monitor=val_accuracy) semantics, kept in memory."""
    def on_train_begin(self, logs=None): self.best, self.w, self.epoch = -1, None, 0
    def on_epoch_end(self, e, logs=None):
        if logs["val_accuracy"] > self.best: self.best, self.w, self.epoch = logs["val_accuracy"], self.model.get_weights(), e + 1
    def on_train_end(self, logs=None): self.model.set_weights(self.w)

def run(name, X, y, repeats=3, k=5):
    build, opt, bs, ep, pat, rlr = CFG[name]
    rows, hist, oof = [], [], np.zeros((repeats, len(y), 3))
    for r in range(repeats):
        for f, (tr, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=r).split(X, y)):
            tf.keras.utils.set_random_seed(1000 * r + f)
            tr2, va = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=r)
            m = build(); m.compile(optimizer=opt(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            bw = BestWeights(); cb = [bw]
            if pat: cb.append(tf.keras.callbacks.EarlyStopping("val_accuracy", patience=pat, mode="max"))
            if rlr: cb.append(tf.keras.callbacks.ReduceLROnPlateau("val_accuracy", factor=0.5, patience=7, mode="max", min_lr=1e-7))
            t = time.time()
            h = m.fit(X[tr2], y[tr2], batch_size=bs, epochs=ep, validation_data=(X[va], y[va]), callbacks=cb, verbose=0)
            p = m.predict(X[te], verbose=0); oof[r, te] = p; yp = p.argmax(1)
            P, R, F, _ = precision_recall_fscore_support(y[te], yp, average="macro", zero_division=0)
            rows.append(dict(repeat=r, fold=f, acc=accuracy_score(y[te], yp), prec=P, rec=R, f1=F,
                             auc=roc_auc_score(y[te], p, multi_class="ovr"), best_epoch=bw.epoch,
                             epochs_run=len(h.history["loss"]), train_acc=float(m.evaluate(X[tr], y[tr], verbose=0)[1]),
                             sec=time.time() - t))
            hist.append({k2: [float(v) for v in vals] for k2, vals in h.history.items()})
            print(name, r, f, {k2: round(v, 3) for k2, v in rows[-1].items() if isinstance(v, float)}, flush=True)
    return rows, hist, oof, m.count_params()

if __name__ == "__main__":
    files, y, classes = load_files()
    print(classes, np.bincount(y), flush=True)
    Xo = np.stack([feat_orig(p) for p in files])[..., None]
    # sanity check: recomputed features must equal the team's processed_data (same split, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(Xo, y, test_size=0.2, random_state=42)
    P = f"{A}/processed_data"
    print("match processed_data:", np.abs(Xtr - np.load(f"{P}/X_train.npy")).max(), np.abs(Xte - np.load(f"{P}/X_test.npy")).max(),
          (ytr == np.load(f"{P}/y_train.npy")).all(), (yte == np.load(f"{P}/y_test.npy")).all(), flush=True)
    Xf = np.stack([feat_fixed(p) for p in files])[..., None]
    np.save(f"{OUT}/y.npy", y); np.save(f"{OUT}/X_orig.npy", Xo); np.save(f"{OUT}/X_fixed.npy", Xf)
    open(f"{OUT}/files.txt", "w").write("\n".join(os.path.relpath(p, A) for p in files))
    jobs = [("CNN", "orig"), ("CRNN", "orig"), ("Transformer", "orig"), ("CNN", "fixed"), ("CRNN", "fixed"), ("Transformer", "fixed")]
    S = f"{OUT}/cv_summary.json"
    summary = json.load(open(S)) if os.path.exists(S) else {}
    for name, fs in jobs:
        if f"{name}|{fs}" in summary: continue
        rows, hist, oof, npar = run(name, Xo if fs == "orig" else Xf, y)
        key = f"{name}|{fs}"
        summary[key] = dict(rows=rows, params=npar,
                            mean={m: float(np.mean([r[m] for r in rows])) for m in ["acc", "prec", "rec", "f1", "auc", "train_acc"]},
                            std={m: float(np.std([r[m] for r in rows], ddof=1)) for m in ["acc", "prec", "rec", "f1", "auc"]},
                            cm=[confusion_matrix(y, oof[r].argmax(1)).tolist() for r in range(len(oof))])
        np.save(f"{OUT}/oof_{name}_{fs}.npy", oof)
        json.dump(hist, open(f"{OUT}/hist_{name}_{fs}.json", "w"))
        json.dump(summary, open(f"{OUT}/cv_summary.json", "w"), indent=1)
        print("==>", key, summary[key]["mean"], summary[key]["std"], flush=True)
