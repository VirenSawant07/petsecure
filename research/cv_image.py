"""Stratified GROUP 5-fold CV for the image models (near-duplicate groups never straddle folds).
 (a) custom CNN of notebook 02 on 48x48 grayscale, Adam, 10 epochs, batch 32 (as in the notebook)
 (b) notebook-03 head (GAP -> Dense256 ReLU -> Dense3 softmax) on frozen, correctly pre-processed ImageNet backbones."""
import os, sys, json, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf
from tensorflow.keras import layers, Sequential
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
D = f"{ROOT}/results/img"
y, groups = np.load(f"{D}/y.npy"), np.load(f"{D}/groups.npy")
S = f"{D}/cv_summary.json"
summary = json.load(open(S)) if os.path.exists(S) else {}

def cnn():                                               # notebook 02 build_cnn()
    return Sequential([layers.Input((48, 48, 1)),
        layers.Conv2D(64, 3, activation="relu"), layers.BatchNormalization(), layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(), layers.Dropout(0.25),
        layers.Conv2D(128, 3, activation="relu"), layers.BatchNormalization(), layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(), layers.Dropout(0.25),
        layers.Flatten(), layers.Dense(512, activation="relu"), layers.Dropout(0.5), layers.Dense(3, activation="softmax")])

def head(d):                                             # notebook 03 build_transfer_model() head
    return Sequential([layers.Input((d,)), layers.Dense(256, activation="relu"), layers.Dense(3, activation="softmax")])

def run(key, X, build, repeats, fit_kw, inner_val):
    rows, oof, hist = [], np.zeros((repeats, len(y), 3)), []
    for r in range(repeats):
        for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X, y, groups)):
            tf.keras.utils.set_random_seed(1000 * r + f)
            m = build(); m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            t = time.time()
            if inner_val:
                tr2, va = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=r)
                h = m.fit(X[tr2], y[tr2], validation_data=(X[va], y[va]), verbose=0, **fit_kw,
                          callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=8, restore_best_weights=True)])
            else:
                h = m.fit(X[tr], y[tr], validation_data=(X[te], y[te]), verbose=0, **fit_kw)   # notebook protocol, no selection
            p = m.predict(X[te], verbose=0); oof[r, te] = p; yp = p.argmax(1)
            P, R, F, _ = precision_recall_fscore_support(y[te], yp, average="macro", zero_division=0)
            rows.append(dict(repeat=r, fold=f, acc=accuracy_score(y[te], yp), prec=P, rec=R, f1=F,
                             auc=roc_auc_score(y[te], p, multi_class="ovr"), epochs_run=len(h.history["loss"]),
                             train_acc=float(m.evaluate(X[tr], y[tr], verbose=0)[1]), sec=time.time() - t, n_test=len(te)))
            hist.append({k: [float(v) for v in vs] for k, vs in h.history.items()})
            print(key, r, f, {k: round(v, 3) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
    summary[key] = dict(rows=rows, params=int(m.count_params()),
                        mean={k: float(np.mean([r[k] for r in rows])) for k in ["acc", "prec", "rec", "f1", "auc", "train_acc"]},
                        std={k: float(np.std([r[k] for r in rows], ddof=1)) for k in ["acc", "prec", "rec", "f1", "auc"]},
                        cm=[confusion_matrix(y, oof[i].argmax(1)).tolist() for i in range(repeats)])
    np.save(f"{D}/oof_{key}.npy", oof); json.dump(hist, open(f"{D}/hist_{key}.json", "w"))
    json.dump(summary, open(S, "w"), indent=1)
    print("==>", key, summary[key]["mean"], summary[key]["std"], flush=True)

jobs = sys.argv[1:] or ["MobileNetV2", "EfficientNetB0", "ResNet50", "VGG16", "CNN"]
for j in jobs:
    if j in summary: continue
    if j == "CNN":
        run("CNN", np.load(f"{D}/gray48.npy"), cnn, 1, dict(epochs=10, batch_size=32), inner_val=False)
    else:
        F = np.load(f"{D}/feat_{j}.npy")
        run(j, F, lambda: head(F.shape[1]), 3, dict(epochs=60, batch_size=32), inner_val=True)
