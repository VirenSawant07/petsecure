"""Recover the notebook-01 80/20 hold-out split (by exact 48x48 pixel match) and evaluate the frozen-backbone
heads on exactly the same 750 test images used by the saved models."""
import os, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf
from tensorflow.keras import layers, Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
D, P = f"{ROOT}/results/img", f"{ROOT}/proj/pet_emotion_without_venv/processed_data"
y = np.load(f"{D}/y.npy")
key = lambda a: [hash(np.rint(im * 255).astype(np.uint8).tobytes()) for im in a]
idx = {}
for i, k in enumerate(key(np.load(f"{D}/gray48.npy"))): idx.setdefault(k, []).append(i)
def locate(arr):
    out = []
    for k in key(arr):
        out.append(idx[k].pop(0) if k in idx and idx[k] else -1)
    return np.array(out)
te = locate(np.load(f"{P}/X_test.npy")); tr = locate(np.load(f"{P}/X_train.npy"))
assert (te >= 0).all() and (tr >= 0).all(), "unmatched images"
assert (y[te] == np.load(f"{P}/y_test.npy")).all() and (y[tr] == np.load(f"{P}/y_train.npy")).all()
np.save(f"{D}/holdout_train_idx.npy", tr); np.save(f"{D}/holdout_test_idx.npy", te)
print("split recovered:", len(tr), len(te))

res = {}
for name in ["MobileNetV2", "EfficientNetB0", "ResNet50", "VGG16"]:
    F = np.load(f"{D}/feat_{name}.npy")
    tf.keras.utils.set_random_seed(42)
    tr2, va = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=42)
    m = Sequential([layers.Input((F.shape[1],)), layers.Dense(256, activation="relu"), layers.Dense(3, activation="softmax")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = m.fit(F[tr2], y[tr2], validation_data=(F[va], y[va]), epochs=60, batch_size=32, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=8, restore_best_weights=True)])
    p = m.predict(F[te], verbose=0); yp = p.argmax(1)
    Pm, Rm, Fm, _ = precision_recall_fscore_support(y[te], yp, average="macro")
    Pc, Rc, Fc, _ = precision_recall_fscore_support(y[te], yp, average=None)
    res[name] = dict(acc=accuracy_score(y[te], yp), prec=Pm, rec=Rm, f1=Fm, auc=roc_auc_score(y[te], p, multi_class="ovr"),
                     cm=confusion_matrix(y[te], yp).tolist(), per_class_f1=Fc.tolist(), per_class_rec=Rc.tolist(),
                     head_params=int(m.count_params()), epochs=len(h.history["loss"]), history=h.history)
    np.save(f"{D}/holdout_probs_{name}.npy", p)
    print(name, {k: round(v, 4) for k, v in res[name].items() if isinstance(v, float)}, res[name]["cm"], flush=True)
json.dump(res, open(f"{D}/holdout_frozen.json", "w"), indent=1, default=float)
