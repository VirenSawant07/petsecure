"""Calibration (ECE, Brier), app confidence tiers (app3.py thresholds) and per-class metrics on out-of-fold predictions."""
import os, json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

ROOT = os.path.dirname(os.path.abspath(__file__))
R = f"{ROOT}/results"

def ece(p, y, bins=10):
    conf, pred = p.max(1), p.argmax(1); e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)

def tier(p):                                    # app3.py confidence_label()
    s = np.sort(p, 1)[:, ::-1]; top, margin = s[:, 0], s[:, 0] - s[:, 1]
    return np.where((top >= 0.85) & (margin >= 0.35), "High", np.where((top >= 0.65) & (margin >= 0.2), "Moderate", "Low"))

def entropy(p):                                 # app3.py normalized_entropy()
    q = np.clip(p, 1e-12, 1); q = q / q.sum(1, keepdims=True)
    return -(q * np.log(q)).sum(1) / np.log(q.shape[1])

def analyse(P, y, classes):
    out = dict(ece=[], brier=[], tiers={}, per_class={})
    for r in range(P.shape[0]):
        p = P[r]; out["ece"].append(ece(p, y)); out["brier"].append(float(((p - np.eye(3)[y]) ** 2).sum(1).mean()))
    p = P.reshape(-1, 3); yy = np.tile(y, P.shape[0]); ok = p.argmax(1) == yy; t = tier(p)
    for name in ["High", "Moderate", "Low"]:
        m = t == name
        out["tiers"][name] = dict(share=float(m.mean()), acc=float(ok[m].mean()) if m.any() else None)
    H = entropy(p); out["entropy_correct"] = float(H[ok].mean()); out["entropy_wrong"] = float(H[~ok].mean())
    Pc, Rc, Fc, _ = precision_recall_fscore_support(yy, p.argmax(1), average=None)
    out["per_class"] = {c: dict(prec=float(a), rec=float(b), f1=float(f)) for c, a, b, f in zip(classes, Pc, Rc, Fc)}
    out["ece"] = float(np.mean(out["ece"])); out["brier"] = float(np.mean(out["brier"]))
    return out

res = {}
ya = np.load(f"{R}/audio/y.npy"); yi = np.load(f"{R}/img/y.npy")
for k in ["CNN_orig", "CRNN_orig", "Transformer_orig", "CNN_fixed", "CRNN_fixed"]:
    f = f"{R}/audio/oof_{k}.npy"
    if os.path.exists(f): res[f"audio {k}"] = analyse(np.load(f), ya, ["angry", "happy", "sad"])
for k in ["MobileNetV2", "EfficientNetB0", "ResNet50", "VGG16", "CNN"]:
    f = f"{R}/img/oof_{k}.npy"
    if os.path.exists(f): res[f"image {k}"] = analyse(np.load(f), yi, ["happy", "sad", "angry"])
json.dump(res, open(f"{R}/analysis.json", "w"), indent=1)
for k, v in res.items():
    print(k, "ECE %.3f Brier %.3f" % (v["ece"], v["brier"]), {t: (round(d["share"], 3), None if d["acc"] is None else round(d["acc"], 3)) for t, d in v["tiers"].items()},
          "H ok/wrong %.3f/%.3f" % (v["entropy_correct"], v["entropy_wrong"]))
