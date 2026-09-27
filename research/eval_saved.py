"""Evaluate the project's saved models on their original held-out test splits.
Preprocessing replicates each training notebook exactly (including its quirks)."""
import os, sys, json, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf, cv2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)
which = sys.argv[1]

def metrics(y, p):
    yp = p.argmax(1)
    P, R, F, _ = precision_recall_fscore_support(y, yp, average="macro", zero_division=0)
    Pc, Rc, Fc, S = precision_recall_fscore_support(y, yp, average=None, zero_division=0)
    try: auc = roc_auc_score(y, p, multi_class="ovr")
    except ValueError: auc = float("nan")
    return dict(acc=accuracy_score(y, yp), prec=P, rec=R, f1=F, auc=auc,
                per_class=dict(prec=Pc.tolist(), rec=Rc.tolist(), f1=Fc.tolist(), support=S.tolist()),
                cm=confusion_matrix(y, yp).tolist(), mean_conf=float(p.max(1).mean()))

res = {}
if which == "audio":
    A = os.path.join(ROOT, "proj/audio_emotion_recognition_project")
    X = np.load(f"{A}/processed_data/X_test.npy"); y = np.load(f"{A}/processed_data/y_test.npy")
    names = ["cnn_audio_emotion.h5", "cnn_audio_emotion_best.h5", "cnn_audio_emotion_final.h5",
             "crnn_audio_emotion.h5", "crnn_audio_emotion_best.h5", "crnn_audio_emotion_final.h5",
             "crnn_audio_emotion_best.keras", "crnn_audio_emotion_final.keras",
             "transformer_audio_emotion_best.keras", "transformer_audio_emotion_final.keras"]
    for n in names:
        m = tf.keras.models.load_model(f"{A}/models/{n}", compile=False)
        p = m.predict(X, verbose=0)[:, :3]          # CNN heads have 5 units; app uses first 3 (as in app3.py)
        p = p / p.sum(1, keepdims=True)
        res[n] = metrics(y, p) | dict(params=int(m.count_params()), out_units=int(m.output_shape[-1]))
        np.save(f"{OUT}/probs_audio_{n}.npy", p)
        print(n, {k: round(v, 4) for k, v in res[n].items() if isinstance(v, float)}, res[n]["cm"])
    np.save(f"{OUT}/y_audio_test.npy", y)
else:
    I = os.path.join(ROOT, "proj/pet_emotion_without_venv")
    X = np.load(f"{I}/processed_data/X_test.npy"); y = np.load(f"{I}/processed_data/y_test.npy")
    m = tf.keras.models.load_model(f"{I}/models/facial_expression_model.h5", compile=False)
    p = m.predict(X.astype("float32"), verbose=0)
    res["facial_expression_model.h5"] = metrics(y, p) | dict(params=int(m.count_params()))
    np.save(f"{OUT}/probs_img_cnn.npy", p); print("cnn", res["facial_expression_model.h5"]["acc"], flush=True)
    # saved transfer models take 48x48x3. "infer" = notebook 03.5 path (already /255, gray repeated to RGB);
    # "double" = notebook 03 path (divided by 255 a second time).
    X3 = X.astype("float32").repeat(3, -1)
    for n in ["resnet50_model.h5", "vgg16_model.h5", "mobilenet_model.h5", "efficientnet_model.h5"]:
        t = time.time()
        m = tf.keras.models.load_model(f"{I}/models/{n}", compile=False)
        for tag, Xin in [("infer", X3), ("double", X3 / 255.0)]:
            p = m.predict(Xin, batch_size=64, verbose=0)
            res[f"{n}|{tag}"] = metrics(y, p) | dict(params=int(m.count_params()), base=m.layers[1].name)
            np.save(f"{OUT}/probs_img_{n}_{tag}.npy", p)
            print(n, tag, round(res[f"{n}|{tag}"]["acc"], 4), res[f"{n}|{tag}"]["cm"],
                  "mean_conf", round(float(p.max(1).mean()), 3), f"{time.time()-t:.0f}s", flush=True)
    np.save(f"{OUT}/y_img_test.npy", y)
json.dump(res, open(f"{OUT}/saved_{which}.json", "w"), indent=1)
