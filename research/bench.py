"""CPU inference latency (batch = 1, median of 50 after warm-up) and on-disk size of each model."""
import os, time, json, platform
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf, librosa
from tensorflow.keras import layers, Sequential

ROOT = os.path.dirname(os.path.abspath(__file__))
AM, IM = f"{ROOT}/proj/audio_emotion_recognition_project/models", f"{ROOT}/proj/pet_emotion_without_venv/models"

def lat(fn, x, n=50):
    for _ in range(5): fn(x)
    t = []
    for _ in range(n):
        s = time.perf_counter(); fn(x); t.append(time.perf_counter() - s)
    return float(np.median(t) * 1000)

res = {}
for name, path in [("audio CNN", f"{AM}/cnn_audio_emotion_best.h5"), ("audio CRNN", f"{AM}/crnn_audio_emotion_best.keras"),
                   ("audio attention", f"{AM}/transformer_audio_emotion_best.keras"), ("visual CNN", f"{IM}/facial_expression_model.h5")]:
    m = tf.keras.models.load_model(path, compile=False)
    x = np.random.rand(1, *m.input_shape[1:]).astype("float32")
    f = tf.function(lambda v: m(v, training=False))
    res[name] = dict(ms=lat(lambda v: f(v).numpy(), x), params=int(m.count_params()), mb=os.path.getsize(path) / 2**20)
    print(name, res[name], flush=True)
A = tf.keras.applications
for name, ctor in [("MobileNetV2", A.MobileNetV2), ("EfficientNetB0", A.EfficientNetB0), ("ResNet50", A.ResNet50), ("VGG16", A.VGG16)]:
    base = ctor(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    m = Sequential([base, layers.Dense(256, activation="relu"), layers.Dense(3, activation="softmax")])
    x = np.random.rand(1, 224, 224, 3).astype("float32") * 255
    f = tf.function(lambda v: m(v, training=False))
    res[name] = dict(ms=lat(lambda v: f(v).numpy(), x), params=int(m.count_params()), mb=m.count_params() * 4 / 2**20)
    print(name, res[name], flush=True)
wav = f"{ROOT}/proj/audio_emotion_recognition_project/dataset/angry/Growl2.wav"
def feat(p):
    a, sr = librosa.load(p, sr=22050)
    return np.resize(librosa.power_to_db(librosa.feature.melspectrogram(y=a, sr=sr), ref=np.max), (128, 128))
res["audio preprocessing"] = dict(ms=lat(feat, wav, 20))
res["_cpu"] = platform.processor(); res["_tf"] = tf.__version__
json.dump(res, open(f"{ROOT}/results/latency.json", "w"), indent=1)
print(res["audio preprocessing"], res["_cpu"])
