"""Trains the corrected image model from the paper (frozen ImageNet EfficientNetB0 + Dense-256 head)
on the notebook's 80:20 split and saves it as models/efficientnet_frozen_emotion.keras for the PetSecure app."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
DATA = os.path.join(ROOT, "data", "images")                        # see data/README.md
CLASSES = ["happy", "sad", "angry"]                        # label order of notebook 01

files, y = [], []
for k, c in enumerate(CLASSES):                            # same traversal as notebook 01
    for f in os.listdir(os.path.join(DATA, c)):
        files.append(os.path.join(DATA, c, f)); y.append(k)
y = np.array(y)
tr, te = train_test_split(np.arange(len(files)), test_size=0.2, random_state=42)

NB = os.path.join(ROOT, "processed_data")        # optional: the original notebooks' saved split, if you have it
if os.path.isdir(NB):
    Xte_nb = np.load(os.path.join(NB, "X_test.npy"))
    g = np.stack([cv2.resize(cv2.cvtColor(cv2.imread(files[i]), cv2.COLOR_BGR2GRAY), (48, 48)) for i in te]) / 255.0
    assert np.abs(g[..., None] - Xte_nb).max() < 1e-6 and (y[te] == np.load(os.path.join(NB, "y_test.npy"))).all()
print("split:", len(tr), "train /", len(te), "test")

def load(i):
    return np.asarray(Image.open(files[i]).convert("RGB").resize((224, 224), Image.BILINEAR), dtype="float32")

base = tf.keras.applications.EfficientNetB0(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
base.trainable = False
feats = base.predict(np.stack([load(i) for i in range(len(files))]), batch_size=64, verbose=0)   # EfficientNet rescales internally

tf.keras.utils.set_random_seed(42)
fit_i, val_i = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=42)
head_in = layers.Input((feats.shape[1],))
head_out = layers.Dense(3, activation="softmax")(layers.Dense(256, activation="relu")(head_in))
head = Model(head_in, head_out)
head.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
head.fit(feats[fit_i], y[fit_i], validation_data=(feats[val_i], y[val_i]), epochs=60, batch_size=32, verbose=0,
         callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=8, restore_best_weights=True)])
print("hold-out accuracy (750 test images): %.2f%%" % (100 * head.evaluate(feats[te], y[te], verbose=0)[1]))

inp = layers.Input((224, 224, 3))
full = Model(inp, head(base(inp, training=False)))
path = os.path.join(ROOT, "models", "efficientnet_frozen_emotion.keras")
full.save(path)
print("saved", path)
