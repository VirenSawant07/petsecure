"""Frozen ImageNet backbones on the original RGB images (224x224, model-specific preprocess_input).
Also writes the 48x48 grayscale tensor (notebook-01 preprocessing) and near-duplicate groups, in one fixed file order."""
import os, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, cv2, tensorflow as tf
from PIL import Image
from scipy.sparse.csgraph import connected_components

ROOT = os.path.dirname(os.path.abspath(__file__))
I = f"{ROOT}/proj/pet_emotion_without_venv/notebooks/dataset"
OUT = f"{ROOT}/results/img"; os.makedirs(OUT, exist_ok=True)
CLASSES = ["happy", "sad", "angry"]                     # label order used by notebook 01

files, y = [], []
for k, c in enumerate(CLASSES):
    for f in sorted(os.listdir(f"{I}/{c}")):
        files.append(f"{I}/{c}/{f}"); y.append(k)
y = np.array(y); np.save(f"{OUT}/y.npy", y)
open(f"{OUT}/files.txt", "w").write("\n".join(os.path.relpath(f, I) for f in files))

# notebook-01 tensor: cv2 gray, resize 48, /255
g = np.stack([cv2.resize(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY), (48, 48)) for f in files])
np.save(f"{OUT}/gray48.npy", (g / 255.0).astype("float32")[..., None])

# near-duplicate groups (dHash hamming <= 4) so duplicates never straddle CV folds
def dhash(p):
    a = np.asarray(Image.open(p).convert("L").resize((9, 8)), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()
H = np.array([dhash(f) for f in files])
adj = (H[:, None, :] != H[None, :, :]).sum(-1) <= 4
n, groups = connected_components(adj, directed=False)
np.save(f"{OUT}/groups.npy", groups); print("groups", n, "of", len(files), flush=True)

rgb = np.stack([np.asarray(Image.open(f).convert("RGB").resize((224, 224), Image.BILINEAR)) for f in files]).astype("float32")
A = tf.keras.applications
backbones = {
    "MobileNetV2": (A.MobileNetV2, A.mobilenet_v2.preprocess_input),
    "EfficientNetB0": (A.EfficientNetB0, A.efficientnet.preprocess_input),
    "ResNet50": (A.ResNet50, A.resnet50.preprocess_input),
    "VGG16": (A.VGG16, A.vgg16.preprocess_input),
}
for name, (ctor, prep) in backbones.items():
    t = time.time()
    m = ctor(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
    feats = m.predict(prep(rgb.copy()), batch_size=64, verbose=0)
    np.save(f"{OUT}/feat_{name}.npy", feats.astype("float32"))
    print(name, feats.shape, m.count_params(), f"{time.time()-t:.0f}s", flush=True)
