"""Dataset statistics for the paper: audio durations/sample rates, image sizes, exact + near duplicates."""
import os, json, hashlib, collections
import numpy as np, soundfile as sf
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
A = f"{ROOT}/proj/audio_emotion_recognition_project/dataset"
I = f"{ROOT}/proj/pet_emotion_without_venv/notebooks/dataset"
out = {"audio": {}, "image": {}}

for c in sorted(os.listdir(A)):
    d, srs, ch = [], [], []
    for f in os.listdir(f"{A}/{c}"):
        info = sf.info(f"{A}/{c}/{f}")
        d.append(info.duration); srs.append(info.samplerate); ch.append(info.channels)
    out["audio"][c] = dict(n=len(d), dur_mean=float(np.mean(d)), dur_std=float(np.std(d)), dur_min=float(np.min(d)),
                           dur_max=float(np.max(d)), dur_total=float(np.sum(d)),
                           sr=dict(collections.Counter(srs)), channels=dict(collections.Counter(ch)))

def dhash(im, n=8):
    g = np.asarray(im.convert("L").resize((n + 1, n)), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()

md5, dh, labels, sizes = {}, [], [], collections.Counter()
for c in sorted(os.listdir(I)):
    ws, hs = [], []
    for f in sorted(os.listdir(f"{I}/{c}")):
        p = f"{I}/{c}/{f}"
        md5.setdefault(hashlib.md5(open(p, "rb").read()).hexdigest(), []).append(c)
        im = Image.open(p); ws.append(im.size[0]); hs.append(im.size[1]); sizes[im.mode] += 1
        dh.append(dhash(im)); labels.append(c)
    out["image"][c] = dict(n=len(ws), w_mean=float(np.mean(ws)), h_mean=float(np.mean(hs)),
                           w_min=int(min(ws)), w_max=int(max(ws)), h_min=int(min(hs)), h_max=int(max(hs)))
dup_groups = [v for v in md5.values() if len(v) > 1]
out["image"]["exact_dup_groups"] = len(dup_groups)
out["image"]["exact_dup_extra_files"] = sum(len(v) - 1 for v in dup_groups)
out["image"]["exact_dup_cross_class_groups"] = sum(len(set(v)) > 1 for v in dup_groups)
H = np.array(dh); L = np.array(labels)
dist = (H[:, None, :] != H[None, :, :]).sum(-1)            # 3750x3750 hamming, fine in memory (int)
iu = np.triu_indices(len(H), 1)
near = dist[iu] <= 4
out["image"]["near_dup_pairs_hamming_le4"] = int(near.sum())
out["image"]["near_dup_pairs_cross_class"] = int((L[iu[0]][near] != L[iu[1]][near]).sum())
out["image"]["modes"] = dict(sizes)
print(json.dumps(out, indent=1))
json.dump(out, open(f"{ROOT}/results/data_profile.json", "w"), indent=1)
