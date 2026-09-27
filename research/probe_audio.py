"""Shortcut probe: predict the audio emotion label from recording metadata only (no acoustic content)."""
import json, numpy as np, soundfile as sf
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
A = "proj/audio_emotion_recognition_project/"
files = open("results/audio/files.txt").read().split("\n"); y = np.load("results/audio/y.npy")
info = [sf.info(A + f) for f in files]
X = np.array([[i.samplerate, i.duration] for i in info])
out = {}
for name, cols in [("samplerate", [0]), ("duration", [1]), ("samplerate+duration", [0, 1])]:
    s = [cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=0), X[:, cols], y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=r)).mean() for r in range(3)]
    out[name] = dict(mean=float(np.mean(s)), std=float(np.std(s)))
    print(name, round(np.mean(s), 4))
json.dump(out, open("results/audio/probe.json", "w"), indent=1)
