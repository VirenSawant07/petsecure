"""Dataset / preprocessing / notebook-training-curve figures (single-column 3.5 in unless noted)."""
import os, json, random
import numpy as np, soundfile as sf, librosa, librosa.display
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = f"{ROOT}/figs"; os.makedirs(FIG, exist_ok=True)
A = f"{ROOT}/proj/audio_emotion_recognition_project"
I = f"{ROOT}/proj/pet_emotion_without_venv/notebooks/dataset"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 7, "axes.titlesize": 7.2,
                     "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.5, "axes.axisbelow": True, "savefig.dpi": 300, "legend.frameon": False})
CLS = ["angry", "happy", "sad"]
COL = {"angry": "#eb6834", "happy": "#1baf7a", "sad": "#2a78d6"}
INK2 = "#52514e"

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ------------------------------------------------ dataset overview
prof = json.load(open(f"{ROOT}/results/data_profile.json"))
files = open(f"{ROOT}/results/audio/files.txt").read().replace("\\", "/").split(); ya = np.load(f"{ROOT}/results/audio/y.npy")
info = [sf.info(f"{A}/{f}") for f in files]
sr = np.array([i.samplerate for i in info]); dur = np.array([i.duration for i in info])
fig = plt.figure(figsize=(3.5, 3.5)); gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.45, height_ratios=[1.05, 1])
a0, a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
b = a0.bar(CLS, [prof["image"][c]["n"] for c in CLS], color=[COL[c] for c in CLS], width=0.6)
a0.bar_label(b, fontsize=6.2, padding=1.5); a0.set_ylim(0, 1500); a0.set_title("(a) Images (3,750)"); a0.set_ylabel("Images")
a0.grid(axis="x", visible=False)
rates = [11025, 16000, 22050, 44100, 48000]; ramp = ["#cfe0f5", "#9cc0ea", "#5f97d9", "#2a6cc0", "#153f7a"]
bottom = np.zeros(3)
for rt, c in zip(rates, ramp):
    v = np.array([np.sum((ya == k) & (sr == rt)) for k in range(3)])
    a1.bar(CLS, v, bottom=bottom, color=c, width=0.6, edgecolor="white", linewidth=0.7, label=f"{rt/1000:g}k")
    bottom += v
for k in range(3): a1.text(k, bottom[k] + 1, str(int(bottom[k])), ha="center", fontsize=6.2)
a1.set_ylim(0, 78); a1.set_title("(b) Audio clips by sampling rate"); a1.set_ylabel("Clips"); a1.grid(axis="x", visible=False)
a1.legend(loc="upper center", ncol=3, fontsize=5.5, handlelength=0.8, columnspacing=0.5, borderaxespad=0, handletextpad=0.3)
rng = np.random.default_rng(0)
for k, c in enumerate(CLS):
    d = dur[ya == k]
    a2.boxplot(d, positions=[k], widths=0.4, showfliers=False, medianprops=dict(color="#0b0b0b", lw=1),
               boxprops=dict(color=INK2, lw=0.7), whiskerprops=dict(color=INK2, lw=0.7), capprops=dict(color=INK2, lw=0.7))
    a2.scatter(k + rng.uniform(-0.14, 0.14, len(d)), d, s=8, color=COL[c], alpha=0.85, lw=0.4, edgecolor="white", zorder=3)
a2.set_yscale("log"); a2.yaxis.set_minor_formatter(mt.NullFormatter()); a2.set_yticks([2, 3, 5, 10, 20, 50]); a2.set_yticklabels(["2", "3", "5", "10", "20", "50"])
a2.set_xticks(range(3)); a2.set_xticklabels(CLS); a2.set_ylabel("Duration (s, log)"); a2.set_title("(c) Audio clip duration"); a2.grid(axis="x", visible=False)
save(fig, "fig_dataset_overview")

# ------------------------------------------------ sample images (3 x 5)
random.seed(7)
fig, axs = plt.subplots(3, 5, figsize=(3.5, 2.2), gridspec_kw=dict(wspace=0.04, hspace=0.05))
for r, c in enumerate(CLS):
    for j, f in enumerate(random.sample(sorted(os.listdir(f"{I}/{c}")), 5)):
        im = Image.open(f"{I}/{c}/{f}").convert("RGB"); s = min(im.size)
        im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2)).resize((200, 200))
        axs[r, j].imshow(im); axs[r, j].set_xticks([]); axs[r, j].set_yticks([]); axs[r, j].grid(False)
        for sp in axs[r, j].spines.values(): sp.set_visible(False)
    axs[r, 0].set_ylabel(c, fontsize=7, labelpad=3)
save(fig, "fig_sample_images")

# ------------------------------------------------ audio examples: per class waveform | log-mel
pick = {"angry": "dataset/angry/Growl2.wav", "happy": "dataset/happy/dog_3.wav", "sad": "dataset/sad/Dog5.wav"}
fig, axs = plt.subplots(3, 2, figsize=(3.5, 3.5), gridspec_kw=dict(width_ratios=[1, 1.25], hspace=0.55, wspace=0.32))
for i, c in enumerate(CLS):
    y, s = librosa.load(f"{A}/{pick[c]}", sr=22050); t = np.arange(len(y)) / s
    axs[i, 0].plot(t, y, color=COL[c], lw=0.4); axs[i, 0].set_xlim(0, t[-1]); axs[i, 0].set_ylabel(c, fontsize=7)
    axs[i, 0].set_title(f"{os.path.basename(pick[c])}: waveform", fontsize=6.5)
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=s), ref=np.max)
    librosa.display.specshow(S, sr=s, x_axis="time", y_axis="mel", ax=axs[i, 1], cmap="magma")
    axs[i, 1].set_title("log-mel (dB)", fontsize=6.5); axs[i, 1].set_ylabel("Hz", fontsize=6.3); axs[i, 1].grid(False)
    axs[i, 1].tick_params(labelsize=5.8); axs[i, 0].tick_params(labelsize=5.8)
    for a in axs[i]: a.set_xlabel("Time (s)" if i == 2 else "", fontsize=6.5)
save(fig, "fig_audio_examples")

# ------------------------------------------------ preprocessing illustration (full width)
f = "dataset/happy/dog_3.wav"
y, s = librosa.load(f"{A}/{f}", sr=22050)
S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=s), ref=np.max)
Xo = np.load(f"{ROOT}/results/audio/X_orig.npy"); Xf = np.load(f"{ROOT}/results/audio/X_fixed.npy")
k = files.index(f)
fig, axs = plt.subplots(1, 3, figsize=(7.16, 1.8), gridspec_kw=dict(width_ratios=[S.shape[1] / 128 * 0.9, 1, 1], wspace=0.35))
axs[0].imshow(S, origin="lower", aspect="auto", cmap="magma", extent=[0, len(y) / s, 0, 128])
axs[0].set_title(f"(a) Log-mel of a {len(y)/s:.1f} s clip"); axs[0].set_xlabel("Time (s)"); axs[0].set_ylabel("Mel band")
axs[1].imshow(Xo[k, ..., 0], origin="lower", aspect="auto", cmap="magma"); axs[1].set_title("(b) Original: np.resize to 128×128")
axs[1].set_xlabel("Column index"); axs[1].set_ylabel("Row index")
axs[2].imshow(Xf[k, ..., 0], origin="lower", aspect="auto", cmap="magma", extent=[0, 3.0, 0, 128])
axs[2].set_title("(c) Corrected: 3.0 s window, 16 kHz"); axs[2].set_xlabel("Time (s)"); axs[2].set_ylabel("Mel band")
for a in axs: a.grid(False)
save(fig, "fig_preprocessing")

# ------------------------------------------------ notebook training curves (2 x 2)
L = json.load(open(f"{ROOT}/results/notebook_logs.json"))
panels = [("audio_cnn_final", "(a) Audio CNN"), ("audio_crnn_best", "(b) Audio CRNN"),
          ("audio_transformer", "(c) Audio attention encoder"), ("image_cnn", "(d) Visual CNN")]
fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.0), gridspec_kw=dict(wspace=0.28, hspace=0.55))
for a, (k, title) in zip(axs.flat, panels):
    d = L[k]; e = np.arange(1, len(d["acc"]) + 1)
    a.plot(e, d["acc"], color="#2a78d6", lw=1.1, label="training")
    a.plot(e, d["val_acc"], color="#eb6834", lw=1.1, label="hold-out")
    bi = int(np.argmax(d["val_acc"]))
    a.scatter([e[bi]], [d["val_acc"][bi]], s=12, color="#eb6834", edgecolor="white", lw=0.7, zorder=4)
    a.annotate(f"{d['val_acc'][bi]*100:.1f}%", (e[bi], d["val_acc"][bi]), xytext=(0, 5), textcoords="offset points", fontsize=5.8,
               ha="center", bbox=dict(fc="white", ec="none", pad=0.5))
    a.set_title(title); a.set_ylim(0.15, 1.08); a.axhline(1 / 3, color="#9a9990", lw=0.6, ls=(0, (3, 2)))
    a.set_xlabel("Epoch")
axs[0, 0].set_ylabel("Accuracy"); axs[1, 0].set_ylabel("Accuracy"); axs[0, 0].legend(loc="lower right", fontsize=5.8, handlelength=1.2)
save(fig, "fig_training_curves")
print("ok")
