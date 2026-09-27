"""Block diagrams for the paper (system architecture, model architectures, CV protocol)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = f"{ROOT}/figs"; os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8, "savefig.dpi": 300})

C = dict(inp="#E8EEF7", pre="#DCEBDD", mod="#FBE7C6", out="#F3DCE5", sto="#E6E1F2", edge="#3A3F4B", ink="#1B1F27", mute="#5B6170")

def box(ax, x, y, w, h, text, fc, fs=6.6, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008", fc=fc, ec=C["edge"], lw=0.8))
    ax.text(x + w / 2 if align == "center" else x + 0.012, y + h / 2, text, ha=align, va="center", fontsize=fs,
            color=C["ink"], linespacing=1.15)

def arrow(ax, p, q, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=C["edge"],
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

def lane(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.01", fc="none",
                                ec="#B8BDC7", lw=0.7, ls=(0, (3, 2))))
    ax.text(x + 0.008, y + h - 0.015, title, ha="left", va="top", fontsize=7, color=C["mute"], style="italic")

# ---------------------------------------------------------------- system architecture
fig, ax = plt.subplots(figsize=(7.16, 2.85)); fig.subplots_adjust(0, 0, 1, 1); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
lane(ax, 0.0, 0.50, 1.0, 0.49, "Online inference path")
lane(ax, 0.0, 0.0, 1.0, 0.46, "Offline training & validation path")
X = [0.012, 0.158, 0.37, 0.565, 0.735]; Wd = [0.126, 0.19, 0.175, 0.15, 0.255]
box(ax, X[0], 0.765, Wd[0], 0.15, "Pet image\n(JPEG / PNG\nupload)", C["inp"])
box(ax, X[0], 0.545, Wd[0], 0.15, "Pet vocalization\n(.wav upload)", C["inp"])
box(ax, X[1], 0.765, Wd[1], 0.15, "RGB resize to 224×224,\nImageNet normalization\n(gray 48×48 for CNN)", C["pre"])
box(ax, X[1], 0.545, Wd[1], 0.15, "Resample 22.05 kHz, STFT,\n128-band mel, power→dB,\n128×128 tensor", C["pre"])
box(ax, X[2], 0.765, Wd[2], 0.15, "Visual branch\nfrozen CNN backbone\n+ dense head", C["mod"])
box(ax, X[2], 0.545, Wd[2], 0.15, "Acoustic branch\nspectrogram CNN /\nCRNN / attention", C["mod"])
box(ax, X[3], 0.655, Wd[3], 0.15, "Softmax posterior\np = [p_angry, p_happy,\np_sad] per modality", C["out"])
box(ax, X[4], 0.755, Wd[4], 0.205, "Confidence analytics\n• top-1 confidence p(1)\n• top-2 margin Δ = p(1) − p(2)\n• normalized entropy U\n• tier: High / Moderate / Low", C["out"], align="left")
box(ax, X[4], 0.545, Wd[4], 0.18, "Report & dashboard (Streamlit)\n• ranking chart, waveform, mel view\n• TXT / JSON / CSV / MD export\n• session history of reports", C["out"], align="left")
for y in (0.84, 0.62):
    arrow(ax, (X[0] + Wd[0], y), (X[1], y)); arrow(ax, (X[1] + Wd[1], y), (X[2], y))
arrow(ax, (X[2] + Wd[2], 0.84), (X[3], 0.765)); arrow(ax, (X[2] + Wd[2], 0.62), (X[3], 0.695))
arrow(ax, (X[3] + Wd[3], 0.745), (X[4], 0.84)); arrow(ax, (X[3] + Wd[3], 0.715), (X[4], 0.635))
Y0, Hh = 0.19, 0.17
box(ax, X[0], Y0, Wd[0], Hh, "Labelled corpora\n3,750 images,\n113 vocalizations\n(angry/happy/sad)", C["sto"])
box(ax, X[1], Y0, Wd[1], Hh, "Pre-processing,\nnear-duplicate grouping\n(dHash), feature caching", C["pre"])
box(ax, X[2], Y0, Wd[2], Hh, "Training: Adam / AdamW,\nsparse categorical\ncross-entropy", C["mod"])
box(ax, X[3], Y0, Wd[3], Hh, "Repeated stratified\n(group) 5-fold CV,\ninner 15 % validation", C["mod"])
box(ax, X[4], Y0, Wd[4], Hh, "Model store (.h5 / .keras)\n+ class map (classes.npy)", C["sto"])
for i in range(4): arrow(ax, (X[i] + Wd[i], Y0 + Hh / 2), (X[i + 1], Y0 + Hh / 2))
arrow(ax, (X[4] + Wd[4] / 2, Y0 + Hh), (X[4] + Wd[4] / 2, 0.545))
ax.text(X[4] + Wd[4] / 2 - 0.01, 0.48, "deploy selected checkpoint", fontsize=6.3, color=C["mute"], va="center", ha="right")
arrow(ax, (X[3] + Wd[3] / 2, Y0), (X[2] + Wd[2] / 2, Y0), rad=-0.5)
ax.text((X[3] + X[2] + Wd[2]) / 2 + 0.03, 0.045, "model selection / hyper-parameter feedback", fontsize=6.3, color=C["mute"], ha="center")
fig.savefig(f"{FIG}/fig_system_arch.png", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---------------------------------------------------------------- model architectures
LC = {"in": "#E8EEF7", "conv": "#FBE7C6", "norm": "#FFF4D6", "pool": "#DCEBDD", "drop": "#EFEFEF", "dense": "#F3DCE5",
      "rnn": "#E6E1F2", "attn": "#D6ECF0", "out": "#F7C9C9", "frozen": "#D9DDE6"}
rows = [
    ("(a) Visual CNN, 48×48 gray input (5.57 M parameters)", [("in", "Input\n48×48×1"), ("conv", "Conv 3×3\n64, ReLU"), ("norm", "Batch\nNorm"),
        ("conv", "Conv 3×3\n64, ReLU"), ("pool", "MaxPool\n2×2"), ("drop", "Dropout\n0.25"), ("conv", "Conv 3×3\n128, ReLU"), ("norm", "Batch\nNorm"),
        ("conv", "Conv 3×3\n128, ReLU"), ("pool", "MaxPool\n2×2"), ("drop", "Dropout\n0.25"), ("dense", "Dense\n512, ReLU"), ("drop", "Dropout\n0.5"),
        ("out", "Softmax\n3")]),
    ("(b) Frozen-backbone transfer model (trainable head: 0.33–0.53 M parameters)", [("in", "Input\n224×224×3"),
        ("frozen", "Frozen ImageNet backbone\nMobileNetV2 | EfficientNetB0\nResNet50 | VGG16"), ("pool", "Global\nAvgPool"),
        ("dense", "Dense\n256, ReLU"), ("out", "Softmax\n3")]),
    ("(c) Spectrogram CNN, 128×128 log-mel input (3.30 M parameters)", [("in", "Log-mel\n128×128×1"), ("conv", "Conv 3×3\n32, ReLU"),
        ("pool", "MaxPool\n2×2"), ("conv", "Conv 3×3\n64, ReLU"), ("pool", "MaxPool\n2×2"), ("conv", "Conv 3×3\n128, ReLU"), ("pool", "MaxPool\n2×2"),
        ("dense", "Flatten,\nDense 128"), ("drop", "Dropout\n0.5"), ("out", "Softmax\n3")]),
    ("(d) Convolutional-recurrent network, CRNN (85 k parameters)", [("in", "Log-mel\n128×128×1"), ("conv", "Conv 3×3\n32, ReLU"),
        ("pool", "MaxPool\n2×2"), ("conv", "Conv 3×3\n64, ReLU"), ("pool", "MaxPool\n2×2"), ("conv", "Conv 3×3\n64, ReLU"), ("pool", "MaxPool\n2×2"),
        ("rnn", "Reshape\n196×64"), ("rnn", "GRU\n64 units"), ("dense", "Dense\n64, ReLU"), ("out", "Softmax\n3")]),
    ("(e) Attention encoder (4.46 M parameters)", [("in", "Log-mel\n128×128×1"), ("dense", "Flatten,\nDense 256"), ("rnn", "Reshape\n1×256"),
        ("attn", "Multi-head\nself-attention\n4 heads, dₖ = 64"), ("norm", "Layer\nNorm"), ("out", "Softmax\n3")]),
]
fig, ax = plt.subplots(figsize=(7.16, 4.2)); fig.subplots_adjust(0, 0, 1, 1); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
H, gap, top, g = 0.125, 0.07, 0.965, 0.009
unit = (1.0 - g * 13) / 14.0
for r, (title, layers) in enumerate(rows):
    y = top - r * (H + gap) - H - 0.03
    ax.text(0.0, y + H + 0.012, title, fontsize=7.2, fontweight="bold", color=C["ink"], va="bottom")
    x = 0.0
    for t, txt in layers:
        ww = unit * (3.0 if t == "frozen" else 1.3 if t == "attn" else 1.0)
        box(ax, x, y, ww, H, txt, LC[t], fs=5.8)
        if x > 0: arrow(ax, (x - g, y + H / 2), (x, y + H / 2))
        x += ww + g
fig.savefig(f"{FIG}/fig_model_arch.png", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ---------------------------------------------------------------- CV protocol
fig, ax = plt.subplots(figsize=(3.5, 2.5)); fig.subplots_adjust(0, 0, 1, 1); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.text(0.0, 0.98, "Outer loop: stratified (group) 5-fold CV, 3 seeds", fontsize=7, fontweight="bold", va="top")
for k in range(5):
    y = 0.79 - k * 0.105
    ax.text(0.0, y + 0.035, f"Fold {k + 1}", fontsize=6.5, va="center")
    for j in range(5):
        fc = "#F7C9C9" if j == k else "#DCEBDD"
        ax.add_patch(FancyBboxPatch((0.12 + j * 0.122, y), 0.112, 0.07, boxstyle="round,pad=0,rounding_size=0.008", fc=fc, ec=C["edge"], lw=0.6))
        ax.text(0.12 + j * 0.122 + 0.056, y + 0.035, "test" if j == k else "train", fontsize=5.8, ha="center", va="center")
box(ax, 0.765, 0.66, 0.235, 0.20, "Inner split of\ntraining folds:\n85 % fit, 15 % val", C["mod"], fs=6.2)
box(ax, 0.765, 0.40, 0.235, 0.20, "Early stopping /\nbest epoch chosen\non val only", C["mod"], fs=6.2)
box(ax, 0.765, 0.14, 0.235, 0.20, "Score held-out\nfold; report\nmean ± SD", C["out"], fs=6.2)
arrow(ax, (0.73, 0.825), (0.765, 0.78)); arrow(ax, (0.8825, 0.66), (0.8825, 0.60)); arrow(ax, (0.8825, 0.40), (0.8825, 0.34))
ax.text(0.0, 0.05, "Images: near-duplicate groups (dHash distance ≤ 4) stay in one fold.", fontsize=6.2, va="center", color=C["mute"])
fig.savefig(f"{FIG}/fig_cv_protocol.png", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
print("ok")
