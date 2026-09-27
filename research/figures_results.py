"""Result figures (hold-out, confusion, ROC, CV, ablation, efficiency, confidence, t-SNE)."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

ROOT = os.path.dirname(os.path.abspath(__file__))
R, FIG = f"{ROOT}/results", f"{ROOT}/figs"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 7.5, "axes.titlesize": 8,
                     "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.5, "axes.axisbelow": True, "savefig.dpi": 300, "legend.frameon": False})
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"         # validated categorical slots 1-4
GRAY, INK, INK2 = "#9a9990", "#0b0b0b", "#52514e"
def save(fig, name): fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

sa, si = json.load(open(f"{R}/saved_audio.json")), json.load(open(f"{R}/saved_image.json"))
hf = json.load(open(f"{R}/img/holdout_frozen.json"))
cva, cvi = json.load(open(f"{R}/audio/cv_summary.json")), json.load(open(f"{R}/img/cv_summary.json"))
lat, an, probe = json.load(open(f"{R}/latency.json")), json.load(open(f"{R}/analysis.json")), json.load(open(f"{R}/audio/probe.json"))
ya_te, yi_te = np.load(f"{R}/y_audio_test.npy"), np.load(f"{R}/y_img_test.npy")
ya, yi = np.load(f"{R}/audio/y.npy"), np.load(f"{R}/img/y.npy")
A_CLS, I_CLS = ["angry", "happy", "sad"], ["happy", "sad", "angry"]

# ------------------------------------------------ hold-out comparison
fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.3), gridspec_kw=dict(height_ratios=[1.15, 1], hspace=0.55))
rows = [("Visual CNN (48×48)", si["facial_expression_model.h5"]["acc"], GRAY),
        ("ResNet50, fine-tuned", si["resnet50_model.h5|infer"]["acc"], GRAY), ("VGG16, fine-tuned", si["vgg16_model.h5|infer"]["acc"], GRAY),
        ("\"MobileNet\" (VGG16 base)", si["mobilenet_model.h5|infer"]["acc"], GRAY), ("EfficientNetB0, fine-tuned", si["efficientnet_model.h5|infer"]["acc"], GRAY),
        ("MobileNetV2, frozen", hf["MobileNetV2"]["acc"], S1), ("EfficientNetB0, frozen", hf["EfficientNetB0"]["acc"], S1),
        ("ResNet50, frozen", hf["ResNet50"]["acc"], S1), ("VGG16, frozen", hf["VGG16"]["acc"], S1)]
yy = np.arange(len(rows))[::-1]
b = ax[0].barh(yy, [r[1] * 100 for r in rows], color=[r[2] for r in rows], height=0.62)
ax[0].bar_label(b, labels=[f"{r[1]*100:.1f}" for r in rows], fontsize=6.3, padding=2)
ax[0].set_yticks(yy); ax[0].set_yticklabels([r[0] for r in rows]); ax[0].set_xlim(0, 100); ax[0].set_xlabel("Hold-out accuracy (%), 750 images")
ax[0].axvline(100 / 3, color=INK2, lw=0.7, ls=(0, (3, 2))); ax[0].text(100 / 3 + 1, len(rows) - 0.35, "chance", fontsize=6, color=INK2)
ax[0].set_title("(a) Visual: as first trained (gray) vs corrected (blue)", fontsize=7.2); ax[0].grid(axis="y", visible=False)
arows = [("CNN (deployed)", "cnn_audio_emotion.h5"), ("CNN, best ckpt", "cnn_audio_emotion_best.h5"), ("CRNN", "crnn_audio_emotion.h5"),
         ("CRNN, best ckpt", "crnn_audio_emotion_best.keras"), ("Attention enc.", "transformer_audio_emotion_best.keras")]
x = np.arange(len(arows)); w = 0.36
b1 = ax[1].bar(x - w / 2, [sa[k]["acc"] * 100 for _, k in arows], w, color=S1, label="accuracy")
b2 = ax[1].bar(x + w / 2, [sa[k]["f1"] * 100 for _, k in arows], w, color=S2, label="macro-F1")
ax[1].bar_label(b1, fmt="%.1f", fontsize=5.4, label_type="center", rotation=90, color="white"); ax[1].bar_label(b2, fmt="%.1f", fontsize=5.4, label_type="center", rotation=90, color="white")
ax[1].set_xticks(x); ax[1].set_xticklabels([a for a, _ in arows], rotation=20, ha="right"); ax[1].set_ylim(0, 100)
ax[1].set_ylabel("Score (%)"); ax[1].legend(loc="upper right", ncol=2, fontsize=6.2)
ax[1].set_title("(b) Acoustic checkpoints, 23-clip hold-out", fontsize=7.2); ax[1].grid(axis="x", visible=False)
save(fig, "fig_holdout_comparison")

# ------------------------------------------------ confusion matrices (hold-out)
def cm_panel(a, cm, labels, title):
    cm = np.array(cm); rn = cm / cm.sum(1, keepdims=True)
    a.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{cm[i, j]}\n{rn[i, j]*100:.0f}%", ha="center", va="center", fontsize=6.2,
                   color="white" if rn[i, j] > 0.55 else INK, linespacing=1.1)
    a.set_xticks(range(3)); a.set_yticks(range(3)); a.set_xticklabels(labels); a.set_yticklabels(labels)
    a.set_xlabel("Predicted"); a.set_title(title, fontsize=7.2); a.grid(False)
    for sp in a.spines.values(): sp.set_visible(False)
fig, ax = plt.subplots(2, 2, figsize=(3.5, 3.55), gridspec_kw=dict(wspace=0.45, hspace=0.5)); ax = ax.flat
cm_panel(ax[0], si["facial_expression_model.h5"]["cm"], I_CLS, "(a) Visual CNN (as trained)")
cm_panel(ax[1], hf["ResNet50"]["cm"], I_CLS, "(b) ResNet50, frozen")
cm_panel(ax[2], sa["cnn_audio_emotion_best.h5"]["cm"], A_CLS, "(c) Audio CNN, best ckpt")
cm_panel(ax[3], sa["crnn_audio_emotion_best.keras"]["cm"], A_CLS, "(d) Audio CRNN, best ckpt")
ax[0].set_ylabel("True"); ax[2].set_ylabel("True")
save(fig, "fig_confusion_holdout")

# ------------------------------------------------ ROC curves (macro-average one-vs-rest)
def macro_roc(y, p):
    Y = label_binarize(y, classes=[0, 1, 2]); grid = np.linspace(0, 1, 201); tprs = []
    for c in range(3):
        f, t, _ = roc_curve(Y[:, c], p[:, c]); tprs.append(np.interp(grid, f, t))
    t = np.mean(tprs, 0); t[0] = 0
    return grid, t, auc(grid, t)
fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.6), gridspec_kw=dict(hspace=0.42))
for (n, c) in zip(["ResNet50", "VGG16", "MobileNetV2", "EfficientNetB0"], [S1, S2, S3, S4]):
    g, t, a = macro_roc(yi_te, np.load(f"{R}/img/holdout_probs_{n}.npy")); ax[0].plot(g, t, color=c, lw=1.5, label=f"{n}, frozen (AUC {a:.3f})")
g, t, a = macro_roc(yi_te, np.load(f"{R}/probs_img_cnn.npy")); ax[0].plot(g, t, color=GRAY, lw=1.5, label=f"Visual CNN, as trained (AUC {a:.3f})")
for (lab, k, c) in [("CNN, best ckpt", "cnn_audio_emotion_best.h5", S1), ("CRNN, best ckpt", "crnn_audio_emotion_best.keras", S2),
                    ("CNN, deployed", "cnn_audio_emotion.h5", S3), ("Attention enc.", "transformer_audio_emotion_best.keras", S4)]:
    g, t, a = macro_roc(ya_te, np.load(f"{R}/probs_audio_{k}.npy")); ax[1].plot(g, t, color=c, lw=1.5, label=f"{lab} (AUC {a:.3f})")
for a_, ttl in zip(ax, ["(a) Visual models, 750-image hold-out", "(b) Acoustic models, 23-clip hold-out"]):
    a_.plot([0, 1], [0, 1], color=GRAY, lw=0.7, ls=(0, (3, 2))); a_.set_xlim(0, 1); a_.set_ylim(0, 1.02)
    a_.set_xlabel("False-positive rate"); a_.set_ylabel("True-positive rate (macro-avg.)"); a_.set_title(ttl, fontsize=7.5)
    a_.legend(loc="lower right", fontsize=6)
save(fig, "fig_roc")

# ------------------------------------------------ CV fold distributions
def strip(a, data, labels, colors, hold=None):
    rng = np.random.default_rng(1)
    for i, (d, c) in enumerate(zip(data, colors)):
        d = np.array(d) * 100
        a.boxplot(d, positions=[i], widths=0.5, showfliers=False, medianprops=dict(color=INK, lw=1),
                  boxprops=dict(color=INK2, lw=0.7), whiskerprops=dict(color=INK2, lw=0.7), capprops=dict(color=INK2, lw=0.7))
        a.scatter(i + rng.uniform(-0.15, 0.15, len(d)), d, s=10, color=c, edgecolor="white", lw=0.4, zorder=3)
        a.text(i, 101, f"{d.mean():.1f}", ha="center", fontsize=6.3, color=INK)
        if hold is not None and hold[i] is not None:
            a.scatter([i + 0.33], [hold[i] * 100], marker="D", s=14, color=INK, zorder=4)
    a.set_xticks(range(len(labels))); a.set_xticklabels(labels, rotation=20, ha="right"); a.set_ylim(20, 106); a.grid(axis="x", visible=False)
fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.4), gridspec_kw=dict(hspace=0.62))
ik = ["CNN", "MobileNetV2", "EfficientNetB0", "ResNet50", "VGG16"]
strip(ax[0], [[r["acc"] for r in cvi[k]["rows"]] for k in ik], ["Visual CNN", "MobileNetV2", "EfficientNetB0", "ResNet50", "VGG16"],
      [GRAY, S1, S1, S1, S1], [si["facial_expression_model.h5"]["acc"]] + [hf[k]["acc"] for k in ik[1:]])
ax[0].set_ylabel("Fold accuracy (%)"); ax[0].set_title("(a) Visual models, group 5-fold CV", fontsize=7.5)
ak = [k for k in ["CNN|orig", "CRNN|orig", "Transformer|orig", "CNN|fixed", "CRNN|fixed", "Transformer|fixed"] if k in cva]
names = {"CNN|orig": "CNN", "CRNN|orig": "CRNN", "Transformer|orig": "Attention", "CNN|fixed": "CNN, corr.", "CRNN|fixed": "CRNN, corr.", "Transformer|fixed": "Attention, corr."}
holdmap = {"CNN|orig": sa["cnn_audio_emotion_best.h5"]["acc"], "CRNN|orig": sa["crnn_audio_emotion_best.keras"]["acc"],
           "Transformer|orig": sa["transformer_audio_emotion_best.keras"]["acc"]}
strip(ax[1], [[r["acc"] for r in cva[k]["rows"]] for k in ak], [names[k] for k in ak],
      [S2 if "fixed" in k else S1 for k in ak], [holdmap.get(k) for k in ak])
ax[1].axhline(probe["samplerate+duration"]["mean"] * 100, color=S2, lw=0.8, ls=(0, (4, 2)))
ax[1].text(len(ak) - 0.5, probe["samplerate+duration"]["mean"] * 100 + 1.2, "metadata-only baseline", fontsize=6, color=INK2, ha="right")
ax[1].set_title("(b) Acoustic models, stratified 5-fold CV ×3", fontsize=7.2); ax[1].set_ylabel("Fold accuracy (%)")
ax[0].scatter([], [], marker="D", s=14, color=INK, label="single hold-out split"); ax[0].legend(loc="lower right", fontsize=6.2)
save(fig, "fig_cv_folds")

# ------------------------------------------------ CV aggregated confusion
best_i = max([k for k in cvi if k != "CNN"], key=lambda k: cvi[k]["mean"]["acc"])
best_a = "CRNN|fixed"
fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.85), gridspec_kw=dict(wspace=0.55))
cm_panel(ax[0], np.sum(cvi[best_i]["cm"], 0), I_CLS, f"(a) {best_i}, out-of-fold")
cm_panel(ax[1], np.sum(cva[best_a]["cm"], 0), A_CLS, "(b) Audio CRNN, corrected input")
ax[0].set_ylabel("True")
save(fig, "fig_cv_confusion")

# ------------------------------------------------ efficiency (accuracy vs latency)
fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.0), gridspec_kw=dict(hspace=0.55))
pts = [("Visual CNN", "visual CNN", cvi["CNN"]["mean"]["acc"], GRAY)] + [(k, k, cvi[k]["mean"]["acc"], S1) for k in ik[1:]]
for n, lk, acc_, c in pts:
    L = lat[lk]; ax[0].scatter(L["ms"], acc_ * 100, s=12 + L["params"] / 2.5e5, color=c, alpha=0.85, edgecolor="white", lw=0.6, zorder=3)
    off = {"MobileNetV2": (-5, -12), "EfficientNetB0": (-7, 6), "ResNet50": (5, 5), "VGG16": (-6, -13), "Visual CNN": (6, -3)}[n]
    ax[0].annotate(f"{n} ({L['params']/1e6:.1f} M)", (L["ms"], acc_ * 100), xytext=off, textcoords="offset points", fontsize=5.8,
                   ha="right" if off[0] < 0 else "left")
ax[0].set_xscale("log"); ax[0].set_xlabel("CPU latency per image (ms, batch 1, log scale)"); ax[0].set_ylabel("CV accuracy (%)")
ax[0].set_ylim(30, 92); ax[0].set_title("(a) Visual models", fontsize=7.5); ax[0].set_xlim(1, 400)
for n, lk, ck in [("CNN", "audio CNN", "CNN|orig"), ("CRNN", "audio CRNN", "CRNN|orig"), ("Attention", "audio attention", "Transformer|orig")]:
    if ck not in cva: continue
    L = lat[lk]; ax[1].scatter(L["ms"], cva[ck]["mean"]["acc"] * 100, s=12 + L["params"] / 2.5e5, color=S1, alpha=0.85, edgecolor="white", lw=0.6, zorder=3)
    ax[1].annotate(f"{n}\n{L['params']/1e6:.2f} M", (L["ms"], cva[ck]["mean"]["acc"] * 100), xytext=(6, -3), textcoords="offset points", fontsize=6, linespacing=1.0)
ax[1].set_xscale("log"); ax[1].set_xlabel("CPU latency per clip (ms, model only, log scale)"); ax[1].set_ylabel("CV accuracy (%)")
ax[1].set_title("(b) Acoustic models", fontsize=7.5); ax[1].set_ylim(30, 92); ax[1].set_xlim(0.5, 60)
save(fig, "fig_efficiency")

# ------------------------------------------------ confidence: reliability + tiers
def reliability(P, y, bins=10):
    p = P.reshape(-1, 3); yy = np.tile(y, P.shape[0]); conf, ok = p.max(1), p.argmax(1) == yy
    e = np.linspace(1 / 3, 1, bins + 1); xs, ys, ns = [], [], []
    for lo, hi in zip(e[:-1], e[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 5: xs.append(conf[m].mean()); ys.append(ok[m].mean()); ns.append(m.sum())
    return np.array(xs), np.array(ys)
fig, ax = plt.subplots(2, 1, figsize=(3.5, 4.2), gridspec_kw=dict(hspace=0.5))
for lab, P, y, c in [(f"Image {best_i}", np.load(f"{R}/img/oof_{best_i}.npy"), yi, S1),
                     ("Audio CNN", np.load(f"{R}/audio/oof_CNN_orig.npy"), ya, S2)]:
    xs, ys = reliability(P, y); ax[0].plot(xs, ys, marker="o", ms=3.5, lw=1.4, color=c, label=lab)
ax[0].plot([1 / 3, 1], [1 / 3, 1], color=GRAY, lw=0.7, ls=(0, (3, 2))); ax[0].set_xlabel("Mean top-1 confidence p(1)")
ax[0].set_ylabel("Observed accuracy"); ax[0].set_title("(a) Reliability of out-of-fold predictions", fontsize=7.5); ax[0].legend(loc="upper left", fontsize=6.3)
tiers = ["High", "Moderate", "Low"]; x = np.arange(3); w = 0.36
for i, (lab, key, c) in enumerate([(f"Image {best_i}", f"image {best_i}", S1), ("Audio CNN", "audio CNN_orig", S2)]):
    acc_ = [an[key]["tiers"][t]["acc"] * 100 for t in tiers]; sh = [an[key]["tiers"][t]["share"] * 100 for t in tiers]
    bb = ax[1].bar(x + (i - 0.5) * w, acc_, w, color=c, label=lab)
    ax[1].bar_label(bb, labels=[f"{a:.0f}%" for a in acc_], fontsize=5.8, padding=1.5)
    for xi, s_ in zip(x + (i - 0.5) * w, sh):
        ax[1].text(xi, 3, f"{s_:.0f}% of preds", rotation=90, ha="center", va="bottom", fontsize=5.2, color="white")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"{t} tier" for t in tiers]); ax[1].set_ylim(0, 118); ax[1].set_ylabel("Accuracy within tier (%)")
ax[1].set_title("(b) Accuracy within each app confidence tier", fontsize=7.2); ax[1].legend(loc="upper right", fontsize=6.3, ncol=2)
ax[1].grid(axis="x", visible=False)
save(fig, "fig_confidence")

# ------------------------------------------------ t-SNE of best backbone embeddings
F = np.load(f"{R}/img/feat_{best_i}.npy")
Z = TSNE(2, perplexity=35, init="pca", random_state=0).fit_transform((F - F.mean(0)) / (F.std(0) + 1e-6))
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for k, (c, col) in enumerate(zip(I_CLS, [S3, S1, S2])):
    m = yi == k; ax.scatter(Z[m, 0], Z[m, 1], s=2.2, color=col, alpha=0.65, lw=0, label=c)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.legend(loc="upper right", markerscale=4, fontsize=6.5)
ax.set_title(f"t-SNE of frozen {best_i} embeddings (3,750 images)", fontsize=7.5)
save(fig, "fig_tsne")
print("ok", best_i, best_a)
