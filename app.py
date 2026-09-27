"""PetSecure: dog emotion recognition from a photo (Vision) or a vocalization (Sound).

Photo model : frozen ImageNet EfficientNetB0 + dense head   (models/efficientnet_frozen_emotion.keras)
Sound model : CRNN on the corrected 16 kHz / 3.0 s log-mel   (models/crnn_corrected_emotion.keras)
Pages       : home · visual · audio        e.g. ?page=visual&photo=photo_sad.jpg  or  ?page=audio&sound=sound_sad.wav
Run with    : streamlit run app.py   (picks up .streamlit/config.toml and ./static); models and measures: inference.py
"""
import io
import json
import logging
from datetime import datetime
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

import ui
from inference import (PHOTO_CLASSES, SOUND_CLASSES, load_audio, margin, predict_photo, predict_sound,
                       tier, uncertainty)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent
EMOTIONS = ui.EMOTIONS
EXAMPLES = ROOT / "examples"
MODEL_CLASSES = json.loads((ROOT / "models" / "crnn_corrected_emotion.json").read_text())["classes"]
assert set(MODEL_CLASSES) == set(PHOTO_CLASSES) == set(SOUND_CLASSES) == set(EMOTIONS), "UI emotions must match the model classes"

st.set_page_config(page_title="PetSecure · Understand how your dog really feels", page_icon=str(ROOT / "static" / "favicon.png"),
                   layout="wide", initial_sidebar_state="collapsed")


# ------------------------------------------------------------------ reports and pickers (models and measures live in inference.py)
def report(kind: str, name: str, p: dict) -> tuple:
    r = {"app": "PetSecure", "analysis": kind, "file": name, "generated_at": datetime.now().isoformat(timespec="seconds"),
         "emotion": max(p, key=p.get), "probabilities": {e: round(p[e], 4) for e in EMOTIONS},
         "confidence_tier": tier(p), "top2_margin": round(margin(p), 4), "uncertainty_index": round(uncertainty(p), 4)}
    txt = (f"PETSECURE {kind.upper()} REPORT\nFile: {name}\nGenerated: {r['generated_at']}\n\n"
           f"Emotion: {r['emotion']} ({100 * p[r['emotion']]:.1f}%)\nConfidence tier: {r['confidence_tier']}\n"
           + "".join(f"  {e:6s} {100 * p[e]:5.1f}%\n" for e in EMOTIONS)
           + f"Top-2 margin: {100 * r['top2_margin']:.1f} points\nUncertainty index: {r['uncertainty_index']}\n\n"
           "Predictions are probabilistic and support, not replace, human judgement. PetSecure is not a substitute for a vet.\n")
    return txt, json.dumps(r, indent=2)


def downloads(kind: str, name: str, p: dict, key: str) -> None:
    txt, js = report(kind, name, p)
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Report (TXT)", txt, f"petsecure_{key}_report.txt", use_container_width=True, key=f"{key}_txt")
    c2.download_button("⬇ Report (JSON)", js, f"petsecure_{key}_report.json", use_container_width=True, key=f"{key}_json")


def example_picker(label: str, pattern: str, param: str):
    options = ["(none)"] + [f.name for f in sorted(EXAMPLES.glob(pattern))]
    wanted = st.query_params.get(param)
    choice = st.selectbox(label, options, index=options.index(wanted) if wanted in options else 0,
                          help="Held-out files the models never saw during training. Files marked “hard” are ones the model gets wrong.")
    return None if choice == "(none)" else EXAMPLES / choice


def model_note(json_name: str, what: str, unit: str) -> str:
    info = json.loads((ROOT / "models" / json_name).read_text())
    return (f"{what}: {info['holdout_correct']} of {info['holdout_total']} held-out {unit} correct "
            f"({100 * info['holdout_correct'] / info['holdout_total']:.1f}%).")


# ------------------------------------------------------------------ navigation
if "page" not in st.session_state:
    st.session_state.page = st.query_params.get("page", "home") if st.query_params.get("page") in ("visual", "audio") else "home"


def go(page: str) -> None:
    st.session_state.page = page
    st.query_params.clear()
    if page != "home":
        st.query_params["page"] = page


page = st.session_state.page
html = lambda s: st.markdown(s, unsafe_allow_html=True)
html(ui.CSS)
IMG_ACC, SND_ACC = "83.1 %", "71.4 %"        # cross-validated accuracy reported in the paper


def back_bar(other: str, other_label: str) -> None:
    html(ui.nav(links=False))
    c1, _, c3 = st.columns([1, 3, 1.3])
    c1.button("← Home", on_click=go, args=("home",), use_container_width=True)
    c3.button(other_label, on_click=go, args=(other,), use_container_width=True)


# ================================================================== HOME
if page == "home":
    html(ui.nav())
    left, right = st.columns([1.1, 1], gap="large", vertical_alignment="center")
    with left:
        html(ui.hero_text(MODEL_CLASSES))
        b1, b2 = st.columns(2)
        b1.button("📷  Analyse a photo", on_click=go, args=("visual",), type="primary", use_container_width=True, key="hero_visual")
        b2.button("🎧  Analyse a sound", on_click=go, args=("audio",), type="secondary", use_container_width=True, key="hero_audio")
    with right:
        html(ui.collage())

    html(ui.section_head("modes", "Choose a mode", "Two ways to listen to your dog",
                         "Each mode has its own model, trained and tested on its own data."))
    c1, c2 = st.columns(2, gap="large")
    with c1, st.container(border=True, key="card_mode_visual"):
        html(ui.mode("vision", "Vision", "Upload a photo of your dog and PetSecure reads its face and body.",
                     ["JPG · PNG", "EfficientNetB0"], f"{IMG_ACC} CV accuracy"))
        st.button("Analyse a photo  →", on_click=go, args=("visual",), type="primary", use_container_width=True, key="go_visual")
    with c2, st.container(border=True, key="card_mode_audio"):
        html(ui.mode("sound", "Sound", "Upload a bark, growl or whine and PetSecure listens to your dog's voice.",
                     ["WAV · FLAC · MP3 · OGG", "Log-mel CRNN"], f"{SND_ACC} CV accuracy"))
        st.button("Analyse a sound  →", on_click=go, args=("audio",), type="primary", use_container_width=True, key="go_audio")

    html(ui.section_head("how", "How it works", "Three simple steps", "No sign-up, no setup. Your file never leaves this session."))
    html(ui.steps(MODEL_CLASSES))
    html(ui.section_head("emotions", "What it can read", "Emotions PetSecure can read",
                         f"Our models recognise {len(MODEL_CLASSES)} emotional states. Here's what each one looks like."))
    html(ui.emotion_cards(ui.EMOTIONS))
    html(ui.section_head("gallery", "Good dogs only", "Made for happy tails", "A few of the faces that inspired us."))
    html(ui.gallery())
    html(ui.section_head("tips", "Tips for pet owners", "Little signs, big meaning", "Quick guides to what your dog may be telling you."))
    html(ui.tips())
    html(ui.section_head("faq", "FAQ", "Questions, answered", "Privacy, accuracy and the files you can use."))
    html(ui.faq(MODEL_CLASSES, IMG_ACC, SND_ACC))

# ================================================================== VISION
elif page == "visual":
    back_bar("audio", "Switch to Sound 🎧")
    html(ui.page_head("📷", "PetSecure Vision", "Reads your dog's mood from its face and body in a photo."))
    left, right = st.columns([1, 1.15], gap="large")
    with left, st.container(border=True, key="card_v_input"):
        html('<div class="card-title">1 · Add a photo</div>')
        upload = st.file_uploader("Dog photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
        example = example_picker("…or try an example photo", "photo_*.jpg", "photo")
        source = upload if upload is not None else example
        image = None
        if source is None:
            html(ui.empty("🐶", "No photo yet. Upload one above or pick an example."))
        else:
            try:
                image = Image.open(source); image.load()
            except Exception:
                st.error("This file could not be opened as an image. Please try a JPG or PNG photo.")
            else:
                name = getattr(source, "name", Path(str(source)).name)
                st.image(image, caption=f"{name} · {image.width} × {image.height} px", use_container_width=True)
    with right, st.container(border=True, key="card_v_result"):
        html('<div class="card-title">2 · What PetSecure sees</div>')
        if image is None:
            html(ui.empty("🔍", "Your dog's reading will appear here."))
        else:
            with st.spinner("Looking at the photo..."):
                p = predict_photo(image)
            html(ui.verdict("This dog looks", p, tier(p)) + ui.bars(p) + ui.gauge(tier(p), margin(p), uncertainty(p))
                 + ui.explain(p, tier(p), "looks"))
            html(ui.note("How it works: the photo is resized to 224 × 224 and described by a frozen ImageNet EfficientNetB0 "
                         "(1,280 features); a small trained classifier turns that description into the three probabilities. "
                         + model_note("efficientnet_frozen_emotion.json", "Model", "photos")))
            downloads("photo", name, p, "photo")

# ================================================================== SOUND
else:
    back_bar("visual", "Switch to Vision 📷")
    html(ui.page_head("🎧", "PetSecure Sound", "Listens to your dog's bark, growl or whine and reads the mood in its voice."))
    with st.container(border=True, key="card_a_input"):
        c1, c2 = st.columns([1.3, 1], gap="large")
        with c1:
            html('<div class="card-title">1 · Add a recording</div>')
            upload = st.file_uploader("Dog vocalization (WAV, FLAC, MP3 or OGG)", type=["wav", "flac", "mp3", "ogg"])
        with c2:
            html('<div class="card-title">&nbsp;</div>')
            example = example_picker("…or try an example recording", "sound_*.wav", "sound")
        data, name = (upload.getvalue(), upload.name) if upload is not None else \
            ((example.read_bytes(), example.name) if example else (None, None))
        if data is not None:
            st.audio(data)
    y = None
    if data is None:
        html(ui.empty("🎙️", "No recording yet. Upload one above or pick an example."))
    else:
        try:
            y, sr = load_audio(io.BytesIO(data))
        except Exception as exc:
            st.error(f"This recording could not be analysed ({exc}). Please try another WAV, FLAC, MP3 or OGG file.")
            y = None
    if y is not None:
        with st.spinner("Listening to the recording..."):
            p, spec = predict_sound(y, sr)
        left, right = st.columns([1.2, 1], gap="large")
        with left, st.container(border=True, key="card_a_signal"):
            html('<div class="card-title">2 · What PetSecure hears</div>')
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(6, 3.6), gridspec_kw=dict(height_ratios=[1, 1.6], hspace=0.45))
            fig.patch.set_alpha(0)
            t = np.arange(len(y)) / sr
            a1.plot(t, y, color="#1f4d3a", lw=0.5); a1.set_xlim(0, t[-1]); a1.set_yticks([])
            a1.set_title(f"Waveform · {name} · {len(y) / sr:.1f} s", fontsize=8, color="#5b665f", loc="left")
            librosa.display.specshow(spec, sr=16000, hop_length=375, x_axis="time", y_axis="mel", fmax=8000, ax=a2, cmap="magma")
            a2.set_title("Log-mel spectrogram given to the model (centre 3.0 s)", fontsize=8, color="#5b665f", loc="left")
            for a in (a1, a2):
                a.tick_params(labelsize=7, colors="#5b665f"); a.set_xlabel(""); a.set_ylabel("")
                for side in a.spines.values():
                    side.set_visible(False)
            st.pyplot(fig, clear_figure=True)
            html('<div class="card-title">Sound fingerprint</div>')
            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)
            m1.metric("Duration", f"{len(y) / sr:.1f} s")
            m2.metric("Loudness", f"{20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9):.0f} dB")
            m3.metric("Brightness", f"{librosa.feature.spectral_centroid(y=y, sr=sr).mean():.0f} Hz",
                      help="Spectral centroid: higher means a sharper, higher-pitched sound.")
            m4.metric("Noisiness", f"{librosa.feature.zero_crossing_rate(y).mean():.3f}", help="Zero-crossing rate.")
        with right, st.container(border=True, key="card_a_result"):
            html('<div class="card-title">3 · The verdict</div>')
            html(ui.verdict("This dog sounds", p, tier(p)) + ui.bars(p) + ui.gauge(tier(p), margin(p), uncertainty(p))
                 + ui.explain(p, tier(p), "sounds"))
            html(ui.note("How it works: the sound is resampled to 16 kHz, the centre 3 seconds become a 128 × 128 log-mel "
                         "spectrogram, and a small convolutional recurrent network (CRNN) reads it. "
                         + model_note("crnn_corrected_emotion.json", "Model", "clips")
                         + " It is weakest on angry sounds, which it often calls happy."))
            downloads("sound", name, p, "sound")

html(ui.footer())
