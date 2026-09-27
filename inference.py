"""Model loading, pre-processing, prediction and confidence measures for PetSecure.

Kept free of any Streamlit code so it can be unit-tested and reused (app, scripts, an API later).
The logic is unchanged from the version evaluated in the paper.
"""
import logging
import os
import time
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
from PIL import Image

log = logging.getLogger("petsecure")

MODEL_DIR = Path(os.environ.get("PETSECURE_MODEL_DIR", Path(__file__).resolve().parent / "models"))
PHOTO_MODEL = MODEL_DIR / "efficientnet_frozen_emotion.keras"
SOUND_MODEL = MODEL_DIR / "crnn_corrected_emotion.keras"
PHOTO_CLASSES = ["happy", "sad", "angry"]   # output order of the photo model
SOUND_CLASSES = ["angry", "happy", "sad"]   # output order of the sound model

# Confidence tiers: (minimum top probability, minimum gap to the runner-up). Same thresholds as the original audio app.
TIERS = (("High", 0.85, 0.35), ("Moderate", 0.65, 0.20))

# Audio front-end (corrected): 16 kHz, centre 3.0 s window, 128 mel bands up to 8 kHz -> 128 x 128 input.
SAMPLE_RATE, HOP, FRAMES, N_MELS, FMAX = 16000, 375, 128, 128, 8000
MIN_SECONDS = 0.1


@lru_cache(maxsize=None)
def load(path: Path):
    import tensorflow as tf                      # imported lazily: the confidence functions don't need TensorFlow
    t0 = time.perf_counter()
    model = tf.keras.models.load_model(path)
    log.info("loaded %s in %.1f s", Path(path).name, time.perf_counter() - t0)
    return model


def tier(p: dict) -> str:
    top, second = sorted(p.values(), reverse=True)[:2]
    for name, min_top, min_gap in TIERS:
        if top >= min_top and top - second >= min_gap:
            return name
    return "Low"


def uncertainty(p: dict) -> float:
    """Normalized entropy: 0 = certain, 1 = all classes equally likely."""
    q = np.clip(np.array(list(p.values())), 1e-12, 1.0)
    return float(-(q * np.log(q)).sum() / np.log(len(q)))


def margin(p: dict) -> float:
    top, second = sorted(p.values(), reverse=True)[:2]
    return top - second


def predict_photo(image: Image.Image) -> dict:
    x = np.asarray(image.convert("RGB").resize((224, 224), Image.BILINEAR), dtype="float32")[None]  # model rescales internally
    t0 = time.perf_counter()
    p = {c: float(v) for c, v in zip(PHOTO_CLASSES, load(PHOTO_MODEL).predict(x, verbose=0)[0], strict=True)}
    log.info("photo -> %s (%.2f) in %.0f ms", max(p, key=p.get), max(p.values()), 1000 * (time.perf_counter() - t0))
    return p


def log_mel(y: np.ndarray, sr: int = SAMPLE_RATE, hop: int = HOP, frames: int = FRAMES) -> np.ndarray:
    """Corrected front-end: centre 3.0 s window (zero-padded if shorter), 128 mel bands up to 8 kHz."""
    n = hop * (frames - 1)
    y = y[(len(y) - n) // 2:][:n] if len(y) > n else np.pad(y, ((n - len(y)) // 2, n - len(y) - (n - len(y)) // 2))
    s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=hop, n_mels=N_MELS, fmax=FMAX)
    return librosa.power_to_db(s, ref=np.max)[:, :frames]


def load_audio(source) -> tuple:
    """Decode any WAV/FLAC/MP3/OGG file (path or file-like) to 16 kHz mono. Raises ValueError for clips shorter than 0.1 s."""
    y, sr = librosa.load(source, sr=SAMPLE_RATE)
    if len(y) < sr * MIN_SECONDS:
        raise ValueError("the recording is shorter than 0.1 s")
    return y, sr


def predict_sound(y: np.ndarray, sr: int = SAMPLE_RATE):
    spec = log_mel(y, sr)
    t0 = time.perf_counter()
    p = load(SOUND_MODEL).predict(spec[None, ..., None], verbose=0)[0]
    p = {c: float(v) for c, v in zip(SOUND_CLASSES, p, strict=True)}
    log.info("sound -> %s (%.2f) in %.0f ms", max(p, key=p.get), max(p.values()), 1000 * (time.perf_counter() - t0))
    return p, spec
