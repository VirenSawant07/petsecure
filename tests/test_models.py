"""Smoke tests: the shipped models load and still give the expected answers on held-out examples."""
from pathlib import Path

import pytest
from PIL import Image

from inference import load_audio, predict_photo, predict_sound

EX = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("file, emotion", [("photo_happy.jpg", "happy"), ("photo_angry.jpg", "angry"), ("photo_sad.jpg", "sad")])
def test_photo_model(file, emotion):
    p = predict_photo(Image.open(EX / file))
    assert max(p, key=p.get) == emotion
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("file, emotion", [("sound_sad.wav", "sad"), ("sound_happy.wav", "happy")])
def test_sound_model(file, emotion):
    p, spec = predict_sound(*load_audio(EX / file))
    assert max(p, key=p.get) == emotion
    assert spec.shape == (128, 128)
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-4)
