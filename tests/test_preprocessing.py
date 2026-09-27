"""The audio front-end must always give the model a 128 x 128 spectrogram, whatever the clip length."""
import io

import numpy as np
import pytest

from inference import SAMPLE_RATE, load_audio, log_mel


@pytest.mark.parametrize("seconds", [0.5, 3.0, 10.0])
def test_log_mel_shape_is_fixed(seconds):
    y = np.random.default_rng(0).normal(size=int(seconds * SAMPLE_RATE)).astype("float32")
    spec = log_mel(y)
    assert spec.shape == (128, 128)
    assert np.isfinite(spec).all() and spec.max() == pytest.approx(0.0)   # decibels relative to the loudest point


def test_rejects_non_audio_file():
    with pytest.raises(RuntimeError):   # soundfile.LibsndfileError: format not recognised
        load_audio(io.BytesIO(b"this is not audio"))
