"""The confidence measures shown next to every prediction."""
import pytest

from inference import margin, tier, uncertainty


@pytest.mark.parametrize("p, expected", [
    ({"angry": 0.02, "happy": 0.95, "sad": 0.03}, "High"),      # strong and far ahead
    ({"angry": 0.04, "happy": 0.12, "sad": 0.84}, "Moderate"),  # just under the High cut-off
    ({"angry": 0.10, "happy": 0.55, "sad": 0.35}, "Low"),       # close call
    ({"angry": 0.00, "happy": 0.86, "sad": 0.14}, "High"),      # exactly on the edges: 0.86 >= 0.85 and gap 0.72 >= 0.35
    ({"angry": 0.40, "happy": 0.60, "sad": 0.00}, "Low"),       # below 0.65
])
def test_tier(p, expected):
    assert tier(p) == expected


def test_margin_is_gap_between_top_two():
    assert margin({"angry": 0.1, "happy": 0.6, "sad": 0.3}) == pytest.approx(0.3)


def test_uncertainty_bounds():
    assert uncertainty({"angry": 1.0, "happy": 0.0, "sad": 0.0}) == pytest.approx(0.0, abs=1e-9)   # certain
    assert uncertainty({"angry": 1 / 3, "happy": 1 / 3, "sad": 1 / 3}) == pytest.approx(1.0)       # pure guess
    assert 0 < uncertainty({"angry": 0.2, "happy": 0.7, "sad": 0.1}) < 1
