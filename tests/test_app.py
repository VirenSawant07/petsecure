"""The Streamlit pages render without errors and the navigation buttons work (headless, no browser needed)."""
from streamlit.testing.v1 import AppTest


def run(page=None):
    at = AppTest.from_file("../app.py", default_timeout=120)   # path is relative to this test file
    if page:
        at.query_params["page"] = page
    return at.run()


def test_home_renders():
    at = run()
    assert not at.exception
    assert any("really" in m.value for m in at.markdown)


def test_home_button_opens_vision():
    at = run()
    at.button(key="go_visual").click().run()
    assert not at.exception
    assert any("PetSecure Vision" in m.value for m in at.markdown)


def test_sound_page_renders():
    at = run("audio")
    assert not at.exception
    assert any("PetSecure Sound" in m.value for m in at.markdown)
