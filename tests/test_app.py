from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError

import extractor
from app import attach_ocr_text, load_uploaded_image


def test_load_uploaded_image_returns_rgb_copy() -> None:
    source = Image.new("RGBA", (2, 2), color=(100, 120, 140, 80))
    image_bytes = BytesIO()
    source.save(image_bytes, format="PNG")

    image = load_uploaded_image(image_bytes.getvalue())

    assert image.mode == "RGB"
    assert image.size == (2, 2)


def test_load_uploaded_image_rejects_invalid_data() -> None:
    with pytest.raises(UnidentifiedImageError):
        load_uploaded_image(b"not an image")


def test_attach_ocr_text_populates_snippets(monkeypatch) -> None:
    snippets = [
        {"id": 1, "color": "yellow", "image": Image.new("RGB", (40, 20), "yellow")},
        {"id": 2, "color": "green", "image": Image.new("RGB", (40, 20), "green")},
    ]
    responses = iter(["First highlight", "Second highlight"])
    monkeypatch.setattr(extractor, "extract_text", lambda _img: next(responses))

    updated, err = attach_ocr_text(snippets)

    assert err is None
    assert updated[0]["text"] == "First highlight"
    assert updated[1]["text"] == "Second highlight"


def test_attach_ocr_text_handles_missing_tesseract(monkeypatch) -> None:
    def raise_missing(_img):
        raise extractor.TesseractNotInstalledError("Tesseract OCR is not installed")

    monkeypatch.setattr(extractor, "extract_text", raise_missing)
    snippets = [{"id": 1, "color": "yellow", "image": Image.new("RGB", (40, 20), "yellow")}]

    updated, err = attach_ocr_text(snippets)

    assert updated[0]["text"] == ""
    assert err == "Tesseract OCR is not installed"


def test_streamlit_app_renders_detections_without_snippets_error() -> None:
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path))
    at.session_state["detections"] = {
        "list": [{"color": "yellow", "bbox": (10, 10, 40, 20), "area": 800}],
        "masks": {"yellow": None},
        "annotated": Image.new("RGB", (100, 100), "white"),
    }
    at.session_state["snippets"] = None
    at.run()

    assert not at.exception
