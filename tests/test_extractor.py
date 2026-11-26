import pytest
import shutil
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pytesseract
import extractor
import detector

@pytest.fixture
def sample_image():
    # Create a simple 100x100 white image
    img = Image.new("RGB", (100, 100), color="white")
    return img

def test_extract_snippets_single(sample_image):
    detections = [{"color": "yellow", "bbox": (10, 10, 30, 20), "area": 600}]
    snippets = extractor.extract_snippets(sample_image, detections, padding=5)
    assert len(snippets) == 1
    snip = snippets[0]
    # Expected padded bbox: (5,5,40,30)
    assert snip["padded_bbox"] == (5, 5, 40, 30)
    # Image size should match padded dimensions
    assert snip["width"] == 40 and snip["height"] == 30
    # Ensure original image not modified
    assert sample_image.size == (100, 100)

def test_extract_snippets_multiple_and_order(sample_image):
    detections = [
        {"color": "green", "bbox": (50, 10, 20, 10), "area": 200},
        {"color": "yellow", "bbox": (10, 40, 30, 20), "area": 600},
    ]
    snippets = extractor.extract_snippets(sample_image, detections, padding=0)
    # Should be sorted top-to-bottom then left-to-right: green (y=10) then yellow (y=40)
    ids = [s["id"] for s in snippets]
    assert ids == [1, 2]
    assert snippets[0]["color"] == "green"
    assert snippets[1]["color"] == "yellow"

def test_padding_and_clamping(sample_image):
    detections = [{"color": "pink", "bbox": (0, 0, 10, 10), "area": 100}]
    snippets = extractor.extract_snippets(sample_image, detections, padding=20)
    snip = snippets[0]
    # Padding should be clamped to image bounds (0,0) start
    assert snip["padded_bbox"] == (0, 0, 30, 30)
    assert snip["width"] == 30 and snip["height"] == 30

def test_empty_detections(sample_image):
    assert extractor.extract_snippets(sample_image, [], padding=5) == []

def test_invalid_bbox_handling(sample_image):
    # bbox with negative coordinates – function should clamp to 0
    detections = [{"color": "yellow", "bbox": (-5, -5, 10, 10), "area": 100}]
    snippets = extractor.extract_snippets(sample_image, detections, padding=0)
    snip = snippets[0]
    assert snip["padded_bbox"] == (0, 0, 10, 10)

def test_png_export(sample_image):
    detections = [{"color": "yellow", "bbox": (10, 10, 20, 20), "area": 400}]
    snippets = extractor.extract_snippets(sample_image, detections, padding=0)
    buf = BytesIO()
    snippets[0]["image"].save(buf, format="PNG")
    data = buf.getvalue()
    # PNG file starts with the standard signature
    assert data[:8] == b"\x89PNG\r\n\x1a\n"

def test_integration_with_detector():
    # Create a synthetic image with a solid yellow rectangle
    img = Image.new("RGB", (100, 100), color="white")
    # Paint a yellow box (approx HSV yellow range used in detector)
    np_img = np.array(img)
    np_img[30:50, 40:80] = (255, 255, 0)  # RGB yellow
    img2 = Image.fromarray(np_img)
    # Run detector
    detections, _, _ = detector.detect_highlights(
        img2,
        min_area=100,
        thresholds={"yellow": detector.DEFAULT_THRESHOLDS["yellow"]},
    )
    # Ensure at least one detection found
    assert len(detections) >= 1
    # Extract snippets
    snippets = extractor.extract_snippets(img2, detections, padding=2)
    assert len(snippets) >= 1
    # Verify snippet bbox is within image bounds
    for snip in snippets:
        x, y, w, h = snip["padded_bbox"]
        assert 0 <= x < img2.width and 0 <= y < img2.height
        assert x + w <= img2.width and y + h <= img2.height


def test_extract_snippets_assigns_sequential_ids_after_sorting(sample_image):
    # Input detections are intentionally out of top-to-bottom order.
    detections = [
        {"color": "yellow", "bbox": (10, 60, 30, 20), "area": 600},
        {"color": "green", "bbox": (50, 10, 20, 10), "area": 200},
        {"color": "pink", "bbox": (10, 10, 20, 10), "area": 200},
    ]
    snippets = extractor.extract_snippets(sample_image, detections, padding=0)
    assert [s["id"] for s in snippets] == [1, 2, 3]
    assert [s["color"] for s in snippets] == ["pink", "green", "yellow"]


def test_extract_text_strips_whitespace_and_preserves_input(monkeypatch):
    img = Image.new("RGB", (120, 40), color=(255, 255, 0))
    orig = img.copy()
    captured = {}

    def fake_image_to_string(processed_img, config=""):
        captured["image"] = processed_img
        captured["config"] = config
        return "  Important Biology Note \n\n"

    monkeypatch.setattr(pytesseract, "image_to_string", fake_image_to_string)
    result = extractor.extract_text(img)

    assert result == "Important Biology Note"
    assert isinstance(captured["image"], Image.Image)
    assert captured["image"].mode == "L"
    assert list(img.getdata()) == list(orig.getdata())


def test_extract_text_missing_tesseract_raises_friendly_error(monkeypatch):
    def raise_missing(*args, **kwargs):
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_string", raise_missing)
    img = Image.new("RGB", (100, 40), color="white")
    with pytest.raises(extractor.TesseractNotInstalledError, match="Tesseract OCR is not installed"):
        extractor.extract_text(img)


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract binary not installed")
def test_extract_text_real_ocr_on_highlighted_text():
    img = Image.new("RGB", (420, 100), color=(255, 255, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=32)
    draw.text((20, 25), "HighlightHunter", fill=(0, 0, 0), font=font)

    text = extractor.extract_text(img)
    assert "HighlightHunter" in text or "Highlight" in text
