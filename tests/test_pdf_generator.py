import pytest
from io import BytesIO
from PIL import Image
import numpy as np
import extractor
import detector
from pdf_generator import generate_revision_pdf

# Helper to create a simple colored image
def make_image(color, size=(200, 100)):
    img = Image.new("RGB", size, color=color)
    return img

def test_generate_single_snippet_pdf():
    img = make_image("yellow", (200, 100))
    detections = [{"color": "yellow", "bbox": (20, 10, 50, 30), "area": 1500}]
    snippets = extractor.extract_snippets(img, detections, padding=0)
    pdf_bytes = generate_revision_pdf(snippets, title="Test PDF")
    assert isinstance(pdf_bytes, bytes)
    # Verify PDF can be opened and has one page (first page + snippet page = 2)
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(pdf_bytes))
    assert len(reader.pages) == 2
    # A4 dimensions in points (1 point = 1/72 inch) ~ 595x842
    page0 = reader.pages[0]
    w = float(page0.mediabox.upper_right[0])
    h = float(page0.mediabox.upper_right[1])
    assert abs(w - 595) < 5 and abs(h - 842) < 5

def test_generate_multiple_snippets_multi_page():
    img = make_image("green", (400, 800))
    # Create many detections to force multiple pages
    detections = []
    y = 0
    for i in range(10):
        detections.append({"color": "green", "bbox": (10, y, 380, 70), "area": 380*70})
        y += 80
    snippets = extractor.extract_snippets(img, detections, padding=5)
    pdf_bytes = generate_revision_pdf(snippets)
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(pdf_bytes))
    # Should be more than 2 pages (first page + at least 2 snippet pages)
    assert len(reader.pages) >= 3

def test_empty_snippets_raises():
    with pytest.raises(ValueError):
        generate_revision_pdf([])

def test_wide_and_tall_snippet_scaling():
    # Wide image
    wide = make_image("pink", (800, 200))
    det_wide = [{"color": "pink", "bbox": (0, 0, 800, 200), "area": 160000}]
    snip_wide = extractor.extract_snippets(wide, det_wide, padding=0)
    # Tall image
    tall = make_image("yellow", (200, 800))
    det_tall = [{"color": "yellow", "bbox": (0, 0, 200, 800), "area": 160000}]
    snip_tall = extractor.extract_snippets(tall, det_tall, padding=0)
    pdf_bytes = generate_revision_pdf(snip_wide + snip_tall)
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(pdf_bytes))
    assert len(reader.pages) >= 2

def test_original_image_immutability():
    img = make_image("green", (300, 300))
    orig_copy = img.copy()
    det = [{"color": "green", "bbox": (50, 50, 100, 100), "area": 10000}]
    snippets = extractor.extract_snippets(img, det, padding=10)
    # Ensure original image unchanged
    assert list(img.getdata()) == list(orig_copy.getdata())

def test_integration_with_detector_and_extractor():
    # Create a synthetic image with two coloured boxes
    base = Image.new("RGB", (200, 200), "white")
    np_img = np.array(base)
    np_img[20:70, 30:130] = (255, 255, 0)   # yellow box
    np_img[120:170, 50:150] = (0, 255, 0)   # green box
    img = Image.fromarray(np_img)
    detections, _, _ = detector.detect_highlights(
        img,
        min_area=100,
        thresholds={"yellow": detector.DEFAULT_THRESHOLDS["yellow"], "green": detector.DEFAULT_THRESHOLDS["green"]},
    )
    snippets = extractor.extract_snippets(img, detections, padding=5)
    pdf_bytes = generate_revision_pdf(snippets)
    from pypdf import PdfReader
    reader = PdfReader(BytesIO(pdf_bytes))
    assert len(reader.pages) >= 2
    # Ensure at least one snippet present
    assert len(snippets) >= 2
