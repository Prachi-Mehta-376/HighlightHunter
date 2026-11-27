import pytest
from io import BytesIO
from PIL import Image
import numpy as np
import extractor
import detector
from pdf_generator import generate_pdf, generate_revision_pdf

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


def test_generate_pdf_creates_valid_pdf_with_text(tmp_path, monkeypatch):
    img = make_image("yellow", (200, 100))
    out_file = tmp_path / "single_report.pdf"
    monkeypatch.setattr(extractor, "extract_text", lambda _img: "Key formula: E = mc^2")

    returned_path = generate_pdf(img, output_path=str(out_file))
    assert returned_path == str(out_file)
    assert out_file.exists()

    from pypdf import PdfReader
    reader = PdfReader(str(out_file))
    assert len(reader.pages) == 1
    page_text = reader.pages[0].extract_text()
    assert "HighlightHunter Report" in page_text
    assert "Key formula: E = mc^2" in page_text


def test_generate_revision_pdf_includes_ocr_text():
    img1 = make_image("yellow", (200, 60))
    img2 = make_image("green", (200, 60))
    snippets = [
        {"id": 1, "color": "yellow", "image": img1, "width": 200, "height": 60, "text": "Photosynthesis converts light energy."},
        {"id": 2, "color": "green", "image": img2, "width": 200, "height": 60, "text": "Mitochondria generate ATP."},
    ]
    pdf_bytes = generate_revision_pdf(snippets, title="Biology Notes")

    from pypdf import PdfReader
    reader = PdfReader(BytesIO(pdf_bytes))
    assert len(reader.pages) == 2

    page0_text = reader.pages[0].extract_text()
    page1_text = reader.pages[1].extract_text()
    assert "Biology Notes" in page0_text
    assert "Total snippets: 2" in page0_text
    assert "Page 1" in page0_text
    assert "Snippet 1 (yellow)" in page1_text
    assert "Photosynthesis converts light energy." in page1_text
    assert "Snippet 2 (green)" in page1_text
    assert "Mitochondria generate ATP." in page1_text
    assert "Page 2" in page1_text


def test_generate_revision_pdf_long_ocr_text_paginates():
    img = make_image("pink", (300, 120))
    long_text = " ".join([f"Sentence {i}: important revision detail for exam prep." for i in range(1, 140)])
    snippets = [
        {"id": 1, "color": "pink", "image": img, "width": 300, "height": 120, "text": long_text},
    ]
    pdf_bytes = generate_revision_pdf(snippets, title="Long OCR Test")

    from pypdf import PdfReader
    reader = PdfReader(BytesIO(pdf_bytes))
    assert len(reader.pages) >= 3
    all_text = "\n".join(page.extract_text() for page in reader.pages)
    assert "Sentence 1:" in all_text
    assert "Sentence 139:" in all_text


def test_generate_revision_pdf_includes_page_headings_for_multi_image_snippets():
    img1 = make_image("yellow", (200, 60))
    img2 = make_image("pink", (200, 60))
    snippets = [
        {
            "id": 1,
            "page": 1,
            "source_name": "notes_p1.png",
            "color": "yellow",
            "image": img1,
            "width": 200,
            "height": 60,
            "text": "Definition from page one.",
        },
        {
            "id": 2,
            "page": 2,
            "source_name": "notes_p2.png",
            "color": "pink",
            "image": img2,
            "width": 200,
            "height": 60,
            "text": "Formula from page two.",
        },
    ]
    pdf_bytes = generate_revision_pdf(snippets, title="Combined Study Guide")

    from pypdf import PdfReader
    reader = PdfReader(BytesIO(pdf_bytes))
    assert len(reader.pages) == 2

    page0_text = reader.pages[0].extract_text()
    page1_text = reader.pages[1].extract_text()
    assert "Combined Study Guide" in page0_text
    assert "Total snippets: 2" in page0_text
    assert "Source pages: 2" in page0_text
    assert "Page 1 (notes_p1.png) – Snippet 1 (yellow)" in page1_text
    assert "Definition from page one." in page1_text
    assert "Page 2 (notes_p2.png) – Snippet 2 (pink)" in page1_text
    assert "Formula from page two." in page1_text


def test_generate_revision_pdf_wraps_long_source_filename_heading():
    img = make_image("green", (200, 60))
    long_name = "very_long_lecture_notes_scan_chapter_12_section_4_highlighted_copy.png"
    snippets = [
        {
            "id": 1,
            "page": 3,
            "source_name": long_name,
            "color": "green",
            "image": img,
            "width": 200,
            "height": 60,
            "text": "Wrapped heading check.",
        }
    ]
    pdf_bytes = generate_revision_pdf(snippets, title="Long Filename Test")

    from pypdf import PdfReader
    reader = PdfReader(BytesIO(pdf_bytes))
    assert len(reader.pages) == 2
    page1_text = reader.pages[1].extract_text().replace("\n", " ")
    assert "Page 3" in page1_text
    assert "Snippet 1 (green)" in page1_text
    assert "Wrapped heading check." in page1_text
