from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, UnidentifiedImageError

import detector
import extractor
from app import (
    attach_ocr_text,
    detect_multiple_images,
    extract_multiple_snippets,
    load_uploaded_image,
)
from pdf_generator import generate_revision_pdf


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
    at.session_state["detections"] = [
        {
            "page": 1,
            "source_name": "note1.png",
            "image": Image.new("RGB", (100, 100), "white"),
            "detections": [{"color": "yellow", "bbox": (10, 10, 40, 20), "area": 800}],
            "masks": {"yellow": None},
            "annotated": Image.new("RGB", (100, 100), "white"),
        }
    ]
    at.session_state["snippets"] = None
    at.run()

    assert not at.exception


def _make_highlight_page(highlights):
    img = Image.new("RGB", (240, 200), "white")
    draw = ImageDraw.Draw(img)
    rgb_map = {"yellow": (255, 255, 0), "green": (0, 255, 0), "pink": (255, 0, 255)}
    for colour, (x, y, w, h) in highlights:
        draw.rectangle([x, y, x + w, y + h], fill=rgb_map[colour])
    return img


def test_detect_multiple_images_processes_each_independently() -> None:
    img1 = _make_highlight_page([("yellow", (20, 20, 80, 25))])
    img2 = Image.new("RGB", (240, 200), "white")  # no highlights
    img3 = _make_highlight_page([("pink", (20, 50, 80, 25))])

    results = detect_multiple_images(
        [("page1.png", img1), ("blank.png", img2), ("page3.png", img3)],
        min_area=100,
        thresholds={c: detector.DEFAULT_THRESHOLDS[c] for c in ["yellow", "green", "pink"]},
    )

    assert len(results) == 3
    assert [r["page"] for r in results] == [1, 2, 3]
    assert [r["source_name"] for r in results] == ["page1.png", "blank.png", "page3.png"]
    assert len(results[0]["detections"]) == 1
    assert results[0]["detections"][0]["color"] == "yellow"
    assert results[1]["detections"] == []
    assert len(results[2]["detections"]) == 1
    assert results[2]["detections"][0]["color"] == "pink"


def test_extract_multiple_snippets_preserves_page_and_reading_order() -> None:
    img1 = _make_highlight_page([])
    img2 = _make_highlight_page([])
    img3 = _make_highlight_page([])
    page_results = [
        {
            "page": 1,
            "source_name": "p1.png",
            "image": img1,
            # Intentionally unsorted on page 1
            "detections": [
                {"color": "green", "bbox": (100, 80, 40, 20), "area": 800},
                {"color": "yellow", "bbox": (100, 10, 40, 20), "area": 800},
                {"color": "pink", "bbox": (10, 10, 40, 20), "area": 800},
            ],
        },
        {
            "page": 2,
            "source_name": "empty.png",
            "image": img2,
            "detections": [],
        },
        {
            "page": 3,
            "source_name": "p3.png",
            "image": img3,
            "detections": [
                {"color": "yellow", "bbox": (15, 30, 50, 20), "area": 1000},
            ],
        },
    ]

    snippets = extract_multiple_snippets(page_results, padding=5)

    assert [s["id"] for s in snippets] == [1, 2, 3, 4]
    assert [(s["page"], s["source_name"], s["color"]) for s in snippets] == [
        (1, "p1.png", "pink"),    # y=10, x=10
        (1, "p1.png", "yellow"),  # y=10, x=100
        (1, "p1.png", "green"),   # y=80, x=100
        (3, "p3.png", "yellow"),  # page 3 after skipping empty page 2
    ]


def test_multi_image_pipeline_combined_pdf_with_selection_and_ocr(monkeypatch) -> None:
    img1 = _make_highlight_page([("yellow", (20, 20, 80, 25))])
    img2 = Image.new("RGB", (240, 200), "white")  # no highlights
    img3 = _make_highlight_page([
        ("green", (20, 20, 80, 25)),
        ("pink", (20, 90, 80, 25)),
    ])

    page_results = detect_multiple_images(
        [("lec1.png", img1), ("lec2_blank.png", img2), ("lec3.png", img3)],
        min_area=100,
        thresholds={c: detector.DEFAULT_THRESHOLDS[c] for c in ["yellow", "green", "pink"]},
    )
    snippets = extract_multiple_snippets(page_results, padding=5)

    ocr_map = {1: "First page concept", 2: "Third page excluded note", 3: "Third page included note"}
    monkeypatch.setattr(extractor, "extract_text", lambda _img: ocr_map[next_ocr()])
    counter = iter([1, 2, 3])
    def next_ocr():
        return next(counter)

    snippets, err = attach_ocr_text(snippets)
    assert err is None
    assert len(snippets) == 3

    # User deselects snippet 2
    selected = [s for s in snippets if s["id"] in {1, 3}]
    pdf_bytes = generate_revision_pdf(selected, title="Multi-Image Revision")

    from pypdf import PdfReader
    reader = PdfReader(BytesIO(pdf_bytes))
    assert len(reader.pages) >= 2
    full_text = "\n".join(page.extract_text() for page in reader.pages)
    assert "Page 1 (lec1.png) – Snippet 1 (yellow)" in full_text
    assert "First page concept" in full_text
    assert "Page 3 (lec3.png) – Snippet 3 (pink)" in full_text
    assert "Third page included note" in full_text
    assert "Third page excluded note" not in full_text
