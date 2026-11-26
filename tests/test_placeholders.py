import pytest
from detector import detect_highlights
from extractor import extract_text
from pdf_generator import generate_pdf
from PIL import Image


def test_detect_highlights_returns_list():
    img = Image.new("RGB", (100, 100), color="white")
    detections, _, _ = detect_highlights(img)
    assert isinstance(detections, list)
    img = Image.new("RGB", (100, 100), color="white")
    result = detect_highlights(img)
    assert isinstance(result, list)


def test_extract_text_returns_str():
    img = Image.new("RGB", (100, 100), color="white")
    result = extract_text(img)
    assert isinstance(result, str)


def test_generate_pdf_returns_path():
    img = Image.new("RGB", (100, 100), color="white")
    path = generate_pdf(img, output_path="test_output.pdf")
    assert isinstance(path, str)
    assert path == "test_output.pdf"
