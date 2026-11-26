import numpy as np
from PIL import Image, ImageDraw
import pytest
from detector import detect_highlights, DEFAULT_THRESHOLDS

def _create_test_image(width=200, height=200, highlights=None, add_text=False):
    """Create a synthetic RGB image.
    *highlights* is a list of tuples (color_name, (x, y, w, h)).
    If *add_text* is True, draw a black line crossing the centre of each highlight.
    """
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    colour_rgb = {
        "yellow": (255, 255, 0),
        "green": (0, 255, 0),
        "pink": (255, 0, 255),
    }
    if highlights:
        for colour, (x, y, w, h) in highlights:
            draw.rectangle([x, y, x + w, y + h], fill=colour_rgb[colour])
            if add_text:
                # simple black vertical line to simulate text
                draw.line([x + w // 2, y, x + w // 2, y + h], fill="black", width=2)
    return img

@pytest.mark.parametrize("color", ["yellow", "green", "pink"])
def test_single_color_detection(color):
    img = _create_test_image(highlights=[(color, (20, 30, 60, 20))])
    detections, masks, annotated = detect_highlights(
        img,
        min_area=100,
        thresholds={color: DEFAULT_THRESHOLDS[color]},
    )
    # Should detect exactly one region of the requested color.
    assert len(detections) == 1
    det = detections[0]
    assert det["color"] == color
    x, y, w, h = det["bbox"]
    # Roughly match the drawn rectangle (allow small tolerance).
    assert abs(x - 20) <= 5 and abs(y - 30) <= 5
    assert abs(w - 60) <= 10 and abs(h - 20) <= 5
    # Mask should be non‑empty for that color.
    assert masks[color].sum() > 0
    # Annotated image must be a different object from the input.
    assert annotated != img

def test_multiple_colors_detection():
    highlights = [
        ("yellow", (10, 10, 50, 15)),
        ("green", (70, 10, 50, 15)),
        ("pink", (130, 10, 50, 15)),
    ]
    img = _create_test_image(highlights=highlights)
    detections, masks, annotated = detect_highlights(
        img,
        min_area=100,
        thresholds={c: DEFAULT_THRESHOLDS[c] for c in ["yellow", "green", "pink"]},
    )
    # Expect three detections, one per colour.
    assert len(detections) == 3
    colors_found = {d["color"] for d in detections}
    assert colors_found == {"yellow", "green", "pink"}

def test_no_highlights():
    img = Image.new("RGB", (200, 200), color="white")
    detections, masks, annotated = detect_highlights(
        img,
        min_area=100,
        thresholds={c: DEFAULT_THRESHOLDS[c] for c in ["yellow", "green", "pink"]},
    )
    assert detections == []
    # All masks should be empty (all zeros).
    for m in masks.values():
        assert m.sum() == 0

def test_highlight_with_text_interruption():
    # Add a black line (text) across the highlight.
    img = _create_test_image(highlights=[("yellow", (20, 30, 60, 20))], add_text=True)
    detections, masks, annotated = detect_highlights(
        img,
        min_area=100,
        thresholds={"yellow": DEFAULT_THRESHOLDS["yellow"]},
    )
    # Should still detect a single region despite the line.
    assert len(detections) == 1
    assert detections[0]["color"] == "yellow"

def test_small_noise_ignored():
    # Small yellow speck below min_area should be filtered out.
    img = _create_test_image(highlights=[("yellow", (10, 10, 5, 5))])
    detections, _, _ = detect_highlights(
        img,
        min_area=100,
        thresholds={"yellow": DEFAULT_THRESHOLDS["yellow"]},
    )
    assert detections == []

def test_adjacent_highlights_separate_lines():
    # Two yellow highlights on different lines should not merge.
    highlights = [
        ("yellow", (10, 10, 60, 15)),
        ("yellow", (10, 40, 60, 15)),
    ]
    img = _create_test_image(highlights=highlights)
    detections, _, _ = detect_highlights(
        img,
        min_area=100,
        thresholds={"yellow": DEFAULT_THRESHOLDS["yellow"]},
    )
    assert len(detections) == 2

# Create a sample image for manual testing (saved in assets folder).
def test_generate_sample_image(tmp_path):
    highlights = [
        ("yellow", (20, 30, 80, 20)),
        ("green", (20, 60, 80, 20)),
        ("pink", (20, 90, 80, 20)),
    ]
    img = _create_test_image(highlights=highlights, add_text=True)
    sample_path = tmp_path / "sample_notes.png"
    img.save(sample_path)
    # Verify file written.
    assert sample_path.exists()


def test_annotated_image_preserves_rgb_channels():
    # Blue background outside highlight must stay blue (not channel-swapped to red).
    img = Image.new("RGB", (200, 200), color=(0, 0, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 60, 120, 90], fill=(255, 255, 0))
    detections, _, annotated = detect_highlights(
        img,
        min_area=100,
        thresholds={"yellow": DEFAULT_THRESHOLDS["yellow"]},
    )
    assert len(detections) == 1
    assert annotated.getpixel((5, 5)) == (0, 0, 200)
