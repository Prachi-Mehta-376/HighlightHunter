# -*- coding: utf-8 -*-
"""Generate synthetic demo images with highlighted text for HighlightHunter.
The images are saved under assets/demo/ as required.
"""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Ensure output directory exists
out_dir = Path(__file__).parent / "assets" / "demo"
out_dir.mkdir(parents=True, exist_ok=True)

# Basic settings
img_width, img_height = 1200, 1600
bg_color = (255, 255, 255)
text_color = (0, 0, 0)
line_spacing = 40
margin = 50
font = ImageFont.load_default()

# Helper to draw multiline text with optional highlight
def create_image(lines, highlights, filename):
    img = Image.new("RGB", (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(img)
    y = margin
    for idx, line in enumerate(lines, start=1):
        # Determine if this line should be highlighted
        hl = highlights.get(idx)
        if hl:
            # Compute text size for rectangle
            bbox = draw.textbbox((0, 0), line, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            rect_y0 = y
            rect_y1 = y + text_h
            rect_x0 = margin - 5
            rect_x1 = margin + text_w + 5
            # Highlight colours: "yellow", "green", "pink"
            colour_map = {
                "yellow": (255, 255, 0),
                "green": (0, 255, 0),
                "pink": (255, 192, 203),
            }
            draw.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=colour_map[hl])
        # Draw the text over the highlight (or plain)
        draw.text((margin, y), line, fill=text_color, font=font)
        y += line_spacing
    img.save(out_dir / filename)

# Content for the demo images (short but representative)
common_lines = [
    "1. Computer Vision is the field that enables machines to see.",
    "2. Image preprocessing includes resizing, filtering, and colour conversion.",
    "3. Convolutional Neural Networks have revolutionized visual recognition.",
    "4. Classical techniques rely on edge detection and colour segmentation.",
    "5. HSV colour space separates intensity from chromatic information.",
    "6. Morphological operations help clean binary masks.",
    "7. Bounding boxes are used to localise regions of interest.",
    "8. PDF generation can compile extracted snippets for revision.",
    "9. Proper documentation ensures reproducibility and future work.",
    "10. This synthetic note demonstrates HighlightHunter capabilities."
]

# 1. yellow_highlights.png – only the first three lines highlighted in yellow
highlights_yellow = {1: "yellow", 2: "yellow", 3: "yellow"}
create_image(common_lines, highlights_yellow, "yellow_highlights.png")

# 2. multicolor_highlights.png – mix of colours across lines
highlights_multi = {1: "yellow", 3: "green", 5: "pink", 7: "yellow", 9: "green"}
create_image(common_lines, highlights_multi, "multicolor_highlights.png")

# 3. handwritten_style_notes.png – simulate handwritten by using a different font if available.
# Fallback to default if not found.
try:
    hand_font = ImageFont.truetype("arial.ttf", 24)
except Exception:
    hand_font = ImageFont.load_default()

def create_handwritten_image():
    img = Image.new("RGB", (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(img)
    y = margin
    for idx, line in enumerate(common_lines, start=1):
        # Randomly highlight some lines
        if idx % 3 == 0:
            hl = "yellow"
            colour_map = {"yellow": (255, 255, 0)}
            # Compute text size for rectangle
            bbox = draw.textbbox((0, 0), line, font=hand_font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.rectangle([margin-5, y, margin+text_w+5, y+text_h], fill=colour_map[hl])
        draw.text((margin, y), line, fill=text_color, font=hand_font)
        y += line_spacing
    img.save(out_dir / "handwritten_style_notes.png")

create_handwritten_image()

# 4. no_highlights.png – plain notes without any highlights
create_image(common_lines, {}, "no_highlights.png")

# 5. long_highlighted_sentences.png – a very long sentence that wraps and is highlighted
long_line = "11. " + "This is a very long highlighted sentence that spans multiple lines when rendered, testing the padding and clipping logic of the extractor module, ensuring that even when highlights touch image boundaries they are correctly processed."
lines_long = common_lines + [long_line]
highlights_long = {11: "pink"}
create_image(lines_long, highlights_long, "long_highlighted_sentences.png")
