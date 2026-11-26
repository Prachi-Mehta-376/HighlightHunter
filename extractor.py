"""Extractor module for cropping highlighted snippets and running OCR."""

from typing import List, Dict
import cv2
import numpy as np
import pytesseract
from PIL import Image


class TesseractNotInstalledError(RuntimeError):
    """Raised when the Tesseract OCR binary is not installed or not on PATH."""


def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Convert a snippet image to a clean grayscale/binarized image for OCR."""
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Upscale small snippets so Tesseract can resolve small font glyphs reliably.
    height, width = gray.shape[:2]
    if height < 80 or width < 200:
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Light Gaussian blur + Otsu binarization separates dark text from highlight fills.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)


def extract_text(image: Image.Image) -> str:
    """Extract text from a highlighted image region using pytesseract OCR.

    Args:
        image: Pillow Image (not modified in place).
    Returns:
        Extracted text string (whitespace-trimmed).
    Raises:
        TesseractNotInstalledError: If the Tesseract binary cannot be found.
    """
    processed = _preprocess_for_ocr(image)
    try:
        text = pytesseract.image_to_string(processed, config="--psm 6")
    except pytesseract.TesseractNotFoundError as exc:
        raise TesseractNotInstalledError(
            "Tesseract OCR is not installed or not found on PATH. "
            "Install Tesseract (e.g., `brew install tesseract` on macOS, "
            "`sudo apt-get install tesseract-ocr` on Linux) and restart the app."
        ) from exc
    return text.strip()


def extract_snippets(
    image: Image.Image,
    detections: List[Dict],
    padding: int = 5,
) -> List[Dict]:
    """Extract cropped snippet images from the original *image* based on *detections*.

    Parameters
    ----------
    image: PIL.Image.Image
        The original RGB image (must not be modified).
    detections: list of dict
        Each dict must contain ``color`` (str) and ``bbox`` (x, y, w, h).
    padding: int, default 5
        Number of pixels to pad each side of the bounding box. Padding is
        clamped to image boundaries.

    Returns
    -------
    snippets: list of dict
        Each dict contains:
        - ``id``: unique integer identifier (starting at 1, ordered top-to-bottom, left-to-right).
        - ``color``: highlight colour.
        - ``orig_bbox``: original (x, y, w, h).
        - ``padded_bbox``: (x, y, w, h) after padding.
        - ``image``: cropped Pillow Image.
        - ``width`` and ``height`` of the cropped image.
    """
    if not detections:
        return []
    img_w, img_h = image.size
    snippets: List[Dict] = []
    for det in detections:
        x, y, w, h = det["bbox"]
        # Apply padding and clamp to image bounds.
        if x < 0:
            x0 = 0
            x1 = min(w + padding, img_w)
        else:
            x0 = max(x - padding, 0)
            x1 = min(x + w + padding, img_w)
        if y < 0:
            y0 = 0
            y1 = min(h + padding, img_h)
        else:
            y0 = max(y - padding, 0)
            y1 = min(y + h + padding, img_h)
        padded_bbox = (x0, y0, x1 - x0, y1 - y0)
        cropped = image.crop((x0, y0, x1, y1))
        snippet = {
            "id": 0,
            "color": det["color"],
            "orig_bbox": (x, y, w, h),
            "padded_bbox": padded_bbox,
            "image": cropped,
            "width": cropped.width,
            "height": cropped.height,
        }
        snippets.append(snippet)
    # Sort snippets in natural reading order: top-to-bottom then left-to-right,
    # then assign sequential IDs starting at 1.
    snippets.sort(key=lambda s: (s["orig_bbox"][1], s["orig_bbox"][0]))
    for idx, snippet in enumerate(snippets, start=1):
        snippet["id"] = idx
    return snippets
