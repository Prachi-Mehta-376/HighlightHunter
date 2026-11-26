"""Extractor module for cropping highlighted snippets from an image.
Provides a placeholder `extract_text` (for compatibility) and a full
`extract_snippets` implementation used by the Streamlit UI.
"""

from typing import List, Dict
from PIL import Image
from io import BytesIO


def extract_text(image: Image.Image) -> str:
    """Placeholder function for extracting text from highlighted regions.

    Args:
        image: Pillow Image.
    Returns:
        Empty string (OCR not implemented).
    """
    # TODO: implement OCR extraction (e.g., using pytesseract)
    return ""


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
        - ``id``: unique integer identifier (starting at 1).
        - ``color``: highlight colour.
        - ``orig_bbox``: original (x, y, w, h).
        - ``padded_bbox``: (x, y, w, h) after padding.
        - ``image``: cropped Pillow Image.
        - ``width`` and ``height`` of the cropped image.
    """
    if not detections:
        return []
    from io import BytesIO
    # Work on a copy to avoid mutating the original image.
    img_w, img_h = image.size
    snippets: List[Dict] = []
    for idx, det in enumerate(detections, start=1):
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
            "id": idx,
            "color": det["color"],
            "orig_bbox": (x, y, w, h),
            "padded_bbox": padded_bbox,
            "image": cropped,
            "width": cropped.width,
            "height": cropped.height,
        }
        snippets.append(snippet)
    # Sort snippets in natural reading order: top‑to‑bottom then left‑to‑right.
    snippets.sort(key=lambda s: (s["orig_bbox"][1], s["orig_bbox"][0]))
    return snippets
