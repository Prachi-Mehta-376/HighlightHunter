import cv2
import numpy as np
from typing import List, Dict, Tuple
from PIL import Image

# Default HSV thresholds for highlights (tuned for typical scanned notes).
# Hue range in OpenCV is [0,179]. Saturation and Value are [0,255].
DEFAULT_THRESHOLDS = {
    "yellow": {
        "lower": (20, 80, 80),
        "upper": (40, 255, 255),
    },
    "green": {
        "lower": (39, 80, 80),  # Upper hue of yellow is 38, start green at 39 to avoid overlap
        "upper": (85, 255, 255),
    },
    "pink": {
        # Pink (magenta) often spans the hue wrap‑around region.
        "lower1": (140, 80, 80),
        "upper1": (179, 255, 255),
        "lower2": (0, 80, 80),
        "upper2": (10, 255, 255),
    },
}


def _create_mask(hsv: np.ndarray, color: str, thresholds: Dict) -> np.ndarray:
    """Create a binary mask for *color* using the supplied *thresholds*.

    Handles the special wrap‑around case for pink.
    """
    if color == "pink":
        mask1 = cv2.inRange(hsv, np.array(thresholds["lower1"]), np.array(thresholds["upper1"]))
        mask2 = cv2.inRange(hsv, np.array(thresholds["lower2"]), np.array(thresholds["upper2"]))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, np.array(thresholds["lower"]), np.array(thresholds["upper"]))
    # Morphological closing to fill small gaps (e.g., around dark text).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # (Optional) horizontal dilation can be applied later if needed.
    # For precise bounding boxes we skip dilation here to keep coordinates tight.
    # horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    # mask = cv2.dilate(mask, horiz_kernel, iterations=1)
    return mask


def _filter_contours(contours: List[np.ndarray], min_area: int, min_width: int, min_height: int, max_aspect: float) -> List[Tuple[int, int, int, int]]:
    """Filter contours by size and shape, returning bounding boxes.

    Returns a list of (x, y, w, h).
    """
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue
        if w < min_width or h < min_height:
            continue
        aspect = w / float(h)
        if aspect > max_aspect:
            continue
        boxes.append((x, y, w, h))
    return boxes


def _merge_boxes(boxes: List[Tuple[int, int, int, int]], iou_thresh: float = 0.2) -> List[Tuple[int, int, int, int]]:
    """Merge boxes that overlap significantly (IoU > *iou_thresh*).

    Simple greedy merging – adequate for small numbers of boxes.
    """
    if not boxes:
        return []
    merged = []
    for box in boxes:
        added = False
        for i, mbox in enumerate(merged):
            ix1 = max(box[0], mbox[0])
            iy1 = max(box[1], mbox[1])
            ix2 = min(box[0] + box[2], mbox[0] + mbox[2])
            iy2 = min(box[1] + box[3], mbox[1] + mbox[3])
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            union = box[2] * box[3] + mbox[2] * mbox[3] - inter
            iou = inter / float(union) if union else 0
            if iou > iou_thresh:
                nx = min(box[0], mbox[0])
                ny = min(box[1], mbox[1])
                nx2 = max(box[0] + box[2], mbox[0] + mbox[2])
                ny2 = max(box[1] + box[3], mbox[1] + mbox[3])
                merged[i] = (nx, ny, nx2 - nx, ny2 - ny)
                added = True
                break
        if not added:
            merged.append(box)
    return merged


def detect_highlights(
    pil_image: Image.Image,
    *,
    min_area: int = 500,
    min_width: int = 20,
    min_height: int = 10,
    max_aspect: float = 10.0,
    thresholds: Dict = DEFAULT_THRESHOLDS,
) -> List:
    """Detect highlighted regions in a *pil_image*.

    Parameters
    ----------
    pil_image: PIL.Image.Image
        The uploaded image (RGB).
    min_area, min_width, min_height, max_aspect:
        Simple size filters to discard noise.
    thresholds:
        Optional custom HSV thresholds dict (same structure as ``DEFAULT_THRESHOLDS``).

    Returns a list ``[detections, masks, annotated]`` where:
    - detections: list of dicts with ``color``, ``bbox`` and ``area``.
    - masks: dict of colour → binary mask.
    - annotated: PIL.Image.Image with coloured bounding boxes.
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    rgb = np.array(pil_image)
    rgb_copy = rgb.copy()
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    detections: List[Dict] = []
    masks: Dict[str, np.ndarray] = {}
    colour_rgb_map = {
        "yellow": (255, 255, 0),
        "green": (0, 255, 0),
        "pink": (255, 0, 255),
    }
    for colour in thresholds.keys():
        mask = _create_mask(hsv, colour, thresholds[colour])
        masks[colour] = mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = _filter_contours(contours, min_area, min_width, min_height, max_aspect)
        boxes = _merge_boxes(boxes)
        box_colour = colour_rgb_map.get(colour, (106, 13, 173))
        for (x, y, w, h) in boxes:
            detections.append({"color": colour, "bbox": (x, y, w, h), "area": int(w * h)})
            cv2.rectangle(rgb_copy, (x, y), (x + w, y + h), box_colour, 2)
    annotated = Image.fromarray(rgb_copy)
    return [detections, masks, annotated]
