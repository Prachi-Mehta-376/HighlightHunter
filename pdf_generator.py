"""PDF generation module using ReportLab for HighlightHunter."""

from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List

from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen import canvas

import extractor

PURPLE_ACCENT = HexColor("#6a0dad")
TEXT_DARK = HexColor("#1f1f1f")
TEXT_MUTED = HexColor("#555555")
PAGE_W, PAGE_H = A4
MARGIN = 40
FOOTER_SPACE = 35
USABLE_W = PAGE_W - 2 * MARGIN


def _to_rgb_reader(image: Image.Image) -> ImageReader:
    """Return a ReportLab ImageReader without mutating the source image."""
    rgb_img = image if image.mode == "RGB" else image.convert("RGB")
    return ImageReader(rgb_img)


def _wrap_text_lines(text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
    """Wrap multi-line text into lines fitting within *max_width* points."""
    wrapped: List[str] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            wrapped.append("")
        else:
            wrapped.extend(simpleSplit(raw_line, font_name, font_size, max_width))
    return wrapped


def _draw_footer(c: canvas.Canvas, page_num: int) -> None:
    """Draw a centered purple footer with the page number."""
    c.saveState()
    c.setFont("Helvetica", 9)
    c.setFillColor(PURPLE_ACCENT)
    footer_text = f"Generated with HighlightHunter – Page {page_num}"
    c.drawCentredString(PAGE_W / 2.0, 22, footer_text)
    c.restoreState()


def generate_pdf(image, output_path="highlight_report.pdf"):
    """Generate a single-image highlight report PDF using ReportLab.

    Args:
        image (PIL.Image.Image): Image object.
        output_path (str): Path to save the generated PDF.
    Returns:
        str: Path to the generated PDF file.
    """
    out_file = Path(output_path)
    if out_file.parent and str(out_file.parent) != ".":
        out_file.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(out_file), pagesize=A4)
    c.setTitle("HighlightHunter Report")

    # Header
    y_cursor = PAGE_H - MARGIN
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(PURPLE_ACCENT)
    c.drawString(MARGIN, y_cursor - 18, "HighlightHunter Report")
    y_cursor -= 32

    c.setFont("Helvetica", 10)
    c.setFillColor(TEXT_MUTED)
    c.drawString(MARGIN, y_cursor, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y_cursor -= 20

    # Scaled image preview
    iw, ih = image.size
    max_img_h = 340.0
    scale = min(USABLE_W / float(iw), max_img_h / float(ih), 1.0)
    disp_w, disp_h = iw * scale, ih * scale
    x_offset = MARGIN + (USABLE_W - disp_w) / 2.0
    c.drawImage(_to_rgb_reader(image), x_offset, y_cursor - disp_h, width=disp_w, height=disp_h)
    y_cursor -= disp_h + 24

    # Extracted OCR text (best effort if Tesseract is missing)
    extracted_text = ""
    try:
        extracted_text = extractor.extract_text(image)
    except extractor.TesseractNotInstalledError:
        extracted_text = ""

    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(PURPLE_ACCENT)
    c.drawString(MARGIN, y_cursor, "Extracted Text")
    y_cursor -= 18

    body_text = extracted_text if extracted_text else "(No text extracted)"
    lines = _wrap_text_lines(body_text, "Helvetica", 10, USABLE_W)
    page_num = 1
    c.setFont("Helvetica", 10)
    c.setFillColor(TEXT_DARK)
    for line in lines:
        if y_cursor < MARGIN + FOOTER_SPACE:
            _draw_footer(c, page_num)
            c.showPage()
            page_num += 1
            y_cursor = PAGE_H - MARGIN
            c.setFont("Helvetica", 10)
            c.setFillColor(TEXT_DARK)
        if line:
            c.drawString(MARGIN, y_cursor, line)
        y_cursor -= 14

    _draw_footer(c, page_num)
    c.showPage()
    c.save()
    return str(output_path)


def generate_revision_pdf(snippets: List[Dict], title: str = "My Revision Notes") -> bytes:
    """Generate a revision-sheet PDF from a list of snippet dictionaries using ReportLab.

    Creates an A4 PDF in memory with a title/summary first page followed by
    snippet pages containing each snippet's label, image, OCR text, and page footer.
    Raises ``ValueError`` when ``snippets`` is empty.

    Parameters
    ----------
    snippets: list of dicts
        Each dict must contain the keys produced by ``extractor.extract_snippets``:
        ``id``, ``color``, ``image`` (Pillow Image), ``width`` and ``height``,
        and may optionally include ``text`` (str).
    title: str
        Custom title shown on the first page.

    Returns
    -------
    bytes
        The binary PDF data.
    """
    if not snippets:
        raise ValueError("No snippets provided for PDF generation.")

    output = BytesIO()
    c = canvas.Canvas(output, pagesize=A4)
    pdf_title = (title or "My Revision Notes").strip() or "My Revision Notes"
    c.setTitle(pdf_title)

    # --- Page 1: Title & Summary ---
    title_lines = _wrap_text_lines(pdf_title, "Helvetica-Bold", 24, USABLE_W)
    y_title = PAGE_H - MARGIN - 40
    c.setFont("Helvetica-Bold", 24)
    c.setFillColor(PURPLE_ACCENT)
    for line in title_lines:
        c.drawCentredString(PAGE_W / 2.0, y_title, line)
        y_title -= 30

    c.setFont("Helvetica", 14)
    c.setFillColor(TEXT_DARK)
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    c.drawCentredString(PAGE_W / 2.0, y_title - 12, date_str)

    c.setFont("Helvetica", 12)
    c.setFillColor(TEXT_MUTED)
    c.drawCentredString(PAGE_W / 2.0, y_title - 42, f"Total snippets: {len(snippets)}")

    _draw_footer(c, 1)
    c.showPage()

    # --- Subsequent Pages: Snippet Images + OCR Text ---
    page_num = 2
    y_cursor = PAGE_H - MARGIN
    min_y = MARGIN + FOOTER_SPACE
    max_single_page_img_h = PAGE_H - 2 * MARGIN - FOOTER_SPACE - 60

    def start_new_page() -> None:
        nonlocal page_num, y_cursor
        _draw_footer(c, page_num)
        c.showPage()
        page_num += 1
        y_cursor = PAGE_H - MARGIN

    for snip in snippets:
        label = f"Snippet {snip['id']} ({snip['color']})"
        img = snip["image"]
        iw, ih = img.size
        scale = min(USABLE_W / float(iw), max_single_page_img_h / float(ih), 1.0)
        disp_w, disp_h = iw * scale, ih * scale

        ocr_text = (snip.get("text") or "").strip()
        wrapped_text = _wrap_text_lines(ocr_text, "Helvetica", 10, USABLE_W) if ocr_text else []
        # Reserve enough height for label + image + at least the first few lines of text if present
        min_text_h = (18 + min(len(wrapped_text), 2) * 14) if wrapped_text else 0
        needed_start_h = 22 + disp_h + 10 + min_text_h

        # Start a new page if the snippet header + image won't fit in remaining space
        if y_cursor - needed_start_h < min_y and y_cursor < PAGE_H - MARGIN:
            start_new_page()

        # Draw snippet label
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(PURPLE_ACCENT)
        c.drawString(MARGIN, y_cursor - 12, label)
        y_cursor -= 22

        # Draw snippet image centered horizontally
        x_offset = MARGIN + (USABLE_W - disp_w) / 2.0
        c.drawImage(_to_rgb_reader(img), x_offset, y_cursor - disp_h, width=disp_w, height=disp_h)
        y_cursor -= disp_h + 12

        # Draw OCR text if available
        if wrapped_text:
            if y_cursor - 16 < min_y:
                start_new_page()
            c.setFont("Helvetica-Bold", 10)
            c.setFillColor(TEXT_MUTED)
            c.drawString(MARGIN, y_cursor - 10, "Extracted Text:")
            y_cursor -= 18

            c.setFont("Helvetica", 10)
            c.setFillColor(TEXT_DARK)
            for line in wrapped_text:
                if y_cursor - 14 < min_y:
                    start_new_page()
                    c.setFont("Helvetica", 10)
                    c.setFillColor(TEXT_DARK)
                if line:
                    c.drawString(MARGIN, y_cursor - 10, line)
                y_cursor -= 14

        y_cursor -= 16

    _draw_footer(c, page_num)
    c.showPage()
    c.save()

    pdf_bytes = output.getvalue()
    output.close()
    return pdf_bytes
