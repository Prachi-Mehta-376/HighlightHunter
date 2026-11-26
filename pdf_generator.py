def generate_pdf(image, output_path="highlight_report.pdf"):
    """Placeholder function for generating a PDF from the image and extracted text.

    Args:
        image (PIL.Image.Image): Image object.
        output_path (str): Path to save the generated PDF.
    Returns:
        str: Path to the generated PDF file.
    """
    # TODO: implement actual PDF generation using ReportLab or similar.
    # For now, just return the intended path without creating a file.
    return output_path

# ---------------------------------------------------------------------------
# New revision‑sheet PDF generator (Sprint 4 backend)
# ---------------------------------------------------------------------------

from typing import List, Dict
from io import BytesIO
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

def _draw_footer(draw: ImageDraw.Draw, page_num: int, width: int, height: int, font: ImageFont.ImageFont):
    footer_text = f"Generated with HighlightHunter – Page {page_num}"
    w = font.getbbox(footer_text)[2] - font.getbbox(footer_text)[0]
    draw.text(((width - w) / 2, height - 30), footer_text, fill="#6a0dad", font=font)

def generate_revision_pdf(snippets: List[Dict], title: str = "My Revision Notes") -> bytes:
    """Generate a revision‑sheet PDF from a list of snippet dictionaries.

    The function creates an A4‑sized PDF (595×842 px @ 72 dpi) entirely in memory.
    It raises ``ValueError`` when ``snippets`` is empty.

    Parameters
    ----------
    snippets: list of dicts
        Each dict must contain the keys produced by ``extractor.extract_snippets``:
        ``id``, ``color``, ``image`` (Pillow Image), ``width`` and ``height``.
    title: str
        Custom title shown on the first page.

    Returns
    -------
    bytes
        The binary PDF data.
    """
    if not snippets:
        raise ValueError("No snippets provided for PDF generation.")

    # A4 dimensions at 72 dpi (points) – Pillow works in pixels.
    PAGE_W, PAGE_H = 595, 842
    MARGIN = 40
    SPACING = 20

    # Fonts – fall back to the default Pillow font.
    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
        header_font = ImageFont.truetype("arial.ttf", 18)
        body_font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        title_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    pages: List[Image.Image] = []
    # First page – title and summary
    first_page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    d = ImageDraw.Draw(first_page)
    # Title
    w = title_font.getbbox(title)[2] - title_font.getbbox(title)[0]
    d.text(((PAGE_W - w) / 2, MARGIN), title, fill="#6a0dad", font=title_font)
    # Subtitle / date
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    w = header_font.getbbox(date_str)[2] - header_font.getbbox(date_str)[0]
    d.text(((PAGE_W - w) / 2, MARGIN + 40), date_str, fill="black", font=header_font)
    # Total snippets count
    count_str = f"Total snippets: {len(snippets)}"
    bbox = body_font.getbbox(count_str)
    w = bbox[2] - bbox[0]
    d.text(((PAGE_W - w) / 2, MARGIN + 80), count_str, fill="black", font=body_font)
    # Footer
    _draw_footer(d, 1, PAGE_W, PAGE_H, body_font)
    pages.append(first_page)

    # Subsequent pages – each snippet with label and image
    cur_page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(cur_page)
    y_cursor = MARGIN
    page_num = 2

    for snip in snippets:
        # Prepare label
        label = f"Snippet {snip['id']} ({snip['color']})"
        label_h = body_font.getbbox(label)[3] - body_font.getbbox(label)[1]
        # Reserve space for label + spacing
        needed_h = label_h + SPACING
        # Compute max height available for image after label
        max_img_h = PAGE_H - MARGIN - y_cursor - needed_h - 30  # reserve bottom margin & footer space
        max_img_w = PAGE_W - 2 * MARGIN

        # Determine scaled size preserving aspect ratio
        img = snip["image"]
        iw, ih = img.size
        # If there is no vertical space left, start a new page first
        if max_img_h <= 0:
            _draw_footer(draw, page_num - 1, PAGE_W, PAGE_H, body_font)
            pages.append(cur_page)
            cur_page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
            draw = ImageDraw.Draw(cur_page)
            y_cursor = MARGIN
            page_num += 1
            # recompute available space after page break
            max_img_h = PAGE_H - MARGIN - y_cursor - needed_h - 30
        scale = min(max_img_w / iw, max_img_h / ih, 1.0)
        disp_w, disp_h = int(iw * scale), int(ih * scale)
        # If the snippet does not fit vertically, start a new page
        if y_cursor + needed_h + disp_h + 30 > PAGE_H:
            # Footer for the current page
            _draw_footer(draw, page_num - 1, PAGE_W, PAGE_H, body_font)
            pages.append(cur_page)
            cur_page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
            draw = ImageDraw.Draw(cur_page)
            y_cursor = MARGIN
            page_num += 1
        # Draw label
        draw.text((MARGIN, y_cursor), label, fill="#6a0dad", font=body_font)
        y_cursor += needed_h
        # Center image horizontally
        x_offset = MARGIN + (max_img_w - disp_w) // 2
        # Paste resized image
        if scale != 1.0:
            img_resized = img.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        else:
            img_resized = img
        cur_page.paste(img_resized, (x_offset, y_cursor))
        y_cursor += disp_h + SPACING

    # Footer for last page
    _draw_footer(draw, page_num, PAGE_W, PAGE_H, body_font)
    pages.append(cur_page)

    # Save all pages to a BytesIO as PDF
    output = BytesIO()
    # Pillow expects the first image to call save with append_images for rest
    pages[0].save(output, format="PDF", save_all=True, append_images=pages[1:])
    pdf_bytes = output.getvalue()
    output.close()
    return pdf_bytes
