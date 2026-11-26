import streamlit as st
import detector
import extractor
from io import BytesIO
from PIL import Image
from pdf_generator import generate_revision_pdf


# Helper to load image bytes and ensure RGB (used by tests and UI)
def load_uploaded_image(data: bytes) -> Image.Image:
    """Load image bytes, ensure RGB mode, and return a Pillow Image.

    Args:
        data: Raw image bytes (e.g., from an uploaded file).
    Returns:
        Pillow Image in RGB mode.
    """
    img = Image.open(BytesIO(data))
    return img.convert("RGB")

from typing import Dict, List, Optional, Tuple


def attach_ocr_text(snippets: List[Dict]) -> Tuple[List[Dict], Optional[str]]:
    """Run OCR on each extracted snippet and attach a ``text`` field.

    Args:
        snippets: List of snippet dicts produced by ``extractor.extract_snippets``.
    Returns:
        Tuple of ``(snippets, ocr_error)``, where ``ocr_error`` is a user-facing
        message if Tesseract OCR is missing, or ``None`` otherwise.
    """
    ocr_error: Optional[str] = None
    for snip in snippets:
        try:
            snip["text"] = extractor.extract_text(snip["image"])
        except extractor.TesseractNotInstalledError as exc:
            snip["text"] = ""
            if ocr_error is None:
                ocr_error = str(exc)
    return snippets, ocr_error


def _invalidate_pdf() -> None:
    """Clear any previously generated PDF bytes when selection or text changes."""
    st.session_state.pdf_bytes = None


def main():
    st.set_page_config(page_title="HighlightHunter", layout="wide")

    # Custom CSS for purple/lavender theme
    st.markdown(
        """
        <style>
        .stApp {background-color: #f8f6ff;}
        .css-1v0mbdj {color: #6a0dad;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Title & description
    st.title("HighlightHunter")
    st.markdown("Detect highlighted text in study notes and generate revision sheets.")

    # Sidebar – image upload and detection settings
    st.sidebar.title("Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
    )

    # Detection configuration (persist in session state)
    if "detect_config" not in st.session_state:
        st.session_state.detect_config = {
            "colors": ["yellow", "green", "pink"],
            "min_area": 500,
        }
    colors = st.sidebar.multiselect(
        "Highlight colours to detect",
        options=["yellow", "green", "pink"],
        default=st.session_state.detect_config["colors"],
    )
    min_area = st.sidebar.slider(
        "Minimum region area (pixels)", min_value=100, max_value=5000,
        value=st.session_state.detect_config["min_area"], step=100
    )
    # Crop padding slider (default 5)
    if "crop_padding" not in st.session_state:
        st.session_state.crop_padding = 5
    padding = st.sidebar.slider(
        "Crop padding (pixels)", min_value=0, max_value=50, value=st.session_state.crop_padding, step=1, key="crop_padding"
    )

    # Reset detection results when settings change or new image uploaded
    if uploaded_file is not None:
        if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
            st.session_state.last_file = uploaded_file.name
            st.session_state.detections = None
            st.session_state.snippets = None
            st.session_state.ocr_error = None
            st.session_state.pdf_bytes = None
    if colors != st.session_state.detect_config["colors"] or min_area != st.session_state.detect_config["min_area"]:
        st.session_state.detect_config["colors"] = colors
        st.session_state.detect_config["min_area"] = min_area
        st.session_state.detections = None
        st.session_state.snippets = None
        st.session_state.ocr_error = None
        st.session_state.pdf_bytes = None
    if padding != st.session_state.get("crop_padding"):
        st.session_state.crop_padding = padding
        st.session_state.snippets = None
        st.session_state.ocr_error = None
        st.session_state.pdf_bytes = None

    if uploaded_file is not None:
        try:
            image = load_uploaded_image(uploaded_file.getvalue())
            st.image(image, caption="Uploaded Image", use_column_width=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Detect Highlights"):
                    detections, masks, annotated = detector.detect_highlights(
                        image,
                        min_area=min_area,
                        thresholds={c: detector.DEFAULT_THRESHOLDS[c] for c in colors},
                    )
                    st.session_state.detections = {
                        "list": detections,
                        "masks": masks,
                        "annotated": annotated,
                    }
                    # Invalidate any previously extracted snippets and PDF
                    st.session_state.snippets = None
                    st.session_state.ocr_error = None
                    st.session_state.pdf_bytes = None

            with col2:
                if st.session_state.get("detections"):
                    st.subheader("Detection Results")
                    st.image(st.session_state.detections["annotated"], caption="Annotated Image", use_column_width=True)
                    total = len(st.session_state.detections["list"])
                    st.write(f"**Total regions detected:** {total}")
                    colour_counts = {c: 0 for c in colors}
                    for det in st.session_state.detections["list"]:
                        colour_counts[det["color"]] += 1
                    for c, cnt in colour_counts.items():
                        st.write(f"{c.title()}: {cnt}")
                    for c, mask in st.session_state.detections["masks"].items():
                        if c in colors:
                            st.image(mask, caption=f"{c.title()} mask", clamp=True, use_column_width=False)
                else:
                    st.info("Upload an image and click **Detect Highlights** to see results.")

            # ------- Extraction UI (appears after detection) -------
            if st.session_state.get("detections"):
                st.subheader("Extract Snippets")
                if st.button("Extract Highlights"):
                    # Run extraction using current image and detection list
                    snippets = extractor.extract_snippets(
                        image,
                        st.session_state.detections["list"],
                        padding=st.session_state.crop_padding,
                    )
                    snippets, ocr_error = attach_ocr_text(snippets)
                    st.session_state.snippets = snippets
                    st.session_state.ocr_error = ocr_error
                    # Initialise selection and editable OCR text state
                    st.session_state.selection = {snip["id"]: True for snip in snippets}
                    for snip in snippets:
                        st.session_state[f"select_{snip['id']}"] = True
                        st.session_state[f"text_{snip['id']}"] = snip["text"]
                    st.session_state.pdf_bytes = None

                if st.session_state.get("ocr_error"):
                    st.warning(st.session_state["ocr_error"])

                # Show extracted snippets and PDF export only after extraction
                if st.session_state.get("snippets") is not None:
                    if not st.session_state.snippets:
                        st.info("No highlighted snippets were extracted.")
                    else:
                        for snip in st.session_state.snippets:
                            col_img, col_ctrl = st.columns([3, 1])
                            with col_img:
                                st.image(snip["image"], caption=f"Snippet {snip['id']} ({snip['color']})", use_column_width=True)
                                text_key = f"text_{snip['id']}"
                                if text_key not in st.session_state:
                                    st.session_state[text_key] = snip.get("text", "")
                                snip["text"] = st.text_area(
                                    f"Extracted Text (Snippet {snip['id']})",
                                    key=text_key,
                                    height=90,
                                    on_change=_invalidate_pdf,
                                )
                            with col_ctrl:
                                # Selection checkbox
                                key_chk = f"select_{snip['id']}"
                                if key_chk not in st.session_state:
                                    st.session_state[key_chk] = True
                                st.session_state.selection[snip['id']] = st.checkbox(
                                    "Include",
                                    value=st.session_state[key_chk],
                                    key=key_chk,
                                    on_change=_invalidate_pdf,
                                )
                                # PNG download button
                                buf = BytesIO()
                                snip["image"].save(buf, format="PNG")
                                st.download_button(
                                    label="Download PNG",
                                    data=buf.getvalue(),
                                    file_name=f"snippet_{snip['id']}.png",
                                    mime="image/png",
                                    key=f"download_png_{snip['id']}",
                                )

                        # PDF generation section
                        st.subheader("Generate Revision Sheet")
                        title = st.text_input(
                            "PDF Title",
                            value="My Revision Notes",
                            key="revision_title",
                            on_change=_invalidate_pdf,
                        )
                        selected_snips = [
                            snip for snip in st.session_state.snippets
                            if st.session_state.selection.get(snip['id'], False)
                        ]
                        total = len(st.session_state.snippets)
                        selected = len(selected_snips)
                        excluded = total - selected
                        st.write(f"Total snippets: {total} | Selected: {selected} | Excluded: {excluded}")

                        if selected > 0:
                            if st.button("Generate Revision PDF"):
                                try:
                                    pdf_bytes = generate_revision_pdf(selected_snips, title=title)
                                    st.session_state.pdf_bytes = pdf_bytes
                                    st.success("PDF generated successfully.")
                                except Exception as e:
                                    st.error(f"Failed to generate PDF: {e}")

                        if st.session_state.get("pdf_bytes"):
                            st.download_button(
                                label="Download Revision PDF",
                                data=st.session_state.pdf_bytes,
                                file_name="HighlightHunter_Revision_Notes.pdf",
                                mime="application/pdf",
                            )
        except Exception as e:
            st.error(f"Error processing image: {e}")
    else:
        st.info("Please upload an image file to begin.")

if __name__ == "__main__":
    main()
