# HighlightHunter

## Overview
HighlightHunter is a beginner‑friendly computer‑vision web app that lets students upload photos of their study notes, detect highlighted sections with OpenCV, crop and run OCR on the highlighted regions with `pytesseract`, edit the extracted text in the UI, and export a formatted revision PDF using `reportlab`.

## Features
- Streamlit UI with purple‑lavender theme
- Image upload (JPG, JPEG, PNG) with RGB preview
- OpenCV HSV highlight detection (`yellow`, `green`, `pink`) with configurable minimum area, padding, annotated preview, and per-colour masks
- Snippet cropping sorted in natural reading order with PNG download per snippet
- OCR text extraction using `pytesseract` with editable text fields in the UI
- ReportLab revision PDF generation (`generate_revision_pdf` and `generate_pdf`) featuring a title page, snippet images, extracted OCR text, page numbers, and automatic page breaks
- Comprehensive `pytest` test suite

## Project Structure
```
HighlightHunter/
├─ app.py                 # Streamlit entry point & UI orchestration
├─ detector.py            # OpenCV highlight detection & annotation
├─ extractor.py           # Snippet cropping & pytesseract OCR extraction
├─ pdf_generator.py       # ReportLab PDF generation
├─ requirements.txt       # Python dependencies
├─ README.md              # This file
├─ .gitignore             # Python & Streamlit ignores
├─ assets/                # Sample and demo images
└─ tests/                 # pytest test suite
```

## System Prerequisite: Tesseract OCR
`pytesseract` requires the **Tesseract OCR** system binary to be installed and available on your `PATH`:

- **macOS (Homebrew)**:
  ```bash
  brew install tesseract
  ```
- **Ubuntu / Debian**:
  ```bash
  sudo apt-get update && sudo apt-get install -y tesseract-ocr
  ```
- **Windows**:
  Download and run the Tesseract installer from the [UB Mannheim Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki), then add the installation folder (e.g., `C:\Program Files\Tesseract-OCR`) to your system `PATH`.

Verify installation:
```bash
tesseract --version
```

## Setup Instructions
1. **Clone the repo** (or download the folder).
2. Open a terminal in the `HighlightHunter` directory.
3. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate     # macOS / Linux
   # .\.venv\Scripts\activate    # Windows
   ```
4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```
   The app will be reachable at `http://localhost:8501`.
6. Run the test suite:
   ```bash
   pytest
   ```
