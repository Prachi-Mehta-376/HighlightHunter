import os
from pathlib import Path
from PIL import Image
import detector, extractor
from pdf_generator import generate_revision_pdf

# Load sample image (synthetic already created)
img_path = Path(r'C:/Users/Prachi Mehta/Desktop/Projects/HighlightHunter/assets/sample_notes.png')
img = Image.open(img_path).convert('RGB')
# Detect highlights for all colors
detections, _, _ = detector.detect_highlights(
    img,
    thresholds={c: detector.DEFAULT_THRESHOLDS[c] for c in ['yellow','green','pink']},
    min_area=100,
)
# Extract snippets (use default padding 5)
snippets = extractor.extract_snippets(img, detections, padding=5)
# Generate PDF bytes
pdf_bytes = generate_revision_pdf(snippets, title='Demo Revision Sheet')
# Save to assets folder
out_path = Path(r'C:/Users/Prachi Mehta/Desktop/Projects/HighlightHunter/assets/demo_revision_notes.pdf')
out_path.write_bytes(pdf_bytes)
print('PDF saved', out_path, 'size', out_path.stat().st_size)
