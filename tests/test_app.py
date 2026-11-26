from io import BytesIO

import pytest
from PIL import Image, UnidentifiedImageError

from app import load_uploaded_image


def test_load_uploaded_image_returns_rgb_copy() -> None:
    source = Image.new("RGBA", (2, 2), color=(100, 120, 140, 80))
    image_bytes = BytesIO()
    source.save(image_bytes, format="PNG")

    image = load_uploaded_image(image_bytes.getvalue())

    assert image.mode == "RGB"
    assert image.size == (2, 2)


def test_load_uploaded_image_rejects_invalid_data() -> None:
    with pytest.raises(UnidentifiedImageError):
        load_uploaded_image(b"not an image")
