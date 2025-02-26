import os
import sys
import tempfile
from pathlib import Path

import fitz
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_pdf_bytes():
    """Create a simple PDF file in memory for testing."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 size

    # Add some text to the page
    page.insert_text((100, 100), "Sample PDF Document", fontsize=16)
    page.insert_text((100, 150), "This is a test document for OCR testing.", fontsize=12)
    page.insert_text((100, 200), "It contains some text that can be extracted.", fontsize=12)

    # Save to bytes
    pdf_bytes = doc.tobytes()
    doc.close()

    return pdf_bytes


@pytest.fixture
def sample_pdf_path(sample_pdf_bytes):
    """Create a temporary PDF file on disk for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(sample_pdf_bytes)
        tmp_path = tmp.name

    yield Path(tmp_path)

    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


@pytest.fixture
def sample_image():
    """Create a sample image with text for OCR testing."""
    # Create a white image
    width, height = 800, 600
    image = Image.new("RGB", (width, height), color="white")

    # Convert to numpy array for easier handling
    img_array = np.array(image)

    return img_array


@pytest.fixture
def test_pdf_path():
    """Path to the real test PDF document."""
    pdf_path = Path(__file__).parent.parent / "docketanalyzer_ocr" / "setup" / "test.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Test PDF not found at {pdf_path}")
    return pdf_path
