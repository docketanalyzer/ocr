import json
import tempfile
from pathlib import Path

import fitz
import numpy as np

from docketanalyzer_ocr.document import (
    PDFDocument,
    extract_native_text,
    page_to_image,
    pdf_document,
)


class TestDocumentCore:
    """Core tests for document processing functionality."""

    def test_document_processing(self, sample_pdf_bytes):
        """Test basic document processing without mocks."""
        # Create document and process
        doc = PDFDocument(sample_pdf_bytes, filename="test.pdf")
        doc.process()

        # Verify document structure
        assert len(doc.pages) > 0
        assert doc.filename == "test.pdf"

        # Check page content
        for page in doc.pages:
            assert page.text
            assert len(page.blocks) > 0

            # Check block structure
            for block in page.blocks:
                assert hasattr(block, "bbox")
                assert hasattr(block, "block_type")
                assert hasattr(block, "lines")

                # Check line structure
                for line in block.lines:
                    assert hasattr(line, "bbox")
                    assert hasattr(line, "content")

    def test_page_to_image_conversion(self, sample_pdf_bytes):
        """Test converting PDF pages to images."""
        doc = fitz.open("pdf", sample_pdf_bytes)
        page = doc[0]

        # Test with default DPI
        img = page_to_image(page)
        assert isinstance(img, np.ndarray)
        assert img.shape[2] == 3  # RGB image

        # Test with custom DPI
        img_high_dpi = page_to_image(page, dpi=300)
        assert img_high_dpi.shape[0] > img.shape[0]  # Higher resolution

        doc.close()

    def test_native_text_extraction(self, sample_pdf_bytes):
        """Test extracting native text from PDF pages."""
        doc = fitz.open("pdf", sample_pdf_bytes)
        page = doc[0]

        text_data = extract_native_text(page, 100)

        assert isinstance(text_data, list)
        assert len(text_data) > 0

        # Check structure of text items
        for item in text_data:
            assert "bbox" in item
            assert "content" in item
            assert isinstance(item["bbox"], tuple)
            assert len(item["bbox"]) == 4
            assert isinstance(item["content"], str)

        doc.close()

    def test_document_save_load(self, sample_pdf_bytes):
        """Test saving and loading document data."""
        # Process document
        doc = pdf_document(sample_pdf_bytes, filename="test.pdf").process()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            doc.save(tmp_path)

        # Load saved data
        with open(tmp_path, "r") as f:
            data = json.load(f)

        # Verify data structure
        assert data["filename"] == "test.pdf"
        assert "pages" in data
        assert len(data["pages"]) > 0

        # Clean up
        tmp_path.unlink()
