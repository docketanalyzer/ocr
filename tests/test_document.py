from unittest.mock import MagicMock, patch

import fitz
import numpy as np

from docketanalyzer_ocr.document import (
    Block,
    Line,
    Page,
    PDFDocument,
    extract_native_text,
    has_images,
    has_text_annotations,
    page_needs_ocr,
    page_to_image,
    process_pdf,
)


class TestDocumentHelpers:
    """Tests for helper functions in document.py."""

    def test_page_to_image(self, sample_pdf_bytes):
        """Test converting a PDF page to an image."""
        doc = fitz.open("pdf", sample_pdf_bytes)
        page = doc[0]

        # Test with default DPI
        img = page_to_image(page)
        assert isinstance(img, np.ndarray)
        assert img.shape[2] == 3  # RGB image

        # Test with custom DPI
        img_high_dpi = page_to_image(page, dpi=300)
        assert isinstance(img_high_dpi, np.ndarray)
        assert img_high_dpi.shape[0] >= img.shape[0]  # Higher resolution

        doc.close()

    def test_extract_native_text(self, sample_pdf_bytes):
        """Test extracting native text from a PDF page."""
        doc = fitz.open("pdf", sample_pdf_bytes)
        page = doc[0]

        text_data = extract_native_text(page)

        assert isinstance(text_data, list)
        # Our sample PDF should have text
        assert len(text_data) > 0
        # Check structure of first text item
        if len(text_data) > 0:
            assert "bbox" in text_data[0]
            assert "content" in text_data[0]
            assert isinstance(text_data[0]["bbox"], tuple)
            assert len(text_data[0]["bbox"]) == 4

        doc.close()

    def test_has_images(self, sample_pdf_bytes):
        """Test checking if a page has images."""
        doc = fitz.open("pdf", sample_pdf_bytes)
        page = doc[0]

        # Our sample PDF doesn't have images
        assert has_images(page) is False

        doc.close()

    def test_has_text_annotations(self, sample_pdf_bytes):
        """Test checking if a page has text annotations."""
        doc = fitz.open("pdf", sample_pdf_bytes)
        page = doc[0]

        # Our sample PDF doesn't have annotations
        assert has_text_annotations(page) is False

        doc.close()

    def test_page_needs_ocr(self, sample_pdf_bytes):
        """Test determining if a page needs OCR."""
        doc = fitz.open("pdf", sample_pdf_bytes)
        page = doc[0]

        # Our sample PDF has text, so it shouldn't need OCR
        assert page_needs_ocr(page) is False

        doc.close()


class TestDocumentComponents:
    """Tests for document component classes (Line, Block, Page)."""

    def test_line_creation(self):
        """Test creating a Line component."""
        mock_block = MagicMock()
        bbox = (10, 20, 100, 30)
        content = "Test line content"

        line = Line(mock_block, 0, bbox, content)

        assert line.block == mock_block
        assert line.i == 0
        assert line.bbox == bbox
        assert line.content == content
        assert line.text == content

    def test_block_creation(self):
        """Test creating a Block component."""
        mock_page = MagicMock()
        bbox = (10, 20, 200, 100)
        lines_data = [
            {"bbox": (10, 20, 100, 30), "content": "Line 1"},
            {"bbox": (10, 40, 150, 50), "content": "Line 2"},
        ]

        block = Block(mock_page, 0, bbox, "text", lines_data)

        assert block.page == mock_page
        assert block.i == 0
        assert block.bbox == bbox
        assert block.block_type == "text"
        assert len(block.lines) == 2
        assert isinstance(block.lines[0], Line)
        assert block.lines[0].content == "Line 1"
        assert block.text == "Line 1\nLine 2"

    def test_page_creation(self):
        """Test creating a Page component."""
        mock_doc = MagicMock()
        blocks_data = [
            {
                "bbox": (10, 20, 200, 100),
                "type": "text",
                "lines": [
                    {"bbox": (10, 20, 100, 30), "content": "Block 1 Line 1"},
                    {"bbox": (10, 40, 150, 50), "content": "Block 1 Line 2"},
                ],
            },
            {
                "bbox": (10, 120, 200, 200),
                "type": "text",
                "lines": [{"bbox": (10, 120, 100, 130), "content": "Block 2 Line 1"}],
            },
        ]

        page = Page(mock_doc, 0, blocks_data)

        assert page.doc == mock_doc
        assert page.i == 0
        assert len(page.blocks) == 2
        assert isinstance(page.blocks[0], Block)
        assert len(page.blocks[0].lines) == 2
        assert page.blocks[0].lines[0].content == "Block 1 Line 1"
        assert page.text == "Block 1 Line 1\nBlock 1 Line 2\n\nBlock 2 Line 1"


class TestPDFDocument:
    """Tests for the PDFDocument class."""

    @patch("docketanalyzer_ocr.document.extract_native_text")
    def test_document_initialization(self, mock_extract_native_text, sample_pdf_bytes):
        """Test initializing a PDFDocument."""
        mock_extract_native_text.return_value = [{"bbox": [10, 20, 100, 30], "content": "Test content"}]

        doc = PDFDocument(sample_pdf_bytes, filename="test.pdf")

        assert doc.filename == "test.pdf"
        assert doc.dpi == 200  # Default DPI
        assert len(doc.doc) > 0  # Should have at least one page

    @patch("docketanalyzer_ocr.document.extract_native_text")
    def test_document_processing(self, mock_extract_native_text, sample_pdf_bytes):
        """Test processing a PDFDocument."""
        mock_extract_native_text.return_value = [{"bbox": [10, 20, 100, 30], "content": "Test content"}]

        doc = PDFDocument(sample_pdf_bytes, filename="test.pdf")
        processed_doc = doc.process()

        assert processed_doc is doc  # Should return self
        assert len(doc) > 0  # Should have at least one page
        assert isinstance(doc[0], Page)

    @patch("docketanalyzer_ocr.document.extract_native_text")
    def test_document_data_export(self, mock_extract_native_text, sample_pdf_bytes, tmp_path):
        """Test exporting document data to JSON."""
        mock_extract_native_text.return_value = [{"bbox": [10, 20, 100, 30], "content": "Test content"}]

        doc = PDFDocument(sample_pdf_bytes, filename="test.pdf")
        doc.process()

        data = doc.data
        assert isinstance(data, dict)
        assert "filename" in data
        assert "pages" in data
        assert len(data["pages"]) > 0

        # Test saving to file
        save_path = tmp_path / "test_output.json"
        doc.save(save_path)
        assert save_path.exists()

    @patch("docketanalyzer_ocr.document.extract_native_text")
    def test_process_pdf_function(self, mock_extract_native_text, sample_pdf_bytes):
        """Test the process_pdf function."""
        mock_extract_native_text.return_value = [{"bbox": [10, 20, 100, 30], "content": "Test content"}]

        doc = process_pdf(sample_pdf_bytes, filename="test.pdf").process()
        raise Exception("test")

        assert isinstance(doc, PDFDocument)
        assert doc.filename == "test.pdf"
        assert len(doc) > 0
