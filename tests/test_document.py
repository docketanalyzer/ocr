import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz
import numpy as np
import pytest
from PIL import Image

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

        assert isinstance(doc, PDFDocument)
        assert doc.filename == "test.pdf"
        assert len(doc) > 0

    @patch("docketanalyzer_ocr.document.upload_to_s3")
    @patch("docketanalyzer_ocr.document.uuid.uuid4")
    def test_upload_to_s3_with_bytes(self, mock_uuid4, mock_upload_to_s3, sample_pdf_bytes):
        """Test uploading PDF bytes to S3."""
        # Setup
        mock_uuid4.return_value = "test-uuid"
        mock_upload_to_s3.return_value = True

        # Create document with bytes
        doc = PDFDocument(sample_pdf_bytes, filename="test.pdf")

        # Call the method
        s3_key = doc._upload_to_s3()

        # Assertions
        assert s3_key == "ocr/test-uuid_test.pdf"
        assert doc._s3_key == s3_key
        mock_upload_to_s3.assert_called_once()
        # Verify the first argument is a path to a temp file
        assert mock_upload_to_s3.call_args[0][0].startswith("/tmp/")
        assert mock_upload_to_s3.call_args[0][1] == s3_key
        # Check if overwrite=True is passed as a keyword argument
        assert mock_upload_to_s3.call_args[1].get("overwrite") is True

    @patch("docketanalyzer_ocr.document.upload_to_s3")
    @patch("docketanalyzer_ocr.document.uuid.uuid4")
    def test_upload_to_s3_with_path(self, mock_uuid4, mock_upload_to_s3, sample_pdf_path):
        """Test uploading PDF from path to S3."""
        # Setup
        mock_uuid4.return_value = "test-uuid"
        mock_upload_to_s3.return_value = True

        # Create document with path
        doc = PDFDocument(sample_pdf_path, filename="test.pdf")

        # Call the method
        s3_key = doc._upload_to_s3()

        # Assertions
        assert s3_key == "ocr/test-uuid_test.pdf"
        assert doc._s3_key == s3_key
        mock_upload_to_s3.assert_called_once_with(sample_pdf_path, s3_key, overwrite=True)

    @patch("docketanalyzer_ocr.document.upload_to_s3")
    def test_upload_to_s3_failure(self, mock_upload_to_s3, sample_pdf_bytes):
        """Test handling of S3 upload failure."""
        # Setup
        mock_upload_to_s3.return_value = False

        # Create document
        doc = PDFDocument(sample_pdf_bytes, filename="test.pdf")

        # Call the method - should raise ValueError
        with pytest.raises(ValueError, match="Failed to upload PDF to S3"):
            doc._upload_to_s3()

    @patch("docketanalyzer_ocr.document.page_to_image")
    def test_page_clip(self, mock_page_to_image, sample_pdf_bytes):
        """Test clipping an image from a page."""
        # Setup
        import numpy as np

        # Create a test image
        test_img = np.zeros((100, 200, 3), dtype=np.uint8)
        test_img[20:40, 30:70] = 255  # White rectangle in the middle

        # Mock the page_to_image function
        mock_page_to_image.return_value = test_img

        # Create document and process it
        doc = PDFDocument(sample_pdf_bytes, filename="test.pdf")
        page = doc[0]
        page.img = test_img  # Set the image directly

        # Test clipping with specific bbox
        bbox = (30, 20, 70, 40)
        clip = page.clip(bbox)

        # Assertions
        assert clip.shape == (20, 40, 3)  # Height, width, channels
        assert np.all(clip == 255)  # Should be all white

        # Test clipping with save
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "clip.png"
            page.clip(bbox, save=str(save_path))
            assert save_path.exists()
            saved_img = np.array(Image.open(save_path))
            assert saved_img.shape == (20, 40, 3)

    @patch("docketanalyzer_ocr.document.extract_native_text")
    def test_save_and_load(self, mock_extract_native_text, sample_pdf_bytes, tmp_path):
        """Test saving and loading document data."""
        # Setup
        mock_extract_native_text.return_value = [
            {"bbox": [10, 20, 100, 30], "content": "Line 1"},
            {"bbox": [10, 40, 100, 50], "content": "Line 2"},
        ]

        # Create and process a document
        doc1 = PDFDocument(sample_pdf_bytes, filename="test.pdf")
        doc1.process()

        # Save to JSON
        json_path = tmp_path / "test_doc.json"
        doc1.save(json_path)

        # Create a new document and load the saved data
        doc2 = PDFDocument(sample_pdf_bytes, filename="different.pdf")
        doc2.load(json_path)

        # Assertions
        assert doc2.filename == "test.pdf"  # Should be updated from the loaded data
        assert len(doc2.pages) == len(doc1.pages)
        assert len(doc2.pages[0].blocks) > 0
        assert len(doc2.pages[0].blocks[0].lines) > 0
        assert doc2.pages[0].blocks[0].lines[0].content == "Line 1"
        # Check for Line 2 only if it exists
        if len(doc2.pages[0].blocks[0].lines) > 1:
            assert doc2.pages[0].blocks[0].lines[1].content == "Line 2"

        # Test loading from dictionary
        doc3 = PDFDocument(sample_pdf_bytes, filename="another.pdf")
        doc3.load(doc1.data)

        # Assertions
        assert doc3.filename == "test.pdf"
        assert len(doc3.pages) == len(doc1.pages)
        assert len(doc3.pages[0].blocks) > 0
        assert len(doc3.pages[0].blocks[0].lines) > 0
        assert doc3.pages[0].blocks[0].lines[0].content == "Line 1"

    @patch("docketanalyzer_ocr.document.extract_native_text")
    def test_process_pdf_with_load(self, mock_extract_native_text, sample_pdf_bytes, tmp_path):
        """Test the process_pdf function with loading existing data."""
        # Setup
        mock_extract_native_text.return_value = [
            {"bbox": [10, 20, 100, 30], "content": "Line 1"},
            {"bbox": [10, 40, 100, 50], "content": "Line 2"},
        ]

        # Create and process a document
        doc1 = PDFDocument(sample_pdf_bytes, filename="test.pdf")
        doc1.process()

        # Save to JSON
        json_path = tmp_path / "test_doc.json"
        doc1.save(json_path)

        # Use process_pdf with load parameter
        doc2 = process_pdf(sample_pdf_bytes, filename="different.pdf", load=json_path)

        # Assertions
        assert doc2.filename == "test.pdf"  # Should be updated from the loaded data
        assert len(doc2.pages) == len(doc1.pages)
        assert len(doc2.pages[0].blocks) > 0
        assert len(doc2.pages[0].blocks[0].lines) > 0
        assert doc2.pages[0].blocks[0].lines[0].content == "Line 1"
        # Check for Line 2 only if it exists
        if len(doc2.pages[0].blocks[0].lines) > 1:
            assert doc2.pages[0].blocks[0].lines[1].content == "Line 2"
