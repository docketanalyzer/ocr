# DocketAnalyzer OCR Tests

This directory contains unit tests for the DocketAnalyzer OCR package.

## Test Structure

The tests are organized by module:

- `test_document.py`: Tests for document processing functionality
- `test_ocr.py`: Tests for OCR functionality
- `test_layout.py`: Tests for layout analysis functionality
- `test_utils.py`: Tests for utility functions
- `test_extract_tess.py`: Tests for Tesseract OCR extraction
- `test_integration.py`: Integration tests for the full OCR pipeline

## Running Tests

To run all tests:

```bash
cd /path/to/project/ocr
pytest tests/
```

To run a specific test file:

```bash
pytest tests/test_document.py
```

To run a specific test:

```bash
pytest tests/test_document.py::TestDocumentHelpers::test_page_to_image
```

## Test Coverage

To run tests with coverage:

```bash
pytest --cov=docketanalyzer_ocr tests/
```

## Required Dependencies

The tests require the following dependencies:

- pytest
- pytest-cov (for coverage reports)
- unittest.mock
- numpy
- PIL
- PyMuPDF (fitz)
- pandas

## Test Fixtures

The tests use fixtures defined in `conftest.py`:

- `sample_pdf_bytes`: A simple PDF file in memory
- `sample_pdf_path`: A temporary PDF file on disk
- `sample_image`: A sample image for OCR testing
- `mock_ocr_result`: Mock OCR result data
- `mock_layout_result`: Mock layout analysis result data
- `mock_s3_file`: Mock S3 file data

## Adding New Tests

When adding new functionality to the package, please add corresponding tests following the existing patterns. 