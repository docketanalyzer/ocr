# Docket Analyzer OCR Tests

This directory contains unit tests for the Docket Analyzer OCR package.

## Test Structure

The tests are organized by module:

- `test_document.py`: Tests for document processing functionality
- `test_ocr.py`: Tests for OCR functionality
- `test_layout.py`: Tests for layout analysis functionality
- `test_utils.py`: Tests for utility functions
- `test_remote.py`: Tests for remote processing functionality
- `test_service.py`: Tests for the OCR service API endpoints

## Running Tests and Code Coverage

```bash
pytest --cov=docketanalyzer_ocr tests/ --cov-report=xml --cov-branch --junitxml=junit.xml -o junit_family=legacy
```

## Code Quality

```bash
ruff format . && ruff check --fix .
```

## Test Fixtures

The tests use fixtures defined in `conftest.py`:

- `sample_pdf_bytes`: A simple PDF file in memory
- `sample_pdf_path`: A temporary PDF file on disk
- `sample_image`: A sample image for OCR testing
- `test_pdf_path`: Path to the real test PDF document
