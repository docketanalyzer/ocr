"""Docket Analyzer OCR Module.

This module provides functionality for extracting text from PDF documents,
particularly legal dockets, using OCR when necessary. It handles both
native PDF text extraction and OCR-based extraction for scanned documents.

Main components:
- Document processing and text extraction
- OCR using PaddleOCR
- Layout analysis
- Utility functions for file handling

Main entry points:
- pdf_document: Process a PDF file and extract text
- load_pdf: Load a PDF file from various sources
"""

from .document import PDFDocument, pdf_document
from .utils import download_from_s3, load_pdf, upload_to_s3

__all__ = [
    "PDFDocument",
    "pdf_document",
    "upload_to_s3",
    "download_from_s3",
    "load_pdf",
]
