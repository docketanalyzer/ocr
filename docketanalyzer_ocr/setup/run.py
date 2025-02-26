"""Docket Analyzer OCR Module - Main Entry Point.

This script serves as the main entry point for the Docket Analyzer OCR module
when run directly. It demonstrates basic usage by processing a test PDF file
and printing the extracted text from each page.
"""
from pathlib import Path
from docketanalyzer_ocr import process_pdf

if __name__ == "__main__":
    # Process a test PDF file and print the extracted text
    path = Path(__file__).parent / "test.pdf"
    doc = process_pdf(path)
    for page in doc.stream():
        print(page.text)
