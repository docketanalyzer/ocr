"""Docket Analyzer OCR Module - Main Entry Point.

This script serves as the main entry point for the Docket Analyzer OCR module
when run directly. It demonstrates basic usage by processing a test PDF file
and printing the extracted text from each page.

Example:
    $ python init.py
"""

from docketanalyzer_ocr import process_pdf

if __name__ == "__main__":
    # Process a test PDF file and print the extracted text
    doc = process_pdf("data/test.pdf")
    for page in doc.stream():
        print(page.text)
