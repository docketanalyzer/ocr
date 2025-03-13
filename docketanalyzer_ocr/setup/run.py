from pathlib import Path

from docketanalyzer_ocr import pdf_document

if __name__ == "__main__":
    # Process a test PDF file to get the additional dependecies downloaded
    path = Path(__file__).parent / "test.pdf"
    doc = pdf_document(path)
    for page in doc.stream():
        print(page.text)
