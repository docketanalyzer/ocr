from pathlib import Path

from docketanalyzer_ocr import pdf_document

if __name__ == "__main__":
    # Process a test PDF file to setup additional dependencies
    path = Path(__file__).parent / "tests" / "fixtures" / "document.pdf"
    doc = pdf_document(path, use_s3=False)

    for page in doc.stream():
        print(page.text)
