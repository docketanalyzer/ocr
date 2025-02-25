from docketanalyzer_ocr import process_pdf


if __name__ == "__main__":
    doc = process_pdf("data/test.pdf")
    for page in doc.stream():
        break
