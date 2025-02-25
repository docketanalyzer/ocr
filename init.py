from docketanalyzer_ocr import process_pdf, load_pdf


if __name__ == "__main__":
    file, filename = load_pdf(s3_key="ocr/test.pdf")
    doc = process_pdf(file, filename=filename)
    for page in doc.stream():
        print(page.text)
