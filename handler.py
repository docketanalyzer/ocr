from collections.abc import Generator
from datetime import datetime

import runpod

from docketanalyzer_ocr import load_pdf, pdf_document


def handler(event: dict) -> Generator[dict, None, None]:
    """RunPod serverless handler for OCR processing.

    This function processes PDF documents for OCR text extraction in a serverless
    environment. It can handle PDFs provided either as binary data or via S3 keys.

    Args:
        event: The event dictionary containing:
            - input: Dictionary with processing parameters:
                - s3_key: Optional S3 key to load the PDF from
                - file: Optional binary PDF data
                - filename: Optional filename for the PDF
                - batch_size: Optional batch size for processing (default: 1)

    Yields:
        dict: Per page results:
            - page: Processed page data
            - seconds_elapsed: Processing time so far
            - progress: Processing progress (0-1)
    """
    start = datetime.now()
    inputs = event.pop("input")
    filename = inputs.get("filename")
    batch_size = inputs.get("batch_size", 1)

    if inputs.get("s3_key"):
        pdf_data, filename = load_pdf(
            s3_key=inputs.pop("s3_key"), filename=filename
        )
    elif inputs.get("file"):
        pdf_data, filename = load_pdf(
            file=inputs.pop("file"),
            filename=filename,
        )
    else:
        raise ValueError("Neither 's3_key' nor 'file' provided in input")

    doc = pdf_document(pdf_data, filename=filename)
    for i, page in enumerate(doc.stream(batch_size=batch_size)):
        duration = (datetime.now() - start).total_seconds()
        yield {
            "page": page.data,
            "seconds_elapsed": duration,
            "progress": i / len(doc),
        }
    doc.close()


runpod.serverless.start(
    {
        "handler": handler,
        "return_aggregate_stream": False,
    }
)
