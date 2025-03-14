import tempfile
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

from docketanalyzer_core import load_s3

BASE_DIR = Path(__file__).resolve().parent


def load_pdf(
    file: bytes | None = None,
    s3_key: str | None = None,
    filename: str | None = None,
) -> tuple[bytes, str]:
    """Loads a PDF file either from binary content or S3.

    This function handles loading a PDF file from either binary or from an S3 bucket.
    It returns the binary content of the PDF file and the filename.

    Args:
        file: PDF file content as bytes. Defaults to None.
        s3_key: S3 key if the PDF should be fetched from S3. Defaults to None.
        filename: Optional filename to use. If not provided, will be derived
            from s3_key or set to a default.

    Returns:
        tuple[bytes, str]: A tuple containing:
            - The binary content of the PDF file
            - The filename of the PDF

    Raises:
        ValueError: If neither file nor s3_key is provided.
    """
    if file is None and s3_key is None:
        raise ValueError("Either file or s3_key must be provided")

    if filename is None:
        filename = Path(s3_key).name if s3_key else "document.pdf"

    # If we already have the file content, just return it
    if file is not None:
        return file, filename

    # Otherwise, we need to download from S3
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_path = Path(temp_file.name)
        load_s3().download(s3_key, str(temp_path))
        return temp_path.read_bytes(), filename


def page_to_image(page: fitz.Page, dpi: int = 200) -> np.ndarray:
    """Converts a PDF page to a numpy image array.

    This function renders a PDF page at the specified DPI and converts it to a numpy
        array. If the resulting image would be too large, it falls back to a
        lower resolution.

    Args:
        page: The pymupdf Page object to convert.
        dpi: The dots per inch resolution to render at. Defaults to 200.

    Returns:
        np.ndarray: The page as a numpy array in RGB format.
    """
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pm = page.get_pixmap(matrix=mat, alpha=False)

    if pm.width > 4500 or pm.height > 4500:
        pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

    img = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
    img = np.array(img)

    return img


def extract_native_text(page: fitz.Page, dpi: int) -> list[dict]:
    """Extracts text content and bounding boxes from a PDF page using native PDF text.

    This function extracts text directly from the PDF's internal structure
        rather than using OCR.

    Args:
        page: The pymupdf Page object to extract text from.
        dpi: The resolution to use when scaling bounding boxes.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - 'bbox': The bounding box coordinates [x1, y1, x2, y2]
            - 'content': The text content of the line
    """
    blocks = page.get_text("dict")["blocks"]
    data = []
    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                content = "".join([span["text"] for span in line["spans"]])
                if content.strip():
                    line["bbox"] = tuple([(dpi / 72) * x for x in line["bbox"]])
                    data.append(
                        {
                            "bbox": line["bbox"],
                            "content": content,
                        }
                    )
    return data


def has_images(page: fitz.Page) -> bool:
    """Checks if a page has images that are large enough to potentially contain text.

    Args:
        page: The pymupdf Page object to check.

    Returns:
        bool: True if the page contains images of a significant size, False otherwise.
    """
    image_list = page.get_images(full=True)

    for _, img_info in enumerate(image_list):
        xref = img_info[0]
        base_image = page.parent.extract_image(xref)

        if base_image:
            width = base_image["width"]
            height = base_image["height"]
            if width > 10 and height > 10:
                return True

    return False


def has_text_annotations(page: fitz.Page) -> bool:
    """Checks if a page has annotations that could contain text.

    Args:
        page: The pymupdf Page object to check.

    Returns:
        bool: True if the page has text-containing annotations, False otherwise.
    """
    annots = page.annots()

    if annots:
        for annot in annots:
            annot_type = annot.type[1]
            if annot_type in [fitz.PDF_ANNOT_FREE_TEXT, fitz.PDF_ANNOT_WIDGET]:
                return True

    return False


def page_needs_ocr(page: fitz.Page) -> bool:
    """Determines if a page needs OCR processing.

    This function checks various conditions to decide if OCR is needed:
    - If the page has no text
    - If the page has CID-encoded text (often indicates non-extractable text)
    - If the page has text annotations
    - If the page has images that might contain text
    - If the page has many drawing paths (might be scanned text)

    Args:
        page: The pymupdf Page object to check.

    Returns:
        bool: True if the page needs OCR processing, False otherwise.
    """
    page_text = page.get_text()

    return (
        page_text.strip() == ""
        or "(cid:" in page_text
        or has_text_annotations(page)
        or has_images(page)
        or len(page.get_drawings()) > 10
    )
