import json
import uuid
from pathlib import Path
from typing import Generator, Iterator, Optional, Union

import fitz
import numpy as np
from PIL import Image
from tqdm import tqdm

from .remote import RunPodClient
from .utils import delete_from_s3, upload_to_s3


def page_to_image(page: fitz.Page, dpi: int = 200) -> np.ndarray:
    """Converts a PDF page to a numpy image array.

    This function renders a PDF page at the specified DPI and converts it to a numpy array.
    If the resulting image would be too large, it falls back to a lower resolution.

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


def extract_native_text(page: fitz.Page) -> list[dict]:
    """Extracts text content and bounding boxes from a PDF page using native PDF text.

    This function extracts text directly from the PDF's internal structure rather than using OCR.

    Args:
        page: The pymupdf Page object to extract text from.

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
                data.append(
                    {
                        "bbox": line["bbox"],
                        "content": "".join([span["text"] for span in line["spans"]]),
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
    # Get all images on the page
    image_list = page.get_images(full=True)

    # Check if there are any images that meet the size criteria
    for img_index, img_info in enumerate(image_list):
        xref = img_info[0]  # Get the xref of the image
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
    # Get all annotations on the page
    annots = page.annots()

    if annots:
        for annot in annots:
            # Check for FreeText or Widget annotations
            annot_type = annot.type[1]  # Get the annotation type
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

    if page_text.strip() == "":
        return True

    if "(cid:" in page_text:
        return True

    if has_text_annotations(page):
        return True

    if has_images(page):
        return True

    paths = page.get_drawings()
    if len(paths) > 10:
        return True

    return False


class DocumentComponent:
    """Base class for document components in the hierarchy.

    This is an abstract base class that defines the common interface and behavior
    for all document components (lines, blocks, pages, etc.) in the document hierarchy.

    Attributes:
        parent_attr: The attribute name that references the parent component.
        child_attr: The attribute name that references the child components.
        text_join: The string used to join text from child components.
    """

    parent_attr = None
    child_attr = None
    text_join = ""

    @property
    def parent(self):
        """Gets the parent component of this component.

        Returns:
            DocumentComponent: The parent component, or None if no parent exists.
        """
        if self.parent_attr is not None:
            return getattr(self, self.parent_attr, None)

    @property
    def children(self) -> list["DocumentComponent"]:
        """Gets the child components of this component.

        Returns:
            list[DocumentComponent]: A list of child components, or an empty list if no children exist.
        """
        if self.child_attr is not None:
            return getattr(self, self.child_attr, [])

    @property
    def page_num(self) -> int:
        """Gets the page number this component belongs to.

        Returns:
            int: The page number (0-indexed).
        """
        if isinstance(self, Page):
            return self.i
        return self.parent.page_num

    @property
    def doc(self) -> "PDFDocument":
        """Gets the document this component belongs to.

        Returns:
            PDFDocument: The parent document.
        """
        return self.parent.doc

    @property
    def text(self) -> str:
        """Gets the text content of this component.

        For Line components, returns the content directly.
        For other components, joins the text of all child components.

        Returns:
            str: The text content.
        """
        if isinstance(self, Line):
            return self.content
        return self.text_join.join([child.text for child in self.children])

    @property
    def id(self) -> str:
        """Gets a unique identifier for this component.

        Returns:
            str: A unique identifier string.
        """
        if isinstance(self, Page):
            return self.i
        return f"{self.parent.id}-{self.i}"

    def clip(
        self,
        bbox: Optional[tuple[float, float, float, float]] = None,
        save: Optional[str] = None,
    ):
        """Clips an image of this component from the parent page.

        Args:
            bbox: The bounding box to clip. If None, uses this component's bbox.
            save: Optional path to save the clipped image to.

        Returns:
            Image: The clipped image.
        """
        bbox = bbox or self.bbox
        return self.parent.clip(bbox, save)

    def __getitem__(self, idx: int) -> "DocumentComponent":
        """Gets a child component by index.

        Args:
            idx: The index of the child component.

        Returns:
            DocumentComponent: The child component at the specified index.
        """
        return self.children[idx]

    def __iter__(self) -> Iterator["DocumentComponent"]:
        """Iterates over child components.

        Yields:
            DocumentComponent: Each child component.
        """
        for child in self.children:
            yield child

    def __len__(self) -> int:
        """Gets the number of child components.

        Returns:
            int: The number of child components.
        """
        return len(self.children)


class Line(DocumentComponent):
    """Represents a line of text in a document.

    This is the lowest level component in the document hierarchy.

    Attributes:
        block: The parent block containing this line.
        i: The index of this line within its parent block.
        bbox: The bounding box coordinates [x1, y1, x2, y2].
        content: The text content of the line.
    """

    parent_attr = "block"

    def __init__(
        self,
        block: "Block",
        i: int,
        bbox: tuple[float, float, float, float],
        content: str,
    ):
        """Initializes a new Line component.

        Args:
            block: The parent block containing this line.
            i: The index of this line within its parent block.
            bbox: The bounding box coordinates [x1, y1, x2, y2].
            content: The text content of the line.
        """
        self.block = block
        self.i = i
        self.bbox = bbox
        self.content = content

    @property
    def data(self) -> dict:
        """Gets a dictionary representation of this line.

        Returns:
            dict: A dictionary containing the line's data.
        """
        return {
            "i": self.i,
            "bbox": self.bbox,
            "content": self.content,
        }


class Block(DocumentComponent):
    """Represents a block of text in a document.

    A block contains one or more lines of text.

    Attributes:
        page: The parent page containing this block.
        i: The index of this block within its parent page.
        bbox: The bounding box coordinates [x1, y1, x2, y2].
        block_type: The type of block (e.g., 'text', 'image', etc.).
        lines: The list of Line components in this block.
    """

    parent_attr = "page"
    child_attr = "lines"
    text_join = "\n"

    def __init__(
        self,
        page: "Page",
        i: int,
        bbox: tuple[float, float, float, float],
        block_type: str = "text",
        lines: list[dict] = [],
    ):
        """Initializes a new Block component.

        Args:
            page: The parent page containing this block.
            i: The index of this block within its parent page.
            bbox: The bounding box coordinates [x1, y1, x2, y2].
            block_type: The type of block. Defaults to 'text'.
            lines: A list of line data to initialize with.
        """
        self.page = page
        self.i = i
        self.bbox = bbox
        self.block_type = block_type
        self.lines = [Line(self, i, line["bbox"], line["content"]) for i, line in enumerate(lines)]

    @property
    def data(self) -> dict:
        """Gets a dictionary representation of this block.

        Returns:
            dict: A dictionary containing the block's data.
        """
        return {
            "i": self.i,
            "bbox": self.bbox,
            "type": self.block_type,
            "lines": [line.data for line in self.lines],
        }


class Page(DocumentComponent):
    """Represents a page in a document.

    A page contains one or more blocks of content.

    Attributes:
        _doc: The parent document containing this page.
        i: The index of this page within the document.
        blocks: The list of Block components on this page.
        img: The image representation of the page (set during processing).
        extracted_text: The extracted text data (set during processing).
        needs_ocr: Whether this page needs OCR processing.
    """

    parent_attr = "doc"
    child_attr = "blocks"
    text_join = "\n\n"

    def __init__(self, doc: "PDFDocument", i: int, blocks: list[dict] = []):
        """Initializes a new Page component.

        Args:
            doc: The parent document containing this page.
            i: The index of this page within the document.
            blocks: A list of block data to initialize with.
        """
        self._doc = doc
        self.i = i
        self.blocks = []
        self.img = None
        self.extracted_text = None
        self.needs_ocr = None
        self.set_blocks(blocks)

    def set_blocks(self, blocks: list[dict]) -> None:
        """Sets the blocks for this page.

        Args:
            blocks: A list of block data to set.
        """
        self.blocks = [
            Block(
                self,
                i,
                block.get("bbox"),
                block.get("type", "text"),
                block.get("lines", []),
            )
            for i, block in enumerate(blocks)
        ]

    def clip(
        self,
        bbox: Optional[tuple[float, float, float, float]] = None,
        save: Optional[str] = None,
    ) -> Image.Image:
        """Clips an image from this page.

        Args:
            bbox: The bounding box to clip. If None, uses the entire page.
            save: Optional path to save the clipped image to.

        Returns:
            Image.Image: The clipped image.
        """
        if self.img is None:
            return None

        x1, y1, x2, y2 = bbox or (0, 0, self.img.shape[1], self.img.shape[0])
        clip = self.img[int(y1) : int(y2), int(x1) : int(x2)]

        if save:
            Image.fromarray(clip).save(save)

        return clip

    @property
    def doc(self) -> "PDFDocument":
        """Gets the document this page belongs to.

        Returns:
            PDFDocument: The parent document.
        """
        return self._doc

    @property
    def data(self) -> dict:
        """Gets a dictionary representation of this page.

        Returns:
            dict: A dictionary containing the page's data.
        """
        return {
            "i": self.i,
            "blocks": [block.data for block in self.blocks],
        }


class PDFDocument:
    """Represents a PDF document.

    This class handles loading, processing, and extracting text from PDF documents.
    It manages the document hierarchy (pages, blocks, lines) and handles OCR when needed.

    Attributes:
        doc: The underlying PyMuPDF document.
        filename: The name of the PDF file.
        dpi: The resolution to use when rendering pages for OCR.
        pages: The list of Page components in the document.
        remote: Whether to use remote processing via RunPod.
    """

    def __init__(
        self,
        file_or_path: Union[bytes, str, Path],
        filename: Optional[str] = None,
        dpi: int = 200,
        remote: bool = False,
    ):
        """Initializes a new PDFDocument.

        Args:
            file_or_path: The PDF file content as bytes, or a path to the PDF file.
            filename: Optional name for the PDF file.
            dpi: The resolution to use when rendering pages for OCR. Defaults to 200.
            remote: Whether to use remote processing via RunPod. Defaults to False.
        """
        if isinstance(file_or_path, bytes):
            self.doc = fitz.open("pdf", file_or_path)
            self.pdf_bytes = file_or_path
            self.pdf_path = None
        else:
            self.doc = fitz.open(file_or_path)
            self.pdf_bytes = None
            self.pdf_path = file_or_path
        self.filename = filename or getattr(file_or_path, "name", "document.pdf")
        self.dpi = dpi
        self.remote = remote
        self.pages = [Page(self, i) for i in range(len(self.doc))]
        self._runpod_client = None
        self._s3_key = None

    @property
    def runpod_client(self) -> RunPodClient:
        """Gets or creates the RunPod client.

        Returns:
            RunPodClient: The RunPod client instance.
        """
        if self._runpod_client is None:
            self._runpod_client = RunPodClient()
        return self._runpod_client

    def _upload_to_s3(self) -> str:
        """Uploads the PDF to S3 with a random filename under the 'ocr' folder.

        Returns:
            str: The S3 key where the file was uploaded.
        """
        # Generate a random filename to avoid collisions
        random_id = str(uuid.uuid4())
        s3_key = f"ocr/{random_id}_{self.filename}"

        # If we have the PDF as bytes, write it to a temporary file first
        if self.pdf_bytes is not None:
            with Path(f"/tmp/{random_id}.pdf").open("wb") as f:
                f.write(self.pdf_bytes)
                temp_path = f.name
        else:
            temp_path = self.pdf_path

        # Upload to S3
        success = upload_to_s3(temp_path, s3_key, overwrite=True)
        if not success:
            raise ValueError(f"Failed to upload PDF to S3 at key: {s3_key}")

        # Clean up temporary file if we created one
        if self.pdf_bytes is not None:
            Path(temp_path).unlink(missing_ok=True)

        self._s3_key = s3_key
        return s3_key

    def stream(self, batch_size: int = 1) -> Generator[Page, None, None]:
        """Processes the document page by page and yields each processed page.

        This is a generator that processes pages in batches and yields each page
        after it has been processed. If remote=True, uses the RunPod client for processing.

        Args:
            batch_size: Number of pages to process in each batch. Defaults to 1.

        Yields:
            Page: Each processed page.
        """
        if self.remote:
            # Use remote processing via RunPod
            try:
                # Upload the PDF to S3
                s3_key = self._upload_to_s3()

                # Call the RunPod endpoint with the S3 key
                for result in self.runpod_client(s3_key=s3_key, filename=self.filename, batch_size=batch_size):
                    # Handle the nested response format from RunPod
                    if "stream" in result:
                        # Process each item in the stream array
                        for stream_item in result["stream"]:
                            if "output" in stream_item and "page" in stream_item["output"]:
                                page_data = stream_item["output"]["page"]
                                page_idx = page_data.get("i", 0)

                                # Update the page with the received data
                                if page_idx < len(self.pages):
                                    self.pages[page_idx].set_blocks(page_data.get("blocks", []))
                                    yield self.pages[page_idx]

                    # Check for completion status
                    status = result.get("status")
                    if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                        break

                    # Also check for status in the stream items
                    if "stream" in result:
                        for stream_item in result["stream"]:
                            if stream_item.get("output", {}).get("status") in ["COMPLETED", "FAILED", "CANCELLED"]:
                                break
            finally:
                # Clean up the S3 file after processing
                if self._s3_key:
                    delete_from_s3(self._s3_key)
                    self._s3_key = None
        else:
            # Use local processing
            for i in tqdm(range(0, len(self.doc), batch_size), desc="Processing PDF"):
                batch_pages = []
                batch_imgs = []

                # Prepare batch
                for j in range(i, min(i + batch_size, len(self.doc))):
                    page = self.doc[j]
                    self.pages[j].img = page_to_image(page, self.dpi)
                    batch_pages.append(page)
                    batch_imgs.append(self.pages[j].img)

                    # Extract native text
                    self.pages[j].extracted_text = extract_native_text(page)

                    # Check if OCR is needed
                    self.pages[j].needs_ocr = page_needs_ocr(page)
                    if not self.pages[j].needs_ocr:
                        self.pages[j].set_blocks(
                            [
                                {
                                    "bbox": line["bbox"],
                                    "type": "text",
                                    "lines": [line],
                                }
                                for line in self.pages[j].extracted_text
                            ]
                        )

                # Process OCR for pages that need it
                for j in range(len(batch_pages)):
                    page_idx = i + j
                    if self.pages[page_idx].needs_ocr:
                        from .ocr import extract_ocr_text

                        self.pages[page_idx].extracted_text = extract_ocr_text(self.pages[page_idx].img)
                        self.pages[page_idx].set_blocks(
                            [
                                {
                                    "bbox": line["bbox"],
                                    "type": "text",
                                    "lines": [line],
                                }
                                for line in self.pages[page_idx].extracted_text
                            ]
                        )

                    yield self.pages[page_idx]

    def process(self, batch_size: int = 1) -> "PDFDocument":
        """Processes the entire document at once.

        This method processes all pages in the document and returns the document itself.
        If remote=True, uses the RunPod client for processing.

        Args:
            batch_size: Number of pages to process in each batch. Defaults to 1.

        Returns:
            PDFDocument: The processed document (self).
        """
        for _ in self.stream(batch_size=batch_size):
            pass
        return self

    @property
    def data(self) -> dict:
        """Gets a dictionary representation of this document.

        Returns:
            dict: A dictionary containing the document's data.
        """
        return {
            "filename": self.filename,
            "pages": [page.data for page in self.pages],
        }

    def save(self, path: Union[str, Path]) -> None:
        """Saves the document data to a JSON file.

        Args:
            path: The path to save the JSON file to.
        """
        with open(path, "w") as f:
            json.dump(self.data, f)

    def load(self, path_or_data: Union[str, Path, dict]) -> "PDFDocument":
        """Loads document data from a JSON file or dictionary.

        Args:
            path_or_data: Either a path to a JSON file or a dictionary of document data.

        Returns:
            PDFDocument: The loaded document (self).
        """
        if isinstance(path_or_data, (str, Path)):
            with open(path_or_data, "r") as f:
                data = json.load(f)
        else:
            data = path_or_data

        self.filename = data.get("filename", self.filename)
        for i, page_data in enumerate(data.get("pages", [])):
            if i < len(self.pages):
                self.pages[i].set_blocks(page_data.get("blocks", []))

        return self

    def __getitem__(self, idx: int) -> Page:
        """Gets a page by index.

        Args:
            idx: The index of the page.

        Returns:
            Page: The page at the specified index.
        """
        return self.pages[idx]

    def __iter__(self) -> Iterator[Page]:
        """Iterates over pages in the document.

        Yields:
            Page: Each page in the document.
        """
        for page in self.pages:
            yield page

    def __len__(self) -> int:
        """Gets the number of pages in the document.

        Returns:
            int: The number of pages.
        """
        return len(self.pages)


def process_pdf(
    file_or_path: Union[bytes, str, Path],
    filename: Optional[str] = None,
    dpi: int = 200,
    load: Optional[Union[str, Path, dict]] = None,
    remote: bool = False,
) -> PDFDocument:
    """Processes a PDF file for text extraction.

    This is the main entry point for processing PDF documents. It creates a PDFDocument
    instance, optionally loads existing data, and processes the document if requested.

    Args:
        file_or_path: The PDF file content as bytes, or a path to the PDF file.
        filename: Optional name for the PDF file.
        dpi: The resolution to use when rendering pages for OCR. Defaults to 200.
        load: Optional path to a JSON file or dictionary with existing document data to load.
        remote: Whether to use remote processing via RunPod. Defaults to False.

    Returns:
        PDFDocument: The created (and possibly processed) document.
    """
    doc = PDFDocument(file_or_path, filename=filename, dpi=dpi, remote=remote)

    if load is not None:
        doc.load(load)

    return doc
