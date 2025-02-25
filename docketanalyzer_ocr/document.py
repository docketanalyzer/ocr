import json
from typing import Optional
import fitz
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def page_to_image(page, dpi=200) -> dict:
    """Convert fitz.Document to image, Then convert the image to numpy array.

    Args:
        page (_type_): pymudoc page
        dpi (int, optional): reset the dpi of dpi. Defaults to 200.

    Returns:
        dict:  {'img': numpy array, 'width': width, 'height': height }
    """
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pm = page.get_pixmap(matrix=mat, alpha=False)

    if pm.width > 4500 or pm.height > 4500:
        pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

    img = Image.frombytes('RGB', (pm.width, pm.height), pm.samples)
    img = np.array(img)

    return img


def extract_native_text(page) -> dict:
    blocks = page.get_text("dict")["blocks"]
    data = []
    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                data.append({
                    "bbox": line["bbox"],
                    "content": ''.join([span["text"] for span in line["spans"]]),
                })
    return data


def has_images(page: fitz.Page) -> bool:
    """Does the page have images that are large enough to contain text

    :param page: fitz Page object
    :return: True if page contains images of a certain size
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
    """Does the page have annotations which could contain text

    :param page: fitz Page object
    :return: if page has annotations that could contain text
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
    """Does the page need OCR

    :param page: fitz Page object
    :return: does page need OCR
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
    parent_attr = None
    child_attr = None
    text_join = ""

    @property
    def parent(self):
        if self.parent_attr is not None:
            return getattr(self, self.parent_attr, None)
    
    @property
    def children(self):
        if self.child_attr is not None:
            return getattr(self, self.child_attr, [])

    @property
    def page_num(self):
        if isinstance(self, Page):
            return self.i
        return self.parent.page_num
    
    @property
    def doc(self):
        if isinstance(self, Page):
            return self._doc
        return self.parent.doc
    
    @property
    def text(self):
        if isinstance(self, Line):
            return self.content
        return self.text_join.join([child.text for child in self.children])

    @property
    def id(self):
        if isinstance(self, Page):
            return self.i
        return f"{self.parent.id}-{self.i}"
    
    def clip(self, bbox=None, save=None):
        bbox = bbox or self.bbox
        return self.parent.clip(bbox, save)
    
    def __getitem__(self, idx):
        return self.children[idx]
    
    def __iter__(self):
        for child in self.children:
            yield child

    def __len__(self):
        return len(self.children)


class Line(DocumentComponent):
    parent_attr = "block"

    def __init__(self, block, i, bbox, content):
        self.block = block
        self.i = i
        self.bbox = bbox
        self.content = content

    @property
    def data(self):
        return {
            "i": self.i,
            "bbox": self.bbox,
            "content": self.content,
        }


class Block(DocumentComponent):
    parent_attr = "page"
    child_attr = "lines"
    text_join = "\n"

    def __init__(self, page, i, bbox, block_type='text', lines=[]):
        self.page = page
        self.i = i
        self.bbox = bbox
        self.block_type = block_type
        self.lines = [
            Line(self, i, line["bbox"], line['content']) 
            for i, line in enumerate(lines)
        ]

    @property
    def data(self):
        return {
            "i": self.i,
            "type": self.block_type,
            "bbox": self.bbox,
            "lines": [line.data for line in self.lines],
        }


class Page(DocumentComponent):
    parent_attr = "doc"
    child_attr = "blocks"
    text_join = "\n\n"

    def __init__(self, doc, i, blocks=[]):
        self._doc = doc
        self.i = i
        self.blocks = []
        self.set_blocks(blocks)
        self.img = None
        self.extracted_text = None
        self.needs_ocr = None

    def set_blocks(self, blocks):
        self.blocks = [
            Block(self, i, block["bbox"], block["type"], block.get("lines", []))
            for i, block in enumerate(blocks)
        ]

    def clip(self, bbox=None, save=None):
        if bbox is None:
            img = self.img
        else:
            with fitz.open('pdf', self.doc.bits) as pdf:
                page = pdf.load_page(self.page_num)
                rect = fitz.Rect(*bbox)
                zoom = fitz.Matrix(3, 3)
                pix = page.get_pixmap(clip=rect, matrix=zoom)
                img = pix.tobytes(output="jpeg", jpg_quality=95)
        if save:
            Path(save).write_bytes(img)
        return img

    @property
    def data(self):
        return {
            "i": self.i,
            "blocks": [block.data for block in self.blocks],
        }


class PDFDocument:
    def __init__(
        self,
        file_or_path: bytes | str | Path,
        filename: Optional[str] = None,
        dpi: int = 200,
    ):
        if isinstance(file_or_path, bytes):
            bits = file_or_path
        else:
            path = Path(file_or_path)
            filename = filename or path.name
            bits = path.read_bytes()
            

        self.filename = filename
        self.bits = bits
        self.dpi = dpi
        self.pages = []
        with fitz.open('pdf', bits) as pdf:
            for i in range(len(pdf)):
                self.pages.append(Page(self, i))

    def stream(self, batch_size=1):
        from .layout import predict_layout


        with fitz.open('pdf', self.bits) as pdf:
            for i, page in enumerate(pdf):
                self[i].img = page_to_image(page, self.dpi)
                self[i].needs_ocr = page_needs_ocr(page)
                if not self[i].needs_ocr:
                    self[i].extracted_text = extract_native_text(page)

        images = [page.img for page in self.pages]
        scale_factor = 72.0 / self.dpi

        for i, page_blocks in tqdm(enumerate(predict_layout(images, batch_size)), total=len(self.pages)):
            for block in page_blocks:
                block['bbox'] = [int(p * scale_factor) for p in block['bbox']]
            
            if self[i].needs_ocr:
                from .ocr import extract_ocr_text
                self[i].extracted_text = extract_ocr_text(self[i].img)
                for line in self[i].extracted_text:
                    line['bbox'] = [int(p * scale_factor) for p in line['bbox']]
            
            lines = pd.DataFrame(self[i].extracted_text)
            lines['x0'] = lines['bbox'].apply(lambda x: x[0])
            lines['y0'] = lines['bbox'].apply(lambda x: x[1])
            lines['x1'] = lines['bbox'].apply(lambda x: x[2])
            lines['y1'] = lines['bbox'].apply(lambda x: x[3])

            for block in page_blocks:
                block_lines = lines[
                    (lines['x1'] > block['bbox'][0]) & 
                    (lines['x0'] < block['bbox'][2]) & 
                    (lines['y1'] > block['bbox'][1]) & 
                    (lines['y0'] < block['bbox'][3])
                ]
                block['lines'] = block_lines[['bbox', 'content']].to_dict('records')
            self[i].set_blocks(page_blocks)
            yield self[i]

    def process(self, batch_size=1):
        for _ in self.stream(batch_size):
            pass
        return self
        
    @property
    def data(self):
        return {
            "filename": self.filename,
            "pages": [page.data for page in self.pages],
        }
    
    def save(self, path):
        path = Path(path)
        path.write_text(json.dumps(self.data, indent=2))

    def load(self, path_or_data):
        if isinstance(path_or_data, str) or isinstance(path_or_data, Path):
            path = Path(path_or_data)
            data = json.loads(path.read_text())
        else:
            data = path_or_data
        self.filename = data["filename"]
        for i, page in enumerate(data["pages"]):
            self[i].set_blocks(page["blocks"])
        
    def __getitem__(self, idx):
        return self.pages[idx]
    
    def __iter__(self):
        for page in self.pages:
            yield page

    def __len__(self):
        return len(self.pages)


def process_pdf(
        file_or_path: bytes | str | Path, 
        filename: Optional[str] = None, 
        dpi: int = 200, batch_size=1, 
        load: Optional[str | Path | dict] = None, 
        process: bool = False,
    ):
    doc = PDFDocument(file_or_path, filename, dpi)
    if load:
        doc.load(load)
    elif process:
        doc.process(batch_size)
    return doc
