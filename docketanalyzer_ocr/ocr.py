from typing import Any, Union

import torch
from paddleocr import PaddleOCR

OCR_MODEL = None


def load_model() -> tuple[PaddleOCR, str]:
    """Loads and initializes the PaddleOCR model.

    This function initializes the OCR model if it hasn't been loaded yet.
    It determines whether to use CPU or CUDA based on availability.

    Returns:
        tuple[PaddleOCR, str]: A tuple containing:
            - The initialized PaddleOCR model
            - The device string ('cpu' or 'cuda')
    """
    global OCR_MODEL

    device = "cpu" if not torch.cuda.is_available() else "cuda"

    if OCR_MODEL is None:
        OCR_MODEL = PaddleOCR(
            lang="en",
            use_gpu=device == "cuda",
            gpu_mem=5000,
            precision="bf16",
            show_log=False,
        )

    return OCR_MODEL, device


def extract_ocr_text(image: Union[str, bytes, Any]) -> list[dict]:
    """Extracts text from an image using OCR.

    This function processes an image with the PaddleOCR model to extract text
    and bounding boxes for each detected text line.

    Args:
        image: The input image. Can be a file path, bytes, or a numpy array.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - 'bbox': The bounding box coordinates [x1, y1, x2, y2]
            - 'content': The extracted text content
    """
    model, _ = load_model()

    result = model.ocr(image, cls=False)
    data = []
    for idx in range(len(result)):
        res = result[idx]
        if res:
            for line in res:
                data.append(
                    {
                        "bbox": line[0][0] + line[0][2],
                        "content": line[1][0],
                    }
                )
    return data
