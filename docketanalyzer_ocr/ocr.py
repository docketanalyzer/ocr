from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor


RECOGNITION_MODEL = None
DETECTION_MODEL = None


def load_model() -> tuple["RecognitionPredictor", "DetectionPredictor"]:
    """Loads and initializes the OCR models.

    Returns:
        tuple[RecognitionPredictor, DetectionPredictor]: A tuple containing:
            - The initialized RecognitionPredictor model
            - The initialized DetectionPredictor model
    """
    from surya.detection import DetectionPredictor
    from surya.recognition import RecognitionPredictor

    global RECOGNITION_MODEL, DETECTION_MODEL

    if RECOGNITION_MODEL is None:
        RECOGNITION_MODEL = RecognitionPredictor()
    if DETECTION_MODEL is None:
        DETECTION_MODEL = DetectionPredictor()

    return RECOGNITION_MODEL, DETECTION_MODEL


def extract_text(imgs: list[Any], langs: list[str] | str | None = "en") -> list[dict]:
    """Extracts text from an image using the OCR service.

    This function sends an image to the OCR service for processing and returns
    the extracted text and bounding boxes.

    Args:
        imgs: A list of input images.
        langs: A list of language codes to use for OCR. Defaults to ['en'].

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - 'bbox': The bounding box coordinates [x1, y1, x2, y2]
            - 'content': The extracted text content
    """
    if isinstance(langs, str):
        langs = [langs]
    langs = [langs] * len(imgs)

    recognition_model, detection_model = load_model()

    preds = recognition_model(imgs, langs, detection_model)

    results = []
    for pred in preds:
        results.append([])
        for line in pred.text_lines:
            results[-1].append({"bbox": line.bbox, "content": line.text})
    return results
