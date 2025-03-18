from docketanalyzer_core import download_file

from .utils import BASE_DIR

LAYOUT_MODEL = None
LAYOUR_MODEL_PATH = (
    BASE_DIR / "data" / "doclayout_yolo_docstructbench_imgsz1280_2501.pt"
)
LAYOUR_MODEL_URL = "https://github.com/docketanalyzer/ocr/raw/refs/heads/main/models/doclayout_yolo_docstructbench_imgsz1280_2501.pt"


LAYOUT_CHOICES = {
    0: "title",
    1: "text",
    2: "abandon",
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}


def merge_overlapping_blocks(blocks: list[dict]) -> list[dict]:
    """Merges overlapping layout blocks based on type priority.

    This function takes a list of detected layout blocks and merges any that overlap,
    keeping the type with the highest priority (lowest number in LAYOUT_CHOICES).

    Args:
        blocks: List of dictionaries, each with 'type' and 'bbox' keys.
               'bbox' is a tuple of (xmin, ymin, xmax, ymax).

    Returns:
        list[dict]: A new list with merged blocks, sorted by vertical
        position (y-coordinate) and then horizontal position (x-coordinate).
    """
    if not blocks:
        return []

    # Merged blocks with different types will get the type with the highest priority
    type_priority = {
        block_type: i for i, block_type in enumerate(LAYOUT_CHOICES.values())
    }

    unprocessed = [block.copy() for block in blocks]
    result = []

    while unprocessed:
        current = unprocessed.pop(0)
        current_bbox = current["bbox"]

        merged = True

        while merged:
            merged = False

            i = 0
            while i < len(unprocessed):
                other = unprocessed[i]
                other_bbox = other["bbox"]

                if box_overlap_pct(current_bbox, other_bbox) > 0.5:
                    current_priority = type_priority[current["type"]]
                    other_priority = type_priority[other["type"]]

                    if other_priority < current_priority:
                        current["type"] = other["type"]

                    current_bbox = merge_boxes(current_bbox, other_bbox)
                    current["bbox"] = current_bbox

                    unprocessed.pop(i)
                    merged = True
                else:
                    i += 1

        result.append(current)

    result.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return result


def box_overlap_pct(
    box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]
) -> float:
    """Calculates the percentage of overlap between two bounding boxes.

    The overlap percentage is calculated as the area of intersection divided by
    the smaller of the two box areas, resulting in a value between 0.0 and 1.0.

    Args:
        box1: Tuple of (xmin, ymin, xmax, ymax) for the first box.
        box2: Tuple of (xmin, ymin, xmax, ymax) for the second box.

    Returns:
        float: The overlap percentage (0.0 to 1.0) where:
            - 0.0 means no overlap
            - 1.0 means one box is completely contained within the other
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate the area of each box
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate intersection coordinates
    x_overlap_min = max(x1_min, x2_min)
    x_overlap_max = min(x1_max, x2_max)
    y_overlap_min = max(y1_min, y2_min)
    y_overlap_max = min(y1_max, y2_max)

    # Check if there is an overlap
    if x_overlap_max <= x_overlap_min or y_overlap_max <= y_overlap_min:
        return 0.0

    # Calculate the area of the intersection
    intersection_area = (x_overlap_max - x_overlap_min) * (
        y_overlap_max - y_overlap_min
    )

    # Calculate the overlap percentage relative to the smaller box
    smaller_area = min(area1, area2)

    return intersection_area / smaller_area


def merge_boxes(
    box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Merges two bounding boxes into one that encompasses both.

    Args:
        box1: Tuple of (xmin, ymin, xmax, ymax) for the first box.
        box2: Tuple of (xmin, ymin, xmax, ymax) for the second box.

    Returns:
        tuple[float, float, float, float]: A new bounding box that contains both
            input boxes.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    merged_box = (
        min(x1_min, x2_min),
        min(y1_min, y2_min),
        max(x1_max, x2_max),
        max(y1_max, y2_max),
    )

    return merged_box


def load_model() -> tuple["YOLOv10", str]:  # noqa: F821
    """Loads and initializes the document layout detection model.

    Returns:
        tuple[YOLOv10, str]: A tuple containing:
            - The initialized YOLOv10 model
            - The device string ('cpu' or 'cuda')
    """
    import torch
    from doclayout_yolo import YOLOv10

    global LAYOUT_MODEL

    device = "cpu" if not torch.cuda.is_available() else "cuda"

    if LAYOUT_MODEL is None:
        if not LAYOUR_MODEL_PATH.exists():
            LAYOUR_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            download_file(
                LAYOUR_MODEL_URL,
                LAYOUR_MODEL_PATH,
                description="Downloading layout model...",
            )
        LAYOUT_MODEL = YOLOv10(LAYOUR_MODEL_PATH, verbose=False)
        LAYOUT_MODEL.to(device)

    return LAYOUT_MODEL, device


def predict_layout(images: list, batch_size: int = 8) -> list[list[dict]]:
    """Predicts document layout elements in a batch of images.

    This function processes a batch of images through the layout detection model
    to identify different document elements like text, tables, figures, etc.

    Args:
        images: List of images to process.
        batch_size: Number of images to process in each batch.

    Returns:
        list[list[dict]]: For each input image, a list of detected layout blocks,
        where each block is a dictionary with 'type' and 'bbox' keys.
    """
    model, _ = load_model()

    results = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        preds = model(batch, verbose=False)

        for pred in preds:
            blocks = []
            for xyxy, cla in zip(
                pred.boxes.xyxy,
                pred.boxes.cls,
                strict=False,
            ):
                bbox = [int(p.item()) for p in xyxy]
                blocks.append(
                    {
                        "type": LAYOUT_CHOICES[int(cla.item())],
                        "bbox": bbox,
                    }
                )
            blocks = merge_overlapping_blocks(blocks)
            results.append(blocks)

    return results
