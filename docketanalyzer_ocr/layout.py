import torch
from doclayout_yolo import YOLOv10

from .utils import BASE_DIR

LAYOUT_MODEL = None


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

    # Create a priority map for faster lookup
    type_priority = {block_type: i for i, block_type in enumerate(LAYOUT_CHOICES.values())}

    # Add default priority for any types not in the list (lowest priority)
    max_priority = len(type_priority)

    # Start with all blocks as unprocessed
    unprocessed = [block.copy() for block in blocks]
    result = []

    while unprocessed:
        # Take a block as the current merged block
        current = unprocessed.pop(0)
        current_bbox = current["bbox"]

        # Flag to check if any merge happened in this iteration
        merged = True

        while merged:
            merged = False

            # Check each remaining unprocessed block
            i = 0
            while i < len(unprocessed):
                other = unprocessed[i]
                other_bbox = other["bbox"]

                # Check for overlap
                if boxes_overlap(current_bbox, other_bbox):
                    # Determine which type to keep based on priority
                    current_priority = type_priority.get(current["type"], max_priority)
                    other_priority = type_priority.get(other["type"], max_priority)

                    # Keep the type with higher priority (lower number)
                    if other_priority < current_priority:
                        current["type"] = other["type"]

                    # Merge the bounding boxes
                    current_bbox = merge_boxes(current_bbox, other_bbox)
                    current["bbox"] = current_bbox

                    # Remove the merged block from unprocessed
                    unprocessed.pop(i)
                    merged = True
                else:
                    i += 1

        # Add the merged block to the result
        result.append(current)

    # Sort by ymin and then xmin
    result.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return result


def boxes_overlap(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> bool:
    """Checks if two bounding boxes overlap.

    Args:
        box1: Tuple of (xmin, ymin, xmax, ymax) for the first box.
        box2: Tuple of (xmin, ymin, xmax, ymax) for the second box.

    Returns:
        bool: True if the boxes overlap, False otherwise.
    """
    # Extract coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Check for overlap
    if x1_max < x2_min or x2_max < x1_min:  # No horizontal overlap
        return False
    if y1_max < y2_min or y2_max < y1_min:  # No vertical overlap
        return False

    return True


def merge_boxes(
    box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Merges two bounding boxes into one that encompasses both.

    Args:
        box1: Tuple of (xmin, ymin, xmax, ymax) for the first box.
        box2: Tuple of (xmin, ymin, xmax, ymax) for the second box.

    Returns:
        tuple[float, float, float, float]: A new bounding box that contains both input boxes.
    """
    # Extract coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Create merged box with min/max coordinates
    merged_box = (
        min(x1_min, x2_min),
        min(y1_min, y2_min),
        max(x1_max, x2_max),
        max(y1_max, y2_max),
    )

    return merged_box


def load_model() -> tuple[YOLOv10, str]:
    """Loads and initializes the document layout detection model.

    Returns:
        tuple[YOLOv10, str]: A tuple containing:
            - The initialized YOLOv10 model
            - The device string ('cpu' or 'cuda')
    """
    global LAYOUT_MODEL

    device = "cpu" if not torch.cuda.is_available() else "cuda"

    if LAYOUT_MODEL is None:
        LAYOUT_MODEL = YOLOv10(BASE_DIR / "models" / "doclayout_yolo_docstructbench_imgsz1280_2501.pt")
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
        preds = model(batch)

        for pred in preds:
            blocks = []
            for xyxy, cla in zip(
                pred.boxes.xyxy,
                pred.boxes.cls,
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
