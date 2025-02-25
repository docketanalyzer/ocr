from doclayout_yolo import YOLOv10
import torch
from .utils import BASE_DIR


LAYOUT_MODEL = None


LAYOUT_CHOICES = {
    0: 'title', 
    1: 'text', 
    2: 'abandon', 
    3: 'figure', 
    4: 'figure_caption', 
    5: 'table', 
    6: 'table_caption', 
    7: 'table_footnote', 
    8: 'isolate_formula', 
    9: 'formula_caption'
}


def merge_overlapping_blocks(blocks):
    """
    Merge all overlapping blocks regardless of type, with type priority.
    
    Args:
        blocks (list): List of dictionaries, each with 'type' and 'bbox' keys.
                      'bbox' is a tuple of (xmin, ymin, xmax, ymax).
    
    Returns:
        list: A new list with merged blocks.
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
        current_bbox = current['bbox']
        
        # Flag to check if any merge happened in this iteration
        merged = True
        
        while merged:
            merged = False
            
            # Check each remaining unprocessed block
            i = 0
            while i < len(unprocessed):
                other = unprocessed[i]
                other_bbox = other['bbox']
                
                # Check for overlap
                if boxes_overlap(current_bbox, other_bbox):
                    # Determine which type to keep based on priority
                    current_priority = type_priority.get(current['type'], max_priority)
                    other_priority = type_priority.get(other['type'], max_priority)
                    
                    # Keep the type with higher priority (lower number)
                    if other_priority < current_priority:
                        current['type'] = other['type']
                    
                    # Merge the bounding boxes
                    current_bbox = merge_boxes(current_bbox, other_bbox)
                    current['bbox'] = current_bbox
                    
                    # Remove the merged block from unprocessed
                    unprocessed.pop(i)
                    merged = True
                else:
                    i += 1
        
        # Add the merged block to the result
        result.append(current)
    
    # Sort by ymin and then xmin
    result.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
    return result


def boxes_overlap(box1, box2):
    """
    Check if two bounding boxes overlap.
    
    Args:
        box1 (tuple): (xmin, ymin, xmax, ymax) of first box
        box2 (tuple): (xmin, ymin, xmax, ymax) of second box
    
    Returns:
        bool: True if boxes overlap, False otherwise
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Check if one box is to the left of the other
    if x1_max < x2_min or x2_max < x1_min:
        return False
    
    # Check if one box is above the other
    if y1_max < y2_min or y2_max < y1_min:
        return False
    
    # If we get here, the boxes overlap
    return True


def merge_boxes(box1, box2):
    """
    Merge two overlapping bounding boxes.
    
    Args:
        box1 (tuple): (xmin, ymin, xmax, ymax) of first box
        box2 (tuple): (xmin, ymin, xmax, ymax) of second box
    
    Returns:
        tuple: The merged bounding box as (xmin, ymin, xmax, ymax)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # The merged box has the minimum of the mins and the maximum of the maxes
    return (
        min(x1_min, x2_min),
        min(y1_min, y2_min),
        max(x1_max, x2_max),
        max(y1_max, y2_max)
    )


def load_model():
    global LAYOUT_MODEL

    if LAYOUT_MODEL is None:
        LAYOUT_MODEL = YOLOv10(BASE_DIR / "models" / "doclayout_yolo_docstructbench_imgsz1280_2501.pt")

    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    LAYOUT_MODEL.to(device)

    return LAYOUT_MODEL, device


def predict_layout(images: list, batch_size: int):
    model, device = load_model()

    for idx in range(0, len(images), batch_size):
        results = [
            image_res.cpu()
            for image_res in model.predict(
                images[idx : idx + batch_size],
                imgsz=1280,
                conf=0.10,
                iou=0.45,
                verbose=False,
                device=device,
            )
        ]
        for result in results:
            blocks = []
            for xyxy, conf, cla in zip(
                result.boxes.xyxy,
                result.boxes.conf,
                result.boxes.cls,
            ):
                bbox = [int(p.item()) for p in xyxy]
                blocks.append({
                    'type': LAYOUT_CHOICES[int(cla.item())],
                    'bbox': bbox,
                    'score': round(float(conf.item()), 3),
                })
            blocks = merge_overlapping_blocks(blocks)
            yield blocks
