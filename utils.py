"""
Utility functions for image preprocessing and visualization.
"""
import cv2
import numpy as np


_CLAHE_CACHE = {}


def get_clahe(clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8)):
    """Get (and cache) a CLAHE instance for the given parameters."""
    key = (float(clahe_clip_limit), tuple(clahe_tile_grid_size))
    clahe = _CLAHE_CACHE.get(key)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=key[0], tileGridSize=key[1])
        _CLAHE_CACHE[key] = clahe
    return clahe


def preprocess_image(image, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8), clahe=None):
    """
    Preprocess image by converting to LAB color space and applying CLAHE on L channel.
    
    Args:
        image: Input BGR image
        clahe_clip_limit: CLAHE clip limit
        clahe_tile_grid_size: CLAHE tile grid size
        
    Returns:
        Preprocessed BGR image
    """
    # Convert BGR to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Extract L channel
    l_channel = lab[:, :, 0]
    
    # Apply CLAHE to L channel
    if clahe is None:
        clahe = get_clahe(clahe_clip_limit, clahe_tile_grid_size)
    l_channel_enhanced = clahe.apply(l_channel)
    
    # Merge channels back
    lab[:, :, 0] = l_channel_enhanced
    
    # Convert back to BGR
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_image


def draw_roi(image, roi, color, label="ROI"):
    """
    Draw ROI rectangle on image.
    
    Args:
        image: Image to draw on
        roi: ROI coordinates (x, y, w, h)
        color: BGR color tuple
        label: Label text for ROI
    """
    x, y, w, h = roi
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def get_centroid_from_bbox(bbox, fmt):
    """
    Calculate centroid from bounding box.

    Args:
        bbox: Bounding box coordinates
              - if fmt == "xywh": (x, y, w, h)
              - if fmt == "xyxy": (x1, y1, x2, y2)
        fmt: Bounding box format: "xywh" or "xyxy"

    Returns:
        (cx, cy) centroid coordinates
    """
    if fmt not in ("xywh", "xyxy"):
        raise ValueError(f"Unsupported bbox format: {fmt!r}. Use 'xywh' or 'xyxy'.")

    if bbox is None or len(bbox) != 4:
        raise ValueError(f"Expected bbox of length 4, got: {bbox!r}")

    if fmt == "xywh":
        x, y, w, h = bbox
        cx = x + w // 2
        cy = y + h // 2
        return (int(cx), int(cy))

    # fmt == "xyxy"
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return (int(cx), int(cy))

def calculate_bbox_overlap(bbox1, bbox2, fmt1="xywh", fmt2=None):
    """
    Calculate overlap ratio between two bounding boxes.
    Returns the ratio of intersection area to the smaller box's area.

    Args:
        bbox1: First bounding box coordinates
        bbox2: Second bounding box coordinates
        fmt1: Format for bbox1: "xywh" or "xyxy" (default "xywh")
        fmt2: Format for bbox2: "xywh" or "xyxy" (default: same as fmt1)

    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    if fmt2 is None:
        fmt2 = fmt1

    if fmt1 not in ("xywh", "xyxy"):
        raise ValueError(f"Unsupported bbox format for bbox1: {fmt1!r}. Use 'xywh' or 'xyxy'.")
    if fmt2 not in ("xywh", "xyxy"):
        raise ValueError(f"Unsupported bbox format for bbox2: {fmt2!r}. Use 'xywh' or 'xyxy'.")

    if bbox1 is None or len(bbox1) != 4:
        raise ValueError(f"Expected bbox1 of length 4, got: {bbox1!r}")
    if bbox2 is None or len(bbox2) != 4:
        raise ValueError(f"Expected bbox2 of length 4, got: {bbox2!r}")

    # Normalize to (x1, y1, x2, y2)
    if fmt1 == "xywh":
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    else:  # "xyxy"
        x1_1, y1_1, x2_1, y2_1 = bbox1

    if fmt2 == "xywh":
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    else:  # "xyxy"
        x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    if area1 <= 0 or area2 <= 0:
        return 0.0

    smaller_area = min(area1, area2)
    return intersection_area / smaller_area


def calculate_detection_overlap(det1, det2, overlap_threshold=0.5, bbox_format="xywh"):
    """
    Calculate overlap between two detections.

    Args:
        det1: First detection dictionary
        det2: Second detection dictionary
        overlap_threshold: Minimum overlap ratio to consider as overlap (default 0.5 = 50%)
        bbox_format: Bounding box format for det['bbox']: "xywh" or "xyxy" (default "xywh")

    Returns:
        True if overlap exceeds threshold, False otherwise
    """
    overlap_ratio = 0.0

    # Try bbox-based overlap first
    if det1.get("bbox") is not None and det2.get("bbox") is not None:
        overlap_ratio = calculate_bbox_overlap(
            det1["bbox"],
            det2["bbox"],
            fmt1=bbox_format,
            fmt2=bbox_format,
        )
    else:
        # If no bbox, use centroid distance as fallback
        if "centroid" in det1 and "centroid" in det2:
            dist = np.sqrt(
                (det1["centroid"][0] - det2["centroid"][0]) ** 2
                + (det1["centroid"][1] - det2["centroid"][1]) ** 2
            )
            # If centroids are very close (< 10 pixels), consider it overlap
            if dist < 10:
                overlap_ratio = 0.6  # Assume overlap if very close

    return overlap_ratio >= overlap_threshold

