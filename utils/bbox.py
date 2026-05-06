"""
Bounding box coordinate format conversions.

Supports converting between common formats:
  - xywh:       [x, y, width, height]  (FathomNet, COCO)
  - xyxy:       [x1, y1, x2, y2]      (Pascal VOC)
  - cxcywh:     [cx, cy, w, h]         (YOLO, normalized 0-1)
  - cxcywh_abs: [cx, cy, w, h]         (YOLO, absolute pixels)
"""

from enum import Enum
from typing import Optional


class BBoxFormat(Enum):
    XYWH = "xywh"           # x, y, width, height (top-left origin) — FathomNet target
    XYXY = "xyxy"           # x1, y1, x2, y2 (top-left, bottom-right)
    CXCYWH_NORM = "cxcywh"  # center_x, center_y, w, h (normalized 0-1)
    CXCYWH_ABS = "cxcywh_abs"  # center_x, center_y, w, h (absolute pixels)


def convert_bbox(
    coords: tuple,
    from_format: BBoxFormat,
    to_format: BBoxFormat = BBoxFormat.XYWH,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> tuple:
    """
    Convert bounding box coordinates between formats.

    Args:
        coords: tuple of 4 numeric values
        from_format: source format
        to_format: target format (default: XYWH for FathomNet)
        image_width: required for normalized ↔ pixel conversions
        image_height: required for normalized ↔ pixel conversions

    Returns:
        tuple of 4 values in the target format
    """
    a, b, c, d = [float(v) for v in coords]

    # Step 1: Convert to intermediate XYWH (absolute pixels)
    if from_format == BBoxFormat.XYWH:
        x, y, w, h = a, b, c, d

    elif from_format == BBoxFormat.XYXY:
        x, y = a, b
        w = c - a
        h = d - b

    elif from_format == BBoxFormat.CXCYWH_NORM:
        if image_width is None or image_height is None:
            raise ValueError(
                "image_width and image_height required for normalized coordinates"
            )
        cx_abs = a * image_width
        cy_abs = b * image_height
        w = c * image_width
        h = d * image_height
        x = cx_abs - w / 2
        y = cy_abs - h / 2

    elif from_format == BBoxFormat.CXCYWH_ABS:
        w, h = c, d
        x = a - w / 2
        y = b - h / 2

    else:
        raise ValueError(f"Unknown source format: {from_format}")

    # Step 2: Convert from XYWH to target format
    if to_format == BBoxFormat.XYWH:
        return (round(x), round(y), round(w), round(h))

    elif to_format == BBoxFormat.XYXY:
        return (round(x), round(y), round(x + w), round(y + h))

    elif to_format == BBoxFormat.CXCYWH_NORM:
        if image_width is None or image_height is None:
            raise ValueError(
                "image_width and image_height required for normalized coordinates"
            )
        cx = (x + w / 2) / image_width
        cy = (y + h / 2) / image_height
        return (cx, cy, w / image_width, h / image_height)

    elif to_format == BBoxFormat.CXCYWH_ABS:
        cx = x + w / 2
        cy = y + h / 2
        return (round(cx), round(cy), round(w), round(h))

    else:
        raise ValueError(f"Unknown target format: {to_format}")
