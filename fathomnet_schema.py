"""
FathomNet target schema definition.
Single source of truth for required/optional fields and validation rules.
"""

REQUIRED_FIELDS = {
    "concept":  "Species or object name (scientific name preferred)",
    "image":    "Image filename or URL",
    "x":        "Bounding box top-left X coordinate in pixels",
    "y":        "Bounding box top-left Y coordinate in pixels",
    "width":    "Bounding box width in pixels",
    "height":   "Bounding box height in pixels",
}

OPTIONAL_FIELDS = {
    "depth":          "Depth in meters",
    "altitude":       "Altitude in meters",
    "latitude":       "Latitude in decimal degrees (-90 to 90)",
    "longitude":      "Longitude in decimal degrees (-180 to 180)",
    "temperature":    "Temperature in Celsius",
    "salinity":       "Salinity in PSU",
    "oxygen":         "Dissolved oxygen in ml/L",
    "pressure":       "Pressure in dbar",
    "observer":       "Who made the annotation",
    "timestamp":      "ISO 8601 timestamp",
    "imagingtype":    "Imaging type (e.g., ROV, AUV, camera)",
    "occluded":       "Whether the object is partially occluded",
    "truncated":      "Whether the object extends beyond image edge",
    "userdefinedkey": "Custom key for linking to source systems (max 56 chars)",
    "altconcept":     "Alternate taxonomy name",
    "groupof":        "Whether annotation represents a group",
}

ALL_FIELDS = {**REQUIRED_FIELDS, **OPTIONAL_FIELDS}

NOAA_NCEI_IMAGE_EXTS = {".jpg", ".png"}

# Common aliases for required fields (used by fuzzy matching)
FIELD_ALIASES = {
    "concept": [
        "class", "class_name", "className", "label", "name", "category",
        "category_name", "species", "fish_species", "species_name", "tag",
        "object_name", "object_class", "annotation", "taxon",
    ],
    "image": [
        "filename", "file_name", "image_path", "img_path", "img_name",
        "image_name", "url", "image_url", "img_url", "path", "file",
        "image_file", "img", "photo", "image_filename",
    ],
    "x": [
        "xmin", "x_min", "left", "bbox_x", "box_x", "x1", "x_start",
        "topleft_x", "roi_x",
    ],
    "y": [
        "ymin", "y_min", "top", "bbox_y", "box_y", "y1", "y_start",
        "topleft_y", "roi_y",
    ],
    "width": [
        "w", "bbox_width", "box_width", "bbox_w", "box_w", "roi_width",
        "roi_w",
    ],
    "height": [
        "h", "bbox_height", "box_height", "bbox_h", "box_h", "roi_height",
        "roi_h",
    ],
}
