"""
Dataset format detection and sampling.

Detects known formats (COCO, YOLO, Pascal VOC, etc.) from file contents
and directory structure. Also samples files for AI-assisted analysis.
"""

import json
import os
from pathlib import Path
from typing import Optional


# ── Format Detection ──────────────────────────────────────────────────────────

KNOWN_FORMATS = [
    "coco_json",
    "yolo",
    "pascal_voc",
    "fishclef_xml",
    "folder_encoded",  # annotations in folder/filenames (like 18600403)
    "csv",
    "unknown",
]


def detect_format(dataset_path: str) -> dict:
    """
    Analyze a dataset path and detect its format.

    Returns:
        dict with keys:
            - format: str (one of KNOWN_FORMATS)
            - confidence: float (0-1)
            - annotation_files: list of annotation file paths
            - image_files: list of image file paths
            - details: str (human-readable explanation)
            - directory_tree: str (sampled directory structure)
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        return {
            "format": "unknown",
            "confidence": 0.0,
            "annotation_files": [],
            "image_files": [],
            "details": f"Path does not exist: {dataset_path}",
            "directory_tree": "",
        }

    # Collect all files. The CLI accepts either a dataset directory or a
    # single annotation file; scan the parent for directory-tree context when
    # a file is provided, but only classify the given file as input.
    all_files = []
    image_files = []
    annotation_files = []
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    ANNOTATION_EXTS = {".json", ".xml", ".csv", ".txt", ".yaml", ".yml"}

    if dataset_path.is_file():
        all_files.append(dataset_path)
        ext = dataset_path.suffix.lower()
        if ext in IMAGE_EXTS:
            image_files.append(dataset_path)
        elif ext in ANNOTATION_EXTS:
            annotation_files.append(dataset_path)
        tree_root = dataset_path.parent
    else:
        for root, dirs, files in os.walk(dataset_path):
            # skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                if f.startswith("."):
                    continue
                fp = Path(root) / f
                all_files.append(fp)
                ext = fp.suffix.lower()
                if ext in IMAGE_EXTS:
                    image_files.append(fp)
                elif ext in ANNOTATION_EXTS:
                    annotation_files.append(fp)
        tree_root = dataset_path

    dir_tree = _build_directory_tree(tree_root, max_depth=3, max_items=10)

    # Try each detector in order of specificity
    result = (
        _detect_coco(dataset_path, annotation_files, image_files)
        or _detect_yolo(dataset_path, annotation_files, image_files)
        or _detect_pascal_voc(dataset_path, annotation_files, image_files)
        or _detect_fishclef(dataset_path, annotation_files, image_files)
        or _detect_folder_encoded(dataset_path, annotation_files, image_files)
        or _detect_csv(dataset_path, annotation_files, image_files)
    )

    if result is None:
        result = {
            "format": "unknown",
            "confidence": 0.0,
            "details": "Could not identify dataset format.",
        }

    result["annotation_files"] = [str(f) for f in annotation_files]
    result["image_files"] = [str(f) for f in image_files[:20]]  # sample
    result["image_count"] = len(image_files)
    result["directory_tree"] = dir_tree

    return result


# ── Individual Format Detectors ───────────────────────────────────────────────

def _detect_coco(path: Path, ann_files: list, img_files: list) -> Optional[dict]:
    """Detect COCO JSON format."""
    json_files = [f for f in ann_files if f.suffix.lower() == ".json"]
    coco_files = []
    total_images = 0
    total_anns = 0
    categories = []
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                # Read just enough to check top-level keys
                data = json.load(f)
            if isinstance(data, dict):
                keys = set(data.keys())
                coco_keys = {"images", "annotations", "categories"}
                if coco_keys.issubset(keys):
                    coco_files.append(jf)
                    total_images += len(data.get("images", []))
                    total_anns += len(data.get("annotations", []))
                    for cat in data.get("categories", []):
                        name = cat.get("name", "")
                        if name and name not in categories:
                            categories.append(name)
        except (json.JSONDecodeError, OSError):
            continue
    if coco_files:
        return {
            "format": "coco_json",
            "confidence": 0.95,
            "details": (
                f"COCO JSON format detected in {len(coco_files)} file(s). "
                f"{total_images} images, {total_anns} annotations, "
                f"{len(categories)} categories: {categories[:10]}"
            ),
            "coco_files": [str(f) for f in sorted(coco_files)],
        }
    return None


def _detect_yolo(path: Path, ann_files: list, img_files: list) -> Optional[dict]:
    """Detect YOLO format (txt files with 'class cx cy w h' per line)."""
    txt_files = [f for f in ann_files if f.suffix.lower() == ".txt"]
    classes_file = None
    label_files = []

    for tf in txt_files:
        if tf.stem.lower() in ("classes", "names", "obj"):
            classes_file = tf
        else:
            label_files.append(tf)

    if not label_files:
        return None

    # Check if txt files follow YOLO format: "class_id cx cy w h" per line
    yolo_count = 0
    for lf in label_files[:5]:  # sample 5 files
        try:
            with open(lf, "r") as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    # All values should be numeric, coords between 0-1
                    vals = [float(p) for p in parts]
                    if all(0 <= v <= 1 for v in vals[1:]):
                        yolo_count += 1
        except (ValueError, OSError):
            continue

    if yolo_count >= 3:
        return {
            "format": "yolo",
            "confidence": 0.85,
            "details": (
                f"YOLO format detected. {len(label_files)} label files. "
                f"Classes file: {classes_file.name if classes_file else 'NOT FOUND'}."
            ),
            "classes_file": str(classes_file) if classes_file else None,
        }
    return None


def _detect_pascal_voc(path: Path, ann_files: list, img_files: list) -> Optional[dict]:
    """Detect Pascal VOC XML format."""
    xml_files = [f for f in ann_files if f.suffix.lower() == ".xml"]
    if not xml_files:
        return None

    import xml.etree.ElementTree as ET

    for xf in xml_files[:3]:
        try:
            tree = ET.parse(xf)
            root = tree.getroot()
            if root.tag == "annotation":
                # Pascal VOC has <annotation><object><bndbox> structure
                objs = root.findall(".//object")
                bndboxes = root.findall(".//bndbox")
                if objs and bndboxes:
                    return {
                        "format": "pascal_voc",
                        "confidence": 0.90,
                        "details": (
                            f"Pascal VOC XML detected. {len(xml_files)} annotation files."
                        ),
                    }
        except (ET.ParseError, OSError):
            continue
    return None


def _detect_fishclef(path: Path, ann_files: list, img_files: list) -> Optional[dict]:
    """Detect FishCLEF-style XML (video > frame > object structure)."""
    xml_files = [f for f in ann_files if f.suffix.lower() == ".xml"]
    if not xml_files:
        return None

    import xml.etree.ElementTree as ET

    for xf in xml_files[:3]:
        try:
            tree = ET.parse(xf)
            root = tree.getroot()
            if root.tag == "video":
                frames = root.findall(".//frame")
                objects = root.findall(".//object")
                if frames and objects:
                    return {
                        "format": "fishclef_xml",
                        "confidence": 0.90,
                        "details": (
                            f"FishCLEF XML detected. {len(xml_files)} video annotation files "
                            f"with <video><frame><object> structure."
                        ),
                    }
        except (ET.ParseError, OSError):
            continue
    return None


def _detect_folder_encoded(
    path: Path, ann_files: list, img_files: list
) -> Optional[dict]:
    """
    Detect datasets where annotations are encoded in folder/file names.
    e.g., species_name/image_WxH+X+Y.jpg
    """
    if ann_files:
        # If there are annotation files, this probably isn't folder-encoded
        return None

    if not img_files:
        return None

    # Check if images are organized into named subfolders
    parent_dirs = set()
    for img in img_files:
        rel = img.relative_to(path)
        if len(rel.parts) >= 2:
            parent_dirs.add(rel.parts[-2])

    if len(parent_dirs) < 2:
        return None

    # Check if filenames contain bbox-like patterns: WxH+X+Y or X_Y_W_H.
    import re

    bbox_patterns = [
        re.compile(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)"),
        re.compile(r"_(-?\d+)_(-?\d+)_(\d+)_(\d+)\.\w+$"),
    ]
    bbox_matches = 0
    for img in img_files[:20]:
        if any(pattern.search(img.name) for pattern in bbox_patterns):
            bbox_matches += 1

    has_bbox_in_name = bbox_matches > len(img_files[:20]) * 0.5

    details = (
        f"No annotation files found. {len(img_files)} images organized into "
        f"{len(parent_dirs)} named folders."
    )
    if has_bbox_in_name:
        details += (
            " Bounding box pattern detected in filenames."
        )

    return {
        "format": "folder_encoded",
        "confidence": 0.6 if has_bbox_in_name else 0.3,
        "details": details,
        "folder_names": sorted(parent_dirs)[:20],
        "has_bbox_in_filename": has_bbox_in_name,
    }


def _detect_csv(path: Path, ann_files: list, img_files: list) -> Optional[dict]:
    """Detect CSV/TSV annotation files."""
    csv_files = [
        f for f in ann_files if f.suffix.lower() in (".csv", ".tsv")
    ]
    if not csv_files:
        return None

    for cf in csv_files[:3]:
        try:
            with open(cf, "r") as f:
                header = f.readline().strip()
            if header:
                # Detect delimiter
                delimiter = "\t" if "\t" in header else ","
                columns = [c.strip().strip('"').strip("'") for c in header.split(delimiter)]
                return {
                    "format": "csv",
                    "confidence": 0.70,
                    "details": (
                        f"CSV file detected: {cf.name}. "
                        f"Columns: {columns}"
                    ),
                    "columns": columns,
                    "delimiter": delimiter,
                }
        except OSError:
            continue
    return None


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_dataset(dataset_path: str, detection_result: dict, max_lines: int = 50) -> dict:
    """
    Create a representative sample of the dataset for AI analysis.

    Returns:
        dict with keys:
            - file_samples: list of {path, content_preview} dicts
            - directory_tree: str
            - image_samples: list of image filenames
            - stats: dict with counts
    """
    dataset_path = Path(dataset_path)
    samples = {
        "file_samples": [],
        "directory_tree": detection_result.get("directory_tree", ""),
        "image_samples": [],
        "stats": {
            "image_count": detection_result.get("image_count", 0),
            "annotation_file_count": len(detection_result.get("annotation_files", [])),
            "format_detected": detection_result["format"],
        },
    }

    # Sample annotation files
    for ann_path in detection_result.get("annotation_files", [])[:3]:
        try:
            with open(ann_path, "r", encoding="utf-8", errors="replace") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line)
            samples["file_samples"].append({
                "path": ann_path,
                "content_preview": "".join(lines),
            })
        except OSError:
            continue

    # Sample image filenames (with paths for structure visibility)
    for img_path in detection_result.get("image_files", [])[:15]:
        try:
            rel = Path(img_path).relative_to(dataset_path)
            samples["image_samples"].append(str(rel))
        except ValueError:
            samples["image_samples"].append(img_path)

    return samples


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_directory_tree(
    path: Path, max_depth: int = 3, max_items: int = 10, _depth: int = 0
) -> str:
    """Build a readable directory tree string."""
    if _depth > max_depth:
        return ""

    indent = "    " * _depth
    result = ""

    if _depth == 0:
        result += f"{path.name}/\n"

    try:
        items = sorted(path.iterdir())
    except PermissionError:
        return result + f"{indent}    [permission denied]\n"

    # Separate dirs and files, skip hidden
    dirs = [i for i in items if i.is_dir() and not i.name.startswith(".")]
    files = [i for i in items if i.is_file() and not i.name.startswith(".")]

    # Show files (limited)
    for f in files[:max_items]:
        result += f"{indent}    {f.name}\n"
    if len(files) > max_items:
        result += f"{indent}    ... ({len(files) - max_items} more files)\n"

    # Show dirs (limited, recurse)
    for d in dirs[:max_items]:
        child_count = sum(1 for _ in d.iterdir()) if d.is_dir() else 0
        result += f"{indent}    {d.name}/ ({child_count} items)\n"
        result += _build_directory_tree(d, max_depth, max_items, _depth + 1)
    if len(dirs) > max_items:
        result += f"{indent}    ... ({len(dirs) - max_items} more folders)\n"

    return result
