"""
Folder-encoded dataset → FathomNet converter.

Handles datasets where annotations are encoded in the directory structure
and filenames rather than in separate annotation files.

Example: species_name/image_WxH+X+Y.jpg
  - concept → folder name
  - bounding box → parsed from filename
"""

import re
from pathlib import Path
from typing import Generator, Optional

from .base import BaseConverter, FathomNetRecord


# Common filename bbox patterns. The mapping values refer to capture groups.
BBOX_PATTERNS = [
    {
        "regex": re.compile(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+)"),
        "groups": {"width": 1, "height": 2, "x": 3, "y": 4},
    },
    {
        "regex": re.compile(r"_(-?\d+)_(-?\d+)_(\d+)_(\d+)\.\w+$"),
        "groups": {"x": 1, "y": 2, "width": 3, "height": 4},
    },
]


class FolderEncodedConverter(BaseConverter):
    format_name = "folder_encoded"
    format_description = "Folder/filename-encoded annotations (concept in folder name, bbox in filename)"

    def convert(
        self,
        dataset_path: str,
        detection_result: dict,
        field_overrides: Optional[dict] = None,
    ) -> Generator[FathomNetRecord, None, None]:
        overrides = field_overrides or {}
        dataset_path = Path(dataset_path)

        # User can override the bbox regex pattern
        custom_pattern = overrides.get("bbox_pattern")
        if custom_pattern:
            bbox_regex = re.compile(custom_pattern)
        else:
            bbox_regex = None  # will try all default patterns

        # User can override how concept is derived
        concept_source = overrides.get("concept_source", "folder_name")

        # User can override group order for a custom bbox regex. Built-in
        # patterns use their documented group orders.
        group_mapping = overrides.get("bbox_groups", {
            "width": 1,
            "height": 2,
            "x": 3,
            "y": 4,
        })

        use_custom_group_mapping = "bbox_groups" in overrides

        # Set to True when images are pre-cropped to the bounding box region.
        # The submission box becomes (0, 0, w, h) and the original position
        # from the filename is preserved as source_x / source_y extra columns.
        # Leave False (default) when images are full frames and the filename
        # coords describe a region within the image.
        images_are_crops = overrides.get("images_are_crops", False)

        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

        for img_path in dataset_path.rglob("*"):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            if img_path.name.startswith("."):
                continue

            # Get concept from folder name
            if concept_source in {"folder_name", "folder.name", "folder", "name"}:
                concept = img_path.parent.name.replace("_", " ").strip()
            else:
                concept = "Unknown"

            # Try to extract bbox from filename
            bbox = self._extract_bbox(
                img_path.name,
                bbox_regex,
                group_mapping,
                use_custom_group_mapping=use_custom_group_mapping,
            )

            if bbox:
                x, y, w, h = bbox
            else:
                self.record_drop(
                    item_type="image",
                    reason="No bounding box pattern found in filename",
                    source=str(img_path),
                    image=img_path.name,
                    source_image_path=str(img_path),
                    concept=concept,
                )
                continue

            if images_are_crops:
                # Image is already cropped to the bbox region. The submission
                # box covers the whole image; original position kept as metadata.
                record = FathomNetRecord(
                    concept=concept,
                    image=img_path.name,
                    x=0,
                    y=0,
                    width=w,
                    height=h,
                    source_image_path=str(img_path),
                )
                record.extra["source_x"] = x
                record.extra["source_y"] = y
            else:
                # Image is a full frame; bbox coords point to a region within it.
                record = FathomNetRecord(
                    concept=concept,
                    image=img_path.name,
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    source_image_path=str(img_path),
                )
            context = {
                "folder": {
                    "name": img_path.parent.name,
                    "path": str(img_path.parent),
                },
                "image": {
                    "filename": img_path.name,
                    "stem": img_path.stem,
                    "extension": img_path.suffix,
                    "path": str(img_path),
                    "relative_path": str(img_path.relative_to(dataset_path))
                    if img_path.is_relative_to(dataset_path)
                    else img_path.name,
                },
                "filename": {
                    "bbox": [x, y, w, h],
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                },
            }
            self._apply_optional_overrides_from_context(record, overrides, context)
            self._apply_extra_columns_from_context(record, overrides, context)
            yield record

    def _extract_bbox(
        self,
        filename: str,
        custom_regex: Optional[re.Pattern],
        group_mapping: dict,
        *,
        use_custom_group_mapping: bool = False,
    ) -> Optional[tuple]:
        """Extract bounding box coordinates from a filename."""
        if custom_regex:
            match = custom_regex.search(filename)
            if match:
                try:
                    w = int(match.group(group_mapping["width"]))
                    h = int(match.group(group_mapping["height"]))
                    x = int(match.group(group_mapping["x"]))
                    y = int(match.group(group_mapping["y"]))
                    return (x, y, w, h)
                except (IndexError, ValueError):
                    return None

        # Try default patterns
        for pattern_config in BBOX_PATTERNS:
            pattern = pattern_config["regex"]
            default_groups = (
                group_mapping if use_custom_group_mapping else pattern_config["groups"]
            )
            match = pattern.search(filename)
            if match:
                try:
                    w = int(match.group(default_groups["width"]))
                    h = int(match.group(default_groups["height"]))
                    x = int(match.group(default_groups["x"]))
                    y = int(match.group(default_groups["y"]))
                    return (x, y, w, h)
                except (IndexError, ValueError):
                    continue

        return None

    def get_field_names(self, dataset_path: str, detection_result: dict) -> list[str]:
        return sorted(self.get_field_samples(dataset_path, detection_result).keys())

    def get_field_samples(
        self, dataset_path: str, detection_result: dict
    ) -> dict[str, list[str]]:
        fields = set()
        samples: dict[str, list[str]] = {}
        dataset_path = Path(dataset_path)
        image_files = [
            Path(f) for f in detection_result.get("image_files", [])
            if Path(f).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        ]
        if not image_files:
            image_files = []
            for img_path in dataset_path.rglob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
                    image_files.append(img_path)
                if len(image_files) >= 20:
                    break

        for img_path in image_files[:20]:
            self._add_field_sample(fields, samples, "folder.name", img_path.parent.name)
            self._add_field_sample(fields, samples, "folder.path", str(img_path.parent))
            self._add_field_sample(fields, samples, "image.filename", img_path.name)
            self._add_field_sample(fields, samples, "image.stem", img_path.stem)
            self._add_field_sample(fields, samples, "image.extension", img_path.suffix)
            self._add_field_sample(fields, samples, "image.path", str(img_path))
            try:
                rel_path = str(img_path.relative_to(dataset_path))
            except ValueError:
                rel_path = img_path.name
            self._add_field_sample(fields, samples, "image.relative_path", rel_path)

            bbox = self._extract_bbox(
                img_path.name,
                custom_regex=None,
                group_mapping={"width": 1, "height": 2, "x": 3, "y": 4},
            )
            if not bbox:
                continue
            x, y, w, h = bbox
            self._add_field_sample(fields, samples, "filename.bbox", [x, y, w, h])
            self._add_field_sample(fields, samples, "filename.x", x)
            self._add_field_sample(fields, samples, "filename.y", y)
            self._add_field_sample(fields, samples, "filename.width", w)
            self._add_field_sample(fields, samples, "filename.height", h)

        for field_name in (
            "folder.name",
            "folder.path",
            "image.filename",
            "image.stem",
            "image.extension",
            "image.path",
            "image.relative_path",
            "filename.bbox",
            "filename.x",
            "filename.y",
            "filename.width",
            "filename.height",
        ):
            self._add_field_sample(fields, samples, field_name, "")

        return {field: samples.get(field, []) for field in sorted(fields)}

    def get_default_field_map(
        self, dataset_path: str, detection_result: dict
    ) -> dict[str, str]:
        return {
            "concept": "folder.name",
            "image": "image.filename",
            "x": "filename.bbox",
            "y": "filename.bbox",
            "width": "filename.bbox",
            "height": "filename.bbox",
        }

    def validate_prerequisites(
        self, dataset_path: str, detection_result: dict
    ) -> list[str]:
        errors = []
        if not detection_result.get("has_bbox_in_filename", False):
            errors.append(
                "No bounding box pattern detected in filenames. "
                "You may need to provide a custom regex pattern."
            )
        if not detection_result.get("folder_names"):
            errors.append("No named subfolders found for concept extraction.")
        return errors
