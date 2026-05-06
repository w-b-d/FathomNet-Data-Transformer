"""
YOLO txt → FathomNet converter.

YOLO format: one .txt file per image, each line is:
    class_id center_x center_y width height
All coordinates are normalized (0-1) relative to image dimensions.
Requires image dimensions to convert to absolute pixels.
"""

from pathlib import Path
from typing import Generator, Optional
from PIL import Image

from .base import BaseConverter, FathomNetRecord
from utils.bbox import convert_bbox, BBoxFormat


class YOLOConverter(BaseConverter):
    format_name = "yolo"
    format_description = "YOLO txt format (class_id cx cy w h, normalized)"

    def convert(
        self,
        dataset_path: str,
        detection_result: dict,
        field_overrides: Optional[dict] = None,
    ) -> Generator[FathomNetRecord, None, None]:
        overrides = field_overrides or {}
        dataset_path = Path(dataset_path)

        # Load class names
        classes = self._load_classes(dataset_path, detection_result, overrides)

        # Find label files
        label_files = [
            Path(f) for f in detection_result.get("annotation_files", [])
            if f.endswith(".txt") and Path(f).stem.lower() not in ("classes", "names", "obj")
        ]

        # Find corresponding images
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

        for label_file in label_files:
            # Find the matching image
            image_path = None
            for ext in IMAGE_EXTS:
                candidate = label_file.with_suffix(ext)
                if candidate.exists():
                    image_path = candidate
                    break

                # Also check mirrored YOLO structure:
                # labels/train/foo.txt -> images/train/foo.jpg
                parts = label_file.parts
                if "labels" in parts:
                    idx = parts.index("labels")
                    candidate = Path(*parts[:idx], "images", *parts[idx + 1:]).with_suffix(ext)
                    if candidate.exists():
                        image_path = candidate
                        break

                # Also check ../images/ directory
                images_dir = label_file.parent.parent / "images"
                candidate = images_dir / (label_file.stem + ext)
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                self.record_drop(
                    item_type="image",
                    reason="No matching image found for YOLO label file",
                    source=str(label_file),
                    image=label_file.stem,
                )
                continue

            # Get image dimensions
            try:
                with Image.open(image_path) as img:
                    img_w, img_h = img.size
            except (OSError, Image.UnidentifiedImageError) as e:
                self.record_drop(
                    item_type="image",
                    reason="YOLO image could not be opened",
                    source=str(label_file),
                    image=image_path.name,
                    source_image_path=str(image_path),
                    detail=str(e),
                )
                continue

            # Parse label file
            try:
                with open(label_file, "r") as f:
                    lines = f.readlines()
            except OSError as e:
                self.record_drop(
                    item_type="annotation_file",
                    reason="YOLO label file could not be read",
                    source=str(label_file),
                    image=image_path.name,
                    source_image_path=str(image_path),
                    detail=str(e),
                )
                continue

            for line_number, line in enumerate(lines, start=1):
                parts = line.strip().split()
                if len(parts) < 5:
                    if line.strip():
                        self.record_drop(
                            item_type="annotation",
                            reason="Malformed YOLO annotation line",
                            source=f"{label_file}:{line_number}",
                            image=image_path.name,
                            source_image_path=str(image_path),
                            detail=line.strip(),
                        )
                    continue

                try:
                    class_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    self.record_drop(
                        item_type="annotation",
                        reason="Invalid YOLO annotation values",
                        source=f"{label_file}:{line_number}",
                        image=image_path.name,
                        source_image_path=str(image_path),
                        detail=line.strip(),
                    )
                    continue

                # Convert normalized center coords to absolute x, y, w, h
                x, y, bw, bh = convert_bbox(
                    (cx, cy, w, h),
                    from_format=BBoxFormat.CXCYWH_NORM,
                    to_format=BBoxFormat.XYWH,
                    image_width=img_w,
                    image_height=img_h,
                )

                concept = classes.get(class_id, f"class_{class_id}")

                record = FathomNetRecord(
                    concept=concept,
                    image=image_path.name,
                    x=x,
                    y=y,
                    width=bw,
                    height=bh,
                    source_image_path=str(image_path),
                )
                context = {
                    "annotation": {
                        "class_id": class_id,
                        "center_x": cx,
                        "center_y": cy,
                        "width": w,
                        "height": h,
                        "bbox": [cx, cy, w, h],
                        "line": line.strip(),
                        "line_number": line_number,
                    },
                    "class": {
                        "id": class_id,
                        "name": concept,
                    },
                    "image": {
                        "filename": image_path.name,
                        "path": str(image_path),
                        "width": img_w,
                        "height": img_h,
                    },
                    "label": {
                        "file": str(label_file),
                        "name": label_file.name,
                        "stem": label_file.stem,
                    },
                }
                self._apply_optional_overrides_from_context(record, overrides, context)
                self._apply_extra_columns_from_context(record, overrides, context)

                yield record

    def _load_classes(
        self, dataset_path: Path, detection_result: dict, overrides: dict
    ) -> dict:
        """Load class ID → name mapping from classes.txt or similar."""
        classes = {}

        # Check for user-provided class file
        class_file = overrides.get("classes_file")
        if class_file:
            class_file = Path(class_file)
        else:
            # Auto-detect
            class_file_str = detection_result.get("classes_file")
            if class_file_str:
                class_file = Path(class_file_str)
            else:
                # Search common names
                for name in ("classes.txt", "names.txt", "obj.names"):
                    candidate = dataset_path / name
                    if candidate.exists():
                        class_file = candidate
                        break

        if class_file and class_file.exists():
            try:
                with open(class_file, "r") as f:
                    for i, line in enumerate(f):
                        name = line.strip()
                        if name:
                            classes[i] = name
            except OSError:
                pass

        return classes

    def get_field_names(self, dataset_path: str, detection_result: dict) -> list[str]:
        return sorted(self.get_field_samples(dataset_path, detection_result).keys())

    def get_field_samples(
        self, dataset_path: str, detection_result: dict
    ) -> dict[str, list[str]]:
        fields = set()
        samples: dict[str, list[str]] = {}
        dataset_path = Path(dataset_path)
        classes = self._load_classes(dataset_path, detection_result, {})

        for class_id, class_name in list(classes.items())[:20]:
            self._add_field_sample(fields, samples, "class.id", class_id)
            self._add_field_sample(fields, samples, "class.name", class_name)

        label_files = [
            Path(f) for f in detection_result.get("annotation_files", [])
            if str(f).endswith(".txt") and Path(f).stem.lower() not in ("classes", "names", "obj")
        ]

        for label_file in label_files[:3]:
            self._add_field_sample(fields, samples, "label.file", str(label_file))
            self._add_field_sample(fields, samples, "label.name", label_file.name)
            self._add_field_sample(fields, samples, "label.stem", label_file.stem)
            try:
                with open(label_file, "r") as f:
                    lines = f.readlines()
            except OSError:
                continue

            for line_number, line in enumerate(lines[:20], start=1):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                self._add_field_sample(fields, samples, "annotation.line", line.strip())
                self._add_field_sample(fields, samples, "annotation.line_number", line_number)
                self._add_field_sample(fields, samples, "annotation.class_id", parts[0])
                self._add_field_sample(fields, samples, "annotation.center_x", parts[1])
                self._add_field_sample(fields, samples, "annotation.center_y", parts[2])
                self._add_field_sample(fields, samples, "annotation.width", parts[3])
                self._add_field_sample(fields, samples, "annotation.height", parts[4])
                self._add_field_sample(fields, samples, "annotation.bbox", parts[1:5])
                if parts[0].isdigit() and int(parts[0]) in classes:
                    self._add_field_sample(fields, samples, "class.name", classes[int(parts[0])])
                    self._add_field_sample(fields, samples, "class.id", parts[0])

        for field_name in (
            "annotation.bbox",
            "annotation.class_id",
            "annotation.center_x",
            "annotation.center_y",
            "annotation.width",
            "annotation.height",
            "class.id",
            "class.name",
            "image.filename",
            "image.path",
            "image.width",
            "image.height",
        ):
            self._add_field_sample(fields, samples, field_name, "")

        return {field: samples.get(field, []) for field in sorted(fields)}

    def get_default_field_map(
        self, dataset_path: str, detection_result: dict
    ) -> dict[str, str]:
        return {
            "concept": "class.name",
            "image": "image.filename",
            "x": "annotation.bbox",
            "y": "annotation.bbox",
            "width": "annotation.bbox",
            "height": "annotation.bbox",
        }

    def validate_prerequisites(
        self, dataset_path: str, detection_result: dict
    ) -> list[str]:
        errors = []
        class_file = detection_result.get("classes_file")
        if not class_file:
            errors.append(
                "No classes.txt file found. Class IDs will be used as concept names "
                "unless you add classes.txt, names.txt, or obj.names to the dataset."
            )
        return errors
