"""
COCO JSON → FathomNet converter.

Handles standard COCO format with images, annotations, and categories arrays.
Supports fuzzy field matching for non-standard COCO variants.
"""

import json
from pathlib import Path
from typing import Generator, Optional

from fathomnet_schema import OPTIONAL_FIELDS
from .base import BaseConverter, FathomNetRecord


class COCOConverter(BaseConverter):
    format_name = "coco_json"
    format_description = "COCO JSON format (images + annotations + categories)"

    def convert(
        self,
        dataset_path: str,
        detection_result: dict,
        field_overrides: Optional[dict] = None,
    ) -> Generator[FathomNetRecord, None, None]:
        overrides = field_overrides or {}
        dataset_path = Path(dataset_path)
        dataset_root = dataset_path.parent if dataset_path.is_file() else dataset_path

        # Find COCO JSON files
        coco_files = detection_result.get("coco_files", [])
        if not coco_files:
            # Fallback: find any json files
            coco_files = [
                f for f in detection_result.get("annotation_files", [])
                if f.endswith(".json")
            ]

        for coco_file in coco_files:
            yield from self._convert_single_file(
                coco_file, dataset_root, overrides
            )

    def _convert_single_file(
        self,
        coco_file: str,
        dataset_path: Path,
        overrides: dict,
    ) -> Generator[FathomNetRecord, None, None]:
        with open(coco_file, "r") as f:
            data = json.load(f)

        # Build lookup tables
        # Category ID → name
        cat_key_name = overrides.get("category_name_field", "name")
        cat_key_id = overrides.get("category_id_field", "id")
        categories = {}
        category_records = {}
        for cat in data.get("categories", []):
            cat_id = cat.get(cat_key_id)
            cat_name = cat.get(cat_key_name, f"unknown_{cat_id}")
            categories[cat_id] = cat_name
            category_records[cat_id] = cat

        # License ID → license info. COCO stores the image's license as an ID
        # and the details in a separate top-level licenses array.
        license_records = {}
        for license_info in data.get("licenses", []):
            if isinstance(license_info, dict) and "id" in license_info:
                license_records[license_info["id"]] = license_info

        # Image ID → image info
        img_key_id = overrides.get("image_id_field", "id")
        img_key_fname = overrides.get("image_filename_field", "file_name")
        images = {}
        for img in data.get("images", []):
            img_id = img.get(img_key_id)
            img_record = dict(img)
            img_record["file_name"] = img.get(img_key_fname, img.get("filename", ""))
            images[img_id] = img_record

        # Determine which COCO file subdirectory images are in
        coco_dir = Path(coco_file).parent
        images_dir = coco_dir / "images"
        if not images_dir.exists():
            images_dir = coco_dir

        # Process annotations
        ann_key_bbox = overrides.get("bbox_field", "bbox")
        ann_key_cat = overrides.get("annotation_category_field", "category_id")
        ann_key_img = overrides.get("annotation_image_field", "image_id")

        # Classes to exclude (optional)
        exclude_concepts = set(overrides.get("exclude_concepts", []))

        for ann in data.get("annotations", []):
            cat_id = ann.get(ann_key_cat)
            img_id = ann.get(ann_key_img)

            concept = categories.get(cat_id, f"unknown_{cat_id}")
            img_info = images.get(img_id, {})
            cat_info = category_records.get(cat_id, {})
            license_info = license_records.get(img_info.get("license"), {})
            file_name = img_info.get("file_name", f"unknown_{img_id}")
            source_image_path = self._resolve_image_path(
                dataset_path, coco_file, file_name
            )

            if concept in exclude_concepts:
                self.record_drop(
                    item_type="annotation",
                    reason=f"Excluded concept: {concept}",
                    source=f"{coco_file}: annotation {ann.get('id', '?')}",
                    image=file_name,
                    source_image_path=source_image_path,
                    concept=concept,
                )
                continue

            bbox = ann.get(ann_key_bbox)
            try:
                invalid_bbox = not bbox or len(bbox) < 4
            except TypeError:
                invalid_bbox = True
            if invalid_bbox:
                self.record_drop(
                    item_type="annotation",
                    reason="Missing or invalid COCO bbox",
                    source=f"{coco_file}: annotation {ann.get('id', '?')}",
                    image=file_name,
                    source_image_path=source_image_path,
                    concept=concept,
                    detail=str(bbox),
                )
                continue

            # COCO bbox is [x, y, width, height] — same as FathomNet
            try:
                x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            except (TypeError, ValueError):
                self.record_drop(
                    item_type="annotation",
                    reason="Invalid COCO bbox values",
                    source=f"{coco_file}: annotation {ann.get('id', '?')}",
                    image=file_name,
                    source_image_path=source_image_path,
                    concept=concept,
                    detail=str(bbox),
                )
                continue

            record = FathomNetRecord(
                concept=concept,
                image=file_name,
                x=round(x),
                y=round(y),
                width=round(w),
                height=round(h),
                source_image_path=source_image_path,
            )

            # Carry over any useful extra fields
            if "score" in ann:
                record.extra["score"] = ann["score"]
            if ann.get("iscrowd"):
                record.groupof = bool(ann["iscrowd"])
            self._apply_optional_overrides(
                record=record,
                overrides=overrides,
                annotation=ann,
                image=img_info,
                category=cat_info,
                license_info=license_info,
                root=data,
            )
            self._apply_extra_columns(
                record=record,
                overrides=overrides,
                annotation=ann,
                image=img_info,
                category=cat_info,
                license_info=license_info,
                root=data,
            )

            yield record

    def _apply_optional_overrides(
        self,
        *,
        record: FathomNetRecord,
        overrides: dict,
        annotation: dict,
        image: dict,
        category: dict,
        license_info: dict,
        root: dict,
    ):
        """Apply FathomNet optional field mappings from COCO structures."""
        for field in OPTIONAL_FIELDS:
            source = overrides.get(field)
            if not source:
                continue
            value = self._lookup_coco_value(
                str(source),
                annotation=annotation,
                image=image,
                category=category,
                license_info=license_info,
                root=root,
            )
            value = self._coerce_optional_value(field, value)
            if value is None or value == "":
                continue
            if hasattr(record, field):
                setattr(record, field, value)
            else:
                record.extra[field] = value

    def _coerce_optional_value(self, field: str, value):
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value == "" or value.lower() in {"na", "n/a", "none", "null"}:
                return None

        if field in {
            "depth",
            "altitude",
            "latitude",
            "longitude",
            "temperature",
            "salinity",
            "oxygen",
            "pressure",
        }:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        if field in {"occluded", "truncated", "groupof"}:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value != 0
            if isinstance(value, str):
                return value.lower() in {"1", "true", "yes", "y"}
            return None

        return str(value)

    def _apply_extra_columns(
        self,
        *,
        record: FathomNetRecord,
        overrides: dict,
        annotation: dict,
        image: dict,
        category: dict,
        license_info: dict,
        root: dict,
    ):
        """Apply user-selected source fields as extra CSV columns."""
        for item in overrides.get("extra_columns", []):
            if isinstance(item, dict):
                source = item.get("source")
                column = item.get("column") or source
            else:
                source = item
                column = item
            if not source or not column:
                continue

            value = self._lookup_coco_value(
                str(source),
                annotation=annotation,
                image=image,
                category=category,
                license_info=license_info,
                root=root,
            )
            value = self._coerce_extra_value(value)
            # A user-selected extra column should exist even when this dataset's
            # value is blank/null for every row. Otherwise the engine never sees
            # the key and the CSV column silently disappears.
            record.extra[str(column)] = "" if value is None else value

    def _coerce_extra_value(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value == "" or value.lower() in {"none", "null"}:
                return None
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return value

    def _lookup_coco_value(
        self,
        source: str,
        *,
        annotation: dict,
        image: dict,
        category: dict,
        license_info: dict,
        root: dict,
    ):
        """Look up a mapped source path such as images.uuid or annotations.id."""
        source = source.strip()
        if not source:
            return None

        containers = {
            "annotation": annotation,
            "annotations": annotation,
            "image": image,
            "images": image,
            "category": category,
            "categories": category,
            "license": license_info,
            "licenses": license_info,
            "info": root.get("info", {}),
        }

        if "." in source:
            prefix, remainder = source.split(".", 1)
            container = containers.get(prefix)
            if container is None:
                return None
            return self._lookup_nested_value(container, remainder)

        for container in (annotation, image, category, license_info, root.get("info", {})):
            value = self._lookup_nested_value(container, source)
            if value is not None:
                return value
        return None

    def _lookup_nested_value(self, data: dict, path: str):
        value = data
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    def _resolve_image_path(
        self, dataset_root: Path, coco_file: str, file_name: str
    ) -> Optional[str]:
        """Resolve a COCO image reference to an actual source file when possible."""
        image_ref = Path(file_name)
        if image_ref.is_absolute() and image_ref.exists():
            return str(image_ref)

        coco_dir = Path(coco_file).parent
        candidates = [
            coco_dir / image_ref,
            coco_dir / "images" / image_ref,
            dataset_root / image_ref,
            dataset_root / "images" / image_ref,
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def get_field_names(self, dataset_path: str, detection_result: dict) -> list[str]:
        """Extract all unique field names from the COCO JSON."""
        samples = self.get_field_samples(dataset_path, detection_result)
        top_level = set()
        for coco_file in self._get_coco_files(detection_result)[:3]:
            try:
                with open(coco_file, "r") as f:
                    data = json.load(f)
                top_level.update(data.keys())
            except (json.JSONDecodeError, OSError):
                continue
        return sorted(top_level | set(samples.keys()))

    def get_field_samples(
        self, dataset_path: str, detection_result: dict
    ) -> dict[str, list[str]]:
        """Extract detected COCO field paths and representative values."""
        fields = set()
        samples: dict[str, list[str]] = {}
        coco_files = self._get_coco_files(detection_result)
        for coco_file in coco_files[:3]:
            try:
                with open(coco_file, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            self._collect_field_samples(
                "info", data.get("info", {}), fields, samples
            )
            for license_info in data.get("licenses", [])[:20]:
                self._collect_field_samples(
                    "licenses", license_info, fields, samples
                )
            for cat in data.get("categories", [])[:50]:
                self._collect_field_samples(
                    "categories", cat, fields, samples
                )
            for img in data.get("images", [])[:100]:
                self._collect_field_samples(
                    "images", img, fields, samples
                )
            for ann in data.get("annotations", [])[:100]:
                self._collect_field_samples(
                    "annotations", ann, fields, samples
                )

        return {field: samples.get(field, []) for field in sorted(fields)}

    def get_default_field_map(
        self, dataset_path: str, detection_result: dict
    ) -> dict[str, str]:
        return {
            "concept": "categories.name",
            "image": "images.file_name",
            "x": "annotations.bbox",
            "y": "annotations.bbox",
            "width": "annotations.bbox",
            "height": "annotations.bbox",
        }

    def _get_coco_files(self, detection_result: dict) -> list[str]:
        coco_files = detection_result.get("coco_files", [])
        if not coco_files:
            coco_files = [
                f for f in detection_result.get("annotation_files", [])
                if str(f).endswith(".json")
            ]
        return [str(f) for f in coco_files]

    def _collect_field_samples(
        self,
        prefix: str,
        value,
        fields: set[str],
        samples: dict[str, list[str]],
        *,
        max_values: int = 4,
    ):
        if not isinstance(value, dict):
            return

        for key, child in value.items():
            field_name = f"{prefix}.{key}"
            fields.add(field_name)
            sample = self._sample_value(child)
            if sample is None:
                continue
            field_samples = samples.setdefault(field_name, [])
            if sample not in field_samples and len(field_samples) < max_values:
                field_samples.append(sample)

            if isinstance(child, dict):
                self._collect_field_samples(
                    field_name,
                    child,
                    fields,
                    samples,
                    max_values=max_values,
                )

    def _sample_value(self, value) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)[:120]
        return str(value)

    def validate_prerequisites(
        self, dataset_path: str, detection_result: dict
    ) -> list[str]:
        errors = []
        coco_files = detection_result.get("coco_files", [])
        if not coco_files:
            errors.append("No COCO JSON file found.")
            return errors

        for coco_file in coco_files:
            try:
                with open(coco_file, "r") as f:
                    data = json.load(f)
                if "categories" not in data:
                    errors.append(f"{coco_file}: missing 'categories' array.")
                if "images" not in data:
                    errors.append(f"{coco_file}: missing 'images' array.")
                if "annotations" not in data:
                    errors.append(f"{coco_file}: missing 'annotations' array.")
            except json.JSONDecodeError as e:
                errors.append(f"{coco_file}: invalid JSON — {e}")
            except OSError as e:
                errors.append(f"{coco_file}: cannot read — {e}")

        return errors
