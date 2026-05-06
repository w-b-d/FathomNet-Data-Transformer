"""
Transformation engine.

Takes a converter + mapping config and produces the final FathomNet CSV.
This is the shared pipeline that all three modes converge into.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

from converters.base import BaseConverter, FathomNetRecord
from mapper.mapping_config import MappingConfig
from fathomnet_schema import NOAA_NCEI_IMAGE_EXTS, REQUIRED_FIELDS


class TransformEngine:
    """Orchestrates the conversion from any format to FathomNet CSV."""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
        self.errors: list[dict] = []
        self.dropped_items: list[dict] = []
        self.submission_target = "noaa-ncei"
        self.image_conversion = "none"
        self.stats = {
            "total_processed": 0,
            "total_written": 0,
            "total_skipped": 0,
            "total_errors": 0,
            "total_dropped": 0,
            "dropped_by_reason": {},
            "images_converted": 0,
            "concepts": {},
        }

    def run(
        self,
        converter: BaseConverter,
        dataset_path: str,
        detection_result: dict,
        config: Optional[MappingConfig] = None,
        image_dirs: Optional[list[str]] = None,
        submission_target: str = "noaa-ncei",
        image_conversion: str = "none",
    ) -> dict:
        """
        Execute the full transformation pipeline.

        Args:
            converter: format-specific converter instance
            dataset_path: path to the dataset
            detection_result: output from detect_format()
            config: mapping configuration

        Returns:
            dict with stats, output path, and any errors
        """
        config = config or MappingConfig()
        self.submission_target = submission_target
        self.image_conversion = image_conversion
        self.records = []
        self.errors = []
        self.dropped_items = []
        self.stats = {
            "total_processed": 0,
            "total_written": 0,
            "total_skipped": 0,
            "total_errors": 0,
            "total_dropped": 0,
            "dropped_by_reason": {},
            "images_converted": 0,
            "concepts": {},
        }
        if hasattr(converter, "reset_drop_report"):
            converter.reset_drop_report()

        # Build field overrides from config for the converter
        field_overrides = {}
        if config.field_map:
            for fathomnet_field, mapping in config.field_map.items():
                if isinstance(mapping, dict) and "source" in mapping:
                    field_overrides[fathomnet_field] = mapping["source"]
                elif isinstance(mapping, str):
                    field_overrides[fathomnet_field] = mapping
        field_overrides.update(
            self._converter_specific_overrides(converter.format_name, field_overrides)
        )
        if config.exclude_concepts:
            field_overrides["exclude_concepts"] = config.exclude_concepts
        if config.extra.get("images_are_crops"):
            field_overrides["images_are_crops"] = True
        if config.extra.get("extra_columns"):
            field_overrides["extra_columns"] = config.extra["extra_columns"]
        if config.extra.get("bbox_pattern"):
            field_overrides["bbox_pattern"] = config.extra["bbox_pattern"]
        if config.extra.get("bbox_groups"):
            field_overrides["bbox_groups"] = config.extra["bbox_groups"]

        # Tell FishCLEF (and any video converter) to save extracted frames
        # directly into the final output dir, so images + CSV end up co-located.
        field_overrides.setdefault("frame_output_dir", str(self.output_dir))

        # AI mode produces field_map entries like
        # "x": {"source": "filename_regex:...", "transform": "capture_group_2"}.
        # Parse these once up front; they get applied per-record below.
        filename_regex_overrides, filename_regex_errors = (
            self._parse_filename_regex_overrides(config)
        )
        if filename_regex_errors:
            for regex_error in filename_regex_errors:
                self.errors.append({
                    "record": None,
                    "error": regex_error,
                })
                self.stats["total_errors"] += 1
                self._record_dropped_item(
                    item_type="config",
                    stage="filename_regex",
                    reason=regex_error,
                )
            output_path = self.output_dir / "metadata.csv"
            self.stats["output_path"] = str(output_path)
            self.stats["unique_concepts"] = 0
            return {
                "stats": self.stats,
                "errors": self.errors[:20],
                "dropped_items": self.dropped_items[:200],
                "output_path": str(output_path),
            }

        # Run conversion
        for record in converter.convert(dataset_path, detection_result, field_overrides):
            self.stats["total_processed"] += 1

            # Overwrite fields with values extracted from the filename via regex
            # (only if the AI mapping requested filename_regex extraction).
            if filename_regex_overrides:
                regex_error = self._apply_filename_regex_overrides(
                    record, filename_regex_overrides
                )
                if regex_error:
                    self.errors.append({
                        "record": record.to_dict(),
                        "error": regex_error,
                    })
                    self.stats["total_errors"] += 1
                    self._record_dropped_item(
                        item_type="annotation",
                        stage="filename_regex",
                        reason=regex_error,
                        record=record,
                    )
                    continue

            # Apply concept aliases
            if config.concept_aliases and record.concept in config.concept_aliases:
                record.concept = config.concept_aliases[record.concept]

            # Apply coordinate offsets
            if config.x_offset:
                record.x += config.x_offset
            if config.y_offset:
                record.y += config.y_offset

            # Clamp negative coordinates to 0 (box extends outside frame)
            if record.x < 0:
                record.width = max(0, record.width + record.x)  # shrink width
                record.x = 0
            if record.y < 0:
                record.height = max(0, record.height + record.y)  # shrink height
                record.y = 0

            # Skip excluded concepts
            if record.concept in config.exclude_concepts:
                self.stats["total_skipped"] += 1
                self._record_dropped_item(
                    item_type="annotation",
                    stage="transform_filter",
                    reason=f"Excluded concept: {record.concept}",
                    record=record,
                )
                continue

            # Basic validation
            error = self._validate_record(record)
            if error:
                self.errors.append({
                    "record": record.to_dict(),
                    "error": error,
                })
                self.stats["total_errors"] += 1
                self._record_dropped_item(
                    item_type="annotation",
                    stage="transform_validation",
                    reason=error,
                    record=record,
                )
                continue

            # Track stats
            self.stats["concepts"][record.concept] = (
                self.stats["concepts"].get(record.concept, 0) + 1
            )

            row = record.to_dict()
            if getattr(record, "source_image_path", None):
                row["_source_image_path"] = record.source_image_path
            if getattr(record, "_source_image_lookup", None):
                row["_source_image_lookup"] = record._source_image_lookup
            self.records.append(row)
            self.stats["total_written"] += 1

        # Copy referenced images into the output folder, handling filename
        # collisions by renaming. This may rewrite record["image"] entries,
        # so it must happen BEFORE the CSV is written.
        self._ingest_converter_drops(getattr(converter, "dropped_items", []))

        copied, url_count = self._copy_images(dataset_path, image_dirs or [])
        self.stats["images_copied"] = copied
        self.stats["url_records"] = url_count
        self._refresh_written_stats()

        # Write CSV
        output_path = self.output_dir / "metadata.csv"
        self._write_csv(output_path)

        self.stats["output_path"] = str(output_path)
        self.stats["unique_concepts"] = len(self.stats["concepts"])

        return {
            "stats": self.stats,
            "errors": self.errors[:20],  # first 20 errors
            "dropped_items": self.dropped_items[:200],
            "output_path": str(output_path),
        }

    def _validate_record(self, record: FathomNetRecord) -> Optional[str]:
        """Basic validation of a single record."""
        if not record.concept or record.concept.strip() == "":
            return "Empty concept"
        if not record.image or record.image.strip() == "":
            return "Empty image filename"
        if record.width <= 0 or record.height <= 0:
            return f"Invalid bbox dimensions: {record.width}x{record.height}"
        if record.x < 0 or record.y < 0:
            return f"Negative coordinates: ({record.x}, {record.y})"
        is_url = record.image.startswith(("http://", "https://"))
        if self.submission_target == "noaa-ncei":
            if is_url:
                return "NOAA-NCEI submissions require image filenames, not URLs"
            # NOAA-NCEI accepts .jpg only, not .jpeg (same format, different ext).
            # Rename so the copy step writes .jpg (lossless rename, no re-encode).
            # We notify the user the first time it happens and tally the total.
            if Path(record.image).suffix.lower() == ".jpeg":
                if self.stats.get("jpeg_renamed", 0) == 0:
                    print(
                        "  Note: .jpeg files will be renamed to .jpg "
                        "(NOAA-NCEI accepts .jpg only — this is a lossless rename, "
                        "the image bytes are unchanged)."
                    )
                self.stats["jpeg_renamed"] = self.stats.get("jpeg_renamed", 0) + 1
                record._source_image_lookup = record.image
                record.image = Path(record.image).with_suffix(".jpg").name
            if (
                self.image_conversion == "none"
                and Path(record.image).suffix.lower() not in NOAA_NCEI_IMAGE_EXTS
            ):
                return "NOAA-NCEI submissions require .jpg or .png images"
        return None

    def _parse_filename_regex_overrides(self, config) -> tuple[dict, list[str]]:
        """
        Build {field: (compiled_regex, capture_group_int)} from any field_map
        entries whose source starts with "filename_regex:". AI mode emits these.

        The capture group is parsed out of the transform string ("capture_group_2",
        "group_3", etc.); defaults to 1 if no number is found.
        """
        import re
        result = {}
        errors = []
        if not config.field_map:
            return result, errors
        for field, mapping in config.field_map.items():
            if not isinstance(mapping, dict):
                continue
            source = mapping.get("source", "")
            if not isinstance(source, str) or not source.startswith("filename_regex:"):
                continue
            pattern_str = source[len("filename_regex:"):]
            try:
                pattern = re.compile(pattern_str)
            except re.error as e:
                errors.append(
                    f"invalid filename_regex pattern for field '{field}': {e}"
                )
                continue
            transform = str(mapping.get("transform") or "").strip()
            if transform:
                num_match = re.search(
                    r"(?:capture[_ -]?)?group[_ -]?(\d+)",
                    transform,
                    re.IGNORECASE,
                )
                if not num_match:
                    errors.append(
                        f"unsupported filename_regex transform for field "
                        f"'{field}': {transform}"
                    )
                    continue
                group_num = int(num_match.group(1))
            else:
                group_num = 1
            if group_num < 1:
                errors.append(
                    f"capture group must be 1 or greater for field '{field}'"
                )
                continue
            if group_num > pattern.groups:
                errors.append(
                    f"capture group {group_num} out of range for field "
                    f"'{field}' (pattern has {pattern.groups})"
                )
                continue
            result[field] = (pattern, group_num)
        return result, errors

    def _apply_filename_regex_overrides(
        self, record: FathomNetRecord, overrides: dict
    ) -> Optional[str]:
        """
        Run each filename_regex extractor against record.image and overwrite
        the matching field. Returns an error string if extraction fails for
        any required-numeric field, otherwise None.

        Special case: the `image` field always has to point to the real file
        the engine will copy. If the regex for `image` does not match, we
        keep the existing filename and log a one-time notice — dropping the
        record over a failed image-name extraction would lose valid data.
        """
        filename = record.image or ""
        if not filename and record.source_image_path:
            filename = Path(record.source_image_path).name
        if not filename:
            return "no filename available for filename_regex extraction"

        for field, (pattern, group_num) in overrides.items():
            match = pattern.search(filename)

            # `image` field is forgiving — keep the real filename if the AI's
            # regex doesn't match. Notify the user once so it's transparent.
            if field == "image":
                if not match:
                    if not self.stats.get("image_regex_fallback_warned"):
                        print(
                            "  Note: AI's filename_regex for 'image' did not "
                            "match the actual filenames; keeping the source "
                            "filenames as-is so the images can still be copied."
                        )
                        self.stats["image_regex_fallback_warned"] = True
                    self.stats["image_regex_fallback"] = (
                        self.stats.get("image_regex_fallback", 0) + 1
                    )
                    continue
                try:
                    record.image = str(match.group(group_num))
                except IndexError:
                    return f"capture group {group_num} out of range for field 'image'"
                continue

            if not match:
                return f"filename_regex did not match '{filename}' for field '{field}'"
            try:
                value = match.group(group_num)
            except IndexError:
                return f"capture group {group_num} out of range for field '{field}'"

            if field in ("x", "y", "width", "height"):
                try:
                    setattr(record, field, int(value))
                except (TypeError, ValueError):
                    return f"could not parse '{value}' as int for field '{field}'"
            elif field == "concept":
                record.concept = str(value)
            else:
                # Optional field — best-effort assignment to the record
                if hasattr(record, field):
                    setattr(record, field, value)
                else:
                    record.extra[field] = value
        return None

    def _converter_specific_overrides(
        self, converter_format: str, generic_overrides: dict
    ) -> dict:
        """Translate generic FathomNet field mappings to converter knobs."""
        def clean(source: str, *prefixes: str) -> str:
            value = str(source).strip()
            for prefix in prefixes:
                if value.startswith(prefix):
                    value = value[len(prefix):]
            if "[" in value:
                value = value.split("[", 1)[0]
            return value

        translated = {}

        if converter_format == "coco_json":
            if "concept" in generic_overrides:
                translated["category_name_field"] = clean(
                    generic_overrides["concept"], "categories."
                )
            if "image" in generic_overrides:
                translated["image_filename_field"] = clean(
                    generic_overrides["image"], "images."
                )
            for target in ("x", "y", "width", "height"):
                source = generic_overrides.get(target)
                if source and "bbox" in str(source):
                    translated["bbox_field"] = clean(source, "annotations.")

        elif converter_format == "pascal_voc":
            if "concept" in generic_overrides:
                translated["concept_field"] = clean(
                    generic_overrides["concept"], "object."
                )
            if "image" in generic_overrides:
                translated["image_field"] = clean(
                    generic_overrides["image"], "annotation.", "image."
                )

        elif converter_format == "fishclef_xml":
            if "concept" in generic_overrides:
                translated["species_attributes"] = clean(
                    generic_overrides["concept"], "object."
                )
            if "x" in generic_overrides:
                translated["x_attr"] = clean(generic_overrides["x"], "object.")
            if "y" in generic_overrides:
                translated["y_attr"] = clean(generic_overrides["y"], "object.")
            if "width" in generic_overrides:
                translated["w_attr"] = clean(generic_overrides["width"], "object.")
            if "height" in generic_overrides:
                translated["h_attr"] = clean(generic_overrides["height"], "object.")

        elif converter_format == "folder_encoded":
            if "concept" in generic_overrides:
                translated["concept_source"] = clean(
                    generic_overrides["concept"], "folder."
                )

        return translated

    def _write_csv(self, output_path: Path):
        """Write all records to a FathomNet-compatible CSV."""
        for record in self.records:
            record.pop("_source_image_path", None)
            record.pop("_source_image_lookup", None)

        if not self.records:
            return

        # Determine all columns present (required + any extras)
        all_columns = list(REQUIRED_FIELDS.keys())
        extra_columns = set()
        for record in self.records:
            for key in record:
                if key not in all_columns:
                    extra_columns.add(key)
        all_columns.extend(sorted(extra_columns))

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
            writer.writeheader()
            for record in self.records:
                writer.writerow(record)

    def _record_dropped_item(
        self,
        *,
        item_type: str,
        stage: str,
        reason: str,
        source: Optional[str] = None,
        image: Optional[str] = None,
        source_image_path: Optional[str] = None,
        concept: Optional[str] = None,
        record: Optional[FathomNetRecord | dict] = None,
        detail: Optional[str] = None,
    ):
        """Track records or images that are not included in the final output."""
        item = {
            "type": item_type,
            "stage": stage,
            "reason": reason,
        }
        if source:
            item["source"] = str(source)
        if image:
            item["image"] = str(image)
        if source_image_path:
            item["source_image_path"] = str(source_image_path)
        if concept:
            item["concept"] = str(concept)
        if record is not None:
            if isinstance(record, FathomNetRecord):
                item["record"] = record.to_dict()
                if record.source_image_path:
                    item["source_image_path"] = record.source_image_path
                if record.image:
                    item.setdefault("image", record.image)
                if record.concept:
                    item.setdefault("concept", record.concept)
            else:
                record_dict = dict(record)
                record_dict.pop("_source_image_lookup", None)
                item["record"] = record_dict
                if record.get("image"):
                    item.setdefault("image", record["image"])
                if record.get("concept"):
                    item.setdefault("concept", record["concept"])
                if record.get("_source_image_path"):
                    item.setdefault("source_image_path", record["_source_image_path"])
        if detail:
            item["detail"] = str(detail)

        self.dropped_items.append(item)
        self.stats["total_dropped"] = self.stats.get("total_dropped", 0) + 1
        by_reason = self.stats.setdefault("dropped_by_reason", {})
        by_reason[reason] = by_reason.get(reason, 0) + 1

    def _ingest_converter_drops(self, dropped_items: list[dict]):
        """Merge converter-local drop reports into the engine report."""
        for item in dropped_items:
            reason = item.get("reason", "Dropped by converter")
            self.dropped_items.append(dict(item))
            self.stats["total_dropped"] = self.stats.get("total_dropped", 0) + 1
            by_reason = self.stats.setdefault("dropped_by_reason", {})
            by_reason[reason] = by_reason.get(reason, 0) + 1

    def _refresh_written_stats(self):
        """Rebuild write stats after image handling may have removed rows."""
        concepts = {}
        for record in self.records:
            concept = record.get("concept")
            if concept:
                concepts[concept] = concepts.get(concept, 0) + 1
        self.stats["concepts"] = concepts
        self.stats["total_written"] = len(self.records)
        self.stats["unique_concepts"] = len(concepts)

    def _copy_images(
        self, dataset_path: str, image_dirs: list[str]
    ) -> tuple[int, int]:
        """
        Copy every image referenced in the CSV into the output folder.

        Handles three real-world cases:
          • URL records (http/https) — left untouched, no copy attempted
          • Filename collisions across source folders — second copy renamed
            to img_1.jpg, img_2.jpg, etc., and the CSV record is rewritten
          • Images outside dataset_path — additional image_dirs are also
            walked (e.g. sibling /images folder when user points at /annotations)

        Returns:
            (copied_count, url_count)
        """
        if not self.records:
            return (0, 0)

        # Build {filename_or_relative_path: [source_paths]} index — list, not single path,
        # so we can detect collisions across folders.
        search_roots = [dataset_path] + [d for d in image_dirs if d != dataset_path]
        index: dict[str, list[Path]] = {}
        seen_paths: set[Path] = set()
        for root in search_roots:
            root_path = Path(root)
            if root_path.is_file():
                root_path = root_path.parent
            try:
                for path in root_path.rglob("*"):
                    if not path.is_file():
                        continue
                    if path.suffix.lower() not in IMAGE_EXTS:
                        continue
                    resolved = path.resolve()
                    if resolved in seen_paths:
                        continue  # same file reachable from two roots
                    seen_paths.add(resolved)
                    index.setdefault(path.name, []).append(path)
                    try:
                        rel = str(path.relative_to(root_path))
                        index.setdefault(rel, []).append(path)
                    except ValueError:
                        pass
            except OSError:
                continue

        copied = 0
        url_count = 0
        converted = 0
        failed_record_ids: set[int] = set()
        # Map source_path_str → final filename in output
        # so identical references in the CSV map to the same copied file.
        rename_map: dict[str, str] = {}
        # Only track names we ACTIVELY assign during this run. We must NOT
        # seed this with the current contents of output_dir — files left over
        # from a previous run would otherwise look like collisions and force
        # every record to be renamed (e.g. img.jpg → img_1.jpg) on every
        # re-run, breaking preview lookups.
        used_names: set[str] = set()
        missing_image_reported: set[str] = set()

        for record in self.records:
            fname = record.get("image")
            if not fname:
                continue

            # Case 1: URL — leave it alone
            if fname.startswith(("http://", "https://")):
                url_count += 1
                continue

            source_hint = record.get("_source_image_path")
            source_lookup = record.get("_source_image_lookup")
            candidates = []
            if source_hint:
                hinted = Path(source_hint)
                if hinted.exists():
                    candidates = [hinted]
            if not candidates and source_lookup:
                candidates = index.get(source_lookup, [])
            if (
                not candidates
                and source_lookup
                and Path(source_lookup).name != source_lookup
            ):
                candidates = index.get(Path(source_lookup).name, [])
            if not candidates:
                candidates = index.get(fname, [])
            if not candidates and Path(fname).name != fname:
                candidates = index.get(Path(fname).name, [])

            source_from_output = False
            # Case 2: no source found, but file already in output (FishCLEF frame)
            if not candidates:
                existing = self.output_dir / Path(fname).name
                if existing.exists():
                    candidates = [existing]
                    source_from_output = True
                else:
                    if fname not in missing_image_reported:
                        self._record_dropped_item(
                            item_type="image",
                            stage="image_copy",
                            reason="Referenced image could not be found or copied; metadata row was kept",
                            image=fname,
                            record=record,
                        )
                        missing_image_reported.add(fname)
                    continue

            # Pick the source path. If multiple sources share this filename,
            # we map each (filename, source_path) pair to its own output name.
            src = candidates[0]
            convert_image = self._should_convert_image(fname)
            key = f"{src.resolve()}::{self.image_conversion if convert_image else 'copy'}"
            if key in rename_map:
                record["image"] = rename_map[key]
                continue  # already copied this exact source

            # Pick a non-colliding output name
            target = self._target_image_name(fname, src, convert_image)
            if target in used_names:
                stem, suffix = Path(target).stem, Path(target).suffix
                i = 1
                while f"{stem}_{i}{suffix}" in used_names:
                    i += 1
                target = f"{stem}_{i}{suffix}"

            dst = self.output_dir / target
            if convert_image:
                if src.resolve() != dst.resolve() or not dst.exists():
                    success, detail = self._convert_image(src, dst, self.image_conversion)
                    if not success:
                        failed_record_ids.add(id(record))
                        self._record_dropped_item(
                            item_type="annotation",
                            stage="image_conversion",
                            reason="Image conversion failed; metadata row was removed",
                            image=fname,
                            source_image_path=str(src),
                            record=record,
                            detail=detail,
                        )
                        continue
                    converted += 1
            elif src.resolve() != dst.resolve() and not dst.exists():
                try:
                    shutil.copy2(src, dst)
                    copied += 1
                except OSError:
                    self._record_dropped_item(
                        item_type="image",
                        stage="image_copy",
                        reason="Referenced image could not be copied; metadata row was kept",
                        image=fname,
                        source_image_path=str(src),
                        record=record,
                    )
                    continue
            elif source_from_output:
                used_names.add(target)
            used_names.add(target)
            rename_map[key] = target
            record["image"] = target

        if failed_record_ids:
            self.records = [
                record for record in self.records
                if id(record) not in failed_record_ids
            ]

        self.stats["images_converted"] = converted

        for record in self.records:
            record.pop("_source_image_path", None)
            record.pop("_source_image_lookup", None)

        return (copied, url_count)

    def _should_convert_image(self, filename: str) -> bool:
        return (
            self.submission_target == "noaa-ncei"
            and self.image_conversion in {"png", "jpg"}
            and Path(filename).suffix.lower() not in NOAA_NCEI_IMAGE_EXTS
        )

    def _target_image_name(self, filename: str, source_path: Path, convert_image: bool) -> str:
        if not convert_image:
            return Path(filename).name
        stem = Path(filename).stem or source_path.stem
        return f"{stem}.{self.image_conversion}"

    def _convert_image(
        self,
        source_path: Path,
        target_path: Path,
        target_format: str,
    ) -> tuple[bool, str]:
        try:
            from PIL import Image
        except ImportError:
            return (False, "Pillow is required for image conversion.")

        try:
            with Image.open(source_path) as img:
                original_size = img.size
                if target_format == "jpg":
                    img = self._prepare_jpeg_image(img)
                    img.save(target_path, "JPEG", quality=95)
                else:
                    img.save(target_path, "PNG")

            with Image.open(target_path) as converted_img:
                if converted_img.size != original_size:
                    try:
                        target_path.unlink()
                    except OSError:
                        pass
                    return (
                        False,
                        f"Converted dimensions changed from {original_size} to {converted_img.size}.",
                    )
        except Exception as e:
            try:
                if target_path.exists():
                    target_path.unlink()
            except OSError:
                pass
            return (False, str(e))

        return (True, "")

    def _prepare_jpeg_image(self, img):
        from PIL import Image

        if img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        ):
            background = Image.new("RGB", img.size, (255, 255, 255))
            alpha = img.convert("RGBA").split()[-1]
            background.paste(img.convert("RGBA"), mask=alpha)
            return background
        if img.mode != "RGB":
            return img.convert("RGB")
        return img

    def get_sample_records(self, n: int = 6, strategy: str = "smart") -> list[dict]:
        """
        Get a sample of records for preview.

        Args:
            n: number of samples
            strategy: "random", "smart" (diverse sampling), or "first"

        Returns:
            list of record dicts
        """
        if not self.records:
            return []

        if strategy == "first":
            return self.records[:n]

        if strategy == "random":
            import random
            return random.sample(self.records, min(n, len(self.records)))

        # Smart sampling: pick diverse records
        import random
        samples = []

        # 1. Most common class — random record from that class
        if self.stats["concepts"]:
            most_common = max(self.stats["concepts"], key=self.stats["concepts"].get)
            pool = [r for r in self.records if r["concept"] == most_common]
            if pool:
                samples.append(random.choice(pool))

        # 2. Rarest class — random record from that class
        if self.stats["concepts"]:
            rarest = min(self.stats["concepts"], key=self.stats["concepts"].get)
            pool = [r for r in self.records if r["concept"] == rarest and r not in samples]
            if pool:
                samples.append(random.choice(pool))

        # 3. Most bounding boxes on one image — random record from that image
        from collections import Counter
        img_counts = Counter(r["image"] for r in self.records)
        if img_counts:
            busiest_img = img_counts.most_common(1)[0][0]
            pool = [r for r in self.records if r["image"] == busiest_img and r not in samples]
            if pool:
                samples.append(random.choice(pool))

        # 4. Smallest bounding box
        smallest = min(self.records, key=lambda r: r["width"] * r["height"])
        if smallest not in samples:
            samples.append(smallest)

        # 5. Largest bounding box
        largest = max(self.records, key=lambda r: r["width"] * r["height"])
        if largest not in samples:
            samples.append(largest)

        # Fill remaining with random picks from different classes
        remaining = [r for r in self.records if r not in samples]
        random.shuffle(remaining)
        seen_concepts = {r["concept"] for r in samples}
        for r in remaining:
            if len(samples) >= n:
                break
            if r["concept"] not in seen_concepts:
                samples.append(r)
                seen_concepts.add(r["concept"])
        # If still not enough, just add more
        for r in remaining:
            if len(samples) >= n:
                break
            if r not in samples:
                samples.append(r)

        return samples[:n]
