"""
Pascal VOC XML → FathomNet converter.

Handles Pascal VOC format with <annotation><object><bndbox> structure.
Bounding boxes are in [xmin, ymin, xmax, ymax] format → converted to [x, y, w, h].
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Generator, Optional

from .base import BaseConverter, FathomNetRecord


class PascalVOCConverter(BaseConverter):
    format_name = "pascal_voc"
    format_description = "Pascal VOC XML format (<annotation><object><bndbox>)"

    def convert(
        self,
        dataset_path: str,
        detection_result: dict,
        field_overrides: Optional[dict] = None,
    ) -> Generator[FathomNetRecord, None, None]:
        overrides = field_overrides or {}
        dataset_path = Path(dataset_path)
        dataset_root = dataset_path.parent if dataset_path.is_file() else dataset_path

        # Field name overrides for non-standard VOC
        name_field = overrides.get("concept_field", "name")
        filename_field = overrides.get("image_field", "filename")

        xml_files = [
            f for f in detection_result.get("annotation_files", [])
            if f.endswith(".xml")
        ]

        for xml_file in xml_files:
            yield from self._convert_single_file(
                xml_file, dataset_root, name_field, filename_field, overrides
            )

    def _convert_single_file(
        self,
        xml_file: str,
        dataset_path: Path,
        name_field: str,
        filename_field: str,
        overrides: dict,
    ) -> Generator[FathomNetRecord, None, None]:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except ET.ParseError as e:
            self.record_drop(
                item_type="annotation_file",
                reason="Pascal VOC XML could not be parsed",
                source=xml_file,
                detail=str(e),
            )
            return
        except OSError as e:
            self.record_drop(
                item_type="annotation_file",
                reason="Pascal VOC XML could not be read",
                source=xml_file,
                detail=str(e),
            )
            return

        if root.tag != "annotation":
            self.record_drop(
                item_type="annotation_file",
                reason="Pascal VOC XML root is not <annotation>",
                source=xml_file,
                detail=root.tag,
            )
            return

        # Get image filename
        fname_el = root.find(filename_field)
        image_name = fname_el.text if fname_el is not None else Path(xml_file).stem
        source_image_path = self._resolve_image_path(
            dataset_path, Path(xml_file), image_name
        )

        for obj_index, obj in enumerate(root.findall(".//object"), start=1):
            # Get concept name
            name_el = obj.find(name_field)
            if name_el is None:
                # Try common alternatives
                for alt in ("label", "class", "category"):
                    name_el = obj.find(alt)
                    if name_el is not None:
                        break
            concept = name_el.text if name_el is not None else "Unknown"

            # Get bounding box — VOC uses xmin/ymin/xmax/ymax
            bndbox = obj.find("bndbox")
            if bndbox is None:
                self.record_drop(
                    item_type="annotation",
                    reason="Pascal VOC object missing bndbox",
                    source=f"{xml_file}: object {obj_index}",
                    image=image_name,
                    source_image_path=source_image_path,
                    concept=concept,
                )
                continue

            try:
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
            except (AttributeError, ValueError, TypeError) as e:
                self.record_drop(
                    item_type="annotation",
                    reason="Pascal VOC object has invalid bbox values",
                    source=f"{xml_file}: object {obj_index}",
                    image=image_name,
                    source_image_path=source_image_path,
                    concept=concept,
                    detail=str(e),
                )
                continue

            record = FathomNetRecord(
                concept=concept,
                image=image_name,
                x=round(xmin),
                y=round(ymin),
                width=round(xmax - xmin),
                height=round(ymax - ymin),
                source_image_path=source_image_path,
            )

            # Optional VOC fields
            difficult = obj.find("difficult")
            if difficult is not None and difficult.text == "1":
                record.extra["difficult"] = True

            truncated_el = obj.find("truncated")
            if truncated_el is not None:
                record.truncated = truncated_el.text == "1"

            occluded_el = obj.find("occluded")
            if occluded_el is not None:
                record.occluded = occluded_el.text == "1"

            context = self._build_record_context(
                root=root,
                obj=obj,
                bndbox=bndbox,
                xml_file=Path(xml_file),
                image_name=image_name,
                source_image_path=source_image_path,
            )
            self._apply_optional_overrides_from_context(record, overrides, context)
            self._apply_extra_columns_from_context(record, overrides, context)

            yield record

    def _resolve_image_path(
        self, dataset_path: Path, xml_file: Path, image_name: str
    ) -> Optional[str]:
        image_ref = Path(image_name)
        if image_ref.is_absolute() and image_ref.exists():
            return str(image_ref)

        candidates = [
            xml_file.parent / image_ref,
            xml_file.parent.parent / "images" / image_ref,
            xml_file.parent.parent / "JPEGImages" / image_ref,
            dataset_path / image_ref,
            dataset_path / "images" / image_ref,
            dataset_path / "JPEGImages" / image_ref,
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    def get_field_names(self, dataset_path: str, detection_result: dict) -> list[str]:
        return sorted(self.get_field_samples(dataset_path, detection_result).keys())

    def get_field_samples(
        self, dataset_path: str, detection_result: dict
    ) -> dict[str, list[str]]:
        fields = set()
        samples: dict[str, list[str]] = {}
        xml_files = [
            f for f in detection_result.get("annotation_files", [])
            if f.endswith(".xml")
        ]
        for xml_file in xml_files[:3]:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                self._collect_xml_field_samples("annotation", root, fields, samples)
                for obj in root.findall(".//object")[:20]:
                    self._collect_xml_field_samples("object", obj, fields, samples)
            except (ET.ParseError, OSError):
                continue
        for field_name in (
            "annotation.filename",
            "object.name",
            "object.bndbox",
            "object.bndbox.xmin",
            "object.bndbox.ymin",
            "object.bndbox.xmax",
            "object.bndbox.ymax",
        ):
            self._add_field_sample(fields, samples, field_name, "")
        return {field: samples.get(field, []) for field in sorted(fields)}

    def get_default_field_map(
        self, dataset_path: str, detection_result: dict
    ) -> dict[str, str]:
        return {
            "concept": "object.name",
            "image": "annotation.filename",
            "x": "object.bndbox",
            "y": "object.bndbox",
            "width": "object.bndbox",
            "height": "object.bndbox",
        }

    def _build_record_context(
        self,
        *,
        root: ET.Element,
        obj: ET.Element,
        bndbox: ET.Element,
        xml_file: Path,
        image_name: str,
        source_image_path: Optional[str],
    ) -> dict:
        return {
            "annotation": self._xml_element_to_value(root),
            "object": self._xml_element_to_value(obj),
            "bndbox": self._xml_element_to_value(bndbox),
            "file": {
                "path": str(xml_file),
                "name": xml_file.name,
                "stem": xml_file.stem,
            },
            "image": {
                "filename": image_name,
                "path": source_image_path or "",
            },
        }

    def _collect_xml_field_samples(
        self,
        prefix: str,
        element: ET.Element,
        fields: set[str],
        samples: dict[str, list[str]],
    ):
        path = f"{prefix}.{element.tag}" if element.tag != prefix else prefix
        value = self._xml_element_to_value(element)
        if isinstance(value, dict):
            self._add_field_sample(fields, samples, path, value)
            for attr_name, attr_value in element.attrib.items():
                self._add_field_sample(fields, samples, f"{path}.{attr_name}", attr_value)
            for child in element:
                self._collect_xml_field_samples(path, child, fields, samples)
        else:
            self._add_field_sample(fields, samples, path, value)

    def _xml_element_to_value(self, element: ET.Element):
        children = list(element)
        data = dict(element.attrib)
        for child in children:
            child_value = self._xml_element_to_value(child)
            if child.tag in data:
                if not isinstance(data[child.tag], list):
                    data[child.tag] = [data[child.tag]]
                data[child.tag].append(child_value)
            else:
                data[child.tag] = child_value

        text = (element.text or "").strip()
        if children or data:
            if text:
                data["text"] = text
            return data
        return text

    def validate_prerequisites(
        self, dataset_path: str, detection_result: dict
    ) -> list[str]:
        errors = []
        xml_files = [
            f for f in detection_result.get("annotation_files", [])
            if f.endswith(".xml")
        ]
        if not xml_files:
            errors.append("No XML annotation files found.")
        return errors
