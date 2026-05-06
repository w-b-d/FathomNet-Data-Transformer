"""
Base converter interface.

All format-specific converters extend BaseConverter and implement convert().
Every converter produces a list of FathomNetRecord dicts — the universal
intermediate format before CSV export.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Generator, Any
from pathlib import Path

from fathomnet_schema import OPTIONAL_FIELDS


@dataclass
class FathomNetRecord:
    """A single row destined for the FathomNet metadata.csv."""

    # Required
    concept: str = ""
    image: str = ""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    # Optional metadata
    depth: Optional[float] = None
    altitude: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    temperature: Optional[float] = None
    salinity: Optional[float] = None
    oxygen: Optional[float] = None
    pressure: Optional[float] = None
    observer: Optional[str] = None
    timestamp: Optional[str] = None
    imagingtype: Optional[str] = None
    occluded: Optional[bool] = None
    truncated: Optional[bool] = None
    userdefinedkey: Optional[str] = None
    altconcept: Optional[str] = None
    groupof: Optional[bool] = None
    source_image_path: Optional[str] = None

    # Extra fields (preserved as key-value tags)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dict, dropping None values and flattening extras."""
        d = {}
        for k, v in asdict(self).items():
            if k in ("extra", "source_image_path"):
                continue
            if v is not None:
                d[k] = v
        d.update(self.extra)
        return d


class BaseConverter:
    """
    Base class for all format converters.

    Subclasses must implement:
        - convert(dataset_path, detection_result, field_overrides) -> Generator[FathomNetRecord]
        - format_name: str (class attribute)
        - format_description: str (class attribute)
    """

    format_name: str = "base"
    format_description: str = "Base converter"

    def __init__(self):
        self.dropped_items: list[dict] = []

    def reset_drop_report(self):
        """Clear converter-local drop tracking before a new run."""
        self.dropped_items = []

    def record_drop(
        self,
        *,
        item_type: str,
        reason: str,
        source: Optional[str] = None,
        image: Optional[str] = None,
        source_image_path: Optional[str] = None,
        concept: Optional[str] = None,
        record: Optional[Any] = None,
        detail: Optional[str] = None,
    ):
        """Track an item skipped before it reaches the transform engine."""
        item = {
            "type": item_type,
            "stage": self.format_name,
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
            if hasattr(record, "to_dict"):
                item["record"] = record.to_dict()
                if getattr(record, "source_image_path", None):
                    item["source_image_path"] = record.source_image_path
                if getattr(record, "image", None):
                    item.setdefault("image", record.image)
                if getattr(record, "concept", None):
                    item.setdefault("concept", record.concept)
            else:
                item["record"] = record
        if detail:
            item["detail"] = str(detail)
        self.dropped_items.append(item)

    def convert(
        self,
        dataset_path: str,
        detection_result: dict,
        field_overrides: Optional[dict] = None,
    ) -> Generator[FathomNetRecord, None, None]:
        """
        Convert dataset to FathomNet records.

        Args:
            dataset_path: path to the dataset root
            detection_result: output from detect_format()
            field_overrides: optional user-specified field name mappings
                e.g. {"concept": "class_name", "image": "img_path"}

        Yields:
            FathomNetRecord instances
        """
        raise NotImplementedError

    def get_field_names(self, dataset_path: str, detection_result: dict) -> list[str]:
        """
        Return the list of field/column names found in this dataset.
        Used for fuzzy matching and manual mapping UI.
        """
        return []

    def get_field_samples(
        self, dataset_path: str, detection_result: dict
    ) -> dict[str, list[str]]:
        """
        Return sample values for detected fields when available.
        Used by interactive metadata-column selection.
        """
        return {}

    def get_default_field_map(
        self, dataset_path: str, detection_result: dict
    ) -> dict[str, str]:
        """
        Return the converter's built-in FathomNet field mappings.
        Used to show what pressing Enter keeps in known-format mode.
        """
        return {}

    def _apply_optional_overrides_from_context(
        self,
        record: FathomNetRecord,
        overrides: dict,
        context: dict[str, Any],
    ):
        """Apply user-selected optional FathomNet field mappings."""
        for field_name in OPTIONAL_FIELDS:
            source = overrides.get(field_name)
            if not source:
                continue

            value = self._lookup_context_value(str(source), context)
            value = self._coerce_optional_value(field_name, value)
            if value is None or value == "":
                continue

            if hasattr(record, field_name):
                setattr(record, field_name, value)
            else:
                record.extra[field_name] = value

    def _apply_extra_columns_from_context(
        self,
        record: FathomNetRecord,
        overrides: dict,
        context: dict[str, Any],
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

            value = self._lookup_context_value(str(source), context)
            value = self._coerce_extra_value(value)
            # Keep selected columns even when every row's value is blank/null.
            # Otherwise the CSV header silently loses the user-selected column.
            record.extra[str(column)] = "" if value is None else value

    def _lookup_context_value(self, source: str, context: dict[str, Any]):
        """Look up a dotted source path in converter-provided context dicts."""
        source = source.strip()
        if not source:
            return None

        if "." in source:
            prefix, remainder = source.split(".", 1)
            container = context.get(prefix)
            if container is not None:
                return self._lookup_nested_value(container, remainder)

        for container in context.values():
            value = self._lookup_nested_value(container, source)
            if value is not None:
                return value
        return None

    def _lookup_nested_value(self, data: Any, path: str):
        value = data
        for part in path.split("."):
            if isinstance(value, dict):
                if part not in value:
                    return None
                value = value[part]
            elif isinstance(value, list) and part.isdigit():
                index = int(part)
                if index >= len(value):
                    return None
                value = value[index]
            else:
                return None
        return value

    def _coerce_optional_value(self, field_name: str, value):
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if value == "" or value.lower() in {"na", "n/a", "none", "null"}:
                return None

        if field_name in {
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

        if field_name in {"occluded", "truncated", "groupof"}:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value != 0
            if isinstance(value, str):
                return value.lower() in {"1", "true", "yes", "y"}
            return None

        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value)

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

    def _add_field_sample(
        self,
        fields: set[str],
        samples: dict[str, list[str]],
        field_name: str,
        value,
        *,
        max_values: int = 4,
    ):
        fields.add(field_name)
        text = self._sample_value_to_text(value)
        if text == "":
            return
        values = samples.setdefault(field_name, [])
        if text not in values and len(values) < max_values:
            values.append(text)

    def _sample_value_to_text(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        return str(value)

    def _collect_xml_field_samples(
        self,
        prefix: str,
        element: Any,
        fields: set[str],
        samples: dict[str, list[str]],
    ):
        """Collect dotted XML element/attribute paths and sample values."""
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

    def _xml_element_to_value(self, element: Any):
        """Convert an XML element into nested dict/text data for lookup."""
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

    def validate_prerequisites(self, dataset_path: str, detection_result: dict) -> list[str]:
        """
        Check if the dataset has everything needed for conversion.

        Returns:
            list of error messages (empty = ready to convert)
        """
        return []
