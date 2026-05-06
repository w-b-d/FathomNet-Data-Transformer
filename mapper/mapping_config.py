"""
Mapping configuration — the universal intermediate format.

All three modes (known format, manual, AI-assisted) produce a MappingConfig.
The transformation engine consumes it.
"""

import json
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class FieldMapping:
    """How to map a single source field to a FathomNet field."""
    source: str                          # source field name or special source
    transform: Optional[str] = None      # optional transform expression
    lookup: Optional[str] = None         # optional lookup file for ID → name
    default: Optional[str] = None        # default if source field is missing


@dataclass
class MappingConfig:
    """Complete mapping configuration for a dataset → FathomNet conversion."""

    # What converter/mode produced this
    source_format: str = "unknown"

    # Field mappings: fathomnet_field → FieldMapping
    field_map: dict = field(default_factory=dict)

    # Classes to exclude from output
    exclude_concepts: list = field(default_factory=list)

    # Concept name transformations
    concept_aliases: dict = field(default_factory=dict)  # "DR" → "Dascyllus Reticulatus"

    # Coordinate adjustments
    x_offset: int = 0
    y_offset: int = 0
    coordinate_format: str = "xywh"  # xywh, xyxy, cxcywh, cxcywh_abs

    # Extra options
    merge_splits: bool = True  # combine train/val/test into one CSV
    extra: dict = field(default_factory=dict)

    def save(self, path: str):
        """Save mapping config to YAML file."""
        path = Path(path).expanduser()
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "MappingConfig":
        """Load mapping config from YAML file."""
        path = Path(path).expanduser()
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("mapping config must contain a YAML object")
        config = cls()
        for k, v in data.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "MappingConfig":
        data = json.loads(json_str)
        config = cls()
        for k, v in data.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config
