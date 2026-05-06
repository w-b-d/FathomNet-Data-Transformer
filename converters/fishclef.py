"""
Video + XML annotation → FathomNet converter.

Handles datasets where:
  - Annotations are in XML files referencing video frames by index
  - Images need to be extracted from video files (.mp4, .flv, .avi, etc.)

Built for FishCLEF-style datasets but works for any video+annotation format
where XML files describe objects per frame with bounding boxes.

Uses the shared VideoLoader utility for frame extraction.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Generator, Optional

from .base import BaseConverter, FathomNetRecord
from utils.video import VideoLoader, HAS_CV2


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


class FishCLEFConverter(BaseConverter):
    format_name = "fishclef_xml"
    format_description = "Video + XML annotations (frame extraction required)"

    def convert(
        self,
        dataset_path: str,
        detection_result: dict,
        field_overrides: Optional[dict] = None,
    ) -> Generator[FathomNetRecord, None, None]:
        overrides = field_overrides or {}
        dataset_path = Path(dataset_path)

        # Locate video and XML directories
        video_dir = self._find_dir(dataset_path, ["videos", "video"])
        xml_dir = self._find_dir(dataset_path, ["gt", "annotations", "xml", "labels"])

        if not video_dir:
            print(f"Could not find video directory under {dataset_path}")
            self.record_drop(
                item_type="video",
                reason="Could not find video directory",
                source=str(dataset_path),
            )
            return
        if not xml_dir:
            print(f"Could not find XML annotation directory under {dataset_path}")
            self.record_drop(
                item_type="annotation_file",
                reason="Could not find XML annotation directory",
                source=str(dataset_path),
            )
            return

        # Output directory for extracted frames
        output_dir = overrides.get("frame_output_dir")
        if not output_dir:
            output_dir = str(dataset_path.parent / "fathomnet_output" / "images")
        os.makedirs(output_dir, exist_ok=True)

        # Configurable XML field names — works for FishCLEF and similar formats
        frame_tag = overrides.get("frame_tag", "frame")
        frame_id_attr = overrides.get("frame_id_attr", "id")
        object_tag = overrides.get("object_tag", "object")
        species_attrs = overrides.get("species_attributes", ["fish_species", "species_name", "name", "label", "class"])
        x_attr = overrides.get("x_attr", "x")
        y_attr = overrides.get("y_attr", "y")
        w_attr = overrides.get("w_attr", "w")
        h_attr = overrides.get("h_attr", "h")

        if isinstance(species_attrs, str):
            species_attrs = [species_attrs]

        # Load videos using shared utility
        print(f"Loading videos from {video_dir}...")
        loader = VideoLoader(str(video_dir))
        loaded = loader.load_all()
        print(f"Loaded {len(loaded)} videos ({sum(loaded.values())} total frames)")

        if not loaded:
            print("No videos could be loaded. Check video format and OpenCV installation.")
            self.record_drop(
                item_type="video",
                reason="No videos could be loaded",
                source=str(video_dir),
            )
            return

        # Process each XML file
        xml_files = sorted(xml_dir.glob("*.xml"))
        for xml_file in xml_files:
            yield from self._process_xml(
                xml_file, loader, output_dir,
                frame_tag, frame_id_attr, object_tag,
                species_attrs, x_attr, y_attr, w_attr, h_attr, overrides,
            )

    def _process_xml(
        self,
        xml_path: Path,
        loader: VideoLoader,
        output_dir: str,
        frame_tag: str,
        frame_id_attr: str,
        object_tag: str,
        species_attrs: list,
        x_attr: str,
        y_attr: str,
        w_attr: str,
        h_attr: str,
        overrides: dict,
    ) -> Generator[FathomNetRecord, None, None]:
        """Process a single XML annotation file."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except (ET.ParseError, OSError) as e:
            print(f"Error parsing {xml_path}: {e}")
            self.record_drop(
                item_type="annotation_file",
                reason="FishCLEF XML could not be parsed or read",
                source=str(xml_path),
                detail=str(e),
            )
            return

        # Match XML to video — try the full stem first, then the part before first dot
        key = xml_path.stem
        if key not in loader.videos:
            key = xml_path.stem.split(".")[0]
        if key not in loader.videos:
            # Try matching by prefix (some datasets append suffixes)
            for vkey in loader.get_video_keys():
                if key.startswith(vkey) or vkey.startswith(key):
                    key = vkey
                    break
            else:
                print(f"No video match for XML: {xml_path.name}")
                self.record_drop(
                    item_type="annotation_file",
                    reason="No matching video for XML annotation file",
                    source=str(xml_path),
                )
                return

        video_name = key

        for frame_el in root.findall(frame_tag):
            frame_id = _safe_int(frame_el.get(frame_id_attr), -1)
            objects = frame_el.findall(object_tag)
            if not objects:
                continue

            # Get the frame from the video
            frame = loader.get_frame(key, frame_id)
            if frame is None:
                self.record_drop(
                    item_type="image",
                    reason="Video frame could not be extracted; annotations skipped",
                    source=f"{xml_path}: frame {frame_id}",
                )
                continue

            # Save frame as image
            image_filename = loader.save_frame(frame, output_dir, video_name, frame_id)
            if image_filename is None:
                self.record_drop(
                    item_type="image",
                    reason="Video frame image could not be saved; annotations skipped",
                    source=f"{xml_path}: frame {frame_id}",
                )
                continue

            source_image_path = str(Path(output_dir) / image_filename)

            # Yield one record per object in this frame
            for obj_el in objects:
                # Try multiple species attribute names
                concept = "Unknown"
                for attr in species_attrs:
                    val = obj_el.get(attr)
                    if val:
                        concept = val
                        break

                x = _safe_int(obj_el.get(x_attr), 0)
                y = _safe_int(obj_el.get(y_attr), 0)
                w = _safe_int(obj_el.get(w_attr), 0)
                h = _safe_int(obj_el.get(h_attr), 0)

                record = FathomNetRecord(
                    concept=concept,
                    image=image_filename,
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    source_image_path=source_image_path,
                )
                record.extra["video_name"] = video_name
                record.extra["frame_number"] = frame_id

                context = {
                    "video": {"name": video_name},
                    "frame": {
                        **frame_el.attrib,
                        "id": frame_id,
                    },
                    "object": self._xml_element_to_value(obj_el),
                    "file": {
                        "path": str(xml_path),
                        "name": xml_path.name,
                        "stem": xml_path.stem,
                    },
                    "image": {
                        "filename": image_filename,
                        "path": source_image_path,
                    },
                }
                self._apply_optional_overrides_from_context(record, overrides, context)
                self._apply_extra_columns_from_context(record, overrides, context)

                yield record

    def _find_dir(self, base: Path, candidates: list) -> Optional[Path]:
        """Find a subdirectory matching one of the candidate names, searching up to 2 levels deep."""
        # Direct children
        for name in candidates:
            d = base / name
            if d.is_dir():
                return d

        # One level deeper
        for child in base.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                for name in candidates:
                    d = child / name
                    if d.is_dir():
                        return d

        return None

    def get_field_names(self, dataset_path: str, detection_result: dict) -> list[str]:
        samples = self.get_field_samples(dataset_path, detection_result)
        if samples:
            return sorted(samples.keys())
        return ["object.fish_species", "object.species_name", "object.name",
                "object.label", "object.class", "object.x", "object.y",
                "object.w", "object.h", "frame.id", "video.name"]

    def get_field_samples(
        self, dataset_path: str, detection_result: dict
    ) -> dict[str, list[str]]:
        fields = set()
        samples: dict[str, list[str]] = {}
        xml_files = [
            Path(f) for f in detection_result.get("annotation_files", [])
            if str(f).endswith(".xml")
        ]
        if not xml_files:
            xml_dir = self._find_dir(
                Path(dataset_path), ["gt", "annotations", "xml", "labels"]
            )
            if xml_dir:
                xml_files = sorted(xml_dir.glob("*.xml"))

        for xml_file in xml_files[:3]:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
            except (ET.ParseError, OSError):
                continue

            self._collect_xml_field_samples("video", root, fields, samples)
            for frame_el in root.findall(".//frame")[:20]:
                self._collect_xml_field_samples("frame", frame_el, fields, samples)
                for obj_el in frame_el.findall(".//object")[:20]:
                    self._collect_xml_field_samples("object", obj_el, fields, samples)

        # These generated fields are always available after frame extraction.
        for field_name in (
            "object.fish_species",
            "object.species_name",
            "object.name",
            "object.label",
            "object.class",
            "object.x",
            "object.y",
            "object.w",
            "object.h",
            "frame.id",
        ):
            self._add_field_sample(fields, samples, field_name, "")
        self._add_field_sample(fields, samples, "video.name", "")
        self._add_field_sample(fields, samples, "image.filename", "")
        self._add_field_sample(fields, samples, "image.path", "")
        self._add_field_sample(fields, samples, "file.name", "")
        self._add_field_sample(fields, samples, "file.path", "")
        return {field: samples.get(field, []) for field in sorted(fields)}

    def get_default_field_map(
        self, dataset_path: str, detection_result: dict
    ) -> dict[str, str]:
        return {
            "concept": "object.fish_species",
            "image": "image.filename",
            "x": "object.x",
            "y": "object.y",
            "width": "object.w",
            "height": "object.h",
        }

    def validate_prerequisites(
        self, dataset_path: str, detection_result: dict
    ) -> list[str]:
        errors = []
        if not HAS_CV2:
            errors.append(
                "OpenCV (cv2) is required for video frame extraction. "
                "Install with: pip install opencv-python"
            )

        dataset_path = Path(dataset_path)
        video_dir = self._find_dir(dataset_path, ["videos", "video"])
        xml_dir = self._find_dir(dataset_path, ["gt", "annotations", "xml", "labels"])

        if not video_dir:
            errors.append(
                f"No video directory found under {dataset_path}. "
                f"Expected: videos/ or video/"
            )
        if not xml_dir:
            errors.append(
                f"No XML annotation directory found under {dataset_path}. "
                f"Expected: gt/, annotations/, or xml/"
            )

        return errors
