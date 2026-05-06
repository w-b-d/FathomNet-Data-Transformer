"""
Post-transformation validator.

Checks the output records against FathomNet requirements and
flags issues with severity levels (error, warning, info).
"""

from collections import Counter
from typing import Optional

from fathomnet_schema import NOAA_NCEI_IMAGE_EXTS, REQUIRED_FIELDS


def validate_output(
    records: list[dict],
    image_dir: Optional[str] = None,
    submission_target: str = "noaa-ncei",
) -> dict:
    """
    Validate transformed records against FathomNet requirements.

    Args:
        records: list of record dicts (output from TransformEngine)
        image_dir: optional path to check if image files exist
        submission_target: "noaa-ncei" or "self-hosted"

    Returns:
        dict with:
            - issues: list of {severity, check, message, affected_count}
            - summary: dict with overall stats
            - passed: bool (True if no errors)
    """
    issues = []
    summary = {
        "total_records": len(records),
        "unique_images": 0,
        "unique_concepts": 0,
    }

    if not records:
        issues.append({
            "severity": "error",
            "check": "empty_output",
            "message": "No records produced. The transformation generated zero output rows.",
            "affected_count": 0,
        })
        return {"issues": issues, "summary": summary, "passed": False}

    # ── Required fields ──────────────────────────────────────────────────

    for field_name in REQUIRED_FIELDS:
        missing = sum(1 for r in records if r.get(field_name) is None or r.get(field_name) == "")
        if missing > 0:
            issues.append({
                "severity": "error",
                "check": f"missing_{field_name}",
                "message": f"'{field_name}' is empty or missing in {missing} records.",
                "affected_count": missing,
            })

    # ── Concept checks ───────────────────────────────────────────────────

    concepts = Counter(r.get("concept", "") for r in records)
    summary["unique_concepts"] = len(concepts)

    # Unknown/empty concepts
    unknown_count = concepts.get("Unknown", 0) + concepts.get("unknown", 0) + concepts.get("", 0)
    if unknown_count > 0:
        pct = unknown_count / len(records) * 100
        issues.append({
            "severity": "warning",
            "check": "unknown_concepts",
            "message": f"{unknown_count} records ({pct:.1f}%) have 'Unknown' or empty concept.",
            "affected_count": unknown_count,
        })

    # Single concept (suspicious)
    if len(concepts) == 1:
        issues.append({
            "severity": "warning",
            "check": "single_concept",
            "message": f"Only one unique concept found: '{list(concepts.keys())[0]}'. Is this intentional?",
            "affected_count": len(records),
        })

    # Concept distribution
    summary["concept_distribution"] = dict(concepts.most_common(20))

    # ── Bounding box checks ──────────────────────────────────────────────

    zero_area = sum(
        1 for r in records
        if r.get("width", 0) <= 0 or r.get("height", 0) <= 0
    )
    if zero_area > 0:
        issues.append({
            "severity": "error",
            "check": "zero_area_bbox",
            "message": f"{zero_area} records have zero or negative bbox area.",
            "affected_count": zero_area,
        })

    negative_coords = sum(
        1 for r in records
        if r.get("x", 0) < 0 or r.get("y", 0) < 0
    )
    if negative_coords > 0:
        issues.append({
            "severity": "error",
            "check": "negative_coordinates",
            "message": f"{negative_coords} records have negative x or y coordinates.",
            "affected_count": negative_coords,
        })

    # Suspiciously tiny boxes (< 5x5 pixels)
    tiny = sum(
        1 for r in records
        if 0 < r.get("width", 0) < 5 and 0 < r.get("height", 0) < 5
    )
    if tiny > 0:
        pct = tiny / len(records) * 100
        issues.append({
            "severity": "warning",
            "check": "tiny_bbox",
            "message": f"{tiny} records ({pct:.1f}%) have very small bboxes (< 5x5 px). Possible coordinate error.",
            "affected_count": tiny,
        })

    # All same coordinates (likely parse error)
    coords = [(r.get("x"), r.get("y"), r.get("width"), r.get("height")) for r in records]
    if len(set(coords)) == 1 and len(records) > 1:
        issues.append({
            "severity": "error",
            "check": "identical_coordinates",
            "message": "ALL records have identical bounding boxes. Likely a parsing error.",
            "affected_count": len(records),
        })

    # ── Image checks ─────────────────────────────────────────────────────

    images = Counter(r.get("image", "") for r in records)
    summary["unique_images"] = len(images)

    if submission_target == "noaa-ncei":
        url_images = [
            img_name for img_name in images
            if str(img_name).startswith(("http://", "https://"))
        ]
        if url_images:
            issues.append({
                "severity": "error",
                "check": "noaa_ncei_image_urls",
                "message": (
                    f"{len(url_images)} images use URLs. NOAA-NCEI submissions "
                    "require local image filenames in the CSV."
                ),
                "affected_count": len(url_images),
            })

        from pathlib import Path
        unsupported_images = [
            img_name for img_name in images
            if not str(img_name).startswith(("http://", "https://"))
            and Path(str(img_name)).suffix.lower() not in NOAA_NCEI_IMAGE_EXTS
        ]
        if unsupported_images:
            issues.append({
                "severity": "error",
                "check": "noaa_ncei_image_format",
                "message": (
                    f"{len(unsupported_images)} images are not .jpg or .png. "
                    "NOAA-NCEI submissions should only include .jpg or .png images."
                ),
                "affected_count": len(unsupported_images),
            })

    # Duplicate annotations (same image + same bbox + same concept)
    seen = set()
    dupes = 0
    for r in records:
        key = (r.get("image"), r.get("concept"), r.get("x"), r.get("y"), r.get("width"), r.get("height"))
        if key in seen:
            dupes += 1
        seen.add(key)
    if dupes > 0:
        issues.append({
            "severity": "warning",
            "check": "duplicate_annotations",
            "message": f"{dupes} duplicate annotations found (same image + concept + bbox).",
            "affected_count": dupes,
        })

    # Check if image files exist (if image_dir provided)
    if image_dir:
        from pathlib import Path
        image_path = Path(image_dir)
        if image_path.exists():
            missing_images = set()
            for img_name in images:
                if str(img_name).startswith(("http://", "https://")):
                    continue
                if not (image_path / img_name).exists():
                    missing_images.add(img_name)
            if missing_images:
                issues.append({
                    "severity": "warning",
                    "check": "missing_image_files",
                    "message": f"{len(missing_images)} referenced images not found in {image_dir}.",
                    "affected_count": len(missing_images),
                })

    # ── Metadata checks ──────────────────────────────────────────────────

    # userdefinedkey length check
    long_keys = sum(
        1 for r in records
        if r.get("userdefinedkey") and len(str(r["userdefinedkey"])) > 56
    )
    if long_keys > 0:
        issues.append({
            "severity": "error",
            "check": "userdefinedkey_too_long",
            "message": f"{long_keys} records have userdefinedkey > 56 characters.",
            "affected_count": long_keys,
        })

    # Latitude/longitude range checks
    bad_geo = 0
    for r in records:
        lat = r.get("latitude")
        lon = r.get("longitude")
        if lat is not None and (lat < -90 or lat > 90):
            bad_geo += 1
        if lon is not None and (lon < -180 or lon > 180):
            bad_geo += 1
    if bad_geo > 0:
        issues.append({
            "severity": "error",
            "check": "invalid_coordinates_geo",
            "message": f"{bad_geo} records have latitude/longitude out of valid range.",
            "affected_count": bad_geo,
        })

    # ── Summary ──────────────────────────────────────────────────────────

    # Bbox area stats
    areas = [r.get("width", 0) * r.get("height", 0) for r in records if r.get("width", 0) > 0]
    if areas:
        summary["bbox_area_min"] = min(areas)
        summary["bbox_area_max"] = max(areas)
        summary["bbox_area_mean"] = round(sum(areas) / len(areas), 1)

    # Optional fields present
    optional_present = []
    for field_name in ["depth", "latitude", "longitude", "timestamp", "observer"]:
        count = sum(1 for r in records if r.get(field_name) is not None)
        if count > 0:
            optional_present.append(f"{field_name} ({count})")
    if optional_present:
        issues.append({
            "severity": "info",
            "check": "optional_fields",
            "message": f"Optional fields populated: {', '.join(optional_present)}",
            "affected_count": 0,
        })
    else:
        issues.append({
            "severity": "info",
            "check": "no_optional_fields",
            "message": "No optional metadata fields populated (depth, lat/lon, timestamp, etc.).",
            "affected_count": 0,
        })

    has_errors = any(i["severity"] == "error" for i in issues)

    return {
        "issues": issues,
        "summary": summary,
        "passed": not has_errors,
    }
