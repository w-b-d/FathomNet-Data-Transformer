"""
Fuzzy field name matching.

Three-tier strategy:
  1. Exact match → auto-accept
  2. Known alias → auto-accept with notification
  3. Fuzzy similarity → suggest to user for confirmation
"""

from difflib import SequenceMatcher
from typing import Optional

from fathomnet_schema import FIELD_ALIASES


# Common abbreviation expansions
ABBREVIATIONS = {
    "img": "image",
    "cat": "category",
    "cls": "class",
    "ann": "annotation",
    "bbox": "boundingbox",
    "bb": "boundingbox",
    "w": "width",
    "h": "height",
    "fn": "filename",
    "fname": "filename",
    "lat": "latitude",
    "lon": "longitude",
    "lng": "longitude",
    "temp": "temperature",
    "sal": "salinity",
    "obs": "observer",
    "ts": "timestamp",
    "dep": "depth",
    "alt": "altitude",
}


def _normalize(name: str) -> str:
    """Normalize a field name for comparison."""
    return name.lower().replace("_", "").replace("-", "").replace(" ", "")


def _expand_abbreviations(name: str) -> str:
    """Expand known abbreviations."""
    normalized = _normalize(name)
    return ABBREVIATIONS.get(normalized, normalized)


def fuzzy_match_fields(
    source_fields: list[str],
    target_fields: Optional[dict] = None,
) -> dict:
    """
    Match source field names to FathomNet target fields.

    Args:
        source_fields: field names from the uploaded dataset
        target_fields: dict of {fathomnet_field: description}, defaults to required fields

    Returns:
        dict with keys:
            - exact: {fathomnet_field: source_field} — perfect matches
            - alias: {fathomnet_field: (source_field, alias_used)} — known alias matches
            - fuzzy: {fathomnet_field: [(source_field, score), ...]} — similarity matches
            - unmatched_target: [fathomnet_fields with no match]
            - unmatched_source: [source_fields not matched to anything]
    """
    if target_fields is None:
        from fathomnet_schema import REQUIRED_FIELDS
        target_fields = REQUIRED_FIELDS

    result = {
        "exact": {},
        "alias": {},
        "fuzzy": {},
        "unmatched_target": [],
        "unmatched_source": list(source_fields),
    }

    matched_sources = set()

    for target_name in target_fields:
        target_norm = _normalize(target_name)

        # Tier 1: Exact match (after normalization)
        exact_found = False
        for src in source_fields:
            if src in matched_sources:
                continue
            if _normalize(src) == target_norm:
                result["exact"][target_name] = src
                matched_sources.add(src)
                exact_found = True
                break

        if exact_found:
            continue

        # Tier 2: Known alias match
        aliases = FIELD_ALIASES.get(target_name, [])
        alias_norms = {_normalize(a): a for a in aliases}
        alias_found = False
        for src in source_fields:
            if src in matched_sources:
                continue
            src_norm = _normalize(src)
            if src_norm in alias_norms:
                result["alias"][target_name] = (src, alias_norms[src_norm])
                matched_sources.add(src)
                alias_found = True
                break

        if alias_found:
            continue

        # Tier 3: Fuzzy similarity
        candidates = []
        for src in source_fields:
            if src in matched_sources:
                continue
            score = _similarity_score(target_name, src)
            if score > 0.4:
                candidates.append((src, round(score, 2)))

        candidates.sort(key=lambda x: -x[1])
        if candidates:
            result["fuzzy"][target_name] = candidates[:3]
        else:
            result["unmatched_target"].append(target_name)

    result["unmatched_source"] = [
        s for s in source_fields if s not in matched_sources
    ]

    return result


def _similarity_score(target: str, source: str) -> float:
    """
    Compute a similarity score between two field names.
    Combines multiple heuristics.
    """
    t_norm = _normalize(target)
    s_norm = _normalize(source)
    t_exp = _expand_abbreviations(target)
    s_exp = _expand_abbreviations(source)

    # Exact after normalization
    if t_norm == s_norm:
        return 0.95

    # Exact after abbreviation expansion
    if t_exp == s_exp:
        return 0.85

    # One contains the other
    if t_norm in s_norm or s_norm in t_norm:
        return 0.75

    # After expansion, one contains the other
    if t_exp in s_exp or s_exp in t_exp:
        return 0.70

    # Sequence matcher (catches typos, partial overlaps)
    seq_score = SequenceMatcher(None, t_norm, s_norm).ratio()
    if seq_score > 0.6:
        return seq_score * 0.8  # scale down slightly

    return 0.0
