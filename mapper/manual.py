"""
Manual field mapping mode.

Interactive CLI flow where the user maps their fields to FathomNet fields.
Uses fuzzy matching to suggest candidates.
"""

from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

from fathomnet_schema import REQUIRED_FIELDS, OPTIONAL_FIELDS
from utils.fuzzy_match import fuzzy_match_fields
from .mapping_config import MappingConfig, FieldMapping

console = Console()


def run_manual_mapping(
    source_fields: list[str],
    source_format: str = "unknown",
) -> MappingConfig:
    """
    Interactive manual field mapping session.

    Args:
        source_fields: list of field names from the uploaded dataset
        source_format: detected format name

    Returns:
        MappingConfig with user-specified field mappings
    """
    config = MappingConfig(source_format=source_format)

    console.print("\n[bold cyan]Manual Field Mapping[/bold cyan]")
    console.print(f"Your dataset has {len(source_fields)} fields:\n")

    # Show source fields
    for i, f in enumerate(source_fields, 1):
        console.print(f"  [{i}] {f}")
    console.print()

    # Run fuzzy matching for suggestions
    match_result = fuzzy_match_fields(source_fields)

    # Map required fields
    console.print("[bold]Required FathomNet fields:[/bold]\n")

    for target_field, description in REQUIRED_FIELDS.items():
        mapping = _map_single_field(
            target_field, description, source_fields, match_result, required=True
        )
        if mapping:
            config.field_map[target_field] = mapping

    # Ask about optional fields
    console.print("\n[bold]Optional fields[/bold] (press Enter to skip each):\n")

    for target_field, description in OPTIONAL_FIELDS.items():
        mapping = _map_single_field(
            target_field, description, source_fields, match_result, required=False
        )
        if mapping:
            config.field_map[target_field] = mapping

    # Ask about coordinate format if x/y mapping suggests xyxy
    console.print()
    coord_format = Prompt.ask(
        "Coordinate format",
        choices=["xywh", "xyxy", "cxcywh", "cxcywh_abs"],
        default="xywh",
    )
    config.coordinate_format = coord_format

    # Ask about class exclusions
    exclude = Prompt.ask(
        "\nClasses to exclude (comma-separated, or Enter to skip)",
        default="",
    )
    if exclude.strip():
        config.exclude_concepts = [c.strip() for c in exclude.split(",")]

    return config


def _map_single_field(
    target_field: str,
    description: str,
    source_fields: list[str],
    match_result: dict,
    required: bool,
) -> Optional[dict]:
    """Prompt the user to map a single field."""
    # Check if we have a suggestion
    suggestion = None
    suggestion_type = None

    if target_field in match_result["exact"]:
        suggestion = match_result["exact"][target_field]
        suggestion_type = "exact match"
    elif target_field in match_result["alias"]:
        suggestion = match_result["alias"][target_field][0]
        suggestion_type = f"known alias of '{match_result['alias'][target_field][1]}'"
    elif target_field in match_result["fuzzy"]:
        top = match_result["fuzzy"][target_field][0]
        suggestion = top[0]
        suggestion_type = f"fuzzy match ({top[1]:.0%})"

    # Build prompt
    prompt_text = f"  {target_field}"
    if suggestion:
        prompt_text += f" [dim](suggested: {suggestion} — {suggestion_type})[/dim]"

    default = suggestion if suggestion else ("" if not required else None)

    response = Prompt.ask(
        prompt_text,
        default=default,
        show_default=True,
    )

    if not response or response.strip() == "":
        if required:
            console.print(f"    [red]WARNING: {target_field} is required![/red]")
        return None

    return {"source": response.strip()}
