#!/usr/bin/env python3
"""
FathomNet Data Transformer — CLI entry point.

Converts any dataset format to FathomNet-compatible metadata.csv.

Three modes:
  [1] Known Format    — deterministic converters (COCO, YOLO, Pascal VOC, etc.)
  [2] Manual Mapping  — interactive field mapping
  [3] AI-Assisted     — Claude API analyzes the dataset

Usage:
    python cli.py                          # interactive mode
    python cli.py /path/to/dataset         # auto-detect and convert
    python cli.py /path/to/dataset --mode known --format coco_json
    python cli.py /path/to/dataset --mode ai --prompt "description of dataset"
"""

import argparse
from copy import deepcopy
import re
import shutil
import sys
import tempfile
from typing import Optional

# Load environment variables from .env file (if present)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; .env won't be auto-loaded
import os

# Add tool directory to path so modules can import each other
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.text import Text

from detectors import detect_format, sample_dataset
from converters import CONVERTER_REGISTRY
from converters.base import BaseConverter
from mapper.ai_mapper import AIMapper
from mapper.manual import run_manual_mapping
from mapper.mapping_config import MappingConfig
from transform.engine import TransformEngine
from validate.validator import validate_output
from preview.csv_preview import show_csv_preview
from preview.image_preview import generate_image_previews
from preview.report import generate_html_report

console = Console()


# ── Main Flow ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Banner
    console.print()
    console.print(Panel(
        "[bold cyan]FathomNet Data Transformer[/bold cyan]\n"
        "[dim]Convert any dataset to FathomNet-compatible format[/dim]",
        border_style="cyan",
    ))

    # Step 1: Get dataset path
    dataset_path = args.dataset or Prompt.ask("\nDataset path")
    dataset_path = os.path.expanduser(dataset_path.strip())
    if not Path(dataset_path).exists():
        console.print(f"[red]Path not found: {dataset_path}[/red]")
        sys.exit(1)

    console.print(f"\nScanning [cyan]{dataset_path}[/cyan]...")

    # Step 2: Detect format
    detection = detect_format(dataset_path)
    console.print(f"\nDetected format: [bold]{detection['format']}[/bold] "
                  f"(confidence: {detection.get('confidence', 0):.0%})")
    console.print(f"  {detection.get('details', '')}")
    console.print(f"  Images: {detection.get('image_count', 0)}")
    console.print(f"  Annotation files: {len(detection.get('annotation_files', []))}")

    # Step 3: Choose mode
    mode = args.mode or choose_mode(detection)

    # Step 4: Get user prompt (for AI mode, or if no annotation files found)
    user_prompt = args.prompt or ""
    if mode == "ai" and not user_prompt:
        if not detection.get("annotation_files"):
            console.print(
                "\n[yellow]No annotation files detected.[/yellow] "
                "A description will help the AI understand your dataset."
            )
        user_prompt = Prompt.ask(
            "\nDescribe this dataset (optional but recommended)",
            default="",
        )

    # Step 5: Run the chosen mode
    # Output: folder name placed next to this script (or absolute path if given)
    script_dir = Path(__file__).resolve().parent
    output_name = args.output or "fathomnet_output"
    final_output_dir = (
        output_name if Path(output_name).is_absolute()
        else str(script_dir / output_name)
    )
    final_review_dir = _get_review_dir(final_output_dir)
    staging_root = Path(tempfile.mkdtemp(prefix="fathomnet_transform_", dir=str(script_dir)))
    output_dir = str(staging_root / "output")
    review_dir = str(staging_root / "review")

    if mode == "known":
        result = run_known_format(
            dataset_path,
            detection,
            args,
            output_dir,
            args.submission_target,
            args.convert_images,
            images_are_crops=args.crops,
        )
    elif mode == "manual":
        result = run_manual_mode(
            dataset_path,
            detection,
            output_dir,
            args.submission_target,
            args.convert_images,
            images_are_crops=args.crops,
            mapping_path=args.mapping,
            bbox_pattern=args.bbox_pattern,
            bbox_groups=args.bbox_order,
        )
    elif mode == "ai":
        result = run_ai_mode(
            dataset_path, detection, user_prompt, output_dir,
            submission_target=args.submission_target,
            image_conversion=args.convert_images,
            images_are_crops=args.crops,
            bbox_pattern=args.bbox_pattern,
            bbox_groups=args.bbox_order,
        )
    else:
        console.print(f"[red]Unknown mode: {mode}[/red]")
        sys.exit(1)

    if result is None:
        console.print("[red]Conversion failed.[/red]")
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        sys.exit(1)

    # Step 6: Validate
    console.print("\n[bold]Running validation...[/bold]")
    validation = validate_output(
        result["engine"].records,
        image_dir=output_dir,
        submission_target=args.submission_target,
    )

    # Step 7: Preview (CSV + images)
    sample_records = result["engine"].get_sample_records(n=6, strategy="smart")
    show_csv_preview(
        sample_records,
        result["stats"],
        validation,
        dropped_items=result.get("dropped_items", []),
    )

    # Step 8: Image preview
    image_dirs = [output_dir] + _get_image_search_dirs(dataset_path, detection)
    previews = generate_image_previews(
        sample_records,
        image_dirs,
        output_dir=str(Path(review_dir) / "previews"),
        all_records=result["engine"].records,
    )

    if previews or result.get("dropped_items"):
        report_path = generate_html_report(
            previews=previews,
            sample_records=sample_records,
            stats=result["stats"],
            validation_result=validation,
            dropped_items=result.get("dropped_items", []),
            output_path=str(Path(review_dir) / "preview.html"),
            auto_open=not args.no_preview,
        )
        console.print(f"\nPreview report: [cyan]{report_path}[/cyan]")
    else:
        console.print("\n[yellow]Could not generate image previews (images not found).[/yellow]")

    # Step 9: Approve / Correct loop
    try:
        run_approval_loop(
            dataset_path, detection, result, validation, user_prompt,
            output_dir, review_dir, image_dirs,
            args.submission_target, args.convert_images,
            images_are_crops=args.crops,
            auto_open_reports=not args.no_preview,
            bbox_pattern=args.bbox_pattern,
            bbox_groups=args.bbox_order,
            final_output_dir=final_output_dir,
            final_review_dir=final_review_dir,
        )
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)


# ── Mode Selection ────────────────────────────────────────────────────────────

def choose_mode(detection: dict) -> str:
    """Interactive mode selection."""
    console.print("\n[bold]How would you like to convert?[/bold]\n")
    console.print("  [1] Known Format    — I know my format (COCO, YOLO, etc.)")
    console.print("  [2] Manual Mapping  — I'll map columns to FathomNet fields")
    console.print("  [3] AI-Assisted     — Let Claude figure it out")

    # Suggest based on detection
    fmt = detection["format"]
    if fmt in CONVERTER_REGISTRY and detection.get("confidence", 0) > 0.7:
        console.print(f"\n  [dim]Recommended: [1] Known Format ({fmt})[/dim]")
        default = "1"
    elif fmt == "unknown" or detection.get("confidence", 0) < 0.5:
        console.print(f"\n  [dim]Recommended: [3] AI-Assisted[/dim]")
        default = "3"
    else:
        default = "1"

    choice = Prompt.ask("\nSelect mode", choices=["1", "2", "3"], default=default)

    return {"1": "known", "2": "manual", "3": "ai"}[choice]


# ── Mode 1: Known Format ─────────────────────────────────────────────────────

def run_known_format(
    dataset_path: str,
    detection: dict,
    args,
    output_dir: str,
    submission_target: str,
    image_conversion: str,
    images_are_crops: bool = False,
) -> dict:
    """Run a deterministic converter for a known format."""
    fmt = args.format if args.format else detection["format"]
    loaded_config = _load_mapping_config(args.mapping) if args.mapping else None
    if loaded_config and not args.format and loaded_config.source_format in CONVERTER_REGISTRY:
        fmt = loaded_config.source_format

    if fmt not in CONVERTER_REGISTRY:
        console.print(f"\n[yellow]No built-in converter for '{fmt}'.[/yellow]")
        console.print("Available converters:")
        for name in CONVERTER_REGISTRY:
            console.print(f"  - {name}")

        fmt = Prompt.ask("Select format", choices=list(CONVERTER_REGISTRY.keys()))

    converter_class = CONVERTER_REGISTRY[fmt]
    converter = converter_class()

    # Check prerequisites
    errors = converter.validate_prerequisites(dataset_path, detection)
    if errors:
        console.print("\n[yellow]Prerequisites check:[/yellow]")
        for err in errors:
            console.print(f"  [yellow]WARNING: {err}[/yellow]")
        if not Confirm.ask("Continue anyway?", default=True):
            return None

    # Show detected fields and ask about overrides
    fields = converter.get_field_names(dataset_path, detection)
    if fields:
        preview = ", ".join(fields[:15])
        suffix = f" ... and {len(fields) - 15} more" if len(fields) > 15 else ""
        console.print(f"\n[dim]Fields found ({len(fields)}): {preview}{suffix}[/dim]")

    # Ask about field overrides
    config = loaded_config or MappingConfig(source_format=fmt)
    if loaded_config:
        if not config.source_format or config.source_format == "unknown":
            config.source_format = fmt
        console.print(f"\n[green]Loaded mapping config:[/green] {args.mapping}")
    elif Confirm.ask("\nCustomize field mappings or metadata columns?", default=False):
        config = _interactive_field_overrides(converter, dataset_path, detection)

    # Ask about class exclusions
    exclude = Prompt.ask(
        "Classes to exclude (comma-separated, or Enter to skip)",
        default="",
    )
    if exclude.strip():
        config.exclude_concepts = [c.strip() for c in exclude.split(",")]

    if images_are_crops:
        config.extra["images_are_crops"] = True
    _apply_bbox_options(config, args.bbox_pattern, args.bbox_order, fmt)

    # Run conversion
    console.print("\n[bold]Running conversion...[/bold]")
    engine = TransformEngine(output_dir=output_dir)
    stats = engine.run(
        converter,
        dataset_path,
        detection,
        config,
        image_dirs=_get_image_search_dirs(dataset_path, detection),
        submission_target=submission_target,
        image_conversion=image_conversion,
    )

    console.print(f"[green]Done![/green] {stats['stats']['total_written']} records written.")
    if stats['stats'].get('jpeg_renamed'):
        console.print(
            f"  [dim]Renamed {stats['stats']['jpeg_renamed']} .jpeg files to .jpg "
            f"(NOAA-NCEI requires .jpg).[/dim]"
        )

    return {
        "engine": engine,
        "stats": stats["stats"],
        "config": config,
        "dropped_items": stats.get("dropped_items", []),
    }


def _interactive_field_overrides(
    converter: BaseConverter, dataset_path: str, detection: dict
) -> MappingConfig:
    """Let user override field mappings for a known-format converter."""
    from utils.fuzzy_match import fuzzy_match_fields
    from fathomnet_schema import OPTIONAL_FIELDS, REQUIRED_FIELDS

    fields = converter.get_field_names(dataset_path, detection)
    field_samples = converter.get_field_samples(dataset_path, detection)
    field_choices = _detected_field_candidates(fields)
    default_field_map = converter.get_default_field_map(dataset_path, detection)
    config = MappingConfig(source_format=converter.format_name)

    if field_choices:
        _print_detected_field_reference(
            field_choices,
            field_samples,
            title="Detected source fields you can map from:",
        )

    console.print("\nFor each FathomNet field, enter the matching field number or name.")
    console.print("Press Enter to keep the shown default.\n")

    for target, desc in REQUIRED_FIELDS.items():
        default_source = default_field_map.get(target, "")
        default_label = _field_default_label(default_source, field_choices)
        override = Prompt.ask(
            f"  {target} ({desc}){default_label}",
            default="",
            show_default=False,
        )
        source = _resolve_field_reference(override, field_choices) or default_source
        if source:
            config.field_map[target] = {"source": source}

    if Confirm.ask("\nMap optional FathomNet fields?", default=False):
        console.print("Enter a source field number or name. Press Enter to skip.\n")
        for target, desc in OPTIONAL_FIELDS.items():
            default_source = default_field_map.get(target, "")
            default_label = _field_default_label(default_source, field_choices)
            override = Prompt.ask(
                f"  {target} ({desc}){default_label}",
                default="",
                show_default=False,
            )
            source = _resolve_field_reference(override, field_choices) or default_source
            if source:
                config.field_map[target] = {"source": source}

    _interactive_extra_columns(fields, config, field_samples)

    return config


def _interactive_extra_columns(
    fields: list[str],
    config: MappingConfig,
    field_samples: Optional[dict[str, list[str]]] = None,
):
    """Let the user include detected source fields as extra CSV columns."""
    field_samples = field_samples or {}
    candidates = _extra_column_candidates(fields, config)
    if not candidates:
        return

    if not Confirm.ask("\nAdd any other detected fields as extra CSV columns?", default=False):
        return

    _print_detected_field_reference(
        candidates,
        field_samples,
        title="Detected fields available for extra columns:",
        show_default_column=True,
    )

    selection = Prompt.ask(
        "\nExtra fields to include (comma-separated numbers or names)",
        default="",
    )
    if not selection.strip():
        return

    selected = _parse_extra_field_selection(selection, candidates)
    if not selected:
        console.print("[yellow]No matching extra fields selected.[/yellow]")
        return

    from fathomnet_schema import ALL_FIELDS

    extra_columns = []
    used_columns = set(ALL_FIELDS)
    for source in selected:
        default_name = _default_extra_column_name(source)
        column = Prompt.ask(
            f"  CSV column name for {source}",
            default=default_name,
        ).strip() or default_name
        column = _sanitize_extra_column_name(column)
        base_column = column
        i = 2
        while column in used_columns:
            column = f"{base_column}_{i}"
            i += 1
        used_columns.add(column)
        extra_columns.append({"source": source, "column": column})

    config.extra["extra_columns"] = extra_columns


def _extra_column_candidates(fields: list[str], config: MappingConfig) -> list[str]:
    """Return detected source fields that are not already mapped."""
    mapped_sources = set()
    for mapping in config.field_map.values():
        if isinstance(mapping, dict) and mapping.get("source"):
            mapped_sources.add(str(mapping["source"]))
        elif isinstance(mapping, str):
            mapped_sources.add(mapping)

    candidates = []
    for field_name in _detected_field_candidates(fields):
        if field_name in mapped_sources:
            continue
        if "." not in field_name:
            continue
        candidates.append(field_name)
    return sorted(dict.fromkeys(candidates))


def _detected_field_candidates(fields: list[str]) -> list[str]:
    """Return user-map-able detected source fields."""
    skip_names = {"images", "annotations", "categories", "info", "licenses", "parts"}
    return sorted(
        dict.fromkeys(
            field_name for field_name in fields
            if field_name and field_name not in skip_names
        )
    )


def _print_detected_field_reference(
    fields: list[str],
    field_samples: dict[str, list[str]],
    *,
    title: str,
    show_default_column: bool = False,
):
    console.print(f"\n{title}")
    for i, field_name in enumerate(fields, 1):
        sample_text = _format_sample_values(field_samples.get(field_name, []))
        sample_suffix = f" [dim]e.g. {sample_text}[/dim]" if sample_text else ""
        if show_default_column:
            console.print(
                f"  [{i}] {field_name}  "
                f"[dim]→ {_default_extra_column_name(field_name)}[/dim]{sample_suffix}"
            )
        else:
            console.print(f"  [{i}] {field_name}{sample_suffix}")


def _resolve_field_reference(value: str, fields: list[str]) -> str:
    token = value.strip()
    if not token:
        return ""
    if token.isdigit():
        index = int(token) - 1
        if 0 <= index < len(fields):
            return fields[index]
        console.print(f"[yellow]No detected field numbered {token}; skipping.[/yellow]")
        return ""
    return token


def _field_default_label(source: str, fields: list[str]) -> str:
    if not source:
        return ""
    try:
        number = fields.index(source) + 1
        return f" [dim](default: [{number}] {source})[/dim]"
    except ValueError:
        return f" [dim](default: {source})[/dim]"


def _parse_extra_field_selection(selection: str, candidates: list[str]) -> list[str]:
    selected = []
    for raw_token in selection.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if token.isdigit():
            index = int(token) - 1
            if 0 <= index < len(candidates):
                selected.append(candidates[index])
            continue
        if token in candidates:
            selected.append(token)

    return list(dict.fromkeys(selected))


def _default_extra_column_name(source: str) -> str:
    return _sanitize_extra_column_name(source)


def _sanitize_extra_column_name(value: str) -> str:
    column = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip()).strip("_")
    if not column:
        column = "extra"
    if column[0].isdigit():
        column = f"field_{column}"
    return column


def _format_sample_values(values: list[str], max_len: int = 80) -> str:
    if not values:
        return ""
    text = ", ".join(str(v) for v in values[:3])
    if len(values) > 3:
        text += ", ..."
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip() + "..."
    return text


# ── Mode 2: Manual Mapping ───────────────────────────────────────────────────

def run_manual_mode(
    dataset_path: str,
    detection: dict,
    output_dir: str,
    submission_target: str,
    image_conversion: str,
    images_are_crops: bool = False,
    mapping_path: Optional[str] = None,
    bbox_pattern: Optional[str] = None,
    bbox_groups: Optional[dict] = None,
) -> dict:
    """Run manual field mapping mode."""
    config = _load_mapping_config(mapping_path) if mapping_path else None
    if config:
        console.print(f"\n[green]Loaded mapping config:[/green] {mapping_path}")
    else:
        # We need to figure out the source fields
        # Try to extract from detection or sample
        source_fields = _extract_source_fields(dataset_path, detection)

        if not source_fields:
            console.print("[yellow]Could not auto-detect field names.[/yellow]")
            fields_input = Prompt.ask("Enter your field names (comma-separated)")
            source_fields = [f.strip() for f in fields_input.split(",")]

        config = run_manual_mapping(source_fields, detection["format"])

    if images_are_crops:
        config.extra["images_are_crops"] = True

    # For manual mode, we use a generic CSV converter or the detected format's converter
    fmt = config.source_format if config.source_format in CONVERTER_REGISTRY else detection["format"]
    _apply_bbox_options(config, bbox_pattern, bbox_groups, fmt)
    if fmt in CONVERTER_REGISTRY:
        converter = CONVERTER_REGISTRY[fmt]()
    else:
        console.print("[yellow]No built-in converter for this format. Trying CSV...[/yellow]")
        # TODO: implement a generic CSV converter
        console.print("[red]Generic CSV converter not yet implemented.[/red]")
        return None

    console.print("\n[bold]Running conversion...[/bold]")
    engine = TransformEngine(output_dir=output_dir)
    stats = engine.run(
        converter,
        dataset_path,
        detection,
        config,
        image_dirs=_get_image_search_dirs(dataset_path, detection),
        submission_target=submission_target,
        image_conversion=image_conversion,
    )

    console.print(f"[green]Done![/green] {stats['stats']['total_written']} records written.")
    if stats['stats'].get('jpeg_renamed'):
        console.print(
            f"  [dim]Renamed {stats['stats']['jpeg_renamed']} .jpeg files to .jpg "
            f"(NOAA-NCEI requires .jpg).[/dim]"
        )
    return {
        "engine": engine,
        "stats": stats["stats"],
        "config": config,
        "dropped_items": stats.get("dropped_items", []),
    }


# ── Mode 3: AI-Assisted ──────────────────────────────────────────────────────

def run_ai_mode(
    dataset_path: str,
    detection: dict,
    user_prompt: str,
    output_dir: str,
    correction_history: list = None,
    submission_target: str = "noaa-ncei",
    image_conversion: str = "none",
    images_are_crops: bool = False,
    bbox_pattern: Optional[str] = None,
    bbox_groups: Optional[dict] = None,
) -> dict:
    """Run AI-assisted conversion using Claude API."""
    correction_history = correction_history or []

    console.print("\n[bold]Analyzing dataset with Claude...[/bold]")

    # Sample the dataset
    sample = sample_dataset(dataset_path, detection)

    # Call Claude
    mapper = AIMapper()
    ai_result = mapper.analyze(
        sample=sample,
        detection_result=detection,
        user_prompt=user_prompt,
        correction_history=correction_history,
    )

    # Check for errors
    if "error" in ai_result:
        if ai_result["error"] == "no_api_key":
            console.print(f"\n[yellow]{ai_result['message']}[/yellow]")
            console.print()
            fmt = detection.get("format", "unknown")
            if fmt != "unknown":
                console.print(
                    f"[cyan]Tip:[/cyan] Your dataset was detected as [bold]{fmt}[/bold]. "
                    f"Try running with:\n"
                    f"  python3 cli.py {dataset_path} --mode known --format {fmt}"
                )
            else:
                console.print(
                    "[cyan]Tip:[/cyan] You can also use --mode manual to map fields yourself."
                )
            return None
        console.print(f"[red]AI analysis error: {ai_result['error']}[/red]")
        if "raw_response" in ai_result:
            console.print(f"[dim]{ai_result['raw_response'][:500]}[/dim]")
        return None

    # Show AI's analysis
    console.print(f"\n[bold cyan]AI Analysis:[/bold cyan]")
    console.print(f"  Format: {ai_result.get('source_format', 'unknown')}")
    console.print(f"  Confidence: {ai_result.get('confidence', 0):.0%}")
    if ai_result.get("notes"):
        console.print(f"  Notes: {ai_result['notes']}")
    if ai_result.get("conversion_steps"):
        console.print("  Steps:")
        for step in ai_result["conversion_steps"]:
            console.print(f"    - {step}")

    # Show field mapping
    console.print("\n  [bold]Field mapping:[/bold]")
    for field, mapping in ai_result.get("field_map", {}).items():
        source = mapping.get("source", "?") if isinstance(mapping, dict) else mapping
        transform = mapping.get("transform", "") if isinstance(mapping, dict) else ""
        extra = f" [dim]({transform})[/dim]" if transform else ""
        console.print(f"    {field} <- {source}{extra}")

    # Ask about any AI questions. If the user provides answers, we re-call
    # Claude with the new context so the mapping actually reflects the
    # answers (otherwise the answers would be silently ignored when the
    # user clicks "Proceed").
    questions = ai_result.get("questions", [])
    if questions:
        console.print("\n  [yellow]Questions from AI:[/yellow]")
        for q in questions:
            console.print(f"    - {q}")
        answer = Prompt.ask("\nAnswer (or press Enter to skip)", default="")
        if answer:
            user_prompt += f"\n\nAdditional context: {answer}"
            console.print(
                "\n[bold]Re-analyzing with your answers...[/bold] "
                "[dim](one extra API call)[/dim]"
            )
            ai_result = mapper.analyze(
                sample=sample,
                detection_result=detection,
                user_prompt=user_prompt,
                correction_history=correction_history,
            )
            if "error" in ai_result:
                console.print(
                    f"[red]Re-analysis failed: {ai_result.get('error')}[/red]"
                )
                return None

            # Show the updated analysis
            console.print(f"\n[bold cyan]Updated AI Analysis:[/bold cyan]")
            console.print(f"  Format: {ai_result.get('source_format', 'unknown')}")
            console.print(f"  Confidence: {ai_result.get('confidence', 0):.0%}")
            if ai_result.get("notes"):
                console.print(f"  Notes: {ai_result['notes']}")
            console.print("\n  [bold]Updated field mapping:[/bold]")
            for field, mapping in ai_result.get("field_map", {}).items():
                source = (
                    mapping.get("source", "?") if isinstance(mapping, dict) else mapping
                )
                transform = (
                    mapping.get("transform", "") if isinstance(mapping, dict) else ""
                )
                extra = f" [dim]({transform})[/dim]" if transform else ""
                console.print(f"    {field} <- {source}{extra}")

    unsupported_sources = _unsupported_ai_mapping_sources(ai_result)
    if unsupported_sources:
        console.print("\n[red]AI proposed mapping sources this tool cannot execute:[/red]")
        for field, source in unsupported_sources:
            console.print(f"  [red]{field}[/red] <- {source}")
        console.print(
            "[yellow]Try a known-format conversion, manual mapping, or ask AI mode "
            "to use detected fields/filename_regex only.[/yellow]"
        )
        return None

    # Confirm mapping
    if not Confirm.ask("\nProceed with this mapping?", default=True):
        console.print("[yellow]Mapping rejected. You can adjust and retry.[/yellow]")
        return None

    # Convert AI result to mapping config
    config = mapper.result_to_config(ai_result)
    if images_are_crops:
        config.extra["images_are_crops"] = True

    # Try to find an appropriate converter
    fmt = ai_result.get("source_format", detection["format"])
    _apply_bbox_options(config, bbox_pattern, bbox_groups, fmt)
    if fmt in CONVERTER_REGISTRY:
        converter = CONVERTER_REGISTRY[fmt]()
    elif detection["format"] in CONVERTER_REGISTRY:
        converter = CONVERTER_REGISTRY[detection["format"]]()
    else:
        console.print(f"[yellow]No built-in converter for '{fmt}'. "
                       f"AI-generated mapping may not fully work yet.[/yellow]")
        return None

    # Run conversion
    console.print("\n[bold]Running conversion...[/bold]")
    engine = TransformEngine(output_dir=output_dir)
    stats = engine.run(
        converter,
        dataset_path,
        detection,
        config,
        image_dirs=_get_image_search_dirs(dataset_path, detection),
        submission_target=submission_target,
        image_conversion=image_conversion,
    )

    console.print(f"[green]Done![/green] {stats['stats']['total_written']} records written.")
    if stats['stats'].get('jpeg_renamed'):
        console.print(
            f"  [dim]Renamed {stats['stats']['jpeg_renamed']} .jpeg files to .jpg "
            f"(NOAA-NCEI requires .jpg).[/dim]"
        )

    return {
        "engine": engine,
        "stats": stats["stats"],
        "config": config,
        "dropped_items": stats.get("dropped_items", []),
        "ai_result": ai_result,
        "user_prompt": user_prompt,
    }


# ── Approval / Correction Loop ───────────────────────────────────────────────

def run_approval_loop(
    dataset_path: str,
    detection: dict,
    result: dict,
    validation: dict,
    user_prompt: str,
    output_dir: str,
    review_dir: str,
    image_dirs: list,
    submission_target: str,
    image_conversion: str,
    images_are_crops: bool = False,
    auto_open_reports: bool = True,
    bbox_pattern: Optional[str] = None,
    bbox_groups: Optional[dict] = None,
    final_output_dir: Optional[str] = None,
    final_review_dir: Optional[str] = None,
    max_attempts: int = 4,
):
    """
    Interactive approve/correct loop.

    User can approve the output, ask for more samples, or describe
    what's wrong for the AI to adjust.
    """
    attempt = 0
    correction_history = []

    while attempt < max_attempts:
        console.print()
        console.print("[bold]What would you like to do?[/bold]")
        console.print(r"  \[y] Looks good — export final CSV")
        console.print(r"  \[n] Something's wrong — describe the issue")
        console.print(r"  \[m] Show more sample images")
        console.print(r"  \[c] Show full class distribution")
        if _current_crop_mode(result, images_are_crops):
            console.print(r"  \[t] Treat images as full frames and rerun")
        else:
            console.print(r"  \[t] Treat images as pre-cropped and rerun")
        console.print(r"  \[r] Show current mapping config")
        console.print(r"  \[q] Quit without exporting")

        choice = Prompt.ask(
            "",
            choices=["y", "n", "m", "c", "t", "r", "q"],
            default="y",
        )

        if choice == "y":
            final_output = final_output_dir or output_dir
            final_review = final_review_dir or review_dir
            _finalize_staged_output(output_dir, review_dir, final_output, final_review)
            output_path = str(Path(final_output) / "metadata.csv")
            result["stats"]["output_path"] = output_path
            console.print(f"\n[bold green]Exported![/bold green] {output_path}")
            console.print(f"  {result['stats']['total_written']} annotations")
            console.print(f"  {validation['summary'].get('unique_images', '?')} images")
            console.print(f"  {validation['summary'].get('unique_concepts', '?')} concepts")
            console.print(f"  Review report: {Path(final_review) / 'preview.html'}")
            return

        elif choice == "n":
            correction = Prompt.ask(
                "\nDescribe what's wrong"
            )
            correction_history.append({
                "mapping": result.get("config", MappingConfig()).to_json()
                           if hasattr(result.get("config", None), "to_json")
                           else str(result.get("config")),
                "correction": correction,
            })

            # Re-run AI mode with correction
            console.print("\n[bold]Re-analyzing with your feedback...[/bold]")
            new_prompt = user_prompt + f"\n\nUser correction: {correction}"
            _clear_staged_output(output_dir)
            new_result = run_ai_mode(
                dataset_path,
                detection,
                new_prompt,
                output_dir,
                correction_history,
                submission_target=submission_target,
                image_conversion=image_conversion,
                images_are_crops=images_are_crops,
                bbox_pattern=bbox_pattern,
                bbox_groups=bbox_groups,
            )

            if new_result is None:
                console.print("[yellow]Could not re-run. Try a different description.[/yellow]")
                continue

            result = new_result
            images_are_crops = _current_crop_mode(result, images_are_crops)
            validation = validate_output(
                result["engine"].records,
                image_dir=output_dir,
                submission_target=submission_target,
            )
            sample_records = result["engine"].get_sample_records(n=6, strategy="smart")
            show_csv_preview(
                sample_records,
                result["stats"],
                validation,
                dropped_items=result.get("dropped_items", []),
            )

            previews = generate_image_previews(
                sample_records,
                image_dirs,
                output_dir=str(Path(review_dir) / "previews"),
                all_records=result["engine"].records,
            )
            if previews or result.get("dropped_items"):
                generate_html_report(
                    previews=previews,
                    sample_records=sample_records,
                    stats=result["stats"],
                    validation_result=validation,
                    dropped_items=result.get("dropped_items", []),
                    output_path=str(Path(review_dir) / "preview.html"),
                    auto_open=auto_open_reports,
                )

            attempt += 1

        elif choice == "m":
            sample_records = result["engine"].get_sample_records(n=6, strategy="random")
            show_csv_preview(
                sample_records,
                result["stats"],
                validation,
                dropped_items=result.get("dropped_items", []),
            )
            previews = generate_image_previews(
                sample_records,
                image_dirs,
                output_dir=str(Path(review_dir) / "previews"),
                all_records=result["engine"].records,
            )
            if previews or result.get("dropped_items"):
                generate_html_report(
                    previews=previews,
                    sample_records=sample_records,
                    stats=result["stats"],
                    validation_result=validation,
                    dropped_items=result.get("dropped_items", []),
                    output_path=str(Path(review_dir) / "preview.html"),
                    auto_open=auto_open_reports,
                )

        elif choice == "c":
            _show_full_distribution(result["stats"])

        elif choice == "t":
            images_are_crops = not _current_crop_mode(result, images_are_crops)
            mode_label = "pre-cropped images" if images_are_crops else "full-frame images"
            console.print(f"\n[bold]Re-running conversion for {mode_label}...[/bold]")
            _clear_staged_output(output_dir)
            new_result = _rerun_current_conversion(
                dataset_path=dataset_path,
                detection=detection,
                result=result,
                output_dir=output_dir,
                submission_target=submission_target,
                image_conversion=image_conversion,
                images_are_crops=images_are_crops,
            )
            if new_result is None:
                images_are_crops = not images_are_crops
                continue

            result = new_result
            validation = _refresh_conversion_review(
                result=result,
                output_dir=output_dir,
                review_dir=review_dir,
                image_dirs=image_dirs,
                submission_target=submission_target,
                strategy="smart",
                auto_open_reports=auto_open_reports,
            )

        elif choice == "r":
            config = result.get("config")
            if config and hasattr(config, "to_json"):
                console.print(f"\n[dim]{config.to_json()}[/dim]")
            else:
                console.print(f"\n[dim]{config}[/dim]")

        elif choice == "q":
            console.print(
                "[yellow]Exiting without export. Temporary conversion files were discarded.[/yellow]"
            )
            return

    console.print(
        f"\n[yellow]Reached {max_attempts} correction attempts. "
        f"Try editing the mapping config manually or use a different mode.[/yellow]"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_mapping_config(mapping_path: Optional[str]) -> Optional[MappingConfig]:
    """Load a saved mapping config, returning None if it cannot be used."""
    if not mapping_path:
        return None
    try:
        return MappingConfig.load(mapping_path)
    except OSError as e:
        console.print(f"[red]Could not read mapping config {mapping_path}: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Could not load mapping config {mapping_path}: {e}[/red]")
    return None


def _finalize_staged_output(
    staged_output_dir: str,
    staged_review_dir: str,
    final_output_dir: str,
    final_review_dir: str,
):
    """Export staged conversion artifacts to the requested final locations."""
    _copy_staged_dir(Path(staged_output_dir), Path(final_output_dir))
    if Path(staged_review_dir).exists():
        _copy_staged_dir(Path(staged_review_dir), Path(final_review_dir))
    _clean_submission_artifacts(final_output_dir)


def _clear_staged_output(output_dir: str):
    """Remove staged output before a rerun inside the same approval session."""
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)


def _copy_staged_dir(source: Path, target: Path):
    if not source.exists():
        return
    if target.exists() and not target.is_dir():
        raise OSError(f"Cannot export to {target}: path exists and is not a directory")
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        destination = target / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, destination)


def _apply_bbox_options(
    config: MappingConfig,
    bbox_pattern: Optional[str],
    bbox_groups: Optional[dict],
    fmt: str,
):
    """Attach folder-encoded bbox filename options to a mapping config."""
    if not bbox_pattern and not bbox_groups:
        return
    if fmt != "folder_encoded":
        console.print(
            "[yellow]Ignoring --bbox-pattern/--bbox-order because the selected "
            f"format is '{fmt}', not folder_encoded.[/yellow]"
        )
        return
    if bbox_pattern:
        config.extra["bbox_pattern"] = bbox_pattern
    if bbox_groups:
        config.extra["bbox_groups"] = bbox_groups


def _unsupported_ai_mapping_sources(ai_result: dict) -> list[tuple[str, str]]:
    """Return AI mapping sources that the transform engine cannot execute."""
    unsupported = []
    for field, mapping in ai_result.get("field_map", {}).items():
        source = mapping.get("source") if isinstance(mapping, dict) else mapping
        if not isinstance(source, str):
            continue
        source = source.strip()
        if source.startswith(("computed:", "constant:")):
            unsupported.append((field, source))
    return unsupported


def _current_crop_mode(result: dict, fallback: bool = False) -> bool:
    """Return whether the current conversion treats images as pre-cropped."""
    config = result.get("config") if result else None
    extra = getattr(config, "extra", None)
    if isinstance(extra, dict):
        return bool(extra.get("images_are_crops", fallback))
    return fallback


def _rerun_current_conversion(
    *,
    dataset_path: str,
    detection: dict,
    result: dict,
    output_dir: str,
    submission_target: str,
    image_conversion: str,
    images_are_crops: bool,
) -> Optional[dict]:
    """Re-run the current converter/config with only crop mode changed."""
    existing_config = result.get("config")
    if isinstance(existing_config, MappingConfig):
        config = deepcopy(existing_config)
    else:
        config = MappingConfig(source_format=detection.get("format", "unknown"))

    if images_are_crops:
        config.extra["images_are_crops"] = True
    else:
        config.extra.pop("images_are_crops", None)

    fmt = None
    ai_result = result.get("ai_result")
    if isinstance(ai_result, dict):
        fmt = ai_result.get("source_format")
    if not fmt or fmt == "unknown":
        fmt = config.source_format
    if not fmt or fmt == "unknown":
        fmt = detection.get("format", "unknown")
    if fmt not in CONVERTER_REGISTRY and detection.get("format") in CONVERTER_REGISTRY:
        fmt = detection["format"]
    if fmt not in CONVERTER_REGISTRY:
        console.print(f"[red]Cannot rerun: no built-in converter for '{fmt}'.[/red]")
        return None

    converter = CONVERTER_REGISTRY[fmt]()
    engine = TransformEngine(output_dir=output_dir)
    stats = engine.run(
        converter,
        dataset_path,
        detection,
        config,
        image_dirs=_get_image_search_dirs(dataset_path, detection),
        submission_target=submission_target,
        image_conversion=image_conversion,
    )

    console.print(f"[green]Done![/green] {stats['stats']['total_written']} records written.")
    if stats["stats"].get("jpeg_renamed"):
        console.print(
            f"  [dim]Renamed {stats['stats']['jpeg_renamed']} .jpeg files to .jpg "
            f"(NOAA-NCEI requires .jpg).[/dim]"
        )

    new_result = dict(result)
    new_result.update({
        "engine": engine,
        "stats": stats["stats"],
        "config": config,
        "dropped_items": stats.get("dropped_items", []),
    })
    return new_result


def _refresh_conversion_review(
    *,
    result: dict,
    output_dir: str,
    review_dir: str,
    image_dirs: list,
    submission_target: str,
    strategy: str,
    auto_open_reports: bool = True,
) -> dict:
    """Rebuild validation, table preview, image previews, and HTML report."""
    validation = validate_output(
        result["engine"].records,
        image_dir=output_dir,
        submission_target=submission_target,
    )
    sample_records = result["engine"].get_sample_records(n=6, strategy=strategy)
    show_csv_preview(
        sample_records,
        result["stats"],
        validation,
        dropped_items=result.get("dropped_items", []),
    )
    previews = generate_image_previews(
        sample_records,
        image_dirs,
        output_dir=str(Path(review_dir) / "previews"),
        all_records=result["engine"].records,
    )
    if previews or result.get("dropped_items"):
        report_path = generate_html_report(
            previews=previews,
            sample_records=sample_records,
            stats=result["stats"],
            validation_result=validation,
            dropped_items=result.get("dropped_items", []),
            output_path=str(Path(review_dir) / "preview.html"),
            auto_open=auto_open_reports,
        )
        console.print(f"\nPreview report: [cyan]{report_path}[/cyan]")
    return validation


def _extract_source_fields(dataset_path: str, detection: dict) -> list[str]:
    """Try to extract field names from the dataset."""
    fmt = detection["format"]
    if fmt in CONVERTER_REGISTRY:
        converter = CONVERTER_REGISTRY[fmt]()
        return converter.get_field_names(dataset_path, detection)
    if detection.get("columns"):
        return detection["columns"]
    return []


def _get_review_dir(output_dir: str) -> str:
    """Keep previews outside the folder users will zip for submission."""
    output_path = Path(output_dir)
    return str(output_path.with_name(f"{output_path.name}_review"))


def _clean_submission_artifacts(output_dir: str):
    """Remove generated review files from old runs in the submission folder."""
    output_path = Path(output_dir)
    for artifact in (output_path / "preview.html", output_path / "previews"):
        if artifact.is_dir():
            shutil.rmtree(artifact)
        elif artifact.exists():
            artifact.unlink()


def _get_image_search_dirs(dataset_path: str, detection: dict) -> list[str]:
    """Build a list of directories where images might be found."""
    dataset_path = Path(dataset_path)
    dataset_root = dataset_path.parent if dataset_path.is_file() else dataset_path
    dirs = [str(dataset_root)]
    if not dataset_root.is_dir():
        return dirs

    # Common image subdirectories
    for subdir in ["images", "image", "imgs", "img", "photos", "JPEGImages"]:
        candidate = dataset_root / subdir
        if candidate.exists():
            dirs.append(str(candidate))
        # Also check one level deeper (e.g., coco_train_data/images/)
        for child in dataset_root.iterdir():
            if child.is_dir():
                candidate = child / subdir
                if candidate.exists():
                    dirs.append(str(candidate))

    return dirs


def _show_full_distribution(stats: dict):
    """Show the full concept distribution."""
    concepts = stats.get("concepts", {})
    if not concepts:
        console.print("[yellow]No concept data available.[/yellow]")
        return

    console.print(f"\n[bold]Full class distribution ({len(concepts)} concepts):[/bold]\n")
    max_count = max(concepts.values()) if concepts else 1
    for concept, count in sorted(concepts.items(), key=lambda x: -x[1]):
        bar_len = min(int(count / max_count * 40), 40)
        bar = "█" * bar_len
        console.print(f"  {concept:35s} {count:>6d}  {bar}")


def _parse_bbox_order_arg(value: str) -> dict[str, int]:
    """Parse x,y,width,height style capture-group order from the CLI."""
    aliases = {
        "x": "x",
        "y": "y",
        "w": "width",
        "width": "width",
        "h": "height",
        "height": "height",
    }
    tokens = [
        token.strip().lower()
        for token in re.split(r"[,\s]+", value.strip())
        if token.strip()
    ]
    fields = []
    for token in tokens:
        if token not in aliases:
            raise argparse.ArgumentTypeError(
                f"unknown bbox-order field '{token}'. Use x,y,width,height."
            )
        fields.append(aliases[token])

    required = {"x", "y", "width", "height"}
    if len(fields) != 4 or set(fields) != required:
        raise argparse.ArgumentTypeError(
            "bbox-order must contain x, y, width, and height exactly once"
        )
    return {field: index for index, field in enumerate(fields, start=1)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="FathomNet Data Transformer — Convert datasets to FathomNet format"
    )
    parser.add_argument("dataset", nargs="?", help="Path to dataset directory or file")
    parser.add_argument("--mode", choices=["known", "manual", "ai"],
                        help="Conversion mode")
    parser.add_argument("--format", help="Dataset format (for known mode)")
    parser.add_argument("--prompt", help="Dataset description (for AI mode)")
    parser.add_argument("--output", "-o",
                        help="Output folder name (created next to cli.py) "
                             "or an absolute path. Default: fathomnet_output")
    parser.add_argument("--mapping", help="Path to saved mapping config (YAML, known/manual modes)")
    parser.add_argument("--submission-target",
                        choices=["noaa-ncei", "self-hosted"],
                        default="noaa-ncei",
                        help="Submission rules to validate against. "
                             "Default: noaa-ncei")
    parser.add_argument("--convert-images",
                        choices=["none", "png", "jpg"],
                        default="none",
                        help="For NOAA-NCEI submissions, convert unsupported "
                             "local image formats to png or jpg. Default: none")
    parser.add_argument("--no-preview", action="store_true",
                        help="Don't auto-open preview in browser")
    parser.add_argument("--crops", action="store_true",
                        help="For folder-encoded datasets: treat images as "
                             "pre-cropped to the bounding box region. Sets the "
                             "submission box to (0, 0, w, h) and stores the "
                             "original filename coordinates as source_x/source_y "
                             "columns in the CSV.")
    parser.add_argument("--bbox-pattern",
                        help="For folder-encoded datasets: custom filename regex "
                             "with capture groups for bbox values.")
    parser.add_argument("--bbox-order",
                        type=_parse_bbox_order_arg,
                        metavar="ORDER",
                        help="For folder-encoded datasets: capture-group order "
                             "for --bbox-pattern or built-in numeric patterns, "
                             "for example x,y,width,height or width,height,x,y.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
