"""
CSV preview — shows a formatted table of sample records in the terminal.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from fathomnet_schema import REQUIRED_FIELDS

console = Console()


def show_csv_preview(
    records: list[dict],
    stats: dict,
    validation_result: dict,
    dropped_items: list[dict] | None = None,
    max_rows: int = 8,
):
    """
    Display a formatted preview of the output CSV in the terminal.

    Args:
        records: sample records to display
        stats: transformation stats
        validation_result: output from validate_output()
        dropped_items: skipped annotations/images to summarize
        max_rows: maximum rows to show
    """
    if not records:
        console.print("[red]No records to preview.[/red]")
        return

    # ── CSV Table ────────────────────────────────────────────────────────

    console.print()
    console.print(
        f"[bold cyan]CSV Preview[/bold cyan] "
        f"({len(records)} of {stats.get('total_written', '?')} rows):"
    )
    console.print()

    table = Table(show_header=True, header_style="bold magenta", show_lines=True)

    # Add required columns
    for col in REQUIRED_FIELDS:
        table.add_column(col, style="white", no_wrap=(col != "image"))

    # Check for extra columns in records
    extra_cols = set()
    for r in records:
        for k in r:
            if k not in REQUIRED_FIELDS:
                extra_cols.add(k)
    for col in sorted(extra_cols):
        table.add_column(col, style="dim")

    # Add rows
    all_cols = list(REQUIRED_FIELDS.keys()) + sorted(extra_cols)
    for r in records[:max_rows]:
        row = []
        for col in all_cols:
            val = r.get(col, "")
            # Truncate long values
            s = str(val) if val is not None else ""
            if len(s) > 35:
                s = s[:32] + "..."
            row.append(s)
        table.add_row(*row)

    console.print(table)

    # ── Flagged Issues ───────────────────────────────────────────────────

    issues = validation_result.get("issues", [])
    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    infos = [i for i in issues if i["severity"] == "info"]

    if errors or warnings:
        console.print()
        console.print("[bold]Flagged Issues:[/bold]")
        for issue in errors:
            console.print(f"  [red]ERROR[/red]   {issue['message']}")
        for issue in warnings:
            console.print(f"  [yellow]WARN[/yellow]    {issue['message']}")
        for issue in infos:
            console.print(f"  [dim]INFO[/dim]    {issue['message']}")
    else:
        console.print("\n  [green]No issues found![/green]")

    # ── Quick Stats ──────────────────────────────────────────────────────

    summary = validation_result.get("summary", {})
    console.print()
    console.print("[bold]Quick Stats:[/bold]")
    console.print(f"  Total annotations: {stats.get('total_written', 0)}")
    console.print(f"  Unique images:     {summary.get('unique_images', '?')}")
    console.print(f"  Unique concepts:   {summary.get('unique_concepts', '?')}")
    if summary.get("bbox_area_min"):
        console.print(
            f"  Bbox area range:   {summary['bbox_area_min']:,} px² → "
            f"{summary['bbox_area_max']:,} px²"
        )
    if stats.get("total_skipped", 0) > 0:
        console.print(f"  Skipped:           {stats['total_skipped']} (excluded concepts)")
    if stats.get("total_errors", 0) > 0:
        console.print(f"  Errors:            [red]{stats['total_errors']}[/red]")
    if stats.get("images_converted", 0) > 0:
        console.print(f"  Images converted:  {stats['images_converted']}")
    if stats.get("total_dropped", 0) > 0:
        console.print(f"  Dropped/issues:    [yellow]{stats['total_dropped']}[/yellow]")

    dropped_items = dropped_items or []
    if dropped_items:
        console.print()
        console.print("[bold]Dropped / unavailable items:[/bold]")
        for item in dropped_items[:8]:
            label = item.get("type", "item")
            reason = item.get("reason", "Dropped")
            image = item.get("image")
            source = item.get("source")
            suffix = []
            if image:
                suffix.append(f"image={image}")
            if source:
                suffix.append(f"source={source}")
            details = f" [dim]({'; '.join(suffix)})[/dim]" if suffix else ""
            console.print(f"  [yellow]{label}[/yellow] {reason}{details}")
        if len(dropped_items) > 8:
            console.print(f"  [dim]... and {len(dropped_items) - 8} more[/dim]")

    # ── Concept Distribution ─────────────────────────────────────────────

    concept_dist = summary.get("concept_distribution", {})
    if concept_dist:
        console.print()
        console.print("[bold]Top concepts:[/bold]")
        for concept, count in list(concept_dist.items())[:10]:
            bar_len = min(int(count / max(concept_dist.values()) * 30), 30)
            bar = "█" * bar_len
            console.print(f"  {concept:30s} {count:>6d}  {bar}")
        if len(concept_dist) > 10:
            console.print(f"  ... and {len(concept_dist) - 10} more")

    console.print()
