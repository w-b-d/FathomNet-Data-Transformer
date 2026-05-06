"""
Image preview — generates preview images with bounding boxes drawn on them.
"""

import os
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


def generate_image_previews(
    records: list[dict],
    image_search_dirs: list[str],
    output_dir: str = "./output/previews",
    all_records: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Generate preview images with bounding boxes drawn on them.

    Args:
        records: sample records to preview (from smart sampling)
        image_search_dirs: directories to search for source images
        output_dir: where to save preview images
        all_records: all records (to draw multiple boxes per image)

    Returns:
        list of {image, preview_path, records} dicts
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        console.print("[red]Pillow is required for image preview. Install with: pip install Pillow[/red]")
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group all records by image (so we can draw all boxes for a given image)
    if all_records:
        records_by_image = {}
        for r in all_records:
            img_name = r.get("image", "")
            if img_name not in records_by_image:
                records_by_image[img_name] = []
            records_by_image[img_name].append(r)
    else:
        records_by_image = {}

    # Color palette for different concepts
    COLORS = [
        (0, 255, 0),      # green
        (255, 100, 0),     # orange
        (0, 200, 255),     # cyan
        (255, 50, 50),     # red
        (255, 255, 0),     # yellow
        (200, 0, 255),     # purple
        (0, 255, 200),     # teal
        (255, 150, 200),   # pink
    ]
    concept_colors = {}
    color_idx = 0

    previews = []

    for sample in records:
        img_name = sample.get("image", "")
        if not img_name:
            continue

        # Find the source image
        img_path = _find_image(img_name, image_search_dirs)
        if img_path is None:
            console.print(f"  [yellow]Image not found: {img_name}[/yellow]")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            # Get all records for this image
            img_records = records_by_image.get(img_name, [sample])

            for r in img_records:
                concept = r.get("concept", "Unknown")
                x = r.get("x", 0)
                y = r.get("y", 0)
                w = r.get("width", 0)
                h = r.get("height", 0)

                # Assign color per concept
                if concept not in concept_colors:
                    concept_colors[concept] = COLORS[color_idx % len(COLORS)]
                    color_idx += 1
                color = concept_colors[concept]

                # Draw bounding box
                for offset in range(2):  # 2px thick
                    draw.rectangle(
                        [x + offset, y + offset, x + w - offset, y + h - offset],
                        outline=color,
                    )

                # Draw label background
                label = concept
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                except (OSError, IOError):
                    font = ImageFont.load_default()

                bbox = font.getbbox(label)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                label_y = max(0, y - text_h - 4)

                draw.rectangle(
                    [x, label_y, x + text_w + 4, label_y + text_h + 4],
                    fill=color,
                )
                draw.text(
                    (x + 2, label_y + 2),
                    label,
                    fill=(0, 0, 0),
                    font=font,
                )

            # Save preview
            preview_filename = f"preview_{len(previews):02d}_{img_name}"
            if len(preview_filename) > 100:
                preview_filename = f"preview_{len(previews):02d}.jpg"
            preview_path = output_path / preview_filename
            img.save(str(preview_path), "JPEG", quality=85)

            previews.append({
                "image": img_name,
                "preview_path": str(preview_path),
                "records": img_records,
                "image_size": img.size,
            })

        except (OSError, Exception) as e:
            console.print(f"  [yellow]Error processing {img_name}: {e}[/yellow]")
            continue

    return previews


def _find_image(filename: str, search_dirs: list[str]) -> Optional[Path]:
    """Search for an image file across multiple directories."""
    for search_dir in search_dirs:
        search_path = Path(search_dir)
        # Direct path
        candidate = search_path / filename
        if candidate.exists():
            return candidate
        # Recursive search
        matches = list(search_path.rglob(filename))
        if matches:
            return matches[0]
    return None
