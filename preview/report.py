"""
HTML report generator.

Creates a single preview.html file with bounding box images + CSV table
that opens in the browser for easy visual verification.
"""

import base64
import io
import mimetypes
import webbrowser
from pathlib import Path
from typing import Optional

from jinja2 import Template


REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>FathomNet Transformer — Preview Report</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f1117; color: #e0e0e0; padding: 2rem; }
  h1 { color: #4fc3f7; margin-bottom: 0.5rem; }
  h2 { color: #81d4fa; margin: 2rem 0 1rem; }
  .subtitle { color: #888; margin-bottom: 2rem; }
  .stats { display: flex; gap: 2rem; flex-wrap: wrap; margin: 1rem 0; }
  .stat { background: #1a1d27; border-radius: 8px; padding: 1rem 1.5rem; }
  .stat-value { font-size: 1.8rem; font-weight: bold; color: #4fc3f7; }
  .stat-label { color: #888; font-size: 0.85rem; }
  .issues { margin: 1rem 0; }
  .issue { padding: 0.5rem 1rem; border-radius: 4px; margin: 0.25rem 0; }
  .issue-error { background: #2d1515; border-left: 3px solid #f44336; }
  .issue-warning { background: #2d2515; border-left: 3px solid #ff9800; }
  .issue-info { background: #152d1a; border-left: 3px solid #4caf50; }
  .preview-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                  gap: 1.5rem; margin: 1rem 0; }
  .preview-card { background: #1a1d27; border-radius: 8px; overflow: hidden; }
  .preview-card img { width: 100%; height: auto; display: block; }
  .preview-info { padding: 0.75rem 1rem; }
  .preview-info .filename { color: #4fc3f7; font-size: 0.85rem; word-break: break-all; }
  .preview-info .annotations { color: #888; font-size: 0.8rem; margin-top: 0.25rem; }
  .drop-table td { vertical-align: top; }
  .drop-reason { color: #ffcc80; }
  .drop-source { color: #777; word-break: break-all; }
  table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.85rem; }
  th { background: #1a1d27; color: #4fc3f7; padding: 0.75rem; text-align: left;
       border-bottom: 2px solid #333; }
  td { padding: 0.5rem 0.75rem; border-bottom: 1px solid #222; }
  tr:hover { background: #1a1d27; }
  .actions { margin: 2rem 0; text-align: center; }
  .btn { display: inline-block; padding: 0.75rem 2rem; border-radius: 6px;
         font-size: 1rem; cursor: pointer; border: none; margin: 0 0.5rem; }
  .btn-approve { background: #4caf50; color: white; }
  .btn-reject { background: #f44336; color: white; }
  footer { text-align: center; color: #555; margin-top: 3rem; font-size: 0.8rem; }
</style>
</head>
<body>

<h1>FathomNet Transformer — Preview Report</h1>
<p class="subtitle">Review the conversion results before exporting</p>

<!-- Stats -->
<div class="stats">
  <div class="stat">
    <div class="stat-value">{{ stats.total_written }}</div>
    <div class="stat-label">Total Annotations</div>
  </div>
  <div class="stat">
    <div class="stat-value">{{ summary.unique_images }}</div>
    <div class="stat-label">Unique Images</div>
  </div>
  <div class="stat">
    <div class="stat-value">{{ summary.unique_concepts }}</div>
    <div class="stat-label">Unique Concepts</div>
  </div>
	  {% if stats.total_skipped > 0 %}
	  <div class="stat">
	    <div class="stat-value">{{ stats.total_skipped }}</div>
	    <div class="stat-label">Skipped</div>
	  </div>
	  {% endif %}
	  {% if stats.total_dropped > 0 %}
	  <div class="stat">
	    <div class="stat-value">{{ stats.total_dropped }}</div>
	    <div class="stat-label">Dropped / Issues</div>
	  </div>
	  {% endif %}
	  {% if stats.images_converted > 0 %}
	  <div class="stat">
	    <div class="stat-value">{{ stats.images_converted }}</div>
	    <div class="stat-label">Images Converted</div>
	  </div>
	  {% endif %}
	</div>

<!-- Issues -->
{% if issues %}
<h2>Flagged Issues</h2>
<div class="issues">
  {% for issue in issues %}
  <div class="issue issue-{{ issue.severity }}">
    <strong>{{ issue.severity | upper }}</strong> — {{ issue.message }}
  </div>
  {% endfor %}
	</div>
	{% endif %}

	<!-- Dropped Items -->
	{% if dropped_items %}
	<h2>Dropped / Unavailable Items</h2>
	{% if dropped_previews %}
	<div class="preview-grid">
	  {% for item in dropped_previews %}
	  <div class="preview-card">
	    <img src="{{ item.image_data }}" alt="{{ item.image }}">
	    <div class="preview-info">
	      <div class="filename">{{ item.image or item.source }}</div>
	      <div class="annotations">
	        <span class="drop-reason">{{ item.reason }}</span>
	        {% if item.concept %}<br>Concept: {{ item.concept }}{% endif %}
	        {% if item.source %}<br><span class="drop-source">{{ item.source }}</span>{% endif %}
	      </div>
	    </div>
	  </div>
	  {% endfor %}
	</div>
	{% endif %}
	<table class="drop-table">
	  <thead>
	    <tr>
	      <th>Type</th>
	      <th>Reason</th>
	      <th>Image</th>
	      <th>Source</th>
	    </tr>
	  </thead>
	  <tbody>
	    {% for item in dropped_items %}
	    <tr>
	      <td>{{ item.type }}</td>
	      <td class="drop-reason">{{ item.reason }}</td>
	      <td>{{ item.image or '' }}</td>
	      <td class="drop-source">{{ item.source or '' }}</td>
	    </tr>
	    {% endfor %}
	  </tbody>
	</table>
	{% endif %}

	<!-- Image Previews -->
<h2>Visual Preview ({{ previews | length }} samples)</h2>
<div class="preview-grid">
  {% for preview in previews %}
  <div class="preview-card">
    <img src="{{ preview.image_data }}" alt="{{ preview.image }}">
    <div class="preview-info">
      <div class="filename">{{ preview.image }}</div>
      <div class="annotations">
        {% for r in preview.records %}
          {{ r.concept }} ({{ r.x }}, {{ r.y }}, {{ r.width }}×{{ r.height }}){% if not loop.last %}; {% endif %}
        {% endfor %}
      </div>
    </div>
  </div>
  {% endfor %}
</div>

<!-- CSV Preview -->
<h2>CSV Preview</h2>
<table>
  <thead>
    <tr>
      {% for col in columns %}
      <th>{{ col }}</th>
      {% endfor %}
    </tr>
  </thead>
  <tbody>
    {% for row in csv_rows %}
    <tr>
      {% for col in columns %}
      <td>{{ row.get(col, '') }}</td>
      {% endfor %}
    </tr>
    {% endfor %}
  </tbody>
</table>

<footer>
  Generated by FathomNet Data Transformer
</footer>

</body>
</html>
"""


def generate_html_report(
    previews: list[dict],
    sample_records: list[dict],
    stats: dict,
    validation_result: dict,
    dropped_items: Optional[list[dict]] = None,
    output_path: str = "./output/preview.html",
    auto_open: bool = True,
    max_dropped_images: int = 12,
) -> str:
    """
    Generate an HTML preview report and optionally open it in the browser.

    Args:
        previews: output from generate_image_previews()
        sample_records: sample records for CSV table
        stats: transformation stats
        validation_result: output from validate_output()
        dropped_items: skipped annotations/images to show in the report
        output_path: where to save the HTML file
        auto_open: whether to open in browser

    Returns:
        path to the generated HTML file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dropped_items = dropped_items or []
    dropped_previews = _prepare_dropped_previews(
        dropped_items, output_file.parent, max_dropped_images
    )

    # Embed images as base64 data URIs
    for preview in previews:
        preview_file = Path(preview["preview_path"])
        if preview_file.exists():
            with open(preview_file, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            preview["image_data"] = f"data:image/jpeg;base64,{img_data}"
        else:
            preview["image_data"] = ""

    # Determine columns for CSV table
    columns = ["concept", "image", "x", "y", "width", "height"]
    extra_cols = set()
    for r in sample_records:
        for k in r:
            if k not in columns:
                extra_cols.add(k)
    columns.extend(sorted(extra_cols))

    # Render template
    template = Template(REPORT_TEMPLATE)
    html = template.render(
        stats=stats,
        summary=validation_result.get("summary", {}),
        issues=validation_result.get("issues", []),
        dropped_items=dropped_items,
        dropped_previews=dropped_previews,
        previews=previews,
        columns=columns,
        csv_rows=sample_records,
    )

    # Write file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)

    if auto_open:
        webbrowser.open(f"file://{output_file.resolve()}")

    return str(output_file.resolve())


def _prepare_dropped_previews(
    dropped_items: list[dict],
    output_dir: Path,
    max_images: int,
) -> list[dict]:
    """Build small embedded thumbnails for dropped items when an image exists."""
    previews = []
    seen_paths = set()

    for item in dropped_items:
        if len(previews) >= max_images:
            break

        image_path = _resolve_dropped_image_path(item, output_dir)
        if not image_path:
            continue

        try:
            key = str(image_path.resolve())
        except OSError:
            key = str(image_path)
        if key in seen_paths:
            continue
        seen_paths.add(key)

        image_data = _encode_thumbnail(image_path)
        if not image_data:
            continue

        preview_item = dict(item)
        preview_item.setdefault("image", image_path.name)
        preview_item["image_data"] = image_data
        previews.append(preview_item)

    return previews


def _resolve_dropped_image_path(item: dict, output_dir: Path) -> Optional[Path]:
    candidates = []

    source_image_path = item.get("source_image_path")
    if source_image_path:
        candidates.append(Path(source_image_path))

    image = item.get("image")
    if image:
        image_path = Path(str(image))
        candidates.append(image_path)
        candidates.append(output_dir / image_path)
        candidates.append(output_dir / image_path.name)

    source = item.get("source")
    if source:
        candidates.append(Path(str(source)))

    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _encode_thumbnail(image_path: Path) -> Optional[str]:
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            img.thumbnail((520, 360))
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=82)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        try:
            mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime_type};base64,{encoded}"
        except OSError:
            return None
