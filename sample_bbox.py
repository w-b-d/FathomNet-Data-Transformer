"""
Interactive bounding-box sampler for converted FathomNet outputs.

Run this after conversion against an output folder containing metadata.csv and
the copied images. It starts a tiny local web server, opens a browser page, and
serves fresh random batches of annotated images until you quit.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import threading
import webbrowser
from collections import defaultdict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, quote, unquote, urlparse

from PIL import Image, ImageDraw, ImageFont


REQUIRED_COLUMNS = {"concept", "image", "x", "y", "width", "height"}

COLORS = [
    (0, 190, 120),
    (245, 130, 32),
    (30, 144, 255),
    (230, 70, 90),
    (170, 95, 220),
    (230, 205, 45),
    (30, 190, 210),
    (245, 115, 175),
]


@dataclass
class SamplerState:
    output_dir: Path
    records_by_image: dict[str, list[dict]]
    image_paths: dict[str, Path]
    batch_size: int
    rng: random.Random

    @property
    def images(self) -> list[str]:
        return list(self.records_by_image.keys())


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Open a browser-based random visual check for an output folder "
            "containing metadata.csv and images."
        )
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=".",
        help="output folder containing metadata.csv and copied images",
    )
    parser.add_argument(
        "--metadata",
        default="metadata.csv",
        help="metadata CSV filename or path (default: metadata.csv inside output_dir)",
    )
    parser.add_argument(
        "--n",
        "-n",
        type=int,
        default=10,
        help="number of random images to show per batch",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="host for the local preview server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="port for the local preview server (default: choose an open port)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional random seed for repeatable batches",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="print the URL without opening the browser",
    )
    return parser.parse_args()


def main(
    output_dir: str,
    metadata: str = "metadata.csv",
    n_samples: int = 10,
    host: str = "127.0.0.1",
    port: int = 0,
    seed: Optional[int] = None,
    auto_open: bool = True,
):
    output_path = Path(output_dir).expanduser().resolve()
    csv_path = Path(metadata).expanduser()
    if not csv_path.is_absolute():
        csv_path = output_path / csv_path

    try:
        records_by_image, image_paths = load_metadata(output_path, csv_path)
    except ValueError as exc:
        print(exc)
        return 1

    batch_size = max(1, min(n_samples, len(records_by_image)))
    state = SamplerState(
        output_dir=output_path,
        records_by_image=records_by_image,
        image_paths=image_paths,
        batch_size=batch_size,
        rng=random.Random(seed),
    )

    handler = make_handler(state)
    server = ThreadingHTTPServer((host, port), handler)
    server.state = state
    url = f"http://{server.server_address[0]}:{server.server_address[1]}/"

    print(
        f"Loaded {sum(len(v) for v in records_by_image.values())} annotations "
        f"across {len(records_by_image)} readable images."
    )
    print(f"Serving random bbox samples at: {url}")
    print("Click Quit in the browser or press Ctrl+C here to stop.")

    if auto_open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping preview server.")
    finally:
        server.server_close()
    return 0


def load_metadata(
    output_dir: Path, csv_path: Path
) -> tuple[dict[str, list[dict]], dict[str, Path]]:
    if not output_dir.is_dir():
        raise ValueError(f"Output folder not found: {output_dir}")
    if not csv_path.is_file():
        raise ValueError(f"Could not find metadata CSV: {csv_path}")

    records_by_image: dict[str, list[dict]] = defaultdict(list)
    image_paths: dict[str, Path] = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fieldnames
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"metadata.csv is missing required columns: {missing_list}")

        for row in reader:
            image_name = (row.get("image") or "").strip()
            if not image_name or image_name.startswith(("http://", "https://")):
                continue
            if not _has_valid_bbox(row):
                continue

            image_path = _resolve_image_path(output_dir, image_name)
            if image_path is None:
                continue

            row["image"] = image_name
            records_by_image[image_name].append(row)
            image_paths[image_name] = image_path

    if not records_by_image:
        raise ValueError(
            f"No readable local images with valid bounding boxes found in {csv_path}"
        )

    return dict(records_by_image), image_paths


def make_handler(state: SamplerState):
    class BBoxSampleHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html()
            elif parsed.path == "/sample":
                self._send_sample()
            elif parsed.path == "/preview":
                query = parse_qs(parsed.query)
                image_name = query.get("image", [""])[0]
                self._send_preview(unquote(image_name))
            elif parsed.path == "/quit":
                self._send_json({"ok": True, "message": "Server stopped."})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                self.send_error(404)

        def _send_html(self):
            html = build_html(
                image_count=len(state.records_by_image),
                annotation_count=sum(len(v) for v in state.records_by_image.values()),
                batch_size=state.batch_size,
                output_dir=str(state.output_dir),
            )
            self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")

        def _send_sample(self):
            images = state.images
            selected = state.rng.sample(images, min(state.batch_size, len(images)))
            payload = []
            for image_name in selected:
                records = state.records_by_image[image_name]
                concepts = sorted({r.get("concept", "Unknown") for r in records})
                payload.append({
                    "image": image_name,
                    "box_count": len(records),
                    "concepts": concepts[:6],
                    "preview_url": f"/preview?image={quote(image_name)}",
                })
            self._send_json({"items": payload})

        def _send_preview(self, image_name: str):
            image_path = state.image_paths.get(image_name)
            records = state.records_by_image.get(image_name)
            if image_path is None or records is None:
                self.send_error(404)
                return

            try:
                data = render_preview_png(image_path, records)
            except OSError as exc:
                self.send_error(500, str(exc))
                return
            self._send_bytes(data, "image/png")

        def _send_json(self, payload: dict):
            self._send_bytes(
                json.dumps(payload).encode("utf-8"),
                "application/json; charset=utf-8",
            )

        def _send_bytes(self, data: bytes, content_type: str):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return BBoxSampleHandler


def render_preview_png(image_path: Path, records: list[dict]) -> bytes:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font()
    color_by_concept: dict[str, tuple[int, int, int]] = {}

    for row in records:
        parsed = _parse_bbox(row)
        if parsed is None:
            continue
        x, y, w, h = parsed
        concept = row.get("concept") or "Unknown"
        if concept not in color_by_concept:
            color_by_concept[concept] = COLORS[len(color_by_concept) % len(COLORS)]
        color = color_by_concept[concept]

        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(image.width - 1, x + w)
        y1 = min(image.height - 1, y + h)
        if x1 <= x0 or y1 <= y0:
            continue

        for offset in range(3):
            draw.rectangle(
                [x0 + offset, y0 + offset, x1 - offset, y1 - offset],
                outline=color,
            )

        label = str(concept)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_y = max(0, y0 - text_h - 6)
        draw.rectangle(
            [x0, label_y, min(image.width, x0 + text_w + 8), label_y + text_h + 6],
            fill=color,
        )
        draw.text((x0 + 4, label_y + 3), label, fill=(0, 0, 0), font=font)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_html(
    *,
    image_count: int,
    annotation_count: int,
    batch_size: int,
    output_dir: str,
) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bounding Box Sampler</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #10141b;
      color: #edf2f7;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: rgba(16, 20, 27, 0.95);
      border-bottom: 1px solid #283242;
      padding: 16px 22px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 22px;
      letter-spacing: 0;
    }}
    .meta {{
      color: #a7b1c2;
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    button {{
      border: 1px solid #4b9eff;
      border-radius: 6px;
      background: #1f6feb;
      color: #fff;
      padding: 9px 13px;
      font-size: 14px;
      cursor: pointer;
    }}
    button.secondary {{
      border-color: #536172;
      background: #232c39;
    }}
    button:hover {{ filter: brightness(1.08); }}
    #status {{
      color: #a7b1c2;
      font-size: 14px;
    }}
    main {{ padding: 22px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
      gap: 18px;
    }}
    figure {{
      margin: 0;
      overflow: hidden;
      border: 1px solid #283242;
      border-radius: 8px;
      background: #18202b;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      background: #0b0f14;
    }}
    figcaption {{
      padding: 10px 12px 12px;
      color: #c6d0df;
      font-size: 13px;
      line-height: 1.35;
    }}
    .filename {{
      color: #79b8ff;
      font-weight: 600;
      overflow-wrap: anywhere;
    }}
    .concepts {{
      margin-top: 4px;
      color: #97a6ba;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Bounding Box Sampler</h1>
    <div class="meta">{annotation_count:,} annotations across {image_count:,} readable local images</div>
    <div class="meta">{output_dir}</div>
    <div class="toolbar">
      <button id="sample" type="button">New Random Batch</button>
      <button id="quit" class="secondary" type="button">Quit</button>
      <span id="status">Loading...</span>
    </div>
  </header>
  <main>
    <div id="grid" class="grid"></div>
  </main>
  <script>
    const batchSize = {batch_size};
    const grid = document.getElementById("grid");
    const status = document.getElementById("status");
    const sampleButton = document.getElementById("sample");
    const quitButton = document.getElementById("quit");

    async function loadBatch() {{
      sampleButton.disabled = true;
      status.textContent = "Sampling...";
      try {{
        const response = await fetch("/sample", {{ cache: "no-store" }});
        const data = await response.json();
        const stamp = Date.now();
        grid.replaceChildren();
        for (const item of data.items) {{
          const conceptsText = item.concepts.length ? item.concepts.join(", ") : "Unknown";
          const figure = document.createElement("figure");
          const img = document.createElement("img");
          const caption = document.createElement("figcaption");
          const filename = document.createElement("div");
          const concepts = document.createElement("div");

          img.src = `${{item.preview_url}}&v=${{stamp}}`;
          img.alt = item.image;
          filename.className = "filename";
          filename.textContent = item.image;
          concepts.className = "concepts";
          concepts.textContent = `${{item.box_count}} box(es) - ${{conceptsText}}`;

          caption.append(filename, concepts);
          figure.append(img, caption);
          grid.append(figure);
        }}
        status.textContent = `Showing ${{data.items.length}} random image(s).`;
      }} catch (error) {{
        status.textContent = `Could not load sample: ${{error}}`;
      }} finally {{
        sampleButton.disabled = false;
      }}
    }}

    sampleButton.addEventListener("click", loadBatch);
    quitButton.addEventListener("click", async () => {{
      quitButton.disabled = true;
      status.textContent = "Stopping server...";
      try {{
        await fetch("/quit", {{ cache: "no-store" }});
        status.textContent = "Server stopped. You can close this tab.";
      }} catch (error) {{
        status.textContent = "Server stopped.";
      }}
    }});
    document.addEventListener("keydown", (event) => {{
      if (event.key === "Enter" || event.key === " ") {{
        event.preventDefault();
        loadBatch();
      }}
    }});
    loadBatch();
  </script>
</body>
</html>"""


def _resolve_image_path(output_dir: Path, image_name: str) -> Optional[Path]:
    image_ref = Path(image_name)
    candidates = []
    if image_ref.is_absolute():
        candidates.append(image_ref)
    else:
        candidates.append(output_dir / image_ref)
        candidates.append(output_dir / image_ref.name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _has_valid_bbox(row: dict) -> bool:
    parsed = _parse_bbox(row)
    return parsed is not None and parsed[2] > 0 and parsed[3] > 0


def _parse_bbox(row: dict) -> Optional[tuple[int, int, int, int]]:
    try:
        x = round(float(row.get("x", "")))
        y = round(float(row.get("y", "")))
        w = round(float(row.get("width", "")))
        h = round(float(row.get("height", "")))
    except (TypeError, ValueError):
        return None
    return x, y, w, h


def _load_font():
    font_candidates = [
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, 14)
        except OSError:
            continue
    return ImageFont.load_default()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        main(
            output_dir=args.output_dir,
            metadata=args.metadata,
            n_samples=args.n,
            host=args.host,
            port=args.port,
            seed=args.seed,
            auto_open=not args.no_open,
        )
    )
