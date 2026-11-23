"""Mark evolutionary transitions on the H&E of each section.

Arrows run from the upstream pattern to the downstream pattern on the tree.
The tissue underneath is the registered H&E of that section.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"E:\annotile")
HE = ROOT / "data" / "he"
RESULTS = ROOT / "results"
WINDOW = 400.0
COLORS = {
    "P1": (244, 162, 97), "P2": (42, 157, 143), "P3": (233, 196, 106), "P4": (231, 111, 81),
    "P5": (109, 89, 122), "P6": (69, 123, 157), "P7": (181, 131, 141), "P8": (38, 70, 83),
}


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _windows() -> list[dict[str, str]]:
    with (RESULTS / "windows.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _matrix(name: str) -> np.ndarray:
    raw = np.loadtxt(HE / f"{name}_he_imagealignment.csv", delimiter=",")
    choice = json.loads((HE / "alignment_choice.json").read_text(encoding="utf-8"))
    if choice.get(name) == "inverse":
        raw = np.linalg.inv(raw)
    return raw


def _overview(name: str) -> tuple[np.ndarray, float]:
    path = HE / f"{name}_he_image.ome.tif"
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        levels = series.levels or [series]
        full = levels[0]
        chosen = levels[4] if len(levels) > 4 else levels[-1]
        image = np.asarray(chosen.asarray())
        scale = full.shape[0] / image.shape[0]
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    return image[:, :, :3].astype(np.uint8), float(scale)


def _points(name: str, rows: list[dict[str, str]], scale: float) -> dict[str, np.ndarray]:
    matrix = _matrix(name)
    grouped: dict[str, list[np.ndarray]] = {}
    for row in rows:
        if row["slice"] != name:
            continue
        x = (int(row["ix"]) + 0.5) * WINDOW
        y = (int(row["iy"]) + 0.5) * WINDOW
        pixel = matrix @ np.array([x, y, 1.0])
        grouped.setdefault(row["pattern"], []).append(pixel[:2] / scale)
    return {pattern: np.vstack(points) for pattern, points in grouped.items()}


def _curve(start: np.ndarray, end: np.ndarray, bend: float) -> list[tuple[float, float]]:
    delta = end - start
    normal = np.array([-delta[1], delta[0]])
    length = float(np.linalg.norm(normal)) or 1.0
    control = (start + end) / 2 + normal / length * bend
    steps = np.linspace(0, 1, 28)
    points = []
    for t in steps:
        point = (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t ** 2 * end
        points.append((float(point[0]), float(point[1])))
    return points


def _arrow(draw: ImageDraw.ImageDraw, start: np.ndarray, end: np.ndarray, bend: float, width: int) -> None:
    path = _curve(start, end, bend)
    draw.line(path, fill=(20, 20, 20), width=width + 4, joint="curve")
    draw.line(path, fill=(255, 255, 255), width=width, joint="curve")
    tip = np.array(path[-1])
    back = np.array(path[-4])
    direction = tip - back
    length = float(np.linalg.norm(direction)) or 1.0
    direction = direction / length
    side = np.array([-direction[1], direction[0]])
    head = 16
    left = tip - direction * head + side * head * 0.55
    right = tip - direction * head - side * head * 0.55
    draw.polygon([tuple(tip), tuple(left), tuple(right)], fill=(255, 255, 255), outline=(20, 20, 20))


def _badge(draw: ImageDraw.ImageDraw, point: np.ndarray, pattern: str, font) -> None:
    x, y = float(point[0]), float(point[1])
    color = COLORS[pattern]
    draw.rounded_rectangle((x - 16, y - 14, x + 16, y + 14), radius=4, fill=color, outline=(255, 255, 255), width=2)
    text = pattern[1:]
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2 - 1), text, fill=(255, 255, 255), font=font)


def _one(name: str, tree: dict) -> Image.Image:
    image, scale = _overview(name)
    canvas = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    paint = ImageDraw.Draw(overlay)
    points = _points(name, _windows(), scale)
    radius = max(7, int(180 / scale))
    for pattern, block in points.items():
        color = COLORS[pattern] + (150,)
        for x, y in block:
            paint.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    canvas = Image.alpha_composite(canvas, overlay)
    finite = np.vstack(list(points.values()))
    pad = 80
    x0 = max(0, int(finite[:, 0].min()) - pad)
    y0 = max(0, int(finite[:, 1].min()) - pad)
    x1 = min(canvas.size[0], int(finite[:, 0].max()) + pad)
    y1 = min(canvas.size[1], int(finite[:, 1].max()) + pad)
    canvas = canvas.crop((x0, y0, x1, y1))
    points = {pattern: block - np.array([x0, y0]) for pattern, block in points.items()}
    longest = max(canvas.size)
    if longest < 1100:
        factor = 1100 / longest
        canvas = canvas.resize((int(canvas.size[0] * factor), int(canvas.size[1] * factor)), Image.Resampling.BILINEAR)
        points = {pattern: block * factor for pattern, block in points.items()}
    draw = ImageDraw.Draw(canvas)
    item = next(row for row in tree["slices"] if row["slice"] == name)
    centers = {pattern: block.mean(axis=0) for pattern, block in points.items()}
    red = set(zip(item["main_path"], item["main_path"][1:]))
    font = _font(18)
    bend_sign = 1
    for parent, kids in item["children"].items():
        if parent not in centers:
            continue
        for kid in kids:
            if kid not in centers:
                continue
            span = float(np.linalg.norm(centers[kid] - centers[parent]))
            _arrow(
                draw,
                centers[parent],
                centers[kid],
                bend_sign * max(28.0, span * 0.18),
                7 if (parent, kid) in red else 4,
            )
            bend_sign *= -1
    for pattern, center in centers.items():
        _badge(draw, center, pattern, font)
    legend = _font(22)
    x, y = canvas.size[0] - 92, 18
    draw.rounded_rectangle((x - 10, 8, canvas.size[0] - 8, 18 + 32 * len(centers)), radius=6, fill=(255, 255, 255))
    for pattern in sorted(centers):
        color = COLORS[pattern]
        draw.rounded_rectangle((x, y, x + 22, y + 22), radius=3, fill=color)
        draw.text((x + 30, y - 2), pattern[1:], fill=(30, 30, 30), font=legend)
        y += 32
    return canvas.convert("RGB")


def main() -> None:
    tree = json.loads((RESULTS / "trajectory.json").read_text(encoding="utf-8"))
    images = []
    out_dir = RESULTS / "figures"
    for name in ("5k", "lc", "luad"):
        image = _one(name, tree)
        path = out_dir / f"transition_{name}.png"
        image.save(path, quality=90)
        images.append(image)
        print("WROTE", path, image.size, flush=True)
    width = sum(image.size[0] for image in images)
    height = max(image.size[1] for image in images)
    sheet = Image.new("RGB", (width, height), (240, 240, 240))
    x = 0
    for image in images:
        sheet.paste(image, (x, 0))
        x += image.size[0]
    sheet.save(out_dir / "transitions.png", quality=90)
    print("WROTE", out_dir / "transitions.png", flush=True)


if __name__ == "__main__":
    main()
