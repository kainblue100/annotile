"""Score additive programs on the eight patterns.

Yan et al., Nature 2026, separate a spatial niche (cells that share a place)
from an ecotype (programs that co-occur across patients). These three donors
and this gene panel cannot support their patient-level archetypes or ecotypes,
and the panel does not contain their ISG15 / HLA-DRA gene set. The scores
below use only genes already measured here. A program is the mean of
gene-wise z-scores across the eight patterns, so two programs can be high in
the same pattern.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"E:\annotile")
RESULTS = ROOT / "results"

MODULES = (
    ("Epithelium", ("EPCAM", "MALL", "SOX2")),
    ("Proliferation", ("MKI67", "UBE2C")),
    ("Stroma", ("COL5A2", "THY1", "PDGFRA")),
    ("Endothelium", ("PECAM1", "CD34", "CD93", "CLEC14A")),
    ("Macrophage", ("CD68", "CD163", "CD14")),
    ("CXCL9", ("CXCL9",)),
    ("STC2", ("STC2",)),
)
GROUPS = {
    "epithelium": ("epi", "epi?", "normal_epi", "AT1", "AT2_like", "cil"),
    "macrophage": ("mph",),
    "lymphoid": ("nkt", "Tfh_CXCL13_region", "b", "plasma"),
    "endothelium": ("endo", "endo?", "lymph_endo"),
    "fibroblast": ("fib", "fib_activated", "fib_SYNPO2"),
}
ORDER = [f"P{i}" for i in range(1, 9)]
PATHS = {"5k": ["P7", "P4"], "lc": ["P5", "P1"], "luad": ["P7", "P2", "P8"]}


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows = _read(RESULTS / "pattern_expression.csv")
    key = "" if "" in rows[0] else list(rows[0])[0]
    genes = [column for column in rows[0] if column != key]
    table = {row[key]: {gene: float(row[gene]) for gene in genes} for row in rows}
    matrix = np.array([[table[pattern][gene] for gene in genes] for pattern in ORDER])
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    z = (matrix - center) / scale
    index = {gene: i for i, gene in enumerate(genes)}
    scores = {}
    for name, members in MODULES:
        columns = [index[gene] for gene in members]
        scores[name] = z[:, columns].mean(axis=1)

    windows = _read(RESULTS / "windows.csv")
    composition: dict[str, dict[str, dict[str, float]]] = {}
    for slc, path in PATHS.items():
        composition[slc] = {}
        for pattern in path:
            chosen = [row for row in windows if row["slice"] == slc and row["pattern"] == pattern]
            weight = sum(float(row["n_cells"]) for row in chosen)
            composition[slc][pattern] = {"cells": int(weight), "windows": len(chosen)}
            for group, columns in GROUPS.items():
                total = 0.0
                for row in chosen:
                    cells = float(row["n_cells"])
                    total += cells * sum(float(row[column]) for column in columns)
                composition[slc][pattern][group] = round(total / weight, 4) if weight else 0.0

    payload = {
        "source": "Yan et al., Nature 2026, spatial niche rather than cross-patient ecotype",
        "not_fit": "No archetype or ecotype was fit. Three donors, no treatment outcome, and the shared panel lacks ISG15, IFI27, IFIT3 and HLA-DRA.",
        "unit": "mean z-score of the named genes across the eight patterns",
        "patterns": ORDER,
        "modules": {
            name: {
                "genes": list(members),
                "scores": {pattern: round(float(scores[name][i]), 3) for i, pattern in enumerate(ORDER)},
            }
            for name, members in MODULES
        },
        "path_composition": composition,
    }
    (RESULTS / "niche_programs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _draw(scores, composition)
    print("WROTE niche_programs")


def _color(value: float) -> tuple[int, int, int]:
    # diverging around 0, clipped at +/-1.5
    t = max(-1.5, min(1.5, value)) / 1.5
    if t >= 0:
        return (int(255 - 80 * t), int(244 - 140 * t), int(236 - 150 * t))
    t = -t
    return (int(255 - 180 * t), int(244 - 90 * t), int(236 - 40 * t))


def _draw(scores: dict[str, np.ndarray], composition: dict) -> None:
    image = Image.new("RGB", (1280, 820), (250, 248, 244))
    draw = ImageDraw.Draw(image)
    title = _font(22)
    label = _font(16)
    small = _font(14)
    draw.text((28, 18), "Additive programs on the eight patterns", font=title, fill=(30, 30, 30))
    names = [name for name, _members in MODULES]
    left, top, cell = 160, 92, 110
    for j, name in enumerate(names):
        draw.text((left + j * cell, top - 22), name, font=small, fill=(40, 40, 40))
    for i, pattern in enumerate(ORDER):
        y = top + 28 + i * 36
        draw.text((40, y + 6), pattern, font=label, fill=(30, 30, 30))
        for j, name in enumerate(names):
            value = float(scores[name][i])
            x = left + j * cell
            draw.rounded_rectangle((x, y, x + cell - 8, y + 30), 4, fill=_color(value))
            draw.text((x + 8, y + 6), f"{value:+.2f}", font=small, fill=(20, 20, 20))
    draw.text((40, 450), "Cell composition on the video branch of each slice", font=title, fill=(30, 30, 30))
    headers = ("epithelium", "macrophage", "lymphoid", "endothelium", "fibroblast")
    x0 = 40
    for j, header in enumerate(headers):
        draw.text((260 + j * 140, 498), header, font=small, fill=(40, 40, 40))
    y = 528
    for slc, path in PATHS.items():
        for pattern in path:
            item = composition[slc][pattern]
            draw.text((x0, y), f"{slc} {pattern}", font=label, fill=(30, 30, 30))
            for j, header in enumerate(headers):
                draw.text((260 + j * 140, y), f"{item[header]:.2f}", font=label, fill=(20, 20, 20))
            y += 28
    image.save(RESULTS / "figures" / "niche_programs.png")


if __name__ == "__main__":
    main()
