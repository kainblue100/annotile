"""Density of one public NSCLC through six aligned CosMx sections.

Pentimalli et al., Cell Systems 2025, cut 34 consecutive 5 µm sections from
one tumor. CosMx was run on every sixth section, 30 µm apart, and STIM aligned
those six planes. The coordinates used here are the published aligned tables.
This is not stacked onto the three Xenium donors, and the cells are not yet
labeled: the file is positions only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"E:\annotile")
STIM = ROOT / "data" / "nsclc3d" / "stim"
OUT = ROOT / "results" / "figures" / "serial_nsclc.png"
PX_PER_UM = 5.6
SECTIONS = (4, 10, 16, 22, 28, 34)
BIN_UM = 80.0


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _load(section: int, aligned: bool) -> np.ndarray:
    name = f"section_{section}_aligned.csv" if aligned else f"section_{section}.csv"
    raw = np.loadtxt(STIM / name, delimiter=",")
    return raw / PX_PER_UM


def _grid(points: np.ndarray, origin: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ix = np.floor((points[:, 0] - origin[0]) / BIN_UM).astype(np.int32)
    iy = np.floor((points[:, 1] - origin[1]) / BIN_UM).astype(np.int32)
    keep = (ix >= 0) & (iy >= 0) & (ix < shape[1]) & (iy < shape[0])
    hist = np.zeros(shape, dtype=np.float64)
    np.add.at(hist, (iy[keep], ix[keep]), 1)
    return hist


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = (a + b) > 0
    x = a[mask]
    y = b[mask]
    if x.size < 10:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom else float("nan")


def main() -> None:
    aligned = {section: _load(section, True) for section in SECTIONS}
    raw = {section: _load(section, False) for section in SECTIONS}
    stacked = np.vstack(list(aligned.values()))
    origin = stacked.min(axis=0)
    high = stacked.max(axis=0)
    shape = (
        int(np.ceil((high[1] - origin[1]) / BIN_UM)) + 1,
        int(np.ceil((high[0] - origin[0]) / BIN_UM)) + 1,
    )
    grids = {section: _grid(points, origin, shape) for section, points in aligned.items()}
    shifts = {}
    for section in SECTIONS:
        if section == 4:
            shifts[str(section)] = 0.0
            continue
        n = min(len(aligned[section]), len(raw[section]))
        delta = aligned[section][:n] - raw[section][:n]
        shifts[str(section)] = round(float(np.median(np.linalg.norm(delta, axis=1))), 1)
    adjacent = []
    for left, right in zip(SECTIONS, SECTIONS[1:]):
        adjacent.append(
            {
                "from": f"section_{left}",
                "to": f"section_{right}",
                "gap_um": 30,
                "density_pearson": round(_corr(grids[left], grids[right]), 3),
            }
        )
    counts = {f"section_{section}": int(len(aligned[section])) for section in SECTIONS}
    payload = {
        "source": "Pentimalli et al., Cell Systems 2025, Zenodo 10.5281/zenodo.7899173",
        "alignment": "Published STIM coordinates. 5.6 px = 1 µm. Profiled sections are 30 µm apart.",
        "not_joined": "These six planes are one NSCLC. They are not stacked with the three Xenium donors.",
        "labels": "Coordinates only. Cell-type labels live in the Seurat object and are not in this table.",
        "cells": counts,
        "median_alignment_shift_um": shifts,
        "adjacent_density_pearson": adjacent,
        "span_um": [round(float(high[0] - origin[0]), 1), round(float(high[1] - origin[1]), 1), 150],
    }
    (ROOT / "results" / "serial_nsclc.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _draw(grids, adjacent, counts)
    print(json.dumps(payload, indent=2))


def _draw(grids: dict[int, np.ndarray], adjacent: list[dict], counts: dict[str, int]) -> None:
    thumb = 280
    image = Image.new("RGB", (980, 760), (248, 246, 242))
    draw = ImageDraw.Draw(image)
    title = _font(22)
    label = _font(16)
    draw.text((24, 16), "One NSCLC, six aligned sections, 30 um apart", font=title, fill=(25, 25, 25))
    peak = max(float(np.quantile(grid, 0.98)) for grid in grids.values()) or 1.0
    for i, section in enumerate(SECTIONS):
        col, row = i % 3, i // 3
        x = 30 + col * 310
        y = 70 + row * 300
        grid = grids[section]
        panel = np.zeros((thumb, thumb, 3), dtype=np.uint8)
        panel[:] = (236, 232, 226)
        ys = np.linspace(0, grid.shape[0], thumb, endpoint=False).astype(int)
        xs = np.linspace(0, grid.shape[1], thumb, endpoint=False).astype(int)
        sample = grid[ys][:, xs]
        ink = np.clip(sample / peak, 0, 1)
        panel[:, :, 0] = (236 - ink * 190).astype(np.uint8)
        panel[:, :, 1] = (232 - ink * 150).astype(np.uint8)
        panel[:, :, 2] = (226 - ink * 80).astype(np.uint8)
        image.paste(Image.fromarray(panel), (x, y))
        z = (i * 30)
        draw.text((x, y + thumb + 6), f"section {section}   z={z} um   n={counts[f'section_{section}']}", font=label, fill=(30, 30, 30))
    y = 690
    text = "  ".join(
        f"{item['from'].split('_')[1]}-{item['to'].split('_')[1]} r={item['density_pearson']:.2f}" for item in adjacent
    )
    draw.text((30, y), "Adjacent density Pearson: " + text, font=label, fill=(30, 30, 30))
    image.save(OUT)


if __name__ == "__main__":
    main()
