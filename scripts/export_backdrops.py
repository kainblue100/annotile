"""Crop the H&E window behind each pattern volume and write a scene file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr
from PIL import Image

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"E:\annotile")
sys.path.insert(0, str(ROOT / "src"))

from annotile.data import SLICES, keep_label  # noqa: E402

OUT = ROOT / "results" / "figures" / "blender"
OUT.mkdir(parents=True, exist_ok=True)


def xy_columns(columns: list[str]) -> tuple[str, str]:
    lower = {c.lower(): c for c in columns}
    x = lower.get("x_centroid") or lower.get("x_location")
    y = lower.get("y_centroid") or lower.get("y_location")
    if x is None or y is None:
        raise RuntimeError(f"centroid table has no x/y columns: {columns}")
    return x, y


def load_cells(root: Path) -> pd.DataFrame:
    frames = []
    for spec in SLICES:
        ann = pd.read_csv(root / spec.annotation, dtype=str).rename(columns={"group": "annotation"})
        cen = pd.read_csv(root / spec.centroids)
        xcol, ycol = xy_columns(list(cen.columns))
        cen = cen.rename(columns={xcol: "x", ycol: "y"})
        cen["cell_id"] = cen["cell_id"].astype(str)
        merged = ann[["cell_id", "annotation"]].merge(cen[["cell_id", "x", "y"]], on="cell_id", how="inner")
        merged = merged[merged["annotation"].map(keep_label)].copy()
        merged["x"] = pd.to_numeric(merged["x"], errors="coerce")
        merged["y"] = pd.to_numeric(merged["y"], errors="coerce")
        merged = merged.dropna(subset=["x", "y"])
        merged["slice"] = spec.name
        frames.append(merged[["slice", "x", "y", "annotation"]])
        print("LOADED", spec.name, len(merged), flush=True)
    return pd.concat(frames, ignore_index=True)


def densest(cells: pd.DataFrame, size: float) -> pd.DataFrame:
    xs = cells["x"].to_numpy()
    ys = cells["y"].to_numpy()
    best = cells.iloc[0:0]
    best_n = -1
    x0s = np.arange(xs.min(), max(xs.max() - size, xs.min()) + 1e-6, 8.0)
    y0s = np.arange(ys.min(), max(ys.max() - size, ys.min()) + 1e-6, 8.0)
    for x0 in x0s:
        for y0 in y0s:
            chosen = cells[(cells["x"] >= x0) & (cells["x"] < x0 + size) & (cells["y"] >= y0) & (cells["y"] < y0 + size)]
            if len(chosen) > best_n:
                best = chosen
                best_n = len(chosen)
    return best


def patch_of(cells: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    for size in (40.0, 80.0, 120.0):
        chosen = densest(cells, size)
        if len(chosen) >= 20 or size == 120.0:
            if len(chosen) > 160:
                center = chosen[["x", "y"]].median()
                distance = (chosen["x"] - center["x"]) ** 2 + (chosen["y"] - center["y"]) ** 2
                chosen = chosen.loc[distance.nsmallest(160).index]
            return chosen.copy(), size
    return cells.copy(), 40.0


def window_cells(cells: pd.DataFrame, row) -> pd.DataFrame:
    x0 = float(row.ix) * 400.0
    y0 = float(row.iy) * 400.0
    return cells[(cells["slice"] == row.slice) & (cells["x"] >= x0) & (cells["x"] < x0 + 400.0) & (cells["y"] >= y0) & (cells["y"] < y0 + 400.0)]


def matrix_for(root: Path, slice_name: str) -> np.ndarray:
    spec = next(item for item in SLICES if item.name == slice_name)
    matrix = np.loadtxt(root / spec.he_alignment, delimiter=",")
    choice = json.loads((root / "data" / "he" / "alignment_choice.json").read_text(encoding="utf-8"))
    if choice.get(slice_name) == "inverse":
        matrix = np.linalg.inv(matrix)
    return matrix


def crop_um(image, matrix: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    corners = [(matrix @ np.array([x, y, 1.0]))[:2] for x in (x0, x1) for y in (y0, y1)]
    arr = np.asarray(corners)
    px0, py0 = np.floor(arr.min(axis=0)).astype(int)
    px1, py1 = np.ceil(arr.max(axis=0)).astype(int)
    shape = image.shape
    channel_first = len(shape) == 3 and shape[0] in (3, 4) and shape[1] > 8 and shape[2] > 8
    height, width = (shape[1], shape[2]) if channel_first else (shape[0], shape[1])
    px0, px1 = int(np.clip(px0, 0, width)), int(np.clip(px1, 0, width))
    py0, py1 = int(np.clip(py0, 0, height)), int(np.clip(py1, 0, height))
    if channel_first:
        crop = np.moveaxis(np.asarray(image[:, py0:py1, px0:px1]), 0, -1)
    else:
        crop = np.asarray(image[py0:py1, px0:px1])
    crop = crop[:, :, :3]
    if crop.dtype != np.uint8:
        peak = float(np.nanmax(crop)) if crop.size else 1.0
        scale = 255.0 / peak if peak > 255 else 1.0
        crop = np.clip(crop * scale, 0, 255).astype(np.uint8)
    return crop


def main() -> None:
    html = (ROOT / "docs" / "volume.html").read_text(encoding="utf-8")
    start = html.index("const DATA = ") + len("const DATA = ")
    end = html.index(";\nconst PATTERN")
    payload = json.loads(html[start:end])
    cells = load_cells(ROOT)
    windows = pd.read_csv(ROOT / "results" / "windows.csv")
    images = {}
    scene = {"thickness_um": payload["thickness_um"], "z_scale": 4.0, "patterns": []}
    for model in payload["patterns"]:
        block = windows[windows["pattern"] == model["pattern"]].sort_values("n_cells", ascending=False)
        chosen = pd.DataFrame()
        host = None
        for row in block.itertuples(index=False):
            found = window_cells(cells, row)
            if found.empty:
                continue
            chosen, _span = patch_of(found)
            host = row
            if len(chosen) >= 20:
                break
        if host is None or chosen.empty:
            raise RuntimeError(model["pattern"])
        local = chosen[["x", "y"]].to_numpy(dtype=float)
        local = local - local.min(axis=0)
        published = np.array([[cell["x"], cell["y"]] for cell in model["cells"]], dtype=float)
        if len(local) == len(published):
            gap = np.abs(np.sort(local[:, 0]) - np.sort(published[:, 0])).max()
        else:
            gap = -1
        print(model["pattern"], host.slice, int(host.ix), int(host.iy), "cells", len(chosen), "published", len(published), "xgap", round(float(gap), 3), flush=True)
        if slice_name_image_missing(images, host.slice):
            spec = next(item for item in SLICES if item.name == host.slice)
            tif = tifffile.TiffFile(ROOT / spec.he_image)
            images[host.slice] = zarr.open(tif.aszarr(level=0), mode="r")
            images[host.slice + "_matrix"] = matrix_for(ROOT, host.slice)
        margin = 90.0
        x0 = float(chosen["x"].min()) - margin
        y0 = float(chosen["y"].min()) - margin
        x1 = float(chosen["x"].max()) + margin
        y1 = float(chosen["y"].max()) + margin
        crop = crop_um(images[host.slice], images[host.slice + "_matrix"], x0, y0, x1, y1)
        image = Image.fromarray(crop).resize((960, 960), Image.Resampling.LANCZOS)
        path = OUT / "backdrops" / f"{model['pattern']}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        scene["patterns"].append(model | {"backdrop": f"backdrops/{model['pattern']}.png"})
        print("CROP", model["pattern"], crop.shape, flush=True)
    (OUT / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    print("SCENE_OK", len(scene["patterns"]), flush=True)


def slice_name_image_missing(images: dict, name: str) -> bool:
    return name not in images


if __name__ == "__main__":
    main()
