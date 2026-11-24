"""H&E crops aligned to the Xenium windows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import zarr
from PIL import Image

from annotile.data import load_cohort
from annotile.pipeline import WINDOW_UM

_OPEN = {}


def alignment(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",")


def _image(path: Path):
    if path not in _OPEN:
        tif = tifffile.TiffFile(path)
        store = tif.aszarr(level=0)
        _OPEN[path] = (tif, zarr.open(store, mode="r"))
    return _OPEN[path][1]


def _crop_array(image, matrix: np.ndarray, ix: int, iy: int, window_um: float) -> np.ndarray:
    corners = []
    for x in (ix * window_um, (ix + 1) * window_um):
        for y in (iy * window_um, (iy + 1) * window_um):
            point = matrix @ np.array([x, y, 1.0])
            corners.append(point[:2])
    arr = np.asarray(corners)
    x0, y0 = np.floor(arr.min(axis=0)).astype(int)
    x1, y1 = np.ceil(arr.max(axis=0)).astype(int)
    shape = image.shape
    if len(shape) == 3 and shape[0] in (3, 4) and shape[1] > 8 and shape[2] > 8:
        height, width = shape[1], shape[2]
        y0, y1 = int(np.clip(y0, 0, height)), int(np.clip(y1, 0, height))
        x0, x1 = int(np.clip(x0, 0, width)), int(np.clip(x1, 0, width))
        if y1 <= y0 or x1 <= x0:
            return np.full((64, 64, 3), 245, dtype=np.uint8)
        crop = np.asarray(image[:, y0:y1, x0:x1])
        crop = np.moveaxis(crop, 0, -1)
    else:
        height = shape[0]
        width = shape[1]
        y0, y1 = int(np.clip(y0, 0, height)), int(np.clip(y1, 0, height))
        x0, x1 = int(np.clip(x0, 0, width)), int(np.clip(x1, 0, width))
        if y1 <= y0 or x1 <= x0:
            return np.full((64, 64, 3), 245, dtype=np.uint8)
        crop = np.asarray(image[y0:y1, x0:x1])
    if crop.ndim == 2:
        crop = np.repeat(crop[:, :, None], 3, axis=2)
    crop = crop[:, :, :3]
    if crop.dtype != np.uint8:
        peak = float(np.nanmax(crop)) if crop.size else 1.0
        scale = 255.0 / peak if peak > 255 else 1.0
        crop = np.clip(crop * scale, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(crop).resize((224, 224), Image.Resampling.BILINEAR))


def _spec_paths(root: Path) -> dict[str, tuple[Path, Path]]:
    found = {}
    for spec in load_cohort(root):
        image = root / spec.he_image
        align = root / spec.he_alignment
        if image.is_file() and align.is_file():
            found[spec.name] = (image, align)
    return found


def _matrix_for(image, matrix: np.ndarray, rows: pd.DataFrame, window_um: float) -> tuple[np.ndarray, str]:
    def brightness(candidate: np.ndarray) -> float:
        values = []
        for row in rows.head(8).itertuples(index=False):
            crop = _crop_array(image, candidate, int(row.ix), int(row.iy), window_um)
            values.append(float(crop.mean()))
        return float(np.median(values)) if values else 255.0

    direct = brightness(matrix)
    if direct < 235:
        return matrix, "direct"
    flipped = np.linalg.inv(matrix)
    return (flipped, "inverse") if brightness(flipped) + 5 < direct else (matrix, "direct")


def he_block(root: Path, windows: pd.DataFrame, window_um: float = WINDOW_UM) -> tuple[np.ndarray | None, dict]:
    paths = _spec_paths(root)
    info = {"he_slices": sorted(paths), "alignment": {}}
    if len(paths) != len(load_cohort(root)):
        return None, info
    chosen = {}
    features = np.zeros((len(windows), 7), dtype=np.float64)
    for slice_name, (image_path, align_path) in paths.items():
        image = _image(image_path)
        matrix, how = _matrix_for(image, alignment(align_path), windows[windows["slice"] == slice_name], window_um)
        chosen[slice_name] = matrix
        info["alignment"][slice_name] = how
    (root / "data" / "he" / "alignment_choice.json").write_text(
        json.dumps(info["alignment"]), encoding="utf-8"
    )
    for i, row in enumerate(windows.itertuples(index=False)):
        image = _image(paths[row.slice][0])
        crop = _crop_array(image, chosen[row.slice], int(row.ix), int(row.iy), window_um)
        rgb = crop.astype(np.float64)
        features[i, 0:3] = rgb.mean(axis=(0, 1))
        features[i, 3:6] = rgb.std(axis=(0, 1))
        gray = rgb.mean(axis=2)
        gy, gx = np.gradient(gray)
        features[i, 6] = np.sqrt(gx * gx + gy * gy).mean()
    return features, info


def crop_window(root: Path, slice_name: str, ix: int, iy: int, window_um: float = WINDOW_UM) -> np.ndarray | None:
    paths = _spec_paths(root)
    if slice_name not in paths:
        return None
    image, align_path = paths[slice_name]
    matrix = alignment(align_path)
    choice = root / "data" / "he" / "alignment_choice.json"
    if choice.is_file():
        how = json.loads(choice.read_text(encoding="utf-8")).get(slice_name, "direct")
        if how == "inverse":
            matrix = np.linalg.inv(matrix)
    return _crop_array(_image(image), matrix, ix, iy, window_um)
