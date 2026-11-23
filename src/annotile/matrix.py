"""Shared-gene expression of each spatial window."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

from annotile.data import SLICES, SliceSpec
from annotile.pipeline import WINDOW_UM, load_slice


def _text(values) -> list[str]:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode().rstrip("\x00").strip())
        else:
            out.append(str(value).rstrip("\x00").strip())
    return out


def read_matrix(path: Path):
    with h5py.File(path, "r") as handle:
        group = handle["matrix"] if "matrix" in handle else handle
        data = np.asarray(group["data"][:])
        indices = np.asarray(group["indices"][:])
        indptr = np.asarray(group["indptr"][:])
        barcodes = _text(group["barcodes"][:])
        names = _text(group["features"]["name"][:])
        if "shape" in group:
            shape = tuple(int(v) for v in group["shape"][:])
        else:
            shape = (len(names), len(barcodes))
    if len(indptr) != shape[1] + 1:
        raise RuntimeError(f"{path.name} is not stored with one column per cell")
    matrix = sparse.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()
    return barcodes, names, matrix


def _match(barcodes: list[str], cell_ids: pd.Series) -> tuple[np.ndarray, float]:
    index = {barcode: i for i, barcode in enumerate(barcodes)}
    rows = cell_ids.map(index)
    if rows.notna().mean() < 0.5:
        alt = {barcode[:-2] if barcode.endswith("-1") else f"{barcode}-1": i for barcode, i in index.items()}
        rows = cell_ids.map(alt)
    matched = rows.notna().to_numpy()
    return rows.fillna(-1).to_numpy().astype(np.int32), float(matched.mean())


def expression_block(root: Path, windows: pd.DataFrame, window_um: float = WINDOW_UM) -> tuple[np.ndarray | None, dict]:
    loaded = []
    info = {"shared_genes": 0, "slices": {}}
    for spec in SLICES:
        path = root / spec.matrix
        if not path.is_file():
            return None, info
        barcodes, names, matrix = read_matrix(path)
        cells, _meta = load_slice(root, spec)
        rows, rate = _match(barcodes, cells["cell_id"])
        cells = cells.loc[rows >= 0].copy()
        cells["row"] = rows[rows >= 0]
        cells["ix"] = np.floor(cells["x"].to_numpy() / window_um).astype(np.int32)
        cells["iy"] = np.floor(cells["y"].to_numpy() / window_um).astype(np.int32)
        cells["key"] = spec.name + "|" + cells["ix"].astype(str) + "|" + cells["iy"].astype(str)
        loaded.append((spec.name, names, matrix, cells, rate))
        info["slices"][spec.name] = {"matrix_join": round(rate, 4), "genes": len(names), "matrix_cells": int(len(cells))}
    shared = set(loaded[0][1])
    for _name, names, _matrix, _cells, _rate in loaded[1:]:
        shared &= set(names)
    shared_names = sorted(shared)
    info["shared_genes"] = len(shared_names)
    if not shared_names:
        return None, info
    window_keys = (windows["slice"] + "|" + windows["ix"].astype(int).astype(str) + "|" + windows["iy"].astype(int).astype(str)).tolist()
    key_to_i = {key: i for i, key in enumerate(window_keys)}
    totals = np.zeros((len(windows), len(shared_names)), dtype=np.float64)
    counts = np.zeros(len(windows), dtype=np.float64)
    for _name, names, matrix, cells, _rate in loaded:
        positions = {gene: i for i, gene in enumerate(names)}
        columns = [positions[gene] for gene in shared_names]
        present = cells["key"].isin(key_to_i)
        use = cells.loc[present]
        if use.empty:
            continue
        dense = matrix[use["row"].to_numpy()][:, columns].toarray().astype(np.float64)
        library = dense.sum(axis=1)
        dense = np.log1p(dense / np.maximum(library, 1.0)[:, None] * 1e4)
        dest = use["key"].map(key_to_i).to_numpy()
        np.add.at(totals, dest, dense)
        np.add.at(counts, dest, 1)
    counts = np.maximum(counts, 1.0)
    means = totals / counts[:, None]
    components = min(16, means.shape[0] - 1, means.shape[1])
    if components < 2:
        return means, info
    scores = PCA(n_components=components, random_state=42).fit_transform(means)
    info["expression_components"] = int(components)
    return scores, info
