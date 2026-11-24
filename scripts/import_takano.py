"""Write Annotile tables for the 17 Takano lung-adenocarcinoma Xenium sections.

The raw Xenium folders stay where they are. This script only writes the
annotation and centroid tables, then appends those sections to the public
manifest. Cell-type names are the ones stored on each Seurat object.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path("/mnt/storage/bzhang/annotile")
RAW = Path("/mnt/storage/bzhang/data/lung_cancer_model/takano_kero/Xenium_Raw")
LABELS = ROOT / "data" / "public" / "takano" / "labels"
OUT = ROOT / "data" / "public" / "takano"
MANIFEST = ROOT / "data" / "public" / "manifest.json"
PAPER = "https://doi.org/10.1038/s41467-024-54671-7"

# rds stem, raw folder, slice name, patient, stage shown on the page
SECTIONS = (
    ("xenium_2", "Xenium_LUAD_No2", "tk_luad2", "Takano_LUAD2", "浸润性"),
    ("xenium_3", "Xenium_LUAD_No3", "tk_luad3", "Takano_LUAD3", "浸润性"),
    ("xenium_14", "Xenium_LUAD_No14", "tk_luad14", "Takano_LUAD14", "浸润性"),
    ("xenium_16", "Xenium_LUAD_No16", "tk_luad16", "Takano_LUAD16", "浸润性"),
    ("xenium_17", "Xenium_LUAD_No17", "tk_luad17", "Takano_LUAD17", "浸润性"),
    ("xenium_tsu_20", "Xenium_TSU-20", "tk_tsu20", "Takano_TSU20", "Noguchi A"),
    ("xenium_tsu_23", "Xenium_TSU-23", "tk_tsu23", "Takano_TSU23", "Noguchi A"),
    ("xenium_tsu_25", "Xenium_TSU-25", "tk_tsu25", "Takano_TSU25", "Noguchi A"),
    ("xenium_tsu_21", "Xenium_TSU-21", "tk_tsu21", "Takano_TSU21", "Noguchi B"),
    ("xenium_tsu_24", "Xenium_TSU-24", "tk_tsu24", "Takano_TSU24", "Noguchi B"),
    ("xenium_tsu_35", "Xenium_TSU-35", "tk_tsu35", "Takano_TSU35", "Noguchi B"),
    ("xenium_tsu_27", "Xenium_TSU-27", "tk_tsu27", "Takano_TSU27", "Noguchi C"),
    ("xenium_tsu_28", "Xenium_TSU-28", "tk_tsu28", "Takano_TSU28", "Noguchi C"),
    ("xenium_tsu_30", "Xenium_TSU-30", "tk_tsu30", "Takano_TSU30", "Noguchi C"),
    ("xenium_tsu_33", "Xenium_TSU-33", "tk_tsu33", "Takano_TSU33", "Noguchi C"),
    ("xenium_tsu_36", "Xenium_TSU-36", "tk_tsu36", "Takano_TSU36", "Noguchi C"),
    ("xenium_tsu_41", "Xenium_TSU-41", "tk_tsu41", "Takano_TSU41", "Noguchi C"),
)


def _barcodes(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        group = handle["matrix"] if "matrix" in handle else handle
        values = group["barcodes"][:]
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode().rstrip("\x00"))
        else:
            out.append(str(value).rstrip("\x00"))
    return out


def _match(labels: pd.DataFrame, cells: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    labels = labels.copy()
    labels["cell_id"] = labels["cell_id"].astype(str)
    cells = cells.copy()
    cells["cell_id"] = cells["cell_id"].astype(str)
    direct = labels.merge(cells, on="cell_id", how="inner")
    if len(direct) >= 0.5 * len(labels):
        return direct, f"id {len(direct)}/{len(labels)}"
    counts = pd.to_numeric(labels["ncount"], errors="coerce").to_numpy()
    totals = pd.to_numeric(cells["total_counts"], errors="coerce").to_numpy()
    candidates = []
    if len(labels) == len(cells):
        candidates.append(cells)
    positive = cells.loc[totals > 0]
    if len(labels) == len(positive):
        candidates.append(positive)
    for chosen in candidates:
        other = pd.to_numeric(chosen["total_counts"], errors="coerce").to_numpy()
        if len(other) < 3 or not np.isfinite(counts).all() or not np.isfinite(other).all():
            continue
        corr = float(np.corrcoef(counts, other)[0, 1])
        if corr < 0.9:
            continue
        aligned = labels.copy()
        aligned["cell_id"] = chosen["cell_id"].to_numpy()
        aligned["x_centroid"] = chosen["x_centroid"].to_numpy()
        aligned["y_centroid"] = chosen["y_centroid"].to_numpy()
        aligned["total_counts"] = chosen["total_counts"].to_numpy()
        return aligned, f"order corr={corr:.4f} {len(aligned)}"
    return direct, f"unmatched id {len(direct)}/{len(labels)}"


def main() -> None:
    missing = [stem for stem, *_rest in SECTIONS if not (LABELS / f"{stem}.csv").is_file()]
    if missing:
        raise SystemExit("label csv still missing: " + ", ".join(missing))
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for stem, folder, name, patient, stage in SECTIONS:
        labels = pd.read_csv(LABELS / f"{stem}.csv", dtype=str)
        cells = pd.read_csv(
            RAW / folder / "cells.csv.gz",
            usecols=["cell_id", "x_centroid", "y_centroid", "total_counts"],
        )
        matched, how = _match(labels, cells)
        matrix = RAW / folder / "cell_feature_matrix.h5"
        barcodes = set(_barcodes(matrix))
        keep = matched["cell_id"].astype(str).isin(barcodes)
        used = matched.loc[keep].copy()
        used = used.dropna(subset=["annotation", "x_centroid", "y_centroid"])
        used = used[used["annotation"].astype(str).str.len() > 0]
        used = used.drop_duplicates("cell_id")
        rate = len(used) / max(len(labels), 1)
        print(f"{name} {how} h5_kept={len(used)} rate={rate:.4f}", flush=True)
        if rate < 0.5:
            raise SystemExit(f"{name}: cell ids do not meet the expression matrix")
        ann = OUT / f"{name}_annotation.csv"
        cen = OUT / f"{name}_centroids.csv"
        ann.write_text(
            "cell_id,group\n" + "".join(f"{cid},{label}\n" for cid, label in zip(used["cell_id"], used["annotation"])),
            encoding="utf-8",
        )
        with cen.open("w", encoding="utf-8") as handle:
            handle.write("cell_id,x_centroid,y_centroid\n")
            for cid, x, y in zip(used["cell_id"], used["x_centroid"], used["y_centroid"]):
                handle.write(f"{cid},{float(x):.4f},{float(y):.4f}\n")
        entries.append({
            "name": name,
            "title": f"Takano Xenium {folder.removeprefix('Xenium_')}（{stage}）",
            "annotation": str(ann.relative_to(ROOT)).replace("\\", "/"),
            "centroids": str(cen.relative_to(ROOT)).replace("\\", "/"),
            "matrix": str(matrix),
            "source": PAPER,
            "centroid_url": "https://ssbd.riken.jp/repository/458/",
            "patient": patient,
        })
    current = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else []
    current = [item for item in current if not str(item.get("name", "")).startswith("tk_")]
    MANIFEST.write_text(json.dumps(current + entries, indent=2), encoding="utf-8")
    print(f"TAKANO_IMPORT_DONE sections={len(entries)} manifest={len(current) + len(entries)}", flush=True)


if __name__ == "__main__":
    main()
