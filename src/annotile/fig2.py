"""Figure laid out like HIPO Figure 2: t-SNE, correlation triangle, representative windows."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.manifold import TSNE

from annotile.data import SLICES
from annotile.pipeline import WINDOW_UM, load_slice
from annotile.portraits import TYPE_COLOR

PATTERN_COLOR = {
    "P1": "#f4a261",
    "P2": "#2a9d8f",
    "P3": "#e9c46a",
    "P4": "#e76f51",
    "P5": "#6d597a",
    "P6": "#457b9d",
    "P7": "#b5838d",
    "P8": "#264653",
}
FINE_COLOR = [
    "#e9c46a",
    "#2a9d8f",
    "#e76f51",
    "#264653",
    "#6d597a",
    "#f4a261",
    "#457b9d",
    "#b5838d",
    "#81b29a",
    "#f2cc8f",
]


def _load_cells(root: Path) -> pd.DataFrame:
    frames = []
    for spec in SLICES:
        cells, _info = load_slice(root, spec)
        cells["slice"] = spec.name
        frames.append(cells)
    cells = pd.concat(frames, ignore_index=True)
    cells["ix"] = np.floor(cells["x"].to_numpy() / WINDOW_UM).astype(np.int32)
    cells["iy"] = np.floor(cells["y"].to_numpy() / WINDOW_UM).astype(np.int32)
    cells["key"] = cells["slice"] + "|" + cells["ix"].astype(str) + "|" + cells["iy"].astype(str)
    return cells


def _stained(image: np.ndarray) -> float:
    rgb = image.astype(np.float32)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    mean = rgb.mean(axis=2)
    return float(((spread > 12) & (mean < 225)).mean())


def _picks_with_images(root: Path, windows: pd.DataFrame, summary: dict, per_pattern: int = 4):
    from annotile.he import crop_window

    chosen = []
    images = {}
    for item in summary["patterns"]:
        block = windows[windows["pattern"] == item["pattern"]].sort_values("n_cells", ascending=False).head(30)
        scored = []
        for row in block.itertuples(index=False):
            image = crop_window(root, row.slice, int(row.ix), int(row.iy))
            if image is None:
                continue
            scored.append((_stained(image), row, image))
        scored.sort(key=lambda item: item[0], reverse=True)
        taken = []
        seen = set()
        for score, row, image in scored:
            if score < 0.35 or row.slice in seen:
                continue
            taken.append((score, row, image))
            seen.add(row.slice)
            if len(taken) == per_pattern:
                break
        if len(taken) < per_pattern:
            have = {(row.slice, int(row.ix), int(row.iy)) for _score, row, _image in taken}
            for score, row, image in scored:
                ident = (row.slice, int(row.ix), int(row.iy))
                if score < 0.2 or ident in have:
                    continue
                taken.append((score, row, image))
                have.add(ident)
                if len(taken) == per_pattern:
                    break
        if not taken:
            fallback = _picks(windows[windows["pattern"] == item["pattern"]], {"patterns": [item]}, per_pattern)
            for row in fallback.itertuples(index=False):
                chosen.append(row._asdict())
            continue
        for _score, row, image in taken:
            key = f"{row.slice}|{int(row.ix)}|{int(row.iy)}"
            chosen.append(
                {
                    "slice": row.slice,
                    "ix": int(row.ix),
                    "iy": int(row.iy),
                    "fine": int(row.fine),
                    "pattern": row.pattern,
                    "n_cells": int(row.n_cells),
                    "key": key,
                }
            )
            images[key] = image
    return pd.DataFrame(chosen), images


def _picks(windows: pd.DataFrame, summary: dict, per_pattern: int = 4) -> pd.DataFrame:
    chosen = []
    for item in summary["patterns"]:
        block = windows[windows["pattern"] == item["pattern"]].sort_values("n_cells", ascending=False)
        taken = []
        seen_slice = set()
        for row in block.itertuples(index=False):
            if row.slice in seen_slice:
                continue
            taken.append(row)
            seen_slice.add(row.slice)
            if len(taken) == per_pattern:
                break
        if len(taken) < per_pattern:
            have = {(row.slice, int(row.ix), int(row.iy)) for row in taken}
            for row in block.itertuples(index=False):
                ident = (row.slice, int(row.ix), int(row.iy))
                if ident in have:
                    continue
                taken.append(row)
                have.add(ident)
                if len(taken) == per_pattern:
                    break
        for row in taken:
            chosen.append(
                {
                    "slice": row.slice,
                    "ix": int(row.ix),
                    "iy": int(row.iy),
                    "fine": int(row.fine),
                    "pattern": row.pattern,
                    "n_cells": int(row.n_cells),
                }
            )
    return pd.DataFrame(chosen)


def _he_images(root: Path, picks: pd.DataFrame) -> dict[str, np.ndarray]:
    try:
        from annotile.he import crop_window
    except Exception:
        return {}
    images = {}
    for row in picks.itertuples(index=False):
        image = crop_window(root, row.slice, int(row.ix), int(row.iy))
        if image is not None:
            images[row.key] = image
    return images


def _order(summary: dict) -> list[int]:
    ordered = []
    for item in summary["patterns"]:
        for name in item["fine_clusters"]:
            ordered.append(int(name[1:]))
    return ordered


def draw(root: Path, out: Path) -> None:
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    windows = pd.read_csv(out / "windows.csv")
    feature_path = out / "features.npy"
    if feature_path.is_file():
        features = np.load(feature_path)
    else:
        features = windows[summary["labels"]].to_numpy(dtype=np.float64)
    fine = windows["fine"].to_numpy()
    order = _order(summary)
    embedding = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(features)
    means = np.vstack([features[fine == i].mean(axis=0) for i in order])
    corr = np.corrcoef(means)
    picks, images = _picks_with_images(root, windows, summary)
    if "key" not in picks.columns:
        picks["key"] = picks["slice"] + "|" + picks["ix"].astype(int).astype(str) + "|" + picks["iy"].astype(int).astype(str)
    if not images:
        images = _he_images(root, picks)
    cells = _load_cells(root)
    cells = cells[cells["key"].isin(set(picks["key"]))]
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _composite(fig_dir / "fig2.png", summary, windows, embedding, fine, order, corr, picks, cells, images)
    _panel_c(fig_dir / "fig2c.png", summary, picks, cells, images)
    print("fig2.png fig2c.png")


def _composite(path, summary, windows, embedding, fine, order, corr, picks, cells, images) -> None:
    fig = plt.figure(figsize=(13.6, 16.2), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.45], hspace=0.22, wspace=0.16)
    _panel_a(fig.add_subplot(grid[0, 0]), embedding, fine, order)
    heat = grid[0, 1].subgridspec(1, 2, width_ratios=[0.22, 1], wspace=0.04)
    _panel_b(fig.add_subplot(heat[0]), fig.add_subplot(heat[1]), summary, order, corr)
    tiles = grid[1, :].subgridspec(len(summary["patterns"]), 4, wspace=0.12, hspace=0.38)
    _fill_tiles(fig, tiles, summary, picks, cells, images)
    fig.text(0.08, 0.955, "A", fontsize=16, fontweight="bold")
    fig.text(0.54, 0.955, "B", fontsize=16, fontweight="bold")
    fig.text(0.08, 0.56, "C", fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _panel_a(ax, embedding, fine, order) -> None:
    for index, cluster in enumerate(order):
        block = embedding[fine == cluster]
        color = FINE_COLOR[index % len(FINE_COLOR)]
        ax.scatter(block[:, 0], block[:, 1], s=12, c=color, linewidths=0, label=f"F{cluster}", rasterized=True)
        ax.text(block[:, 0].mean(), block[:, 1].mean(), f"F{cluster}", fontsize=8, ha="center", va="center")
    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right", frameon=False, fontsize=8, markerscale=1.3)
    ax.set_title("Fine clusters", fontsize=11)


def _panel_b(ax_strip, ax, summary, order, corr) -> None:
    masked = np.ma.array(corr, mask=np.triu(np.ones_like(corr, dtype=bool), k=1))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#ffffff")
    image = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1, aspect="equal")
    names = [f"F{i}" for i in order]
    ax.set_xticks(range(len(order)), names, rotation=90, fontsize=8)
    ax.set_yticks([])
    ax.set_title("Correlation", fontsize=11)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    pattern_of = {}
    for item in summary["patterns"]:
        for name in item["fine_clusters"]:
            pattern_of[int(name[1:])] = item["pattern"]
    strip = np.arange(len(order))[:, None]
    colors = [PATTERN_COLOR[pattern_of[i]] for i in order]
    ax_strip.imshow(strip, aspect="auto", cmap=ListedColormap(colors))
    ax_strip.set_xticks([])
    centers = []
    labels = []
    start = 0
    current = pattern_of[order[0]]
    for pos, cluster in enumerate(order + [None]):
        pattern = pattern_of[cluster] if cluster is not None else None
        if pattern != current:
            centers.append((start + pos - 1) / 2)
            labels.append(current)
            start = pos
            current = pattern
    ax_strip.set_yticks(centers, labels, fontsize=8)
    ax_strip.tick_params(length=0)
    for spine in ax_strip.spines.values():
        spine.set_visible(False)


def _fill_tiles(fig, tiles, summary, picks, cells, images) -> None:
    grouped = {name: frame for name, frame in picks.groupby("pattern", sort=False)}
    for row, item in enumerate(summary["patterns"]):
        pattern = item["pattern"]
        block = grouped.get(pattern, picks.iloc[0:0])
        color = PATTERN_COLOR.get(pattern, "#44403c")
        for col in range(4):
            ax = fig.add_subplot(tiles[row, col])
            if col >= len(block):
                ax.axis("off")
                continue
            sample = block.iloc[col]
            key = sample["key"] if "key" in sample else None
            points = cells[cells["key"] == key] if key is not None else cells.iloc[0:0]
            _draw_window(ax, points, sample, color, ylabel=pattern if col == 0 else None, image=images.get(key))


def _draw_window(ax, points, sample, border: str, ylabel: str | None, image=None) -> None:
    if image is not None:
        ax.imshow(image)
    else:
        if len(points):
            colors = [TYPE_COLOR.get(label, "#a8a29e") for label in points["annotation"]]
            ax.scatter(points["x"], points["y"], c=colors, s=3, linewidths=0, rasterized=True)
        x0 = float(sample["ix"]) * WINDOW_UM
        y0 = float(sample["iy"]) * WINDOW_UM
        ax.set_xlim(x0, x0 + WINDOW_UM)
        ax.set_ylim(y0 + WINDOW_UM, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{sample['slice']}  F{int(sample['fine'])}", fontsize=8, color=border)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12, color=border, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_color(border)
        spine.set_linewidth(2.4)


def _panel_c(path, summary, picks, cells, images) -> None:
    rows = len(summary["patterns"])
    fig = plt.figure(figsize=(12.4, 2.55 * rows), facecolor="white")
    tiles = fig.add_gridspec(rows, 4, wspace=0.14, hspace=0.42)
    _fill_tiles(fig, tiles, summary, picks, cells, images)
    if not images:
        used = sorted(set(cells["annotation"]))
        handles = [Patch(facecolor=TYPE_COLOR.get(label, "#a8a29e"), edgecolor="none", label=label) for label in used]
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    draw(args.root, args.out)


if __name__ == "__main__":
    main()
