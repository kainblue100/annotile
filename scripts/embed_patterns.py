"""Map the 785 windows into t-SNE and UMAP, colored by pattern.

The features are the same matrix used for clustering: cell-type fractions,
16 expression components, and the H&E color, spread, and texture. Lines join
pattern centroids along the within-slice evolutionary tree. Red is the branch
played in the video.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.manifold import TSNE

ROOT = Path(r"E:\annotile")
RESULTS = ROOT / "results"
COLORS = {
    "P1": "#f4a261",
    "P2": "#2a9d8f",
    "P3": "#e9c46a",
    "P4": "#e76f51",
    "P5": "#6d597a",
    "P6": "#457b9d",
    "P7": "#b5838d",
    "P8": "#264653",
}


def _patterns() -> list[str]:
    with (RESULTS / "windows.csv").open(newline="", encoding="utf-8") as handle:
        return [row["pattern"] for row in csv.DictReader(handle)]


def _edges() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    tree = json.loads((RESULTS / "trajectory.json").read_text(encoding="utf-8"))
    gray = set()
    red = set()
    for item in tree["slices"]:
        for parent, kids in item["children"].items():
            for kid in kids:
                gray.add((parent, kid))
        path = item["main_path"]
        red.update(zip(path, path[1:]))
    return gray, red


def _centroids(embedding: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
    names = sorted(set(labels))
    out = {}
    for name in names:
        mask = np.array([label == name for label in labels])
        out[name] = embedding[mask].mean(axis=0)
    return out


def _panel(ax, embedding: np.ndarray, labels: list[str], title: str, gray, red) -> None:
    for name in [f"P{i}" for i in range(1, 9)]:
        mask = np.array([label == name for label in labels])
        if not mask.any():
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=16,
            c=COLORS[name],
            alpha=0.78,
            linewidths=0,
            label=name,
            zorder=2,
        )
    centers = _centroids(embedding, labels)
    for parent, kid in gray:
        if (parent, kid) in red or (kid, parent) in red:
            continue
        a, b = centers[parent], centers[kid]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#b5b5b5", lw=1.6, zorder=3)
    for parent, kid in red:
        a, b = centers[parent], centers[kid]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#b03a2e", lw=2.4, zorder=4)
    for name, point in centers.items():
        ax.scatter(point[0], point[1], s=90, c=COLORS[name], edgecolors="white", linewidths=1.2, zorder=5)
        ax.text(point[0], point[1], name, ha="center", va="center", fontsize=7, color="white", zorder=6, fontweight="bold")
    ax.set_title(title, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#e7e5e4")


def main() -> None:
    features = np.load(RESULTS / "features.npy")
    labels = _patterns()
    if len(labels) != len(features):
        raise SystemExit(f"{len(labels)} windows and {len(features)} feature rows")
    tsne = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(features)
    mapped = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42).fit_transform(features)
    gray, red = _edges()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.1), facecolor="white")
    _panel(axes[0], tsne, labels, "t-SNE", gray, red)
    _panel(axes[1], mapped, labels, "UMAP", gray, red)
    handles, names = axes[1].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    out = RESULTS / "figures" / "pattern_embedding.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    np.savez_compressed(RESULTS / "embedding_2d.npz", tsne=tsne.astype(np.float32), umap=mapped.astype(np.float32))
    print("WROTE", out)


if __name__ == "__main__":
    main()
