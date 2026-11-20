"""Draw one picture of each tissue pattern."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

TYPE_COLOR = {
    "epi": "#c2410c",
    "epi?": "#fb923c",
    "normal_epi": "#fdba74",
    "AT1": "#fde68a",
    "AT2_like": "#eab308",
    "cil": "#0f766e",
    "endo": "#1d4ed8",
    "endo?": "#60a5fa",
    "lymph_endo": "#312e81",
    "fib": "#65a30d",
    "fib_activated": "#166534",
    "fib_SYNPO2": "#a3e635",
    "nkt": "#7c3aed",
    "Tfh_CXCL13_region": "#c4b5fd",
    "b": "#db2777",
    "plasma": "#f9a8d4",
    "mph": "#b45309",
    "mast": "#78716c",
}


def _rgb(value: str) -> np.ndarray:
    return np.array([int(value[i : i + 2], 16) for i in (1, 3, 5)], dtype=float) / 255


def _colors(block: pd.DataFrame, labels: list[str]) -> np.ndarray:
    palette = np.vstack([_rgb(TYPE_COLOR.get(label, "#a8a29e")) for label in labels])
    mixed = block[labels].to_numpy(dtype=float) @ palette
    return np.clip(mixed, 0, 1)


def draw_portraits(windows: pd.DataFrame, summary: dict, fig_dir: Path) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = summary["labels"]
    slices = summary["slice_names"]
    limits = {}
    for name in slices:
        block = windows[windows["slice"] == name]
        limits[name] = (
            int(block["ix"].min()) - 0.6,
            int(block["ix"].max()) + 1.6,
            int(block["iy"].min()) - 0.6,
            int(block["iy"].max()) + 1.6,
        )
    written = []
    for item in summary["patterns"]:
        pattern = item["pattern"]
        path = fig_dir / f"pattern_{pattern}.png"
        _one(windows, item, labels, slices, limits, path)
        written.append(path.name)
    return written


def _one(windows, item, labels, slices, limits, path: Path) -> None:
    pattern = item["pattern"]
    ranked = sorted(item["annotation_fraction"].items(), key=lambda kv: -kv[1])
    shown = [(name, value) for name, value in ranked if value >= 0.04][:6]
    fig, axes = plt.subplots(1, len(slices), figsize=(4.1 * len(slices), 5.4), squeeze=False)
    for ax, name in zip(axes[0], slices):
        tissue = windows[windows["slice"] == name]
        chosen = tissue[tissue["pattern"] == pattern]
        for ix, iy in zip(tissue["ix"].to_numpy(), tissue["iy"].to_numpy()):
            ax.add_patch(Rectangle((ix, iy), 1, 1, facecolor="#e7e5e4", edgecolor="none"))
        colors = _colors(chosen, labels) if len(chosen) else np.zeros((0, 3))
        for ix, iy, color in zip(chosen["ix"].to_numpy(), chosen["iy"].to_numpy(), colors):
            ax.add_patch(Rectangle((ix, iy), 1, 1, facecolor=color, edgecolor="none"))
        x0, x1, y0, y1 = limits[name]
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{name}   {len(chosen)} windows", fontsize=11)
        for spine in ax.spines.values():
            spine.set_color("#d6d3d1")
    bits = "   ".join(f"{name} {100 * value:.0f}%" for name, value in shown)
    fig.suptitle(f"{pattern}    {bits}", fontsize=13, y=0.98)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=TYPE_COLOR.get(name, "#a8a29e"), edgecolor="none", label=f"{name} {100 * value:.0f}%")
        for name, value in shown
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main_from_results(out: Path) -> None:
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    windows = pd.read_csv(out / "windows.csv")
    names = draw_portraits(windows, summary, out / "figures")
    print(" ".join(names))
