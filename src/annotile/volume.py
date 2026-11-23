"""3D cell models of each pattern.

Plane positions are the Xenium centroids. Depth and size follow the same
kind of constraints as CellSqueeze3D: centers stay inside a histological
section, radii stay near a cell-type prior, and spheres do not interpenetrate.
Each cell is drawn as a membrane, a cytoplasm, and a nucleus, the three parts
used by the synthetic cell models of Alon, Yu, and Zhou.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from annotile.fig2 import PATTERN_COLOR
from annotile.portraits import TYPE_COLOR

THICKNESS_UM = 10.0
PATCH_UM = 40.0
MAX_CELLS = 160
RADIUS_UM = {
    "epi": 8.5,
    "epi?": 8.0,
    "normal_epi": 7.5,
    "AT1": 6.5,
    "AT2_like": 7.0,
    "cil": 7.5,
    "endo": 5.5,
    "endo?": 5.5,
    "lymph_endo": 5.0,
    "fib": 7.0,
    "fib_activated": 7.5,
    "fib_SYNPO2": 6.5,
    "nkt": 4.5,
    "Tfh_CXCL13_region": 5.0,
    "b": 4.2,
    "plasma": 5.0,
    "mph": 7.0,
    "mast": 5.5,
}


def _radius(label: str) -> float:
    return RADIUS_UM.get(label, 6.0)


def _densest(cells: pd.DataFrame, size: float) -> pd.DataFrame:
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


def _patch(cells: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    for size in (PATCH_UM, 80.0, 120.0):
        chosen = _densest(cells, size)
        if len(chosen) >= 20 or size == 120.0:
            if len(chosen) > MAX_CELLS:
                center = chosen[["x", "y"]].median()
                distance = (chosen["x"] - center["x"]) ** 2 + (chosen["y"] - center["y"]) ** 2
                chosen = chosen.loc[distance.nsmallest(MAX_CELLS).index]
            return chosen.copy(), size
    return cells.copy(), PATCH_UM


def place(xy: np.ndarray, radius: np.ndarray, thickness: float = THICKNESS_UM, seed: int = 42) -> np.ndarray:
    """Push cell centers apart along depth until the spheres stop intersecting."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.0, thickness, size=len(radius))
    for _ in range(60):
        delta = z[:, None] - z[None, :]
        flat = xy[:, None, :] - xy[None, :, :]
        dist = np.sqrt(np.sum(flat * flat, axis=2) + delta * delta) + 1e-6
        overlap = radius[:, None] + radius[None, :] - dist
        np.fill_diagonal(overlap, 0.0)
        overlap = np.maximum(overlap, 0.0)
        push = (overlap * np.sign(delta)).sum(axis=1)
        hits = np.maximum((overlap > 0).sum(axis=1), 1)
        z = np.clip(z + 0.45 * push / hits, 0.0, thickness)
    return z


def _window_cells(cells: pd.DataFrame, row) -> pd.DataFrame:
    x0 = float(row.ix) * 400.0
    y0 = float(row.iy) * 400.0
    block = cells[(cells["slice"] == row.slice) & (cells["x"] >= x0) & (cells["x"] < x0 + 400.0) & (cells["y"] >= y0) & (cells["y"] < y0 + 400.0)]
    return block


def build_volumes(windows: pd.DataFrame, cells: pd.DataFrame, summary: dict, fig_dir: Path, page: Path) -> dict:
    fig_dir.mkdir(parents=True, exist_ok=True)
    models = []
    for item in summary["patterns"]:
        block = windows[windows["pattern"] == item["pattern"]].sort_values("n_cells", ascending=False)
        chosen = pd.DataFrame()
        host = None
        span = PATCH_UM
        for row in block.itertuples(index=False):
            found = _window_cells(cells, row)
            if found.empty:
                continue
            chosen, span = _patch(found)
            host = row
            if len(chosen) >= 20:
                break
        if host is None or chosen.empty:
            continue
        xy = chosen[["x", "y"]].to_numpy(dtype=float)
        xy = xy - xy.min(axis=0)
        labels = chosen["annotation"].astype(str).tolist()
        radius = np.array([_radius(label) for label in labels], dtype=float)
        depth = place(xy, radius)
        models.append({
            "pattern": item["pattern"],
            "slice": str(host.slice),
            "fine": int(host.fine),
            "patch_um": span,
            "cells_in_patch": int(len(chosen)),
            "cells": [
                {
                    "x": round(float(xy[i, 0]), 2),
                    "y": round(float(xy[i, 1]), 2),
                    "z": round(float(depth[i]), 2),
                    "r": round(float(radius[i]), 2),
                    "type": labels[i],
                    "color": TYPE_COLOR.get(labels[i], "#a8a29e"),
                }
                for i in range(len(labels))
            ],
        })
    _draw(fig_dir / "pattern_volume.png", models)
    payload = {"thickness_um": THICKNESS_UM, "patterns": models}
    page.write_text(_page(payload), encoding="utf-8")
    summary["volume"] = {
        "thickness_um": THICKNESS_UM,
        "patterns": [
            {
                "pattern": item["pattern"],
                "slice": item["slice"],
                "fine": item["fine"],
                "patch_um": item["patch_um"],
                "cells": item["cells_in_patch"],
            }
            for item in models
        ],
    }
    return summary["volume"]


def _draw(path: Path, models: list[dict]) -> None:
    rows = max(len(models), 1)
    fig, axes = plt.subplots(rows, 3, figsize=(9.6, 2.55 * rows), squeeze=False, facecolor="white")
    titles = ("top", "side", "midsection")
    for index, model in enumerate(models):
        cells = model["cells"]
        xy = np.array([[cell["x"], cell["y"]] for cell in cells])
        z = np.array([cell["z"] for cell in cells])
        radius = np.array([cell["r"] for cell in cells])
        colors = [cell["color"] for cell in cells]
        _top(axes[index, 0], xy, z, radius, colors)
        _side(axes[index, 1], xy[:, 0], z, radius, colors, model["patch_um"])
        _cut(axes[index, 2], xy, z, radius, colors, model["patch_um"])
        axes[index, 0].set_ylabel(f"{model['pattern']}  {model['slice']}", fontsize=9)
        for col, title in enumerate(titles):
            if index == 0:
                axes[index, col].set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _paint(ax, order, draw) -> None:
    for i in order:
        draw(i)


def _top(ax, xy, z, radius, colors) -> None:
    span = max(float(xy.max()) if len(xy) else 1.0, 1.0)
    order = np.argsort(z)

    def draw(i):
        membrane = Circle((xy[i, 0], xy[i, 1]), radius[i] * 1.12, facecolor=colors[i], edgecolor="none", alpha=0.28)
        body = Circle((xy[i, 0], xy[i, 1]), radius[i], facecolor=colors[i], edgecolor="#1c1917", linewidth=0.25, alpha=0.9)
        nucleus = Circle((xy[i, 0], xy[i, 1]), radius[i] * 0.42, facecolor="#1c1917", edgecolor="none", alpha=0.55)
        ax.add_patch(membrane)
        ax.add_patch(body)
        ax.add_patch(nucleus)

    _paint(ax, order, draw)
    ax.set_xlim(-2, span + 2)
    ax.set_ylim(span + 2, -2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _side(ax, x, z, radius, colors, span) -> None:
    order = np.argsort(x)

    def draw(i):
        ax.add_patch(Circle((x[i], z[i]), radius[i] * 1.12, facecolor=colors[i], edgecolor="none", alpha=0.22))
        ax.add_patch(Circle((x[i], z[i]), radius[i], facecolor=colors[i], edgecolor="#1c1917", linewidth=0.25, alpha=0.9))
        ax.add_patch(Circle((x[i], z[i]), radius[i] * 0.42, facecolor="#1c1917", edgecolor="none", alpha=0.55))

    _paint(ax, order, draw)
    ax.axhspan(0, THICKNESS_UM, color="#e7e5e4", zorder=0)
    ax.set_xlim(-2, span + 2)
    ax.set_ylim(-4, THICKNESS_UM + 8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _cut(ax, xy, z, radius, colors, span) -> None:
    plane = THICKNESS_UM / 2
    hit = np.abs(z - plane) < radius
    order = np.argsort(np.abs(z - plane))
    for i in order:
        if not hit[i]:
            continue
        reach = float(np.sqrt(max(radius[i] ** 2 - (z[i] - plane) ** 2, 0.0)))
        ax.add_patch(Circle((xy[i, 0], xy[i, 1]), reach, facecolor=colors[i], edgecolor="#1c1917", linewidth=0.25, alpha=0.92))
        core = min(reach * 0.55, radius[i] * 0.42)
        ax.add_patch(Circle((xy[i, 0], xy[i, 1]), core, facecolor="#1c1917", edgecolor="none", alpha=0.5))
    ax.set_xlim(-2, span + 2)
    ax.set_ylim(span + 2, -2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _page(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    colors = json.dumps(PATTERN_COLOR)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Annotile 模式的三维模型</title>
<style>
  body {{ margin: 0; background: #f6f4ef; color: #1c1917; font: 16px/1.6 "Microsoft YaHei", sans-serif; }}
  header {{ max-width: 1100px; margin: 0 auto; padding: 28px 20px 0; }}
  a {{ color: #9a3412; }}
  #bar {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }}
  button {{ border: 2px solid #e7e5e4; background: white; border-radius: 8px; padding: 6px 12px; cursor: pointer; font: inherit; }}
  button.on {{ color: white; }}
  #view {{ width: min(1100px, calc(100% - 24px)); height: 72vh; margin: 0 auto 12px; display: block; background: white; border: 1px solid #e7e5e4; }}
  p {{ max-width: 78ch; }}
</style>
</head>
<body>
<header>
<p><a href="index.html">返回说明页</a></p>
<h1>每个模式的三维细胞模型</h1>
<p id="caption"></p>
<div id="bar"></div>
</header>
<canvas id="view"></canvas>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const DATA = {data};
const PATTERN = {colors};
const ZSCALE = 4;
const canvas = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio || 1);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 5000);
camera.position.set(80, 90, 120);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(20, 20, 20);
scene.add(new THREE.AmbientLight(0xffffff, 0.72));
const sun = new THREE.DirectionalLight(0xffffff, 0.75);
sun.position.set(40, 80, 60);
scene.add(sun);
const root = new THREE.Group();
scene.add(root);
function sphere(cell, scale, opacity) {{
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(cell.r * scale, 22, 16),
    new THREE.MeshPhongMaterial({{ color: cell.color, transparent: opacity < 1, opacity, shininess: 30 }})
  );
  mesh.position.set(cell.x, cell.z * ZSCALE, cell.y);
  return mesh;
}}
function show(model) {{
  while (root.children.length) root.remove(root.children[0]);
  const slab = new THREE.Mesh(
    new THREE.BoxGeometry(model.patch_um, DATA.thickness_um * ZSCALE, model.patch_um),
    new THREE.MeshPhongMaterial({{ color: 0xe7e5e4, transparent: true, opacity: 0.28, depthWrite: false }})
  );
  slab.position.set(model.patch_um / 2, DATA.thickness_um * ZSCALE / 2, model.patch_um / 2);
  root.add(slab);
  model.cells.forEach((cell) => {{
    root.add(sphere(cell, 1.12, 0.18));
    root.add(sphere(cell, 1, 0.94));
    const nucleus = sphere(cell, 0.42, 0.85);
    nucleus.material = new THREE.MeshPhongMaterial({{ color: 0x1c1917 }});
    root.add(nucleus);
  }});
  controls.target.set(model.patch_um / 2, DATA.thickness_um * ZSCALE / 2, model.patch_um / 2);
  document.getElementById("caption").textContent =
    model.pattern + " · " + model.slice + " · 细簇 F" + model.fine +
    " · " + model.cells_in_patch + " 个细胞，取自该模式里最密的 " + model.patch_um +
    " µm 见方。灰色盒子是 " + DATA.thickness_um + " µm 厚的切片。显示时深度放大了 " + ZSCALE + " 倍，球的半径仍是微米。拖动可旋转。";
  document.querySelectorAll("#bar button").forEach((button) => {{
    const on = button.dataset.pattern === model.pattern;
    button.classList.toggle("on", on);
    button.style.background = on ? PATTERN[model.pattern] : "white";
  }});
}}
const bar = document.getElementById("bar");
DATA.patterns.forEach((model, index) => {{
  const button = document.createElement("button");
  button.textContent = model.pattern;
  button.dataset.pattern = model.pattern;
  button.style.borderColor = PATTERN[model.pattern];
  button.onclick = () => show(model);
  bar.appendChild(button);
  if (index === 0) show(model);
}});
function frame() {{
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== width || canvas.height !== height) {{
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
  }}
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}}
frame();
</script>
</body>
</html>
"""
