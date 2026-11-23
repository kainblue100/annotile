"""Within-slice spatial trajectories for the eight Annotile patterns.

The graph follows pseudo-time-space (Pham et al., Nat Commun 2023): a step is
allowed only between windows that share a side on the same section, and the
cost of that step is the distance in the window feature that already mixes
cell-type fractions, expression components, and H&E color and texture.
Pseudotime is the shortest such path from the window richest in differentiated
lung epithelium. Three donors are not joined into one tree.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"E:\annotile")
RESULTS = ROOT / "results"
DIFF = ("normal_epi", "AT1", "AT2_like", "cil")
GENES = ("EPCAM", "MKI67", "UBE2C", "EGFR", "COL5A2", "PECAM1")
COLORS = {
    "P1": (244, 162, 97), "P2": (42, 157, 143), "P3": (233, 196, 106), "P4": (231, 111, 81),
    "P5": (109, 89, 122), "P6": (69, 123, 157), "P7": (181, 131, 141), "P8": (38, 70, 83),
}
MIN_BORDERS = 3


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _dijkstra(start: int, neighbors: list[list[tuple[int, float]]], count: int) -> np.ndarray:
    dist = np.full(count, np.inf)
    dist[start] = 0.0
    used = np.zeros(count, dtype=bool)
    for _ in range(count):
        node = -1
        best = np.inf
        for i in range(count):
            if not used[i] and dist[i] < best:
                best = dist[i]
                node = i
        if node < 0 or not np.isfinite(best):
            break
        used[node] = True
        for nxt, weight in neighbors[node]:
            cand = dist[node] + weight
            if cand < dist[nxt]:
                dist[nxt] = cand
    return dist


def _path(graph: dict[str, list[tuple[str, float]]], start: str, end: str) -> list[str]:
    previous = {start: ""}
    dist = {start: 0.0}
    left = set(graph)
    while left:
        node = min(left, key=lambda name: dist.get(name, 1e18))
        if node not in dist or node == end:
            left.remove(node)
            if node == end:
                break
            continue
        left.remove(node)
        for nxt, weight in graph[node]:
            cand = dist[node] + weight
            if cand < dist.get(nxt, 1e18):
                dist[nxt] = cand
                previous[nxt] = node
    if end not in previous:
        return [start]
    order = []
    node = end
    while node:
        order.append(node)
        node = previous[node]
    return order[::-1]


def _expression() -> tuple[dict[str, dict[str, float]], list[str]]:
    rows = _read(RESULTS / "pattern_expression.csv")
    key = "" if "" in rows[0] else list(rows[0])[0]
    genes = [column for column in rows[0] if column != key]
    table = {row[key]: {gene: float(row[gene]) for gene in genes} for row in rows}
    return table, genes


def _children(graph: dict[str, list[tuple[str, float]]], root: str, pseudotime: dict[str, float]) -> dict[str, list[str]]:
    distance = {root: 0.0}
    previous = {root: ""}
    pending = set(graph)
    while pending:
        node = min(pending, key=lambda name: distance.get(name, 1e18))
        if node not in distance:
            break
        pending.remove(node)
        for nxt, weight in graph[node]:
            cand = distance[node] + weight
            if cand < distance.get(nxt, 1e18):
                distance[nxt] = cand
                previous[nxt] = node
    children: dict[str, list[str]] = defaultdict(list)
    for node, parent in previous.items():
        if parent:
            children[parent].append(node)
    for parent in children:
        children[parent].sort(key=lambda name: pseudotime.get(name, 0.0))
    return children


def _leaves(children: dict[str, list[str]], root: str) -> int:
    kids = children.get(root, [])
    if not kids:
        return 1
    return sum(_leaves(children, kid) for kid in kids)


def _place(node: str, children: dict[str, list[str]], depth: int, cursor: list[int], pos: dict[str, tuple[float, float]]) -> None:
    kids = children.get(node, [])
    if not kids:
        pos[node] = (float(depth), float(cursor[0]))
        cursor[0] += 1
        return
    for kid in kids:
        _place(kid, children, depth + 1, cursor, pos)
    pos[node] = (float(depth), sum(pos[kid][1] for kid in kids) / len(kids))


def _draw(slices: list[dict], out: Path) -> None:
    title = _font(26)
    label = _font(22)
    small = _font(16)
    panels = []
    for item in slices:
        pos: dict[str, tuple[float, float]] = {}
        _place(item["root_window"]["pattern"], item["children"], 0, [0], pos)
        panels.append((item, pos, _leaves(item["children"], item["root_window"]["pattern"])))
    height = 36 + sum(70 + max(leaves, 1) * 92 for _item, _pos, leaves in panels)
    canvas = Image.new("RGB", (1080, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    top = 28
    trunk = {(left, right) for item, _pos, _leaves in panels for left, right in zip(item["main_path"], item["main_path"][1:])}
    for item, pos, leaves in panels:
        draw.text((28, top), item["slice"], fill=(20, 20, 20), font=title)
        origin_y = top + 58
        points = {}
        for name, (depth, row) in pos.items():
            points[name] = (170 + depth * 210, origin_y + row * 92)
        root = item["root_window"]["pattern"]
        for parent, kids in item["children"].items():
            if not kids:
                continue
            px, py = points[parent]
            fork = px + 78
            child_y = [points[kid][1] for kid in kids]
            draw.line([(px + 28, py), (fork, py)], fill=(150, 150, 150), width=3)
            if len(kids) > 1:
                draw.line([(fork, min(child_y)), (fork, max(child_y))], fill=(150, 150, 150), width=3)
            for kid in kids:
                kx, ky = points[kid]
                color = (176, 58, 46) if (parent, kid) in trunk else (150, 150, 150)
                width = 7 if (parent, kid) in trunk else 3
                draw.line([(fork, ky), (kx - 30, ky)], fill=color, width=width)
                if len(kids) > 1:
                    draw.line([(fork, py), (fork, ky)], fill=color, width=width)
        for name, (x, y) in points.items():
            color = COLORS[name]
            draw.ellipse((x - 28, y - 28, x + 28, y + 28), fill=color, outline=(30, 30, 30), width=2)
            draw.text((x - 14, y - 12), name, fill=(255, 255, 255), font=label)
            when = next(row["pseudotime"] for row in item["patterns"] if row["pattern"] == name)
            draw.text((x - 22, y + 34), f"{when:.2f}", fill=(70, 70, 70), font=small)
        top += 70 + max(leaves, 1) * 92
    canvas.save(out)


def main() -> None:
    windows = _read(RESULTS / "windows.csv")
    features = np.load(RESULTS / "features.npy")
    if len(features) != len(windows):
        raise SystemExit(f"features {len(features)} != windows {len(windows)}")
    groups = defaultdict(list)
    for index, row in enumerate(windows):
        groups[row["slice"]].append(index)
    expression, gene_names = _expression()
    slices = []
    for slice_name in ("5k", "lc", "luad"):
        index = groups[slice_name]
        local = features[index]
        rows = [windows[i] for i in index]
        score = np.array([sum(float(row[name]) for name in DIFF) for row in rows])
        root_local = int(score.argmax())
        grid = {(int(row["ix"]), int(row["iy"])): i for i, row in enumerate(rows)}
        neighbors = [[] for _ in rows]
        borders = defaultdict(list)
        for i, row in enumerate(rows):
            ix, iy = int(row["ix"]), int(row["iy"])
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                other = grid.get((ix + dx, iy + dy))
                if other is None:
                    continue
                weight = float(np.linalg.norm(local[i] - local[other]))
                neighbors[i].append((other, weight))
                neighbors[other].append((i, weight))
                pair = tuple(sorted((row["pattern"], rows[other]["pattern"])))
                if pair[0] != pair[1]:
                    borders[pair].append(weight)
        distances = _dijkstra(root_local, neighbors, len(rows))
        finite = np.isfinite(distances)
        scale = float(np.median(distances[finite])) if finite.any() else 1.0
        scale = scale if scale > 0 else 1.0
        distances = distances / scale
        by_pattern = defaultdict(list)
        for i, row in enumerate(rows):
            if np.isfinite(distances[i]):
                by_pattern[row["pattern"]].append(float(distances[i]))
        pattern_rows = []
        for pattern, values in sorted(by_pattern.items()):
            pattern_rows.append({
                "pattern": pattern,
                "windows": len(values),
                "pseudotime": round(float(np.median(values)), 4),
                "differentiated": round(float(np.mean([
                    sum(float(rows[i][name]) for name in DIFF)
                    for i, row in enumerate(rows) if row["pattern"] == pattern and np.isfinite(distances[i])
                ])), 4),
            })
        edges = []
        graph = defaultdict(list)
        for (left, right), weights in borders.items():
            if len(weights) < MIN_BORDERS:
                continue
            if left not in by_pattern or right not in by_pattern:
                continue
            edges.append({
                "left": left,
                "right": right,
                "borders": len(weights),
                "feature_distance": round(float(np.median(weights)), 4),
            })
            graph[left].append((right, float(np.median(weights))))
            graph[right].append((left, float(np.median(weights))))
        root_pattern = rows[root_local]["pattern"]
        graph.setdefault(root_pattern, [])
        pseudotime = {item["pattern"]: item["pseudotime"] for item in pattern_rows}
        children = _children(graph, root_pattern, pseudotime)
        reachable = [root_pattern, *pseudotime]
        tip = max(pseudotime, key=lambda name: pseudotime[name])
        order = _path(graph, root_pattern, tip)
        gradient = []
        for gene in GENES:
            if gene not in gene_names:
                continue
            gradient.append({
                "gene": gene,
                "values": {name: round(expression[name][gene], 3) for name in order if name in expression},
            })
        slices.append({
            "slice": slice_name,
            "root_window": {"ix": int(rows[root_local]["ix"]), "iy": int(rows[root_local]["iy"]), "pattern": root_pattern},
            "root_score": round(float(score[root_local]), 4),
            "reachable_windows": int(finite.sum()),
            "windows": len(rows),
            "patterns": pattern_rows,
            "edges": edges,
            "children": {parent: kids for parent, kids in children.items()},
            "main_path": order,
            "expression_along_path": gradient,
        })
    payload = {
        "method": "Within each section, pseudotime is the shortest path along windows that share a side. The step cost is Euclidean distance in the saved window features: cell-type fraction, 16 expression components, and H&E mean, spread, and texture. The root window is the one with the highest normal_epi + AT1 + AT2_like + cil fraction. A pattern edge is kept when at least 3 window borders carry it. The three sections are not joined.",
        "references": [
            "Pham et al., Nat Commun 2023, pseudo-time-space: a trajectory step has to be a spatial neighbor, and H&E morphology is part of the state",
            "Ren et al., Nat Commun 2022, SpaceFlow: pseudotime on a graph that already contains spatial neighbors",
            "Kueckelhaus et al., Nat Commun 2024, SPATA2: read the expression change along the spatial path",
            "SpaTrack, Cell Systems 2025: transition cost combines expression distance and physical distance; separate donors are not one time course",
        ],
        "not_regvelo": "These Xenium matrices have no spliced or unspliced counts, so no RNA velocity was estimated",
        "feature_file": "results/features.npy",
        "feature_width": int(features.shape[1]),
        "slices": slices,
        "video_segments": [
            {"slice": item["slice"], "path": item["main_path"]}
            for item in slices if len(item["main_path"]) > 1
        ],
        "donors": "Three sections, one donor each. A path is the similarity order inside that section. It is not a lineage measured in one patient, and it is not a path across donors.",
    }
    (RESULTS / "trajectory.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _draw(slices, RESULTS / "figures" / "trajectory.png")
    print(json.dumps({
        "segments": payload["video_segments"],
        "slices": [
            {
                "slice": item["slice"],
                "root": item["root_window"],
                "path": item["main_path"],
                "edges": item["edges"],
                "pseudotime": {row["pattern"]: row["pseudotime"] for row in item["patterns"]},
            }
            for item in slices
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
