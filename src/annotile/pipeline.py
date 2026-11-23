"""Window composition, fine clusters, and BIC-merged tissue patterns."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from annotile.data import SLICES, SliceSpec, keep_label

SEED = 42
WINDOW_UM = 400.0
MIN_CELLS = 30
K_CANDIDATES = (6, 8, 10, 12, 14, 16, 18, 20)
G_MAX = 8


def _xy(columns: list[str]) -> tuple[str, str]:
    lower = {c.lower(): c for c in columns}
    x = lower.get("x_centroid") or lower.get("x_location")
    y = lower.get("y_centroid") or lower.get("y_location")
    if x is None or y is None:
        raise RuntimeError(f"centroid table has no x/y columns: {columns}")
    return x, y


def load_slice(root: Path, spec: SliceSpec) -> tuple[pd.DataFrame, dict]:
    ann = pd.read_csv(root / spec.annotation, dtype=str)
    clu = pd.read_csv(root / spec.clusters, dtype=str)
    cen = pd.read_csv(root / spec.centroids)
    xcol, ycol = _xy(list(cen.columns))
    cen = cen.rename(columns={xcol: "x", ycol: "y"})
    cen["cell_id"] = cen["cell_id"].astype(str)
    ann = ann.rename(columns={"group": "annotation"})
    clu = clu.rename(columns={"group": "seurat"})
    merged = ann[["cell_id", "annotation"]].merge(
        clu[["cell_id", "seurat"]], on="cell_id", how="left"
    )
    n_table = int(len(merged))
    merged = merged.merge(cen[["cell_id", "x", "y"]], on="cell_id", how="inner")
    n_xy = int(len(merged))
    if n_table == 0 or n_xy / n_table < 0.5:
        raise RuntimeError(
            f"{spec.name}: only {n_xy} of {n_table} annotated cells matched centroids"
        )
    merged["keep"] = merged["annotation"].map(keep_label)
    dropped = (
        merged.loc[~merged["keep"], "annotation"].value_counts().astype(int).to_dict()
    )
    kept = merged.loc[merged["keep"]].copy()
    kept["x"] = pd.to_numeric(kept["x"], errors="coerce")
    kept["y"] = pd.to_numeric(kept["y"], errors="coerce")
    kept = kept.dropna(subset=["x", "y"])
    info = {
        "name": spec.name,
        "title": spec.title,
        "source": spec.source,
        "cells_annotated": n_table,
        "cells_with_centroid": n_xy,
        "join_rate": round(n_xy / n_table, 4),
        "cells_kept": int(len(kept)),
        "dropped": dropped,
    }
    return kept, info


def windows_for(cells: pd.DataFrame, labels: list[str], window_um: float, min_cells: int) -> pd.DataFrame:
    work = cells.copy()
    work["ix"] = np.floor(work["x"].to_numpy() / window_um).astype(np.int32)
    work["iy"] = np.floor(work["y"].to_numpy() / window_um).astype(np.int32)
    counts = (
        work.groupby(["ix", "iy", "annotation"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=labels, fill_value=0)
    )
    totals = counts.sum(axis=1)
    keep = totals >= min_cells
    counts = counts.loc[keep]
    totals = totals.loc[keep]
    fractions = counts.div(totals, axis=0)
    fractions["n_cells"] = totals.to_numpy()
    return fractions.reset_index()


def choose_k(features: np.ndarray) -> tuple[int, list[dict], np.ndarray]:
    rng = np.random.default_rng(SEED)
    sample_n = min(4000, len(features))
    sample = features[rng.choice(len(features), sample_n, replace=False)]
    rows = []
    best_k = K_CANDIDATES[0]
    best_score = -1.0
    labels = None
    for k in K_CANDIDATES:
        if k >= len(features):
            continue
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        full = model.fit_predict(features)
        score = float(silhouette_score(sample, model.predict(sample), metric="euclidean"))
        rows.append({"k": k, "silhouette": round(score, 4)})
        if score > best_score + 0.01:
            best_score = score
            best_k = k
            labels = full
        elif labels is None:
            labels = full
            best_k = k
            best_score = score
    if labels is None:
        raise RuntimeError("no fine clustering candidate could be fit")
    return best_k, rows, labels


def merge_patterns(means: np.ndarray) -> tuple[np.ndarray, int, list[dict]]:
    scaled = StandardScaler().fit_transform(means)
    upper = min(G_MAX, len(means) - 1)
    records = []
    best = None
    best_g = None
    best_bic = None
    for g in range(2, upper + 1):
        model = GaussianMixture(
            n_components=g,
            covariance_type="diag",
            reg_covar=1e-3,
            n_init=10,
            random_state=SEED,
        )
        label = model.fit_predict(scaled)
        bic = float(model.bic(scaled))
        records.append({"g": g, "bic": round(bic, 2)})
        if best_bic is None or bic < best_bic:
            best_bic = bic
            best = label
            best_g = g
    if best is None or best_g is None:
        raise RuntimeError("gaussian mixture was not fit")
    return best, best_g, records


def _scale(block: np.ndarray) -> np.ndarray:
    center = block.mean(axis=0)
    spread = block.std(axis=0)
    spread = np.where(spread < 1e-8, 1.0, spread)
    return (block - center) / spread


def _features(root: Path, window: pd.DataFrame, labels: list[str], window_um: float) -> tuple[np.ndarray, dict]:
    from annotile.he import he_block
    from annotile.matrix import expression_block

    blocks = [_scale(window[labels].to_numpy(dtype=np.float64))]
    note = ["cell-annotation fraction"]
    details = {"shared_genes": 0, "expression_components": 0, "he": False, "matrix_joins": {}}
    expr, expr_info = expression_block(root, window, window_um)
    if expr is not None:
        blocks.append(_scale(expr))
        components = int(expr_info.get("expression_components", expr.shape[1]))
        note.append(f"{components} expression components from {expr_info.get('shared_genes', 0)} shared genes")
        details["shared_genes"] = int(expr_info.get("shared_genes", 0))
        details["expression_components"] = components
        details["matrix_joins"] = {
            name: item.get("matrix_join") for name, item in expr_info.get("slices", {}).items()
        }
    image, _image_info = he_block(root, window, window_um)
    if image is not None:
        blocks.append(_scale(image))
        note.append("H&E mean color, color spread, and texture")
        details["he"] = True
    return np.hstack(blocks), {"feature": "; ".join(note), **details}


def _corr(means: np.ndarray) -> np.ndarray:
    if len(means) < 2:
        return np.ones((len(means), len(means)))
    return np.corrcoef(means)


def run(root: Path, out: Path, window_um: float = WINDOW_UM, min_cells: int = MIN_CELLS) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    slice_info = []
    frames = []
    for spec in SLICES:
        cells, info = load_slice(root, spec)
        cells["slice"] = spec.name
        frames.append(cells)
        slice_info.append(info)
    all_cells = pd.concat(frames, ignore_index=True)
    labels = sorted(all_cells["annotation"].unique())
    parts = []
    for spec in SLICES:
        part = windows_for(all_cells[all_cells["slice"] == spec.name], labels, window_um, min_cells)
        part["slice"] = spec.name
        parts.append(part)
    window = pd.concat(parts, ignore_index=True)
    features, feature_info = _features(root, window, labels, window_um)
    np.save(out / "features.npy", features.astype(np.float32))
    for item in slice_info:
        item["matrix_join"] = feature_info["matrix_joins"].get(item["name"])
    fine_k, silhouette_rows, fine = choose_k(features)
    window["fine"] = fine
    means = np.vstack([features[fine == i].mean(axis=0) for i in range(fine_k)])
    correlation = _corr(means)
    raw_pattern, g_chosen, bic_rows = merge_patterns(means)
    # Name patterns by how many cells they cover, largest first.
    cell_weight = window.groupby("fine")["n_cells"].sum()
    totals = {}
    for i, p in enumerate(raw_pattern.tolist()):
        totals[p] = totals.get(p, 0) + int(cell_weight.get(i, 0))
    order = sorted(totals, key=lambda p: (-totals[p], p))
    rename = {old: f"P{rank}" for rank, old in enumerate(order, start=1)}
    fine_to_pattern = {i: rename[int(p)] for i, p in enumerate(raw_pattern.tolist())}
    window["pattern"] = window["fine"].map(fine_to_pattern)

    pattern_rows = []
    for name in [rename[p] for p in order]:
        block = window[window["pattern"] == name]
        composition = block[labels].multiply(block["n_cells"], axis=0).sum(axis=0)
        composition = (composition / composition.sum()).astype(float)
        members = sorted(i for i, p in fine_to_pattern.items() if p == name)
        pattern_rows.append(
            {
                "pattern": name,
                "fine_clusters": [f"F{i}" for i in members],
                "windows": int(len(block)),
                "cells": int(block["n_cells"].sum()),
                "annotation_fraction": {lab: round(float(composition[lab]), 4) for lab in labels},
                "top_annotations": [
                    lab for lab, val in composition.sort_values(ascending=False).items() if val >= 0.15
                ],
            }
        )

    slice_fraction = {}
    cluster_overlay = []
    for spec in SLICES:
        block = window[window["slice"] == spec.name]
        covered = int(block["n_cells"].sum())
        fractions = {}
        for row in pattern_rows:
            n = int(block.loc[block["pattern"] == row["pattern"], "n_cells"].sum())
            fractions[row["pattern"]] = round(n / covered, 4) if covered else 0.0
        slice_fraction[spec.name] = fractions
        cells = all_cells[all_cells["slice"] == spec.name].copy()
        cells["ix"] = np.floor(cells["x"].to_numpy() / window_um).astype(np.int32)
        cells["iy"] = np.floor(cells["y"].to_numpy() / window_um).astype(np.int32)
        assigned = cells.merge(
            block[["ix", "iy", "pattern"]], on=["ix", "iy"], how="inner"
        )
        for pattern, sub in assigned.groupby("pattern"):
            top = sub["seurat"].fillna("NA").value_counts(normalize=True).head(3)
            cluster_overlay.append(
                {
                    "slice": spec.name,
                    "pattern": pattern,
                    "cells": int(len(sub)),
                    "top_seurat": [
                        {"cluster": str(k), "fraction": round(float(v), 4)} for k, v in top.items()
                    ],
                }
            )
        info = next(item for item in slice_info if item["name"] == spec.name)
        info["windows"] = int(len(block))
        info["cells_in_windows"] = covered
        info["cells_outside_windows"] = int(info["cells_kept"] - covered)

    names = [row["pattern"] for row in pattern_rows]
    matrix = np.array([[slice_fraction[spec.name][p] for p in names] for spec in SLICES])
    if len(SLICES) > 1:
        slice_corr = np.corrcoef(matrix).round(4).tolist()
    else:
        slice_corr = [[1.0]]

    agreement = _neighbor_agreement(window)
    summary = {
        "method": {
            "window_um": window_um,
            "min_cells": min_cells,
            "seed": SEED,
            "fine_k": fine_k,
            "silhouette": silhouette_rows,
            "merged_patterns": g_chosen,
            "bic": bic_rows,
            "feature": feature_info["feature"],
            "shared_genes": feature_info["shared_genes"],
            "expression_components": feature_info["expression_components"],
            "he": feature_info["he"],
        },
        "labels": labels,
        "slices": slice_info,
        "windows": int(len(window)),
        "neighbor_agreement": round(agreement, 4),
        "patterns": pattern_rows,
        "slice_pattern_fraction": slice_fraction,
        "slice_correlation": slice_corr,
        "slice_names": [spec.name for spec in SLICES],
        "cluster_overlay": cluster_overlay,
    }
    _write_tables(out, window, labels, summary, correlation, fine_to_pattern)
    _figures(fig_dir, window, labels, summary, correlation, fine_to_pattern, all_cells, window_um)
    from annotile.matrix import attach_expression
    attach_expression(root, window, summary, fig_dir)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _neighbor_agreement(window: pd.DataFrame) -> float:
    keys = {
        (row.slice, int(row.ix), int(row.iy)): row.pattern
        for row in window.itertuples(index=False)
    }
    same = 0
    total = 0
    for (sl, ix, iy), pattern in keys.items():
        for dx, dy in ((1, 0), (0, 1)):
            other = keys.get((sl, ix + dx, iy + dy))
            if other is None:
                continue
            total += 1
            same += int(other == pattern)
    return same / total if total else 0.0


def _write_tables(out, window, labels, summary, correlation, fine_to_pattern) -> None:
    rows = []
    for item in summary["patterns"]:
        row = {
            "pattern": item["pattern"],
            "windows": item["windows"],
            "cells": item["cells"],
            "fine_clusters": " ".join(item["fine_clusters"]),
            "top_annotations": " ".join(item["top_annotations"]),
        }
        row.update(item["annotation_fraction"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "patterns.csv", index=False)
    frac = pd.DataFrame(summary["slice_pattern_fraction"]).T
    frac.index.name = "slice"
    frac.to_csv(out / "slice_fractions.csv")
    fine_rows = []
    for i, pattern in sorted(fine_to_pattern.items()):
        fine_rows.append({"fine": f"F{i}", "pattern": pattern, "mean_correlation_to_self_pattern": ""})
    order = sorted(fine_to_pattern)
    corr = pd.DataFrame(correlation, index=[f"F{i}" for i in order], columns=[f"F{i}" for i in order])
    corr.to_csv(out / "fine_correlation.csv")
    pd.DataFrame(
        [{"fine": f"F{i}", "pattern": fine_to_pattern[i]} for i in order]
    ).to_csv(out / "fine_to_pattern.csv", index=False)
    window[["slice", "ix", "iy", "n_cells", "fine", "pattern"] + labels].to_csv(
        out / "windows.csv", index=False
    )


def _figures(fig_dir, window, labels, summary, correlation, fine_to_pattern, cells, window_um) -> None:
    names = [row["pattern"] for row in summary["patterns"]]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(names), 1)))
    color_of = {name: colors[i] for i, name in enumerate(names)}

    comp = np.array([
        [row["annotation_fraction"][lab] for lab in labels] for row in summary["patterns"]
    ])
    fig, ax = plt.subplots(figsize=(10, 3.2 + 0.28 * len(names)))
    im = ax.imshow(comp, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(0.5, float(comp.max())))
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names)
    ax.set_title("Annotation fraction in each pattern")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(fig_dir / "pattern_composition.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bottom = np.zeros(len(summary["slice_names"]))
    x = np.arange(len(summary["slice_names"]))
    for name in names:
        vals = np.array([summary["slice_pattern_fraction"][sl][name] for sl in summary["slice_names"]])
        ax.bar(x, vals, bottom=bottom, color=color_of[name], label=name, width=0.7)
        bottom += vals
    ax.set_xticks(x, summary["slice_names"])
    ax.set_ylabel("Cell fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Pattern composition of each slice")
    ax.legend(ncol=min(len(names), 5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "slice_patterns.png", dpi=140)
    plt.close(fig)

    order = sorted(fine_to_pattern)
    ordered = correlation[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(ordered, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(order)), [f"F{i}" for i in order], rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)), [f"F{i}\n{fine_to_pattern[i]}" for i in order], fontsize=7)
    ax.set_title("Correlation of fine-cluster means")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(fig_dir / "fine_correlation.png", dpi=140)
    plt.close(fig)

    bic = summary["method"]["bic"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot([row["g"] for row in bic], [row["bic"] for row in bic], marker="o")
    chosen = summary["method"]["merged_patterns"]
    ax.axvline(chosen, color="#b45309", lw=1)
    ax.set_xlabel("Number of patterns")
    ax.set_ylabel("BIC")
    ax.set_title("Mixture size chosen by BIC")
    fig.tight_layout()
    fig.savefig(fig_dir / "bic.png", dpi=140)
    plt.close(fig)

    sil = summary["method"]["silhouette"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot([row["k"] for row in sil], [row["silhouette"] for row in sil], marker="o")
    ax.axvline(summary["method"]["fine_k"], color="#b45309", lw=1)
    ax.set_xlabel("Fine clusters")
    ax.set_ylabel("Silhouette")
    ax.set_title("Fine cluster size")
    fig.tight_layout()
    fig.savefig(fig_dir / "silhouette.png", dpi=140)
    plt.close(fig)

    for spec_name in summary["slice_names"]:
        sub = cells[cells["slice"] == spec_name]
        grid = window[window["slice"] == spec_name][["ix", "iy", "pattern"]]
        mapped = sub.copy()
        mapped["ix"] = np.floor(mapped["x"].to_numpy() / window_um).astype(np.int32)
        mapped["iy"] = np.floor(mapped["y"].to_numpy() / window_um).astype(np.int32)
        mapped = mapped.merge(grid, on=["ix", "iy"], how="inner")
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        for name in names:
            block = mapped[mapped["pattern"] == name]
            if block.empty:
                continue
            ax.scatter(
                block["x"], block["y"], s=0.15, c=[color_of[name]], linewidths=0, label=name, rasterized=True
            )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(spec_name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=12, fontsize=8, frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(fig_dir / f"space_{spec_name}.png", dpi=140)
        plt.close(fig)
    from annotile.portraits import draw_portraits
    draw_portraits(window, summary, fig_dir)
