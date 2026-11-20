"""Slice tables and the annotation labels kept for pattern finding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SliceSpec:
    name: str
    title: str
    annotation: str
    clusters: str
    centroids: str
    source: str
    centroid_url: str


SLICES: tuple[SliceSpec, ...] = (
    SliceSpec(
        name="5k",
        title="Prime 5K 肺腺癌",
        annotation="data/GHIST_input/data_input/5k/xenium_5kt_by_annotation.csv",
        clusters="data/GHIST_input/data_input/5k/xenium_5kt_by_seu_clusters_res_0.3.csv",
        centroids="data/centroids/5k_cells.csv.gz",
        source="https://www.10xgenomics.com/datasets/xenium-human-lung-cancer-post-xenium-technote",
        centroid_url=(
            "https://cf.10xgenomics.com/samples/xenium/3.0.0/"
            "Xenium_Prime_Human_Lung_Cancer_FFPE/"
            "Xenium_Prime_Human_Lung_Cancer_FFPE_cells.csv.gz"
        ),
    ),
    SliceSpec(
        name="lc",
        title="免疫组库面板肺癌",
        annotation="data/GHIST_input/data_input/lc/xenium_lc_by_annotation.csv",
        clusters="data/GHIST_input/data_input/lc/xenium_lc_by_seu_clusters_res_0.3.csv",
        centroids="data/centroids/lc_cells.csv.gz",
        source=(
            "https://www.10xgenomics.com/datasets/"
            "ffpe-human-lung-cancer-data-with-human-immuno-oncology-profiling-panel-and-custom-add-on-1-standard"
        ),
        centroid_url=(
            "https://cf.10xgenomics.com/samples/xenium/2.0.0/"
            "Xenium_V1_Human_Lung_Cancer_Addon_FFPE/"
            "Xenium_V1_Human_Lung_Cancer_Addon_FFPE_cells.csv.gz"
        ),
    ),
    SliceSpec(
        name="luad",
        title="多模态分割肺腺癌",
        annotation="data/GHIST_input/data_input/luad/01_seurat_process/xenium_luad_by_annotation.csv",
        clusters="data/GHIST_input/data_input/luad/01_seurat_process/xenium_luad_by_seu_clusters_res_0.3.csv",
        centroids="data/centroids/luad_cells.csv.gz",
        source=(
            "https://www.10xgenomics.com/datasets/"
            "preview-data-ffpe-human-lung-cancer-with-xenium-multimodal-cell-segmentation-1-standard"
        ),
        centroid_url=(
            "https://cf.10xgenomics.com/samples/xenium/2.0.0/"
            "Xenium_V1_humanLung_Cancer_FFPE/"
            "Xenium_V1_humanLung_Cancer_FFPE_cells.csv.gz"
        ),
    ),
)

# Notes shipped with the annotation tables mark these groups as mixed,
# unused, or too uncertain to keep. epi? and endo? are kept: the lc note
# says those two were checked against the image and retained.
DROP_EXACT = frozenset(
    {
        "low_feat_cell",
        "mph_CXCL9_region",
        "fib?",
        "nkt?CCL19",
        "lymph_endo?CCL19",
    }
)
KEEP_MARKED = frozenset({"epi?", "endo?"})


def keep_label(label: str) -> bool:
    if label in DROP_EXACT:
        return False
    if "?" in label and label not in KEEP_MARKED:
        return False
    return True
