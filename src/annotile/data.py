"""Slice tables and the annotation labels kept for pattern finding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SliceSpec:
    name: str
    title: str
    annotation: str
    clusters: str
    centroids: str
    source: str
    centroid_url: str
    matrix: str
    he_image: str
    he_alignment: str
    patient: str = ""
    group: str = ""


# Shared bins so Xenium labels and CosMx labels can sit in one composition.
# Neutrophils are counted with myeloid: the Xenium tables have no neutrophil label,
# and a column that is zero on every Xenium slice would mark the platform.
# Club-like cells count as airway here. Where the annotation table of the Takano
# sections places a club-like or AT2-like cluster in the tumor region, the import
# script has already rewritten that label to "tumor" for that section.
COARSE = {
    "AT1": "alveolar",
    "AT1_like": "alveolar",
    "AT2_like": "alveolar",
    "normal_epi": "alveolar",
    "cil": "airway",
    "club_like": "airway",
    "club_1_like": "airway",
    "club_2_like": "airway",
    "epi": "tumor",
    "epi?": "tumor",
    "epi_secretory": "tumor",
    "fib": "fibroblast",
    "fib_activated": "fibroblast",
    "fib_SYNPO2": "fibroblast",
    "fib_PI16": "fibroblast",
    "fib_IL6": "fibroblast",
    "vasc_fib": "fibroblast",
    "pericyte": "fibroblast",
    "pericytes": "fibroblast",
    "endo": "endothelial",
    "endo?": "endothelial",
    "endo_art": "endothelial",
    "lymph_endo": "endothelial",
    "mph": "myeloid",
    "mph_SPP1": "myeloid",
    "LAMP3_DC": "myeloid",
    "plasma_DC": "myeloid",
    "neutrophil": "myeloid",
    "nkt": "t_nk",
    "t": "t_nk",
    "Tfh_CXCL13_region": "t_nk",
    "b": "b_cell",
    "plasma": "plasma",
    "mast": "mast",
    "alveolar": "alveolar",
    "airway": "airway",
    "tumor": "tumor",
    "fibroblast": "fibroblast",
    "endothelial": "endothelial",
    "myeloid": "myeloid",
    "t_nk": "t_nk",
    "b_cell": "b_cell",
}


def to_coarse(label: str) -> str:
    return COARSE.get(label, "")


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
        matrix="data/matrices/5k_cell_feature_matrix.h5",
        he_image="data/he/5k_he_image.ome.tif",
        he_alignment="data/he/5k_he_imagealignment.csv",
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
        matrix="data/matrices/lc_cell_feature_matrix.h5",
        he_image="data/he/lc_he_image.ome.tif",
        he_alignment="data/he/lc_he_imagealignment.csv",
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
        matrix="data/matrices/luad_cell_feature_matrix.h5",
        he_image="data/he/luad_he_image.ome.tif",
        he_alignment="data/he/luad_he_imagealignment.csv",
    ),
)

# Notes shipped with the annotation tables mark these groups as mixed,
# unused, or too uncertain to keep. epi? and endo? are kept: the lc note
# says those two were checked against the image and retained.
DROP_EXACT = frozenset(
    {
        "low_feat_cell",
        "mph_CXCL9_region",
        "CXCL9_region",
        "fib?",
        "nkt?CCL19",
        "lymph_endo?CCL19",
        # Takano sections: unnamed clusters, doublets, a mixed niche cluster,
        # and one cluster that pools every immune cell of a low-count section.
        "unknown",
        "unknown_IL6",
        "immune",
        "plasma_fib_doub",
        "mph_alv_epi_doub",
        "fib_SYNPO2_endo_art_niche",
    }
)
KEEP_MARKED = frozenset({"epi?", "endo?"})


def load_cohort(root: Path) -> tuple[SliceSpec, ...]:
    """The three 10x Xenium slices, plus the slices listed in a manifest.

    The manifest is data/public/manifest.json unless ANNOTILE_MANIFEST names
    another file. Paths in a manifest may be absolute, so large raw folders
    can stay where they are.
    """
    import json
    import os

    chosen = os.environ.get("ANNOTILE_MANIFEST", "")
    manifest = Path(chosen) if chosen else root / "data" / "public" / "manifest.json"
    if not manifest.is_file():
        if chosen:
            raise FileNotFoundError(f"ANNOTILE_MANIFEST points to a missing file: {manifest}")
        return _only(SLICES)
    extra = []
    for item in json.loads(manifest.read_text(encoding="utf-8")):
        extra.append(
            SliceSpec(
                name=item["name"],
                title=item["title"],
                annotation=item["annotation"],
                clusters=item.get("clusters", ""),
                centroids=item["centroids"],
                source=item["source"],
                centroid_url=item.get("centroid_url", ""),
                matrix=item["matrix"],
                he_image=item.get("he_image", ""),
                he_alignment=item.get("he_alignment", ""),
                patient=item.get("patient", item["name"]),
            )
        )
    return _only(SLICES + tuple(extra))


def _only(specs: tuple[SliceSpec, ...]) -> tuple[SliceSpec, ...]:
    """ANNOTILE_ONLY=tk_ keeps sections whose names start with that prefix."""
    import os

    prefix = os.environ.get("ANNOTILE_ONLY", "").strip()
    if not prefix:
        return specs
    chosen = tuple(spec for spec in specs if spec.name.startswith(prefix))
    if not chosen:
        raise RuntimeError(f"ANNOTILE_ONLY={prefix} matched no sections")
    return chosen


def keep_label(label: str) -> bool:
    if label in DROP_EXACT:
        return False
    if "?" in label and label not in KEEP_MARKED:
        return False
    return True
