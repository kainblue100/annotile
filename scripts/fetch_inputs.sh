#!/usr/bin/env bash
# Download the annotation tables and the matching 10x cell-centroid files.
set -euo pipefail
root="${1:-.}"
mkdir -p "$root/data/centroids"
if [ ! -d "$root/data/GHIST_input/.git" ]; then
  git clone --depth 1 https://github.com/D10ny3u3/GHIST_input.git "$root/data/GHIST_input"
fi
fetch() {
  url="$1"
  dest="$2"
  if [ -s "$dest" ]; then
    echo "have $dest"
    return
  fi
  curl -L --fail --retry 3 --retry-delay 2 -o "$dest.partial" "$url"
  mv "$dest.partial" "$dest"
}
base="https://cf.10xgenomics.com/samples/xenium"
fetch "$base/3.0.0/Xenium_Prime_Human_Lung_Cancer_FFPE/Xenium_Prime_Human_Lung_Cancer_FFPE_cells.csv.gz" "$root/data/centroids/5k_cells.csv.gz"
fetch "$base/2.0.0/Xenium_V1_Human_Lung_Cancer_Addon_FFPE/Xenium_V1_Human_Lung_Cancer_Addon_FFPE_cells.csv.gz" "$root/data/centroids/lc_cells.csv.gz"
fetch "$base/2.0.0/Xenium_V1_humanLung_Cancer_FFPE/Xenium_V1_humanLung_Cancer_FFPE_cells.csv.gz" "$root/data/centroids/luad_cells.csv.gz"
