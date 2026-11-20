# Annotile

Annotile 把注释好的空间切片聚成组织模式。小窗的特征是细胞类型比例，对应 HIPO 流程里的图块特征，并多出细胞注释这一维。

Zhang、Liu、Chen 等，Genome Medicine，2025。DOI [10.1186/s13073-025-01526-5](https://doi.org/10.1186/s13073-025-01526-5)。

流程说明和这次三张肺 Xenium 切片的每一步结果在 [介绍页](docs/index.html)。

## 数据

细胞注释来自 [D10ny3u3/GHIST_input](https://github.com/D10ny3u3/GHIST_input) 的 `data_input`。细胞坐标是这三份公开的 10x `cells.csv.gz`：

- Prime 5K：`Xenium_Prime_Human_Lung_Cancer_FFPE_cells.csv.gz`
- 免疫组库面板：`Xenium_V1_Human_Lung_Cancer_Addon_FFPE_cells.csv.gz`
- 多模态分割：`Xenium_V1_humanLung_Cancer_FFPE_cells.csv.gz`

目录：

```text
data/GHIST_input/
data/centroids/5k_cells.csv.gz
data/centroids/lc_cells.csv.gz
data/centroids/luad_cells.csv.gz
```

## 运行

```bash
python -m venv .venv
.venv/bin/pip install -e .
python -m annotile run --root . --out results
python -m annotile report --result results/summary.json --docs docs
```

种子是 42。这次发表的结果使用边长 400 µm 的小窗，少于 30 个细胞的小窗不进入聚类。这个尺度接近 HIPO 在 5 倍镜下 224 像素图块覆盖的组织范围。
