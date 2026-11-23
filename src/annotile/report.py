"""Write the static page from a finished summary."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{item}</th>" for item in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{item}</td>" for item in row) + "</tr>")
    return "<table><thead><tr>" + head + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def render(summary: dict) -> str:
    method = summary["method"]
    sil = ", ".join(f"k={row['k']} → {row['silhouette']:.3f}" for row in method["silhouette"])
    bic = ", ".join(f"G={row['g']} → {row['bic']:.0f}" for row in method["bic"])
    drop_rows = []
    for item in summary["slices"]:
        dropped = item["dropped"]
        text = "无" if not dropped else "，".join(f"{name} {count}" for name, count in dropped.items())
        drop_rows.append([
            item["name"],
            f"{item['cells_annotated']:,}",
            _pct(item["join_rate"]),
            f"{item['cells_kept']:,}",
            f"{item['cells_in_windows']:,}",
            f"{item['cells_outside_windows']:,}",
            text,
        ])
    pattern_rows = []
    for item in summary["patterns"]:
        top = "、".join(
            f"{name} {_pct(item['annotation_fraction'][name])}" for name in item["top_annotations"]
        ) or "没有单一类型达到 15%"
        pattern_rows.append([
            item["pattern"],
            " ".join(item["fine_clusters"]),
            f"{item['windows']:,}",
            f"{item['cells']:,}",
            top,
        ])
    fraction_headers = ["切片"] + [item["pattern"] for item in summary["patterns"]]
    fraction_rows = []
    for name in summary["slice_names"]:
        vals = summary["slice_pattern_fraction"][name]
        fraction_rows.append([name] + [_pct(vals[item["pattern"]]) for item in summary["patterns"]])
    corr = summary["slice_correlation"]
    corr_headers = ["切片"] + summary["slice_names"]
    corr_rows = []
    for i, name in enumerate(summary["slice_names"]):
        corr_rows.append([name] + [f"{corr[i][j]:.2f}" for j in range(len(summary["slice_names"]))])
    composition_headers = ["细胞类型"] + [item["pattern"] for item in summary["patterns"]]
    composition_rows = []
    for label in summary["labels"]:
        fractions = [item["annotation_fraction"].get(label, 0.0) for item in summary["patterns"]]
        if max(fractions) < 0.02:
            continue
        composition_rows.append([label] + [_pct(value) for value in fractions])
    expression = summary.get("expression") or {}
    genes = expression.get("genes") or []
    values = expression.get("values") or {}
    expression_headers = ["基因"] + [item["pattern"] for item in summary["patterns"]]
    expression_rows = []
    for index, gene in enumerate(genes):
        expression_rows.append(
            [gene] + [f"{values.get(item['pattern'], [])[index]:.2f}" for item in summary["patterns"]]
        )
    top_rows = []
    for item in summary["patterns"]:
        picked = (expression.get("top") or {}).get(item["pattern"], [])
        text = "、".join(f"{row['gene']} {row['value']:.2f}" for row in picked) or "—"
        top_rows.append([item["pattern"], text])
    overlay_rows = []
    for item in summary["cluster_overlay"]:
        tops = "，".join(
            f"簇 {row['cluster']} {_pct(row['fraction'])}" for row in item["top_seurat"]
        )
        overlay_rows.append([item["slice"], item["pattern"], f"{item['cells']:,}", tops])
    bic_boundary = ""
    if method["bic"] and method["merged_patterns"] == method["bic"][-1]["g"]:
        bic_boundary = (
            "<p>BIC 一直降到搜索上限。上限由细簇个数决定，所以这次合并很少："
            "多数细簇各自留成一个模式。</p>"
        )
    kept = "、".join(summary["labels"])
    joins = []
    for item in summary["slices"]:
        rate = item.get("matrix_join")
        if rate is not None:
            joins.append(f"{item['name']} {_pct(rate)}")
    matrix_line = ""
    if joins:
        matrix_line = "<p>表达矩阵和注释细胞编号的匹配率：" + "，".join(joins) + "。</p>"
    portrait_blocks = []
    for item in summary["patterns"]:
        top = "、".join(
            f"{name} {_pct(item['annotation_fraction'][name])}" for name in item["top_annotations"]
        ) or "没有单一类型达到 15%"
        portrait_blocks.append(
            f"<h3>{item['pattern']}</h3>"
            f"<p>{item['cells']:,} 个细胞，{item['windows']:,} 个小窗。主要类型是 {top}。</p>"
            f"<img src=\"figures/pattern_{item['pattern']}.png\" alt=\"{item['pattern']} 在三张切片上的样子\">"
        )
    sources = "".join(
        f"<li><a href=\"{item['source']}\">{item['title']}</a>（{item['name']}）</li>"
        for item in summary["slices"]
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Annotile：带细胞注释的切片模式</title>
<style>
  body {{ margin: 0; background: #f6f4ef; color: #1c1917; font: 17px/1.65 "Source Han Sans SC", "Noto Sans SC", "Microsoft YaHei", sans-serif; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 40px 20px 80px; }}
  h1 {{ font-size: 34px; line-height: 1.25; margin-bottom: 8px; }}
  h2 {{ margin-top: 42px; font-size: 22px; }}
  p, li {{ max-width: 74ch; }}
  code {{ font-family: ui-monospace, Consolas, monospace; font-size: 0.92em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 14px 0 22px; background: white; font-size: 14.5px; }}
  th, td {{ border: 1px solid #e7e5e4; padding: 7px 9px; text-align: left; vertical-align: top; }}
  th {{ background: #292524; color: white; font-weight: 600; }}
  img {{ width: 100%; height: auto; background: white; border: 1px solid #e7e5e4; }}
  .step {{ color: #9a3412; font-weight: 700; }}
  a {{ color: #9a3412; }}
</style>
</head>
<body>
<main>
<h1>Annotile</h1>
<p>把已经标好细胞类型的空间切片划成小窗，把细胞类型、共有基因表达和配准后的 H&amp;E 外观合在一起聚类，得到可以在切片之间比较的组织模式。</p>
<p>输入来自三张公开的人肺 Xenium 切片。细胞类型注释来自 <a href="https://github.com/D10ny3u3/GHIST_input">GHIST_input</a>。细胞坐标、H&amp;E 和 <code>cell_feature_matrix.h5</code> 来自对应的 10x Genomics 公开数据。形态学流程参照 Zhang、Liu、Chen 等人在 <a href="https://doi.org/10.1186/s13073-025-01526-5">Genome Medicine 2025</a> 上的 HIPO：先得到一批细簇，再用高斯混合按 BIC 把相似细簇合并成模式，最后看每张切片由哪些模式组成。</p>
<p>HIPO 的细簇特征是 H&amp;E 图块的对比学习向量。这里的小窗同时使用细胞类型比例、三张切片共有基因的表达主成分，以及配准 H&amp;E 的颜色和纹理。细胞注释是比原文多出来的那一维。转录组 Seurat 簇没有放进跨切片的特征里，因为不同实验的簇编号不是同一种细胞状态；它们只在最后用来描述每个模式里混有哪些转录组簇。</p>

<h2><span class="step">数据</span></h2>
<ul>{sources}</ul>
<p>注释表只有细胞编号和类型。坐标按细胞编号拼回去。三张切片的编号匹配率都在下面的表里。随机种子是 {method['seed']}。</p>

<h2><span class="step">1. 留下可用的细胞注释</span></h2>
<p>注释说明里，<code>low_feat_cell</code>、<code>mph_CXCL9_region</code>、<code>fib?</code>，以及 CCL19 混合群（表中的 <code>lymph_endo?CCL19</code>）被标成混合或不采用。带问号、又不在保留名单里的标签也去掉。说明明确写了对照图像后保留的 <code>epi?</code> 和 <code>endo?</code> 留在分析里。<code>fib_SYNPO2</code> 按原表保留，说明里把它进一步看成血管平滑肌。</p>
<p>保留下来的类型：{kept}。</p>
{_table(["切片", "注释细胞", "坐标匹配", "过滤后", "进入小窗", "小窗以外", "去掉的标签"], drop_rows)}

<h2><span class="step">2. 把切片切成小窗</span></h2>
<p>每个细胞按质心落入边长 {method['window_um']:.0f} µm 的方格。少于 {method['min_cells']} 个细胞的方格去掉，相当于原文里丢掉组织太少的图块。三张切片一共 {summary['windows']:,} 个小窗。相邻小窗属于同一模式的比例是 {_pct(summary['neighbor_agreement'])}，用来看这些模式在组织里是成片的还是散点。</p>

<h2><span class="step">3. 小窗特征</span></h2>
<p>每个小窗先算细胞类型比例。三张切片再对上表达矩阵里共有的 {method.get('shared_genes', 0)} 个基因，取每个细胞的 log1p 标准化表达，在窗内平均后做主成分，这次保留 {method.get('expression_components', 0)} 个成分。H&amp;E 用配准矩阵裁出同一小窗，加入平均颜色、颜色波动和纹理。这三块各自标准化后拼在一起做聚类。</p>
{matrix_line}

<h2><span class="step">4. 细簇</span></h2>
<p>对全部小窗做 k-means。k 从 6 试到 20。每次在固定的 4000 个小窗（不足则全部）上计算轮廓系数，只有比当前选择高出 0.01 才换成更大的 k。选中的细簇数是 {method['fine_k']}。</p>
<p>{sil}</p>
<img src="figures/silhouette.png" alt="不同细簇数的轮廓系数">

<h2><span class="step">5. 合并成模式</span></h2>
<p>每个细簇取成员小窗的平均组成。对这些平均向量做标准化，再用对角协方差的高斯混合去拟合。组数从 2 试到细簇数减 1，并且不超过 8，取得分最低的 BIC。选中 {method['merged_patterns']} 个模式，按包含细胞数从多到少称为 P1、P2……。细簇之间的相关矩阵只用来展示哪些细簇被放在一起，不作为混合模型的输入：相关矩阵的行数和列数相同，满协方差估不稳。</p>
<p>{bic}</p>
{bic_boundary}
<img src="figures/bic.png" alt="不同模式数的 BIC">
<img src="figures/fine_correlation.png" alt="细簇平均组成的相关矩阵">
{_table(["模式", "包含的细簇", "小窗", "细胞", "主要细胞类型"], pattern_rows)}
<img src="figures/pattern_composition.png" alt="每个模式的细胞类型比例">

<h2><span class="step">每个模式的细胞组成</span></h2>
<p>下表是该模式里各细胞类型占该模式细胞数的比例。任一模式都不到 2% 的类型没有列出。</p>
{_table(composition_headers, composition_rows)}

<h2><span class="step">每个模式的基因表达</span></h2>
<p>表达矩阵参加了聚类：三张切片共有的 {expression.get('shared_genes', method.get('shared_genes', 0))} 个基因先在每个细胞里做成 log1p(CP10k)，再在小窗内平均，取 16 个主成分，和细胞组成、H&amp;E 一起使用。下表是每个模式相对其他模式升高最多的 8 个基因，数值是按细胞数加权的平均 log1p(CP10k)。热图把这些基因在全部模式里的同一套数值放在一起。</p>
{_table(["模式", "升高的基因和表达值"], top_rows)}
{_table(expression_headers, expression_rows)}
<img src="figures/pattern_expression.png" alt="每个模式的基因表达">

<h2><span class="step">每个模式的样子</span></h2>
<p>版式对应原文图 2。A 的每个点是一个小窗，t-SNE 用的是注释、表达和 H&amp;E 拼在一起的特征，颜色是细簇。B 是这些特征的细簇相关矩阵，只保留下三角。C 的每一格是该模式的一张代表性 H&amp;E 小窗，边框颜色和 B 的模式色一致。</p>
<img src="figures/fig2.png" alt="按原文图2排版的细簇、相关矩阵和代表性小窗">
<img src="figures/fig2c.png" alt="每个模式的代表性小窗">

<h2><span class="step">6. 每张切片的模式组成</span></h2>
<p>一张切片的向量是各个模式所占的细胞比例。下表是这次运行的结果。三张切片来自三个公开实验、各一个供体，切片之间的相关只能描述这三张片子，不能当成患者分型。</p>
{_table(fraction_headers, fraction_rows)}
<img src="figures/slice_patterns.png" alt="三张切片的模式比例">
<p>切片之间的 Pearson 相关：</p>
{_table(corr_headers, corr_rows)}
<img src="figures/space_5k.png" alt="5k 切片上的模式">
<img src="figures/space_lc.png" alt="lc 切片上的模式">
<img src="figures/space_luad.png" alt="luad 切片上的模式">

<h2><span class="step">7. 模式里的转录组簇</span></h2>
<p>Seurat 分辨率 0.3 的簇编号只在各自分片内部有意义。下表是每个模式里细胞数最多的三个转录组簇。</p>
{_table(["切片", "模式", "细胞", "主要转录组簇"], overlay_rows)}

<h2>复现</h2>
<p>在放有 <code>data/GHIST_input</code> 和三份 <code>data/centroids/*.csv.gz</code> 的目录运行：</p>
<p><code>python -m annotile run --root . --out results</code></p>
<p><code>python -m annotile report --result results/summary.json --docs docs</code></p>
<p>页面上的数字和 <code>results/summary.json</code> 是同一次运行写出来的。</p>
</main>
</body>
</html>
"""


def write_site(result: Path, docs: Path) -> None:
    summary = json.loads(result.read_text(encoding="utf-8"))
    docs.mkdir(parents=True, exist_ok=True)
    figure_src = result.parent / "figures"
    figure_dst = docs / "figures"
    if figure_dst.exists():
        shutil.rmtree(figure_dst)
    shutil.copytree(figure_src, figure_dst)
    (docs / "index.html").write_text(render(summary), encoding="utf-8")
