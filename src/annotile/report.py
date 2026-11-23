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
    volume = summary.get("volume") or {}
    volume_rows = [
        [item["pattern"], item["slice"], f"F{item['fine']}", f"{item['patch_um']:.0f} µm", str(item["cells"])]
        for item in volume.get("patterns", [])
    ]
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
  img, video {{ width: 100%; height: auto; background: white; border: 1px solid #e7e5e4; }}
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

<h2><span class="step">每个模式的三维模型</span></h2>
<p>平面位置用的是这个模式里细胞最密的一块组织上的 Xenium 质心。细胞半径按类型给定。细胞中心只能落在 10 µm 厚的切片里，球与球不能互相穿透，这是 CellSqueeze3D 那类物理约束。每个球分成膜、细胞质和核，对应 Alon、Yu、Zhou 用来合成三维细胞的三个部分。下图从左到右是俯视、侧面和切过切片中平面的截面。侧面里灰色带是切片的厚度，细胞可以凸出切片，因为切片切过的是更大的细胞。</p>
{_table(["模式", "切片", "细簇", "小块边长", "细胞"], volume_rows)}
<img src="figures/pattern_volume.png" alt="每个模式的俯视、侧面和截面">
<p><a href="volume.html">旋转查看这八个模型</a>。那里把深度放大了 4 倍，否则 10 µm 的厚度贴在几十微米宽的组织上几乎是一条线。球的半径没有被放大。</p>
<p>下面两张是用 Blender Cycles 渲染的同一批细胞。玻璃盒后面是这块组织自己的 H&amp;E。深度放大了 4 倍，半径仍是按细胞类型给定的微米。</p>
<img src="figures/blender_volume.png" alt="八个模式的玻璃盒渲染">
<p>近景是把玻璃盒里的球放大。每一格是该模式这块组织里半径最大的一颗细胞，再加上离它最近的两颗。位置是 Xenium 质心，大小是按细胞类型给定的半径。外层是半透明的膜，里面是细胞质和核，颜色收成紫红色，用来接近合成细胞图的近景。膜上的起伏是渲染时加上的，不是从 H&amp;E 描出来的细胞边界。</p>
<img src="figures/blender_closeup.png" alt="八个模式的细胞近景">

<h2><span class="step">模式之间的进化</span></h2>
<p>这里没有剪接和未剪接的计数，所以没有做 RegVelo 的 RNA velocity。每张切片单独做空间拟时序，做法对应 Pham 等人 2023 年 Nature Communications 里的 pseudo-time-space：只有边相邻的小窗才能连一步，这一步的代价是小窗特征的距离。特征就是聚类用的那一份，含细胞类型比例、153 个共有基因的 16 个主成分，以及 H&amp;E 的颜色、色差和纹理。根是这张切片上分化上皮比例最高的小窗，分化上皮按正常上皮、AT1、AT2 样和纤毛细胞相加。Ren 等人 2022 年的 SpaceFlow 也是在带空间邻居的图上算伪时间。Kueckelhaus 等人 2024 年的 SPATA2 沿空间路径读表达变化；这里基因表达只汇总到模式，所以下面的数字是模式之间的差。SpaTrack（Cell Systems 2025）把表达距离和物理距离一起放进转移代价。三张切片各一个供体，没有接成一条轨迹。</p>
<p>进化图按切片分开画。根是这张切片上分化上皮比例最高的小窗所在的模式。两个模式至少有 3 个相邻小窗才连边，每个模式挂到特征距离更短的上游。节点下的数字是伪时间。红枝通向伪时间最远的模式，视频播的就是这一支。</p>
<p>5k 上 P7 分出 P1、P6 和 P4，视频是 P7 → P4。这一支 EPCAM 从 2.06 降到 0.89，PECAM1 从 0.70 升到 2.18。P1 再连到只有 1 个小窗的 P3。lc 上没有 P7，根是 P5，分出 P3 和 P1，视频是 P5 → P1。EPCAM 从 1.84 升到 2.96，UBE2C 从 0.16 升到 0.25。P3 是这张片子里细胞最多的模式。luad 上 P7 分出 P1 和 P2，P2 再分出 P6 和 P8，视频是 P7 → P2 → P8。到 P2 时 EPCAM 到 4.32、MKI67 到 0.30；到 P8 时 PECAM1 到 4.03。三张切片各来自一个供体。这是同一张切片里的相似顺序，不是一个病人体内测到的谱系。</p>
<img src="figures/trajectory.png" alt="三张切片的进化树">
<p>下面三张把这棵树标在切片上。底图是配准后的 H&amp;E，圆点是 400 µm 小窗，颜色和编号对应 P1 到 P8。白箭头从上游模式的中心指向下游，较粗的是视频里的那一支。Xenium 只覆盖切片的一部分，所以点不会铺满整张 H&amp;E。</p>
<img src="figures/transition_5k.png" alt="5k 切片上的模式转移">
<img src="figures/transition_lc.png" alt="lc 切片上的模式转移">
<img src="figures/transition_luad.png" alt="luad 切片上的模式转移">
<video src="figures/evolution.mp4" controls playsinline poster="figures/trajectory.png"></video>
<p>视频按 5k、lc、luad 播放红枝。玻璃盒里的细胞从一块组织的质心变到下一块，颜色和大小跟着细胞类型变。深度放大 4 倍。用的是 EEVEE。玻璃盒后面是 BodyParts3D 的成人肺叶模型，来自 Database Center for Life Science，许可是 CC BY-SA 2.1 JP，只作为器官外形。这三张 Xenium 切片来自三个供体，不是同一肺的连续切片，所以没有把它们叠成 <a href="https://kainblue100.github.io/coda-mouse-lung-volume/">小鼠肺体积</a>那种序列重建。细胞没有跨模式追踪。</p>
<h2><span class="step">同一例肺癌的连续切片</span></h2>
<p>公开数据里有一套人类肺癌连续切片：Pentimalli 等人，Cell Systems 2025。一块非小细胞肺癌蜡块切了 34 张连续 5 µm 切片，第 4、10、16、22、28、34 张做了 CosMx，层间距 30 µm，用 STIM 配准。坐标在 <a href="https://doi.org/10.5281/zenodo.7899173">Zenodo 10.5281/zenodo.7899173</a>。下图用的是他们公布的配准坐标，5.6 像素等于 1 µm。六层共 340,644 个细胞位置，范围约 4.0 mm × 4.0 mm × 150 µm。相邻两层、80 µm 格子上的细胞密度 Pearson 相关依次是 0.69、0.49、0.50、0.67、0.54。空白区在六层里都还在，说明配准后的结构能对上。</p>
<p>这六层是同一例肿瘤，没有和 5k、lc、luad 叠到一起，也没有把 P1–P8 标上去。坐标表只有位置，细胞类型在他们的 Seurat 对象里，这次没有读。电子显微镜和 CT 没有拿来顶替：连续切片本身是公开的，CT 给不出这些小窗里的细胞组成。</p>
<img src="figures/serial_nsclc.png" alt="同一例非小细胞肺癌的六张配准切片">
<h2><span class="step">程序可以叠在同一个模式上</span></h2>
<p>公众号那篇讲的是 Yan 等人，Nature 2026，三阴性乳腺癌化疗前的四层：archetype 是患者级癌细胞程序，metaprogram 是同一肿瘤里可叠加的表达程序，ecotype 是患者之间哪些状态一起出现或一起缺失，spatial niche 才是组织坐标上真正相邻的局部组成。Ecotype 不等于空间相邻。原文是治疗前活检和后来病理完全缓解的关联，癌种不是肺。</p>
<p>这里没有照做 archetype 和 ecotype。供体只有 3 个，没有化疗结局，153 个共有基因里没有他们用的 ISG15、IFI27、IFIT3 和 HLA-DRA。CXCL9 和 STC2 各是一个基因，不是他们的干扰素程序或低氧程序。用得上的是 niche 这一层，以及程序可以叠加：一个模式不必只属于一种状态。</p>
<p>下图的数是每个基因在 8 个模式上的 z 分数，再在一组基因里平均。P2 上皮最高（+1.38），增殖并不高（−0.10）。P3 的增殖（+2.59）、基质（+1.32）和 CXCL9（+2.01）同时高；P3 在 lc 上占了大部分细胞，这个分数主要来自 lc。P5 内皮 +1.95，P8 内皮 +1.14。视频那一支的细胞比例：5k 从 P7 到 P4，上皮 0.47 降到 0.21，内皮 0.06 升到 0.34。lc 从 P5 到 P1，上皮 0.27 升到 0.48，内皮 0.43 降到 0.07，淋巴样细胞 0.13 升到 0.26。luad 从 P7 到 P2 再到 P8，上皮是 0.53、0.59、0.11，到 P8 时内皮升到 0.41。</p>
<img src="figures/niche_programs.png" alt="八个模式上可叠加的表达程序">
<p>同一套 785 个小窗再降到二维。每个点是一个小窗，颜色是模式，大点是该模式全部分窗的中心。灰线和红线是上面的进化树，画到这两个坐标里；红线仍是视频里的那一支。t-SNE 用困惑度 30、PCA 初始化和种子 42。UMAP 用 15 个邻居、min_dist 0.1 和种子 42。线表示切片上相邻，不表示这两个图里的距离。P2 在两张图里都单独成团，P1 和 P3 挨在一起，P4 自成一大团，P6、P7、P8 是小团。</p>
<img src="figures/pattern_embedding.png" alt="模式在 t-SNE 和 UMAP 中的位置">

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
    volume_page = result.parent / "volume.html"
    if volume_page.is_file():
        shutil.copyfile(volume_page, docs / "volume.html")
    (docs / "index.html").write_text(render(summary), encoding="utf-8")
