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


def render(summary: dict, figures: set[str] | None = None) -> str:
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
    space_images = "".join(
        f"<img src=\"figures/space_{name}.png\" alt=\"{name} 上的模式\">"
        for name in summary["slice_names"]
    )
    sources = "".join(
        f"<li><a href=\"{item['source']}\">{item['title']}</a>（{item['name']}）</li>"
        for item in summary["slices"]
    )
    serial = summary.get("serial_fraction_correlation") or []
    if serial:
        bits = [
            f"{item['patient']}（{'、'.join(item['sections'])}）{item['pearson']:.2f}"
            for item in serial
        ]
        serial_line = "<p>同一病人多张切片的模式比例 Pearson 相关：" + "；".join(bits) + "。</p>"
    else:
        serial_line = ""
    n_slices = len(summary["slices"])
    n_patients = len(summary.get("patients") or [])
    centered = method.get("expression_centering") == "per-section"
    center_line = "每个切片先减去自己的均值，再取主成分，切片之间的整体高低不会单独变成一个模式。" if centered else "再取主成分。"

    def has(name: str) -> bool:
        return figures is None or name in figures

    def image(name: str, alt: str) -> str:
        if not has(name):
            return ""
        return f'<img src="figures/{name}" alt="{alt}">'
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
<p>把已经标好细胞类型的空间切片划成小窗，把细胞类型和共有基因表达合在一起聚类，得到可以在切片之间比较的组织模式。</p>
<p>这一版队列共 {n_slices} 张切片、{n_patients} 个病人。三张公开的人肺 Xenium 切片，细胞类型来自 <a href="https://github.com/D10ny3u3/GHIST_input">GHIST_input</a>，坐标和表达来自 10x Genomics。He 等人 <a href="https://doi.org/10.1038/s41587-022-01483-z">Nature Biotechnology 2022</a> 的 CosMx 非小细胞肺癌是五个病人、八张切片（<a href="https://doi.org/10.6084/m9.figshare.25976224">10.6084/m9.figshare.25976224</a>）。Takano 等人 <a href="https://doi.org/10.1038/s41467-024-54671-7">Nature Communications 2024</a> 的 17 张 Xenium 肺腺癌也在里面：5 张浸润性、12 张早期，每张切片一位病人，细胞类型是原作者按切片聚类后写上的名字。只在三张 Xenium 上跑的时候，相邻小窗一致率是 83.6%，八个模式里只有一个在三张切片上都达到 5% 的细胞。</p>
<p>聚类分两步，对应 Wang、Venet、Sotiriou 等人 <a href="https://doi.org/10.1038/s41467-024-54145-w">Nature Communications 2024</a> 的做法：每张切片内部先做 k-means，再把这些局部簇的平均特征放到一起做第二次 k-means，得到切片之间共用的模式。一个病人若有多张切片，这些切片的总权重等于只有一张切片的病人。细胞类型先收成同一套粗类，否则各批数据的名字对不上。CosMx 没有和 Xenium 配准的 H&amp;E，共同特征里不含 H&amp;E。种子是 {method['seed']}。</p>

<h2><span class="step">数据</span></h2>
<ul>{sources}</ul>
<p>注释表只有细胞编号和类型。坐标按细胞编号拼回去。三张切片的编号匹配率都在下面的表里。随机种子是 {method['seed']}。</p>

<h2><span class="step">1. 留下可用的细胞注释</span></h2>
<p>注释说明里，<code>low_feat_cell</code>、<code>mph_CXCL9_region</code>、<code>fib?</code>，以及 CCL19 混合群（表中的 <code>lymph_endo?CCL19</code>）被标成混合或不采用。带问号、又不在保留名单里的标签也去掉。说明明确写了对照图像后保留的 <code>epi?</code> 和 <code>endo?</code> 留在分析里。<code>fib_SYNPO2</code> 按原表保留，说明里把它进一步看成血管平滑肌。Takano 这 17 张里，未命名簇、双细胞、把整张切片免疫细胞合成一类的 <code>immune</code>，以及一个成纤维和内皮混在一起的生态位簇，也去掉。</p>
<p>保留下来的类型：{kept}。</p>
{_table(["切片", "注释细胞", "坐标匹配", "过滤后", "进入小窗", "小窗以外", "去掉的标签"], drop_rows)}

<h2><span class="step">2. 把切片切成小窗</span></h2>
<p>每个细胞按质心落入边长 {method['window_um']:.0f} µm 的方格。少于 {method['min_cells']} 个细胞的方格去掉。这次一共 {summary['windows']:,} 个小窗。相邻小窗属于同一模式的比例是 {_pct(summary['neighbor_agreement'])}。达到 5% 细胞、并且出现在至少两个病人里的模式有 {summary.get('patterns_in_two_patients', 0)} 个。</p>

<h2><span class="step">3. 小窗特征</span></h2>
<p>每个小窗先算粗类细胞的比例。全部切片共有的 {method.get('shared_genes', 0)} 个基因，在细胞里做成 log1p(CP10k)，窗内平均后，{center_line}主成分是 {method.get('expression_components', 0)} 个。有配准 H&amp;E 的切片才会加入颜色和纹理；这一版因为 CosMx 没有配准 H&amp;E，共同特征里没有这一块。</p>
{matrix_line}

<h2><span class="step">4. 细簇</span></h2>
<p>每一张切片单独做 k-means。k 从 4 试到 12，轮廓系数至少高出 0.01 才换成更大的 k。占该切片细胞不足 5% 的局部簇并进最近的簇。局部簇一共 {method['fine_k']} 个。</p>
<p>{sil}</p>
<img src="figures/silhouette.png" alt="不同细簇数的轮廓系数">

<h2><span class="step">5. 合并成模式</span></h2>
<p>第二次 k-means 的输入是每个局部簇的平均特征，不是小窗本身。k 试 4、6、8，同样要求轮廓系数高出 0.01。选中 {method['merged_patterns']} 个共用模式，按细胞数从多到少称为 P1、P2……。下图的数是这次第二次聚类的轮廓系数，不是 BIC。</p>
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
<p>表达矩阵参加了聚类：各切片共有的 {expression.get('shared_genes', method.get('shared_genes', 0))} 个基因先在每个细胞里做成 log1p(CP10k)，再在小窗内平均，取 {method.get('expression_components', 0)} 个主成分，和细胞组成一起使用。下表是每个模式相对其他模式升高最多的 8 个基因，数值是按细胞数加权的平均 log1p(CP10k)。</p>
{_table(["模式", "升高的基因和表达值"], top_rows)}
{_table(expression_headers, expression_rows)}
<img src="figures/pattern_expression.png" alt="每个模式的基因表达">

<h2><span class="step">每个模式的三维模型</span></h2>
<p>平面位置用的是这个模式里细胞最密的一块组织上的细胞质心。细胞半径按类型给定。细胞中心只能落在 10 µm 厚的切片里，球与球不能互相穿透。下图从左到右是俯视、侧面和切过切片中平面的截面。</p>
{_table(["模式", "切片", "细簇", "小块边长", "细胞"], volume_rows)}
<img src="figures/pattern_volume.png" alt="每个模式的俯视、侧面和截面">
<p><a href="volume.html">旋转查看这些模型</a>。深度放大了 4 倍。上一版的 Blender 玻璃盒和近景是按当时的 P1–P8 渲染的，没有放在这一版页面上。</p>

<h2><span class="step">模式之间的进化</span></h2>
<p>每张切片单独做空间拟时序，对应 Pham 等人 2023 年 Nature Communications 的 pseudo-time-space：只有边相邻的小窗才能连一步，代价是这一版聚类用的特征距离。根是肺泡或气道上皮比例最高的小窗。切片之间不接成一条轨迹，也没有做 RNA velocity。</p>
<p>进化图按切片分开画。根是这张切片上分化上皮比例最高的小窗所在的模式。两个模式至少有 3 个相邻小窗才连边，每个模式挂到特征距离更短的上游。节点下的数字是伪时间。红枝通向伪时间最远的模式，视频播的就是这一支。</p>
<p>上一版三张 Xenium 上的 P7→P4 等路径不再沿用。这一版每张切片从肺泡或气道上皮比例最高的小窗往外走，只连接空间上相邻的小窗。图里的编号是这一版的模式。切片之间不接成一条轨迹。</p>
{image("trajectory.png", "各切片的空间路径")}
<p>上一版的转移箭头、玻璃盒和视频是按当时的 P1–P8 渲染的，编号对不上这一版，所以没有再放在这页上。</p>
<h2><span class="step">同一例肺癌的连续切片</span></h2>
<p>公开数据里有一套人类肺癌连续切片：Pentimalli 等人，Cell Systems 2025。一块非小细胞肺癌蜡块切了 34 张连续 5 µm 切片，第 4、10、16、22、28、34 张做了 CosMx，层间距 30 µm，用 STIM 配准。坐标在 <a href="https://doi.org/10.5281/zenodo.7899173">Zenodo 10.5281/zenodo.7899173</a>。下图用的是他们公布的配准坐标，5.6 像素等于 1 µm。六层共 340,644 个细胞位置，范围约 4.0 mm × 4.0 mm × 150 µm。相邻两层、80 µm 格子上的细胞密度 Pearson 相关依次是 0.69、0.49、0.50、0.67、0.54。空白区在六层里都还在，说明配准后的结构能对上。</p>
<p>这六层是同一例肿瘤，没有和 5k、lc、luad 叠到一起，也没有把 P1–P8 标上去。坐标表只有位置，细胞类型在他们的 Seurat 对象里，这次没有读。电子显微镜和 CT 没有拿来顶替：连续切片本身是公开的，CT 给不出这些小窗里的细胞组成。</p>
{image("serial_nsclc.png", "同一例非小细胞肺癌的六张配准切片")}
<h2><span class="step">程序可以叠在同一个模式上</span></h2>
<p>公众号那篇讲的是 Yan 等人，Nature 2026，三阴性乳腺癌化疗前的四层：archetype 是患者级癌细胞程序，metaprogram 是同一肿瘤里可叠加的表达程序，ecotype 是患者之间哪些状态一起出现或一起缺失，spatial niche 才是组织坐标上真正相邻的局部组成。Ecotype 不等于空间相邻。原文是治疗前活检和后来病理完全缓解的关联，癌种不是肺。</p>
<p>这里没有照做 archetype 和 ecotype。这一版是 {n_patients} 个病人、{method.get('shared_genes', 0)} 个共有基因，没有化疗结局。旧的三张切片有 153 个共有基因，里面也没有他们用的 ISG15、IFI27、IFIT3 和 HLA-DRA。用得上的是空间上的模式：一个模式不必只属于一种状态。</p>
<p>没有化疗结局，所以没有做患者级 archetype，也没有把共现网络叫成 ecotype。</p>
{image("niche_programs.png", "模式上可叠加的表达程序")}
{image("pattern_embedding.png", "模式在 t-SNE 和 UMAP 中的位置")}

<h2><span class="step">6. 每张切片的模式组成</span></h2>
<p>一张切片的向量是各个模式所占的细胞比例。相关只能说明这些切片彼此像不像，不能单独当成治疗分型。同一病人的连续切片若都在队列里，它们的比例相关写在表后。</p>
{_table(fraction_headers, fraction_rows)}
<img src="figures/slice_patterns.png" alt="每张切片的模式比例">
<p>切片之间的 Pearson 相关：</p>
{_table(corr_headers, corr_rows)}
{serial_line}
{space_images}

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
    names = {path.name for path in figure_dst.iterdir()} if figure_dst.is_dir() else set()
    (docs / "index.html").write_text(render(summary, names), encoding="utf-8")
