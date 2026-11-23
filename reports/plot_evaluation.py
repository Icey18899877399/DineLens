"""Export standalone academic plots from the saved ASAP test results."""

import argparse
import csv
import json
from itertools import groupby
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

from aspect_detection.visualization_data import load_visualization_data

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs" / "autodl-20261123" / "runs"
DEFAULT_OUTPUT = Path(__file__).with_name("figures")
MM = 1 / 25.4
INK = "#242424"
GRID = "#dedede"
BLUE = "#486988"
RED = "#973e50"
ASPECT_STYLE = (
    {"facecolor": "#f5f8fa", "edgecolor": BLUE, "hatch": "///", "linewidth": .62},
    {"facecolor": "#fbf7f1", "edgecolor": "#9a5d31", "hatch": "xx", "linewidth": .62},
)
OVERALL_STYLE = (
    {"facecolor": "#f8f6fb", "edgecolor": "#6e628a", "hatch": "///", "linewidth": .62},
    {"facecolor": "#f1f7f4", "edgecolor": "#477568", "hatch": "xx", "linewidth": .62},
)
CONFUSION_PALETTES = (
    ("#fdf390", "#cdec9e", "#a5deb6", "#8ecfc1", "#91c0c8", "#99aec6", "#a099bf", "#9e7fa9"),
    ("#f7f9fc", "#326389"),
)
OVERALL_FIELDS = (
    ("方面提及\nMacro-F1", "aspect_macro_f1"),
    ("方面情感\nMacro-F1", "sentiment_macro_f1"),
    ("端到端负面\nMicro-F1", "end_to_end_negative_f1"),
)


def configure_style():
    for cjk in ("SimSun", "Noto Serif CJK SC", "Microsoft YaHei", "PingFang SC", "SimHei"):
        try:
            findfont(FontProperties(family=cjk), fallback_to_default=False)
        except ValueError:
            continue
        break
    else:
        raise RuntimeError("请安装支持简体中文的字体，例如宋体或思源宋体")
    mpl.rcParams.update({
        "font.family": ["Times New Roman", cjk],
        "font.size": 8.5,
        "axes.linewidth": .6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": .55,
        "ytick.major.width": .55,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "legend.fontsize": 8.2,
        "hatch.linewidth": .38,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
    })


def _canvas(width, height, rect):
    figure = plt.figure(figsize=(width * MM, height * MM))
    return figure, figure.add_axes(rect)


def aspect_figure(data):
    rows = data["aspects"]
    figure, axis = _canvas(166, 158, [.235, .096, .705, .803])
    positions = []
    position = 0
    label_transform = axis.get_yaxis_transform()
    for group, members in groupby(rows, key=lambda row: row["group"]):
        axis.text(-.247, position, group, transform=label_transform, va="center", fontsize=9.4)
        position += .78
        for row in members:
            positions.append(position)
            axis.text(-.214, position, row["name"], transform=label_transform, va="center", fontsize=8.5)
            position += 1
        position += .26
    y = np.array(positions)
    axis.set_axisbelow(True)
    axis.grid(axis="x", linewidth=.45, color=GRID)
    for model, style, offset, field in zip(data["models"], ASPECT_STYLE, (-.19, .19), ("baseline", "bert")):
        axis.barh(y + offset, [row[field]["f1"] for row in rows], height=.34, label=model["name"], **style)
    axis.set(xlim=(0, 1), xticks=np.linspace(0, 1, 6), yticks=[], ylim=(y[-1] + .68, -.53),
             xlabel="方面提及识别 F1")
    axis.tick_params(axis="x", length=3)
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, borderaxespad=0,
                handlelength=1.7, columnspacing=1.2)
    return figure


def overall_figure(data):
    figure, axis = _canvas(148, 87, [.125, .245, .83, .622])
    x = np.arange(len(OVERALL_FIELDS))
    axis.set_axisbelow(True)
    axis.grid(axis="y", linewidth=.45, color=GRID)
    for model, style, offset in zip(data["models"], OVERALL_STYLE, (-.17, .17)):
        values = [model[key] for _, key in OVERALL_FIELDS]
        bars = axis.bar(x + offset, values, width=.26, label=model["name"], **style)
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=2.4, fontsize=8.4)
    axis.set(xlim=(-.55, 2.55), ylim=(0, 1), yticks=np.linspace(0, 1, 6), ylabel="F1", xticks=x,
             xticklabels=[label for label, _ in OVERALL_FIELDS])
    axis.tick_params(axis="x", length=0, pad=7)
    axis.tick_params(axis="y", length=3)
    axis.legend(loc="lower center", bbox_to_anchor=(.5, 1.025), ncol=2, borderaxespad=0, handlelength=1.7)
    return figure


def _contrast(cmap, value):
    rgb = np.array(cmap(value)[:3])
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    luminance = linear @ [0.2126, 0.7152, 0.0722]
    return "white" if luminance < .179 else "black"


def confusion_figure(data, index):
    model = data["models"][index]
    labels = data["sentiment_labels"]
    counts = model["confusion"]["counts"]
    support = model["confusion"]["row_support"]
    rates = np.array(model["confusion"]["rates"], dtype=float)
    figure, axis = _canvas(116, 102, [.220, .176, .58, .669])
    cmap = LinearSegmentedColormap.from_list(f"sentiment_{index}", CONFUSION_PALETTES[index])
    image = axis.imshow(rates, cmap=cmap, vmin=0, vmax=1)
    for i, j in np.ndindex(rates.shape):
        value = rates[i, j]
        if np.isfinite(value):
            text, color = f"{value:.1%}\n({counts[i][j]:,})", _contrast(cmap, value)
        else:
            text, color = "—", INK
        axis.text(j, i, text, ha="center", va="center", fontsize=8.4, color=color, linespacing=1.25)
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
             yticklabels=[f"{name}\n(n={n:,})" for name, n in zip(labels, support)],
             xlabel="预测情感", ylabel="人工情感")
    axis.tick_params(length=0, pad=5, labelsize=8.1)
    axis.tick_params(axis="y", pad=7)
    axis.set_title(model["name"], fontsize=10.1, pad=7)
    for spine in axis.spines.values():
        spine.set_visible(True)
    scale = figure.add_axes([.836, .243, .024, .540])
    bar = figure.colorbar(image, cax=scale, ticks=np.linspace(0, 1, 5), format=PercentFormatter(xmax=1))
    bar.ax.tick_params(length=2, labelsize=7.7)
    bar.set_label("行内比例", size=8.2, labelpad=4)
    bar.outline.set_linewidth(.5)
    return figure


def candidates_figure(data):
    evidence = data["evidence"]
    total = evidence["n_pairs_with_gold_negative_aspect"]
    bins = (("zero", "无候选"), ("one", "单候选"), ("multiple", "多候选"))
    counts = [evidence["candidate_counts"][key] for key, _ in bins]
    figure, axis = _canvas(134, 81, [.145, .177, .80, .727])
    axis.set_axisbelow(True)
    axis.grid(axis="y", linewidth=.45, color=GRID)
    bars = axis.bar(range(len(bins)), counts, width=.52, facecolor="#f6f8fa",
                    edgecolor="#527590", hatch="///", linewidth=.62)
    axis.bar_label(bars, labels=[f"{n} ({n / total:.1%})" for n in counts], padding=3.5, fontsize=8.6)
    axis.set(ylim=(0, max(counts) * 1.22), xticks=range(len(bins)), xticklabels=[name for _, name in bins],
             ylabel="评论－方面组合数")
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axis.tick_params(axis="x", length=0, pad=6)
    axis.tick_params(axis="y", length=3)
    return figure


def _quote(axis, text, span, pos):
    start, end = span["start"], span["end"]
    texts = [
        TextArea(text[:start], textprops={"size": 9}),
        TextArea(text[start:end], textprops={"size": 9, "color": RED}),
        TextArea(text[end:], textprops={"size": 9}),
    ]
    axis.add_artist(AnnotationBbox(HPacker(children=texts, align="baseline", pad=0, sep=0), pos,
                                  xycoords="data", frameon=False, box_alignment=(0, .5)))


def example_figure(data):
    example = data["evidence"]["example"]
    spans = example["methods"]
    figure, axis = _canvas(165, 51, [.051, .1792, .898, .6808])
    axis.set_axis_off()
    figure.text(.055, .956, f"演示方面：{example['aspect']}", color="#505050", fontsize=8.5, va="top")
    columns = (.008, .184, .906)
    for x, label in zip(columns, ("项目", "评论原文与抽取片段", "字符区间")):
        axis.text(x, .823, label, fontsize=8.7, va="center")
    axis.hlines([.93, .73, .08], 0, 1, color=INK, linewidth=[.8, .5, .8])
    axis.text(columns[0], .622, "原评论", va="center")
    _quote(axis, example["text"], spans["tfidf"], (columns[1], .622))
    for y, key, name in ((.409, "tfidf", "TF-IDF"), (.195, "textrank", "TextRank")):
        span = spans[key]
        axis.text(columns[0], y, name, va="center")
        axis.text(columns[1], y, span["text"], color=RED, va="center", fontsize=9)
        axis.text(columns[2], y, f"[{span['start']}, {span['end']})", va="center")
    axis.set(xlim=(0, 1), ylim=(.08, 1))
    return figure


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_tables(output, data):
    source = output / "source_data"
    source.mkdir(exist_ok=True)
    write_csv(source / "fig01_aspect_f1.csv", [
        {"aspect": row["aspect"], "name": row["name"], "group": row["group"], "gold_mention_support": row["support"],
         **{f"{model}_{metric}": value for model in ("baseline", "bert") for metric, value in row[model].items()}}
        for row in data["aspects"]
    ])
    write_csv(source / "fig02_overall_f1.csv", [
        {"model": model["name"], "test_reviews": data["test_reviews"], "test_pairs": data["test_pairs"],
         "gold_mentioned_sentiment_pairs": data["sentiment_support"],
         **{key: value for key, value in model.items() if key.endswith("_f1")}}
        for model in data["models"]
    ])
    labels = data["sentiment_labels"]
    for model, name in zip(data["models"], ("fig03_confusion_baseline", "fig04_confusion_bert")):
        write_csv(source / f"{name}.csv", [
            {"model": model["name"], "gold": gold, "predicted": prediction,
             "count": model["confusion"]["counts"][i][j], "gold_row_support": model["confusion"]["row_support"][i],
             "row_proportion": model["confusion"]["rates"][i][j]}
            for i, gold in enumerate(labels) for j, prediction in enumerate(labels)
        ])
    evidence = data["evidence"]
    write_csv(source / "fig05_evidence_candidates.csv", [
        {"candidate_count_bin": key, "count": value, "total_pairs": evidence["n_pairs_with_gold_negative_aspect"]}
        for key, value in evidence["candidate_counts"].items()
    ])
    (source / "fig06_evidence_example.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def export_figures(data, output, formats=("svg", "pdf", "png")):
    configure_style()
    output.mkdir(parents=True, exist_ok=True)
    source_tables(output, data)
    figures = (
        ("fig01_aspect_f1", aspect_figure),
        ("fig02_overall_f1", overall_figure),
        ("fig03_confusion_baseline", lambda source: confusion_figure(source, 0)),
        ("fig04_confusion_bert", lambda source: confusion_figure(source, 1)),
        ("fig05_evidence_candidates", candidates_figure),
        ("fig06_evidence_example", example_figure),
    )
    for name, draw in figures:
        figure = draw(data)
        try:
            for format_name in formats:
                figure.savefig(output / f"{name}.{format_name}", dpi=300, facecolor="white",
                               bbox_inches="tight", pad_inches=.055)
        finally:
            plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="导出独立的餐前点评科研图件")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_figures(load_visualization_data(RESULTS), args.output)
    print(f"六张独立图件已保存到 {args.output}")


if __name__ == "__main__":
    main()

