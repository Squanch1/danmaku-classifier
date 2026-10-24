#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
有标签 EDA: 分析 5,027 条标注样本的标签分布与特征交叉。

产出:
  1) 控制台: 标签分布 / 每标签长度统计 / 长度桶×标签交叉表 / mode×标签交叉表 / 来源构成
  2) figs/ 三张图:
     fig6_label_dist.png   标签分布(有序蓝色阶)
     fig7_len_x_label.png  长度桶 × 标签构成(100% 堆叠)
     fig8_mode_x_label.png 弹幕类型 × 标签构成(100% 堆叠)

用法: python data/scripts/eda_labeled.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import Config

conf = Config()

DATA_DIR = Path(__file__).resolve().parent.parent
LABELS = [1, 2, 3, 4]
LEN_BUCKETS = ["1-5", "6-10", "11-20", "21-30", "31-50", "51-100"]
MODE_NAMES = {1: "滚动", 4: "底部", 5: "顶部"}

# 有序标签用蓝色顺序阶(浅→深 = 低质→高质),取自已验证 ramp 的 250/350/450/600
LABEL_COLORS = {1: "#86b6ef", 2: "#5598e7", 3: "#2a78d6", 4: "#184f95"}

# 绘图风格(与 eda_raw.py 一致的浅色表面)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.grid.axis": "y",
    "grid.color": GRIDLINE,
    "grid.linewidth": 1,
})


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.set_axisbelow(True)


def length_bucket(n):
    if n <= 5:
        return "1-5"
    if n <= 10:
        return "6-10"
    if n <= 20:
        return "11-20"
    if n <= 30:
        return "21-30"
    if n <= 50:
        return "31-50"
    return "51-100"


# ---------------- 数据装载 ----------------

def load_labeled():
    df = pd.read_json(DATA_DIR / "train_data.jsonl", lines=True)
    df["text_len"] = df["text"].str.len()
    df["len_bucket"] = df["text_len"].map(length_bucket)
    return df


def load_mode_map():
    """text -> mode,来自原始弹幕文件。"""
    mode_map = {}
    for line in open(DATA_DIR / "danmaku_raw.jsonl", encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        mode_map[row["text"]] = row.get("mode", 0)
    return mode_map


def load_source_sets():
    """文本 -> 来源(pilot / candidates / quality)。"""
    sources = {}
    for name, path in [("pilot", DATA_DIR / "pilot_labeled.jsonl"),
                       ("candidates", DATA_DIR / "candidates_high_labeled.jsonl")]:
        for line in open(path, encoding="utf-8"):
            try:
                sources[json.loads(line)["text"]] = name
            except (json.JSONDecodeError, KeyError):
                pass
    for csv_path in sorted((DATA_DIR / "scripts").glob("quality_labeled_*.csv")):
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("内容"):
                    sources.setdefault(row["内容"], "quality")
    return sources


# ---------------- 控制台统计 ----------------

def console_stats(df, sources):
    n = len(df)
    print("=" * 60)
    print(f"有标签样本: {n} 条")
    print("\n标签分布:")
    vc = df["label"].value_counts().sort_index()
    for lab in LABELS:
        print(f"  {lab}级: {vc.get(lab, 0):>5} 条 ({vc.get(lab, 0) / n:.1%})")

    print("\n每标签文本长度(字符):")
    for lab in LABELS:
        s = df.loc[df["label"] == lab, "text_len"]
        print(f"  {lab}级: 均值 {s.mean():.1f} | 中位数 {s.median():.0f} | "
              f"P90 {s.quantile(0.9):.0f} | 最长 {s.max()}")

    print("\n长度桶 × 标签交叉表(行内占比%):")
    ct = pd.crosstab(df["len_bucket"], df["label"]).reindex(LEN_BUCKETS)
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    print(pct.round(1).to_string())

    df["mode"] = df["text"].map(load_mode_map()).map(MODE_NAMES)
    print("\nmode × 标签交叉表(行内占比%):")
    ct2 = pd.crosstab(df["mode"], df["label"]).reindex(list(MODE_NAMES.values()))
    pct2 = ct2.div(ct2.sum(axis=1), axis=0) * 100
    print(pct2.round(1).to_string())

    print("\n来源构成(每条文本按 pilot > candidates > quality 归属):")
    df["source"] = df["text"].map(lambda t: sources.get(t, "?"))
    ct3 = pd.crosstab(df["source"], df["label"])
    print(ct3.to_string())
    return df, ct, pct, ct2, pct2


# ---------------- 图表 ----------------

def fig_label_dist(df):
    vc = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar([f"{lab}级" for lab in LABELS], [vc.get(lab, 0) for lab in LABELS],
           width=0.6, color=[LABEL_COLORS[lab] for lab in LABELS])
    total = len(df)
    for bar, lab in zip(ax.patches, LABELS):
        v = vc.get(lab, 0)
        ax.annotate(f"{v:,} ({v / total:.1%})",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=10,
                    color=INK_PRIMARY, xytext=(0, 3), textcoords="offset points")
    ax.set_ylim(0, vc.max() * 1.2)
    ax.set_ylabel("条数")
    ax.set_title("标签分布(1 垃圾 → 4 优质)")
    style_ax(ax)
    fig.tight_layout()
    return fig


def fig_stacked_pct(categories, pct, title, xlabel):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(categories))
    for lab in LABELS:
        vals = pct[lab].values
        ax.bar(range(len(categories)), vals, bottom=bottom, width=0.62,
               color=LABEL_COLORS[lab], label=f"{lab}级",
               edgecolor=SURFACE, linewidth=2)   # 2px 表面间隙
        bottom += vals
    ax.set_xticks(range(len(categories)), categories)
    ax.set_ylim(0, 100)
    ax.set_ylabel("占比(%)")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False, ncol=4)
    style_ax(ax)
    fig.tight_layout()
    return fig


def save_figs(df, pct_len, pct_mode):
    figs_dir = DATA_DIR / "figs"
    figs_dir.mkdir(exist_ok=True)
    for name, fig in [("fig6_label_dist", fig_label_dist(df)),
                      ("fig7_len_x_label", fig_stacked_pct(
                          LEN_BUCKETS, pct_len, "长度桶 × 标签构成(100% 堆叠)", "文本长度(字符)")),
                      ("fig8_mode_x_label", fig_stacked_pct(
                          list(MODE_NAMES.values()), pct_mode, "弹幕类型 × 标签构成(100% 堆叠)", "弹幕类型"))]:
        path = figs_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"已保存图表: {path}")
        plt.close(fig)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    df = load_labeled()
    sources = load_source_sets()
    df, ct_len, pct_len, ct_mode, pct_mode = console_stats(df, sources)
    save_figs(df, pct_len, pct_mode)


if __name__ == "__main__":
    main()
