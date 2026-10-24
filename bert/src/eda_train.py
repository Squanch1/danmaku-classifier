#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练集 EDA: bert/raw_data/train_data.jsonl

四个维度:
  1) 标签分布
  2) 文本长度(分布直方图 + 长度桶 × 标签交叉)
  3) mode × 标签交叉
  4) 各等级高频字符 n-gram(1/2-gram Top10)

产出:
  控制台统计 + bert/raw_data/figs/ 下四张图(浅色表面, 蓝色顺序阶)

用法: python bert/src/eda_train.py
"""

import collections
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().parent.parent / "raw_data" / "train_data.jsonl"
FIGS_DIR = DATA_FILE.parent / "figs"

LABELS = [1, 2, 3, 4]
LEN_BUCKETS = ["1-5", "6-10", "11-20", "21-30", "31-50", "51-100"]
MODE_NAMES = {1: "滚动(1)", 4: "底部(4)", 5: "顶部(5)"}

# 绘图风格(与项目此前 EDA 一致的浅色表面)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
# 有序标签用蓝色顺序阶(浅→深 = 低质→高质)
LABEL_COLORS = {1: "#86b6ef", 2: "#5598e7", 3: "#2a78d6", 4: "#184f95"}

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


def load_data():
    rows = [json.loads(l) for l in DATA_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["text_len"] = df["text"].str.len()
    df["len_bucket"] = df["text_len"].map(length_bucket)
    return df


def load_mode_map():
    """text -> mode,从原始弹幕文件回填(合并数据可能不含 mode 字段)。"""
    raw = Path(__file__).resolve().parent.parent.parent / "data" / "danmaku_raw.jsonl"
    mode_map = {}
    if raw.exists():
        for line in open(raw, encoding="utf-8"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            mode_map[row["text"]] = row.get("mode", 0)
    return mode_map


# ---------------- 维度 1: 标签分布 ----------------

def dim_label_dist(df):
    print("=" * 60)
    print(f"维度1: 标签分布({len(df)} 条)")
    vc = df["label"].value_counts().sort_index()
    for lab in LABELS:
        v = vc.get(lab, 0)
        print(f"  {lab}级: {v:>5} 条 ({v / len(df):.1%})")

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
    ax.set_title("训练集标签分布(1 垃圾 → 4 优质)")
    style_ax(ax)
    fig.tight_layout()
    return fig, vc


# ---------------- 维度 2: 文本长度 ----------------

def dim_length(df):
    print("\n" + "=" * 60)
    print("维度2: 文本长度")
    s = df["text_len"]
    print(f"  长度: 均值 {s.mean():.1f} | 中位数 {s.median():.0f} | "
          f"P90 {s.quantile(0.9):.0f} | 最短 {s.min()} | 最长 {s.max()}")
    print("  长度分桶:")
    buckets = (df["len_bucket"].value_counts().reindex(LEN_BUCKETS)).fillna(0).astype(int)
    for k in LEN_BUCKETS:
        v = buckets.get(k, 0)
        print(f"    {k:>6} 字: {v:>5} 条 ({v / len(df):.1%})")

    print("  长度桶 × 标签交叉表(行内占比%):")
    ct = pd.crosstab(df["len_bucket"], df["label"]).reindex(LEN_BUCKETS)
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    print(pct.round(1).to_string())

    # 图1: 长度直方图(按桶)
    fig1, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(LEN_BUCKETS, [buckets.get(k, 0) for k in LEN_BUCKETS],
           width=0.62, color=BLUE)
    for bar, k in zip(ax.patches, LEN_BUCKETS):
        v = buckets.get(k, 0)
        ax.annotate(f"{v:,}\n{v / len(df):.1%}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9,
                    color=INK_SECONDARY, xytext=(0, 3), textcoords="offset points")
    ax.set_ylim(0, buckets.max() * 1.18)
    ax.set_xlabel("文本长度(字符)")
    ax.set_ylabel("条数")
    ax.set_title("训练集文本长度分布")
    style_ax(ax)
    fig1.tight_layout()

    # 图2: 长度桶 × 标签 100% 堆叠
    fig2, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(LEN_BUCKETS))
    for lab in LABELS:
        vals = pct[lab].values
        ax.bar(range(len(LEN_BUCKETS)), vals, bottom=bottom, width=0.62,
               color=LABEL_COLORS[lab], label=f"{lab}级",
               edgecolor=SURFACE, linewidth=2)
        bottom += vals
    ax.set_xticks(range(len(LEN_BUCKETS)), LEN_BUCKETS)
    ax.set_ylim(0, 100)
    ax.set_ylabel("占比(%)")
    ax.set_xlabel("文本长度(字符)")
    ax.set_title("长度桶 × 标签构成(100% 堆叠)")
    ax.legend(loc="upper right", frameon=False, ncol=4)
    style_ax(ax)
    fig2.tight_layout()
    return (fig1, fig2), pct


# ---------------- 维度 3: mode × 标签 ----------------

def dim_mode(df):
    print("\n" + "=" * 60)
    print("维度3: mode × 标签")
    df = df.copy()
    mode_map = load_mode_map()
    df["mode_name"] = df["text"].map(lambda t: MODE_NAMES.get(mode_map.get(t, 0), "未知"))
    unknown = (df["mode_name"] == "未知").sum()
    if unknown:
        print(f"  [提示] {unknown} 条文本在原始池中未找到 mode(标记为未知)")
    print(df["mode_name"].value_counts().to_string())
    ct = pd.crosstab(df["mode_name"], df["label"]).reindex(list(MODE_NAMES.values()))
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    print("mode × 标签交叉表(行内占比%):")
    print(pct.round(1).to_string())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    cats = list(MODE_NAMES.values())
    bottom = np.zeros(len(cats))
    for lab in LABELS:
        vals = pct[lab].values
        ax.bar(range(len(cats)), vals, bottom=bottom, width=0.62,
               color=LABEL_COLORS[lab], label=f"{lab}级",
               edgecolor=SURFACE, linewidth=2)
        bottom += vals
    ax.set_xticks(range(len(cats)), cats)
    ax.set_ylim(0, 100)
    ax.set_ylabel("占比(%)")
    ax.set_xlabel("弹幕类型")
    ax.set_title("弹幕类型 × 标签构成(100% 堆叠)")
    ax.legend(loc="upper right", frameon=False, ncol=4)
    style_ax(ax)
    fig.tight_layout()
    return fig, pct


# ---------------- 维度 4: 各等级高频 n-gram ----------------

def char_ngrams(texts, n):
    counter = collections.Counter()
    for t in texts:
        for i in range(len(t) - n + 1):
            gram = t[i:i + n]
            if gram.isalnum():          # 中文/字母/数字通过, 标点与 emoji 过滤
                counter[gram] += 1
    return counter


def dim_ngram(df, top=10):
    print("\n" + "=" * 60)
    print(f"维度4: 各等级高频字符 n-gram(Top {top})")
    for lab in LABELS:
        subset = df[df["label"] == lab]["text"]
        print(f"\n  {lab}级({len(subset)} 条):")
        for n in (1, 2):
            grams = char_ngrams(subset, n)
            top_grams = grams.most_common(top)
            print(f"    {n}-gram: " + ", ".join(f"{g}:{c:,}" for g, c in top_grams))
    return None


# ---------------- 主流程 ----------------

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    print(f"分析文件: {DATA_FILE}")
    df = load_data()

    figs = []
    fig, _ = dim_label_dist(df)
    figs.append(("fig1_label_dist", fig))
    (fig2_len, fig3_len), _ = dim_length(df)
    figs.append(("fig2_len_dist", fig2_len))
    figs.append(("fig3_len_x_label", fig3_len))
    fig, _ = dim_mode(df)
    figs.append(("fig4_mode_x_label", fig))
    dim_ngram(df)

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in figs:
        path = FIGS_DIR / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"已保存图表: {path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
