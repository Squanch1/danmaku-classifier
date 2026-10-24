# 探索性数据分析(无标签阶段)
# 产出:
#   1) 控制台统计: 规模/缺失值/长度分布/mode分布/来源视频/高频n-gram
#   2) figs/ 下三张图(浅色表面, 已验证调色板)
#   3) pilot_sample.jsonl —— 按 mode×长度分层抽样的试标注样本
from config import Config

import collections
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

conf = Config()

# ---------------- 绘图风格(浅色表面) ----------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"      # 分类槽位 1 / 单一顺序色
GREEN = "#008300"     # 分类槽位 2
MAGENTA = "#e87ba4"   # 分类槽位 3

MODE_NAMES = {1: "滚动弹幕(1)", 4: "底部弹幕(4)", 5: "顶部弹幕(5)"}
# 分类色按固定槽位分配给实体(mode 值),不按排名——筛选数据变化时颜色不漂移
MODE_COLORS = {1: BLUE, 4: GREEN, 5: MAGENTA}

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
    """隐藏上/右边框,网格线置于柱子之下。"""
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


# ---------------- 统计 ----------------

def text_stats(df):
    df = df.copy()
    df["text_len"] = df["text"].str.len()
    print("=" * 60)
    print(f"数据规模: {df.shape[0]:,} 条 | {df.shape[1]} 列 | 列名: {list(df.columns)}")
    print("\n缺失值:")
    print(df.isnull().sum().to_string())
    print("\n文本长度(字符):")
    print(df["text_len"].describe(percentiles=[.25, .5, .75, .9, .99]).round(1).to_string())
    print("\n长度分桶:")
    df["len_bucket"] = df["text_len"].map(length_bucket)
    buckets = (df["len_bucket"].value_counts()
               .reindex(["1-5", "6-10", "11-20", "21-30", "31-50", "51-100"]))
    for k, v in buckets.items():
        print(f"  {k:>6} 字: {v:>7,} 条 ({v / len(df):.1%})")
    print("\nmode 分布:")
    for m, v in sorted(df["mode"].value_counts().items()):
        print(f"  {MODE_NAMES.get(m, m)}: {v:>7,} 条 ({v / len(df):.1%})")
    print(f"\n来源视频: {df['bvid'].nunique()} 个")
    print("Top 15 视频:")
    for title, v in df["title"].value_counts().head(15).items():
        print(f"  {v:>6,} 条  {str(title)[:38]}")
    return df


def char_ngram_counts(texts, n):
    """字符级 n-gram 计数(中文/字母/数字通过, 标点与 emoji 过滤)。"""
    counter = collections.Counter()
    for t in texts:
        for i in range(len(t) - n + 1):
            gram = t[i:i + n]
            if gram.isalnum():
                counter[gram] += 1
    return counter


def ngram_stats(df, top=20):
    print(f"\n字符 n-gram(语料级, 过滤标点/emoji, Top {top}):")
    for n in (1, 2):
        counter = char_ngram_counts(df["text"], n)
        print(f"  {n}-gram: " + ", ".join(f"{g}:{c:,}" for g, c in counter.most_common(top)))


# ---------------- 图表 ----------------

def fig_text_length(df):
    buckets = (df["len_bucket"].value_counts()
               .reindex(["1-5", "6-10", "11-20", "21-30", "31-50", "51-100"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(buckets.index, buckets.values, width=0.68, color=BLUE)
    total = len(df)
    for bar, v in zip(ax.patches, buckets.values):
        ax.annotate(f"{v:,}\n{v / total:.1%}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9,
                    color=INK_SECONDARY, xytext=(0, 3), textcoords="offset points")
    ax.set_ylim(0, buckets.max() * 1.18)
    ax.set_xlabel("文本长度(字符)")
    ax.set_ylabel("弹幕条数")
    ax.set_title("弹幕文本长度分布")
    style_ax(ax)
    fig.tight_layout()
    return fig


def fig_mode_dist(df):
    counts = df["mode"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar([MODE_NAMES[m] for m in counts.index],
           counts.values, width=0.6,
           color=[MODE_COLORS[m] for m in counts.index])
    total = len(df)
    for bar, v in zip(ax.patches, counts.values):
        ax.annotate(f"{v:,} ({v / total:.1%})",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=10,
                    color=INK_PRIMARY, xytext=(0, 3), textcoords="offset points")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_ylabel("弹幕条数")
    ax.set_title("弹幕类型(mode)分布")
    style_ax(ax)
    fig.tight_layout()
    return fig


def fig_top_videos(df, top=15):
    counts = df["title"].value_counts().head(top).sort_values()
    labels = [str(t)[:16] + ("…" if len(str(t)) > 16 else "") for t in counts.index]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(labels, counts.values, height=0.6, color=BLUE)
    for bar, v in zip(ax.patches, counts.values):
        ax.annotate(f"{v:,}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    ha="left", va="center", fontsize=9,
                    color=INK_SECONDARY, xytext=(4, 0), textcoords="offset points")
    ax.set_xlim(0, counts.max() * 1.15)
    ax.set_xlabel("弹幕条数")
    ax.set_title(f"弹幕来源视频 Top {top}")
    style_ax(ax)
    fig.tight_layout()
    return fig


def fig_ngram_top(df, n, top=5):
    """高频字符 n-gram Top N 直方图。"""
    counter = char_ngram_counts(df["text"], n)
    grams, vals = zip(*counter.most_common(top))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(grams, vals, width=0.6, color=BLUE)
    for bar, v in zip(ax.patches, vals):
        ax.annotate(f"{v:,}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center", va="bottom", fontsize=10,
                    color=INK_PRIMARY, xytext=(0, 3), textcoords="offset points")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.set_ylabel("出现次数")
    ax.set_title(f"高频字符 {n}-gram Top {top}")
    style_ax(ax)
    fig.tight_layout()
    return fig


# ---------------- 分层抽样(供试标注) ----------------

def make_pilot_sample(df, n, seed):
    """按 mode × 长度桶 分层抽样,每层保底 5 条,不足随机补齐。"""
    strata = ["mode", "len_bucket"]
    sizes = df.groupby(strata).size()
    quota = {k: max(5, round(n * s / len(df))) for k, s in sizes.items()}

    parts = []
    for k, grp in df.groupby(strata):
        parts.append(grp.sample(n=min(len(grp), quota[k]), random_state=seed))
    pilot = pd.concat(parts)
    if len(pilot) < n:
        rest = df[~df.index.isin(pilot.index)].sample(n - len(pilot), random_state=seed)
        pilot = pd.concat([pilot, rest])
    pilot = pilot.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\n分层抽样(目标 {n} 条, 实际 {len(pilot)} 条):")
    got = pilot.groupby(strata).size()
    for k, v in sorted(got.items()):
        print(f"  mode={k[0]} len={k[1]:>6}: 配额 {quota[k]:>3} -> 实抽 {v}")
    pilot["label"] = None   # 待标注占位
    return pilot


def save_figs(figs_dir, df):
    Path(figs_dir).mkdir(exist_ok=True)
    for name, make_fig in [("fig1_text_length", fig_text_length),
                           ("fig2_mode_dist", fig_mode_dist),
                           ("fig3_top_videos", fig_top_videos),
                           ("fig4_top_unigrams", lambda d: fig_ngram_top(d, n=1)),
                           ("fig5_top_bigrams", lambda d: fig_ngram_top(d, n=2))]:
        path = Path(figs_dir) / f"{name}.png"
        fig = make_fig(df)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"已保存图表: {path}")
        plt.close(fig)


# ---------------- 主流程 ----------------

def eda():
    df = pd.read_json(conf.raw_data_path, lines=True)
    df = text_stats(df)
    ngram_stats(df)
    save_figs(conf.figs_dir, df)

    pilot = make_pilot_sample(df, conf.pilot_sample_size, conf.seed)
    pilot[["text", "time_s", "mode", "bvid", "title", "label"]].to_json(
        conf.pilot_sample_path, orient="records", lines=True, force_ascii=False)
    print(f"\n试标注样本已导出: {conf.pilot_sample_path}")

    plt.show()


if __name__ == "__main__":
    eda()
